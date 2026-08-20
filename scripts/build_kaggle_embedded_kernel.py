# Copyright 2026 HiperMaximus
"""Build and verify a single-file embedded Kaggle script kernel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess  # noqa: S404
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import cast

from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT

PAYLOAD_SCHEMA_VERSION = "spec0001.kaggle_payload_manifest.v1"
DEFAULT_KERNEL_DIR = Path("kaggle/kernels/setup_smoke")
GIT_EXECUTABLE = shutil.which("git") or "git"
DEFAULT_READY_MARKER = "KAGGLE_SETUP_SMOKE_READY = True"
RUNTIME_SELECTION_KERNEL_ID = "maximusshtefan/eqvae-runtime-selection"
SELECTED_RUNTIME_DEBUG_KERNEL_ID = "maximusshtefan/eqvae-selected-runtime-debug"
SELECTED_RUNTIME_LR_RANGE_KERNEL_ID = "maximusshtefan/eqvae-selected-runtime-lr-range"
SELECTED_RUNTIME_FULL_KERNEL_ID = "maximusshtefan/eqvae-selected-runtime-full"
SO2_RUNTIME_READINESS_KERNEL_ID = "maximusshtefan/eqvae-so2-runtime-readiness"
SO2_PRELAUNCH_KERNEL_ID = "maximusshtefan/eqvae-so2-prelaunch"
SO2_SELECTED_RUNTIME_FULL_KERNEL_ID = (
    "maximshtefan/eqvae-so2-selected-runtime-full-session7"
)
SO2_TRAINING_LAUNCHER_FILES = (
    Path("kaggle/kernels/so2_prelaunch/kernel-metadata.json"),
    Path("kaggle/kernels/so2_prelaunch/run_template.py"),
    Path("kaggle/kernels/so2_selected_runtime_full/kernel-metadata.json"),
    Path("kaggle/kernels/so2_selected_runtime_full/run_template.py"),
)
RUNTIME_SELECTION_V8_ARTIFACT_ROOT = Path(
    "runs/kaggle/real_data_runtime_pretest_v8",
)
RUNTIME_SELECTION_V8_ARTIFACTS = (
    Path("benchmark/runtime_proof.json"),
    Path("benchmark/runtime_matrix.csv"),
    Path("benchmark/dataloader_matrix.csv"),
    Path("benchmark/numerical_checks.csv"),
    Path("benchmark/corruption_checks.csv"),
    Path("benchmark/gate_health_summary.json"),
    Path("metrics/gate_health.csv"),
)
RUNTIME_SELECTION_BASELINE_ARTIFACTS = (
    Path("runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"),
    Path("runs/kaggle/runtime_selection_v5/benchmark/runtime_proof.json"),
)
SELECTED_RUNTIME_FULL_CONFIG = Path(
    "configs/spec0001/non_eq_vae_selected_runtime_full.json",
)
SELECTED_RUNTIME_PLAN = Path("configs/spec0001/non_eq_vae_selected_runtime.json")
FULL_TARGET_UPDATES_PATTERN = re.compile(r"(?m)^FULL_TARGET_UPDATES = \d+$")
FULL_HALF_EPOCH_INTERVAL_PATTERN = re.compile(r"(?m)^FULL_HALF_EPOCH_INTERVAL = \d+$")
EMBEDDED_B64_PATTERN = re.compile(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    flags=re.DOTALL,
)
EMBEDDED_ZIP_HASH_PATTERN = re.compile(
    r'EMBEDDED_PAYLOAD_ZIP_SHA256 = "(?P<sha>[0-9a-f]{64})"',
)
EMBEDDED_MANIFEST_HASH_PATTERN = re.compile(
    r'EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "(?P<sha>[0-9a-f]{64})"',
)


@dataclass(frozen=True)
class BuildArgs:
    """Parsed builder arguments."""

    repo_root: Path
    kernel_dir: Path
    template_path: Path
    output_run_path: Path
    verify_only: bool
    allow_dirty: bool
    ready_marker: str


def main() -> int:
    """Build or verify an embedded Kaggle kernel.

    Returns:
        Process exit status.

    """
    args = _parse_args()
    if args.verify_only:
        verify_run_file(args)
        _write_line(f"ok: verified embedded payload in {args.output_run_path}")
        return 0

    run_text = build_run_text(args)
    args.output_run_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_run_path.write_text(run_text, encoding="utf-8")
    _write_line(f"ok: wrote {args.output_run_path}")
    verify_run_file(args)
    _write_line(f"ok: verified embedded payload in {args.output_run_path}")
    return 0


def build_run_text(args: BuildArgs) -> str:
    """Return generated uploadable `run.py` text.

    Returns:
        Python source text containing the embedded payload.

    """
    manifest = _payload_manifest(args.repo_root, args.template_path, args.kernel_dir)
    manifest_bytes = _canonical_manifest_bytes(manifest)
    zip_bytes = _payload_zip_bytes(
        args.repo_root,
        manifest_bytes,
        args.kernel_dir,
    )
    payload_b64 = "\n".join(
        textwrap.wrap(base64.b64encode(zip_bytes).decode("ascii"), width=76),
    )
    substitutions = {
        "embedded_payload_b64": payload_b64,
        "embedded_payload_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "embedded_payload_manifest_sha256": hashlib.sha256(
            manifest_bytes,
        ).hexdigest(),
    }
    template = Template(args.template_path.read_text(encoding="utf-8"))
    run_text = template.safe_substitute(substitutions)
    if (
        FULL_TARGET_UPDATES_PATTERN.search(run_text) is not None
        or FULL_HALF_EPOCH_INTERVAL_PATTERN.search(run_text) is not None
    ):
        _, target, half = _derive_full_schedule(args.repo_root)
        run_text = _apply_full_schedule_substitution(
            run_text,
            target=target,
            half=half,
        )
    return run_text


def _assert_eqvae_is_repo_root(repo_root: Path) -> None:
    # The baked schedule MUST derive from the same eqvae tree the payload ships, or
    # run.py trains for a step count inconsistent with the data-root constant it
    # loads -- silently, since --verify-only re-derives through the same ambient
    # import. The old file-path loader enforced this structurally by reading
    # repo_root/src/eqvae directly; a plain `import eqvae` resolves through the
    # interpreter instead, which is the editable .pth target and ignores
    # --repo-root. Re-assert it explicitly.
    import eqvae  # noqa: PLC0415

    shipped = (repo_root / "src" / "eqvae").resolve()
    imported = Path(str(eqvae.__file__)).parent.resolve()
    if imported != shipped:
        message = (
            f"imported eqvae ({imported}) is not the tree being shipped "
            f"({shipped}); the baked schedule would not match the payload. Run "
            f"the build with the venv of --repo-root, or drop --repo-root."
        )
        raise RuntimeError(message)


def _derive_full_schedule(repo_root: Path) -> tuple[int, int, int]:
    # Spec 0011 S8b: (updates, target, half) derived from the selected plan's global
    # batch and the config epochs via the single-sourced floor helper. Raises on a
    # malformed plan/config so a bad full-kernel build fails closed.
    _assert_eqvae_is_repo_root(repo_root)
    plan = cast(
        "dict[str, object]",
        json.loads(
            (repo_root / SELECTED_RUNTIME_PLAN).read_text(
                encoding="utf-8",
            ),
        ),
    )
    config = cast(
        "dict[str, object]",
        json.loads(
            (repo_root / SELECTED_RUNTIME_FULL_CONFIG).read_text(encoding="utf-8"),
        ),
    )
    global_batch = plan.get("global_batch_size")
    training = config.get("training")
    epochs = (
        cast("dict[str, object]", training).get("epochs")
        if isinstance(training, dict)
        else None
    )
    if (
        not isinstance(global_batch, int)
        or isinstance(global_batch, bool)
        or global_batch <= 0
    ):
        message = "selected plan global_batch_size must be a positive integer"
        raise RuntimeError(message)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        message = "full config training.epochs must be a positive integer"
        raise RuntimeError(message)
    updates = training_steps_per_epoch(
        real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
        global_batch_size=global_batch,
    )
    return updates, epochs * updates, updates // 2


def _apply_full_schedule_substitution(
    run_text: str,
    *,
    target: int,
    half: int,
) -> str:
    # Spec 0011 S8b: bake the plan-derived schedule into the full kernel's run.py so the
    # constants track a non-24 batch; fail closed if either is not rewritten once.
    run_text, target_count = FULL_TARGET_UPDATES_PATTERN.subn(
        f"FULL_TARGET_UPDATES = {target}",
        run_text,
    )
    run_text, half_count = FULL_HALF_EPOCH_INTERVAL_PATTERN.subn(
        f"FULL_HALF_EPOCH_INTERVAL = {half}",
        run_text,
    )
    if target_count != 1 or half_count != 1:
        message = (
            "full schedule substitution must rewrite each constant exactly once "
            f"(target={target_count}, half={half_count})"
        )
        raise RuntimeError(message)
    return run_text


def verify_run_file(args: BuildArgs) -> None:
    """Verify that the generated `run.py` embeds a fresh payload.

    Raises:
        RuntimeError: If the embedded payload is stale or malformed.

    """
    run_text = args.output_run_path.read_text(encoding="utf-8")
    if args.ready_marker not in run_text:
        message = f"generated kernel is missing ready marker: {args.ready_marker}"
        raise RuntimeError(message)

    zip_bytes = _embedded_zip_bytes(run_text)
    expected_zip_hash = _required_match(EMBEDDED_ZIP_HASH_PATTERN, run_text)
    actual_zip_hash = hashlib.sha256(zip_bytes).hexdigest()
    if actual_zip_hash != expected_zip_hash:
        message = "embedded payload zip SHA-256 does not match generated constant"
        raise RuntimeError(message)

    manifest = _manifest_from_zip(zip_bytes)
    manifest_bytes = _canonical_manifest_bytes(manifest)
    expected_manifest_hash = _required_match(
        EMBEDDED_MANIFEST_HASH_PATTERN,
        run_text,
    )
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_hash != expected_manifest_hash:
        message = "embedded payload manifest SHA-256 does not match generated constant"
        raise RuntimeError(message)

    template_path = _template_path_for_verify(
        manifest=manifest,
        repo_root=args.repo_root,
        fallback_template_path=args.template_path,
    )
    expected_run_text = _render_template_with_embedded_payload(
        run_text=run_text,
        template_path=template_path,
        repo_root=args.repo_root,
    )
    if run_text != expected_run_text:
        message = "generated run.py wrapper does not match current run_template.py"
        raise RuntimeError(message)
    _validate_zip_members(
        zip_bytes=zip_bytes,
        manifest=manifest,
        repo_root=args.repo_root,
    )
    _validate_manifest_against_source(
        manifest=manifest,
        repo_root=args.repo_root,
        template_path=template_path,
        allow_dirty=args.allow_dirty,
    )


def _render_template_with_embedded_payload(
    *,
    run_text: str,
    template_path: Path,
    repo_root: Path,
) -> str:
    """Render the current wrapper around the already-verified embedded payload.

    Returns:
        Exact generated source expected around the embedded archive.

    """
    substitutions = {
        "embedded_payload_b64": _required_match(EMBEDDED_B64_PATTERN, run_text),
        "embedded_payload_zip_sha256": _required_match(
            EMBEDDED_ZIP_HASH_PATTERN,
            run_text,
        ),
        "embedded_payload_manifest_sha256": _required_match(
            EMBEDDED_MANIFEST_HASH_PATTERN,
            run_text,
        ),
    }
    expected = Template(template_path.read_text(encoding="utf-8")).safe_substitute(
        substitutions,
    )
    if (
        FULL_TARGET_UPDATES_PATTERN.search(expected) is not None
        or FULL_HALF_EPOCH_INTERVAL_PATTERN.search(expected) is not None
    ):
        _, target, half = _derive_full_schedule(repo_root)
        expected = _apply_full_schedule_substitution(
            expected,
            target=target,
            half=half,
        )
    return expected


def _parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(
        description="Build an embedded Kaggle script-kernel run.py.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--kernel-dir",
        default=str(Path(__file__).resolve().parents[1] / DEFAULT_KERNEL_DIR),
    )
    parser.add_argument("--template")
    parser.add_argument("--output-run")
    parser.add_argument("--ready-marker", default=DEFAULT_READY_MARKER)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow verifying a payload built from the current dirty worktree.",
    )
    namespace = parser.parse_args()
    repo_root = Path(cast("str", namespace.repo_root)).resolve()
    kernel_dir = Path(cast("str", namespace.kernel_dir))
    if not kernel_dir.is_absolute():
        kernel_dir = (repo_root / kernel_dir).resolve()
    template_arg = cast("str | None", namespace.template)
    output_run_arg = cast("str | None", namespace.output_run)
    template_path = (
        Path(template_arg)
        if template_arg is not None
        else kernel_dir / "run_template.py"
    )
    output_run_path = (
        Path(output_run_arg)
        if output_run_arg is not None
        else kernel_dir / _metadata_code_file(kernel_dir)
    )
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()
    if not output_run_path.is_absolute():
        output_run_path = (repo_root / output_run_path).resolve()
    return BuildArgs(
        repo_root=repo_root,
        kernel_dir=kernel_dir,
        template_path=template_path,
        output_run_path=output_run_path,
        verify_only=cast("bool", namespace.verify_only),
        allow_dirty=cast("bool", namespace.allow_dirty),
        ready_marker=cast("str", namespace.ready_marker),
    )


def _metadata_code_file(kernel_dir: Path) -> str:
    metadata_path = kernel_dir / "kernel-metadata.json"
    payload = cast(
        "dict[str, object]",
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )
    code_file = payload.get("code_file")
    if not isinstance(code_file, str) or not code_file:
        message = f"{metadata_path} must declare a non-empty code_file"
        raise RuntimeError(message)
    return code_file


def _payload_manifest(
    repo_root: Path,
    template_path: Path,
    kernel_dir: Path,
) -> dict[str, object]:
    entries = {
        "src/eqvae": _digest_tree(repo_root / "src" / "eqvae"),
        "configs/spec0001": _digest_tree(repo_root / "configs" / "spec0001"),
        "docs/data/ubc_ocean_masked_holdout_ids.csv": _digest_file(
            repo_root / "docs" / "data" / "ubc_ocean_masked_holdout_ids.csv",
        ),
        "pyproject.toml": _digest_file(repo_root / "pyproject.toml"),
        "uv.lock": _digest_file(repo_root / "uv.lock"),
    }
    if _is_runtime_selection_kernel(kernel_dir):
        entries.update(_runtime_selection_entry_hashes(repo_root))
    elif _ships_legacy_selected_runtime_baseline(kernel_dir):
        entries.update(_selected_runtime_baseline_entry_hashes(repo_root))
    if _is_so2_runtime_readiness_kernel(kernel_dir):
        entries["configs/spec0015"] = _digest_tree(
            repo_root / "configs" / "spec0015",
        )
    if _is_so2_training_kernel(kernel_dir):
        entries["configs/spec0016"] = _digest_tree(
            repo_root / "configs" / "spec0016",
        )
        entries.update(
            {
                str(path): _digest_file(repo_root / path)
                for path in SO2_TRAINING_LAUNCHER_FILES
            },
        )
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repo_root, "status", "--short")),
        "template": {
            "path": _manifest_path(repo_root=repo_root, path=template_path),
            "sha256": _digest_file(template_path),
        },
        "entries": entries,
    }


def _payload_zip_bytes(
    repo_root: Path,
    manifest_bytes: bytes,
    kernel_dir: Path,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path, archive_name in _payload_files(repo_root, kernel_dir):
            archive.write(path, archive_name)
        archive.writestr("payload_manifest.json", manifest_bytes)
    return buffer.getvalue()


def _payload_files(
    repo_root: Path,
    kernel_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    roots = (
        (repo_root / "src" / "eqvae", Path("src/eqvae")),
        (repo_root / "configs" / "spec0001", Path("configs/spec0001")),
    )
    if _is_so2_runtime_readiness_kernel(kernel_dir):
        roots = (
            *roots,
            (repo_root / "configs" / "spec0015", Path("configs/spec0015")),
        )
    if _is_so2_training_kernel(kernel_dir):
        roots = (
            *roots,
            (repo_root / "configs" / "spec0016", Path("configs/spec0016")),
        )
    files: list[tuple[Path, str]] = []
    for source_root, archive_root in roots:
        for path in sorted(candidate for candidate in source_root.rglob("*")):
            if not path.is_file() or _is_ignored_payload_file(path):
                continue
            archive_name = archive_root / path.relative_to(source_root)
            files.append((path, archive_name.as_posix()))
    files.extend(
        (repo_root / relative, relative.as_posix())
        for relative in (
            Path("docs/data/ubc_ocean_masked_holdout_ids.csv"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        )
    )
    if _is_so2_training_kernel(kernel_dir):
        files.extend(
            (repo_root / relative, relative.as_posix())
            for relative in SO2_TRAINING_LAUNCHER_FILES
        )
    if _is_runtime_selection_kernel(kernel_dir):
        files.extend(_runtime_selection_payload_files(repo_root))
    elif _ships_legacy_selected_runtime_baseline(kernel_dir):
        files.extend(_selected_runtime_baseline_payload_files(repo_root))
    return tuple(files)


def _is_runtime_selection_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) == RUNTIME_SELECTION_KERNEL_ID


def _is_so2_runtime_readiness_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) == SO2_RUNTIME_READINESS_KERNEL_ID


def _is_so2_training_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) in {
        SO2_PRELAUNCH_KERNEL_ID,
        SO2_SELECTED_RUNTIME_FULL_KERNEL_ID,
    }


def _is_selected_runtime_debug_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) == SELECTED_RUNTIME_DEBUG_KERNEL_ID


def _is_selected_runtime_full_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) == SELECTED_RUNTIME_FULL_KERNEL_ID


def _is_selected_runtime_lr_range_kernel(kernel_dir: Path) -> bool:
    return _kernel_id(kernel_dir) == SELECTED_RUNTIME_LR_RANGE_KERNEL_ID


def _is_selected_runtime_kernel(kernel_dir: Path) -> bool:
    return (
        _is_selected_runtime_debug_kernel(
            kernel_dir,
        )
        or _is_selected_runtime_full_kernel(
            kernel_dir,
        )
        or _is_selected_runtime_lr_range_kernel(kernel_dir)
    )


def _ships_legacy_selected_runtime_baseline(kernel_dir: Path) -> bool:
    """Return whether compatibility tests still require the historical v5 pair.

    Returns:
        Whether to package the historical v5 plan/proof pair.

    """
    return _is_selected_runtime_debug_kernel(
        kernel_dir,
    ) or _is_selected_runtime_full_kernel(kernel_dir)


def _kernel_id(kernel_dir: Path) -> str:
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        return ""
    payload = cast(
        "dict[str, object]",
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )
    value = payload.get("id")
    return value if isinstance(value, str) else ""


def _runtime_selection_v8_payload_files(
    repo_root: Path,
) -> tuple[tuple[Path, str], ...]:
    return tuple(
        (
            repo_root / RUNTIME_SELECTION_V8_ARTIFACT_ROOT / relative,
            (RUNTIME_SELECTION_V8_ARTIFACT_ROOT / relative).as_posix(),
        )
        for relative in RUNTIME_SELECTION_V8_ARTIFACTS
    )


def _runtime_selection_payload_files(
    repo_root: Path,
) -> tuple[tuple[Path, str], ...]:
    return (
        *_runtime_selection_v8_payload_files(repo_root),
        *(
            (repo_root / relative, relative.as_posix())
            for relative in RUNTIME_SELECTION_BASELINE_ARTIFACTS
        ),
    )


def _selected_runtime_baseline_payload_files(
    repo_root: Path,
) -> tuple[tuple[Path, str], ...]:
    return tuple(
        (repo_root / relative, relative.as_posix())
        for relative in RUNTIME_SELECTION_BASELINE_ARTIFACTS
    )


def _runtime_selection_v8_entry_hashes(repo_root: Path) -> dict[str, str]:
    return {
        (RUNTIME_SELECTION_V8_ARTIFACT_ROOT / relative).as_posix(): _digest_file(
            repo_root / RUNTIME_SELECTION_V8_ARTIFACT_ROOT / relative,
        )
        for relative in RUNTIME_SELECTION_V8_ARTIFACTS
    }


def _runtime_selection_entry_hashes(repo_root: Path) -> dict[str, str]:
    hashes = _runtime_selection_v8_entry_hashes(repo_root)
    hashes.update(_selected_runtime_baseline_entry_hashes(repo_root))
    return hashes


def _selected_runtime_baseline_entry_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): _digest_file(repo_root / relative)
        for relative in RUNTIME_SELECTION_BASELINE_ARTIFACTS
    }


def _is_ignored_payload_file(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _manifest_path(*, repo_root: Path, path: Path) -> str:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _template_path_for_verify(
    *,
    manifest: dict[str, object],
    repo_root: Path,
    fallback_template_path: Path,
) -> Path:
    if fallback_template_path.exists():
        return fallback_template_path
    raw_template = manifest.get("template")
    if not isinstance(raw_template, dict):
        return fallback_template_path
    template = cast("dict[str, object]", raw_template)
    template_path = template.get("path")
    if not isinstance(template_path, str) or not template_path:
        return fallback_template_path
    path = Path(template_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _validate_manifest_against_source(  # noqa: C901, PLR0912
    *,
    manifest: dict[str, object],
    repo_root: Path,
    template_path: Path,
    allow_dirty: bool,
) -> None:
    errors: list[str] = []
    if manifest.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        errors.append("unexpected payload manifest schema_version")

    current_commit = _git_output(repo_root, "rev-parse", "HEAD")
    current_dirty = bool(_git_output(repo_root, "status", "--short"))
    if manifest.get("git_commit") != current_commit:
        errors.append("payload git_commit does not match current HEAD")
    if manifest.get("git_dirty") is not current_dirty:
        errors.append("payload git_dirty does not match current worktree state")
    if not allow_dirty and current_dirty:
        errors.append("payload was built from a dirty git worktree")

    template_error = _manifest_template_error(
        manifest=manifest,
        repo_root=repo_root,
        template_path=template_path,
    )
    if template_error is not None:
        errors.append(template_error)

    expected_entries = {
        "src/eqvae": _digest_tree(repo_root / "src" / "eqvae"),
        "configs/spec0001": _digest_tree(repo_root / "configs" / "spec0001"),
        "docs/data/ubc_ocean_masked_holdout_ids.csv": _digest_file(
            repo_root / "docs" / "data" / "ubc_ocean_masked_holdout_ids.csv",
        ),
        "pyproject.toml": _digest_file(repo_root / "pyproject.toml"),
        "uv.lock": _digest_file(repo_root / "uv.lock"),
    }
    kernel_dir = repo_root / _metadata_kernel_dir(manifest)
    if _is_runtime_selection_kernel(kernel_dir):
        expected_entries.update(_runtime_selection_entry_hashes(repo_root))
    elif _ships_legacy_selected_runtime_baseline(kernel_dir):
        expected_entries.update(_selected_runtime_baseline_entry_hashes(repo_root))
    if _is_so2_runtime_readiness_kernel(kernel_dir):
        expected_entries["configs/spec0015"] = _digest_tree(
            repo_root / "configs" / "spec0015",
        )
    if _is_so2_training_kernel(kernel_dir):
        expected_entries["configs/spec0016"] = _digest_tree(
            repo_root / "configs" / "spec0016",
        )
        expected_entries.update(
            {
                str(path): _digest_file(repo_root / path)
                for path in SO2_TRAINING_LAUNCHER_FILES
            },
        )
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, dict):
        errors.append("payload manifest entries must be an object")
    else:
        entries = cast("dict[str, object]", raw_entries)
        for key, expected in expected_entries.items():
            if entries.get(key) != expected:
                errors.append(f"payload entry {key!r} is stale")

    if errors:
        raise RuntimeError("\n".join(errors))


def _manifest_template_error(
    *,
    manifest: dict[str, object],
    repo_root: Path,
    template_path: Path,
) -> str | None:
    expected_template = {
        "path": _manifest_path(repo_root=repo_root, path=template_path),
        "sha256": _digest_file(template_path),
    }
    raw_template = manifest.get("template")
    if not isinstance(raw_template, dict):
        return "payload manifest template must be an object"
    template = cast("dict[str, object]", raw_template)
    if template != expected_template:
        return "payload template does not match current run_template.py"
    return None


def _validate_zip_members(
    *,
    zip_bytes: bytes,
    manifest: dict[str, object],
    repo_root: Path,
) -> None:
    kernel_dir = repo_root / _metadata_kernel_dir(manifest)
    expected_sources = {
        archive_name: path
        for path, archive_name in _payload_files(repo_root, kernel_dir)
    }
    expected_names = set(expected_sources)
    expected_names.add("payload_manifest.json")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        member_names = archive.namelist()
        actual_names = set(member_names)
        if len(member_names) != len(actual_names):
            message = "payload zip contains duplicate member names"
            raise RuntimeError(message)
        stale_members = [
            name
            for name, source in expected_sources.items()
            if name in actual_names and archive.read(name) != source.read_bytes()
        ]
        manifest_bytes_match = "payload_manifest.json" in actual_names and archive.read(
            "payload_manifest.json",
        ) == _canonical_manifest_bytes(manifest)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unexpected:
            details.append(f"unexpected={unexpected!r}")
        message = "payload zip members do not match expected file set"
        if details:
            message = f"{message}: {', '.join(details)}"
        raise RuntimeError(message)
    if stale_members:
        message = f"payload zip member bytes are stale: {sorted(stale_members)!r}"
        raise RuntimeError(message)
    if not manifest_bytes_match:
        message = "payload zip manifest bytes are not canonical"
        raise RuntimeError(message)


def _metadata_kernel_dir(manifest: dict[str, object]) -> Path:
    raw_template = manifest.get("template")
    if not isinstance(raw_template, dict):
        return DEFAULT_KERNEL_DIR
    template = cast("dict[str, object]", raw_template)
    path = template.get("path")
    if not isinstance(path, str):
        return DEFAULT_KERNEL_DIR
    template_path = Path(path)
    if template_path.name != "run_template.py":
        return template_path.parent
    return template_path.parent


def _embedded_zip_bytes(run_text: str) -> bytes:
    payload_b64 = _required_match(EMBEDDED_B64_PATTERN, run_text)
    return base64.b64decode(payload_b64.encode("ascii"))


def _manifest_from_zip(zip_bytes: bytes) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                message = f"unsafe embedded payload path: {name}"
                raise RuntimeError(message)
        manifest = cast("object", json.loads(archive.read("payload_manifest.json")))
    if not isinstance(manifest, dict):
        message = "embedded payload manifest must be a JSON object"
        raise TypeError(message)
    return cast("dict[str, object]", manifest)


def _required_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if match is None:
        message = f"generated run.py is missing pattern: {pattern.pattern}"
        raise RuntimeError(message)
    return match.group(1)


def _canonical_manifest_bytes(manifest: dict[str, object]) -> bytes:
    text = f"{json.dumps(manifest, indent=2, sort_keys=True)}\n"
    return text.encode("utf-8")


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        if _is_ignored_payload_file(item):
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(_digest_file(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        (GIT_EXECUTABLE, *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_line(message: str) -> None:
    sys.stdout.write(f"{message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
