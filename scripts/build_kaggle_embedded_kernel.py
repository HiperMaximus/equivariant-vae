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

PAYLOAD_SCHEMA_VERSION = "spec0001.kaggle_payload_manifest.v1"
DEFAULT_KERNEL_DIR = Path("kaggle/kernels/setup_smoke")
GIT_EXECUTABLE = shutil.which("git") or "git"
DEFAULT_READY_MARKER = "KAGGLE_SETUP_SMOKE_READY = True"
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
    manifest = _payload_manifest(args.repo_root, args.template_path)
    manifest_bytes = _canonical_manifest_bytes(manifest)
    zip_bytes = _payload_zip_bytes(args.repo_root, manifest_bytes)
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
    return template.safe_substitute(substitutions)


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
    _validate_manifest_against_source(
        manifest=manifest,
        repo_root=args.repo_root,
        template_path=template_path,
        allow_dirty=args.allow_dirty,
    )


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
    repo_root = Path(namespace.repo_root).resolve()
    kernel_dir = Path(namespace.kernel_dir)
    if not kernel_dir.is_absolute():
        kernel_dir = (repo_root / kernel_dir).resolve()
    template_path = (
        Path(cast("str", namespace.template))
        if namespace.template is not None
        else kernel_dir / "run_template.py"
    )
    output_run_path = (
        Path(cast("str", namespace.output_run))
        if namespace.output_run is not None
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
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    code_file = payload.get("code_file")
    if not isinstance(code_file, str) or not code_file:
        message = f"{metadata_path} must declare a non-empty code_file"
        raise RuntimeError(message)
    return code_file


def _payload_manifest(repo_root: Path, template_path: Path) -> dict[str, object]:
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repo_root, "status", "--short")),
        "template": {
            "path": _manifest_path(repo_root=repo_root, path=template_path),
            "sha256": _digest_file(template_path),
        },
        "entries": {
            "src/eqvae": _digest_tree(repo_root / "src" / "eqvae"),
            "configs/spec0001": _digest_tree(repo_root / "configs" / "spec0001"),
            "pyproject.toml": _digest_file(repo_root / "pyproject.toml"),
            "uv.lock": _digest_file(repo_root / "uv.lock"),
        },
    }


def _payload_zip_bytes(repo_root: Path, manifest_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path, archive_name in _payload_files(repo_root):
            archive.write(path, archive_name)
        archive.writestr("payload_manifest.json", manifest_bytes)
    return buffer.getvalue()


def _payload_files(repo_root: Path) -> tuple[tuple[Path, str], ...]:
    roots = (
        (repo_root / "src" / "eqvae", Path("src/eqvae")),
        (repo_root / "configs" / "spec0001", Path("configs/spec0001")),
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
        for relative in (Path("pyproject.toml"), Path("uv.lock"))
    )
    return tuple(files)


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
    template = manifest.get("template")
    if not isinstance(template, dict):
        return fallback_template_path
    template_path = template.get("path")
    if not isinstance(template_path, str) or not template_path:
        return fallback_template_path
    path = Path(template_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _validate_manifest_against_source(
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
        "pyproject.toml": _digest_file(repo_root / "pyproject.toml"),
        "uv.lock": _digest_file(repo_root / "uv.lock"),
    }
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        errors.append("payload manifest entries must be an object")
    else:
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
    template = manifest.get("template")
    if not isinstance(template, dict):
        return "payload manifest template must be an object"
    if template != expected_template:
        return "payload template does not match current run_template.py"
    return None


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
        manifest = json.loads(archive.read("payload_manifest.json"))
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
