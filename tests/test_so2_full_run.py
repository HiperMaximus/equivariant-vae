# Copyright 2026 HiperMaximus
"""Packaging contracts for the one-off Spec 0016 SO2 launchers."""
# ruff: noqa: PLR0913, PLR0917, S603

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from eqvae.benchmarking.so2_prelaunch import execution_identity

if TYPE_CHECKING:
    from collections.abc import Callable

_CASES = (
    (
        "so2_prelaunch",
        "KAGGLE_SO2_PRELAUNCH_READY = True",
        "EQVAE_SO2_PRELAUNCH_IMPORT_ONLY",
        "EQVAE_SO2_PRELAUNCH_OUTPUT_DIR",
        "benchmark/so2_prelaunch_import.json",
    ),
    (
        "so2_selected_runtime_full",
        "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True",
        "EQVAE_SO2_FULL_IMPORT_ONLY",
        "EQVAE_SO2_FULL_OUTPUT_DIR",
        "benchmark/so2_full_import.json",
    ),
)


@pytest.mark.parametrize(
    ("kernel_name", "ready_marker", "import_flag", "output_flag", "artifact"),
    _CASES,
)
def test_so2_kernel_payload_builds_and_imports(
    tmp_path: Path,
    kernel_name: str,
    ready_marker: str,
    import_flag: str,
    output_flag: str,
    artifact: str,
) -> None:
    """Both upload scripts must extract and import their exact embedded source."""
    repository = Path(__file__).resolve().parents[1]
    kernel_dir = repository / "kaggle/kernels" / kernel_name
    run_path = tmp_path / kernel_name / "run.py"
    subprocess.run(
        (
            sys.executable,
            str(repository / "scripts/build_kaggle_embedded_kernel.py"),
            "--repo-root",
            str(repository),
            "--kernel-dir",
            str(kernel_dir),
            "--output-run",
            str(run_path),
            "--ready-marker",
            ready_marker,
            "--allow-dirty",
        ),
        cwd=repository,
        check=True,
    )
    output_dir = tmp_path / f"{kernel_name}_output"
    environment = os.environ.copy()
    environment[import_flag] = "1"
    environment[output_flag] = str(output_dir)
    environment["EQVAE_SKIP_TORCH_UPGRADE"] = "1"
    subprocess.run(
        (sys.executable, str(run_path)),
        cwd=run_path.parent,
        env=environment,
        check=True,
    )
    payload = cast(
        "dict[str, object]",
        json.loads((output_dir / artifact).read_text(encoding="utf-8")),
    )
    assert payload["status"] == "pass"
    if kernel_name == "so2_prelaunch":
        extracted = output_dir / "embedded_payload"
        assert execution_identity(extracted) == execution_identity(repository)
    else:
        assert payload == {
            "status": "pass",
            "fresh_start": False,
            "resume_checkpoint": (
                "/kaggle/input/datasets/maximshtefan/eqvae-so2-session5-step45000/step_045000.pt"
            ),
            "resume_checkpoint_sha256": (
                "703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93"
            ),
        }


def test_so2_full_launcher_resumes_exact_so2_checkpoint_only() -> None:
    """Session 6 must attach only UBC and the verified SO2 continuation input."""
    repository = Path(__file__).resolve().parents[1]
    kernel_dir = repository / "kaggle/kernels/so2_selected_runtime_full"
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
        ),
    )
    assert metadata["id"] == "maximshtefan/eqvae-so2-selected-runtime-full"
    source = (kernel_dir / "run_template.py").read_text(encoding="utf-8")
    assert metadata["dataset_sources"] == [
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
        "maximshtefan/eqvae-so2-session5-step45000",
    ]
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []
    assert '"--resume"' in source
    assert (
        "/kaggle/input/datasets/maximshtefan/eqvae-so2-session5-step45000/step_045000.pt"
        in source
    )
    assert "703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93" in source
    assert "RESUME_CHECKPOINT_BYTES = 16_440_368" in source
    assert "RESUME_MOUNT_WAIT_SECONDS = 600" in source
    assert "_wait_for_resume_checkpoint(resume_checkpoint)" in source
    assert "eqvae-baseline-session" not in source
    assert "EQVAE_SO2_FULL_RESUME" in source
    assert '"fresh_start": False' in source


def test_so2_full_resume_checkpoint_validation_fails_closed(tmp_path: Path) -> None:
    """The continuation wrapper must reject missing or changed checkpoint bytes."""
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "kaggle/kernels/so2_selected_runtime_full/run_template.py"),
        run_name="spec0016_so2_resume_template_test",
    )
    validate = cast("Callable[[Path], None]", namespace["_validate_resume_checkpoint"])
    checkpoint = tmp_path / "step_045000.pt"
    checkpoint.write_bytes(b"exact checkpoint fixture")
    validate.__globals__["RESUME_CHECKPOINT_SHA256"] = hashlib.sha256(
        checkpoint.read_bytes(),
    ).hexdigest()
    validate.__globals__["RESUME_CHECKPOINT_BYTES"] = checkpoint.stat().st_size
    validate(checkpoint)
    checkpoint.write_bytes(b"alter checkpoint fixture")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate(checkpoint)
    checkpoint.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        validate(checkpoint)


def test_so2_full_mount_wait_allows_delayed_read_only_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed Kaggle mount must still receive normal exact validation."""
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "kaggle/kernels/so2_selected_runtime_full/run_template.py"),
        run_name="spec0016_so2_resume_mount_wait_test",
    )
    wait_for_mount = cast(
        "Callable[[Path], None]",
        namespace["_wait_for_resume_checkpoint"],
    )
    validate = cast("Callable[[Path], None]", namespace["_validate_resume_checkpoint"])
    checkpoint = tmp_path / "step_045000.pt"
    fixture = b"delayed checkpoint fixture"
    function_globals = wait_for_mount.__globals__
    function_globals["RESUME_CHECKPOINT_SHA256"] = hashlib.sha256(fixture).hexdigest()
    function_globals["RESUME_CHECKPOINT_BYTES"] = len(fixture)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        checkpoint.write_bytes(fixture)

    time_module = function_globals["time"]
    monkeypatch.setattr(time_module, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time_module, "sleep", sleep)
    wait_for_mount(checkpoint)
    validate(checkpoint)
    assert sleeps == [10]


def test_so2_full_launcher_validates_before_exact_resume_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paid subprocess cannot start before the selected checkpoint validates."""
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "kaggle/kernels/so2_selected_runtime_full/run_template.py"),
        run_name="spec0016_so2_resume_execution_test",
    )
    main = cast("Callable[[], int]", namespace["main"])
    function_globals = main.__globals__
    checkpoint = tmp_path / "step_045000.pt"
    checkpoint.write_bytes(b"execution checkpoint fixture")
    payload = tmp_path / "payload"
    (payload / "src").mkdir(parents=True)
    output = tmp_path / "output"
    events: list[str] = []
    original_validate = cast(
        "Callable[[Path], None]",
        namespace["_validate_resume_checkpoint"],
    )
    function_globals["RESUME_CHECKPOINT_SHA256"] = hashlib.sha256(
        checkpoint.read_bytes(),
    ).hexdigest()
    function_globals["RESUME_CHECKPOINT_BYTES"] = checkpoint.stat().st_size

    def ensure_latest_torch() -> None:
        return None

    def extract(_destination: Path) -> Path:
        return payload

    function_globals["_ensure_latest_torch"] = ensure_latest_torch
    function_globals["_extract"] = extract

    def validate(path: Path) -> None:
        original_validate(path)
        events.append("validated")

    def wait_for_mount(path: Path) -> None:
        assert path == checkpoint.resolve()
        events.append("mount_ready")

    def run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert events == ["mount_ready", "validated"]
        assert command[-2:] == ("--resume", str(checkpoint.resolve()))
        assert "eqvae-baseline-session" not in " ".join(command)
        assert kwargs["cwd"] == payload
        events.append("launched")
        return subprocess.CompletedProcess(command, 0)

    function_globals["_wait_for_resume_checkpoint"] = wait_for_mount
    function_globals["_validate_resume_checkpoint"] = validate
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("EQVAE_SO2_FULL_OUTPUT_DIR", str(output))
    monkeypatch.setenv("EQVAE_SO2_FULL_RESUME", str(checkpoint))
    monkeypatch.delenv("EQVAE_SO2_FULL_IMPORT_ONLY", raising=False)
    assert main() == 0
    assert events == ["mount_ready", "validated", "launched"]


def test_so2_full_push_guard_requires_fresh_proof_and_cost_acceptance() -> None:
    """A full write needs fresh proof, cost acceptance, and the session-1 core."""
    repository = Path(__file__).resolve().parents[1]
    script = (repository / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for required in (
        "guard_so2_full_push_ready",
        "KAGGLE_SO2_FULL_COST_CONFIRMED",
        "so2_prelaunch_verdict.json",
        "validate_prelaunch_artifacts",
        "prelaunch_identity",
        "resume_identity",
        "so2_continuation_resume_execution_core_changed",
        "EXPECTED_CONTINUATION_WRAPPER_SHA256",
        "preflight-so2-prelaunch",
        "preflight-so2-selected-runtime-full",
    ):
        assert required in script
    for required in (
        'so2_full_resume_authority_dir="runs/kaggle/so2_selected_runtime_full_v5_session5_remote"',
        'so2_full_resume_dataset_dir="runs/kaggle/so2_session5_resume_dataset"',
        'so2_full_resume_dataset_slug="maximshtefan/eqvae-so2-session5-step45000"',
        'EXPECTED_RESUME_COMMIT = "e1b9e9f9a28299f4604a768720345ae9cd7c2fb3"',
        'EXPECTED_DATASET_SLUG = "maximshtefan/eqvae-so2-session5-step45000"',
        '"703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93"',
        '"f9210ea3d8fc3b9739d74e0aef69821c4e5bd0af612edb5a6c62743fe91e262c"',
        "EXPECTED_STEP = 45000",
        'expected_files = {"dataset-metadata.json", "step_045000.pt"}',
    ):
        assert required in script


def test_prelaunch_runtime_proof_rejects_nested_amp_fallback(tmp_path: Path) -> None:
    """A green summary cannot hide an unapplied dual-T4/FP16 runtime bundle."""
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "kaggle/kernels/so2_prelaunch/run_template.py"),
        run_name="spec0016_template_test",
    )
    validate = cast(
        "Callable[[Path], bool]",
        namespace["_runtime_proofs_pass"],
    )
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    documents = {
        "selected_runtime_plan_applied.json": {"status": "local_pass"},
        "checkpoint_resume_proof.json": {"status": "local_pass"},
        "gate_health_summary.json": {"status": "local_pass"},
        "training_summary.json": {
            "ddp_rank_device_proof": {"status": "local_pass"},
            "amp_execution": {
                "enabled": True,
                "grad_scaler_enabled": True,
                "autocast_dtype": "float16",
                "requested_autocast_dtype": "float16",
                "local_amp_status": "executed_amp_fp16_conservative",
            },
        },
    }
    for name, payload in documents.items():
        (benchmark / name).write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    assert validate(tmp_path)
    cast(
        "dict[str, object]",
        cast("dict[str, object]", documents["training_summary.json"])["amp_execution"],
    )["enabled"] = False
    (benchmark / "training_summary.json").write_text(
        f"{json.dumps(documents['training_summary.json'])}\n",
        encoding="utf-8",
    )
    assert not validate(tmp_path)
