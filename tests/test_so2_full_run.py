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
                "/kaggle/input/eqvae-so2-session2-step18000/step_018000.pt"
            ),
            "resume_checkpoint_sha256": (
                "5911ad37a1ed3f8a92055e45717be496d18545426e56667e1989a3da9a525ec4"
            ),
        }


def test_so2_full_launcher_resumes_exact_so2_checkpoint_only() -> None:
    """Session 3 must attach only real data and the exact SO2 commit point."""
    repository = Path(__file__).resolve().parents[1]
    kernel_dir = repository / "kaggle/kernels/so2_selected_runtime_full"
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
        ),
    )
    source = (kernel_dir / "run_template.py").read_text(encoding="utf-8")
    assert metadata["dataset_sources"] == [
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
        "maximusshtefan/eqvae-so2-session2-step18000",
    ]
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []
    assert '"--resume"' in source
    assert "/kaggle/input/eqvae-so2-session2-step18000/step_018000.pt" in source
    assert "5911ad37a1ed3f8a92055e45717be496d18545426e56667e1989a3da9a525ec4" in source
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
    checkpoint = tmp_path / "step_018000.pt"
    checkpoint.write_bytes(b"exact checkpoint fixture")
    validate.__globals__["RESUME_CHECKPOINT_SHA256"] = hashlib.sha256(
        checkpoint.read_bytes(),
    ).hexdigest()
    validate(checkpoint)
    checkpoint.write_bytes(b"changed checkpoint fixture")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate(checkpoint)
    checkpoint.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        validate(checkpoint)


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
    checkpoint = tmp_path / "step_018000.pt"
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

    def ensure_latest_torch() -> None:
        return None

    def extract(_destination: Path) -> Path:
        return payload

    function_globals["_ensure_latest_torch"] = ensure_latest_torch
    function_globals["_extract"] = extract

    def validate(path: Path) -> None:
        original_validate(path)
        events.append("validated")

    def run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert events == ["validated"]
        assert command[-2:] == ("--resume", str(checkpoint.resolve()))
        assert "eqvae-baseline-session" not in " ".join(command)
        assert kwargs["cwd"] == payload
        events.append("launched")
        return subprocess.CompletedProcess(command, 0)

    function_globals["_validate_resume_checkpoint"] = validate
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("EQVAE_SO2_FULL_OUTPUT_DIR", str(output))
    monkeypatch.setenv("EQVAE_SO2_FULL_RESUME", str(checkpoint))
    monkeypatch.delenv("EQVAE_SO2_FULL_IMPORT_ONLY", raising=False)
    assert main() == 0
    assert events == ["validated", "launched"]


def test_so2_full_push_guard_requires_fresh_proof_and_cost_acceptance() -> None:
    """A full remote write stays blocked until the measured one-off gate passes."""
    repository = Path(__file__).resolve().parents[1]
    script = (repository / "scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    for required in (
        "guard_so2_full_push_ready",
        "KAGGLE_SO2_FULL_COST_CONFIRMED",
        "so2_prelaunch_verdict.json",
        "validate_prelaunch_artifacts",
        "EXPECTED_CONTINUATION_WRAPPER_SHA256",
        "preflight-so2-prelaunch",
        "preflight-so2-selected-runtime-full",
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
