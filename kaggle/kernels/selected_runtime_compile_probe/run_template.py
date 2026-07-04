# Copyright 2026 HiperMaximus
"""Generated single-file Kaggle compiled fast-path probe launcher template.

Extracts the embedded repo payload, then launches the dual-T4 compiled fast-path
probe under ``torchrun --standalone --nproc_per_node=2`` so both ranks exercise
the exact recipe. The probe (``eqvae.benchmarking.compiled_fastpath_probe``) writes
NON-PROMOTABLE proof/matrix/manifest artifacts on rank 0; this launcher validates
that only those artifacts were written and that they block promotion. It attaches
no Kaggle dataset (``dataset_sources == []``), so it stages in seconds.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path
from typing import cast

KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True
COMPILE_PROBE_BENCHMARK_KIND = "kaggle_compiled_fastpath_probe"
COMPILE_PROBE_BENCHMARK_SOURCE = "kaggle_no_dataset_synthetic_compiled_fastpath"
COMPILE_PROBE_STATUS_SCOPE = "non_promotable_compiled_fastpath_probe"
COMPILE_PROBE_MODULE = "eqvae.benchmarking.compiled_fastpath_probe"
COMPILE_PROBE_PROOF_FILENAME = "compiled_fastpath_probe_proof.json"
COMPILE_PROBE_MATRIX_FILENAME = "compiled_fastpath_probe_matrix.csv"
COMPILE_PROBE_MANIFEST_FILENAME = "compiled_fastpath_probe_manifest.json"
COMPILE_PROBE_ALLOWED_ARTIFACTS = {
    COMPILE_PROBE_PROOF_FILENAME,
    COMPILE_PROBE_MATRIX_FILENAME,
    COMPILE_PROBE_MANIFEST_FILENAME,
}
COMPILE_PROBE_REQUIRED_BLOCKED_CLAIMS = {
    "runtime_selection",
    "full_run_readiness",
    "real_data_throughput",
    "convergence",
    "paper_evidence",
    "final_speedup_on_real_data",
}
DEFAULT_KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
# The probe needs dual-T4 NCCL, so it only ever runs on Kaggle; a hung run would
# otherwise burn the whole GPU session, so bound it (mirrors the synthetic-timing
# child torchrun timeout).
PROBE_TIMEOUT_SECONDS = 1800
EMBEDDED_PAYLOAD_B64 = """
$embedded_payload_b64
"""
EMBEDDED_PAYLOAD_ZIP_SHA256 = "$embedded_payload_zip_sha256"
EMBEDDED_PAYLOAD_MANIFEST_SHA256 = "$embedded_payload_manifest_sha256"


def main() -> int:
    """Run the compiled fast-path probe launcher from an embedded payload.

    Returns:
        Process exit status.

    """
    output_dir = _output_dir()
    return _run_compiled_fastpath_probe(output_dir)


def _run_compiled_fastpath_probe(output_dir: Path) -> int:
    _require_python_version()
    payload_dir = _extract_payload(_payload_extract_dir(output_dir))
    payload_src = payload_dir / "src"
    _launch_dual_t4_probe(payload_src=payload_src, output_dir=output_dir)
    _validate_compiled_fastpath_probe_artifacts(output_dir=output_dir)
    return 0


def _launch_dual_t4_probe(*, payload_src: Path, output_dir: Path) -> None:
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        COMPILE_PROBE_MODULE,
        "--output-dir",
        str(output_dir),
    )
    subprocess.run(  # noqa: S603
        command,
        check=True,
        env=_probe_environment(payload_src=payload_src),
        cwd=str(output_dir),
        timeout=PROBE_TIMEOUT_SECONDS,
    )


def _probe_environment(*, payload_src: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("EQVAE_DATA_ROOT", None)
    existing = environment.get("PYTHONPATH")
    entries = [str(payload_src)]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return environment


def _output_dir() -> Path:
    configured = os.environ.get("EQVAE_OUTPUT_DIR")
    output_dir = Path(configured) if configured else DEFAULT_KAGGLE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _payload_extract_dir(output_dir: Path) -> Path:
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/eqvae_compiled_fastpath_probe_payload")
    if Path("/tmp").exists():  # noqa: S108
        return Path("/tmp/eqvae_compiled_fastpath_probe_payload")  # noqa: S108
    return output_dir / "embedded_payload"


def _require_python_version() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036
        message = (
            "eqvae compiled fast-path probe requires Python >= 3.12 because active "
            "source uses Python 3.12 type-alias syntax"
        )
        raise RuntimeError(message)


def _extract_payload(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    zip_bytes = base64.b64decode(EMBEDDED_PAYLOAD_B64.encode("ascii"))
    actual_zip_hash = hashlib.sha256(zip_bytes).hexdigest()
    if actual_zip_hash != EMBEDDED_PAYLOAD_ZIP_SHA256:
        message = "embedded payload zip SHA-256 mismatch"
        raise RuntimeError(message)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                message = f"unsafe embedded payload path: {member.filename}"
                raise RuntimeError(message)
        archive.extractall(destination)

    manifest_path = destination / "payload_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != EMBEDDED_PAYLOAD_MANIFEST_SHA256:
        message = "embedded payload manifest SHA-256 mismatch"
        raise RuntimeError(message)
    return destination


def _validate_compiled_fastpath_probe_artifacts(*, output_dir: Path) -> None:
    benchmark_dir = output_dir / "benchmark"
    observed = {path.name for path in benchmark_dir.iterdir()}
    if observed != COMPILE_PROBE_ALLOWED_ARTIFACTS:
        message = f"unexpected compiled fast-path probe artifacts: {sorted(observed)}"
        raise RuntimeError(message)
    proof = json.loads(
        (benchmark_dir / COMPILE_PROBE_PROOF_FILENAME).read_text(encoding="utf-8"),
    )
    manifest = json.loads(
        (benchmark_dir / COMPILE_PROBE_MANIFEST_FILENAME).read_text(encoding="utf-8"),
    )
    for payload in (proof, manifest):
        _validate_non_promotable_payload(payload)
    _assert_under_kaggle_working(output_dir)


def _validate_non_promotable_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        message = "compiled fast-path probe artifact must be a JSON object"
        raise TypeError(message)
    data = cast("dict[str, object]", payload)
    errors: list[str] = []
    if data.get("benchmark_kind") != COMPILE_PROBE_BENCHMARK_KIND:
        errors.append("wrong benchmark_kind")
    if data.get("benchmark_source") != COMPILE_PROBE_BENCHMARK_SOURCE:
        errors.append("wrong benchmark_source")
    if data.get("status_scope") != COMPILE_PROBE_STATUS_SCOPE:
        errors.append("wrong status_scope")
    if data.get("full_run_eligible") is not False:
        errors.append("full_run_eligible must be false")
    blocked_claims = data.get("blocked_claims")
    if not isinstance(blocked_claims, dict):
        errors.append("blocked_claims must be an object")
    elif set(cast("dict[str, object]", blocked_claims)) != (
        COMPILE_PROBE_REQUIRED_BLOCKED_CLAIMS
    ):
        errors.append("blocked_claims must match the required probe claims")
    errors.extend(
        f"{source_field} must be empty"
        for source_field in (
            "dataset_sources",
            "competition_sources",
            "kernel_sources",
            "model_sources",
        )
        if data.get(source_field) != []
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def _assert_under_kaggle_working(path: Path) -> None:
    kaggle_working = Path("/kaggle/working").resolve()
    resolved = path.resolve()
    if resolved != kaggle_working and kaggle_working not in resolved.parents:
        message = (
            "compiled fast-path probe output must resolve under "
            f"{kaggle_working}, got {resolved}"
        )
        raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
