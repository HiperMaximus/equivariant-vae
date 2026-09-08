# Copyright 2026 HiperMaximus
"""Guard and generated-wrapper tests for the Spec 0021 delivery workflow."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from kaggle.kernels.ubc_ocean_latent_finalizer import run_template as finalizer_template
from kaggle.kernels.ubc_ocean_latent_inference import run_template as inference_template
from scripts.build_kaggle_embedded_kernel import BuildArgs, verify_run_file

from eqvae.cli import (
    build_ubc_latent_kernel,
    finalize_ubc_latent_stores,
    generate_ubc_latent_stores,
    stage_ubc_latent_resume,
)
from eqvae.inference.input_bundle import ResumeBundleAuthority

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL = REPO_ROOT / "scripts" / "kaggle_kernel.sh"
TEMPLATE = REPO_ROOT / "kaggle/kernels/ubc_ocean_latent_inference/run_template.py"
READY_MARKER = "KAGGLE_UBC_OCEAN_LATENT_INFERENCE_READY = True"
DATASET_REFERENCE = "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
SAVED_OUTPUT_LIMIT = 20_000_000_000
BASH = shutil.which("bash") or "/bin/bash"


@pytest.fixture
def repo_tmp() -> Iterator[Path]:
    """Keep generated payload paths beneath the repo, as the builder requires.

    Yields:
        One unique, cleanup-bound directory beneath ``runs/local_tmp``.

    """
    parent = REPO_ROOT / "runs" / "local_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="spec0021_workflow_", dir=parent))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_compact_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n",
        encoding="utf-8",
    )


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pilot_inputs(root: Path) -> tuple[Path, Path]:
    contract = root / "spec0021_input_contract.json"
    receipt = root / "input_dataset_receipt.json"
    _write_json(contract, {"schema_version": "spec0021.input_bundle.v1"})
    _write_json(
        receipt,
        {
            "schema_version": "spec0021.input_dataset_receipt.v1",
            "dataset_reference": DATASET_REFERENCE,
            "dataset_version": 1,
            "input_contract_sha256": hashlib.sha256(
                contract.read_bytes(),
            ).hexdigest(),
        },
    )
    return contract, receipt


def _fresh_production_config(
    *,
    run: int,
    contract: Path,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "spec0021.inference_config.v2",
        "mode": "production",
        "run_number": run,
        "spec_sha256": hashlib.sha256(
            (
                REPO_ROOT / "docs/specs/0021-dual-model-wsi-latent-inference.md"
            ).read_bytes(),
        ).hexdigest(),
        "input_dataset_receipt": dict(receipt),
        "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "work_manifest_sha256": build_ubc_latent_kernel.WORK_HASHES[run],
        "normal_checkpoint_sha256": build_ubc_latent_kernel.NORMAL_CHECKPOINT_SHA256,
        "so2_checkpoint_sha256": build_ubc_latent_kernel.SO2_CHECKPOINT_SHA256,
        "pilot_authority_sha256": "d" * 64,
        "selected_recipe": {
            "batch_size": 8,
            "d2h": "synchronous",
            "numeric": "FP32",
            "execution": "eager",
        },
        "expected_binary_output_bytes": 2
        * (
            build_ubc_latent_kernel.LATENT_HEADER_BYTES
            + build_ubc_latent_kernel.WORK_ROW_COUNTS[run]
            * build_ubc_latent_kernel.LATENT_RECORD_BYTES
        ),
        "saved_output_limit_bytes": SAVED_OUTPUT_LIMIT,
        "resume_dataset_receipt": None,
    }


@pytest.mark.parametrize("mode", ["pilot", "production"])
def test_inference_wrapper_keeps_only_publishable_output(  # noqa: PLR0915
    repo_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Verified embedded mode alone authorizes production and no temp is published.

    The exact working-tree boundary is a DELIBERATE POLICY: Kaggle captures the
    whole working directory, so successful runtime files outside ``dataset/``
    would silently escape the declared artifact allow-list.

    """
    working_root = repo_tmp / mode / "working"
    output_root = working_root / "dataset"
    scratch_root = working_root / ".spec0021_scratch"
    private_root = working_root / ".spec0021_private"
    working_root.mkdir(parents=True)
    monkeypatch.setattr(inference_template, "WORKING_ROOT", working_root)
    monkeypatch.setattr(inference_template, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(inference_template, "SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr(inference_template, "PRIVATE_ROOT", private_root)
    config_bytes = f"{json.dumps({'mode': mode}, sort_keys=True)}\n".encode()
    authority_bytes = b"{}\n"
    monkeypatch.setattr(
        inference_template,
        "EMBEDDED_INFERENCE_CONFIG_B64",
        base64.b64encode(config_bytes).decode("ascii"),
    )
    monkeypatch.setattr(
        inference_template,
        "EMBEDDED_INFERENCE_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        inference_template,
        "EMBEDDED_PILOT_AUTHORITY_B64",
        base64.b64encode(authority_bytes).decode("ascii"),
    )
    monkeypatch.setattr(
        inference_template,
        "EMBEDDED_PILOT_AUTHORITY_SHA256",
        hashlib.sha256(authority_bytes).hexdigest(),
    )

    def fake_extract(destination: Path) -> Path:
        assert observed["torch_upgraded"] is True
        (destination / "src").mkdir(parents=True)
        return destination

    def skip_dependencies() -> None:
        return

    def fake_torch_upgrade() -> None:
        observed["torch_upgraded"] = True

    def accept_import_origin(*_args: object) -> None:
        return

    observed: dict[str, object] = {}

    def fake_worker(arguments: Sequence[str]) -> int:
        observed["arguments"] = arguments
        observed["confirmation"] = os.environ.get(
            inference_template.PRODUCTION_CONFIRMATION_ENV,
        )
        observed["compiler_cache"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        authority_path = Path(
            os.environ[inference_template.PILOT_AUTHORITY_ENV],
        )
        assert authority_path.parent == private_root
        assert authority_path.read_bytes() == authority_bytes
        if mode == "pilot":
            for name in inference_template.PILOT_OUTPUT_ALLOWLIST:
                (output_root / name).write_text("evidence\n", encoding="utf-8")
        else:
            (output_root / "production-owned-artifact.bin").write_bytes(b"done")
        return 0

    monkeypatch.setattr(inference_template, "_ensure_latest_torch", fake_torch_upgrade)
    monkeypatch.setattr(inference_template, "_install_dependencies", skip_dependencies)
    monkeypatch.setattr(inference_template, "_extract_payload", fake_extract)
    monkeypatch.setattr(
        inference_template,
        "_assert_import_origin",
        accept_import_origin,
    )
    monkeypatch.setattr(generate_ubc_latent_stores, "main", fake_worker)
    prior_sys_path = list(sys.path)
    try:
        assert inference_template.main() == 0
    finally:
        sys.path[:] = prior_sys_path

    assert observed["confirmation"] == ("1" if mode == "production" else None)
    assert observed["compiler_cache"] == str(scratch_root / "inductor")
    assert {path.name for path in working_root.iterdir()} == {"dataset"}
    assert not private_root.exists()
    assert not scratch_root.exists()
    assert inference_template.PRODUCTION_CONFIRMATION_ENV not in os.environ
    assert inference_template.PILOT_AUTHORITY_ENV not in os.environ


def test_pilot_builder_binds_exact_metadata_config_and_wrapper(
    repo_tmp: Path,
) -> None:
    """Generate one upload directory with exact source order and config bytes."""
    contract, receipt = _pilot_inputs(repo_tmp)
    output = repo_tmp / "kernels"
    assert (
        build_ubc_latent_kernel.main(
            [
                "pilot",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
            ],
        )
        == 0
    )
    kernel_dir = output / "pilot"
    assert {path.name for path in kernel_dir.iterdir()} == {
        "kernel-metadata.json",
        "run.py",
        "spec0021_inference_config.json",
    }
    metadata = cast(
        "dict[str, object]",
        json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")),
    )
    assert metadata == {
        "id": "maximusshtefan/eqvae-ubc-ocean-latent-pilot",
        "title": "eqvae UBC-OCEAN latent pilot",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REFERENCE],
        "competition_sources": ["UBC-OCEAN"],
        "kernel_sources": [],
        "model_sources": [],
    }
    config_path = kernel_dir / "spec0021_inference_config.json"
    config_bytes = config_path.read_bytes()
    config = cast("dict[str, object]", json.loads(config_bytes))
    assert config["mode"] == "pilot"
    assert config["run_number"] is None
    assert config["input_dataset_receipt"] == json.loads(
        receipt.read_text(encoding="utf-8"),
    )
    assert (
        generate_ubc_latent_stores._validate_config(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            config_path,
            expected_mode="pilot",
        )["mode"]
        == "pilot"
    )
    assert (
        config_bytes
        == (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )

    run_text = (kernel_dir / "run.py").read_text(encoding="utf-8")
    match = re.search(
        r'EMBEDDED_INFERENCE_CONFIG_B64 = "(?P<payload>[A-Za-z0-9+/=]+)"',
        run_text,
    )
    assert match is not None
    assert base64.b64decode(match.group("payload"), validate=True) == config_bytes
    verify_run_file(
        BuildArgs(
            repo_root=REPO_ROOT,
            kernel_dir=kernel_dir,
            template_path=TEMPLATE,
            output_run_path=kernel_dir / "run.py",
            verify_only=True,
            allow_dirty=True,
            ready_marker=READY_MARKER,
        ),
    )
    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedded inference config differs"):
        verify_run_file(
            BuildArgs(
                repo_root=REPO_ROOT,
                kernel_dir=kernel_dir,
                template_path=TEMPLATE,
                output_run_path=kernel_dir / "run.py",
                verify_only=True,
                allow_dirty=True,
                ready_marker=READY_MARKER,
            ),
        )


def test_production_all_derives_five_ordered_run_configs_from_authority(
    repo_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind five fresh runs and reject any derived payload above Kaggle's cap.

    The 20 GB ceiling is a DELIBERATE PLATFORM POLICY: accepting a merely
    positive number would let a valid run finish inference but lose its saved
    dataset when Kaggle captures ``/kaggle/working``.

    """
    contract, receipt = _pilot_inputs(repo_tmp)
    authority = repo_tmp / "spec0021_pilot_authority.json"
    authority_payload: dict[str, object] = {
        "schema_version": "spec0021.pilot_authority.v2",
        "status": "smoke_passed",
        "smoke_passed": True,
        "smoke_wsi_id": 15188,
        "smoke_patch_count": 16,
        "matrix_sha256": "a" * 64,
        "input_dataset_receipt_sha256": _canonical_hash(
            cast(
                "dict[str, object]",
                json.loads(receipt.read_text(encoding="utf-8")),
            ),
        ),
        "selected_recipe": {
            "batch_size": 8,
            "d2h": "synchronous",
            "numeric": "FP32",
            "execution": "eager",
        },
    }
    _write_json(authority, authority_payload)
    pilot_receipt = repo_tmp / "pilot_receipt.json"
    _write_json(
        pilot_receipt,
        {
            "schema_version": "spec0021.pilot_receipt.v1",
            "status": "smoke_passed",
            "authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
            "matrix_sha256": "a" * 64,
        },
    )
    captured: list[tuple[Path, Mapping[str, object]]] = []

    def capture_build(
        *,
        repo_root: Path,
        kernel_dir: Path,
        config: Mapping[str, object],
    ) -> None:
        assert repo_root == REPO_ROOT
        captured.append((kernel_dir, config))

    monkeypatch.setattr(build_ubc_latent_kernel, "_build_one", capture_build)
    assert (
        build_ubc_latent_kernel.main(
            [
                "production-all",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(repo_tmp / "kernels"),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
                "--pilot-authority",
                str(authority),
                "--pilot-receipt",
                str(pilot_receipt),
            ],
        )
        == 0
    )
    assert [path.name for path, _config in captured] == [
        "run_01",
        "run_02",
        "run_03",
        "run_04",
        "run_05",
    ]
    for run_number, (_path, config) in enumerate(captured, start=1):
        assert config["mode"] == "production"
        assert config["run_number"] == run_number
        assert config["input_dataset_receipt"] == json.loads(
            receipt.read_text(encoding="utf-8"),
        )
        assert config["saved_output_limit_bytes"] == SAVED_OUTPUT_LIMIT
        assert config["expected_binary_output_bytes"] == 2 * (
            build_ubc_latent_kernel.LATENT_HEADER_BYTES
            + build_ubc_latent_kernel.WORK_ROW_COUNTS[run_number]
            * build_ubc_latent_kernel.LATENT_RECORD_BYTES
        )
    monkeypatch.setattr(
        build_ubc_latent_kernel,
        "KAGGLE_SAVED_OUTPUT_LIMIT_BYTES",
        build_ubc_latent_kernel.PUBLISHED_METADATA_RESERVE_BYTES,
    )
    with pytest.raises(ValueError, match="exceeds Kaggle's saved-output cap"):
        build_ubc_latent_kernel.main(
            [
                "production-all",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(repo_tmp / "too_small"),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
                "--pilot-authority",
                str(authority),
                "--pilot-receipt",
                str(pilot_receipt),
            ],
        )


def test_resume_builder_reuses_fixed_run_id_and_adds_only_pinned_resume_source(
    repo_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A continuation changes only its receipt/source, preserving run identity.

    The resume dataset is a DELIBERATE POLICY boundary: attaching any unrelated
    dataset would let mutable or cross-run bytes reach the frozen encoders.

    """
    contract, receipt = _pilot_inputs(repo_tmp)
    authority = repo_tmp / "spec0021_pilot_authority.json"
    receipt_payload = cast(
        "dict[str, object]",
        json.loads(receipt.read_text(encoding="utf-8")),
    )
    _write_json(
        authority,
        {
            "schema_version": "spec0021.pilot_authority.v2",
            "status": "smoke_passed",
            "smoke_passed": True,
            "smoke_wsi_id": 15188,
            "smoke_patch_count": 16,
            "matrix_sha256": "a" * 64,
            "input_dataset_receipt_sha256": _canonical_hash(receipt_payload),
            "selected_recipe": {
                "batch_size": 8,
                "d2h": "synchronous",
                "numeric": "FP32",
                "execution": "eager",
            },
        },
    )
    pilot_receipt = repo_tmp / "pilot_receipt.json"
    _write_json(
        pilot_receipt,
        {
            "schema_version": "spec0021.pilot_receipt.v1",
            "status": "smoke_passed",
            "authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
            "matrix_sha256": "a" * 64,
        },
    )
    run_number = 3
    resume_reference = "maximusshtefan/eqvae-ubc-ocean-latent-run-03-resume"
    resume_receipt = repo_tmp / "resume_receipt.json"
    resume_payload: dict[str, object] = {
        "schema_version": "spec0021.resume_dataset_receipt.v1",
        "dataset_reference": resume_reference,
        "dataset_version": 2,
        "run_number": run_number,
        "provenance_sha256": "a" * 64,
        "input_bundle_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "run_config_sha256": "b" * 64,
        "work_manifest_sha256": build_ubc_latent_kernel.WORK_HASHES[run_number],
        "remote_files": [{"logical_name": "dataset-metadata.json", "size": 1}],
        "remote_listing_sha256": "c" * 64,
    }
    _write_json(resume_receipt, resume_payload)

    def fake_run_text(_args: BuildArgs) -> str:
        return f"{READY_MARKER}\n"

    monkeypatch.setattr(build_ubc_latent_kernel, "build_run_text", fake_run_text)
    output_root = repo_tmp / "kernels"
    assert (
        build_ubc_latent_kernel.main(
            [
                "resume",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output_root),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
                "--pilot-authority",
                str(authority),
                "--pilot-receipt",
                str(pilot_receipt),
                "--run-number",
                str(run_number),
                "--resume-receipt",
                str(resume_receipt),
            ],
        )
        == 0
    )
    kernel_dir = output_root / "run_03"
    config = cast(
        "dict[str, object]",
        json.loads(
            (kernel_dir / "spec0021_inference_config.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
        ),
    )
    assert kernel_dir.name == "run_03"
    assert config["mode"] == "production"
    assert config["run_number"] == run_number
    assert config["resume_dataset_receipt"] == resume_payload
    assert metadata["dataset_sources"] == [
        DATASET_REFERENCE,
        resume_reference,
    ]


def test_resume_stager_derives_next_version_from_verified_prior_receipt(
    repo_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume versions advance exactly once so an old attachment cannot drift.

    This is a DERIVED relationship between the last verified receipt and the
    next immutable bundle, not a manually selected version number.

    """
    contract = repo_tmp / "bundle/spec0021_input_contract.json"
    run_config = repo_tmp / "run/spec0021_inference_config.json"
    artifacts = repo_tmp / "artifacts/dataset"
    for path in (contract, run_config):
        _write_json(path, {})
    artifacts.mkdir(parents=True)
    prior = repo_tmp / "resume_receipt.json"
    expected_version = 5
    _write_json(
        prior,
        {
            "schema_version": "spec0021.resume_dataset_receipt.v1",
            "dataset_reference": (
                "maximusshtefan/eqvae-ubc-ocean-latent-run-02-resume"
            ),
            "dataset_version": 4,
            "run_number": 2,
        },
    )
    captured: dict[str, object] = {}

    def capture_stage(destination: Path, **kwargs: object) -> ResumeBundleAuthority:
        captured.update(kwargs)
        captured["destination"] = destination
        return ResumeBundleAuthority(
            provenance_sha256="a" * 64,
            dataset_slug=cast("str", kwargs["dataset_slug"]),
            dataset_version=cast("int", kwargs["dataset_version"]),
            run_number=2,
            input_bundle_sha256=cast("str", kwargs["input_bundle_sha256"]),
            run_config_sha256=cast("str", kwargs["run_config_sha256"]),
            work_manifest_sha256="b" * 64,
        )

    monkeypatch.setattr(stage_ubc_latent_resume, "stage_resume_bundle", capture_stage)
    assert (
        stage_ubc_latent_resume.main(
            [
                "2",
                str(artifacts.parent),
                "--input-contract",
                str(contract),
                "--run-config",
                str(run_config),
                "--prior-receipt",
                str(prior),
                "--output-root",
                str(repo_tmp / "resume"),
            ],
        )
        == 0
    )
    assert captured["artifacts_dir"] == artifacts
    assert captured["dataset_version"] == expected_version
    assert cast("Path", captured["destination"]).name == "version_0005"


def test_finalizer_builder_binds_cpu_kernel_and_five_output_sources(  # noqa: PLR0914, PLR0915
    repo_tmp: Path,
) -> None:
    """Remote finalization reads five outputs without duplicating their payload.

    The five kernel sources and CPU-only metadata are DELIBERATE POLICY: adding
    a dataset or GPU would change both the provenance graph and resource route.

    """
    contract, receipt = _pilot_inputs(repo_tmp / "authority")
    output_root = repo_tmp / "kernels"
    receipt_payload = cast(
        "dict[str, object]",
        json.loads(receipt.read_text(encoding="utf-8")),
    )
    for run in range(1, 6):
        _write_compact_json(
            output_root / f"run_{run:02d}" / "spec0021_inference_config.json",
            _fresh_production_config(
                run=run,
                contract=contract,
                receipt=receipt_payload,
            ),
        )

    run_one_config = output_root / "run_01" / "spec0021_inference_config.json"
    valid_run_one = run_one_config.read_bytes()
    incomplete = cast("dict[str, object]", json.loads(valid_run_one))
    del incomplete["resume_dataset_receipt"]
    _write_compact_json(run_one_config, incomplete)
    with pytest.raises(ValueError, match="fields differ"):
        build_ubc_latent_kernel.main(
            [
                "finalizer",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output_root),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
            ],
        )
    run_one_config.write_bytes(valid_run_one)
    terminal_run_one = cast("dict[str, object]", json.loads(valid_run_one))
    terminal_run_one["resume_dataset_receipt"] = {
        "schema_version": "spec0021.resume_dataset_receipt.v1",
        "dataset_reference": "maximusshtefan/eqvae-ubc-ocean-latent-run-01-resume",
        "dataset_version": 2,
        "run_number": 1,
        "provenance_sha256": "a" * 64,
        "input_bundle_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "run_config_sha256": "b" * 64,
        "work_manifest_sha256": build_ubc_latent_kernel.WORK_HASHES[1],
        "remote_files": [{"logical_name": "spec0021_resume_contract.json", "size": 1}],
        "remote_listing_sha256": "c" * 64,
    }
    _write_compact_json(run_one_config, terminal_run_one)

    assert (
        build_ubc_latent_kernel.main(
            [
                "finalizer",
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output_root),
                "--input-contract",
                str(contract),
                "--input-receipt",
                str(receipt),
            ],
        )
        == 0
    )
    finalizer = output_root / "finalizer"
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (finalizer / "kernel-metadata.json").read_text(encoding="utf-8"),
        ),
    )
    expected_sources = [
        f"maximusshtefan/eqvae-ubc-ocean-latent-run-{run:02d}" for run in range(1, 6)
    ]
    assert metadata["id"] == "maximusshtefan/eqvae-ubc-ocean-latent-finalizer"
    assert metadata["title"] == "eqvae UBC-OCEAN latent finalizer"
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_internet"] == "true"
    assert metadata["dataset_sources"] == [DATASET_REFERENCE]
    assert metadata["kernel_sources"] == expected_sources
    assert metadata["competition_sources"] == []
    config = cast(
        "dict[str, object]",
        json.loads(
            (finalizer / "spec0021_inference_config.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    assert config["schema_version"] == "spec0021.finalizer_config.v1"
    assert config["kernel_sources"] == expected_sources
    assert set(cast("dict[str, object]", config["production_configs"])) == {
        f"run_{run:02d}" for run in range(1, 6)
    }
    embedded_configs = cast("dict[str, object]", config["production_configs"])
    embedded_run_one = cast("dict[str, object]", embedded_configs["run_01"])
    embedded_production = cast("dict[str, object]", embedded_run_one["config"])
    assert (
        embedded_production["resume_dataset_receipt"]
        == terminal_run_one["resume_dataset_receipt"]
    )
    verify_run_file(
        BuildArgs(
            repo_root=REPO_ROOT,
            kernel_dir=finalizer,
            template_path=(
                REPO_ROOT / "kaggle/kernels/ubc_ocean_latent_finalizer/run_template.py"
            ),
            output_run_path=finalizer / "run.py",
            verify_only=True,
            allow_dirty=True,
            ready_marker=READY_MARKER,
        ),
    )
    env = os.environ.copy()
    env.update(
        {
            "EQVAE_LATENT_KERNEL_ROOT": str(output_root),
            "EQVAE_LATENT_INPUT_BUNDLE_DIR": str(contract.parent),
            "EQVAE_LATENT_INPUT_AUTHORITY_DIR": str(receipt.parent),
        },
    )
    preflight = subprocess.run(  # noqa: S603
        (str(SHELL), "preflight-latent-inference", "finalizer"),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    finalizer_config_path = finalizer / "spec0021_inference_config.json"
    valid_finalizer_config = finalizer_config_path.read_bytes()
    invalid_finalizer_config = cast(
        "dict[str, object]",
        json.loads(valid_finalizer_config),
    )
    invalid_records = cast(
        "dict[str, object]",
        invalid_finalizer_config["production_configs"],
    )
    invalid_record = cast("dict[str, object]", invalid_records["run_01"])
    invalid_production = cast("dict[str, object]", invalid_record["config"])
    invalid_production["selected_recipe"] = {}
    invalid_record["sha256"] = hashlib.sha256(
        (
            json.dumps(invalid_production, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode(),
    ).hexdigest()
    _write_compact_json(finalizer_config_path, invalid_finalizer_config)
    invalid_preflight = subprocess.run(  # noqa: S603
        (str(SHELL), "preflight-latent-inference", "finalizer"),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert invalid_preflight.returncode != 0
    assert "selected recipe is invalid" in invalid_preflight.stderr
    finalizer_config_path.write_bytes(valid_finalizer_config)
    metadata["kernel_sources"] = [*expected_sources, "owner/unrelated-output"]
    _write_json(finalizer / "kernel-metadata.json", metadata)
    marker = repo_tmp / "remote-called"
    fake_bin = repo_tmp / "bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    guarded_env = {
        **env,
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
    }
    refused = subprocess.run(  # noqa: S603
        (str(SHELL), "push", str(finalizer)),
        cwd=REPO_ROOT,
        env=guarded_env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert refused.returncode != 0
    assert "finalizer metadata differs" in refused.stderr
    assert not marker.exists()


def test_finalizer_mount_layout_uses_only_run_directory_symlinks(
    repo_tmp: Path,
) -> None:
    """Observed Kaggle notebook outputs use run symlinks, never artifact copies.

    This matches the one-off run-01 probe and protects the 73 GiB read-only
    payload from copying or artifact-level symlink substitution.

    """
    input_root = repo_tmp / "input"
    pair_root = repo_tmp / "pairs"
    input_root.mkdir()
    source_roots: dict[int, Path] = {}
    for run in range(1, 6):
        slug = f"eqvae-ubc-ocean-latent-run-{run:02d}"
        mount_root = input_root / "notebooks" / "maximusshtefan" / slug
        source_root = mount_root / "dataset"
        source_root.mkdir(parents=True)
        for name in (
            "__output__.json",
            "__results__.html",
            "__script__.ipynb",
            "__script__.py",
            "custom.css",
        ):
            (mount_root / name).touch()
        source_roots[run] = source_root
        for name in (
            f"normal_vae_mu_run_{run:02d}_of_05.bin",
            f"normal_vae_mu_run_{run:02d}_of_05.json",
            f"so2_vae_mu_run_{run:02d}_of_05.bin",
            f"so2_vae_mu_run_{run:02d}_of_05.json",
            f"spec0021_pair_audit_run_{run:02d}_of_05.json",
        ):
            (source_root / name).touch()
    finalizer_template._prepare_pair_root(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        input_root,
        pair_root,
    )
    assert all(path.is_symlink() for path in pair_root.iterdir())
    finalize_ubc_latent_stores._pair_paths(pair_root)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    audit = source_roots[1] / "spec0021_pair_audit_run_01_of_05.json"
    audit.unlink()
    target = repo_tmp / "redirected-audit.json"
    target.touch()
    audit.symlink_to(target)
    with pytest.raises(ValueError, match="allow-list"):
        finalize_ubc_latent_stores._pair_paths(pair_root)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001


def test_finalizer_mount_discovery_rejects_flat_nested_ambiguity(
    repo_tmp: Path,
) -> None:
    """One source mounted through two conventions must fail, never win by order."""
    input_root = repo_tmp / "input"
    input_root.mkdir()
    config: dict[str, object] = {
        "input_dataset_receipt": {
            "dataset_reference": DATASET_REFERENCE,
            "dataset_version": 1,
        },
    }
    for root in (
        input_root / "eqvae-ubc-ocean-latent-inputs",
        input_root
        / "datasets"
        / "maximusshtefan"
        / "eqvae-ubc-ocean-latent-inputs"
        / "versions"
        / "1",
    ):
        root.mkdir(parents=True)
        (root / "spec0021_input_contract.json").touch()
    with pytest.raises(ValueError, match="exactly one immutable"):
        finalizer_template._find_input_contract(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            input_root,
            config,
        )


def test_finalizer_mount_discovery_rejects_preprobe_kernel_layout(
    repo_tmp: Path,
) -> None:
    """The disproved root and kernels families cannot satisfy a run mount."""
    input_root = repo_tmp / "input"
    pair_root = repo_tmp / "pairs"
    for run in range(1, 6):
        slug = f"eqvae-ubc-ocean-latent-run-{run:02d}"
        source_root = input_root / "kernels" / "maximusshtefan" / slug
        source_root.mkdir(parents=True)
        (source_root / f"spec0021_pair_audit_run_{run:02d}_of_05.json").touch()
    with pytest.raises(ValueError, match="kernel-output mount for run 01; found 0"):
        finalizer_template._prepare_pair_root(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            input_root,
            pair_root,
        )


def test_finalizer_wrapper_invokes_validator_and_publishes_only_small_index(  # noqa: PLR0914, PLR0915
    repo_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote wrapper scans mounts but leaves only six indexes and one audit.

    This is the disk-safety invariant: neither the embedded code/config nor any
    73 GiB shard may remain under the Kaggle working output after success.

    """
    working = repo_tmp / "working"
    input_root = repo_tmp / "input"
    payload_root = working / ".payload"
    pair_root = working / ".pairs"
    config_root = working / ".configs"
    receipt_path = working / ".receipt.json"
    config_path = working / ".finalizer.json"
    output_root = working / "dataset"
    working.mkdir()
    input_root.mkdir()
    input_mount = (
        input_root
        / "datasets"
        / "maximusshtefan"
        / "eqvae-ubc-ocean-latent-inputs"
        / "versions"
        / "1"
    )
    input_mount.mkdir(parents=True)
    contract = input_mount / "spec0021_input_contract.json"
    contract.write_text("input contract\n", encoding="utf-8")
    for run in range(1, 6):
        slug = f"eqvae-ubc-ocean-latent-run-{run:02d}"
        mount_root = input_root / "notebooks" / "maximusshtefan" / slug
        source_root = mount_root / "dataset"
        source_root.mkdir(parents=True)
        for name in (
            "__output__.json",
            "__results__.html",
            "__script__.ipynb",
            "__script__.py",
            "custom.css",
        ):
            (mount_root / name).touch()
        for name in (
            f"normal_vae_mu_run_{run:02d}_of_05.bin",
            f"normal_vae_mu_run_{run:02d}_of_05.json",
            f"so2_vae_mu_run_{run:02d}_of_05.bin",
            f"so2_vae_mu_run_{run:02d}_of_05.json",
            f"spec0021_pair_audit_run_{run:02d}_of_05.json",
        ):
            (source_root / name).touch()
    receipt: dict[str, object] = {
        "schema_version": "spec0021.input_dataset_receipt.v1",
        "dataset_reference": DATASET_REFERENCE,
        "dataset_version": 1,
    }
    production_configs: dict[str, object] = {}
    for run in range(1, 6):
        production = {
            "schema_version": "spec0021.inference_config.v2",
            "mode": "production",
            "run_number": run,
        }
        encoded = (
            json.dumps(production, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        production_configs[f"run_{run:02d}"] = {
            "config": production,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    config: dict[str, object] = {
        "schema_version": "spec0021.finalizer_config.v1",
        "mode": "finalizer",
        "spec_sha256": hashlib.sha256(b"spec\n").hexdigest(),
        "input_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "input_dataset_receipt": receipt,
        "kernel_sources": list(finalizer_template.KERNEL_SOURCES),
        "production_configs": production_configs,
    }
    encoded_config = (
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    monkeypatch.setattr(finalizer_template, "INPUT_ROOT", input_root)
    monkeypatch.setattr(finalizer_template, "WORKING_ROOT", working)
    monkeypatch.setattr(finalizer_template, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(finalizer_template, "PAYLOAD_ROOT", payload_root)
    monkeypatch.setattr(finalizer_template, "PAIR_ROOT", pair_root)
    monkeypatch.setattr(finalizer_template, "CONFIG_ROOT", config_root)
    monkeypatch.setattr(finalizer_template, "RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(finalizer_template, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        finalizer_template,
        "EMBEDDED_INFERENCE_CONFIG_B64",
        base64.b64encode(encoded_config).decode("ascii"),
    )
    monkeypatch.setattr(
        finalizer_template,
        "EMBEDDED_INFERENCE_CONFIG_SHA256",
        hashlib.sha256(encoded_config).hexdigest(),
    )

    def fake_extract(destination: Path) -> Path:
        (destination / "src").mkdir(parents=True)
        spec = destination / "docs/specs/0021-dual-model-wsi-latent-inference.md"
        spec.parent.mkdir(parents=True)
        spec.write_bytes(b"spec\n")
        return destination

    observed: dict[str, object] = {}

    def fake_finalizer(arguments: Sequence[str]) -> int:
        observed["arguments"] = tuple(arguments)
        assert all(path.is_symlink() for path in pair_root.iterdir())
        assert len(
            list(config_root.glob("run_*/spec0021_inference_config.json")),
        ) == len(finalizer_template.KERNEL_SOURCES)
        views = output_root / "views"
        views.mkdir(parents=True)
        for task in ("cancer", "tissue"):
            for split in ("train", "validation", "test"):
                (views / f"{task}_{split}_locations.csv").write_text(
                    "location\n",
                    encoding="utf-8",
                )
        (output_root / "spec0021_latent_store_global_audit.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(finalizer_template, "_extract_payload", fake_extract)

    def fake_assert_import_origin(_module_file: Path, _payload_src: Path) -> None:
        return

    monkeypatch.setattr(
        finalizer_template,
        "_assert_import_origin",
        fake_assert_import_origin,
    )
    monkeypatch.setattr(finalize_ubc_latent_stores, "main", fake_finalizer)
    prior_sys_path = list(sys.path)
    try:
        assert finalizer_template.main() == 0
    finally:
        sys.path[:] = prior_sys_path
    assert tuple(path.name for path in working.iterdir()) == ("dataset",)
    assert "--pair-root" in cast("tuple[str, ...]", observed["arguments"])


def test_shell_build_and_preflight_pilot_use_guarded_local_paths(
    repo_tmp: Path,
) -> None:
    """Exercise both positive shell branches without contacting Kaggle."""
    contract, receipt = _pilot_inputs(repo_tmp / "sources")
    bundle = repo_tmp / "bundle"
    authority = repo_tmp / "authority"
    bundle.mkdir()
    authority.mkdir()
    shutil.copy2(contract, bundle / "spec0021_input_contract.json")
    shutil.copy2(receipt, authority / "input_dataset_receipt.json")
    env = os.environ.copy()
    env.update(
        {
            "EQVAE_LATENT_KERNEL_ROOT": str(repo_tmp / "kernels"),
            "EQVAE_LATENT_INPUT_BUNDLE_DIR": str(bundle),
            "EQVAE_LATENT_INPUT_AUTHORITY_DIR": str(authority),
        },
    )
    built = subprocess.run(  # noqa: S603
        (str(SHELL), "build-latent-inference", "pilot"),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert built.returncode == 0, built.stderr
    assert "Spec 0021 latent inference pilot preflight" in built.stdout
    preflight = subprocess.run(  # noqa: S603
        (str(SHELL), "preflight-latent-inference", "pilot"),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert preflight.returncode == 0, preflight.stderr

    (authority / "input_dataset_receipt.json").unlink()
    staged = subprocess.run(  # noqa: S603
        (str(SHELL), "build-latent-inference", "pilot"),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert staged.returncode == 0, staged.stderr
    marker = repo_tmp / "remote_called"
    fake_bin = repo_tmp / "bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    guarded_env = {
        **env,
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "KAGGLE_DISABLE_FRESH_OAUTH": "1",
        "KAGGLE_PUSH_CONFIRMED": "1",
        "KAGGLE_FULL_DATASET_CONFIRMED": "1",
    }
    refused = subprocess.run(  # noqa: S603
        (str(SHELL), "push", str(repo_tmp / "kernels" / "pilot")),
        cwd=REPO_ROOT,
        env=guarded_env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert refused.returncode != 0
    assert "push requires a pinned input dataset receipt" in refused.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("arguments", "environment", "message"),
    [
        (
            ("publish-latent-inputs",),
            {"KAGGLE_PUSH_CONFIRMED": "1"},
            "KAGGLE_DATASET_WRITE_CONFIRMED=1",
        ),
        (
            ("publish-latent-inputs",),
            {"KAGGLE_DATASET_WRITE_CONFIRMED": "1"},
            "KAGGLE_PUSH_CONFIRMED=1",
        ),
        (
            ("publish-latent-inputs",),
            {
                "KAGGLE_PUSH_CONFIRMED": "true",
                "KAGGLE_DATASET_WRITE_CONFIRMED": "1",
            },
            "KAGGLE_PUSH_CONFIRMED=1",
        ),
        (("verify-latent-inputs",), {}, "KAGGLE_REMOTE_CONFIRMED=1"),
        (
            ("preflight-latent-inference", "invalid"),
            {},
            "requires pilot, production-all, finalizer, or run-XX",
        ),
        (
            ("build-latent-inference", "invalid"),
            {},
            "requires pilot, production-all, finalizer, or resume XX",
        ),
        (
            ("publish-latent-resume", "01"),
            {"KAGGLE_PUSH_CONFIRMED": "1"},
            "KAGGLE_DATASET_WRITE_CONFIRMED=1",
        ),
        (
            ("publish-latent-resume", "01"),
            {"KAGGLE_DATASET_WRITE_CONFIRMED": "1"},
            "KAGGLE_PUSH_CONFIRMED=1",
        ),
        (("verify-latent-resume", "01"), {}, "KAGGLE_REMOTE_CONFIRMED=1"),
    ],
)
def test_delivery_guards_refuse_before_any_remote_command(
    tmp_path: Path,
    arguments: tuple[str, ...],
    environment: Mapping[str, str],
    message: str,
) -> None:
    """Only the exact confirmation value one may reach a Kaggle command."""
    marker = tmp_path / "remote_called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    env = os.environ.copy()
    for name in (
        "KAGGLE_PUSH_CONFIRMED",
        "KAGGLE_DATASET_WRITE_CONFIRMED",
        "KAGGLE_REMOTE_CONFIRMED",
        "KAGGLE_FULL_DATASET_CONFIRMED",
    ):
        env.pop(name, None)
    env.update(environment)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["KAGGLE_DISABLE_FRESH_OAUTH"] = "1"
    result = subprocess.run(  # noqa: S603
        (str(SHELL), *arguments),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not marker.exists()


def test_delivery_shell_is_syntactically_valid() -> None:
    """Keep new case branches parseable before any local or remote action."""
    subprocess.run(  # noqa: S603
        (BASH, "-n", str(SHELL)),
        check=True,
        cwd=REPO_ROOT,
        shell=False,
    )
