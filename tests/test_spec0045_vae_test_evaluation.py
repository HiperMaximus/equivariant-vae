# Copyright 2026 HiperMaximus
# pyright: reportPrivateUsage=false
# ruff: noqa: D103, PLC2701, PLR2004, SLF001
"""Focused population, latent-reader, and metric tests for Spec 0045."""

from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path
from typing import cast

import pytest
import scripts.build_vae_test_evaluation as builder
import torch

from eqvae.data.latent_shards import (
    LATENT_RECORD_BYTES,
    LATENT_SHARD_HEADER_SIZE,
    LATENT_VALUES,
    make_latent_shard_header,
)
from eqvae.evaluation import vae_test_runtime
from eqvae.evaluation.vae_test import (
    EXPECTED_WSI_PATCH_COUNTS,
    REDACTED_LOCATION_HEADER,
    load_test_locations,
    paired_reconstruction_metrics,
    reconstruction_metrics,
    redact_test_locations,
    sha256_file,
)
from eqvae.evaluation.vae_test_runtime import (
    _LatentReader,
    _publish_completed_output,
)

SOURCE = Path(
    "runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/views/"
    "cancer_test_locations.csv",
)
ORACLE = Path("runs/local/ubc_ocean_eval_consumption/cancer_test.csv")
SOURCE_SHA256 = "3599baeb4b70d0414e3f7fa9bf8a359c91b2433613dde186b58f6a711f95e66b"
ORACLE_SHA256 = "1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc"
REDACTED_SHA256 = "81f7897d1e4373b7250f14a70c63e865a3026f81c2a99acf6e65c3e331119a7a"
ROW_IDENTITY_SHA256 = "0f7fc4f01961bdb4f2dd71d7c1b7cd0455e6bb175d7dff095a88bfaa27e34fd5"


def test_redaction_reproduces_exact_label_free_test_population(tmp_path: Path) -> None:
    output = tmp_path / "vae_test_locations.csv"
    record = redact_test_locations(
        source_path=SOURCE,
        oracle_path=ORACLE,
        output_path=output,
        expected_source_sha256=SOURCE_SHA256,
        expected_oracle_sha256=ORACLE_SHA256,
    )
    assert record == {
        "bytes": 2_518_967,
        "row_count": 67_138,
        "row_identity_sha256": ROW_IDENTITY_SHA256,
        "sha256": REDACTED_SHA256,
        "wsi_count": 23,
    }
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REDACTED_LOCATION_HEADER
        assert len(list(reader)) == 67_138
    assert not {
        "diagnosis_label",
        "diagnosis_index",
        "tissue_label",
    } & set(REDACTED_LOCATION_HEADER)


def test_location_loader_rejects_byte_and_population_mutations(tmp_path: Path) -> None:
    output = tmp_path / "locations.csv"
    output.write_bytes(SOURCE.read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        load_test_locations(
            output,
            expected_sha256="0" * 64,
            expected_row_count=67_138,
            expected_wsi_patch_counts=EXPECTED_WSI_PATCH_COUNTS,
        )

    tiny = tmp_path / "tiny.csv"
    tiny.write_text(
        ",".join(REDACTED_LOCATION_HEADER) + "\n1,0,0,1,0,0,test\n1,1,1,1,128,0,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invariant"):
        load_test_locations(
            tiny,
            expected_sha256=sha256_file(tiny),
            expected_row_count=2,
            expected_wsi_patch_counts={1: 2},
        )


def test_metric_contract_uses_normalized_and_projected_image_domains() -> None:
    target = torch.linspace(-1.0, 1.0, 16 * 16, dtype=torch.float32).reshape(
        1,
        1,
        16,
        16,
    )
    exact = reconstruction_metrics(
        reconstruction=target,
        target_normalized=target,
    )
    assert exact["mae_norm"].item() == pytest.approx(0.0)
    assert exact["mse_norm"].item() == pytest.approx(0.0)
    assert torch.isposinf(exact["psnr_img"]).all()
    assert exact["ssim_img"].item() == pytest.approx(1.0)

    shifted = torch.clamp(target + 0.2, -1.0, 1.0)
    paired = paired_reconstruction_metrics(
        normal_reconstruction=target,
        so2_reconstruction=shifted,
        target_normalized=target,
    )
    assert tuple(paired) == (
        "normal_mae_norm",
        "normal_mse_norm",
        "normal_psnr_img",
        "normal_ssim_img",
        "so2_mae_norm",
        "so2_mse_norm",
        "so2_psnr_img",
        "so2_ssim_img",
    )
    assert paired["so2_mae_norm"].item() > 0.0
    assert paired["so2_psnr_img"].item() < float("inf")


def test_latent_reader_resolves_exact_fp32_record(tmp_path: Path) -> None:
    values = torch.arange(LATENT_VALUES, dtype=torch.float32)
    payload = values.numpy().tobytes(order="C")
    binary = tmp_path / "latent.bin"
    binary.write_bytes(
        make_latent_shard_header(
            tensor_count=1,
            payload_crc32=zlib.crc32(payload),
        )
        + payload,
    )
    assert binary.stat().st_size == LATENT_SHARD_HEADER_SIZE + LATENT_RECORD_BYTES
    reader = _LatentReader(binary, expected_bytes=binary.stat().st_size)
    observed = reader[0].clone()
    reader.close()
    assert observed.shape == (16, 32, 32)
    assert torch.equal(observed.flatten(), values)
    with pytest.raises(IndexError):
        reader[1]


def test_remote_input_contract_never_serializes_nonfinite_json() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        json.dumps({"bad": float("inf")}, allow_nan=False)


def test_remote_publication_writes_authenticated_status_last(tmp_path: Path) -> None:
    staging = tmp_path / ".building"
    output = tmp_path / "accepted"
    staging.mkdir()
    for name in (
        "per_patch_metrics.csv.gz",
        "run_contract.json",
        "runtime.json",
        "wsi_evidence.json",
    ):
        (staging / name).write_bytes(name.encode())
    manifest = _publish_completed_output(
        staging=staging,
        output=output,
        row_count=67_138,
        wsi_count=23,
    )
    status = cast(
        "dict[str, object]",
        json.loads((output / "status.json").read_text(encoding="utf-8")),
    )
    output_files = cast("dict[str, object]", manifest["output_files"])
    assert not staging.exists()
    assert output_files.keys() == {
        "per_patch_metrics.csv.gz",
        "run_contract.json",
        "runtime.json",
        "wsi_evidence.json",
    }
    assert status["status"] == "pass"
    assert status["manifest_sha256"] == sha256_file(output / "manifest.json")
    assert "status.json" not in output_files


def test_remote_publication_failure_never_exposes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".building"
    output = tmp_path / "accepted"
    staging.mkdir()
    for name in (
        "per_patch_metrics.csv.gz",
        "run_contract.json",
        "runtime.json",
        "wsi_evidence.json",
    ):
        (staging / name).write_bytes(name.encode())
    original_write_json = vae_test_runtime._write_json

    def fail_on_status(path: Path, value: object) -> None:
        if path.name == "status.json":
            message = "injected status failure"
            raise OSError(message)
        original_write_json(path, value)

    monkeypatch.setattr(vae_test_runtime, "_write_json", fail_on_status)
    with pytest.raises(OSError, match="injected status failure"):
        _publish_completed_output(
            staging=staging,
            output=output,
            row_count=67_138,
            wsi_count=23,
        )
    assert not output.exists()


def test_built_input_is_exact_label_free_and_state_only() -> None:
    root = Path("runs/local/vae_test_evaluation_input")
    result = builder.validate_input(actor="maximshtefan")
    assert result["input_contract_sha256"] == (
        "b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1"
    )
    contract = cast(
        "dict[str, object]",
        json.loads((root / builder.CONTRACT_NAME).read_text(encoding="utf-8")),
    )
    assert set(cast("dict[str, object]", contract["files"])) == {
        "vae_test_locations.csv",
        "normal_vae_state.pt",
        "so2_vae_state.pt",
    }
    forbidden = {"diagnosis", "tissue", "label", "truth", "target", "oracle"}
    header = (
        (root / "vae_test_locations.csv")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()[0]
    )
    assert not any(term in header.lower() for term in forbidden)
    for branch in ("normal_vae", "so2_vae"):
        payload = cast(
            "dict[str, torch.Tensor]",
            torch.load(
                root / f"{branch}_state.pt",
                map_location="cpu",
                weights_only=True,
            ),
        )
        assert all(isinstance(value, torch.Tensor) for value in payload.values())


def test_uploadable_kernel_is_compact_and_binds_final_input() -> None:
    kernel = Path("kaggle/kernels/vae_test_reconstruction/run.py")
    source = kernel.read_text(encoding="utf-8")
    assert kernel.stat().st_size == 77_593
    assert kernel.stat().st_size < 1_000_000
    assert 'INPUT_CONTRACT_SHA256 = "b5a32ebf' in source
    assert "SPEC0045_VAE_TEST_RECONSTRUCTION_READY = True" in source
    compile(source, str(kernel), "exec")


def test_shell_route_requires_private_receipt_and_one_use_claim() -> None:
    source = Path("scripts/kaggle_kernel.sh").read_text(encoding="utf-8")
    assert 'KAGGLE_VAE_TEST_ROUTE_ACTIVE=1 "$0" push' in source
    assert '"${KAGGLE_VAE_TEST_EVALUATION_CONFIRMED:-}" != "1"' in source
    assert '[[ -f "$vae_test_input_receipt" ]]' in source
    assert '[[ ! -e "$vae_test_launch_claim" ]]' in source
    assert "validate-claimed-launch --actor" in source
    assert '"schema_version": "spec0045.input_dataset_receipt.v1"' in source
    assert 'with receipt.open("x"' in source
    assert (
        'vae_test_accepted_reference="maximshtefan/'
        'eqvae-frozen-vae-full-test-reconstruction/1"' in source
    )
    assert (
        'vae_test_launch_receipt_sha256="bd7360a9ec6831b107b7b2235cfae370'
        '60553db6434f064d949e0129788c6f3f"' in source
    )
    assert '"$reference" != "$vae_test_accepted_reference"' in source
    assert '"$receipt_sha256" != "$vae_test_launch_receipt_sha256"' in source
