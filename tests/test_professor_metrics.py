# pyright: reportAny=false
# Copyright 2026 HiperMaximus
# ruff: noqa: PLR2004
"""Focused acceptance tests for the Spec 0044 local renderer."""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from eqvae.evaluation import professor_metrics
from eqvae.evaluation.professor_metrics import (
    DEFAULT_INPUT_CONTRACT,
    Fixed25Data,
    build_professor_metrics_package,
    centered_moving_average,
    compute_fixed25_metrics,
    load_input_contract,
    validate_physical_skip_rows,
    validate_session_layout,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def _sha256(path: Path) -> str:
    """Return a fixture file digest.

    Returns:
        Hexadecimal SHA-256.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic fixture JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write one compact fixture CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_small_repository(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Path]]:
    """Build a compact repository that exercises every acceptance path.

    Returns:
        Repository root, input contract path, and representative mutation targets.

    """
    repository = tmp_path / "repository"
    monkeypatch.setattr(professor_metrics, "FIXED25_COUNT", 2)
    monkeypatch.setattr(professor_metrics, "FINAL_COUNTER", 3)
    monkeypatch.setattr(professor_metrics, "IMAGE_SPATIAL_SIZE", 16)
    monkeypatch.setattr(professor_metrics, "EXPECTED_ANGLES", (90,))

    selector_path = repository / "configs/spec0001/fixed_25_validation_patches.json"
    _write_json(selector_path, {"fixture": True})
    identities = [
        {
            "rank": index,
            "sample_id": f"validation:fixture:{index}",
            "wsi_id": str(index + 1),
            "label": index,
            "x": index,
            "y": index + 2,
            "file_index": index,
            "row_index": index,
        }
        for index in range(2)
    ]
    originals = torch.arange(2 * 3 * 16 * 16, dtype=torch.int64).remainder(256)
    originals = originals.reshape(2, 3, 16, 16).to(torch.uint8)
    target = originals.to(torch.float32).div(255.0).mul(2.0).sub(1.0)
    originals_payload = {
        "images_uint8": originals,
        "images_domain": "uint8[0,255]; model_domain = x / 255 * 2 - 1",
        "identities": identities,
        "selector_sha256": _sha256(selector_path),
    }

    train_fields = [
        "event_id",
        "rank",
        "successful_optimizer_update_count",
        "loss",
        "l1_loss",
        "ssim_loss",
        "learning_rate",
        "grad_norm",
        "param_update_norm",
        "nonfinite_count",
        "amp_step_skipped",
    ]
    validation_fields = [
        "rank",
        "optimizer_step",
        "view",
        "sample_count",
        "loss",
        "l1_loss",
        "ssim_loss",
    ]
    equivariance_fields = [
        "optimizer_step",
        "angle_degrees",
        "metric_name",
        "mean",
        "n",
        "data_source",
        "promotable",
    ]
    sessions: list[dict[str, Any]] = []
    consumed: dict[str, str] = {}
    final_hashes: dict[str, str] = {}
    mutation_target: Path | None = None
    for model_index, model in enumerate(("normal", "so2")):
        relative_root = Path("runs") / model
        root = repository / relative_root
        fixed_root = root / "artifacts/fixed25"
        fixed_root.mkdir(parents=True)
        torch.save(originals_payload, fixed_root / "originals.pt")
        _write_json(
            fixed_root / "manifest.json",
            {
                "schema": "spec0010.fixed25_equivariance.manifest.v1",
                "data_source": "real",
                "promotable": True,
                "boundary_optimizer_steps": [1, 2, 3],
                "selector": {"identities": identities},
            },
        )
        train_rows: list[dict[str, Any]] = [
            {
                "event_id": f"{model}-{rank}-{counter}",
                "rank": rank,
                "successful_optimizer_update_count": counter,
                "loss": counter + rank + model_index / 10,
                "l1_loss": counter / 10 + rank / 100,
                "ssim_loss": counter / 20 + rank / 100,
                "learning_rate": counter / 1000,
                "grad_norm": 1.0,
                "param_update_norm": 0.1,
                "nonfinite_count": 0,
                "amp_step_skipped": 0,
            }
            for rank in (0, 1)
            for counter in (1, 2, 3)
        ]
        if model == "normal":
            train_rows.append({
                "event_id": "normal-skipped",
                "rank": 0,
                "successful_optimizer_update_count": 2,
                "loss": 99,
                "l1_loss": 99,
                "ssim_loss": 99,
                "learning_rate": 0.002,
                "grad_norm": "inf",
                "param_update_norm": 0,
                "nonfinite_count": 1,
                "amp_step_skipped": 1,
            })
        _write_csv(root / "metrics/train_steps.csv", train_rows, train_fields)
        validation_rows = [
            {
                "rank": rank,
                "optimizer_step": counter,
                "view": "clean",
                "sample_count": 1 if rank == 0 else 3,
                "loss": counter + rank,
                "l1_loss": counter / 10 + rank / 10,
                "ssim_loss": counter / 20 + rank / 20,
            }
            for counter in (1, 2, 3)
            for rank in (0, 1)
        ]
        _write_csv(
            root / "metrics/validation_metrics.csv",
            validation_rows,
            validation_fields,
        )
        equivariance_rows = [
            {
                "optimizer_step": counter,
                "angle_degrees": 90,
                "metric_name": "equivariance_error_25_patches",
                "mean": model_index + counter / 10,
                "n": 2,
                "data_source": "real",
                "promotable": "true",
            }
            for counter in (1, 2, 3)
        ]
        _write_csv(
            root / "metrics/equivariance_25.csv",
            equivariance_rows,
            equivariance_fields,
        )
        _write_json(root / "benchmark/artifact_manifest.json", {"fixture": True})
        _write_json(
            root / "benchmark/selected_runtime_full_summary.json",
            {"fixture": True},
        )
        for counter in (1, 2, 3):
            boundary = fixed_root / f"boundary_{counter:06d}"
            boundary.mkdir()
            prediction = (target + (model_index + 1) * 0.02 * counter).to(torch.float16)
            reconstruction_path = boundary / "reconstruction_progress.pt"
            torch.save(
                {"optimizer_step": counter, "reconstruction": prediction},
                reconstruction_path,
            )
            consumed[
                (relative_root / reconstruction_path.relative_to(root)).as_posix()
            ] = _sha256(reconstruction_path)
            if model == "normal" and counter == 1:
                mutation_target = reconstruction_path
        rotated_ground_truth = torch.rot90(target, 1, dims=(-2, -1)).to(torch.float16)
        rotated_path = fixed_root / "boundary_000003/rotated_angle_90.pt"
        torch.save(
            {
                "angle_degrees": 90,
                "ground_truth": rotated_ground_truth,
                "rotated_input_reconstruction": (rotated_ground_truth + 0.02).to(
                    torch.float16,
                ),
                "rotated_embedding_reconstruction": (
                    rotated_ground_truth + 0.03 + model_index * 0.01
                ).to(torch.float16),
            },
            rotated_path,
        )
        consumed[(relative_root / rotated_path.relative_to(root)).as_posix()] = _sha256(
            rotated_path,
        )
        hashes = {
            relative: _sha256(root / relative)
            for relative in (
                "metrics/train_steps.csv",
                "metrics/validation_metrics.csv",
                "metrics/equivariance_25.csv",
                "artifacts/fixed25/manifest.json",
                "benchmark/artifact_manifest.json",
                "benchmark/selected_runtime_full_summary.json",
            )
        }
        final_reconstruction = fixed_root / "boundary_000003/reconstruction_progress.pt"
        final_hashes[model] = _sha256(final_reconstruction)
        sessions.append({
            "model": model,
            "path": relative_root.as_posix(),
            "counter_first": 1,
            "counter_last": 3,
            "committed_rank_rows": 6,
            "amp_skipped_attempt_rows": 1 if model == "normal" else 0,
            "hashes": hashes,
        })

    pca_path = (
        repository
        / "runs/local/frozen_vae_rotation_orbits/05-paper-style-latent-pca.png"
    )
    pca_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(pca_path)
    pca_relative = pca_path.relative_to(repository).as_posix()
    consumed[pca_relative] = _sha256(pca_path)
    contract = {
        "schema_version": 1,
        "successful_update_counter": {
            "first": 1,
            "last": 3,
            "validation_boundaries": [1, 2, 3],
        },
        "session_count_by_model": {"normal": 1, "so2": 1},
        "accepted_committed_physical_skip": [],
        "fixed25": {
            "selector_sha256": _sha256(selector_path),
            "originals_sha256": _sha256(
                repository / "runs/normal/artifacts/fixed25/originals.pt",
            ),
            "rotated_grid_selector_rank": 1,
            "rotated_grid_sample_id": identities[1]["sample_id"],
            "pca_reference_path": pca_relative,
            "normal_final_reconstruction_sha256": final_hashes["normal"],
            "so2_final_reconstruction_sha256": final_hashes["so2"],
        },
        "consumed_artifacts_sha256": consumed,
        "sessions": sessions,
    }
    contract_path = repository / "contract.json"
    _write_json(contract_path, contract)
    assert mutation_target is not None
    mutation_targets = {
        "boundary": mutation_target,
        "session_csv": repository / "runs/normal/metrics/train_steps.csv",
        "rotated": (
            repository
            / "runs/normal/artifacts/fixed25/boundary_000003/rotated_angle_90.pt"
        ),
        "pca": pca_path,
    }
    return repository, contract_path, mutation_targets


def test_canonical_contract_is_hash_valid_and_complete() -> None:
    """The tracked contract must remain byte-pinned and cover all ten sessions."""
    contract = load_input_contract(DEFAULT_INPUT_CONTRACT)
    validate_session_layout(contract)
    assert len(contract["sessions"]) == 10
    assert [session["model"] for session in contract["sessions"]].count("normal") == 3
    assert [session["model"] for session in contract["sessions"]].count("so2") == 7


def test_session_layout_rejects_a_mirror_and_range_gap() -> None:
    """Mirror or range drift would double-count or omit scientific history."""
    contract = load_input_contract(DEFAULT_INPUT_CONTRACT)
    mirrored = deepcopy(contract)
    mirrored["sessions"][0]["path"] += "_remote"
    with pytest.raises(ValueError, match="Mirror session"):
        validate_session_layout(mirrored)

    gapped = deepcopy(contract)
    gapped["sessions"][1]["counter_first"] += 1
    with pytest.raises(ValueError, match="Non-contiguous"):
        validate_session_layout(gapped)


def test_fixed25_metrics_use_locked_domains_and_paired_deltas() -> None:
    """A known tensor pair protects the normalized/image-domain metric boundary."""
    target = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    normal = torch.full_like(target, 0.2)
    so2 = torch.full_like(target, -0.4)
    identities = [
        {
            "rank": index,
            "sample_id": f"sample:{index}",
            "wsi_id": "1",
            "label": index,
            "x": 0,
            "y": 0,
        }
        for index in range(2)
    ]
    fixed = Fixed25Data(
        originals_uint8=torch.zeros_like(target, dtype=torch.uint8),
        target_norm=target,
        predictions={"normal": normal, "so2": so2},
        identities=identities,
        final_roots={"normal": Path("normal"), "so2": Path("so2")},
        source_hashes={},
    )
    rows, summary = compute_fixed25_metrics(fixed)
    assert len(rows) == 4
    assert rows[0]["mae_norm"] == pytest.approx(0.2)
    assert rows[0]["mse_norm"] == pytest.approx(0.04)
    assert rows[0]["psnr_img_db"] == pytest.approx(20.0, abs=1e-5)
    assert summary["models"]["normal"]["mae_norm"]["std_population"] == pytest.approx(
        0.0,
    )
    assert summary["paired_deltas"]["normal_minus_so2_mae_norm"][
        "mean"
    ] == pytest.approx(-0.2)


def test_centered_smoother_is_display_only_and_rejects_even_window() -> None:
    """Display smoothing must be centered, deterministic, and non-mutating."""
    x = np.arange(1, 8, dtype=np.float64)
    y = np.asarray([1.0, 2.0, 6.0, 4.0, 5.0, 9.0, 1.0], dtype=np.float64)
    original = y.copy()
    smooth_x, smooth_y = centered_moving_average(x, y, 3)
    np.testing.assert_array_equal(smooth_x, np.asarray([2, 3, 4, 5, 6]))
    np.testing.assert_allclose(smooth_y, np.convolve(y, np.ones(3) / 3, mode="valid"))
    np.testing.assert_array_equal(y, original)
    with pytest.raises(ValueError, match="odd"):
        centered_moving_average(x, y, 4)


def test_physical_skip_validator_accepts_only_locked_signature() -> None:
    """Only the two-rank legacy counter-14,007 exception may enter a train curve."""
    accepted = [
        ("normal", 0, 14_007, float("inf"), 0.0, 1),
        ("normal", 1, 14_007, float("inf"), 0.0, 1),
    ]
    validate_physical_skip_rows(accepted)
    with pytest.raises(ValueError, match="Unexpected committed physical-skip"):
        validate_physical_skip_rows([
            *accepted,
            ("so2", 0, 20_000, float("inf"), 0.0, 1),
        ])


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    """A rerun must preserve every byte in an already accepted directory."""
    output = tmp_path / "accepted"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_professor_metrics_package(
            input_contract=tmp_path / "does-not-exist.json",
            output_dir=output,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_fixture_package_is_atomic_hash_bound_and_mutation_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove one full build, one lost output race, and rejection of source drift."""
    repository, contract_path, mutation_targets = _build_small_repository(
        tmp_path,
        monkeypatch,
    )
    output = repository / "runs/local/package"

    def build() -> str:
        try:
            build_professor_metrics_package(
                input_contract=contract_path,
                output_dir=output,
                repository_root=repository,
                smoothing_window=3,
                require_canonical_contract=False,
            )
        except FileExistsError:
            return "refused"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build) for _index in range(2)]
        outcomes = sorted(future.result() for future in futures)
    assert outcomes == ["published", "refused"]
    manifest = cast(
        "dict[str, Any]",
        json.loads((output / "manifest.json").read_text(encoding="utf-8")),
    )
    status = cast(
        "dict[str, Any]",
        json.loads((output / "status.json").read_text(encoding="utf-8")),
    )
    assert _sha256(output / "manifest.json") == status["manifest_sha256"]
    for relative, expected in manifest["output_files_sha256"].items():
        assert _sha256(output / relative) == expected
    with (output / "metrics/training_dashboard_series.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    normal_train_one = next(
        row
        for row in rows
        if row["model"] == "normal"
        and row["series"] == "train_rank_mean"
        and row["recorded_successful_update_counter"] == "1"
    )
    assert float(normal_train_one["objective"]) == pytest.approx(1.5)
    normal_validation_one = next(
        row
        for row in rows
        if row["model"] == "normal"
        and row["series"] == "clean_validation"
        and row["recorded_successful_update_counter"] == "1"
    )
    assert float(normal_validation_one["objective"]) == pytest.approx(1.75)
    with (output / "metrics/amp_skipped_attempts.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        skipped = list(csv.DictReader(handle))
    assert [(row["model"], row["event_id"]) for row in skipped] == [
        ("normal", "normal-skipped"),
    ]

    for mutation_class, mutation_target in mutation_targets.items():
        original_bytes = mutation_target.read_bytes()
        mutation_target.write_bytes(original_bytes + b"drift")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            build_professor_metrics_package(
                input_contract=contract_path,
                output_dir=repository / f"runs/local/mutated-{mutation_class}",
                repository_root=repository,
                smoothing_window=3,
                require_canonical_contract=False,
            )
        mutation_target.write_bytes(original_bytes)
