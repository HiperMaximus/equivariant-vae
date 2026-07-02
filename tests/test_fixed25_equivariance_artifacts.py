# Copyright 2026 HiperMaximus
"""Focused CPU tests for the Spec 0010 fixed-25 equivariance protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from eqvae.benchmarking.io import JsonObject

from eqvae.artifacts.fixed25_equivariance import (
    EQUIVARIANCE_25_COLUMNS,
    MEASURED_K_VALUES,
    REQUIRED_EQUIVARIANCE_METRICS,
    ROT90_EXACTNESS_METRIC,
    Fixed25Config,
    Fixed25Patches,
    compute_rot90_exactness,
    evaluate_boundary,
    first3_to_rgb,
    headline_equivariance_at_k0,
    load_fixed25_patches,
    parse_fixed25_config,
    pca_to_rgb,
    rot90_k,
    validation_shard_spec_for,
    write_equivariance_csv,
    write_manifest,
    write_originals,
)
from eqvae.cli.fixed25_equivariance import main as fixed25_main
from eqvae.cli.fixed25_originals import main as fixed25_originals_main
from eqvae.data.fixed_selectors import (
    FIXED_25_VALIDATION_KIND,
    FixedSelectorGenerationContext,
    generate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard
from eqvae.data.training_batches import PatchTrainingDataset, PatchTrainingDatasetSpec
from eqvae.models.non_equivariant_vae import build_non_equivariant_vae

_PATCH_SIZE = 16
_VALIDATION_COUNT = 25
_EXPECTED_ROWS = len(REQUIRED_EQUIVARIANCE_METRICS) * len(MEASURED_K_VALUES)
_PLACEHOLDER_SELECTOR = Path("configs/spec0001/fixed_25_validation_patches.json")
_FLOAT16_TOLERANCE = 5e-2
_EXACTNESS_TOLERANCE = 1e-4
_ZERO_ABS = 1e-6
_ROT90_ROW_TOLERANCE = 1e-3
_LATENT_TOLERANCE = 1e-5
_PCA_SPATIAL = 8

_CONFIG_BLOCK: dict[str, object] = {
    "enabled": True,
    "selector_config": "configs/spec0001/fixed_25_validation_patches.json",
    "expected_count": 25,
    "expected_per_label": 5,
    "rotation": {"method": "rot90", "dims": [2, 3], "k_values": [0, 1, 2, 3]},
    "latent": {
        "transform": "rot90_scalar_field",
        "source": "posterior_mu_deterministic",
        "channels": 16,
        "spatial": 2,
    },
    "sampled_latent": {"paired_epsilon": True, "epsilon_seed": 123},
    "equivariance": {"error_eps": 1e-8},
    "save_every_boundary": True,
    "pca": {
        "methods": ["pca_top3", "first3"],
        "components": 3,
        "fit_scope": "per_image",
        "sign_convention": "unpinned",
    },
    "error_maps": {"masked": False},
    "promotable_requires_real_data": True,
}


_manual_seed = cast("Callable[[int], object]", torch.manual_seed)


def _load_pt(path: Path) -> dict[str, Tensor]:
    return cast("dict[str, Tensor]", torch.load(path, weights_only=False))


def _load_json_obj(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _config(overrides: dict[str, object] | None = None) -> Fixed25Config:
    block = {**_CONFIG_BLOCK}
    if overrides:
        block.update(overrides)
    parsed = parse_fixed25_config(
        cast("JsonObject", {"fixed25_equivariance": block}),
        default_epsilon_seed=999,
    )
    assert parsed is not None
    return parsed


def _build_validation_shard(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    write_synthetic_patch_shard(
        bin_path=root / VALIDATION_BIN_NAME,
        csv_path=root / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=_VALIDATION_COUNT,
            image_size=_PATCH_SIZE,
            channels=3,
            seed=7,
        ),
        include_idx=True,
    )
    # A complete data root also needs a train shard for the path resolver used by
    # the standalone CLI; the fixed-25 protocol only reads the validation split.
    write_synthetic_patch_shard(
        bin_path=root / TRAIN_BIN_NAME,
        csv_path=root / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=_VALIDATION_COUNT,
            image_size=_PATCH_SIZE,
            channels=3,
            seed=8,
        ),
        include_idx=False,
    )
    return root


def _write_selector(root: Path, path: Path) -> None:
    document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=PatchShardSpec(
            bin_path=root / VALIDATION_BIN_NAME,
            csv_path=root / VALIDATION_CSV_NAME,
            image_size=_PATCH_SIZE,
            # The canonical selector is CRC-validated (crc_checked=True); the
            # fixed-25 load path validates with CRC too, and crc_checked is
            # compared for equality, so fixtures must generate CRC-validated.
            validate_crc=True,
        ),
        source_split="validation",
        context=FixedSelectorGenerationContext(data_root=root),
    )
    write_fixed_selector_document(path=path, document=document)


def _validation_dataset(root: Path) -> PatchTrainingDataset:
    return PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=root / VALIDATION_BIN_NAME,
            csv_path=root / VALIDATION_CSV_NAME,
            split="validation",
            image_size=_PATCH_SIZE,
            channels=3,
        ),
    )


def _load_patches(tmp_path: Path) -> Fixed25Patches:
    root = _build_validation_shard(tmp_path)
    selector_path = tmp_path / "selector.json"
    _write_selector(root, selector_path)
    dataset = _validation_dataset(root)
    try:
        return load_fixed25_patches(
            config=_config(),
            selector_path=selector_path,
            validation_shard_spec=validation_shard_spec_for(
                validation_bin_path=root / VALIDATION_BIN_NAME,
                validation_csv_path=root / VALIDATION_CSV_NAME,
                image_size=_PATCH_SIZE,
                validate_crc=True,
            ),
            validation_dataset=dataset,
        )
    finally:
        dataset.close()


def test_rot90_exactness_and_k0_equivariance_are_zero(tmp_path: Path) -> None:
    """Exact rot90 round-trips to zero and k=0 gives zero equivariance error."""
    _manual_seed(0)
    model = build_non_equivariant_vae()
    patches = _load_patches(tmp_path)

    exactness = compute_rot90_exactness(
        model=model,
        patches=patches,
        device=torch.device("cpu"),
    )
    headline_k0 = headline_equivariance_at_k0(model=model, images=patches.images)

    assert exactness < _EXACTNESS_TOLERANCE
    assert abs(headline_k0) < _ZERO_ABS


def test_pca_to_rgb_shape_range_and_reproducible() -> None:
    """The EQ-VAE PCA visualization is a reproducible RGB image in [0, 1]."""
    _manual_seed(1)
    latent = torch.randn(1, 16, _PCA_SPATIAL, _PCA_SPATIAL)
    expected_shape = (1, 3, _PCA_SPATIAL, _PCA_SPATIAL)

    rgb = pca_to_rgb(latent)
    rgb_again = pca_to_rgb(latent)
    fallback = first3_to_rgb(latent)

    assert rgb.shape == expected_shape
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0
    assert torch.equal(rgb, rgb_again)
    assert fallback.shape == expected_shape


def test_parse_config_rejects_non_rot90_convention() -> None:
    """Parsing fails closed on a masked, interpolated, or sampled-z convention."""
    with pytest.raises(ValueError, match="masked"):
        _config({"error_maps": {"masked": True}})
    with pytest.raises(ValueError, match="k_values"):
        _config({"rotation": {"method": "rot90", "dims": [2, 3], "k_values": [0, 1]}})
    with pytest.raises(ValueError, match="posterior_mu_deterministic"):
        _config(
            {
                "latent": {
                    "transform": "rot90_scalar_field",
                    "source": "sampled_z",
                    "channels": 16,
                    "spatial": 2,
                },
            },
        )


def test_load_fixed25_fails_closed_on_placeholder(tmp_path: Path) -> None:
    """The committed placeholder selector is refused, never resampled."""
    root = _build_validation_shard(tmp_path)
    dataset = _validation_dataset(root)
    try:
        with pytest.raises(ValueError, match=r"not ready|requires_real_data"):
            load_fixed25_patches(
                config=_config(),
                selector_path=_PLACEHOLDER_SELECTOR,
                validation_shard_spec=validation_shard_spec_for(
                    validation_bin_path=root / VALIDATION_BIN_NAME,
                    validation_csv_path=root / VALIDATION_CSV_NAME,
                    image_size=_PATCH_SIZE,
                    validate_crc=True,
                ),
                validation_dataset=dataset,
            )
    finally:
        dataset.close()


def test_load_fixed25_fails_closed_on_count_mismatch(tmp_path: Path) -> None:
    """A tampered selector with fewer than 25 rows is refused."""
    root = _build_validation_shard(tmp_path)
    selector_path = tmp_path / "selector.json"
    _write_selector(root, selector_path)
    payload = _load_json_obj(selector_path)
    selectors = cast("list[object]", payload["selectors"])
    payload["selectors"] = selectors[:24]
    selector_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = _validation_dataset(root)
    try:
        with pytest.raises(ValueError, match=r"count|24|25"):
            load_fixed25_patches(
                config=_config(),
                selector_path=selector_path,
                validation_shard_spec=validation_shard_spec_for(
                    validation_bin_path=root / VALIDATION_BIN_NAME,
                    validation_csv_path=root / VALIDATION_CSV_NAME,
                    image_size=_PATCH_SIZE,
                    validate_crc=True,
                ),
                validation_dataset=dataset,
            )
    finally:
        dataset.close()


def test_fixed25_load_requires_crc_validated_selector(tmp_path: Path) -> None:
    """The fixed-25 load path validates with CRC (Option Y).

    The canonical selector is CRC-validated (``crc_checked=True``) and the real
    full run validates the fixed-25 selector shard with CRC, so a non-CRC selector
    (``crc_checked=False``) must be refused (``_validate_source`` compares
    ``crc_checked`` for equality) and a CRC-validated one must load. This pins the
    end-to-end consistency that lets a generated real selector load in the run.
    """
    root = _build_validation_shard(tmp_path)
    no_crc_selector = tmp_path / "no_crc_selector.json"
    write_fixed_selector_document(
        path=no_crc_selector,
        document=generate_fixed_selector_document(
            selector_kind=FIXED_25_VALIDATION_KIND,
            shard_spec=PatchShardSpec(
                bin_path=root / VALIDATION_BIN_NAME,
                csv_path=root / VALIDATION_CSV_NAME,
                image_size=_PATCH_SIZE,
                validate_crc=False,
            ),
            source_split="validation",
            context=FixedSelectorGenerationContext(data_root=root),
        ),
    )
    shard_spec = validation_shard_spec_for(
        validation_bin_path=root / VALIDATION_BIN_NAME,
        validation_csv_path=root / VALIDATION_CSV_NAME,
        image_size=_PATCH_SIZE,
        validate_crc=True,
    )
    dataset = _validation_dataset(root)
    try:
        with pytest.raises(ValueError, match="crc_checked"):
            load_fixed25_patches(
                config=_config(),
                selector_path=no_crc_selector,
                validation_shard_spec=shard_spec,
                validation_dataset=dataset,
            )
        crc_selector = tmp_path / "crc_selector.json"
        _write_selector(root, crc_selector)
        patches = load_fixed25_patches(
            config=_config(),
            selector_path=crc_selector,
            validation_shard_spec=shard_spec,
            validation_dataset=dataset,
        )
        assert patches.images.shape[0] == _VALIDATION_COUNT
    finally:
        dataset.close()


def test_evaluate_boundary_writes_artifacts_and_rows(tmp_path: Path) -> None:
    """A boundary writes every required artifact and a full equivariance table."""
    _manual_seed(2)
    model = build_non_equivariant_vae()
    patches = _load_patches(tmp_path)
    fixed25_dir = tmp_path / "out" / "fixed25"
    write_originals(fixed25_dir=fixed25_dir, patches=patches)

    rows = evaluate_boundary(
        model=model,
        patches=patches,
        config=_config(),
        fixed25_dir=fixed25_dir,
        optimizer_step=6250,
        device=torch.device("cpu"),
        data_source="synthetic",
        promotable=False,
    )

    boundary = fixed25_dir / "boundary_006250"
    assert (fixed25_dir / "originals.pt").exists()
    assert (fixed25_dir / "originals.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (boundary / "reconstruction_progress.pt").exists()
    assert (boundary / "latent_mu.pt").exists()
    assert (boundary / "grids" / "rotated_input_vs_latent_grid.png").exists()
    assert (boundary / "latent_pca_eqvae_style.png").exists()
    assert (boundary / "latent_first3.png").exists()
    for degrees in (90, 180, 270):
        assert (boundary / f"rotated_angle_{degrees}.pt").exists()
        assert (boundary / f"error_maps_angle_{degrees}.pt").exists()

    assert len(rows) == _EXPECTED_ROWS
    assert {row["metric_name"] for row in rows} == set(REQUIRED_EQUIVARIANCE_METRICS)
    assert {row["angle_degrees"] for row in rows} == {"90", "180", "270"}
    assert all(row["n"] == "25" for row in rows)
    assert all(row["promotable"] == "false" for row in rows)
    for row in rows:
        if row["metric_name"] == ROT90_EXACTNESS_METRIC:
            assert abs(float(row["value"])) < _ROT90_ROW_TOLERANCE


def test_embedding_rotation_uses_deterministic_mu(tmp_path: Path) -> None:
    """The rotated-embedding reconstruction is D(rot90(mu)), not a sampled z path."""
    _manual_seed(3)
    model = build_non_equivariant_vae()
    model.eval()
    patches = _load_patches(tmp_path)
    fixed25_dir = tmp_path / "out" / "fixed25"

    evaluate_boundary(
        model=model,
        patches=patches,
        config=_config(),
        fixed25_dir=fixed25_dir,
        optimizer_step=0,
        device=torch.device("cpu"),
        data_source="synthetic",
        promotable=False,
    )

    saved = _load_pt(fixed25_dir / "boundary_000000" / "rotated_angle_90.pt")
    with torch.no_grad():
        mu, _ = model.encode(patches.images)
        expected = model.decode(rot90_k(mu, 1))
    embedding_recon = saved["rotated_embedding_reconstruction"].to(torch.float32)
    assert torch.allclose(embedding_recon, expected, atol=_FLOAT16_TOLERANCE)


def test_image_and_latent_share_one_rotation_k(tmp_path: Path) -> None:
    """The image and latent paths apply the same k read from the manifest."""
    _manual_seed(4)
    model = build_non_equivariant_vae()
    model.eval()
    patches = _load_patches(tmp_path)
    fixed25_dir = tmp_path / "out" / "fixed25"

    evaluate_boundary(
        model=model,
        patches=patches,
        config=_config(),
        fixed25_dir=fixed25_dir,
        optimizer_step=0,
        device=torch.device("cpu"),
        data_source="synthetic",
        promotable=False,
    )
    write_manifest(
        fixed25_dir=fixed25_dir,
        config=_config(),
        patches=patches,
        data_source="synthetic",
        promotable=False,
        rot90_exactness_error=0.0,
    )

    manifest = _load_json_obj(fixed25_dir / "manifest.json")
    rotation = cast("dict[str, object]", manifest["rotation"])
    assert rotation["method"] == "rot90"
    assert rotation["k_values"] == [0, 1, 2, 3]

    rotated = _load_pt(fixed25_dir / "boundary_000000" / "rotated_angle_90.pt")
    latent = _load_pt(fixed25_dir / "boundary_000000" / "latent_mu.pt")
    ground_truth = rotated["ground_truth"].to(torch.float32)
    assert torch.allclose(
        ground_truth,
        rot90_k(patches.images, 1),
        atol=_FLOAT16_TOLERANCE,
    )
    with torch.no_grad():
        mu, _ = model.encode(patches.images)
    assert torch.allclose(
        latent["rotated_latent_of_mu_90"],
        rot90_k(mu, 1),
        atol=_LATENT_TOLERANCE,
    )


def test_equivariance_csv_round_trip(tmp_path: Path) -> None:
    """The equivariance CSV keeps the required columns and metric names."""
    _manual_seed(5)
    model = build_non_equivariant_vae()
    patches = _load_patches(tmp_path)
    rows = evaluate_boundary(
        model=model,
        patches=patches,
        config=_config(),
        fixed25_dir=tmp_path / "fixed25",
        optimizer_step=6250,
        device=torch.device("cpu"),
        data_source="synthetic",
        promotable=False,
    )
    csv_path = tmp_path / "metrics" / "equivariance_25.csv"
    write_equivariance_csv(path=csv_path, rows=rows)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    assert fieldnames == EQUIVARIANCE_25_COLUMNS
    assert {row["metric_name"] for row in loaded} == set(REQUIRED_EQUIVARIANCE_METRICS)
    assert all(row["n"] == "25" for row in loaded)


def test_standalone_cli_runs_over_checkpoint(tmp_path: Path) -> None:
    """The standalone CLI evaluates a checkpoint and writes the fixed-25 tree."""
    _manual_seed(6)
    root = _build_validation_shard(tmp_path)
    selector_path = tmp_path / "selector.json"
    _write_selector(root, selector_path)
    model = build_non_equivariant_vae()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seeds": {"latent_seed": 20260612},
                "data": {"image_size": _PATCH_SIZE},
                "model": {"normalization": {"num_groups": 8}},
                "fixed25_equivariance": _CONFIG_BLOCK,
            },
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "standalone"

    status = fixed25_main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(root),
            "--selector",
            str(selector_path),
            "--optimizer-step",
            "0",
        ],
    )

    assert status == 0
    assert (output_dir / "metrics" / "equivariance_25.csv").exists()
    fixed25_dir = output_dir / "artifacts" / "fixed25"
    assert (fixed25_dir / "originals.pt").exists()
    assert (fixed25_dir / "boundary_000000" / "latent_mu.pt").exists()
    manifest = _load_json_obj(fixed25_dir / "manifest.json")
    # Safe default without --promotable: artifacts are synthetic / non-promotable.
    assert manifest["data_source"] == "synthetic"
    assert manifest["promotable"] is False


def test_originals_cli_writes_selected_patches(tmp_path: Path) -> None:
    """The originals CLI saves the 25 selected patches without a checkpoint."""
    _manual_seed(7)
    root = _build_validation_shard(tmp_path)
    selector_path = tmp_path / "selector.json"
    _write_selector(root, selector_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seeds": {"latent_seed": 20260612},
                "data": {"image_size": _PATCH_SIZE},
                "fixed25_equivariance": _CONFIG_BLOCK,
            },
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "originals"

    status = fixed25_originals_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(root),
            "--selector",
            str(selector_path),
        ],
    )

    assert status == 0
    fixed25_dir = output_dir / "artifacts" / "fixed25"
    assert (fixed25_dir / "originals.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    saved = _load_pt(fixed25_dir / "originals.pt")
    assert saved["images"].shape[0] == _VALIDATION_COUNT
    assert len(saved["identities"]) == _VALIDATION_COUNT
    # No boundary/model artifacts: this step archives only the selected originals.
    assert not (fixed25_dir / "boundary_000000").exists()
    assert not (fixed25_dir / "manifest.json").exists()


def test_originals_cli_fails_closed_on_placeholder(tmp_path: Path) -> None:
    """Without an explicit selector, the tracked placeholder is refused."""
    root = _build_validation_shard(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seeds": {"latent_seed": 20260612},
                "data": {"image_size": _PATCH_SIZE},
                "fixed25_equivariance": _CONFIG_BLOCK,
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"not ready|requires_real_data"):
        fixed25_originals_main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--data-root",
                str(root),
            ],
        )
