# Copyright 2026 HiperMaximus
"""Tests for the spec 0001 HED stain corruptor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import skimage.color as skimage_color
import torch

from eqvae.benchmarking.stain_corruptor_qa import (
    LocalStainCorruptorQaRequest,
    write_local_stain_corruptor_qa,
)
from eqvae.corruption.stain import (
    BRANCHLESS_ALL_STRATEGY,
    CONSERVATIVE_DEFAULT_PROFILE,
    CORRUPTION_VERSION,
    FSQ_LEGACY_WIDE_PROFILE,
    INDEXED_MASKED_STRATEGY,
    StainCorruptionParameters,
    StainCorruptionProfile,
    StainCorruptor,
    clean_validation_passthrough,
    corrupt_normalized_batch,
    derive_corruption_seed,
    hed_to_rgb,
    profile_from_config,
    profile_from_name,
    rgb_to_hed,
    sample_corruption_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from eqvae.config import JsonObject

EXPECTED_QA_COUNT = 25
MASK_CARDINALITY_BATCH_SIZE = 4
HELPER_STRATEGY_BATCH_SIZE = 2


def test_rgb_to_hed_matches_scikit_oracle_channel_first() -> None:
    """Torch NCHW conversion matches channel-last scikit-image `rgb2hed`."""
    rgb_fixture = (
        ((0.2, 0.4, 0.8), (0.9, 0.7, 0.3)),
        ((1.0, 1.0, 1.0), (1.0e-6, 0.5, 0.25)),
    )
    rgb_hwc = _as_float64_array(rgb_fixture)
    expected = _scikit_rgb2hed(rgb_hwc)
    rgb_nchw = torch.tensor(rgb_fixture, dtype=torch.float64).permute(2, 0, 1)
    rgb_nchw = rgb_nchw.unsqueeze(0)

    actual = rgb_to_hed(rgb_nchw).squeeze(0).permute(1, 2, 0).numpy()

    np.testing.assert_allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_hed_to_rgb_matches_scikit_oracle_channel_first() -> None:
    """Torch NCHW inverse conversion matches scikit-image `hed2rgb`."""
    hed_fixture = (
        ((0.0, 0.05, 0.02), (0.1, 0.0, 0.03)),
        ((0.2, 0.1, 0.0), (0.4, 0.05, 0.02)),
    )
    hed_hwc = _as_float64_array(hed_fixture)
    expected = _scikit_hed2rgb(hed_hwc)
    hed_nchw = torch.tensor(hed_fixture, dtype=torch.float64).permute(2, 0, 1)
    hed_nchw = hed_nchw.unsqueeze(0)

    actual = hed_to_rgb(hed_nchw).squeeze(0).permute(1, 2, 0).numpy()

    np.testing.assert_allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_valid_hed_manifold_rgb_round_trip_is_stable() -> None:
    """Identity stain parameters round-trip RGB generated from nonnegative HED."""
    hed = torch.tensor(
        [
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[0.05, 0.0], [0.1, 0.2]],
                [[0.02, 0.03], [0.0, 0.04]],
            ],
        ],
        dtype=torch.float64,
    )
    rgb = hed_to_rgb(hed)

    round_tripped = hed_to_rgb(rgb_to_hed(rgb))

    torch.testing.assert_close(round_tripped, rgb, atol=1.0e-6, rtol=1.0e-6)


def test_profile_parser_locks_default_and_fsq_wide_values() -> None:
    """Config profile fields must match the named locked corruption profile."""
    conservative = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)
    fsq_wide = profile_from_name(FSQ_LEGACY_WIDE_PROFILE)

    assert conservative.he_alpha_range == (0.80, 1.20)
    assert conservative.he_beta_range == (-0.05, 0.05)
    assert conservative.residual_alpha_range == (0.98, 1.02)
    assert fsq_wide.he_alpha_range == (0.75, 1.25)
    assert fsq_wide.he_beta_range == (-0.10, 0.10)
    assert fsq_wide.residual_beta_range == (-0.01, 0.01)
    assert profile_from_config(_profile_config_payload(conservative)) == conservative


def test_profile_parser_rejects_mismatched_named_ranges() -> None:
    """Profile names cannot silently drift from their numeric ranges."""
    payload = _profile_config_payload(profile_from_name(CONSERVATIVE_DEFAULT_PROFILE))
    payload["he_beta_range"] = [-0.10, 0.10]

    with pytest.raises(ValueError, match="do not match locked profile"):
        profile_from_config(payload)


def test_semantic_seed_is_stable_and_excludes_rank() -> None:
    """Seed derivation changes with semantic fields and has no rank input."""
    base = derive_corruption_seed(
        corruption_seed=20260611,
        split="train",
        semantic_sample_key="train:wsi_a:1:10:20",
        corruption_step=7,
        corruption_view="train_corrupted",
    )

    assert base == derive_corruption_seed(
        corruption_seed=20260611,
        split="train",
        semantic_sample_key="train:wsi_a:1:10:20",
        corruption_step=7,
        corruption_view="train_corrupted",
    )
    assert base != derive_corruption_seed(
        corruption_seed=20260611,
        split="train",
        semantic_sample_key="train:wsi_a:1:10:21",
        corruption_step=7,
        corruption_view="train_corrupted",
    )
    assert base != derive_corruption_seed(
        corruption_seed=20260611,
        split="train",
        semantic_sample_key="train:wsi_a:1:10:20",
        corruption_step=8,
        corruption_view="train_corrupted",
    )


def test_sampling_is_deterministic_and_does_not_advance_global_rng() -> None:
    """Per-sample generators make corruption draws independent of global RNG."""
    profile = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)
    keys = ("train:wsi_a:1:10:20", "train:wsi_b:2:30:40")
    state = torch.get_rng_state()

    first = sample_corruption_parameters(
        batch_shape=(2, 3, 8, 8),
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=keys,
        corruption_step=0,
        corruption_view="train_corrupted",
    )
    state_after_first = torch.get_rng_state()
    second = sample_corruption_parameters(
        batch_shape=(2, 3, 8, 8),
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=keys,
        corruption_step=0,
        corruption_view="train_corrupted",
    )

    assert torch.equal(state, state_after_first)
    assert torch.equal(first.applied_mask, second.applied_mask)
    assert torch.equal(first.alpha, second.alpha)
    assert torch.equal(first.beta, second.beta)
    assert torch.equal(first.noise_std, second.noise_std)
    assert torch.equal(first.noise, second.noise)
    assert first.sample_seeds == second.sample_seeds


def test_clean_validation_passthrough_consumes_no_rng() -> None:
    """Clean validation/test views skip the corruptor entirely."""
    inputs = torch.randn((2, 3, 8, 8), dtype=torch.float32)
    state = torch.get_rng_state()

    outputs = clean_validation_passthrough(inputs)

    assert outputs is inputs
    assert torch.equal(state, torch.get_rng_state())


def test_corrupt_normalized_batch_preserves_public_contract() -> None:
    """Corrupted outputs preserve public shape, dtype, device, and range."""
    inputs = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    profile = StainCorruptionProfile(
        name="always_on_test",
        corrupt_prob=1.0,
        he_alpha_range=(0.80, 1.20),
        he_beta_range=(-0.05, 0.05),
        residual_alpha_range=(0.98, 1.02),
        residual_beta_range=(-0.01, 0.01),
        noise_std_range=(0.0, 0.05),
    )

    result = corrupt_normalized_batch(
        inputs,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=("train:wsi_a:1:10:20", "train:wsi_b:2:30:40"),
        corruption_step=0,
        corruption_view="train_corrupted",
    )

    assert result.corrupted.shape == inputs.shape
    assert result.corrupted.dtype == inputs.dtype
    assert result.corrupted.device == inputs.device
    assert float(result.corrupted.min().item()) >= -1.0
    assert float(result.corrupted.max().item()) <= 1.0
    assert all(item.applied for item in result.metadata)
    assert not torch.equal(result.corrupted, inputs)


def test_zero_probability_corruption_keeps_clean_samples_unchanged() -> None:
    """Bernoulli-off samples keep the exact clean input tensor values."""
    inputs = torch.linspace(-0.8, 0.8, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    profile = StainCorruptionProfile(
        name="always_off_test",
        corrupt_prob=0.0,
        he_alpha_range=(0.80, 1.20),
        he_beta_range=(-0.05, 0.05),
        residual_alpha_range=(0.98, 1.02),
        residual_beta_range=(-0.01, 0.01),
        noise_std_range=(0.0, 0.05),
    )

    result = corrupt_normalized_batch(
        inputs,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=("train:wsi_a:1:10:20", "train:wsi_b:2:30:40"),
        corruption_step=0,
        corruption_view="train_corrupted",
    )

    torch.testing.assert_close(result.corrupted, inputs, atol=0.0, rtol=0.0)
    assert not any(item.applied for item in result.metadata)


@pytest.mark.parametrize(
    "applied_mask",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, False, True, False),
        (True, True, True, True),
    ],
)
def test_indexed_masked_matches_branchless_public_contract(
    applied_mask: tuple[bool, ...],
) -> None:
    """Indexed corruption preserves branchless RNG and public outputs."""
    inputs = torch.linspace(
        -0.9,
        0.9,
        steps=MASK_CARDINALITY_BATCH_SIZE * 3 * 8 * 8,
    ).reshape(MASK_CARDINALITY_BATCH_SIZE, 3, 8, 8)
    keys = tuple(
        f"train:wsi_{index}:1:{10 + index}:{20 + index}"
        for index in range(MASK_CARDINALITY_BATCH_SIZE)
    )
    profile = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)
    module = StainCorruptor()
    batch_shape = cast("tuple[int, int, int, int]", tuple(inputs.shape))
    sampled = sample_corruption_parameters(
        batch_shape=batch_shape,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=keys,
        corruption_step=0,
        corruption_view="train_corrupted",
    )
    parameters = _parameters_with_mask(sampled, applied_mask=applied_mask)

    branchless = module.apply_with_parameters(
        inputs,
        parameters,
        semantic_sample_keys=keys,
        profile_name=profile.name,
        strategy=BRANCHLESS_ALL_STRATEGY,
    )
    indexed = module.apply_with_parameters(
        inputs,
        parameters,
        semantic_sample_keys=keys,
        profile_name=profile.name,
        strategy=INDEXED_MASKED_STRATEGY,
    )

    torch.testing.assert_close(indexed.corrupted, branchless.corrupted)
    assert [item.applied for item in indexed.metadata] == list(applied_mask)
    assert [item.as_json() for item in indexed.metadata] == [
        item.as_json() for item in branchless.metadata
    ]
    for sample_index, applied in enumerate(applied_mask):
        if applied:
            torch.testing.assert_close(
                indexed.combined[sample_index],
                branchless.combined[sample_index],
            )
        else:
            torch.testing.assert_close(
                indexed.corrupted[sample_index],
                inputs[sample_index],
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                indexed.combined[sample_index],
                inputs[sample_index],
                atol=0.0,
                rtol=0.0,
            )


def test_public_helper_strategies_preserve_semantic_outputs_and_rng() -> None:
    """Public strategy dispatch preserves semantic RNG and final outputs."""
    inputs = torch.linspace(-0.9, 0.9, steps=4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
    keys = (
        "train:wsi_a:1:10:20",
        "train:wsi_b:2:30:40",
        "train:wsi_c:3:50:60",
        "train:wsi_d:4:70:80",
    )
    profile = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)
    state = torch.get_rng_state()

    branchless = corrupt_normalized_batch(
        inputs,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=keys,
        corruption_step=3,
        corruption_view="train_corrupted",
        strategy=BRANCHLESS_ALL_STRATEGY,
    )
    after_branchless = torch.get_rng_state()
    indexed = corrupt_normalized_batch(
        inputs,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=keys,
        corruption_step=3,
        corruption_view="train_corrupted",
        strategy=INDEXED_MASKED_STRATEGY,
    )

    torch.testing.assert_close(indexed.corrupted, branchless.corrupted)
    assert [item.as_json() for item in indexed.metadata] == [
        item.as_json() for item in branchless.metadata
    ]
    assert torch.equal(state, after_branchless)
    assert torch.equal(state, torch.get_rng_state())


def test_corrupt_normalized_batch_accepts_indexed_masked_strategy() -> None:
    """Public corruption helper can run the indexed strategy."""
    inputs = torch.linspace(-0.8, 0.8, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    profile = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)

    result = corrupt_normalized_batch(
        inputs,
        profile=profile,
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=("train:wsi_a:1:10:20", "train:wsi_b:2:30:40"),
        corruption_step=0,
        corruption_view="train_corrupted",
        strategy=INDEXED_MASKED_STRATEGY,
    )

    assert result.corrupted.shape == inputs.shape
    assert len(result.metadata) == HELPER_STRATEGY_BATCH_SIZE


def test_corrupt_normalized_batch_rejects_unknown_strategy() -> None:
    """Unknown corruption strategy names fail before benchmark selection."""
    inputs = torch.linspace(-0.8, 0.8, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    profile = profile_from_name(CONSERVATIVE_DEFAULT_PROFILE)

    with pytest.raises(ValueError, match="Unknown corruption strategy"):
        corrupt_normalized_batch(
            inputs,
            profile=profile,
            corruption_seed=20260611,
            split="train",
            semantic_sample_keys=("train:wsi_a:1:10:20", "train:wsi_b:2:30:40"),
            corruption_step=0,
            corruption_view="train_corrupted",
            strategy="surprise_strategy",
        )


def test_module_apply_with_identity_parameters_round_trips_valid_rgb() -> None:
    """Identity parameters preserve valid HED-manifold RGB before noise."""
    module = StainCorruptor()
    hed = torch.tensor(
        [
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[0.05, 0.0], [0.1, 0.2]],
                [[0.02, 0.03], [0.0, 0.04]],
            ],
        ],
        dtype=torch.float32,
    )
    inputs = (hed_to_rgb(hed) * 2.0) - 1.0
    params = sample_corruption_parameters(
        batch_shape=(1, 3, 2, 2),
        profile=StainCorruptionProfile(
            name="identity_test",
            corrupt_prob=1.0,
            he_alpha_range=(1.0, 1.0),
            he_beta_range=(0.0, 0.0),
            residual_alpha_range=(1.0, 1.0),
            residual_beta_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
        ),
        corruption_seed=20260611,
        split="train",
        semantic_sample_keys=("train:wsi_a:1:10:20",),
        corruption_step=0,
        corruption_view="train_corrupted",
    )

    result = module.apply_with_parameters(
        inputs,
        params,
        semantic_sample_keys=("train:wsi_a:1:10:20",),
        profile_name="identity_test",
    )

    torch.testing.assert_close(result.corrupted, inputs, atol=1.0e-5, rtol=1.0e-5)


def test_local_stain_corruptor_qa_artifact(tmp_path: Path) -> None:
    """Local QA writer emits non-promotable JSON and PNG artifacts."""
    config_path = _write_tiny_qa_config(tmp_path)

    output_path = write_local_stain_corruptor_qa(
        LocalStainCorruptorQaRequest(
            config_path=config_path,
            output_dir=tmp_path,
            run_name="local_stain_qa",
        ),
    )

    payload = _load_json(output_path)
    visual_artifacts = payload["visual_artifacts"]
    checks = payload["checks"]
    assert isinstance(visual_artifacts, dict)
    assert isinstance(checks, dict)
    assert output_path == tmp_path / "benchmark" / "stain_corruptor_qa.json"
    assert payload["status"] == "local_pass"
    assert payload["benchmark_kind"] == "local_synthetic_stain_corruptor_qa"
    assert payload["full_run_eligible"] is False
    assert payload["corruption_version"] == CORRUPTION_VERSION
    assert checks["output_range_pass"] is True
    assert checks["target_preserved"] is True
    assert checks["clean_validation_rng_advanced"] is False
    assert checks["sample_count"] == EXPECTED_QA_COUNT
    assert visual_artifacts["fixed_real_25_status"] == "committed"
    visual_path = tmp_path / cast("str", visual_artifacts["synthetic_grid_path"])
    assert visual_path.exists()
    assert visual_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _as_float64_array(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _scikit_rgb2hed(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    func = cast(
        "Callable[[NDArray[np.float64]], object]",
        skimage_color.rgb2hed,
    )
    return _as_float64_array(func(rgb))


def _scikit_hed2rgb(hed: NDArray[np.float64]) -> NDArray[np.float64]:
    func = cast(
        "Callable[[NDArray[np.float64]], object]",
        skimage_color.hed2rgb,
    )
    return _as_float64_array(func(hed))


def _profile_config_payload(profile: StainCorruptionProfile) -> JsonObject:
    return {
        "profile_name": profile.name,
        "corrupt_prob": profile.corrupt_prob,
        "he_alpha_range": list(profile.he_alpha_range),
        "he_beta_range": list(profile.he_beta_range),
        "residual_alpha_range": list(profile.residual_alpha_range),
        "residual_beta_range": list(profile.residual_beta_range),
        "noise_std_range": list(profile.noise_std_range),
    }


def _write_tiny_qa_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "stain_qa_config.json"
    payload = {
        "schema_version": "spec0001.v0",
        "status": "stain_qa_test",
        "seeds": {
            "data_seed": 1234,
            "corruption_seed": 4321,
        },
        "data": {
            "kind": "synthetic",
            "image_size": 8,
            "channels": 3,
        },
        "corruption": {
            "kind": "tellez_hed_gaussian",
            "implementation_status": "corruption_ready",
            "corruption_version": CORRUPTION_VERSION,
            "profile_name": CONSERVATIVE_DEFAULT_PROFILE,
            "corrupt_prob": 0.3,
            "he_alpha_range": [0.8, 1.2],
            "he_beta_range": [-0.05, 0.05],
            "residual_alpha_range": [0.98, 1.02],
            "residual_beta_range": [-0.01, 0.01],
            "noise_std_range": [0.0, 0.05],
            "clean_validation_consumes_rng": False,
        },
    }
    config_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return config_path


def _parameters_with_mask(
    parameters: StainCorruptionParameters,
    *,
    applied_mask: tuple[bool, ...],
) -> StainCorruptionParameters:
    mask = torch.tensor(
        applied_mask,
        dtype=torch.bool,
        device=parameters.applied_mask.device,
    ).view(len(applied_mask), 1, 1, 1)
    return StainCorruptionParameters(
        applied_mask=mask,
        alpha=parameters.alpha,
        beta=parameters.beta,
        noise_std=parameters.noise_std,
        noise=parameters.noise,
        sample_seeds=parameters.sample_seeds,
    )


def _load_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
