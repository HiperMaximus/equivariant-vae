# Copyright 2026 HiperMaximus
"""Spec 0010 fixed-25 embedding-equivariance evaluation protocol.

This is an evaluation / inspection protocol, decoupled from training (see
``docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`` and
``docs/specs/0010-fixed25-embedding-equivariance-evaluation-protocol.md``). It
probes the embedding space of an autoencoder with exact ``torch.rot90`` at
``{0, 90, 180, 270}`` degrees, following EQ-VAE (arXiv:2502.09509): the
rotation-equivariance error of the latent is a proxy for how smooth / structured
the embedding is. The identical protocol runs on the non-equivariant baseline and
the future ``SO(2)``-steerable model over the same frozen 25 validation patches so
their embedding spaces can be compared visually and numerically.

Nothing here touches the loss, the optimizer, or the training objective.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from eqvae.corruption.stain import clean_validation_passthrough
from eqvae.data.dataloaders import normalize_uint8_batch
from eqvae.data.fixed_selectors import (
    FIXED_25_VALIDATION_COUNT,
    FIXED_25_VALIDATION_KIND,
    FIXED_25_VALIDATION_PER_LABEL,
    load_fixed_selector_document,
    validate_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.models.non_equivariant_vae import NonEquivariantVAE, clamp_logvar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from torch import Tensor

    from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue
    from eqvae.data.fixed_selectors import FixedSelectorDocument
    from eqvae.data.training_batches import PatchTrainingDataset

# Rotation convention (single source, shared by the image and latent paths).
ROTATION_METHOD = "rot90"
ROTATION_DIMS: tuple[int, int] = (2, 3)
ANGLE_K_VALUES: tuple[int, ...] = (0, 1, 2, 3)
MEASURED_K_VALUES: tuple[int, ...] = (1, 2, 3)
DEGREES_PER_K = 90
_FULL_TURN_K = 4

# Latent geometry of the spec 0001 baseline (``16 x 32 x 32`` for a 256 input).
RGB_CHANNELS = 3
PCA_COMPONENTS = 3
_MONTAGE_COLUMNS = 5
_MINMAX_EPS = 1e-8
DEFAULT_ERROR_EPS = 1e-8

# CSV contract for ``metrics/equivariance_25.csv``.
EQUIVARIANCE_25_COLUMNS: tuple[str, ...] = (
    "optimizer_step",
    "angle_degrees",
    "metric_name",
    "value",
    "mean",
    "std",
    "n",
    "data_source",
    "promotable",
)
EQUIVARIANCE_HEADLINE_METRIC = "equivariance_error_25_patches"
LATENT_MU_METRIC = "latent_mu_equivariance_error"
LATENT_LOGVAR_METRIC = "latent_logvar_equivariance_error"
RECONSTRUCTION_METRIC = "reconstruction_equivariance_error"
SAMPLED_LATENT_METRIC = "sampled_latent_equivariance_error"
ROT90_EXACTNESS_METRIC = "rot90_exactness_error"
REQUIRED_EQUIVARIANCE_METRICS: tuple[str, ...] = (
    EQUIVARIANCE_HEADLINE_METRIC,
    LATENT_MU_METRIC,
    LATENT_LOGVAR_METRIC,
    RECONSTRUCTION_METRIC,
    SAMPLED_LATENT_METRIC,
    ROT90_EXACTNESS_METRIC,
)

# Stable relative artifact names under ``artifacts/fixed25/``.
FIXED25_DIRNAME = "fixed25"
ORIGINALS_PT = "originals.pt"
ORIGINALS_PNG = "originals.png"
MANIFEST_JSON = "manifest.json"
RECONSTRUCTION_PROGRESS_PT = "reconstruction_progress.pt"
RECONSTRUCTION_PROGRESS_PNG = "reconstruction_progress.png"
LATENT_MU_PT = "latent_mu.pt"
GRID_PNG = "grids/rotated_input_vs_latent_grid.png"
PCA_PNG = "latent_pca_eqvae_style.png"
FIRST3_PNG = "latent_first3.png"


# Image-domain reconstruction / error-map tensors are archived at half precision
# to bound full-save-every-boundary disk growth (the scientifically load-bearing
# latent arrays stay float32); PNG montages remain the primary visual artifact.
IMAGE_TENSOR_DTYPE = torch.float16
LATENT_TENSOR_DTYPE = torch.float32


def _f16(tensor: Tensor) -> Tensor:
    return tensor.detach().to(IMAGE_TENSOR_DTYPE).cpu()


def _f32(tensor: Tensor) -> Tensor:
    return tensor.detach().to(LATENT_TENSOR_DTYPE).cpu()


def _boundary_dirname(optimizer_step: int) -> str:
    return f"boundary_{optimizer_step:06d}"


def rotated_pt_name(angle_degrees: int) -> str:
    """Return the per-angle rotated-artifact filename.

    Returns:
        Relative ``rotated_angle_{deg}.pt`` filename.

    """
    return f"rotated_angle_{angle_degrees}.pt"


def error_maps_pt_name(angle_degrees: int) -> str:
    """Return the per-angle error-map filename.

    Returns:
        Relative ``error_maps_angle_{deg}.pt`` filename.

    """
    return f"error_maps_angle_{angle_degrees}.pt"


@dataclass(frozen=True)
class Fixed25Config:
    """Resolved ``fixed25_equivariance`` config block, shared by both models."""

    enabled: bool
    selector_config: str
    expected_count: int
    expected_per_label: int
    rotation_method: str
    rotation_dims: tuple[int, int]
    rotation_k_values: tuple[int, ...]
    latent_transform: str
    latent_source: str
    latent_channels: int
    latent_spatial: int
    paired_epsilon: bool
    epsilon_seed: int
    error_eps: float
    save_every_boundary: bool
    pca_methods: tuple[str, ...]
    pca_components: int
    pca_fit_scope: str
    pca_sign_convention: str
    error_maps_masked: bool
    promotable_requires_real_data: bool


def parse_fixed25_config(
    effective_config: JsonObject,
    *,
    default_epsilon_seed: int,
) -> Fixed25Config | None:
    """Parse the shared ``fixed25_equivariance`` block from an effective config.

    Args:
        effective_config: The merged model-base + run config object.
        default_epsilon_seed: Seed used when the block omits an explicit
            ``sampled_latent.epsilon_seed`` (the run ``latent_seed``).

    Returns:
        The resolved config, or ``None`` when the block is absent. Raises
        ``ValueError`` (via ``_validate_convention``) if the block violates the
        locked evaluation convention (rotation method, ``k`` set, masked error
        maps, or latent source).

    """
    block = effective_config.get("fixed25_equivariance")
    if not isinstance(block, dict):
        return None
    rotation = _sub_object(block, "rotation")
    latent = _sub_object(block, "latent")
    sampled = _sub_object(block, "sampled_latent")
    equivariance = _sub_object(block, "equivariance")
    pca = _sub_object(block, "pca")
    error_maps = _sub_object(block, "error_maps")

    rotation_method = _opt_str(rotation, "method") or ROTATION_METHOD
    rotation_k_values = _int_tuple(rotation, "k_values") or ANGLE_K_VALUES
    rotation_dims = _int_tuple(rotation, "dims") or ROTATION_DIMS
    latent_source = _opt_str(latent, "source") or "posterior_mu_deterministic"
    error_maps_masked = _opt_bool(error_maps, "masked")

    _validate_convention(
        rotation_method=rotation_method,
        rotation_k_values=rotation_k_values,
        rotation_dims=rotation_dims,
        latent_source=latent_source,
        error_maps_masked=error_maps_masked,
    )

    return Fixed25Config(
        enabled=_opt_bool(block, "enabled"),
        selector_config=_opt_str(block, "selector_config") or "",
        expected_count=_opt_int(block, "expected_count") or FIXED_25_VALIDATION_COUNT,
        expected_per_label=(
            _opt_int(block, "expected_per_label") or FIXED_25_VALIDATION_PER_LABEL
        ),
        rotation_method=rotation_method,
        rotation_dims=(rotation_dims[0], rotation_dims[1]),
        rotation_k_values=rotation_k_values,
        latent_transform=_opt_str(latent, "transform") or "rot90_scalar_field",
        latent_source=latent_source,
        latent_channels=_opt_int(latent, "channels") or 16,
        latent_spatial=_opt_int(latent, "spatial") or 32,
        paired_epsilon=_opt_bool(sampled, "paired_epsilon"),
        epsilon_seed=_opt_int(sampled, "epsilon_seed") or default_epsilon_seed,
        error_eps=_opt_float(equivariance, "error_eps") or DEFAULT_ERROR_EPS,
        save_every_boundary=_opt_bool(block, "save_every_boundary"),
        pca_methods=_str_tuple(pca, "methods") or ("pca_top3", "first3"),
        pca_components=_opt_int(pca, "components") or PCA_COMPONENTS,
        pca_fit_scope=_opt_str(pca, "fit_scope") or "per_image",
        pca_sign_convention=_opt_str(pca, "sign_convention") or "unpinned",
        error_maps_masked=error_maps_masked,
        promotable_requires_real_data=_opt_bool(block, "promotable_requires_real_data"),
    )


def _validate_convention(
    *,
    rotation_method: str,
    rotation_k_values: tuple[int, ...],
    rotation_dims: tuple[int, ...],
    latent_source: str,
    error_maps_masked: bool,
) -> None:
    if rotation_method != ROTATION_METHOD:
        message = (
            f"fixed25 rotation.method must be {ROTATION_METHOD!r}, got "
            f"{rotation_method!r}"
        )
        raise ValueError(message)
    if tuple(rotation_k_values) != ANGLE_K_VALUES:
        message = (
            f"fixed25 rotation.k_values must be {ANGLE_K_VALUES}, got "
            f"{tuple(rotation_k_values)}"
        )
        raise ValueError(message)
    if tuple(rotation_dims) != ROTATION_DIMS:
        message = f"fixed25 rotation.dims must be {ROTATION_DIMS}, got {rotation_dims}"
        raise ValueError(message)
    if latent_source != "posterior_mu_deterministic":
        message = (
            "fixed25 latent.source must be 'posterior_mu_deterministic' (never "
            f"sampled z), got {latent_source!r}"
        )
        raise ValueError(message)
    if error_maps_masked:
        message = "fixed25 error_maps.masked must be false (full-frame only)"
        raise ValueError(message)


def rot90_k(tensor: Tensor, k: int) -> Tensor:
    """Rotate a batched spatial tensor by ``90*k`` degrees with exact ``rot90``.

    Applied identically to the image (``3 x H x W``) and the latent map
    (``C x H x W``); it is an exact spatial permutation with no interpolation.

    Returns:
        The rotated tensor.

    """
    return torch.rot90(tensor, k % _FULL_TURN_K, dims=ROTATION_DIMS)


def _rot90_inverse(tensor: Tensor, k: int) -> Tensor:
    return rot90_k(tensor, (_FULL_TURN_K - (k % _FULL_TURN_K)) % _FULL_TURN_K)


def _per_image_flatten(tensor: Tensor) -> Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _per_image_sq_l2(tensor: Tensor) -> Tensor:
    return _per_image_flatten(tensor).pow(2).sum(dim=1)


def _per_image_l2(tensor: Tensor) -> Tensor:
    return _per_image_flatten(tensor).pow(2).sum(dim=1).sqrt()


def _minmax01(image: Tensor) -> Tensor:
    """Per-image min-max normalize a ``[B, 3, H, W]`` tensor into ``[0, 1]``.

    Returns:
        The normalized tensor.

    """
    flat = image.reshape(image.shape[0], -1).to(torch.float32)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    scale = (maximum - minimum).clamp_min(_MINMAX_EPS)
    return ((flat - minimum) / scale).reshape(image.shape)


def pca_to_rgb(latent: Tensor) -> Tensor:  # noqa: PLR0914
    """Project a latent map onto its top-3 principal components as RGB.

    Transcribed from EQ-VAE ``evaluation/vis_latent.py`` ``pca_to_rgb``: the PCA
    is fit **per image** over that image's ``H*W`` spatial positions across the
    channel dimension, projected onto the top-3 eigenvectors (sign unpinned), then
    reshaped channel-last and per-image min-max normalized.

    Args:
        latent: A ``[B, C, H, W]`` latent tensor (``C >= 3``).

    Returns:
        A ``[B, 3, H, W]`` tensor in ``[0, 1]``.

    """
    latent = latent.detach()
    batch, channels, height, width = latent.shape
    out = torch.empty(
        (batch, PCA_COMPONENTS, height, width),
        dtype=torch.float32,
    )
    for index in range(batch):
        single = latent[index : index + 1].to(torch.float32)
        flat = single.permute(0, 2, 3, 1).reshape(-1, channels)
        centered = flat - flat.mean(dim=0, keepdim=True)
        divisor = max(1, centered.shape[0] - 1)
        covariance = (centered.t() @ centered) / divisor
        eigh = cast(
            "Callable[[Tensor], tuple[Tensor, Tensor]]",
            torch.linalg.eigh,
        )
        _, eigenvectors = eigh(covariance)
        top = eigenvectors[:, -PCA_COMPONENTS:]
        projected = centered @ top
        # Reshape channel-last (H, W, 3) THEN move channels first; a bare
        # reshape(B, 3, H, W) would scramble the spatial layout.
        rgb = projected.reshape(1, height, width, PCA_COMPONENTS).permute(0, 3, 1, 2)
        out[index] = _minmax01(rgb)[0]
    return out


def first3_to_rgb(latent: Tensor) -> Tensor:
    """Return the first-3-channels fallback RGB visualization.

    Args:
        latent: A ``[B, C, H, W]`` latent tensor (``C >= 3``).

    Returns:
        A ``[B, 3, H, W]`` tensor in ``[0, 1]``.

    """
    return _minmax01(latent.detach()[:, :PCA_COMPONENTS].to(torch.float32))


@dataclass(frozen=True)
class Fixed25Patches:
    """The frozen 25 validation patches plus selector provenance."""

    images: Tensor
    identities: tuple[JsonObject, ...]
    selector_sha256: str
    selector_status: str
    selector_schema: str
    selector_seed: str


def load_fixed25_patches(
    *,
    config: Fixed25Config,
    selector_path: Path,
    validation_shard_spec: PatchShardSpec,
    validation_dataset: PatchTrainingDataset,
) -> Fixed25Patches:
    """Load and validate the 25 fixed validation patches, failing closed.

    Reuses the spec 0001 selector loader/validator. Raises rather than resampling
    on any of: placeholder / not-ready status, wrong schema, wrong kind/split,
    count ``!= 25``, any label without exactly 5 rows, noncanonical rows, or a
    per-row identity mismatch against the current validation shard.

    Returns:
        The loaded patches (normalized ``[-1, 1]`` images) and provenance.

    Raises:
        ValueError: On any fail-closed condition above.

    """
    document = load_fixed_selector_document(selector_path)
    if document.selector_kind != FIXED_25_VALIDATION_KIND:
        message = (
            f"fixed25 selector must be {FIXED_25_VALIDATION_KIND!r}, got "
            f"{document.selector_kind!r}"
        )
        raise ValueError(message)
    if document.source_split != "validation":
        message = "fixed25 selector must target the validation split"
        raise ValueError(message)
    validate_fixed_selector_document(
        document=document,
        shard_spec=validation_shard_spec,
        expected_kind=FIXED_25_VALIDATION_KIND,
    )
    _validate_fixed25_counts(document=document, config=config)
    row_indices, identities = _fixed25_rows(
        document=document,
        validation_dataset=validation_dataset,
    )
    images_uint8 = torch.stack(
        [validation_dataset[row_index].image_uint8 for row_index in row_indices],
        dim=0,
    )
    images = clean_validation_passthrough(normalize_uint8_batch(images_uint8))
    return Fixed25Patches(
        images=images,
        identities=tuple(identities),
        selector_sha256=_sha256_file(selector_path),
        selector_status=document.status,
        selector_schema="spec0001.fixed_selector.v1",
        selector_seed=document.selector_seed,
    )


def _validate_fixed25_counts(
    *,
    document: FixedSelectorDocument,
    config: Fixed25Config,
) -> None:
    if document.expected_count != config.expected_count:
        message = (
            f"fixed25 selector expected_count {document.expected_count} != configured "
            f"{config.expected_count}"
        )
        raise ValueError(message)
    if document.expected_per_label != config.expected_per_label:
        message = (
            f"fixed25 selector expected_per_label {document.expected_per_label} != "
            f"configured {config.expected_per_label}"
        )
        raise ValueError(message)
    if len(document.selectors) != config.expected_count:
        message = (
            f"fixed25 selector realized count {len(document.selectors)} != "
            f"{config.expected_count}"
        )
        raise ValueError(message)
    per_label: dict[int, int] = {}
    for selector in document.selectors:
        per_label[selector.label] = per_label.get(selector.label, 0) + 1
    bad = {
        label: count
        for label, count in per_label.items()
        if count != config.expected_per_label
    }
    if bad:
        message = (
            f"fixed25 selector labels without {config.expected_per_label} "
            f"rows each: {bad}"
        )
        raise ValueError(message)


def _fixed25_rows(
    *,
    document: FixedSelectorDocument,
    validation_dataset: PatchTrainingDataset,
) -> tuple[tuple[int, ...], list[JsonObject]]:
    records = validation_dataset.records
    row_indices: list[int] = []
    identities: list[JsonObject] = []
    for selector in document.selectors:
        if selector.row_index < 0 or selector.row_index >= len(records):
            message = (
                "fixed25 selector row_index outside validation dataset: "
                f"{selector.row_index}"
            )
            raise ValueError(message)
        record = records[selector.row_index]
        expected = (
            record.file_index,
            record.row_index,
            record.sample_id("validation"),
            record.wsi_id,
            record.label,
            record.x,
            record.y,
        )
        observed = (
            selector.file_index,
            selector.row_index,
            selector.sample_id,
            selector.wsi_id,
            selector.label,
            selector.x,
            selector.y,
        )
        if observed != expected:
            message = f"fixed25 selector row mismatch at rank {selector.rank}"
            raise ValueError(message)
        row_indices.append(selector.row_index)
        identities.append(
            cast(
                "JsonObject",
                {
                    "rank": selector.rank,
                    "sample_id": selector.sample_id,
                    "wsi_id": selector.wsi_id,
                    "label": selector.label,
                    "x": selector.x,
                    "y": selector.y,
                    "file_index": selector.file_index,
                    "row_index": selector.row_index,
                },
            ),
        )
    if len(set(row_indices)) != len(row_indices):
        message = "fixed25 selector contains duplicate row indices"
        raise ValueError(message)
    return tuple(row_indices), identities


@dataclass(frozen=True)
class _AnglePass:
    angle_degrees: int
    ground_truth: Tensor
    rotated_input_recon: Tensor
    rotated_embedding_recon: Tensor
    input_recon_error_map: Tensor
    embedding_recon_error_map: Tensor
    rotated_mu_of_input: Tensor
    rotated_latent_of_mu: Tensor
    rows: tuple[CsvRow, ...]


def evaluate_boundary(  # noqa: PLR0913
    *,
    model: NonEquivariantVAE,
    patches: Fixed25Patches,
    config: Fixed25Config,
    fixed25_dir: Path,
    optimizer_step: int,
    device: torch.device,
    data_source: str,
    promotable: bool,
) -> tuple[CsvRow, ...]:
    """Run the fixed-25 protocol for one half-epoch boundary and write artifacts.

    Under ``torch.no_grad`` on the primary rank only: writes clean reconstruction
    progress, per-angle rotated-input / rotated-embedding reconstructions, full
    frame error maps, latent arrays, the composite grid, and the PCA / first-3
    visualizations, then returns the equivariance CSV rows for this boundary.

    Returns:
        The per-angle equivariance rows for ``metrics/equivariance_25.csv``.

    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return _evaluate_boundary_no_grad(
                model=model,
                patches=patches,
                config=config,
                fixed25_dir=fixed25_dir,
                optimizer_step=optimizer_step,
                device=device,
                data_source=data_source,
                promotable=promotable,
            )
    finally:
        if was_training:
            model.train()


def _evaluate_boundary_no_grad(  # noqa: PLR0913
    *,
    model: NonEquivariantVAE,
    patches: Fixed25Patches,
    config: Fixed25Config,
    fixed25_dir: Path,
    optimizer_step: int,
    device: torch.device,
    data_source: str,
    promotable: bool,
) -> tuple[CsvRow, ...]:
    images = patches.images.to(device=device, dtype=torch.float32)
    mu0, logvar0 = model.encode(images)
    logvar0_c = clamp_logvar(logvar0)
    recon0 = model.decode(mu0)
    eps = _sampled_epsilon(config=config, reference=mu0, device=device)
    z0 = mu0 + torch.exp(0.5 * logvar0_c) * eps

    boundary_dir = fixed25_dir / _boundary_dirname(optimizer_step)
    rows: list[CsvRow] = []
    angle_passes: list[_AnglePass] = []
    latent_payload: dict[str, Tensor] = {"mu_clean": _f32(mu0)}
    for k in MEASURED_K_VALUES:
        angle = _angle_pass(
            model=model,
            images=images,
            mu0=mu0,
            logvar0_c=logvar0_c,
            z0=z0,
            eps=eps,
            k=k,
            error_eps=config.error_eps,
            optimizer_step=optimizer_step,
            data_source=data_source,
            promotable=promotable,
        )
        angle_passes.append(angle)
        rows.extend(angle.rows)
        latent_payload[f"rotated_latent_of_mu_{angle.angle_degrees}"] = _f32(
            angle.rotated_latent_of_mu,
        )
        latent_payload[f"mu_of_rotated_input_{angle.angle_degrees}"] = _f32(
            angle.rotated_mu_of_input,
        )
        _atomic_torch_save(
            boundary_dir / rotated_pt_name(angle.angle_degrees),
            {
                "angle_degrees": angle.angle_degrees,
                "ground_truth": _f16(angle.ground_truth),
                "rotated_input_reconstruction": _f16(angle.rotated_input_recon),
                "rotated_embedding_reconstruction": _f16(angle.rotated_embedding_recon),
            },
        )
        _atomic_torch_save(
            boundary_dir / error_maps_pt_name(angle.angle_degrees),
            {
                "angle_degrees": angle.angle_degrees,
                "input_recon_vs_ground_truth": _f16(angle.input_recon_error_map),
                "embedding_recon_vs_input_recon": _f16(angle.embedding_recon_error_map),
            },
        )

    _atomic_torch_save(
        boundary_dir / RECONSTRUCTION_PROGRESS_PT,
        {
            "optimizer_step": optimizer_step,
            "reconstruction": _f16(recon0),
        },
    )
    _atomic_torch_save(boundary_dir / LATENT_MU_PT, latent_payload)
    _write_boundary_images(
        boundary_dir=boundary_dir,
        images=images,
        recon0=recon0,
        mu0=mu0,
        angle_passes=angle_passes,
    )
    return tuple(rows)


def _angle_pass(  # noqa: PLR0913, PLR0914
    *,
    model: NonEquivariantVAE,
    images: Tensor,
    mu0: Tensor,
    logvar0_c: Tensor,
    z0: Tensor,
    eps: Tensor,
    k: int,
    error_eps: float,
    optimizer_step: int,
    data_source: str,
    promotable: bool,
) -> _AnglePass:
    angle_degrees = DEGREES_PER_K * k
    rotated_input = rot90_k(images, k)
    mu_of_rotated, logvar_of_rotated = model.encode(rotated_input)
    logvar_of_rotated_c = clamp_logvar(logvar_of_rotated)
    rotated_mu0 = rot90_k(mu0, k)
    rotated_logvar0_c = rot90_k(logvar0_c, k)

    rotated_input_recon = model.decode(mu_of_rotated)
    rotated_embedding_recon = model.decode(rotated_mu0)

    # Sampled path: R_k z(x, eps) vs z(R_k x, R_k eps) with the SAME eps.
    rotated_z0 = rot90_k(z0, k)
    z_of_rotated = mu_of_rotated + torch.exp(0.5 * logvar_of_rotated_c) * rot90_k(
        eps,
        k,
    )
    sampled_rotated_latent_recon = model.decode(rotated_z0)
    sampled_rotated_input_recon = model.decode(z_of_rotated)

    headline = _per_image_sq_l2(rotated_mu0 - mu_of_rotated) / (
        _per_image_sq_l2(mu_of_rotated) + error_eps
    )
    latent_mu = _per_image_l2(rotated_mu0 - mu_of_rotated)
    latent_logvar = _per_image_l2(rotated_logvar0_c - logvar_of_rotated_c)
    reconstruction = _per_image_l2(rotated_embedding_recon - rotated_input_recon)
    sampled = _per_image_l2(sampled_rotated_latent_recon - sampled_rotated_input_recon)
    exactness = torch.maximum(
        _per_image_l2(_rot90_inverse(rot90_k(images, k), k) - images),
        _per_image_l2(_rot90_inverse(rotated_mu0, k) - mu0),
    )

    rows = tuple(
        _metric_row(
            optimizer_step=optimizer_step,
            angle_degrees=angle_degrees,
            metric_name=name,
            values=values,
            data_source=data_source,
            promotable=promotable,
        )
        for name, values in (
            (EQUIVARIANCE_HEADLINE_METRIC, headline),
            (LATENT_MU_METRIC, latent_mu),
            (LATENT_LOGVAR_METRIC, latent_logvar),
            (RECONSTRUCTION_METRIC, reconstruction),
            (SAMPLED_LATENT_METRIC, sampled),
            (ROT90_EXACTNESS_METRIC, exactness),
        )
    )
    return _AnglePass(
        angle_degrees=angle_degrees,
        ground_truth=rotated_input,
        rotated_input_recon=rotated_input_recon,
        rotated_embedding_recon=rotated_embedding_recon,
        input_recon_error_map=rotated_input_recon - rotated_input,
        embedding_recon_error_map=rotated_embedding_recon - rotated_input_recon,
        rotated_mu_of_input=mu_of_rotated,
        rotated_latent_of_mu=rotated_mu0,
        rows=rows,
    )


def headline_equivariance_at_k0(
    *,
    model: NonEquivariantVAE,
    images: Tensor,
    error_eps: float = DEFAULT_ERROR_EPS,
) -> float:
    """Return the headline equivariance error at ``k = 0`` (must be ``0``).

    Used by tests to prove the convention: at the identity rotation every
    equivariance error is exactly zero.

    Returns:
        The mean over patches of ``r_0(x)`` (identically zero up to float error).

    """
    with torch.no_grad():
        mu0, _ = model.encode(images)
        rotated_mu0 = rot90_k(mu0, 0)
        headline = _per_image_sq_l2(rotated_mu0 - mu0) / (
            _per_image_sq_l2(mu0) + error_eps
        )
        return float(headline.mean().item())


def _sampled_epsilon(
    *,
    config: Fixed25Config,
    reference: Tensor,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.epsilon_seed)
    return torch.randn(
        reference.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def _metric_row(  # noqa: PLR0913
    *,
    optimizer_step: int,
    angle_degrees: int,
    metric_name: str,
    values: Tensor,
    data_source: str,
    promotable: bool,
) -> CsvRow:
    mean = values.mean()
    std = ((values - mean).pow(2).mean()).sqrt()
    return {
        "optimizer_step": str(optimizer_step),
        "angle_degrees": str(angle_degrees),
        "metric_name": metric_name,
        "value": _fmt(float(mean.item())),
        "mean": _fmt(float(mean.item())),
        "std": _fmt(float(std.item())),
        "n": str(int(values.shape[0])),
        "data_source": data_source,
        "promotable": "true" if promotable else "false",
    }


def _fmt(value: float) -> str:
    return format(value, ".10g")


def write_originals(*, fixed25_dir: Path, patches: Fixed25Patches) -> None:
    """Write the immutable 25 clean originals (``.pt`` plus a PNG montage).

    Written once per run (the FSQ immutable structural baseline).
    """
    _atomic_torch_save(
        fixed25_dir / ORIGINALS_PT,
        {
            "images": patches.images.detach().cpu(),
            "identities": list(patches.identities),
            "selector_sha256": patches.selector_sha256,
        },
    )
    _atomic_write_png(
        fixed25_dir / ORIGINALS_PNG,
        _montage([patches.images[i] for i in range(patches.images.shape[0])]),
    )


def write_manifest(  # noqa: PLR0913
    *,
    fixed25_dir: Path,
    config: Fixed25Config,
    patches: Fixed25Patches,
    data_source: str,
    promotable: bool,
    rot90_exactness_error: float,
) -> None:
    """Write ``artifacts/fixed25/manifest.json`` with the shared metadata.

    Records the resolved rotation convention (once, shared by the image and
    latent paths), selector provenance/identities, the promotability label, and
    the boundary steps covered so far (scanned from disk for resume safety).
    """
    boundary_steps = _covered_boundary_steps(fixed25_dir)
    payload = cast(
        "JsonObject",
        {
            "schema": "spec0010.fixed25_equivariance.manifest.v1",
            "data_source": data_source,
            "promotable": promotable,
            "rotation": {
                "method": config.rotation_method,
                "dims": list(config.rotation_dims),
                "k_values": list(config.rotation_k_values),
                "angles_degrees": [DEGREES_PER_K * k for k in config.rotation_k_values],
            },
            "measured_angles_degrees": [DEGREES_PER_K * k for k in MEASURED_K_VALUES],
            "latent": {
                "transform": config.latent_transform,
                "source": config.latent_source,
                "channels": config.latent_channels,
                "spatial": config.latent_spatial,
            },
            "sampled_latent": {
                "paired_epsilon": config.paired_epsilon,
                "epsilon_seed": config.epsilon_seed,
            },
            "equivariance": {"error_eps": config.error_eps},
            "pca": {
                "methods": list(config.pca_methods),
                "components": config.pca_components,
                "fit_scope": config.pca_fit_scope,
                "sign_convention": config.pca_sign_convention,
            },
            "error_maps": {"masked": config.error_maps_masked},
            "tensor_dtypes": {
                "image_domain": str(IMAGE_TENSOR_DTYPE).removeprefix("torch."),
                "latent": str(LATENT_TENSOR_DTYPE).removeprefix("torch."),
            },
            "rot90_exactness_error": rot90_exactness_error,
            "selector": {
                "config": config.selector_config,
                "sha256": patches.selector_sha256,
                "schema": patches.selector_schema,
                "status": patches.selector_status,
                "seed": patches.selector_seed,
                "expected_count": config.expected_count,
                "expected_per_label": config.expected_per_label,
                "identities": list(patches.identities),
            },
            "boundary_optimizer_steps": list(boundary_steps),
        },
    )
    _atomic_write_json(fixed25_dir / MANIFEST_JSON, payload)


def compute_rot90_exactness(
    *,
    model: NonEquivariantVAE,
    patches: Fixed25Patches,
    device: torch.device,
) -> float:
    """Return the maximum ``rot90`` round-trip error over the 25 patches.

    Sanity check for the manifest: exact ``rot90`` round-trips to ``0`` in both
    the image and latent domains.

    Returns:
        The max over patches and angles of ``||R_{-k}(R_k t) - t||``.

    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            images = patches.images.to(device=device, dtype=torch.float32)
            mu0, _ = model.encode(images)
            worst = 0.0
            for k in MEASURED_K_VALUES:
                image_error = _per_image_l2(
                    _rot90_inverse(rot90_k(images, k), k) - images,
                ).max()
                latent_error = _per_image_l2(
                    _rot90_inverse(rot90_k(mu0, k), k) - mu0,
                ).max()
                worst = max(
                    worst,
                    float(image_error.item()),
                    float(latent_error.item()),
                )
            return worst
    finally:
        if was_training:
            model.train()


def _covered_boundary_steps(fixed25_dir: Path) -> tuple[int, ...]:
    if not fixed25_dir.exists():
        return ()
    steps: list[int] = []
    for path in fixed25_dir.glob("boundary_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("boundary_")
        if suffix.isdigit():
            steps.append(int(suffix))
    return tuple(sorted(steps))


def _write_boundary_images(
    *,
    boundary_dir: Path,
    images: Tensor,
    recon0: Tensor,
    mu0: Tensor,
    angle_passes: Sequence[_AnglePass],
) -> None:
    _atomic_write_png(
        boundary_dir / RECONSTRUCTION_PROGRESS_PNG,
        _montage([recon0[i] for i in range(recon0.shape[0])]),
    )
    _atomic_write_png(
        boundary_dir / GRID_PNG,
        _rotated_grid(images=images, recon0=recon0, angle_passes=angle_passes),
    )
    _atomic_write_png(
        boundary_dir / PCA_PNG,
        _montage(
            list(pca_to_rgb(mu0.detach().cpu())),
            unit_domain=True,
        ),
    )
    _atomic_write_png(
        boundary_dir / FIRST3_PNG,
        _montage(
            list(first3_to_rgb(mu0.detach().cpu())),
            unit_domain=True,
        ),
    )


def _rotated_grid(
    *,
    images: Tensor,
    recon0: Tensor,
    angle_passes: Sequence[_AnglePass],
) -> NDArray[np.uint8]:
    # Issue-#4 grid for the first fixed patch: rows {0, 90, 180, 270} x columns
    # {ground truth, rotated-input recon, rotated-embedding recon}. The k=0 row
    # leaves the rotated-embedding (latent) column blank (identity).
    blank = torch.zeros_like(images[0])
    row_cells: list[list[Tensor]] = [[images[0], recon0[0], blank]]
    domains: list[list[bool]] = [[False, False, False]]
    for angle in angle_passes:
        row_cells.append(
            [
                angle.ground_truth[0],
                angle.rotated_input_recon[0],
                angle.rotated_embedding_recon[0],
            ],
        )
        domains.append([False, False, False])
    return _grid(row_cells, domains)


def _grid(
    rows: Sequence[Sequence[Tensor]],
    domains: Sequence[Sequence[bool]],
) -> NDArray[np.uint8]:
    row_arrays = [
        np.concatenate(
            [
                _to_uint8_hwc(cell, unit_domain=domain)
                for cell, domain in zip(row, row_domains, strict=True)
            ],
            axis=1,
        )
        for row, row_domains in zip(rows, domains, strict=True)
    ]
    return np.concatenate(row_arrays, axis=0)


def _montage(
    cells: Sequence[Tensor],
    *,
    unit_domain: bool = False,
) -> NDArray[np.uint8]:
    tiles = [_to_uint8_hwc(cell, unit_domain=unit_domain) for cell in cells]
    if not tiles:
        return np.zeros((1, 1, RGB_CHANNELS), dtype=np.uint8)
    columns = _MONTAGE_COLUMNS
    rows = (len(tiles) + columns - 1) // columns
    shape = cast("tuple[int, ...]", tiles[0].shape)
    height = shape[0]
    width = shape[1]
    canvas = np.zeros((rows * height, columns * width, RGB_CHANNELS), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = tile
    return canvas


def _to_uint8_hwc(image_chw: Tensor, *, unit_domain: bool) -> NDArray[np.uint8]:
    tensor = image_chw.detach().to(torch.float32).cpu()
    if unit_domain:
        scaled = tensor.clamp(0.0, 1.0)
    else:
        scaled = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    byte = (scaled * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    return cast(
        "NDArray[np.uint8]",
        byte.permute(1, 2, 0).contiguous().numpy(),
    )


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def write_equivariance_csv(*, path: Path, rows: Sequence[CsvRow]) -> None:
    """Write ``metrics/equivariance_25.csv`` with the Spec 0010 column contract.

    Used by the standalone evaluator; the in-run runner writes the same columns
    through its own durable interval flush.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=EQUIVARIANCE_25_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _atomic_write_png(path: Path, array_hwc: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(_encode_png_rgb(array_hwc))
    tmp_path.replace(path)


def _encode_png_rgb(array_hwc: NDArray[np.uint8]) -> bytes:
    array = np.ascontiguousarray(array_hwc, dtype=np.uint8)
    shape = cast("tuple[int, ...]", array.shape)
    height = shape[0]
    width = shape[1]
    filtered = np.zeros((height, 1 + width * RGB_CHANNELS), dtype=np.uint8)
    filtered[:, 1:] = array.reshape(height, width * RGB_CHANNELS)
    raw = filtered.tobytes()
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw, 6)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_shard_spec_for(
    *,
    validation_bin_path: Path,
    validation_csv_path: Path,
    image_size: int,
    validate_crc: bool,
) -> PatchShardSpec:
    """Return the validation shard spec used to validate the fixed-25 selector.

    Returns:
        A ``PatchShardSpec`` for the validation shard.

    """
    return PatchShardSpec(
        bin_path=validation_bin_path,
        csv_path=validation_csv_path,
        image_size=image_size,
        channels=RGB_CHANNELS,
        validate_crc=validate_crc,
    )


def _sub_object(obj: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def _opt_str(obj: Mapping[str, JsonValue], key: str) -> str:
    value = obj.get(key)
    return value if isinstance(value, str) else ""


def _opt_int(obj: Mapping[str, JsonValue], key: str) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _opt_float(obj: Mapping[str, JsonValue], key: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _opt_bool(obj: Mapping[str, JsonValue], key: str) -> bool:
    value = obj.get(key)
    return value if isinstance(value, bool) else False


def _int_tuple(obj: Mapping[str, JsonValue], key: str) -> tuple[int, ...]:
    value = obj.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(
        item for item in value if isinstance(item, int) and not isinstance(item, bool)
    )


def _str_tuple(obj: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = obj.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


__all__ = [
    "EQUIVARIANCE_25_COLUMNS",
    "EQUIVARIANCE_HEADLINE_METRIC",
    "FIXED25_DIRNAME",
    "MANIFEST_JSON",
    "MEASURED_K_VALUES",
    "ORIGINALS_PT",
    "REQUIRED_EQUIVARIANCE_METRICS",
    "Fixed25Config",
    "Fixed25Patches",
    "compute_rot90_exactness",
    "evaluate_boundary",
    "first3_to_rgb",
    "headline_equivariance_at_k0",
    "load_fixed25_patches",
    "parse_fixed25_config",
    "pca_to_rgb",
    "rot90_k",
    "rotated_pt_name",
    "validation_shard_spec_for",
    "write_equivariance_csv",
    "write_manifest",
    "write_originals",
]
