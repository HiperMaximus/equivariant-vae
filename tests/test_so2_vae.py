# Copyright 2026 HiperMaximus
"""Focused acceptance tests for the fixed Spec 0014 full SO(2) VAE."""

from __future__ import annotations

import math
from collections import Counter
from inspect import getsource
from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch._dynamo as torch_dynamo  # noqa: PLC2701
from torch import nn
from torch.nn import functional

from eqvae.models.so2_architecture_probe import (
    _PROFILE_9,  # noqa: PLC2701
    L_LAYOUT,
    R_LAYOUT,
    FixedF01FieldNorm,
    FixedF01RadialGate,
    _F01ToF01Conv,  # noqa: PLC2701
    _F01ToScalarConv,  # noqa: PLC2701
    _FixedF01Downsample2x,  # noqa: PLC2701
    _FixedF01Upsample2x,  # noqa: PLC2701
    _ScalarToF01Conv,  # noqa: PLC2701
)
from eqvae.models.so2_vae import SO2VAE, build_so2_vae
from eqvae.training.optim import (
    SpecAdamWConfig,
    build_adamw_parameter_groups,
    create_adamw_optimizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from eqvae.models.non_equivariant_vae import VaeForwardOutput

# pyright: reportPrivateUsage=false

_EXPECTED_SIGNATURES = {
    "R->A": 1,
    "A->A": 7,
    "A->B": 2,
    "B->B": 6,
    "B->C": 2,
    "C->C": 6,
    "C->D": 2,
    "D->D": 7,
    "D->L": 2,
    "L->D": 1,
    "D->C": 2,
    "C->B": 2,
    "B->A": 2,
    "A->R": 1,
}
_EXPECTED_COEFFICIENTS = 1_172_304
_EXPECTED_NORM_PARAMETERS = 3_600
_EXPECTED_GATE_PARAMETERS = 4_096
_EXPECTED_BIASES = 35
_EXPECTED_PARAMETERS = 1_180_035
_LEARNED_CONVOLUTION_COUNT = 43
_HIDDEN_CONVOLUTION_COUNT = 38
_SCALAR_MM_COUNT = 10
_BRANCH_RESAMPLERS_PER_DIRECTION = 6
_GATE_COUNT = 34
_HIDDEN_KERNEL_SIZE = 7


def _learned_convolutions(model: SO2VAE) -> tuple[nn.Module, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, _ScalarToF01Conv | _F01ToF01Conv | _F01ToScalarConv)
    )


def _signature(module: nn.Module) -> str:
    if isinstance(module, _ScalarToF01Conv):
        input_name = "R" if module.input_copies == R_LAYOUT.n0 else "L"
        return f"{input_name}->{module.output_layout.name}"
    if isinstance(module, _F01ToF01Conv):
        return f"{module.input_layout.name}->{module.output_layout.name}"
    if isinstance(module, _F01ToScalarConv):
        output_name = "R" if module.output_copies == R_LAYOUT.n0 else "L"
        return f"{module.input_layout.name}->{output_name}"
    raise AssertionError(type(module).__name__)


def _parameter_ids(module_type: type[nn.Module], model: SO2VAE) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, module_type)
        for parameter in module.parameters(recurse=False)
    }


def _rotate_scalar(values: torch.Tensor, degrees: int) -> torch.Tensor:
    if degrees % 90 == 0:
        return torch.rot90(values, degrees // 90, dims=(-2, -1))
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transform = values.new_tensor(
        ((cosine, sine, 0.0), (-sine, cosine, 0.0)),
    ).unsqueeze(0)
    transform = transform.expand(values.shape[0], -1, -1)
    grid = functional.affine_grid(transform, list(values.shape), align_corners=False)
    return functional.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _relative_rms(
    observed: torch.Tensor,
    expected: torch.Tensor,
    *,
    crop: int,
) -> float:
    if crop:
        observed = observed[..., crop:-crop, crop:-crop]
        expected = expected[..., crop:-crop, crop:-crop]
    numerator = torch.sqrt((observed - expected).double().square().mean())
    denominator = torch.sqrt(expected.double().square().mean()).clamp_min(1e-12)
    return float((numerator / denominator).detach())


def _smooth_inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(14014)
    values = torch.randn(1, 3, 64, 64, generator=generator)
    for _ in range(4):
        values = functional.avg_pool2d(values, kernel_size=5, stride=1, padding=2)
    return values / values.square().mean().sqrt()


def _gradient_is_nonzero(parameter: nn.Parameter) -> bool:
    gradient = parameter.grad
    return gradient is not None and bool(gradient.abs().max() > 0)


def _assert_tokens_in_order(source: str, tokens: tuple[str, ...]) -> None:
    offsets = tuple(source.index(token) for token in tokens)
    assert offsets == tuple(sorted(offsets))


def test_full_topology_and_parameter_partition_are_exact() -> None:
    """Pin every learned map and parameter role so assembly cannot alter capacity."""
    model = build_so2_vae()
    learned = _learned_convolutions(model)
    signatures = Counter(_signature(module) for module in learned)
    named_parameters = dict(model.named_parameters())
    coefficient_ids = {
        id(parameter)
        for name, parameter in named_parameters.items()
        if name.rsplit(".", 1)[-1].startswith("coeff")
    }
    norm_ids = _parameter_ids(FixedF01FieldNorm, model)
    gate_ids = _parameter_ids(FixedF01RadialGate, model)
    bias_ids = {
        id(parameter)
        for name, parameter in named_parameters.items()
        if name.rsplit(".", 1)[-1] == "bias"
    }

    assert signatures == _EXPECTED_SIGNATURES
    assert len(learned) == _LEARNED_CONVOLUTION_COUNT
    assert (
        sum(
            named_parameters[name].numel()
            for name in named_parameters
            if id(named_parameters[name]) in coefficient_ids
        )
        == _EXPECTED_COEFFICIENTS
    )
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in norm_ids
        )
        == _EXPECTED_NORM_PARAMETERS
    )
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in gate_ids
        )
        == _EXPECTED_GATE_PARAMETERS
    )
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in bias_ids
        )
        == _EXPECTED_BIASES
    )
    assert coefficient_ids | norm_ids | gate_ids | bias_ids == {
        id(parameter) for parameter in model.parameters()
    }
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == _EXPECTED_PARAMETERS
    )
    assert model.stem_conv.kernel_size == _PROFILE_9.kernel_size
    assert all(
        module.kernel_size == _HIDDEN_KERNEL_SIZE
        for module in learned
        if module is not model.stem_conv
    )


def test_branch_local_resampling_count_and_order_are_locked() -> None:
    """Keep all 12 branch-local resamplers before their transition projections."""
    model = build_so2_vae()
    downsamplers = tuple(
        module
        for module in model.modules()
        if isinstance(module, _FixedF01Downsample2x)
    )
    upsamplers = tuple(
        module for module in model.modules() if isinstance(module, _FixedF01Upsample2x)
    )
    assert len(downsamplers) == len(upsamplers) == _BRANCH_RESAMPLERS_PER_DIRECTION

    _assert_tokens_in_order(
        getsource(type(model.encoder_blocks[2]).forward),
        (
            "self.main_conv1(inputs)",
            "self.main_norm1(main)",
            "self.main_gate(main)",
            "self.main_downsample(main)",
            "self.main_conv2(main)",
            "self.main_norm2(main)",
            "self.skip_downsample(skip)",
            "self.skip_conv(skip)",
            "self.skip_norm(skip)",
            "self.output_gate(main + skip)",
        ),
    )
    _assert_tokens_in_order(
        getsource(type(model.decoder_blocks[2]).forward),
        (
            "self.main_upsample(main)",
            "self.main_conv1(main)",
            "self.main_norm1(main)",
            "self.main_gate(main)",
            "self.main_conv2(main)",
            "self.main_norm2(main)",
            "self.skip_upsample(skip)",
            "self.skip_conv(skip)",
            "self.skip_norm(skip)",
            "self.output_gate(main + skip)",
        ),
    )


def test_deployment_shapes_and_external_vae_contract() -> None:
    """Prove the locked 256-to-32 schedule and baseline-compatible VAE semantics."""
    deployment = build_so2_vae().to(device="meta")
    inputs = torch.empty(2, R_LAYOUT.channels, 256, 256, device="meta")
    eps = torch.empty(2, L_LAYOUT.channels, 32, 32, device="meta")
    output = cast("VaeForwardOutput", deployment(inputs, eps=eps))
    assert output.reconstruction.shape == (2, R_LAYOUT.channels, 256, 256)
    assert output.mu.shape == output.logvar.shape == eps.shape
    assert output.logvar_clamped.shape == output.z.shape == eps.shape
    assert output.eps is eps
    assert deployment.latent_channels == L_LAYOUT.channels

    model = build_so2_vae()
    with torch.no_grad():
        model.logvar_head.coeff00.zero_()
        model.logvar_head.coeff01.zero_()
        model.logvar_head.bias.fill_(10.0)
    eager_eps = torch.randn(1, L_LAYOUT.channels, 1, 1)
    eager = cast(
        "VaeForwardOutput",
        model(torch.randn(1, R_LAYOUT.channels, 8, 8), eps=eager_eps),
    )
    assert torch.equal(eager.logvar_clamped, torch.full_like(eager.logvar, 4.0))
    assert int(eager.logvar_clamp_count) == eager.logvar.numel()
    assert torch.equal(eager.eps, eager_eps)
    with pytest.raises(ValueError, match="eps shape"):
        model.reparameterize(
            mu=eager.mu,
            logvar=eager.logvar_clamped,
            eps=torch.randn(1, L_LAYOUT.channels, 2, 2),
        )
    assert not any(isinstance(module, nn.Tanh) for module in model.modules())
    assert not bool(model.output_head.coeff00.count_nonzero())
    assert not bool(model.output_head.coeff01.count_nonzero())
    assert not bool(model.output_head.bias.count_nonzero())


def test_full_forward_preserves_contraction_and_one_conv2d_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch any copy loop, second learned convolution, or alternate expansion path."""
    model = build_so2_vae()
    learned_conv2d_calls = 0
    fixed_resampling_conv2d_calls = 0
    bmm_calls = 0
    mm_calls = 0
    original_conv2d = functional.conv2d
    original_bmm = torch.bmm
    original_mm = torch.mm

    def counted_conv2d(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal learned_conv2d_calls, fixed_resampling_conv2d_calls
        groups = cast("int", kwargs.get("groups", 1))
        if groups == 1:
            learned_conv2d_calls += 1
        else:
            fixed_resampling_conv2d_calls += 1
        return original_conv2d(*args, **kwargs)  # type: ignore[call-overload]

    def counted_bmm(inputs: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        nonlocal bmm_calls
        bmm_calls += 1
        return original_bmm(inputs, matrices)

    def counted_mm(inputs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        nonlocal mm_calls
        mm_calls += 1
        return original_mm(inputs, matrix)

    monkeypatch.setattr(functional, "conv2d", counted_conv2d)
    monkeypatch.setattr(torch, "bmm", counted_bmm)
    monkeypatch.setattr(torch, "mm", counted_mm)
    output = cast(
        "VaeForwardOutput",
        model(
            torch.randn(1, R_LAYOUT.channels, 8, 8),
            eps=torch.randn(1, L_LAYOUT.channels, 1, 1),
        ),
    )

    assert output.reconstruction.shape == (1, R_LAYOUT.channels, 8, 8)
    assert learned_conv2d_calls == _LEARNED_CONVOLUTION_COUNT
    assert fixed_resampling_conv2d_calls == _BRANCH_RESAMPLERS_PER_DIRECTION
    assert bmm_calls == _HIDDEN_CONVOLUTION_COUNT
    assert mm_calls == _SCALAR_MM_COUNT


def test_optimizer_groups_and_gate_families_cover_the_so2_parameters() -> None:
    """Apply decay and half-rate semantics and inspect every fixed SO(2) gate."""
    model = build_so2_vae()
    config = SpecAdamWConfig(learning_rate=2e-3, weight_decay=1e-5)
    groups, summary = build_adamw_parameter_groups(model, config=config)
    group_by_name = {group["name"]: group for group in groups}
    group_ids = {
        name: {id(parameter) for parameter in group["params"]}
        for name, group in group_by_name.items()
    }
    coefficient_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1].startswith("coeff")
    }
    gate_ids = _parameter_ids(FixedF01RadialGate, model)
    assert summary.all_trainable_parameters_covered_once
    assert summary.gate_parameters_in_gate_no_decay_group
    assert coefficient_ids <= group_ids["decay"]
    assert gate_ids == group_ids["gate_no_decay"]
    assert group_by_name["gate_no_decay"]["lr"] == pytest.approx(1e-3)
    assert group_by_name["gate_no_decay"]["weight_decay"] == pytest.approx(0.0)
    gates = tuple(
        module for module in model.modules() if isinstance(module, FixedF01RadialGate)
    )
    assert len(gates) == _GATE_COUNT
    for gate in gates:
        for a_parameter, b_parameter in (
            (gate.f0_a, gate.f0_b),
            (gate.f1_a, gate.f1_b),
        ):
            assert a_parameter.dtype == torch.float32
            assert b_parameter.dtype == torch.float32
            assert torch.isfinite(a_parameter).all()
            assert torch.isfinite(b_parameter).all()
            assert torch.allclose(a_parameter, torch.ones_like(a_parameter))
            assert torch.allclose(b_parameter, torch.zeros_like(b_parameter))
            assert torch.isfinite(torch.sigmoid(a_parameter + b_parameter)).all()


def test_two_reconstruction_steps_drive_gradients_through_the_zero_head() -> None:
    """Require true gradient updates upstream after the zero RGB head opens."""
    cast("Callable[[int], torch.Generator]", torch.manual_seed)(24014)
    model = build_so2_vae()
    optimizer, _summary = create_adamw_optimizer(
        model,
        config=SpecAdamWConfig(learning_rate=1e-3, weight_decay=0.0),
    )
    inputs = torch.randn(1, R_LAYOUT.channels, 8, 8)
    target = torch.randn_like(inputs)
    eps = torch.randn(1, L_LAYOUT.channels, 1, 1)
    representative_names = (
        "output_head.coeff00",
        "decoder_blocks.7.main_conv1.coeff00",
        "mu_head.coeff00",
        "encoder_blocks.0.main_conv1.coeff00",
        "stem_conv.coeff00",
        "stem_gate.f0_a",
        "stem_gate.f1_a",
    )
    parameters = dict(model.named_parameters())

    optimizer.zero_grad(set_to_none=True)
    first = cast("VaeForwardOutput", model(inputs, eps=eps))
    first_loss = functional.mse_loss(first.reconstruction, target)
    cast("Callable[[], None]", first_loss.backward)()
    assert _gradient_is_nonzero(parameters["output_head.coeff00"])
    assert not _gradient_is_nonzero(parameters["stem_conv.coeff00"])
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    before_second = {
        name: parameters[name].detach().clone() for name in representative_names
    }
    optimizer.zero_grad(set_to_none=True)
    second = cast("VaeForwardOutput", model(inputs, eps=eps))
    second_loss = functional.mse_loss(second.reconstruction, target)
    cast("Callable[[], None]", second_loss.backward)()
    assert torch.isfinite(second_loss)
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert all(_gradient_is_nonzero(parameters[name]) for name in representative_names)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert all(
        not torch.equal(parameters[name], before_second[name])
        for name in representative_names
    )
    assert all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )


def test_amp_facing_full_model_keeps_master_state_fp32_and_finite() -> None:
    """Exercise mixed-dtype boundaries without mutating FP32 master state."""
    model = build_so2_vae()
    inputs = torch.randn(1, R_LAYOUT.channels, 8, 8)
    eps = torch.randn(1, L_LAYOUT.channels, 1, 1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = cast("VaeForwardOutput", model(inputs, eps=eps))
        loss = (
            output.reconstruction.float().square().mean()
            + output.mu.float().square().mean()
        )
    cast("Callable[[], None]", loss.backward)()
    assert output.reconstruction.dtype == torch.bfloat16
    assert output.mu.dtype == output.logvar.dtype == torch.bfloat16
    assert output.logvar_clamped.dtype == output.z.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(
        buffer.dtype == torch.float32 and buffer.is_contiguous()
        for buffer in model.buffers()
    )
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_fixed_shape_fullgraph_repeats_without_recompile(
    *,
    autocast_enabled: bool,
) -> None:
    """Capture base/autocast model-plus-loss and reject fixed-contract recompiles."""
    model = build_so2_vae()
    inputs = torch.randn(1, R_LAYOUT.channels, 8, 8)
    eps = torch.randn(1, L_LAYOUT.channels, 1, 1)
    target = torch.randn_like(inputs)

    def model_loss(
        batch: torch.Tensor,
        noise: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        output = cast("VaeForwardOutput", model(batch, eps=noise))
        return functional.mse_loss(output.reconstruction.float(), expected) + (
            0.01 * output.mu.float().square().mean()
        )

    compiled = cast(
        "Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            model_loss,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        ),
    )
    torch_dynamo.reset()
    torch_dynamo.config.error_on_recompile = True
    try:
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            first = compiled(inputs, eps, target)
        cast("Callable[[], None]", first.backward)()
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            second = compiled(inputs, eps, target)
    finally:
        torch_dynamo.config.error_on_recompile = False
    assert torch.isfinite(first)
    torch.testing.assert_close(second, first, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("degrees", "maximum_error", "maximum_floor"),
    [(30, 0.50, 0.05), (90, 0.65, 1e-12)],
)
def test_full_model_sampled_equivariance_is_bounded_and_reportable(  # noqa: PLR0914
    degrees: int,
    maximum_error: float,
    maximum_floor: float,
) -> None:
    """Bound scalar endpoint composition while separating the resampling floor."""
    cast("Callable[[int], torch.Generator]", torch.manual_seed)(14014)
    model = build_so2_vae().eval()
    with torch.no_grad():
        model.output_head.coeff00.normal_(std=0.02)
        model.output_head.coeff01.normal_(std=0.02)
        inputs = _smooth_inputs()
        eps = torch.randn(1, L_LAYOUT.channels, 8, 8)
        mu, logvar = model.encode(inputs)
        latent, _used_eps = model.reparameterize(
            mu=mu,
            logvar=logvar.float().clamp(-8.0, 4.0),
            eps=eps,
        )
        reconstruction = model.decode(latent)
        independent_latent = torch.randn(1, L_LAYOUT.channels, 8, 8)
        for _ in range(2):
            independent_latent = functional.avg_pool2d(
                independent_latent,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        independent_latent /= independent_latent.square().mean().sqrt()
        independent_reconstruction = model.decode(independent_latent)
        full = cast("VaeForwardOutput", model(inputs, eps=eps))

        transformed_inputs = _rotate_scalar(inputs, degrees)
        transformed_eps = _rotate_scalar(eps, degrees)
        transformed_mu, transformed_logvar = model.encode(transformed_inputs)
        transformed_latent, _ = model.reparameterize(
            mu=transformed_mu,
            logvar=transformed_logvar.float().clamp(-8.0, 4.0),
            eps=transformed_eps,
        )
        transformed_reconstruction = model.decode(transformed_latent)
        transformed_independent_reconstruction = model.decode(
            _rotate_scalar(independent_latent, degrees),
        )
        transformed_full = cast(
            "VaeForwardOutput",
            model(transformed_inputs, eps=transformed_eps),
        )

    latent_crop = 1
    image_crop = 8
    errors = {
        "mu_full": _relative_rms(transformed_mu, _rotate_scalar(mu, degrees), crop=0),
        "mu_crop": _relative_rms(
            transformed_mu,
            _rotate_scalar(mu, degrees),
            crop=latent_crop,
        ),
        "logvar_full": _relative_rms(
            transformed_logvar,
            _rotate_scalar(logvar, degrees),
            crop=0,
        ),
        "logvar_crop": _relative_rms(
            transformed_logvar,
            _rotate_scalar(logvar, degrees),
            crop=latent_crop,
        ),
        "latent_full": _relative_rms(
            transformed_latent,
            _rotate_scalar(latent, degrees),
            crop=0,
        ),
        "latent_crop": _relative_rms(
            transformed_latent,
            _rotate_scalar(latent, degrees),
            crop=latent_crop,
        ),
        "decoder_full": _relative_rms(
            transformed_reconstruction,
            _rotate_scalar(reconstruction, degrees),
            crop=0,
        ),
        "decoder_crop": _relative_rms(
            transformed_reconstruction,
            _rotate_scalar(reconstruction, degrees),
            crop=image_crop,
        ),
        "independent_decoder_full": _relative_rms(
            transformed_independent_reconstruction,
            _rotate_scalar(independent_reconstruction, degrees),
            crop=0,
        ),
        "independent_decoder_crop": _relative_rms(
            transformed_independent_reconstruction,
            _rotate_scalar(independent_reconstruction, degrees),
            crop=image_crop,
        ),
        "forward_full": _relative_rms(
            transformed_full.reconstruction,
            _rotate_scalar(full.reconstruction, degrees),
            crop=0,
        ),
        "forward_crop": _relative_rms(
            transformed_full.reconstruction,
            _rotate_scalar(full.reconstruction, degrees),
            crop=image_crop,
        ),
    }
    transform_floor = _relative_rms(
        _rotate_scalar(_rotate_scalar(inputs, degrees), -degrees),
        inputs,
        crop=image_crop,
    )
    downsample = _FixedF01Downsample2x(R_LAYOUT.channels)
    downsampled = cast("torch.Tensor", downsample(inputs))
    transformed_downsampled = cast(
        "torch.Tensor",
        downsample(_rotate_scalar(inputs, degrees)),
    )
    downsample_phase_errors = {
        "downsample_phase_full": _relative_rms(
            transformed_downsampled,
            _rotate_scalar(downsampled, degrees),
            crop=0,
        ),
        "downsample_phase_crop": _relative_rms(
            transformed_downsampled,
            _rotate_scalar(downsampled, degrees),
            crop=4,
        ),
    }
    assert math.isfinite(transform_floor)
    assert transform_floor <= maximum_floor
    assert all(math.isfinite(value) for value in errors.values())
    assert max(errors.values()) <= maximum_error, errors
    assert all(math.isfinite(value) for value in downsample_phase_errors.values())
    assert max(downsample_phase_errors.values()) <= maximum_error, (
        downsample_phase_errors
    )
