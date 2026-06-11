# 0005: Gated Activation Policy

Status: accepted
Date: 2026-06-11

## Decision

Spec 0001 uses learned two-parameter gates without learned activation amplitude
`gamma`.

Scalar fields in both models use the same pointwise gate:

```text
gate_i = sigmoid(a_i * x_i + b_i)
out_i = gate_i * x_i
```

Nontrivial continuous `SO(2)` irrep fields use the equivariant radial gate:

```text
r = sqrt(||v||**2 + eps) = sqrt(u**2 + w**2 + eps)
gate = sigmoid(a_i * r + b_i)
out = gate * v
```

First-run initialization and optimization:

- initialize `a_i = 1`, `b_i = 0`;
- do not use learned activation amplitude `gamma`;
- configure radial-gate `eps` explicitly and start with `eps = 1e-4`;
- activation gate parameters use `weight_decay = 0`;
- activation gate parameters use `lr_multiplier = 0.5`;
- include gate parameters in model parameter counts and report them separately;
- benchmark and log gate health before the first full run, including
  saturation, `a,b` ranges, gate gradients/updates, and input/output RMS.

## Rationale

The learned `a,b` gate parameters intentionally add pointwise activation
expressivity to the scalar paths. They are used in both the non-equivariant
baseline and `SO(2)` scalar/trivial fields so this extra scalar expressivity is
matched rather than hidden in only one model.

Nontrivial `SO(2)` vector/irrep fields cannot use arbitrary componentwise SiLU,
ReLU, or GELU without breaking equivariance. A radial gate is the practical
equivariant replacement because it uses an invariant radius and multiplies the
whole vector by one scalar.

The radial norm is computed as `sqrt(||v||**2 + eps)`. The `eps` term is not a
mathematical change to the representation; it is a numerical guard for stable
gradients near zero norm and must be large enough to behave safely under FP16/AMP
execution.

The activation amplitude `gamma` is intentionally omitted because normalization
affine parameters and learned convolutions already control amplitude. Adding
`gamma` is a later ablation if the equivariant model is underpowered.

## Consequences

Paper claims must state that the equivariant model uses equivariance-preserving
radial gates for non-scalar fields. Gate parameter counts and saturation
statistics should be logged so learned activation expressivity is visible rather
than hidden. The gate-health benchmark is instrumentation, not an activation
ablation: spec 0001 should not branch into multiple nonlinearities unless a
later spec explicitly opens that question.
