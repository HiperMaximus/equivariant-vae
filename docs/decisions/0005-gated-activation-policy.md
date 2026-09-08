# Decision 0005: Gated Activation Policy

Status: active

## Decision

Scalar fields in both models use:

```text
gate_i = sigmoid(a_i * x_i + b_i)
out_i = gate_i * x_i
```

Nontrivial F1 fields use the equivariant radial form:

```text
r = sqrt(u**2 + w**2 + eps)
gate = sigmoid(a_i * r + b_i)
out = gate * (u,w)
```

Parameters initialize at `a = 1`, `b = 0`; no learned amplitude `gamma` is
used. The configured epsilon must be safe under the selected FP16 runtime.

## Consequences

- Baseline and `SO(2)` branches retain comparable learned gating capacity.
- Gate health records ranges, saturation, gradients, updates and input/output
  RMS.
- Pointwise nonlinearities may not split or mix F1 components independently.
