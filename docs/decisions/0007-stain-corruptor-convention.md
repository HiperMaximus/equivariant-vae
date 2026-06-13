# 0007: Stain Corruptor Convention

Status: accepted
Date: 2026-06-13

## Decision

Spec 0001 uses a scikit-image-compatible HED stain-coordinate convention as the
reference oracle, but runtime corruption is implemented in repo-owned PyTorch
code so it can run in the compiled training path.

The public corruptor API takes and returns NCHW RGB tensors normalized to
`[-1, 1]`. Internally, the corruptor converts to RGB `[0, 1]`, applies
scikit-compatible HED/RGB math, applies stain-coordinate jitter and image-space
Gaussian noise, converts back to `[-1, 1]`, and clamps the final corrupted input
to `[-1, 1]`.

The first profile uses conservative H/E jitter and tiny third-axis residual
jitter. The residual-axis jitter is an anti-corruption-signature device, not a
claim about biological DAB variation in the H&E dataset. Wider historical FSQ
H/E ranges are a named benchmark profile, not the first-run default.

Corruption RNG is stateless per sample. The semantic seed is derived from the
base corruption seed, split, semantic patch key `{split}:{wsi_id}:{label}:{x}:{y}`,
corruption step, corruption view, and corruption version. Rank/world size are
logged as execution context but are not part of the semantic per-sample seed.
Clean validation and clean test views do not call the corruptor or consume
corruption RNG.

## Rationale

The historical FSQ corruptor followed the right broad idea: implement the stain
transform directly in Torch rather than calling an image library in the hot path.
However, it mixed an ambiguous channel-first matrix convention, a historical
linear-RGB optical-density path, and global RNG calls. Those details are too
fragile for benchmark evidence.

Using scikit-image as an oracle gives a documented convention for tests, while
the PyTorch implementation keeps training compile-friendly. Excluding rank and
physical file order from the semantic seed keeps corruption comparable when the
same patch moves between single-GPU, DDP, branchless, indexed, or resumed runs.

## Consequences

The next local corruption slice must prove HED/RGB oracle agreement, per-channel
semantics, deterministic stateless RNG, clean-validation RNG non-consumption,
range/clamp telemetry, and visual QA through a non-promotable
`benchmark/stain_corruptor_qa.json` artifact before corruption is integrated
into real training.
