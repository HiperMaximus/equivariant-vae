# Decision 0007: Stain Corruptor Convention

Status: active

## Decision

Training corruption uses repo-owned PyTorch HED/RGB transforms compatible with
the pinned scikit-image oracle. The public API accepts and returns NCHW RGB
tensors normalized to `[-1,1]`.

The corruptor converts to RGB `[0,1]`, applies conservative H/E jitter, a small
residual-axis jitter and image-space Gaussian noise, then converts back and
clamps the corrupted input to `[-1,1]`. The clean normalized patch remains the
reconstruction target.

Runtime corruption uses native fast RNG. Bit-identical per-sample corruption is
not required; numerical comparisons use documented tolerances. Validation is
clean or deterministically constructed according to its named view.

## Consequences

- Matrix orientation and normalization are pinned by oracle tests.
- Corruption telemetry must prove a nonzero input-target delta.
- Residual-axis jitter is an implementation device, not a biological DAB claim
  for H&E slides.
