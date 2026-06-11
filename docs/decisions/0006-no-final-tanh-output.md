# 0006: No Final Tanh Output

Status: accepted
Date: 2026-06-11

## Decision

Spec 0001 does not use a final `tanh`, sigmoid, or hard clamp inside the model
forward path.

The decoder ends with a zero-initialized 5x5 RGB convolution that returns raw
normalized RGB reconstruction values. L1 is computed directly on raw `x_hat` and
`x_clean` in normalized image coordinates. SSIM, PSNR, saved images, and visual
artifacts use a clamped image-domain projection:

```text
x_hat_img = clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)
```

## Rationale

The final `tanh` can hide boundary behavior and introduce saturation in a
denoising objective. Zero-initializing the final RGB head gives a stable initial
midpoint reconstruction while leaving the model free to learn the output range.
Range telemetry must expose excursions below `-1` or above `1` instead of
silently clipping them in the model.

## Consequences

Training logs must include reconstruction range telemetry. Any downstream metric
or artifact that expects image values in `[0, 1]` must explicitly project and
clamp the raw model output outside the forward path.
