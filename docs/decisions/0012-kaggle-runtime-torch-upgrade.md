# Decision 0012: Kaggle Run Kernels Upgrade Torch at Runtime (internet ON; narrows 0011)

Date: 2026-07-17; clarified 2026-08-02

Decision: every Kaggle kernel sets `enable_internet = "true"` and runs
`pip install --upgrade torch torchvision torchaudio` at the top of `run.py`,
before importing `eqvae`, so the run executes on the newest release available from
PyPI instead of Kaggle's older base-image torch. Kaggle's installed version is never a
fallback authority. Our CODE still ships as the embedded payload
(decision 0011) and code sources remain embedded/empty — only the torch stack now
comes from PyPI. Current and experimental runtime modes exposed by that installed release
are valid benchmark candidates; bounded execution and measured dual-T4 epoch throughput decide
whether they stay, not an age or stability preference. This narrows, and for the kernels supersedes, the
`enable_internet = "false"` face of decision 0011's hermeticity invariant.

Declared dataset sources may include the real patch dataset and, for later sessions, the private
resume dataset; this does not relax the embedded-code rule.

The selected runtime records a complete performance fingerprint: torch version/build,
`torch.version.cuda`, driver version, GPU name/count/compute capability, and relevant compiler/
backend versions. Every later kernel still upgrades to latest; any performance-relevant fingerprint
mismatch from the selection run forces a new bounded bake-off before paid training. Floating latest
is intentional, but cross-stack performance assumptions are not.

## Why

Kaggle's base image lags the repo's torch target (measured 2026-07-17: the Kaggle
GPU image ships torch 2.10.0+cu128; local is 2.13). The compiled fast-path recipe
(`torch.compile` + `ddp_optimizer_whole_step` + compiled autograd) is exactly the
surface that shifts between torch minors, so a green LOCAL gate on 2.13 would be
validating a version Kaggle does not run — a broken-30h-run risk. Upgrading on
Kaggle pins local/Kaggle torch PARITY and rides each minor's speedups (the run is
speed-first, GOAL.md).

## Why re-making the reverted flip is sound this time

A flip to `enable_internet = "true"` was applied and then reverted on 2026-07-16
(decision 0010, "verify the premise before changing a pin"). That revert was
correct THEN: the flip was unmotivated and the hermeticity premise held. The
premise is now explicitly re-examined and only PARTIALLY relaxed:

- Hermeticity has three faces (decision 0011): embedded code, empty `*_sources`,
  internet off. Only the third is relaxed. The "undeclared CODE-source" risk that
  Spec 0003 Known Risks warns about is NOT reintroduced — code stays baked in and
  the source lists stay empty.
- The one thing traded is torch-version reproducibility (floating latest instead
  of a pinned wheel). Mitigated by (a) local always being upgraded and gated
  before each paid run, minimising the gap, and (b) the run recording
  `torch.__version__` / `torch.version.cuda` in telemetry (FU-045), so the exact
  version is captured.

Measured 2026-07-17 on a throwaway GPU probe: with internet on,
`pip install --upgrade torch torchvision torchaudio` → torch 2.13.0+cu130 (the T4
driver supports CUDA 13), with matmul / autograd / `torch.compile` all verified
computing on the T4. RAPIDS/cudf conflict warnings appear but are harmless — our
active code imports none of torchvision/torchaudio/RAPIDS. The two CPU kernels
(`setup_smoke`, `fixed25_selector`) use `--index-url .../whl/cpu` → torch 2.13+cpu,
matching the local venv exactly.

## Mechanism

A stdlib-only `_ensure_latest_torch(*, cpu_only)` helper in each
`kaggle/kernels/*/run_template.py` runs the pip upgrade before the first torch
import. It no-ops unless `/kaggle/working` exists, so the local gate and simulation
never touch the network; `EQVAE_SKIP_TORCH_UPGRADE=1` forces it off on Kaggle too.
GPU kernels use the default index (cu130); the two CPU kernels use the cpu index.
`enable_internet = "true"` lives in each `kernel-metadata.json`, each template's
`KERNEL_METADATA` mirror, and the `scripts/kaggle_kernel.sh` push-guard validators.

## When this folds away

If the repo goes public and delivery switches to `pip install git+...@<sha>`
(decision 0011's trigger), internet is already on and torch rides along with that
install — this note merges into that transition.
