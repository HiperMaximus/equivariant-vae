# Equivariant VAE Research Repository

This repository contains the completed matched comparison of a normal
denoising VAE and a repo-owned continuous-`SO(2)` steerable VAE for UBC-OCEAN
histopathology patches, plus downstream WSI/tissue evaluation and the working
paper.

The professor-facing result is:

- `reports/professor/informe_final_experimento_eqvae.docx`;
- `reports/professor/informe_final_experimento_eqvae.pdf`;
- 29 Letter pages and 22 inline figures;
- final Spanish update on
  [GitHub issue #6](https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496).

The issue remains open. Exact results, hashes, limitations and the active
experiment frontier are in [CURRENT.md](CURRENT.md).

## Read First

Read [AGENTS.md](AGENTS.md), [CURRENT.md](CURRENT.md), [GOAL.md](GOAL.md), and
only the scientific spec relevant to the current task. Numerical parameters
live in machine-readable contracts; a rerun should normally be a small change
to an existing contract or runner.

## Current Scientific State

- Both VAEs completed 60,000 updates and are frozen.
- Both produce FP32 posterior-`mu` embeddings of shape `[16,32,32]`.
- The shared 152-WSI cohort uses a frozen 106/23/23 WSI split.
- Full foreground embeddings cover 1,750,221 patches for each model.
- MIL diagnosis, tissue label efficiency and reconstruction sealed tests are
  complete and remain isolated from tuning.
- The former all-25 dense rotation diagnostic is superseded because it mixed
  opposite conventions at cardinal angles. Spec 0050 corrected the operator and
  completed the fixed-validation post-hoc geometry audit.
- The matrix-free SLQ calibration and first exact-`C4` Stage A comparison are
  complete. The next experiment tests decoder-fiber bridges, four-sided
  pullback geodesics, tangent transport and free quarter-turn continuation.

The validated conclusion remains mixed for reconstruction and downstream
tasks. Corrected rotation evidence does not demonstrate a robust SO(2)-specific
regularity advantage, a transferable shared low-dimensional action, or local
content--pose factorization. The raw-`mu` exact-quarter control does not favor
SO(2); separately, the SO(2) decoder realizes the prescribed spatial C4 action
at 90/180/270 degrees, without establishing continuous SO(2), encoder, or full
D4/O(2) equivariance.

## Repository Layout

```text
src/eqvae/          model, data, training, inference and evaluation code
configs/            frozen experiment contracts
tests/              local verification
kaggle/kernels/     CLI-managed Kaggle script kernels
docs/specs/         detailed implementation and experiment contracts
docs/decisions/     settled design decisions
runs/               ignored local/remote evidence and receipts
reports/professor/  final advisor-facing report
paper/sipaim2026/   working manuscript subtree
```

## Python

Dependency truth is `pyproject.toml` plus `uv.lock`. Use focused tests for the
mathematics being changed. Local execution uses CPU PyTorch; CUDA, Inductor and
dual-T4 behavior are measured on Kaggle.

## Kaggle

Use the generic CLI workflow:

```bash
./scripts/kaggle_kernel.sh help
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
```

Preserve every resource using its canonical owner-qualified locator and version. See
[docs/kaggle_cli_workflow.md](docs/kaggle_cli_workflow.md).

## Paper And Overleaf

SIPAIM 2026 was not submitted. The working manuscript is
`paper/sipaim2026`; its tracked advisor-facing PDF is
`paper/sipaim2026/sipaim2026.pdf`.

Compile through:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

Never push the whole repository to Overleaf; use
`scripts/sipaim_overleaf_sync.sh`. The thesis repository is separate.

## Current Boundary

Presentation of the final report is ready. Paper, thesis, Overleaf, commits,
pushes, public derived-data release, further issue updates and instrumented WSI
attribution are separate tasks.
