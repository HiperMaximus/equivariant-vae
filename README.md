# Equivariant VAE Research Repository

This repository contains the completed matched comparison of a normal
denoising VAE and a repo-owned continuous-`SO(2)` steerable VAE for UBC-OCEAN
histopathology patches, plus downstream WSI/tissue evaluation and the working
paper.

The professor-facing result is:

- `reports/professor/informe_final_experimento_eqvae.docx`;
- `reports/professor/informe_final_experimento_eqvae.pdf`;
- 20 pages and 15 figures;
- final Spanish update on
  [GitHub issue #6](https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496).

The issue remains open. Exact results, hashes, limitations and authorization
boundaries are in [CURRENT.md](CURRENT.md).

## Read First

1. [AGENTS.md](AGENTS.md)
2. [CURRENT.md](CURRENT.md)
3. [GOAL.md](GOAL.md)
4. [Requirements](docs/repo_goal_and_requirements.md)
5. [Issue image inventory](docs/issue_image_inventory.md)
6. [Architecture contract](docs/equivariant_vae_transition_plan.md)
7. [Kaggle workflow](docs/kaggle_cli_workflow.md)
8. [Data and behavior contract](docs/behavior_inventory_kaggle.md)
9. [Specs index](docs/specs/README.md)
10. [Decisions index](docs/decisions/README.md)

Run the repository preflight before substantial work:

```bash
./scripts/agent_preflight.sh
```

## Current Scientific State

- Both VAEs completed 60,000 updates and are frozen.
- Both produce FP32 posterior-`mu` embeddings of shape `[16,32,32]`.
- The shared 152-WSI cohort uses a frozen 106/23/23 WSI split.
- Full foreground embeddings cover 1,750,221 patches for each model.
- MIL diagnosis, tissue label efficiency and reconstruction sealed tests are
  complete; their one-shot authorities are consumed.
- The all-25 rotation diagnostic evaluates every integer angle from 0° to
  359°.
- No Kaggle job is active.

The conclusion is mixed: normal is slightly better descriptively on
reconstruction; `SO(2)` has favorable downstream point estimates without a
general primary-test advantage; and `SO(2)` is locally smoother at one-degree
resolution without more uniform traversal or stronger PCA planarity.

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

## Python Workflow

Dependency truth is `pyproject.toml` plus `uv.lock`; do not add a root
`requirements.txt`. Use the existing Python 3.12 `.venv`:

```bash
./scripts/python_quality.sh
```

The quality script does not install dependencies. Ask before running:

```bash
uv sync --locked --python 3.12 --group dev
```

Local verification uses CPU PyTorch. CUDA, Inductor and dual-T4 behavior belong
on Kaggle.

## Kaggle

Use the guarded CLI workflow:

```bash
./scripts/kaggle_kernel.sh help
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
```

Remote reads and writes require explicit user authorization and the exact
confirmation variables enforced by the script. Preserve every resource using
its canonical owner-qualified locator and version. See
[docs/kaggle_cli_workflow.md](docs/kaggle_cli_workflow.md).

## Paper And Overleaf

SIPAIM 2026 was not submitted. The working manuscript is
`paper/sipaim2026`; its tracked advisor-facing PDF is
`paper/sipaim2026/sipaim2026.pdf`.

Compile through:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

Never push the whole repository to Overleaf. Remote reads, pulls and pushes
require explicit authorization and must use
`scripts/sipaim_overleaf_sync.sh`. The thesis repository is separate and must
not be edited without an explicit request.

## Current Boundary

Presentation of the final report is ready. Paper, thesis, Overleaf, commits,
pushes, public derived-data release, further issue updates and instrumented WSI
attribution are separate tasks.
