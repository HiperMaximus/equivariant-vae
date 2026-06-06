# Equivariant VAE Paper Repository

This repository supports the SIPAIM 2026 paper and the experiments behind it.
The active research direction is a comparison between a normal denoising VAE and
a continuous `SO(2)`-steerable denoising VAE for histopathology patch
representations.

## Current Source Of Truth

Read the canonical landing sequence before changing architecture, evaluation,
paper text, workflow, or Overleaf sync:

- [AGENTS.md](AGENTS.md)
- [CURRENT.md](CURRENT.md)
- [GOAL.md](GOAL.md)
- [docs/repo_goal_and_requirements.md](docs/repo_goal_and_requirements.md)
- [docs/issue_image_inventory.md](docs/issue_image_inventory.md)
- [docs/equivariant_vae_transition_plan.md](docs/equivariant_vae_transition_plan.md)
- [docs/overleaf_sync_workflow.md](docs/overleaf_sync_workflow.md)
- [docs/agentic_review_workflow.md](docs/agentic_review_workflow.md)
- [docs/spec_driven_development.md](docs/spec_driven_development.md)
- [docs/specs/README.md](docs/specs/README.md)
- [docs/decisions/README.md](docs/decisions/README.md)

Keep these files current. If old notes become misleading, replace or delete
them instead of leaving contradictory historical artifacts in place.

## Active Paper

Paper source:

```text
paper/sipaim2026
```

Tracked advisor-facing PDF:

```text
paper/sipaim2026/sipaim2026.pdf
```

Overleaf project:

```text
https://www.overleaf.com/project/69c614433cbc9e46cf226d24
```

Compile and refresh the tracked PDF:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

Follow the full safe Overleaf workflow in
[docs/overleaf_sync_workflow.md](docs/overleaf_sync_workflow.md). The short
version is: check/setup, ask for permission before pull/push, compile, commit,
then subtree-push with explicit confirmation.

```bash
./scripts/sipaim_overleaf_sync.sh check
./scripts/sipaim_overleaf_sync.sh setup
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull

# edit paper/sipaim2026
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Update SIPAIM paper"

OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

Do not push the whole repo to Overleaf.

## Agent Preflight

Before substantial Codex or Claude work, run:

```bash
./scripts/agent_preflight.sh
```

Claude should read [CLAUDE.md](CLAUDE.md), but that file is only an adapter to
the canonical repo instructions.

The same local checks are available as VS Code tasks when opening this repo.

For substantial workflow, architecture, evaluation, or paper-claim changes, use
the adversarial clean-context subagent process in
[docs/agentic_review_workflow.md](docs/agentic_review_workflow.md).

For substantial implementation work, use spec-driven development:
[docs/spec_driven_development.md](docs/spec_driven_development.md). The first
active implementation spec is
[docs/specs/0001-translatable-normal-vae-baseline.md](docs/specs/0001-translatable-normal-vae-baseline.md).

For Python changes, run the strict quality gate:

```bash
uv sync --python 3.12 --group dev
./scripts/python_quality.sh
```

It targets Python 3.12 with uv, CPU-only local PyTorch on Linux, Ruff `ALL`, and
strict BasedPyright. See
[docs/specs/0002-strict-python-quality-gate.md](docs/specs/0002-strict-python-quality-gate.md).

## Current Experiment Horizon

The comparison should be fair by construction:

- same histopathology patch pipeline;
- same train/validation/test split policy;
- same latent target;
- same macro architecture schedule;
- same optimizer budget and validation access;
- same metric scripts;
- same qualitative artifact protocol.

The non-equivariant baseline must avoid operations that do not translate cleanly
to the continuous `SO(2)` steerable model.

## Required Evaluation Artifacts

Do not lose these advisor/issue requirements:

- SSIM, MAE, MSE, PSNR with mean, standard deviation, and sample count `n`;
- boxplots for reconstruction metrics;
- training/evaluation dashboards analogous to the GitHub issue screenshots;
- fixed 25-patch original/reconstruction visualizations;
- rotated-input qualitative artifacts with fixed continuous angles;
- EQ-VAE-style latent visualization with top principal components, latent maps,
  transformed latent maps, and difference/error maps;
- equivariance tests for nonlinearities, normalization, upsampling, VAE
  sampling, and latent statistics.

The detailed tracker is:

```text
docs/repo_goal_and_requirements.md
```

The issue screenshot inventory is:

```text
docs/issue_image_inventory.md
```

## Related Repository

The thesis repository is separate:

```text
/home/maximus/Documents/Tesis/Tesis
https://github.com/HiperMaximus/Tesis.git
```

Update the thesis only after paper results and claims stabilize.
