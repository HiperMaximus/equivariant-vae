# Specs

This directory contains implementation and experiment specs.

Use these specs to define what "done" means before changing code, evaluation
logic, paper artifacts, or workflow tooling.

Read first:

1. `../spec_driven_development.md`
2. `template.md`
3. the active specs in the table below

## Status Values

- `draft`: useful planning text, not ready for implementation.
- `draft active`: active workstream, but blocked by open questions or missing
  verification details.
- `locked / implementation-ready`: enough contract, acceptance criteria, and
  commands exist to start coding.
- `implemented`: accepted and verified.
- `superseded`: retained only for history, with a pointer to the replacement.

## Active Specs

| Spec | Status | Blocked By | Next Action |
| --- | --- | --- | --- |
| `0001-translatable-normal-vae-baseline.md` | Normal-VAE control implemented and verified at update 60000; beta `0.01`, fixed-25 evidence, and final checkpoint are locked. | Matched continuous `SO(2)` implementation/run, downstream probes, and sealed masked-WSI test evidence. | Preserve the shared baseline contract during the separately authorized full equivariant-VAE implementation. |
| `0002-strict-python-quality-gate.md` | Active and passing: 701 tests passed, 1 expected GPU-only skip, Ruff clean, BasedPyright 0 errors on 2026-08-12. | None. | Rerun `./scripts/python_quality.sh` for Python changes. |
| `0003-kaggle-cli-execution-workflow.md` | Implemented and proven through guarded push/status/output plus three checkpoint-only baseline sessions. | Remote actions still require explicit approval and confirmation variables. | Reuse the guarded workflow for the continuous-`SO(2)` experiment. |
| `0006-selected-runtime-local-mechanics.md` | Implemented, locally verified, and exercised by the completed baseline. | None. | Maintain fail-closed runtime identity/checkpoint checks. |
| `0007-real-ubc-ddp-amp-selected-runtime-runner.md` | Implemented and remote-verified through update 60000. | None for the normal-VAE runner. | Reuse shared mechanics; change only architecture-specific `SO(2)` details. |
| `0008-canonical-fixed32-and-remote-debug-tiny-readiness.md` | Implemented; debug/resume/tiny proof complete. | None. | Retain as the short pre-long-run proof pattern. |
| `0009-first-full-selected-runtime-training-run.md` | Superseded historical v1; durability/DDP lessons are incorporated by Spec 0011. | None; v1 is not a resume base. | Read only for history. |
| `0004-sipaim-paper-scaffold.md` | Draft active scaffold; the normal-VAE result is now available evidence. | Continuous `SO(2)`, downstream WSI classifier, and sealed test results remain pending. | Keep claims bounded until the matched comparison is complete. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |
| `0010-fixed25-equivariance-artifact-protocol.md` | Implemented, committed, and exercised at all 20 baseline boundaries with frozen real originals. | Matched continuous-`SO(2)` evidence remains. | Reuse the identical selector, angles, latent views, grids, and error maps. |
| `0011-reusable-goal-derived-runtime-and-compiled-fastpath.md` | Baseline runtime search, beta decision, three-session execution, update-60000 checkpoint, metrics, and visual verification complete; its selected bundle also passed Spec 0013 mechanics transfer. | Full-`SO(2)`-VAE implementation and later training authorization. | Reuse the selected compiled bundle in narrow dual-T4 checks when the executable full model exposes a concrete compile/VRAM/throughput question; do not recreate a tuner. |
| `0012-continuous-so2-vae-architecture.md` | Radial/F2 decision, equal-copy F01 count/init handoff, and Spec 0013 mechanics acceptance are complete. | Full-VAE coding authorization. | Keep profiles/layout/mechanics fixed; do not assemble the VAE without explicit authorization. |
| `0013-fixed-f01-architecture-probe.md` | Complete: padded-`bmm`/direct compiled mechanics pass correctness, runtime ratios, DDP/AMP/compile, and VRAM; raw CV is retained diagnostically by explicit user decision. | Full-VAE coding authorization. | Do not rerun or add mechanics/runtime arms; request separate authorization before assembling the VAE. |

Guard authorization phrases retained for local push-guard scripts:

- synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`;
- capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`;
- runtime-selection kernel contract is `runtime_selection_kernel_ready`;
- selected-runtime debug/tiny gate contract is `selected_runtime_debug_gate_contract_ready`.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
