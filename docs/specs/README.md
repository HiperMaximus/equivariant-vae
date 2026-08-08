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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; narrow local slices through `corruption_ready` are implemented; runtime-selection v5 selected AMP conservative dual-T4 bs12 indexed-mask at `27.381321` samples/sec; compact relaxed scalar-gate AMP v6 completed fail-closed and kept v5; Spec 0006 local selected-runtime mechanics, Spec 0007 real selected-runtime runner, and Spec 0009 full-run local workflow are locally implemented; Spec 0008 remote debug/tiny v5 passed strict output verification; the first Spec 0009 full-kernel push was accepted as version 1, later canceled by Kaggle, and downloaded with checkpoints through update 43750; the successful FSQ notebook is recorded as runtime-reference material only | Broad implementation remains blocked by a completed and strict-verified full selected-runtime run proof, fixed real 25-patch visual QA, real evaluator/artifact proof, future `SO(2)` count ceiling, final adversarial spec review, and the sealed masked-WSI test shard for paper claims. The immutable v5 selected-runtime artifact still says `full_training_launch_ready = false`, but the external Spec 0008 proof now satisfies the pre-long-run debug/tiny gate. FSQ quantization, PixelShuffle/sub-pixel upsampling, final `tanh`, exact old corruptor behavior, and rot90/discrete-latent artifacts remain quarantined from the continuous `SO(2)` path; however, the useful FSQ fixed-25 rotated-input/transformed-latent evaluation pattern is reused as a repo-owned rot90 `{0,90,180,270}` replacement, now IMPLEMENTED and committed as Spec 0010 (the real fixed-25 selector is generated and committed, FU-041) for paper-promotable qualitative evidence. | Broad Spec 0001 stays draft active. Spec 0010 (`fixed25-equivariance-artifact-protocol`) is IMPLEMENTED and committed (gate green) and carries the fixed-25 rotated/latent artifact protocol; the real fixed-25 selector is generated and committed (FU-041), and local resume-policy work remains before any resume upload/push. |
| `0002-strict-python-quality-gate.md` | active strict gate; semantic test-intent and redundancy contract added 2026-08-02; last verified pre-v4 tree passed 675 tests with 1 skipped | The current substantial v4/partial-v3 dirty tree has not passed the full gate; test intent cannot be certified by docstring presence alone. | During each touched-area audit, keep only the cheapest unique goal-relevant failures and rerun the repo quality gate before handoff. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; the CLI push/status/output path is proven; v5 is a historical fallback only; Spec 0009 adds dedicated full-run actions; full-kernel v1 was canceled | Canceled v1 has resumable checkpoints but no metrics/benchmark, so strict verification fails. Remote reads/writes require fresh explicit approval plus confirmation variables. | Maintain the guarded CLI while Spec 0011 v4 replaces the historical runtime before any full-run/resume action. |
| `0006-selected-runtime-local-mechanics.md` | implemented / locally verified; adversarial fixes added for strict linked-runtime-proof validation, checkpoint progress consistency, observed FP32/AMP-off local telemetry, and structured-readiness-derived push blocking | None for the local mechanics slice. Full-run launch remains outside this spec. | Maintain local non-promotable guarantees while Spec 0009 adds the full-run workflow. |
| `0007-real-ubc-ddp-amp-selected-runtime-runner.md` | implemented / locally verified; extended by Spec 0009 full-run mode | No remaining local runner blocker for the Spec 0009 full-run launch surface. Remote full-run execution remains outside this spec and still requires explicit approval. | Maintain runner/debug non-promotable guarantees while Spec 0009 owns the first full-run guard and verifier. |
| `0008-canonical-fixed32-and-remote-debug-tiny-readiness.md` | implemented; historical v5 debug/tiny output passes strict verification | None within Spec 0008; its proof remains a prerequisite pattern, not approval of v5 as the new runtime. | Locked Spec 0011 v4 must select and re-prove the current runtime before any full run. |
| `0009-first-full-selected-runtime-training-run.md` | version 1 canceled after checkpoint `step_043750.pt`; future atomic interval metrics/checkpoints and boundary breadcrumbs are implemented locally | V1 lacks recoverable metrics/benchmark and is not a resume base; the fresh full run is blocked by Spec 0011 v4 runtime selection, real-data quality/LR gates, and private resume publication. | Finish Spec 0011 through promotion and resume workflow, then launch a fresh paper-promotable run only with explicit approval. |
| `0004-sipaim-paper-scaffold.md` | draft active / scaffold slice implemented | The paper can be outlined and grounded in the thesis/repo evidence, but selected runtime, full VAE runs, continuous `SO(2)` results, downstream WSI classifier results, and sealed masked-WSI test evidence are still pending. | Keep the SIPAIM paper compile-safe, use thesis figures only as paper-local copied assets, and leave result claims as explicit placeholders until evidence exists. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |
| `0010-fixed25-equivariance-artifact-protocol.md` | IMPLEMENTED and committed (Spec 0010 `8457233`, FU-040); the real fixed-25 selector is generated, promoted, and committed (FU-041 `36797c5`; frozen originals in `docs/data/fixed25/`); 6-lens post-impl adversarial review integrated; see decision 0009 | Real fixed-25 selector is DONE and committed (FU-041 `36797c5`); FU-039 is decided (fresh restart); the remaining external blocker is the paper-promotable full run itself (Spec 0011 S19). Settled: rot90 `{0,90,180,270}`; full-save-every-boundary; unmasked/full-frame error maps; PCA = EQ-VAE `pca_to_rgb` + first3; retired `reconstruction_samples.pt` from the full run; scale/LPIPS out of scope; shared `model_base` eval-config; standalone in-run + offline evaluator. | Implemented (decision 0009): `src/eqvae/artifacts/fixed25_equivariance.py` + `src/eqvae/cli/fixed25_equivariance.py`; runner boundary-loop evaluator + FU-039-durable `metrics/equivariance_25.csv` (merge-not-gather, resume-prefix validator, broadcast-guarded rank-0 eval); gate requires fixed-25 (incl. error maps + `promotable=real`) and retires the placeholder; shared `fixed25_equivariance` config block; CPU tests. Same frozen 25 validation images for BOTH models. Extends the Spec 0009 runner/verifier. Remaining: none for the protocol itself; the paper-promotable full run that exercises it is Spec 0011 S19 (Kaggle). |
| `0011-reusable-goal-derived-runtime-and-compiled-fastpath.md` | v4 locked / implementation-ready; two independent xhigh review tracks have no remaining blocker/high; canonical six-file evidence is tracked; partial v3 code remains fail-closed | No remote action is authorized. Kaggle GPU time is not minimized; the finite search is resumable without obsolete total caps. | Refactor the partial controller around exhaustive acceleration inventory, proved contextual exclusions, maximal-cover/direct-interaction recipes, joint recipe/batch search, and audited reuse of the 309 rows. |

Guard authorization phrases retained for local push-guard scripts:

- synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`;
- capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`;
- runtime-selection kernel contract is `runtime_selection_kernel_ready`;
- selected-runtime debug/tiny gate contract is `selected_runtime_debug_gate_contract_ready`.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
