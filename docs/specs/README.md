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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; narrow local slices through `corruption_ready` are implemented; kaggle smoke is `kaggle_smoke_ready`; synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`; capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`; synthetic timing remote v4 is complete; capped real-data runtime pretest v4/v5/v6/v7/v8 artifacts are downloaded and non-promotable; selected-runtime writer and Kaggle executor/kernel plumbing for `v8_shortlist_eager_amp_then_dual_gate` are implemented and remote v3 wrote `selected_runtime.json`; the local `selected_runtime_v3_efficiency_followup` contract and executor policy plumbing are implemented; runtime-selection v4 was downloaded and failed closed on writer false negatives; runtime-selection v5 downloaded to `runs/kaggle/runtime_selection_v5`, passed strict local replay, and selected AMP conservative dual-T4 bs12 indexed-mask at `27.381321` samples/sec; the successful FSQ notebook is recorded as runtime-reference material only | Broad implementation remains blocked by fixed real 25-patch visual QA, a compact broader AMP/non-conservative follow-up before the first long run, selected-runtime debug, checkpoint/resume proof, tiny-overfit proof, real train/resume/evaluator/artifact writers, future `SO(2)` count ceiling, real fixed selector generation from real Kaggle shards, final adversarial spec review, and the sealed masked-WSI test shard for paper claims. Remote v8 fixed the v7 quantile evidence-plumbing failure and was used only as shortlist provenance. Runtime-selection v3 completed and selected `dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked` with passing dual-T4 timing and linked proofs; runtime-selection v5 supersedes it for efficiency but is not full-training-launch-ready. A less conservative AMP policy such as `amp_scalar_gate_relaxed` should be tested and may replace v5 only if it passes the same proof/debug/replay/tiny-overfit gates. FSQ quantization, PixelShuffle/sub-pixel upsampling, final `tanh`, exact old corruptor behavior, and rot90/discrete-latent artifacts remain quarantined from the continuous `SO(2)` path. | Use `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json` as the fallback selected runtime. Next, implement the broader AMP follow-up and selected-runtime debug/resume/tiny-overfit gate. Do not ask for a long real training launch until those proof artifacts pass. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; synthetic setup-smoke remote v1 passed; synthetic timing remote v4 passed; real-data runtime pretest remote v4/v5/v6/v7/v8 artifacts are downloaded; selected-runtime kernel path is guarded as `runtime_selection_kernel_ready`; `./scripts/kaggle_kernel.sh preflight-runtime-selection` is the mandatory local semantic preflight before any future runtime-selection push | Runtime-selection v3 downloaded to `runs/kaggle/runtime_selection_v3`, proved real dual-T4 DDP timing, wrote `benchmark/selected_runtime.json`, and confirmed the selected row. Runtime-selection v4 downloaded to `runs/kaggle/runtime_selection_v4` and failed closed on proof-policy false negatives; local replay after `fc5227d` passes. Runtime-selection v5 downloaded to `runs/kaggle/runtime_selection_v5`, wrote selected runtime, and passed strict local replay under current `main`. Full training/full-run launchers are still not implemented. | User approved the selected-runtime efficiency-follow-up Kaggle push/status/output with the normal guard variables. Other remote reads/writes still need explicit approval plus `KAGGLE_REMOTE_CONFIRMED=1` for reads/status/output and `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1` for pushes with real data attachments. Before any future runtime-selection push or approval request, run `./scripts/kaggle_kernel.sh preflight-runtime-selection`. Next workflow step is selected-runtime debug/resume/tiny-overfit, not a long real training launch. |
| `0004-sipaim-paper-scaffold.md` | draft active / scaffold slice implemented | The paper can be outlined and grounded in the thesis/repo evidence, but selected runtime, full VAE runs, continuous `SO(2)` results, downstream WSI classifier results, and sealed masked-WSI test evidence are still pending. | Keep the SIPAIM paper compile-safe, use thesis figures only as paper-local copied assets, and leave result claims as explicit placeholders until evidence exists. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |

Latest real-data pretest update, 2026-06-20: remote v8 completed and was
downloaded to `runs/kaggle/real_data_runtime_pretest_v8`. It is still
non-promotable and wrote no `benchmark/selected_runtime.json`. The v7
`quantile() input tensor is too large` evidence-plumbing failure is fixed:
paired-numerical and corruption failed-candidate evidence counts are both zero.
Six eager single-visible-T4 FP32 rows are capped-pretest-passing: bs4/bs8/bs12
crossed with `branchless_all` and `indexed_masked`. Eager bs32 remains
`runtime_OutOfMemoryError`, dual-T4 train-step measurement remains pending, and
compiled rows remain diagnostic/ineligible until full compile-settle evidence
exists. The next slice is recorded in
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` as
`v8_shortlist_eager_amp_then_dual_gate`; v8 artifacts are shortlist-only and
must not be promoted into `benchmark/selected_runtime.json`. The slice requires
real dual-T4 train-step timing before selection; missing, failed, or skipped
dual timing keeps selected-runtime writing blocked.

Selected-runtime plumbing update, 2026-06-20: the separate
`v8_shortlist_eager_amp_then_dual_gate` writer now lives in
`src/eqvae/benchmarking/runtime_selection.py` with CLI
`src/eqvae/cli/runtime_selection_benchmark.py` and focused tests in
`tests/test_runtime_selection_benchmark.py`. It records v8 hashes as
shortlist-only provenance, writes this benchmark's own proof/matrix/linked
safety/model-count artifacts, rewrites compiled pass rows to diagnostic
`ineligible`, and refuses `benchmark/selected_runtime.json` unless dual-T4
train-step timing plus linked proof pass. The linked proof now explicitly
requires train and validation dataloader rank coverage, candidate-bound
gate-health row ids, child-process launch proof, scoped numerical/corruption
rows, and a hash-linked `benchmark/stain_corruptor_qa.json`. The adversarial
follow-up hardening now requires 25 measured dataloader batches plus
wait/throughput thresholds, three numerical batch indices, train and validation
corruption-check rows, strict stain-QA candidate coverage, and exact embedded
v8 payload membership. The default local run is intentionally blocked because
it has no real dual-T4 evidence.
The Kaggle executor and kernel wrapper are now implemented in
`src/eqvae/benchmarking/runtime_selection_executor.py`,
`src/eqvae/cli/runtime_selection_executor.py`, and
`kaggle/kernels/runtime_selection`; the generated single-file kernel embeds
only the required v8 provenance artifacts and is guarded by
`runtime_selection_kernel_ready`.
Runtime-selection v1 was downloaded to `runs/kaggle/runtime_selection_v1`; it
proved the dual-T4 timing gate but correctly refused selected-runtime writing
because linked single-visible proof rows failed. The local v2 patch fixes the
observed false negatives by accepting `model_inventory.csv` in the wrapper,
normalizing `local_pass` gate-health rows before eligibility is computed, and
requiring the clean-validation RNG flag only for validation corruption rows.
Runtime-selection v2 completed and downloaded to
`runs/kaggle/runtime_selection_v2`; it fixed those v1 blockers but still
refused selected-runtime writing because the executor emitted gate-health rows
for branchless single-visible candidates only. The local v3 patch binds those
single-visible gate-health rows to same-shape indexed candidates after the
indexed runtime rows have passed linked evidence.
Runtime-selection v3 completed and downloaded to
`runs/kaggle/runtime_selection_v3`; it wrote
`benchmark/selected_runtime.json` selecting
`dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`.
That selected row is the safety baseline for debug/full-run planning. The local
efficiency-follow-up contract now adds first-class `runtime_policy_id` binding,
stable compile eligibility, AMP skip blockers, and policy-bound linked proofs.
Runtime-selection v4 ran that follow-up and failed closed: it found an otherwise
clean AMP conservative row at `samples_sec = 25.220604` with zero AMP skips and
an estimated 10-epoch wall time around 33.0 hours, but no selected runtime was
written because small selected-row numerical drift and nonselected-row proof
failures were treated as global blockers. Local commit `fc5227d` repairs that
writer policy, and replaying the v4 artifacts through the patch produced proof
`pass` for the intended AMP row. Runtime-selection v5 completed and selected
that AMP conservative row at `27.381321` samples/sec with zero AMP skips and
strict local replay pass. It replaces v3 as the efficient selected-runtime
artifact, but cannot unlock full training until a compact broader AMP check and
selected-runtime debug, checkpoint/resume, and tiny-overfit proofs pass.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
