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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; narrow local slices through `corruption_ready` are implemented; kaggle smoke is `kaggle_smoke_ready`; synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`; capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`; synthetic timing remote v4 is complete; capped real-data runtime pretest v4/v5/v6/v7/v8 artifacts are downloaded and non-promotable; selected-runtime writer and Kaggle executor/kernel plumbing for `v8_shortlist_eager_amp_then_dual_gate` are implemented, committed as `fba9d98`, pushed to Kaggle version 1, and fail-closed | Broad implementation remains blocked by fixed real 25-patch visual QA, selected-runtime debug, real train/resume/evaluator/artifact writers, future `SO(2)` count ceiling, selected-runtime benchmark proof on real data, real fixed selector generation from real Kaggle shards, final adversarial spec review, and the sealed masked-WSI test shard for paper claims. Remote v8 fixed the v7 quantile evidence-plumbing failure, wrote no `selected_runtime.json`, and produced six capped-pretest passing eager single-visible-T4 FP32 rows: bs4/bs8/bs12 crossed with `branchless_all` and `indexed_masked`. Eager bs32 remains `runtime_OutOfMemoryError`, dual-T4 train-step measurement remains pending in the v8 pretest, compiled rows remain diagnostic/ineligible, and the selected-runtime writer refuses `selected_runtime.json` without real dual-T4 timing plus linked proof. Runtime-selection Kaggle version 1 still reported `KernelWorkerStatus.RUNNING` as of `2026-06-20T17:31:27-05:00`. | Poll `status-runtime-selection` with `KAGGLE_REMOTE_CONFIRMED=1` until complete, then download `output-runtime-selection runs/kaggle/runtime_selection_v1` and inspect the strict selected-runtime artifacts. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; synthetic setup-smoke remote v1 passed; synthetic timing remote v4 passed; real-data runtime pretest remote v4/v5/v6/v7/v8 artifacts are downloaded; selected-runtime kernel path is guarded as `runtime_selection_kernel_ready` | Remote v8 completed and downloaded to `runs/kaggle/real_data_runtime_pretest_v8`; it kept the capped pretest non-promotable, fixed the quantile failure, produced zero failed candidate evidence entries, and still did not select a runtime. Full training/full-run launchers are still not implemented. | User approved the selected-runtime Kaggle push/status/output on 2026-06-20 with the normal guard variables. Other remote reads/writes still need explicit approval plus `KAGGLE_REMOTE_CONFIRMED=1` for reads/status/output and `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1` for pushes with real data attachments. |
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

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
