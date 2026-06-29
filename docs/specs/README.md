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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; narrow local slices through `corruption_ready` are implemented; Kaggle runtime-selection v5 selected AMP conservative dual-T4 bs12 indexed-mask at `27.381321` samples/sec; compact relaxed scalar-gate AMP v6 completed fail-closed and kept v5; local selected-runtime debug/checkpoint-resume/artifact/tiny-overfit proof plumbing now exists as synthetic, fail-closed contract evidence; Spec 0006 local selected-runtime mechanics are implemented and locally verified; Spec 0007 real selected-runtime runner is implemented locally with synthetic dry-run and `ubc-pre-shuffled` support while remaining non-promotable; Spec 0008 local `remote_generate` readiness is implemented and locally verified; selected-runtime debug/tiny kernel contract is `selected_runtime_debug_gate_contract_ready`; the gate now rejects fabricated fixed-32 selector JSON and schema-valid synthetic shard replay as non-canonical local evidence unless Spec 0008 runs in `remote_generate` mode and downloaded remote artifacts prove the canonical selector; the successful FSQ notebook is recorded as runtime-reference material only | Broad implementation remains blocked by fixed real 25-patch visual QA, remote selected-runtime debug, remote checkpoint/resume proof, remote tiny-overfit proof, real train/resume/evaluator/artifact proof, future `SO(2)` count ceiling, real fixed selector generation from real Kaggle shards, final adversarial spec review, and the sealed masked-WSI test shard for paper claims. Runtime-selection v5 supersedes v3 for efficiency but is not full-training-launch-ready. The selected-runtime debug gate remains non-promotable and remote-pass-blocked until downloaded remote UBC artifacts prove selected-runtime plan application/resume/tiny bounds, canonical selector generation, and exact real-dataset metadata. FSQ quantization, PixelShuffle/sub-pixel upsampling, final `tanh`, exact old corruptor behavior, and rot90/discrete-latent artifacts remain quarantined from the continuous `SO(2)` path. | Broad Spec 0001 stays draft active. Next action is an explicit-user-approved narrow Spec 0008 selected-runtime debug/tiny push, not a long run. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; synthetic setup-smoke remote v1 passed; synthetic timing remote v4 passed; real-data runtime pretest remote v4/v5/v6/v7/v8 artifacts are downloaded; selected-runtime kernel path is guarded as `runtime_selection_kernel_ready`; `./scripts/kaggle_kernel.sh preflight-runtime-selection` is the mandatory local semantic preflight before any future runtime-selection push; selected-runtime runner local preflight is `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`; selected-runtime debug/tiny kernel contract is `selected_runtime_debug_gate_contract_ready` with `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`; selected-runtime debug push also has structured `eqvae.cli.selected_runtime_gate --verify-push-ready --selector-generation-mode remote_generate` and fixed32 readiness checks; local `eqvae.cli.train` synthetic proof plumbing, Spec 0006 mechanics, Spec 0007 real-runner code, and Spec 0008 local fixed32 readiness exist for debug/resume/artifact/tiny-overfit schemas | Runtime-selection v5 downloaded to `runs/kaggle/runtime_selection_v5`, wrote selected runtime, and passed strict local replay under current `main`. Runtime-selection v6 downloaded to `runs/kaggle/runtime_selection_v6`, replayed locally, wrote no selected runtime, and kept v5 because relaxed AMP was slower. The selected-runtime debug/tiny launch surface is fail-closed and non-promotable; local `remote_generate` readiness is implemented, but canonical real fixed-32 selector generation and remote debug/tiny proof are still not passed until an approved Kaggle run is downloaded and verified. | Other remote reads/writes still need explicit approval plus `KAGGLE_REMOTE_CONFIRMED=1` for reads/status/output and `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1` for pushes with real data attachments. Before any future runtime-selection push or approval request, run `./scripts/kaggle_kernel.sh preflight-runtime-selection`; before any selected-runtime debug/tiny push or approval request, run `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` and `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`. Remote debug/tiny push may be requested only after the shell push guard passes, exact real-dataset metadata is attached, and the user explicitly approves. The approved remote kernel then generates and validates the canonical fixed-32 selector before training; downloaded outputs must pass `eqvae.cli.selected_runtime_gate --verify-output`; it is still not approval for a long real training launch. |
| `0006-selected-runtime-local-mechanics.md` | implemented / locally verified; adversarial fixes added for strict linked-runtime-proof validation, checkpoint progress consistency, observed FP32/AMP-off local telemetry, and structured-readiness-derived push blocking | Real Kaggle proof, canonical real fixed-32 selector generation, and remote selected-runtime debug/tiny push remain blocked until Spec 0008 readiness passes in `remote_generate` mode, exact real-dataset metadata is attached, and the user explicitly approves a later remote action. | Maintain local non-promotable guarantees. Next work is outside Spec 0006/0007: canonical fixed-32 selector boundary, then local preflights and explicit user approval before any remote debug/tiny action. |
| `0007-real-ubc-ddp-amp-selected-runtime-runner.md` | implemented / locally verified | Remote proof is intentionally sequenced through Spec 0008. No Kaggle push or long full run belongs to this spec. | Maintain the local runner and preflight; proceed to Spec 0008 for canonical fixed-32 selector plus narrow remote debug/tiny readiness. |
| `0008-canonical-fixed32-and-remote-debug-tiny-readiness.md` | remote debug/tiny readiness proved by selected-runtime debug/tiny v5; local readiness implemented / locally verified | v4 was approved, pushed, reached `KernelWorkerStatus.ERROR`, and downloaded to `runs/kaggle/selected_runtime_debug_v4`. It proved canonical selector generation plus debug/resume/plan/gate/manifest artifact writing, and it proved the fixed32 tiny full-batch sampler (`48/24` effective samples, observed bs12 only). It still failed tiny-overfit on one early full-batch AMP overflow per rank at tiny `optimizer_step_index = 3` (`batch_size = 12`, `grad_norm = inf`, `nonfinite_count = 125`, `amp_step_skipped = 1`). Local follow-up set an explicit conservative GradScaler init scale (`16384.0`), records it in training/tiny/plan-applied artifacts, and hardens wrapper/verifier checks to require scaler evidence plus direct nested tiny metric-row proof for both ranks over steps `1..128`. The first approved v5 push attempt then failed before upload because Kaggle CLI 2.2.1 reused a stale cached OAuth token; `scripts/kaggle_oauth_exec.py` now routes authenticated Kaggle calls through a fresh temporary token-file path, and selected-runtime `api-check` passes with only the known quota warning after removing the raw token probe. After fresh approval, Kaggle accepted selected-runtime debug/tiny version 5; it completed, downloaded to `runs/kaggle/selected_runtime_debug_v5`, and passed strict `--verify-output` with canonical real fixed-32 selector generation, no launch blockers, zero tiny AMP skips, zero tiny nonfinite rows, and 256 nested tiny metric rows over 128 optimizer steps. | Next candidate action is the first full real selected-runtime run, with fresh explicit approval. This is not pre-approved by Spec 0008. |
| `0004-sipaim-paper-scaffold.md` | draft active / scaffold slice implemented | The paper can be outlined and grounded in the thesis/repo evidence, but selected runtime, full VAE runs, continuous `SO(2)` results, downstream WSI classifier results, and sealed masked-WSI test evidence are still pending. | Keep the SIPAIM paper compile-safe, use thesis figures only as paper-local copied assets, and leave result claims as explicit placeholders until evidence exists. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |

Guard authorization phrases retained for local push-guard scripts:

- synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`;
- capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`;
- selected-runtime debug/tiny gate contract is `selected_runtime_debug_gate_contract_ready`.

Latest real-data pretest update, 2026-06-20: remote v8 completed and was
downloaded to `runs/kaggle/real_data_runtime_pretest_v8`. It is still
non-promotable and wrote no `benchmark/selected_runtime.json`. The v7
`quantile() input tensor is too large` evidence-plumbing failure is fixed:
paired-numerical and corruption failed-candidate evidence counts are both zero.
Six eager single-visible-T4 FP32 rows are capped-pretest-passing: bs4/bs8/bs12
crossed with `branchless_all` and `indexed_masked`. Eager bs32 remains
`runtime_OutOfMemoryError`, dual-T4 train-step measurement remained pending in
that capped pretest, and compiled rows remain diagnostic/ineligible until full
compile-settle evidence exists. The later selected-runtime slice was recorded in
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
strict local replay pass. Runtime-selection v6 tested compact relaxed scalar
gate AMP against that v5 fallback, downloaded to
`runs/kaggle/runtime_selection_v6`, replayed locally, and kept v5 because the
relaxed row reached only `25.288828` samples/sec and wrote no selected runtime.
v5 replaces v3 as the efficient selected-runtime artifact, but cannot unlock
full training until selected-runtime debug, checkpoint/resume, and tiny-overfit
proofs pass. The local contract runner for those proof artifacts now exists as
`python -m eqvae.cli.train`; it currently supports only synthetic local data,
writes `full_run_eligible = false`, and leaves the real UBC/Kaggle proof
pending. The dedicated selected-runtime debug/tiny kernel contract now exists at
`kaggle/kernels/selected_runtime_debug` with CLI
`python -m eqvae.cli.selected_runtime_gate`; it embeds v5, writes exact
fail-closed local artifacts, and is covered by
`./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`. It is still not
remote-pass-ready because the Spec 0008 debug wrapper, canonical real fixed-32
selector, and downloaded remote debug/tiny proofs are missing.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
