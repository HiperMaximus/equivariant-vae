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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; narrow local slices through `corruption_ready` are implemented; runtime-selection v5 selected AMP conservative dual-T4 bs12 indexed-mask at `27.381321` samples/sec; compact relaxed scalar-gate AMP v6 completed fail-closed and kept v5; Spec 0006 local selected-runtime mechanics, Spec 0007 real selected-runtime runner, and Spec 0009 full-run local workflow are locally implemented; Spec 0008 remote debug/tiny v5 passed strict output verification; the first Spec 0009 full-kernel push was accepted as version 1, later canceled by Kaggle, and downloaded with checkpoints through update 43750; the successful FSQ notebook is recorded as runtime-reference material only | Broad implementation remains blocked by a completed and strict-verified full selected-runtime run proof, fixed real 25-patch visual QA, real evaluator/artifact proof, future `SO(2)` count ceiling, final adversarial spec review, and the sealed masked-WSI test shard for paper claims. The immutable v5 selected-runtime artifact still says `full_training_launch_ready = false`, but the external Spec 0008 proof now satisfies the pre-long-run debug/tiny gate. FSQ quantization, PixelShuffle/sub-pixel upsampling, final `tanh`, exact old corruptor behavior, and rot90/discrete-latent artifacts remain quarantined from the continuous `SO(2)` path; however, the useful FSQ fixed-25 rotated-input/transformed-latent evaluation pattern is reused as a repo-owned rot90 `{0,90,180,270}` replacement, now IMPLEMENTED as Spec 0010 (uncommitted; real selector still needed) for paper-promotable qualitative evidence. | Broad Spec 0001 stays draft active. Spec 0010 (`fixed25-equivariance-artifact-protocol`) is now IMPLEMENTED (uncommitted, gate green) and carries the fixed-25 rotated/latent artifact protocol; its remaining blocker is the real fixed-25 selector, and local resume-policy work also remains before any resume upload/push. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; synthetic setup-smoke remote v1 passed; synthetic timing remote v4 passed; real-data runtime pretest remote v8 artifacts are downloaded; runtime-selection v5 selected AMP conservative dual-T4 bs12 indexed-mask and runtime-selection v6 kept v5; selected-runtime debug/tiny v5 passed strict output verification; selected-runtime runner local preflight is `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`; selected-runtime debug/tiny kernel remains non-promotable; Spec 0009 adds dedicated full-run preflight/status/output actions and `kaggle/kernels/selected_runtime_full`; full-kernel version 1 was accepted by Kaggle on 2026-06-29 and later returned `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` | Canceled v1 downloaded to `runs/kaggle/selected_runtime_full_v1` with checkpoints through update 43750 but no metrics/benchmark summaries; strict full verification fails as expected. Remote reads/writes remain blocked by explicit approval and confirmation variables. | Do local resume-policy work before any new remote action: support canceled-output checkpoint continuation safely, then ask exact approval for any checkpoint upload/source attachment/resume push. |
| `0006-selected-runtime-local-mechanics.md` | implemented / locally verified; adversarial fixes added for strict linked-runtime-proof validation, checkpoint progress consistency, observed FP32/AMP-off local telemetry, and structured-readiness-derived push blocking | None for the local mechanics slice. Full-run launch remains outside this spec. | Maintain local non-promotable guarantees while Spec 0009 adds the full-run workflow. |
| `0007-real-ubc-ddp-amp-selected-runtime-runner.md` | implemented / locally verified; extended by Spec 0009 full-run mode | No remaining local runner blocker for the Spec 0009 full-run launch surface. Remote full-run execution remains outside this spec and still requires explicit approval. | Maintain runner/debug non-promotable guarantees while Spec 0009 owns the first full-run guard and verifier. |
| `0008-canonical-fixed32-and-remote-debug-tiny-readiness.md` | remote debug/tiny readiness proved by selected-runtime debug/tiny v5; local readiness implemented / locally verified | v4 was approved, pushed, reached `KernelWorkerStatus.ERROR`, and downloaded to `runs/kaggle/selected_runtime_debug_v4`. It proved canonical selector generation plus debug/resume/plan/gate/manifest artifact writing, and it proved the fixed32 tiny full-batch sampler (`48/24` effective samples, observed bs12 only). It still failed tiny-overfit on one early full-batch AMP overflow per rank at tiny `optimizer_step_index = 3` (`batch_size = 12`, `grad_norm = inf`, `nonfinite_count = 125`, `amp_step_skipped = 1`). Local follow-up set an explicit conservative GradScaler init scale (`16384.0`), records it in training/tiny/plan-applied artifacts, and hardens wrapper/verifier checks to require scaler evidence plus direct nested tiny metric-row proof for both ranks over steps `1..128`. The first approved v5 push attempt then failed before upload because Kaggle CLI 2.2.1 reused a stale cached OAuth token; `scripts/kaggle_oauth_exec.py` now routes authenticated Kaggle calls through a fresh temporary token-file path, and selected-runtime `api-check` passes with only the known quota warning after removing the raw token probe. After fresh approval, Kaggle accepted selected-runtime debug/tiny version 5; it completed, downloaded to `runs/kaggle/selected_runtime_debug_v5`, and passed strict `--verify-output` with canonical real fixed-32 selector generation, no launch blockers, zero tiny AMP skips, zero tiny nonfinite rows, and 256 nested tiny metric rows over 128 optimizer steps. | Next candidate action is the first full real selected-runtime run, with fresh explicit approval. This is not pre-approved by Spec 0008. |
| `0009-first-full-selected-runtime-training-run.md` | version 1 canceled after checkpoint `step_043750.pt`; future two-phase interval flushing and boundary log/barrier breadcrumbs implemented locally | Kaggle accepted `maximusshtefan/eqvae-selected-runtime-full` version 1, then returned `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`. Downloaded outputs include resumable checkpoints but no `metrics/` or `benchmark/`; v1's first 43750 metric rows are unrecoverable, so a naive resume push would still be incomplete. Future launches now atomic-flush metrics/partial summaries before interval checkpoint exposure, refresh them with checkpoint hashes after save, print half-epoch boundary breadcrumbs, and wait at a final full-run boundary barrier. The current full-run artifact surface still lacks `fixed25_equivariance_artifact_protocol` and is training/checkpoint evidence only until that protocol is implemented. | FU-039 DECIDED (2026-07-02): the first paper-promotable run restarts from scratch (discard v1 as a resume base) for a complete training curve; remaining FU-039 work is verifying the full config/kernel launch fresh. `fixed25_equivariance_artifact_protocol` is IMPLEMENTED as Spec 0010 and committed; the DDP-correctness fixes (FU-007/008/012/020) are committed. The remaining gating blocker before a paper-promotable full launch is FU-041 (generate the real fixed-25 selector, needs the real UBC validation shard); any remote action still needs exact approval. |
| `0004-sipaim-paper-scaffold.md` | draft active / scaffold slice implemented | The paper can be outlined and grounded in the thesis/repo evidence, but selected runtime, full VAE runs, continuous `SO(2)` results, downstream WSI classifier results, and sealed masked-WSI test evidence are still pending. | Keep the SIPAIM paper compile-safe, use thesis figures only as paper-local copied assets, and leave result claims as explicit placeholders until evidence exists. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |
| `0010-fixed25-equivariance-artifact-protocol.md` | IMPLEMENTED and committed (Spec 0010 `8457233`, FU-040); the real fixed-25 selector is generated, promoted, and committed (FU-041 `36797c5`; frozen originals in `docs/data/fixed25/`); 6-lens post-impl adversarial review integrated; see decision 0009 | Real fixed-25 selector (needs real Kaggle validation `.bin/.csv`) — a real full run FAILS CLOSED until it exists (couples to the undecided FU-039 v1-continuation); and committing the work. Settled: rot90 `{0,90,180,270}`; full-save-every-boundary; unmasked/full-frame error maps; PCA = EQ-VAE `pca_to_rgb` + first3; retired `reconstruction_samples.pt` from the full run; scale/LPIPS out of scope; shared `model_base` eval-config; standalone in-run + offline evaluator. | Implemented (decision 0009): `src/eqvae/artifacts/fixed25_equivariance.py` + `src/eqvae/cli/fixed25_equivariance.py`; runner boundary-loop evaluator + FU-039-durable `metrics/equivariance_25.csv` (merge-not-gather, resume-prefix validator, broadcast-guarded rank-0 eval); gate requires fixed-25 (incl. error maps + `promotable=real`) and retires the placeholder; shared `fixed25_equivariance` config block; CPU tests. Same frozen 25 validation images for BOTH models. Extends the Spec 0009 runner/verifier. Remaining: none for the protocol itself; the paper-promotable full run that exercises it is Spec 0011 S19 (Kaggle). |

| `0011-reusable-goal-derived-runtime-and-compiled-fastpath.md` | draft active; **Phase 1 (S1–S10) + Phase 2 (S11–S13) + Phase 3 (S15 `357ada6`/S16 `3298a57`) DONE (local)**; S14 decomposed — **S14a DONE (`1dc3901`, local); S14b dual-T4 executor branch DONE (`7256cd3`, local); S14b single-GPU pretest surface DONE (`aabd886`, local)**; S14c local-authorable next; Kaggle S17/S19 gated | Promote the compiled fast-path + probe-derived bigger batch into the first paper-promotable full run, built as a REUSABLE per-(model×hardware) SEARCH-then-run mechanism (the search emits `selected_runtime.json`; config carries no batch/LR; validators check relationships, not literals) so the future equivariant model re-runs it unchanged. Supersedes Spec 0009's frozen schedule. Design = 17 steps (4 phases) + adversarial critique (5 must-fixes) folded in; drop_last=True kept → `floor(P/G)` schedule; provenance = folded honest generator. See the per-step `(DONE — …)` tags in the spec body for the state of record. | **Phase 3 COMPLETE.** S16 (`3298a57`, local): compiled whole-step (`torch.compile(step, dynamic=False)`) + train-only inline corruption + `drop_last=True` flip via the `_safe_drop_last` empty-shard guard + wired the S15-deferred dynamo config + extracted `compiled_autograd_context`; behavior-preserving on the eager v5 plan (parser acceptance + observation mirror deferred to S17). **S14a DONE (`1dc3901`, local):** the seven measured compiled-recipe knobs are threaded config→`_RuntimePolicy`→`RowSpec`→CSV via ONE shared `_recipe_knob_columns` producer helper (both producers), so a compiled winner carries its MEASURED recipe into `_selected_runtime_payload`; `_encode_ddp_config` reuses `_row_spec_payload`; behavior-preserving (guards still fail-close `'step'`, no execution change); gate 446, adversarial review clean (5 seams mutation-proven). **S14b dual-T4 executor branch DONE (`7256cd3`, local):** opened the executor's two `'step'` guards + added the compiled-whole-step branch to `_run_ddp_rank_row` (`_build_compiled_ddp_step`/`_run_compiled_ddp_step_batch` mirror the runner S16 code; settle drives the compiled step to warm before the counter reset, the numerical-proof loop stays byte-identical eager, warmup/measured route through the compiled step); extracted `_build_eager_ddp_optimizer` (eager path unchanged); promoted `model_requires_buffer_broadcast` to shared `fastpath_recipe`; two fail-closed guards (`amp_off_fp32` only, `static_graph=False` only) keep the measured recipe faithful; gate 452, review 0-confirmed. **S14b single-GPU pretest surface DONE (`aabd886`, local):** added `compile_scope=='step'` to `real_data_runtime_pretest` (single-GPU `_build_compiled_step`/`_run_compiled_step_batch`, no DDP), mirroring the dual-T4 branch; widened the `_run_stage1_rows` guard + the secondary evidence surfaces (`_unique_train_step_target_rows`, `_compile_evidence_pass_for_row` step==model_forward parity, `implemented_compile_scopes`, `_compile_settle_proof`); `_run_child_row` grows a `run_one_step` dispatch (settle drives the compiled step; eager path byte-identical); step stays ineligible exactly like model_forward (fail-closed ceiling preserved); one guard (`amp_off_fp32` only — `static_graph` N/A single-GPU); gate 459, 4-lens review (1 LOW + 1 test-gap fixed, mutation-proven); compiled EXECUTION stays a Kaggle observation. **NEXT (local, new window per step):** S14c = fold the probe VRAM feasibility sweep + the grid `efficiency_followup` step-policy wiring. Then **Kaggle-only (user-driven):** S17 (run generator → new `selected_runtime.json` + de-pin the plan validators + activate the observation mirror), S19 (~30h full run); plus the queued LR-finder. Each step = one gated commit + adversarial review + explicit approval, in a NEW window per step; never push to origin. |

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
v5 replaces v3 as the efficient selected-runtime artifact. Its immutable
`benchmark/selected_runtime.json` still records
`full_training_launch_ready = false`, but Spec 0008 later proved the downstream
selected-runtime debug/checkpoint-resume/gate/tiny lane on Kaggle and strict
downloaded-output verification passed for
`runs/kaggle/selected_runtime_debug_v5`. The dedicated selected-runtime
debug/tiny kernel contract exists at `kaggle/kernels/selected_runtime_debug`
with CLI `python -m eqvae.cli.selected_runtime_gate`; it embeds v5, writes
bounded non-promotable proof artifacts, and is covered by
`./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`. The first full
selected-runtime training run now has a dedicated Spec 0009 local kernel, guard,
runner schedule, verifier, and approval gate, but the remote run itself remains
blocked until fresh explicit user approval.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
