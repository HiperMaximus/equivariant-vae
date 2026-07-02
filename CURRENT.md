# Current Repository Status

Last updated: 2026-07-02

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE using a repo-owned,
   compile-compatible implementation, with `escnn` as a reference.

Current short state: runtime-selection v5 is the selected fallback runtime
(`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`,
`27.381321` samples/sec, about 30.4 hours for 10 epochs). Runtime-selection v6
tested relaxed scalar-gate AMP, was slower (`25.288828` samples/sec), and kept
v5 fail-closed. Local synthetic selected-runtime debug/checkpoint-resume/
artifact/tiny-overfit proof plumbing exists, and Spec 0006 local mechanics are
implemented and locally verified. The train and gate paths now share the v5
`SelectedRuntimePlan` parser, including strict linked `runtime_proof.json`
status/write-decision/rank/return-code validation and tokenized
`torchrun --standalone --nproc_per_node=2` validation; the train path writes a
full plan-applied proof that fails locally for unexecuted dual-T4 CUDA AMP/DDP
fields, plus local UBC-format mechanics, AMP/progress, checkpoint schema v5,
and structured local readiness artifacts. Synthetic local selected-runtime runs
exercise `PatchTrainingDataset`, collation, normalization, `indexed_masked`
train corruption, clean validation RNG isolation, integrated simulated AMP-skip
progress semantics, strict pre-restore checkpoint/progress rejection, and
observed local FP32/AMP-off row telemetry in `metrics/train_steps.csv`. All
local artifacts remain non-promotable with `full_run_eligible = false`. The
Spec 0007 selected-runtime real runner now exists as
`eqvae.cli.selected_runtime_train`: it supports local dry-run/synthetic proof
and the real `ubc-pre-shuffled` data surface, consumes the shared v5
`SelectedRuntimePlan`, applies the selected batch/corruption/dataloader/
zero-grad policy, implements the AMP/GradScaler train-step path with FP32
objective islands, writes schema-v5 checkpoint/resume proof, emits
`metrics/train_steps.csv` plus selected-runtime gate-health rows, and records
tokenized `torchrun --standalone --nproc_per_node=2` launch plus DDP
rank/device proof artifacts. Local dry-runs still fail the full dual-T4/AMP
plan-applied proof and keep `remote_pass_ready = false`, as intended. The
runner honors the selected DDP static-graph/bucket-view flags exactly,
gathers per-rank metric/gate evidence, writes artifacts only on rank 0,
records selected-runtime AMP/CUDA/DDP checkpoint state when active, and blocks
readiness if an AMP skip is observed. The new local-only preflight is
`./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`. The
push-readiness CLI now also consumes the structured readiness artifact instead
of relying on config booleans alone. The selected-runtime
debug/resume/artifact/tiny-overfit Kaggle gate contract exists as
`selected_runtime_debug_gate_contract_ready`; it is locally preflightable and
fail-closed. Spec 0008 local `remote_generate` readiness is now implemented:
the fixed-32 selector generator/readiness preflight proves synthetic selector
determinism while rejecting synthetic selectors as canonical-real evidence, the
selected-runtime debug wrapper generates and validates the canonical selector
before bounded debug/tiny training, passes the generated selector into
`eqvae.cli.selected_runtime_train`, and exposes a strict
`eqvae.cli.selected_runtime_gate --verify-output` post-download verifier. The
debug/tiny v5 run completed, was downloaded to
`runs/kaggle/selected_runtime_debug_v5`, and passed strict output verification:
canonical real selector generation, selected-runtime plan application,
checkpoint/resume, gate health, manifest, and tiny-overfit bounds all have no
remaining launch blockers.
The first approved v5 push attempt on 2026-06-29 exposed stale Kaggle OAuth
cache behavior in local CLI calls: `kaggle auth print-access-token` could mint
a token, but normal `kaggle kernels ...` calls reused a rejected cached token.
The repo wrapper now runs authenticated Kaggle calls through
`scripts/kaggle_oauth_exec.py`, which generates a fresh OAuth token and passes
it to the child CLI via a temporary 0600 token file; selected-runtime
`api-check` no longer uses the raw `kaggle auth print-access-token` probe and
now proves auth through wrapped endpoint calls. A fresh live selected-runtime
read-only preflight passes except for the already-known quota endpoint warning.
This is workflow plumbing only; selected-runtime debug/tiny v5 has now
completed, downloaded, and passed the strict `--verify-output` gate. The debug
kernel remains non-promotable by design (`full_run_eligible = false`), but Spec
0008's remote debug/resume/artifact/gate/tiny readiness proof has no remaining
launch blockers.
The first long selected-runtime training kernel push was explicitly approved
and launched on Kaggle as version 1; it later returned
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` and was downloaded locally to
`runs/kaggle/selected_runtime_full_v1`. The download contains resumable
checkpoints through update 43750 but no metrics/benchmark summaries, so it is
not a completed or verified full run.
Spec 0009 is implemented locally: `configs/spec0001/non_eq_vae_selected_runtime_full.json`
defines the 10-epoch/125000-update selected-runtime full config; the runner now
has a `kaggle_selected_runtime_full_train` path with stochastic seeded train VAE
reparameterization, half-epoch validation metrics, validation-best/final/latest
four interval checkpoint policy, stricter full-run schedule validation, and
long-run resume proof fields for GradScaler/CUDA RNG/sampler/progress identity;
`kaggle/kernels/selected_runtime_full` is the dedicated full launcher; and
`scripts/kaggle_kernel.sh` has guarded `preflight/status/output` actions plus a
full-run push guard. The debug/tiny kernel remains non-promotable and must not
be reused for the long run. After the v1 cancellation, the local runner was
hardened to use a two-phase full-run interval artifact flush: metrics,
validation, gate-health, and fail-closed partial summaries are fsync/atomic
written before exposing the new interval checkpoint; after checkpoint/best-model
save, the same artifacts are refreshed with checkpoint hashes. DDP ranks now
broadcast rank-0 flush failures so all ranks fail together instead of hanging.

Full-run Kaggle launch requirements for the next agent: use only the dedicated
full kernel path, not the debug/tiny kernel. The correct push action is the
generic guarded push with the full kernel directory:
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_full`.
There is no `push-selected-runtime-full` action. The real push guard requires a
clean Git worktree with committed source provenance; if the ignored generated
`kaggle/kernels/selected_runtime_full/run.py` was built while the repo was
dirty, first commit the source/template/config/docs, then rebuild it locally
with `./scripts/kaggle_kernel.sh build kaggle/kernels/selected_runtime_full`,
then rerun the guarded push. Remote status and output download are separate
remote reads and still require explicit approval plus `KAGGLE_REMOTE_CONFIRMED=1`:
`./scripts/kaggle_kernel.sh status-selected-runtime-full` and
`./scripts/kaggle_kernel.sh output-selected-runtime-full runs/kaggle/selected_runtime_full_v1`.
After a push, do not wait in-turn on the long run; if an approved status check
shows `RUNNING`, record the local time and suggested next polling time here.

Important artifact-scope correction, 2026-06-30: local review against
`GOAL.md`, `docs/repo_goal_and_requirements.md`, `docs/issue_image_inventory.md`,
Spec 0001, and the historical FSQ notebook confirms that Spec 0009's current
full-run runner does not yet save the advisor/issue-required fixed-25
equivariance artifact protocol. It writes durable train/validation/gate CSVs,
summaries, checkpoints, and a minimal deterministic reconstruction tensor, but
not fixed-25 original/reconstruction progress, rotated-input versus
transformed-latent reconstructions, latent/embedding arrays or PCA maps,
full-frame error maps, or an `equivariance_error_25_patches` metric. Treat
any future Spec 0009 verifier pass as training/checkpoint evidence only until a
focused fixed-25 rotated/latent artifact spec is implemented or explicitly
deferred. The runner now at least prints half-epoch full-run boundary
breadcrumbs and waits at a final full-run boundary barrier after the
flush/checkpoint refresh, mirroring the useful FSQ logging/synchronization
pattern for pulled Kaggle logs. `fixed25_equivariance_artifact_protocol` is now IMPLEMENTED as Spec 0010
(`docs/specs/0010-fixed25-equivariance-artifact-protocol.md`) on the working tree
(uncommitted, 2026-07-01): new `src/eqvae/artifacts/fixed25_equivariance.py` +
`src/eqvae/cli/fixed25_equivariance.py`, runner boundary-loop evaluator with
FU-039-durable `metrics/equivariance_25.csv` (merge-not-gather for the global
rank-0 rows, resume-prefix validator, broadcast-guarded rank-0 eval), gate + config
+ CPU tests; gate green (263 passed, 0 type errors) with a 6-lens post-impl
adversarial review integrated (9 confirmed findings). Remaining OPEN before
paper-promotable use: generate the REAL fixed-25 selector (the tracked config is the
placeholder, so a real run FAILS CLOSED until then) and commit the work. It is an
EVALUATION/INSPECTION protocol
(decision `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`),
decoupled from training: it probes the embedding space with exact rot90 as a
PROXY for embedding smoothness/structure (EQ-VAE idea), to COMPARE the
non-equivariant baseline vs the future `SO(2)`-steerable model on the SAME frozen
25 validation images. `SO(2)` steerability is a property of the equivariant
model's convolutions, not of this evaluation. Required scope: canonical fixed-25 selector
guard, fixed originals, per-boundary reconstruction progress, exact rot90
`{0,90,180,270}` rotated-input reconstructions, transformed-latent
reconstructions from deterministic posterior `mu`, full-frame error maps,
latent/embedding arrays, EQ-VAE-style PCA/latent maps, `n=25` equivariance
metrics (headline = EQ-VAE Appendix C.3 / FSQ normalized-latent-L2 ratio),
manifest metadata for rotation/angle policy (no mask; full-frame error maps), atomic writes, boundary
logs/barriers, verifier checks, and focused tests. Rotation model corrected
2026-07-01 to rot90 `{0,90,180,270}` (not continuous angles) per the FSQ
reference (`kaggle/train_runs:1056-1097`) and the EQ-VAE paper §3.3 the advisor
cited; continuous angles are deferred to the future `SO(2)` spec. Do not copy FSQ
quantization/codebook/discrete-index artifacts; replace them with
continuous-latent statistics for the normal VAE and future `SO(2)` model.

DDP correctness pass, 2026-07-02 (uncommitted, working tree): the open HIGH
DDP-correctness follow-ups are fixed in
`src/eqvae/training/selected_runtime_runner.py`, with matching strict-verifier
assertions in `src/eqvae/benchmarking/selected_runtime_gate.py` and CPU tests
(including simulated `world_size=2` cases) in
`tests/test_selected_runtime_full_run.py`. FU-007: each rank seeds its
reparameterization eps generator from `data_seed + rank` (rank 0 unchanged), and
the full summary carries `per_rank_reparameterization_eps_divergent` with a
fail-closed blocker. FU-012: after a resume restores rank-0's generator, each
rank re-applies its per-rank offset from `(data_seed, rank, start_step)` (DDP
only). FU-008: best_model.pt is selected on the cross-rank sample-weighted
`deterministic_denoising` view (never clean, never `min()` over views, never
rank-0-local), recorded via `best_validation_selection_view`/`_reduction` and
asserted by the verifier; a no-validation-boundary fallback is labeled
`train_l1_no_validation_best` so the verifier rejects it. A 5-lens adversarial
subagent review plus two focused delta verifiers found and fixed a DDP boundary
deadlock hazard (added `_synchronized_amp_step_skipped`, a per-step cross-rank
AMP-skip agreement check that fails fast — this also implements the former
FU-020), a fallback-labeling honesty gap, and a missing end-to-end
best-selection test; no residual defects. All changes are no-ops for
single-process (`world_size == 1`) runs. Gate green: `./scripts/python_quality.sh`
= 272 passed, basedpyright clean; `git diff --check` clean; repo
`./scripts/agent_preflight.sh` clean. Resolved and deleted from
`docs/open_follow_ups.md`: FU-007, FU-008, FU-012, FU-020. These fixes gate a
VALID dual-T4 full run. FU-039 is now DECIDED (restart from scratch, discard v1
as a resume base; see below); the remaining gating blocker is FU-041 (real
fixed-25 selector).
Not yet committed; awaiting explicit commit approval.

Spec 0009 full-run remote push status, 2026-06-29: the first approved command
used the nonexistent action `push-selected-runtime-full`; the script printed
usage and exited locally before any Kaggle remote call. The corrected approved
command first blocked locally, again before any Kaggle remote call, because the
real push guard rejected dirty-worktree payload provenance. The source changes
were committed as `c02b538` (`Implement selected runtime full run prep`), the
ignored `kaggle/kernels/selected_runtime_full/run.py` was rebuilt from that
clean commit, and the clean guarded command
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_full`
was accepted by Kaggle at `2026-06-29 18:10 -0500` as
`maximusshtefan/eqvae-selected-runtime-full` version 1:
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-full. No
full-output verification has been run yet. The first approved status read at
`2026-06-29 19:11 -0500` returned
`KernelWorkerStatus.RUNNING`. The next approved status read at
`2026-06-30 11:36 -0500` returned
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; the first full run is no longer
running and has not produced verified local artifacts. The canceled-run outputs
were downloaded after explicit approval and inspected locally; see the next
paragraph. Next action is local resume-policy work, not another remote command.

Spec 0009 full-run v1 output inspection, 2026-06-30: after explicit approval,
the canceled-run outputs were downloaded with
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-selected-runtime-full runs/kaggle/selected_runtime_full_v1`.
Downloaded artifacts are local only and ignored by Git. Kaggle returned
checkpoints through `checkpoints/step_043750.pt` plus `best_model.pt`, embedded
payload files, and `eqvae-selected-runtime-full.log`; it did not return
`benchmark/` or `metrics/` directories. The log contains only DDP startup
warnings and no traceback or Python error, so the local evidence points to an
external Kaggle cancellation/runtime cutoff rather than a model/training
exception, but the downloaded log does not explicitly state the exact cutoff
reason. The latest checkpoint is schema `spec0001.checkpoint.v5`, selected row
`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`,
AMP policy `amp_fp16_conservative`, `optimizer_step =
successful_optimizer_update_count = 43750` (3.5 epochs of the planned 10,
35% complete, 81250 updates remaining), and SHA256
`ceeeb62a789ce38d123b443bd06edfc4ab41f76b5d2bf1474b39a232afeb3e54`.
It includes optimizer state, LR scheduler state, AMP GradScaler state
(`scale = 262144.0`), CUDA RNG state, named `train_data` generator state, beta
progress, DDP sampler progress, and selected-runtime identity. `best_model.pt`
is from update 31250 with validation L1 `0.1223133542` and SHA256
`a2fcabd928bf6c1f781939725010540854a2627bddbfe3c8dd4c43a548cd5824`.
Strict full-output verification failed as expected with
`selected_runtime_full_output_benchmark_dir_missing`, so v1 is not a completed
or paper-promotable run. Root cause of the missing CSVs/summaries was confirmed
against the historical FSQ run: FSQ appended CSVs during training before
checkpoints, while Spec 0009 v1 wrote interval checkpoints during the loop but
deferred CSV and summary writes until normal final teardown. Local future-proof
fix is now implemented in `src/eqvae/training/selected_runtime_runner.py`: full
runs use a two-phase boundary flush: atomic `metrics/train_steps.csv`,
`metrics/validation_metrics.csv`, `metrics/gate_health.csv`, partial
non-promotable benchmark summaries, and a partial manifest are written before
the new interval checkpoint is exposed, then refreshed with checkpoint hashes
after checkpoint/best-model save; final completion overwrites them with full
eligible artifacts. DDP ranks broadcast rank-0 artifact-write and
checkpoint-save failures so all ranks fail together. A follow-up adversarial
review of that interval flush found and fixed one more high-severity bug: it
prepended the resume-history prefix to every rank's rows before the all-gather,
so a resumed dual-T4 run would duplicate the pre-resume metric/validation rows
`world_size` times and later fail strict verification. The prefix is now kept
out of the all-gathered per-rank rows and prepended once after the gather,
mirroring the final-artifact path; simulated `world_size=2` regression test
`test_full_interval_flush_dedups_resume_prefix_under_simulated_ddp` guards it.
Focused test
`test_full_interval_flush_writes_resume_history_and_partial_artifacts` proves
the pre-checkpoint flush leaves resume-readable train history, the
post-checkpoint refresh records the checkpoint hash, and strict full-output
verification still rejects the incomplete run. `tests/test_kaggle_embedded_kernel.py`
also no longer assumes pytest scratch directories live under `/tmp`, so the
quality gate can use workspace-local scratch that is cleaned after use. The
latest local code also adds explicit full-run boundary logs and a final
boundary barrier after the second flush/checkpoint refresh; focused test
`test_full_boundary_logging_waits_at_barrier` covers the helper. Separate new
finding: Spec 0009 still lacks the fixed-25 rotated-input/transformed-latent
artifact protocol required by the issue images and Spec 0001, so do not treat a
future training/checkpoint verifier pass as equivariant embedding evidence. This
is tracked as `fixed25_equivariance_artifact_protocol` / FU-040 and is now
drafted as Spec 0010 (status draft, awaiting user approval) before any
paper-promotable full launch.
FU-039 v1 decision, DECIDED 2026-07-02: the first paper-promotable full run
RESTARTS FROM SCRATCH (fresh run from optimizer step 0). It does NOT resume from
`runs/kaggle/selected_runtime_full_v1/step_043750.pt`, because the user needs a
complete continuous training curve and v1 has zero recoverable metric rows (its
CSVs were never written before Kaggle cancelled), so a resume would leave steps
1-43750 permanently blank. v1's checkpoints stay on disk as a local record only;
they are NOT a resume source and must NOT be uploaded/attached as one. The
checkpoint-only-prefix continuation option is dropped. The two-phase interval
flush (committed `8d86f6a`) means a fresh run writes metrics at every half-epoch
boundary and survives cancellation, so the v1 data loss cannot recur.
FU-039 verification DONE 2026-07-02 (uncommitted working tree): confirmed by
reading that the full config carries no resume field, the dedicated kernel
(`kaggle/kernels/selected_runtime_full/run_template.py`) adds `--resume` ONLY when
`EQVAE_SELECTED_RUNTIME_FULL_RESUME` is set/non-empty (unset => fresh), the runner
resolves `start_step == 0` when no checkpoint loads
(`selected_runtime_runner.py:961-965`), and the interval-flush context is attached
iff `_is_full_run` (`:1032`). New CPU test
`test_fresh_full_run_flushes_metrics_at_first_boundary` drives a FRESH
`_run_train_steps` (start_step 0, real interval-flush context) and, via a spy on
`_write_interval_artifact_flush`, proves the FIRST half-epoch boundary persists
train + validation metric rows mid-loop (before any teardown). A 3-lens
clean-context adversarial workflow (test-soundness, fresh-launch-claim,
gate-regression), each verifying against the real files, returned zero findings.
FU-039 entry deleted from `docs/open_follow_ups.md`; Spec 0009 "Remaining
Blockers" updated. The first paper-promotable full launch still also needs FU-041
(the real fixed-25 selector) — see FU-041 status below.
Latest local verification (2026-07-02, this window): full gate
`./scripts/python_quality.sh` = `277 passed`, `0 errors, 0 warnings, 0 notes`
(basedpyright clean); `git diff --check` clean; repo `./scripts/agent_preflight.sh`
and workspace `/home/maximus/Documents/Tesis/agent_preflight.sh` both pass. Local
selector-kernel preflight `./scripts/kaggle_kernel.sh preflight-fixed25-selector` =
`23 passed` (build + validate + import-simulation). Heavy local gates used workspace
scratch under `/home/maximus/Documents/Tesis/.agent-tmp/equivariant-vae`; empty it
after use.
FU-041 status (2026-07-02): approach (b) CHOSEN and BUILT locally; commit + push
still pending user approval. The user picked generating the REAL selector inside a
dedicated CPU Kaggle kernel (approach b), and requested: validation-only (verified,
5-layer fail-closed), save BOTH which patches (selector JSON identities) AND the
images, and no GPU. Built locally (uncommitted working tree):
  - CRC-consistency fix (Option Y): the fixed-25 selector is validated with
    `validate_crc=True` end-to-end (`_prepare_fixed25_runtime`, the standalone
    `fixed25_equivariance` CLI, and the new `fixed25_originals` CLI) so a real
    CRC-validated selector (`crc_checked=True`, honoring
    `canonical_overwrite_requires_crc`) LOADS in the run. Empirically necessary:
    a `--validate-crc` selector failed the old `validate_crc=False` load path
    (`crc_checked` mismatch). Regression test added; makes the baked
    `generation_command` and the placeholder CRC policy internally consistent.
  - New CLI `src/eqvae/cli/fixed25_originals.py`: writes `artifacts/fixed25/
    originals.pt` + montage `originals.png` from a selector (no model), fail-closed
    on the placeholder.
  - New CPU kernel `kaggle/kernels/fixed25_selector` (`enable_gpu:false`, no
    machine_shape, no T4 quota): runs `select_fixed_patches --kind
    fixed_25_validation --validate-crc` then `fixed25_originals`, writing the
    selector JSON + originals to `/kaggle/working`; import-only mode + push guard
    (`guard_fixed25_selector_push_ready`: clean tree, CPU/dataset metadata, spec
    token, required literals); `preflight-fixed25-selector`/`status-fixed25-selector`
    /`output-fixed25-selector` wired; `run.py` gitignored; Spec 0010 addendum added.
  - Verification: full gate `277 passed` (4 new tests), selector preflight
    `23 passed`, and a 3-lens clean-context adversarial review (crc-correctness,
    kernel-correctness, shell-and-guard) returned ZERO findings.
Committed as `ce86c97` (2026-07-02); worktree clean; `run.py` rebuilt from that
clean commit. The approved push was BLOCKED by the Claude Code auto-mode permission
classifier (flagged data-exfiltration: a kernel push uploads the embedded `src/eqvae`
tree to Kaggle, which is how every kernel push works). This is a harness guardrail,
not a code issue; it must be run by the user or with auto mode off / a Bash
permission rule. Exact command (worktree is already clean + committed, so it will
pass the push guard):
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/fixed25_selector`
then `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-fixed25-selector`,
and after COMPLETE `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-fixed25-selector runs/kaggle/fixed25_selector`.
Remaining (gated): the one approved remote push
(`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh
push kaggle/kernels/fixed25_selector`) requires a clean committed worktree +
rebuilt `run.py`; then `status-fixed25-selector` / `output-fixed25-selector`
(`KAGGLE_REMOTE_CONFIRMED=1`) to download the selector + originals for review, then
commit the generated selector to the tracked config (overwrites the placeholder;
needs approval). No remote action taken this window.
Section C note: the generated `kaggle/kernels/selected_runtime_full/run.py` is
gitignored and STALE (built 2026-06-30, before Spec 0010 `8457233`, DDP fixes
`504c863`, and the FU-039 decision `402ec4a`); it MUST be rebuilt from a clean
commit via `./scripts/kaggle_kernel.sh build kaggle/kernels/selected_runtime_full`
before any full-run push.
Commit `d02204c` (`Implement selected runtime local mechanics`) recorded Spec
0006 plus adversarial fixes. Spec 0007 is implemented locally and Spec 0008
remote debug/tiny readiness is proved by v5. On 2026-06-28/29, the user
approved only the narrow selected-runtime debug/tiny Kaggle actions before the
later explicit full-kernel push approval recorded above. Historical provenance
follows.

2026-06-29 first full-run planning update: three read-only adversarial subagent
audits reviewed the post-Spec-0008 state. Findings: (1) no guarded first full
selected-runtime training Kaggle kernel/workflow exists in
`scripts/kaggle_kernel.sh` or `kaggle/kernels`; (2) the only selected-runtime
kernel is the bounded debug/tiny proof kernel, and it must not be reused as a
long-run launcher; (3) `selected_runtime_train` needs full-run schedule,
validation, checkpoint, readiness, and resume work before a 10-epoch run is
safe; (4) several docs still pointed toward a completed debug/tiny v5 action.
This handoff updates Spec 0008 status, the spec index, GOAL, Kaggle workflow
docs, and adds
`docs/specs/0009-first-full-selected-runtime-training-run.md`. Verification for
this docs/spec pass: `git diff --check`, repo `./scripts/agent_preflight.sh`,
and workspace `/home/maximus/Documents/Tesis/agent_preflight.sh` passed after
the edits. Next concrete action: implement Spec 0009 locally, then run its full
local verification and adversarial review before asking for exact remote
approval.

Spec 0009 wording follow-up, 2026-06-29: an external review flagged that the
current selected-runtime train step feeds zero epsilon into the VAE, so
optimization behaves deterministically as `z = mu`. That is acceptable for the
existing debug/tiny and paired-numerical proof lanes, but it would be wrong for
the first full VAE run. Spec 0009 now explicitly requires stochastic seeded
reparameterization during full-run optimization, permits zero/fixed epsilon only
for deterministic validation/artifacts/debug/tiny/numerical checks, and adds an
acceptance artifact for train reparameterization proof. Verification for this
wording fix: `git diff --check`, trailing-whitespace scan on the edited files,
repo `./scripts/agent_preflight.sh`, and workspace
`/home/maximus/Documents/Tesis/agent_preflight.sh` passed.

Spec 0009 local implementation update, 2026-06-29: the full-run surface was
implemented locally before the later approved Kaggle push. Added the full config,
dedicated `selected_runtime_full` Kaggle kernel, full-run shell guards and
preflight, strict `--verify-full-output` verifier, and focused tests. The
runner derives the selected v5 schedule into exactly 10 epochs, 12500 updates
per epoch, 125000 target optimizer updates, and 6250-update half-epoch
intervals; full-run training samples stochastic seeded epsilon, while zero/fixed
epsilon stays confined to deterministic debug/tiny/validation/artifact/numerical
lanes; validation metrics are scheduled for clean and deterministic denoising
views; interval, final, and validation-best checkpoints are written with latest
four interval retention; resume proof now covers GradScaler, CUDA RNG,
sampler/progress offset, selected-runtime identity, optimizer/scheduler, and
beta progress. Adversarial review initially found two high-severity blockers:
`--verify-full-output` could accept skinny train/checkpoint evidence, and the
resume proof/full kernel did not prove/allow full restart discipline strongly
enough. Both are fixed locally: the verifier now requires complete per-rank
train-step coverage for every successful update, exact latest-four interval
checkpoint retention and manifest hashes, and explicit GradScaler/CUDA RNG
restore-attempt/restored evidence; the full kernel accepts an explicit
`EQVAE_SELECTED_RUNTIME_FULL_RESUME` checkpoint hook. A second post-fix
adversarial review found no remaining high-severity blocker, but flagged a
medium preflight dirty-bypass leak and a low mislabeled
`selected_runtime_full_summary.retained_interval_checkpoints` field. Both are
fixed: the dirty bypass is accepted only by the explicit local preflight guard
mode and is rejected by the real push guard, and both full-run summaries now
report interval checkpoints separately from `final.pt` and `best_model.pt`
while the manifest still hashes all retained checkpoints. A final post-edit
adversarial review found no high-severity blocker and one medium resume-output
gap: a resumed long run could overwrite metrics with post-resume rows only and
then fail strict full-output verification. That is fixed locally: resume now
loads and merges pre-resume train/validation rows, carries retained interval
checkpoint metadata and prior best-validation state, and full-run resume fails
closed when required prior train history is missing. Local verification passed
after those fixes, using ignored repo-local `runs/local_tmp/...` scratch for
heavy temp output and deleting each scratch directory after use. The local
preflight/quality scripts now default to process-unique self-cleaning scratch
under `runs/local_tmp/...`, and the runner preflight no longer leaves
`runs/local_preflight` artifacts:
`tests/test_selected_runtime_runner.py` (`8 passed`),
`tests/test_selected_runtime_full_run.py` (`10 passed`),
`./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` (`52 passed`),
`./scripts/kaggle_kernel.sh preflight-selected-runtime-full` (`11 passed`),
strict debug v5 `--verify-output`, `./scripts/python_quality.sh` (`242
passed`, basedpyright clean), `git diff --check`, repo
`./scripts/agent_preflight.sh`, and workspace
`/home/maximus/Documents/Tesis/agent_preflight.sh`. Temporary scratch pressure
was handled by deleting only known local scratch directories under `/tmp` and
ignored `runs/local_tmp`; no source artifacts were removed. This local
verification is the evidence behind commit `c02b538` and the later approved
full-kernel version 1 push. Status reads and output downloads remain separate
approval-gated remote actions.

Selected-runtime debug/tiny v5 auth/push status, 2026-06-29: after explicit
approval for the narrow selected-runtime debug/tiny v5 push, the local push
guard and embedded-kernel checks passed, but the actual Kaggle upload failed
before remote execution with Kaggle's authentication-required message. A direct
selected-runtime status call also failed until the CLI was given a freshly
minted OAuth token. Root cause: local Kaggle CLI 2.2.1 could generate a fresh
token from `~/.kaggle/credentials.json`, while normal API calls were still using
a stale cached token from the same credentials file. The selected-runtime slug
was correct. New helper `scripts/kaggle_oauth_exec.py` avoids shell token
substitution and token printing by generating a fresh token with the installed
Kaggle SDK, writing it only to a temporary 0600 token file for the child
process, and deleting that file when the child exits. `scripts/kaggle_kernel.sh`
now routes authenticated Kaggle reads/writes through that helper when OAuth
credentials exist, and `api-check` accepts an optional kernel directory so
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
kaggle/kernels/selected_runtime_debug` checks the actual selected-runtime
kernel. Follow-up adversarial review found that `api-check` still had one raw
`kaggle auth print-access-token` probe before the wrapped endpoint checks; that
probe has been removed and covered by `tests/test_kaggle_oauth_exec.py`.
Fresh live read-only verification of `api-check
kaggle/kernels/selected_runtime_debug` passed with the known quota warning, and
`status-selected-runtime-debug` now succeeds through the wrapper and reports
the old v4 `KernelWorkerStatus.ERROR`. After fresh explicit approval for the
narrow retry, the guarded selected-runtime API preflight passed through the
fresh OAuth wrapper, Kaggle accepted
`maximusshtefan/eqvae-selected-runtime-debug` version 5 at
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-debug, and
the immediate guarded status read at `2026-06-29 03:27 -0500` returned
`KernelWorkerStatus.RUNNING`. The guarded follow-up status read returned
`KernelWorkerStatus.COMPLETE`. Outputs were downloaded to
`runs/kaggle/selected_runtime_debug_v5`, and strict local verification passed:

```bash
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_v5 \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
```

The gate summary has `status = "local_pass"`,
`status_scope = "permission_gated_remote_debug_tiny_proof"`,
`launch_blockers_remaining = []`, and component statuses `local_pass` for
artifact manifest, checkpoint/resume, gate health, real UBC debug,
selected-runtime plan application, and tiny overfit, plus selector generation
`pass`. The selector proof is canonical real UBC:
`fixed_32_selector_real = true`, `remote_selector_generation_ready = true`,
`selector_count = 32`, dataset slug
`maximusshtefan/patches-pre-shuffled-ubc-ocean`, train CSV SHA256
`8fc4959f7de006eed259f818ef2cc4ea03d1f3ec6ba483bf7229c04562f22a52`, train
binary size `58982400064`, header CRC32 `1289496176`, and selector SHA256
`15e32a5e54210588bcbe4bfb55afd0f32799b184d39b89ccbf098627cdeee4a1`.
Tiny-overfit passed the hardened AMP proof with `grad_scaler_init_scale =
16384.0`, `amp_step_skipped_count = 0`, `nonfinite_count = 0`, 128 optimizer
steps, 256 nested tiny metric rows over ranks 0/1, observed batch sizes `[12]`,
`l1_improvement_fraction = 0.3546171902297934`, and
`recon_loss_improvement_fraction = 0.30851685095892983`. The first full real
selected-runtime run is now the next candidate action, but it requires fresh
explicit approval and is not launched by this proof.

Selected-runtime debug/tiny v1 push status, 2026-06-28: local source commit
`5591737` (`Remove stale selected runtime blocker`) was clean, the selected
debug preflight passed, the Kaggle API preflight passed useful checks while
still warning on the known quota/files endpoints, and Kaggle accepted
`maximusshtefan/eqvae-selected-runtime-debug` version 1 at
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-debug. The
single immediate guarded status read at `2026-06-28 14:34:43 -0500` returned
`KernelWorkerStatus.RUNNING`. Do not actively poll in-turn; the next concrete
action is, after about `2026-06-28 15:05 -0500` or later, run
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-selected-runtime-debug`.
The resumed guarded status read at `2026-06-28 15:23:22 -0500` still returned
`KernelWorkerStatus.RUNNING`; the next status check should be after about
`2026-06-28 15:55 -0500` or later.
The next guarded status read returned `KernelWorkerStatus.COMPLETE`, and outputs
were downloaded to `runs/kaggle/selected_runtime_debug_v1`. Strict verification
failed, as intended for a non-passing remote proof:
`selected_runtime_output_missing_metric_train_steps.csv` and
`selected_runtime_output_unexpected_metric_train_metrics.csv`. Artifact
inspection found the root blocker happened before training:
`benchmark/fixed32_selector_readiness.json` has
`failure_kind = fixed_32_selector_masked_holdout_unavailable` with
`validation_detail` resolving the selector's relative
`docs/data/ubc_ocean_masked_holdout_ids.csv` from `/kaggle/working` instead of
the embedded payload. The wrapper therefore correctly stayed fail-closed and
did not launch debug or tiny training. Local fix is implemented but not remotely
pushed: the selected-runtime debug wrapper now calls
`fixed32_selector_status` from the embedded payload CWD so the holdout CSV is
visible, and `eqvae.cli.selected_runtime_gate --verify-output` now matches the
documented command without requiring debug/tiny config arguments. Verification
after the fix: focused tests passed (`23 passed`),
`./scripts/kaggle_kernel.sh preflight-selected-runtime-debug` passed, the
downloaded v1 verifier now runs with the documented command and still reports
the v1 metric-artifact failure, `git diff --check` passed, and
`./scripts/python_quality.sh` passed (`215 passed`, `0 errors, 0 warnings, 0
notes`). The next concrete action requires explicit user approval for a new
narrow selected-runtime debug/tiny Kaggle push, expected as v2; this is still
not approval for the long full run. The v1 verifier command is:

```bash
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_v1 \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
```

Selected-runtime debug/tiny v2 push status, 2026-06-28: after user approval for
the narrow rerun, local source commit `fabbfb8` (`Prove selected runtime debug
holdout resolution`) was clean. A stronger regression proved the actual v1 root
fix rather than a symptom workaround: from a fake `/kaggle/working`-style CWD
without `docs/data`, direct `fixed32_selector_status` fails with
`fixed_32_selector_masked_holdout_unavailable`, while the wrapper helper
validates from the embedded payload CWD and reaches the intended
`fixed_32_selector_not_canonical_real_ubc` result for synthetic data. Local
verification before push passed, and Kaggle accepted
`maximusshtefan/eqvae-selected-runtime-debug` version 2 at
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-debug. The
immediate guarded status read at `2026-06-28 17:05:49 -0500` returned
`KernelWorkerStatus.RUNNING`; the next guarded status read at
`2026-06-28 17:45:03 -0500` returned `KernelWorkerStatus.ERROR`. Outputs were
downloaded to `runs/kaggle/selected_runtime_debug_v2`. The v2 selector proof is
real progress: `benchmark/fixed32_selector_readiness.json` has `status =
"pass"`, `fixed_32_selector_real = true`, `remote_selector_generation_ready =
true`, `selector_status.canonical_real_ubc = true`, `selector_count = 32`, and
the locked real UBC train CSV/bin/header fingerprints. Strict verification
still fails, as it must, because only the selector artifacts and log exist; the
missing artifacts are the training/checkpoint/gate/tiny summaries and
`metrics/train_steps.csv`/`metrics/gate_health.csv`. The Kaggle log root cause
is later than v1 and precise:
`ValueError: eps shape torch.Size([12, 16, 32, 32]) does not match mu shape
torch.Size([8, 16, 32, 32])`. The fixed-32 selector has 32 patches and the v5
runtime batch cap is 12, so the third single-process batch was size 8 while the
runner built explicit VAE epsilon from the nominal configured batch size. Local
fixes after v2 are implemented and verified: the runner now sizes explicit
epsilon from the realized input batch and records actual metric batch size; a
regression drives fixed-32/bs12 for three steps and proves `12, 12, 8`; the
selected-runtime debug wrapper now launches phase1/resume/tiny through
`python -m torch.distributed.run --standalone --nproc_per_node=2 -m
eqvae.cli.selected_runtime_train` with the embedded payload on `PYTHONPATH`
instead of direct single-process calls; the shell push guard requires that
distributed launcher path; and the post-download verifier now hash-links the
selector readiness to the downloaded selector, validates canonical selector
metadata locally, replays artifact-manifest hashes, checks gate-health CSV
content, and tightens train-step CSV parsing. Adversarial subagent review found
the partial-batch epsilon fix to be a true source fix, then flagged the verifier
and wrapper hardening items, which are now addressed. Local verification after
the v2 fixes: focused selected-runtime gate/runner/wrapper tests passed,
`./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` passed,
`./scripts/kaggle_kernel.sh preflight-selected-runtime-debug` passed from the
final worktree, `./scripts/python_quality.sh` passed (`221 passed`, `0 errors,
0 warnings, 0 notes`), `git diff --check` passed, and v2 `--verify-output`
still reports the expected missing remote training artifacts. The next concrete
action is to commit these local fixes, run repo/workspace preflights, then ask
for explicit user approval for a new narrow selected-runtime debug/tiny Kaggle
push, expected as v3. This remains not approval for the long full run. The v2
verifier command is:

```bash
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_v2 \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
```

Selected-runtime debug/tiny v3 push status, 2026-06-28: after explicit user
approval for the narrow rerun only, local source commit `09b5b24` (`Fix
selected runtime debug v2 blockers`) was clean. The selected-runtime debug
preflight passed immediately before push, rebuilding the generated embedded
kernel and proving the payload matched the current worktree. The Kaggle API
preflight passed the useful read checks while keeping the known quota/files
endpoint warnings. Kaggle accepted `maximusshtefan/eqvae-selected-runtime-debug`
version 3 at
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-debug. The
single immediate guarded status read at `2026-06-28 18:25:05 -0500` returned
`KernelWorkerStatus.RUNNING`. The guarded follow-up status read on 2026-06-29
returned `KernelWorkerStatus.ERROR`, and outputs were downloaded to
`runs/kaggle/selected_runtime_debug_v3`. The initial strict verification before
local hardening failed only on:

- `selected_runtime_output_tiny_overfit_not_pass`;
- `selected_runtime_output_gate_summary_not_pass`.

The v3 artifacts are real progress but not remote proof. The canonical selector,
selected-runtime plan application, checkpoint/resume proof, gate-health CSV,
artifact manifest, and debug/resume phase all wrote the expected artifacts.
The root gate summary is still `status = "fail"` with
`launch_blockers_remaining = ["tiny_overfit"]`; component `local_pass` values
inside downloaded artifacts are evidence rows, not a remote-pass claim. Only a
zero-blocker `--verify-output` result plus a passing gate summary may mark Spec
0008 remotely proved.

The precise v3 tiny blocker is not convergence. The tiny phase reached 128
successful optimizer updates and improved strongly
(`l1_improvement_fraction = 0.42601120821087546`,
`recon_loss_improvement_fraction = 0.37984046690403395`), but it failed the
Spec 0008 zero-tolerance AMP/nonfinite contract:
`amp_step_skipped_count = 2` and `nonfinite_count = 500`. The offending metric
rows are one per rank at optimizer step index 3 with per-rank `batch_size = 4`,
`grad_norm = inf`, and `amp_step_skipped = 1`. This came from cycling the
canonical 32-patch selector under dual-rank DDP with per-rank batch size 12,
which produced a repeated 12/4 per-rank microbatch pattern in tiny mode.

Local follow-up after v3 is implemented and locally verified but not remotely
proved. The runner keeps the v2 partial-batch eps source fix and its regression,
but `kaggle_tiny_overfit` with the fixed-32 selector now uses a deterministic
`fixed32_tiny_full_batch_repeated` sampler policy: the selector remains 32
unique canonical patches, while selector-order rows are repeated only to make
tiny-overfit microbatches full-sized for the selected bs12 AMP runtime. The
training, debug, and tiny summaries record `train_sampler_policy`,
`train_effective_global_epoch_samples`,
`train_effective_per_rank_epoch_samples`, and
`fixed_train_repeated_to_full_batch`; the tiny summary also records
`observed_batch_sizes`, `amp_step_skipped_count`, and aggregate
`nonfinite_count`. The wrapper and post-download verifier now require those tiny
fields, including derived v5 DDP effective samples `48` global / `24` per rank
and `observed_batch_sizes = [12]`. The runner training summary and local
readiness now aggregate nonfinite rows over all metric rows instead of only
looking at the final step. Focused local tests prove the DDP padding math covers
all 32 selector rows while avoiding tail microbatches, and the old debug-path
partial-batch regression still proves `12, 12, 8`.

Verification after the local v3 fix: `tests/test_selected_runtime_runner.py`
passed (`8 passed`); focused selected-runtime gate/runner/embedded-kernel tests
passed (`49 passed`); `./scripts/kaggle_kernel.sh
preflight-fixed32-selector-readiness` passed (`11 passed`);
`./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` passed (`50
passed` plus script checks); `./scripts/kaggle_kernel.sh
preflight-selected-runtime-debug` passed (`11 passed`, regenerated and verified
the embedded payload, then `28 passed`); the documented synthetic selector CLI
command passed; remote-generate push readiness passed;
`./scripts/python_quality.sh` passed (`224 passed`, `0 errors, 0 warnings, 0
notes`); `git diff --check` passed; repo `./scripts/agent_preflight.sh` passed;
and workspace `/home/maximus/Documents/Tesis/agent_preflight.sh` passed. The
downloaded v3 verifier command remains:

```bash
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_v3 \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
```

Under the hardened verifier it now reports the historical tiny failure plus the
new missing-sampler-evidence blockers:
`selected_runtime_output_tiny_overfit_not_pass`,
`selected_runtime_output_tiny_amp_skips_nonzero`,
`selected_runtime_output_tiny_nonfinite_nonzero`,
`selected_runtime_output_tiny_sampler_policy_mismatch`,
`selected_runtime_output_tiny_not_repeated_to_full_batch`,
`selected_runtime_output_tiny_global_epoch_samples_mismatch`,
`selected_runtime_output_tiny_per_rank_epoch_samples_mismatch`,
`selected_runtime_output_tiny_batch_sizes_not_full`, and
`selected_runtime_output_gate_summary_not_pass`. It fails as expected because
the v3 artifacts predate the local sampler/evidence fix. The next concrete
action after committing and preflighting this local source fix is to ask for
explicit user approval for a new narrow selected-runtime debug/tiny Kaggle push,
expected as v4. This is still not approval for the first full long run.

Selected-runtime debug/tiny v4 status, 2026-06-29: after explicit user approval
for the narrow rerun only, local source commit `ce72fa0` (`Fix selected runtime
tiny proof sampler`) was clean and Kaggle accepted
`maximusshtefan/eqvae-selected-runtime-debug` version 4. The guarded follow-up
status read at `2026-06-29 01:51:13 -0500` returned
`KernelWorkerStatus.ERROR`, and outputs were downloaded to
`runs/kaggle/selected_runtime_debug_v4`. The v4 artifacts prove the sampler fix
worked: tiny uses `train_sampler_policy = fixed32_tiny_full_batch_repeated`,
`fixed_train_repeated_to_full_batch = true`, effective samples `48` global /
`24` per rank, and `observed_batch_sizes = [12]`. v4 is still not a passing
remote proof. Strict verification fails because both ranks overflowed once at
tiny `optimizer_step_index = 3` on full `batch_size = 12` rows with
`grad_norm = inf`, `nonfinite_count = 125` per rank, and
`amp_step_skipped = 1`; the retry after the GradScaler scale was reduced
succeeded, and the tiny run still completed 128 successful updates with about
`0.3399` L1 improvement and `0.2948` reconstruction-loss improvement.

Local follow-up after v4 is implemented but not remotely proved. The selected
AMP-conservative runner now uses an explicit conservative GradScaler startup
scale (`16384.0` instead of PyTorch's default `65536.0`) and records that scaler
policy in `training_summary.json`, `tiny_overfit_summary.json`, and the
selected-runtime plan-applied proof as a runner AMP extension. The wrapper and
post-download verifier require that scaler evidence. The verifier also now
hashes and inspects the nested
`tiny_overfit_phase/metrics/train_steps.csv` rows directly, requiring both DDP
ranks, steps `1..128`, full bs12 rows, finite `grad_norm`, zero AMP skips, and
zero nonfinite counts. Under the hardened verifier, downloaded v4 correctly
fails with scaler-evidence and row-level tiny blockers, including
`selected_runtime_output_tiny_train_steps_amp_skip`,
`selected_runtime_output_tiny_train_steps_nonfinite`, and
`selected_runtime_output_tiny_train_steps_grad_norm_nonfinite`. Focused local
tests after this fix passed (`51 passed`). The next concrete action is to run
the full local preflight/quality gates, commit, then request explicit user
approval for a new narrow selected-runtime debug/tiny Kaggle push, expected as
v5. This remains not approval for the first full long run.

Synthetic timing is now
completed provenance for screening: remote versions 1, 2, 3, and 4 completed
successfully as non-promotable evidence, with v4 as the current
5-warmup/25-measured repeat-shortlist run. The active real-data state is:
remote v4 passed the canonical identity/hash/CRC/window plus clean-validation
loader proof lane; remote v5 completed as non-promotable candidate evidence with
two eligible eager single-T4 bs4 FP32 rows and no `selected_runtime.json`; the
v6 follow-up adds phase timings, eager-first train-step evidence ordering,
CUDA cache cleanup between candidate evidence attempts, failed-candidate
diagnostics, and runtime-proof evidence counters. Remote v6 completed, artifacts
are downloaded under `runs/kaggle/real_data_runtime_pretest_v6`, and inspection
found no new runtime selection: two eager single-T4 bs4 FP32 rows remain
eligible, eager bs8/bs12 candidate evidence failed with hash-only
`candidate_train_step_RuntimeError` diagnostics, eager bs32 rows are recorded
as `runtime_OutOfMemoryError`, compiled rows remain diagnostic/ineligible, and
no `selected_runtime.json` was written. The local v7 diagnostics follow-up adds
bounded `failure_message_excerpt` fields to failed candidate evidence so the
next artifact can expose the actual exception. After an approved Kaggle API
preflight and GPU-quota confirmation, v7 was pushed successfully; Kaggle
accepted version 7 at
`https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
The immediate guarded post-push status read returned
`KernelWorkerStatus.RUNNING` at `2026-06-19T23:38:51-05:00`, and the next
guarded poll returned `KernelWorkerStatus.COMPLETE` at
`2026-06-20T02:21:15-05:00`. Outputs are downloaded under
`runs/kaggle/real_data_runtime_pretest_v7`; artifact inspection confirmed
`runtime_proof.status = pretest_incomplete`, `selection_ready = false`,
`selected_runtime_written = false`, no `selected_runtime.json`, two eligible
bs4 eager FP32 pass rows, and the expected non-promotable capped-pretest
blocking claims. The v7 diagnostic fix worked: failed candidate evidence in
the manifest now exposes
`failure_message_excerpt = "quantile() input tensor is too large"` for the repeated
`candidate_train_step_RuntimeError` hash
`757ab3828da1202c080e587121c92ffa9210d9ecace6cb28842a62504733fc14`.
The 2026-06-20 local selected-runtime slice now exists under
`src/eqvae/benchmarking/runtime_selection.py`, with a local CLI at
`src/eqvae/cli/runtime_selection_benchmark.py` and focused tests in
`tests/test_runtime_selection_benchmark.py`. It records v8 artifact hashes as
`candidate_shortlist_only` provenance, writes its own model-count,
runtime-proof, runtime-matrix, dataloader, numerical, corruption, gate-health,
and linked evidence paths, enforces eager single-visible-T4 bs8/bs12 FP32
confirmation with bs4 fallback before AMP follow-up, keeps compiled rows
diagnostic-only, and refuses `benchmark/selected_runtime.json` unless the
separate benchmark has passing dual-T4 DDP train-step timing plus linked safety
proofs. The selected-runtime gate now requires train and validation dataloader
rank coverage, candidate-bound gate-health row ids, child-process
`torchrun --nproc_per_node=2` proof, and a hash-linked
`benchmark/stain_corruptor_qa.json`. The local default path is intentionally
fail-closed and writes a failed
`runtime_proof.json` rather than a selected runtime when real dual timing is not
provided. The selected-runtime Kaggle executor/kernel path now exists at
`src/eqvae/benchmarking/runtime_selection_executor.py`,
`src/eqvae/cli/runtime_selection_executor.py`, and
`kaggle/kernels/runtime_selection`. The generated script kernel embeds only the
required v8 provenance files, re-runs the selected-runtime benchmark on Kaggle,
and is guarded by `runtime_selection_kernel_ready`. Commit `fba9d98`
(`Add selected runtime Kaggle executor`) was created on 2026-06-20, Kaggle
accepted runtime-selection version 1 at
`https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection`. A guarded
status poll later returned `KernelWorkerStatus.ERROR`, and outputs were
downloaded under `runs/kaggle/runtime_selection_v1`. Version 1 still produced
the strict benchmark artifacts, including a passing real dual-T4 DDP timing
gate, but wrote no `benchmark/selected_runtime.json`: the writer blocked on
linked single-visible row proof because executor gate-health rows were left
`full_run_eligible = false` and train corruption rows were incorrectly required
to carry the validation clean-RNG flag. The wrapper then raised because
`model_inventory.csv` was not in its expected artifact allow-list. The local v2
fix adds `model_inventory.csv` to the allow-list, normalizes `local_pass`
gate-health rows before eligibility computation while keeping failed non-gate
rows ineligible, and requires `clean_validation_rng_advanced = false` only for
validation corruption rows. Focused regression tests cover these cases and
adversarial subagent review found no selected-runtime fail-open blocker. Commit
`96e41f4` (`Fix runtime selection v1 proof plumbing`) was pushed as
runtime-selection Kaggle version 2 after clean local gates and a clean
rebuild/validate. Version 2 completed and downloaded under
`runs/kaggle/runtime_selection_v2`; it fixed the wrapper error and preserved the
passing dual-T4 timing proof, but still wrote no `benchmark/selected_runtime.json`
because gate-health rows were missing for the three passing single-visible
`indexed_masked` candidates. The local v3 fix expands branchless single-visible
gate-health rows to same-shape indexed candidates only after those runtime rows
already pass linked evidence. Commit `b6b024a`
(`Bind indexed runtime gate evidence`) was pushed as runtime-selection Kaggle
version 3; it completed and downloaded under
`runs/kaggle/runtime_selection_v3`. Version 3 wrote
`benchmark/selected_runtime.json` with `status = "pass"` and
`runtime_proof.status = "pass"`. The selected row is
`dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`: dual T4 DDP,
`world_size = 2`, `nproc_per_node = 2`, per-device batch size 12, global batch
24, FP32 eager/no compile, `indexed_masked` corruption, `samples_sec =
14.035497`, projected epoch time about 356.24 minutes, and 10-epoch wall-time
projection about 59.37 hours. No GitHub or Overleaf push was performed or
approved. This is the proof-clean safety baseline, not the final efficiency
answer for a 60h+ run. Runtime-selection v4 ran the efficiency follow-up and
failed closed: it did not write `selected_runtime.json` because the writer
treated small selected-row numerical drift and nonselected-row linked-proof
failures as global blockers. The fastest otherwise clean v4 row was
`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
at `samples_sec = 25.220604` with zero AMP skips and a projected 10-epoch wall
time around 33.0 hours, but that result remains unpromoted until a corrected
remote rerun writes the proof artifact. Local commit `fc5227d` relaxes only
bounded finite numerical drift, keeps AMP skips and large drift as row
blockers, and scopes linked proof to the selected candidate. Local replay of
the v4 artifacts through that patch wrote the intended selected runtime with
proof `pass`. Runtime-selection v5 completed, downloaded to
`runs/kaggle/runtime_selection_v5`, passed strict local replay under current
`main`, and wrote `benchmark/selected_runtime.json`. The selected row is
`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`:
dual T4 DDP, per-device batch size 12, global batch size 24, AMP conservative
FP16 autocast with FP32 loss, no compile, contiguous layout,
`indexed_masked` corruption, zero AMP skips, gate-health pass, `samples_sec =
27.381321`, and estimated 10-epoch wall time `109563.740875` seconds
(about 30.4 hours). The selected numerical rows have no nonfinite values, no
AMP skips, gate-health pass, and only the expected small
`dual_t4_numerical_delta_failed` row. The selected payload still sets
`full_training_launch_ready = false` with launch blockers for selected-runtime
debug proof, checkpoint/resume proof, and tiny-overfit proof. Do not ask to
launch the first real 60h-scale run until those gates pass. New user
preference recorded 2026-06-21: before the first long real run, also run a
small broader AMP/non-conservative follow-up to test whether a less
conservative AMP policy such as `amp_scalar_gate_relaxed` can beat v5 without
catastrophic failures. v5 remains the safe selected-runtime fallback unless the
aggressive AMP follow-up passes the same local preflight, linked proof, strict
local replay, debug/resume, and tiny-overfit gates. The current local working
slice adds a real scalar-gate relaxed precision switch, updates the
runtime-selection efficiency follow-up to use v5 as fallback, and adds the
`amp_fp16_scalar_gate_relaxed` policy row. Local commit `580a844`
(`Add compact relaxed AMP runtime selection follow-up`) was created because the
Kaggle payload guard requires a clean Git commit. Kaggle accepted runtime-
selection version 6 on 2026-06-21; the one guarded post-push status read
returned `KernelWorkerStatus.RUNNING` at 2026-06-21 14:53:39 -05. On the next
guarded status read, v6 was `KernelWorkerStatus.COMPLETE`; outputs were
downloaded to `runs/kaggle/runtime_selection_v6` and replayed locally through
the current writer at `/tmp/eqvae_runtime_selection_v6_replay`. The relaxed row
passed runtime/gate-health with zero AMP skips but reached only `25.288828`
samples/sec versus the v5 fallback's `27.381321`, with one bounded
`dual_t4_numerical_delta_failed` row. The replay regenerated the same
fail-closed decision: `runtime_proof.status = "fail"`,
`selected_runtime_written = false`, no `benchmark/selected_runtime.json`, and
blocker `selected_runtime_reuses_configured_baseline_no_replacement`. v5 remains
the selected-runtime fallback; the next gate is real UBC/Kaggle selected-runtime
debug/resume/artifact/tiny-overfit proof using v5.
The working historical FSQ training reference is `kaggle/train_runs`: it trained
correctly and should be used as the source for broad macro-architecture and
runtime-efficiency ideas, while FSQ quantization/codebooks/rounding/discrete
latents and sub-pixel/PixelShuffle upsampling stay out of the continuous
`SO(2)` route.

Clean-context adversarial
subagent reviews were run on 2026-06-05, 2026-06-10, 2026-06-11, a focused
scaffold-readiness check on 2026-06-12, and a focused v7 handoff/guard audit on
2026-06-20 plus a focused v7 quantile/evidence review on 2026-06-20. The
2026-06-11 passes
confirmed that the previous `4x4` latent target was inconsistent with the
FSQ-successor spatial-coherence goal, that the historical HED corruptor must not
be copied as-is, and that the benchmark specs were directionally right but not
implementation-ready until launch topology, schemas, thresholds, dataloader
throughput, paired numerical checks, selected-runtime debug, and tiny-overfit
gates were made explicit. A local Kaggle CLI execution scaffold now exists; it
is not broad/full-run Kaggle-push-ready. The no-dataset setup smoke passed on
Kaggle, while real-data smoke paths attach the 60 GB+ dataset and are guarded by
`KAGGLE_FULL_DATASET_CONFIRMED=1`. The no-dataset synthetic binary timing
kernel/guard path now exists on GitHub; remote Kaggle version 3 produced
downloaded ignored broad-screen benchmark evidence at
`runs/kaggle/synthetic_timing_2gib_v3`, and remote Kaggle version 4 produced
the repeated-shortlist evidence at
`runs/kaggle/synthetic_timing_repeat_2gib_v4`.

Spec-driven development is now an active repo workflow. The first active spec is
`docs/specs/0001-translatable-normal-vae-baseline.md`, now reopened as
`draft active` and not implementation-ready. The reopened direction is:
`32x32x16` scalar Gaussian latent, no FSQ quantizer or learned bottleneck scale
`s`, corrected Tellez-style HED/OD stain corruption plus per-image Gaussian
noise `Uniform(0.0, 0.05)`, full-mixing scalar Conv2d baseline channels with the
same learned pointwise scalar gate family used by future `SO(2)` scalar/trivial
fields, future `SO(2)` radial gates for nontrivial irrep fields, no activation
`gamma`, radial-gate `eps = 1e-4` as the first FP16-safe candidate, no final
`tanh` output, a zero-initialized final RGB convolution, and
`L1 + 0.1 * (1 - SSIM) + beta * KL`. L1 uses raw normalized output; SSIM,
PSNR, saved images, and artifacts use an explicit clamped image-domain
projection outside the model forward path. The precision/autograd policy is now
explicit: AMP may cover the main convolutional forward after benchmarking, while
corruption runs FP32/no-grad, posterior/KL/loss/radial-gate numerics run FP32
with gradients where needed, and clean validation must not consume corruption
RNG. The Kaggle benchmark must now select the fastest practical precision,
compile, corruption, layout, DDP, and CUDA backend policy. It should prefer
material speedups even when bitwise determinism or small numerical agreement
gets worse, while still blocking catastrophic failures through dataloader,
non-finite, AMP-skip, DDP, checkpoint/resume, artifact, and gate-health
telemetry. It is not final-paper-claim-ready until the
sealed masked-WSI test shard is generated and locked.
The 2026-06-13 HED/stain corruptor spec-lock pass completed a focused
literature and historical-FSQ review plus adversarial subagent review. Spec 0001
now records `corruption_ready` for the local correctness/QA slice:
scikit-image-compatible HED semantics are the oracle, runtime code must be
repo-owned PyTorch, the public API remains NCHW RGB `[-1, 1]`, the internal HED
domain is RGB `[0, 1]` without the historical sRGB-to-linear step, tiny third
HED residual-axis jitter is kept as anti-signature jitter rather than biological
DAB, conservative corruption is the default, FSQ-wide values are a named
benchmark profile, RNG is stateless from semantic patch keys and excludes rank,
clean validation/test consume no corruption RNG, `branchless_all` is first, and
`benchmark/stain_corruptor_qa.json` is the non-promotable local QA artifact.
The local implementation now exists in `src/eqvae/corruption/stain.py` with
focused `tests/test_stain_corruptor.py` and
`src/eqvae/benchmarking/stain_corruptor_qa.py`; scikit-image 0.26.0 is a
dev/test oracle, not a runtime import in active `src/eqvae` corruption code. The
canonical short decision note is
`docs/decisions/0007-stain-corruptor-convention.md`.
The 2026-06-17 synthetic Kaggle timing spec pass added
`kaggle_synthetic_timing_contract_ready`. The 2026-06-18 implementation added
the no-dataset GPU kernel at `kaggle/kernels/synthetic_timing`, a dedicated
push guard in `scripts/kaggle_kernel.sh`, deterministic streaming UBC-format
shard generation, active loader/collate/normalization proof, and
single-visible-T4 plus dual-T4/DDP child-process timing attempts. Remote
Kaggle version 1 completed with `status = "synthetic_timing_pass"` in all three
JSON artifacts, 16/16 matrix rows passing, and both `single_visible_t4` and
`dual_t4_ddp` modes passing on the historical compact profile. Remote Kaggle
version 2 completed on the current `synthetic_binary_2gib_histology_like_v1`
profile with 10,912 total patches, 5,456 train / 5,456 validation, 16/16 matrix
rows passing, zero fit-probe rows, and both accelerator modes passing. Version 3
reran the same profile after adversarial review, preserving per-rank DDP device
assignments, child/torchrun return codes, and exact
`effective_samples_per_epoch = 300000` for `drop_last = false`. It writes only
non-promotable synthetic timing artifacts and did not write
`benchmark/selected_runtime.json`. Remote version 4 reran the v3 top-four
shortlist with `warmup_steps = 5`, `measured_steps = 25`, and `repeats = 1`;
all four rows passed, the repeat gate is marked complete in the recommendations
artifact, and it still writes no `benchmark/selected_runtime.json`. It may
screen/order rows for the real-data benchmark but cannot unlock
selected-runtime debug/full runs.
The canonical short decision note is
`docs/decisions/0008-kaggle-synthetic-timing-pretest.md`.
Strict Python quality is also an active workflow via
`docs/specs/0002-strict-python-quality-gate.md`.
Kaggle CLI execution is scaffolded via
`docs/specs/0003-kaggle-cli-execution-workflow.md`,
`docs/kaggle_cli_workflow.md`, `scripts/kaggle_kernel.sh`, and
`kaggle/kernels/non_eq_vae_debug`. The debug kernel now contains only the
narrow capped `kaggle_smoke_ready` launcher: it runs bundled repo code from the
ignored payload, resolves the pre-shuffled UBC dataset, carries sample metadata
for deterministic HED corruption, executes at most three train steps and one
clean-validation batch, and writes non-promotable
`benchmark/kaggle_smoke.json`. It is not runtime selection, convergence
evidence, a full benchmark, or a full run. The first remote real-data smoke
version finished as `KernelWorkerStatus.ERROR` with `ModuleNotFoundError: No
module named 'eqvae'`, because the Kaggle CLI did not make the sibling payload
directory available to the uploaded script. It produced no benchmark evidence.
The new setup-only scaffold lives in `kaggle/kernels/setup_smoke`: it attaches
no dataset, requests no GPU, generates an ignored single-file `run.py` with an
embedded zipped payload, creates tiny synthetic UBC-format shards under the
output directory, and writes non-promotable
`benchmark/kaggle_setup_smoke.json`. It is setup/API/import/artifact evidence
only, not real-data loader evidence or runtime selection.
The Kaggle behavior inventory now lives at
`docs/behavior_inventory_kaggle.md`. Dataset slugs were confirmed through the
Kaggle CLI, and the debug kernel metadata now points at
`maximusshtefan/patches-pre-shuffled-ubc-ocean`.
Important dataset nuance: that dataset is the confirmed pre-shuffled
train/validation patch source, with `ubc_train_shuffled.*` and
`ubc_ocean_valid.*` files verified through the Kaggle CLI on 2026-06-10. It does
not contain a held-out test shard. The split was checked against official
UBC-OCEAN metadata on 2026-06-10: train has 322 non-TMA WSIs and 300000 patch
rows, validation has 39 non-TMA WSIs and 30000 patch rows, train/validation WSI
overlap is zero, and both splits have zero overlap with the 152 supplemental-mask
WSIs. The exact masked holdout candidate list is
`docs/data/ubc_ocean_masked_holdout_ids.csv`; the sealed test shard itself still
needs to be generated. The
`kaggle/generate_dataset_Classification_With_Masks` notebook is the current
test-set-generation starting point, but as committed it still writes train/valid
splits rather than `test` files. User-confirmed split intent: train/validation
uses WSIs without supplemental masks; WSIs with non-exhaustive supplemental masks
are reserved for the held-out autoencoder test set and later supervised
experiments.
A clean-context adversarial review pass on 2026-06-10 checked the agentic
workflow and Kaggle data contract. It found and fixed stale onboarding references
to the Kaggle mask notebook, missing preflight coverage for the masked-holdout
CSV, loose Kaggle spec-index readiness checks, and an ambiguity in the patch CSV
metadata schema. The new holdout CSV is tracked so repo preflight can verify it as
tracked.
An additional clean-context adversarial coding-readiness audit on 2026-06-11
found that the repo is not yet safe for broad spec 0001 implementation. It is
ready for a spec-relock/scaffolding decision pass only. The audit added or
confirmed blockers for count verification of the ResNet-like residual schedule
with branch-local non-naive ResNet-D/anti-aliased-style projection/downsample
primitives, strict-quality debt route, package/import policy, JSON config
policy, fixed validation/tiny-overfit selector generation, CPU compile/float16
smoke constraints, baseline rotated/latent artifact semantics, and
local-vs-Kaggle acceptance separation. The analytic Conv2d baseline count target
is now recorded in spec 0001, and the local instantiated topology-count slice now
generates and verifies `benchmark/model_count.json` plus
`benchmark/model_inventory.csv`.
The 2026-06-12 focused check found that spec 0001 was still formally not locked
even for a scaffold unless a narrow exception was recorded. That exception is now
documented in `docs/specs/0001-translatable-normal-vae-baseline.md` and
`docs/specs/README.md`: `src/eqvae`, `configs/spec0001`, analytic
`model_count` schema output, and local synthetic benchmark schema output are
allowed as a scaffold/unblock slice only. The narrow local scaffold is now
`scaffold_schema_ready`, meaning its local JSON/CSV schema contracts pass with
`status = "schema_pass"` and `full_run_eligible = false`; it is not a Kaggle
runtime selection. Spec 0001 remains not locked for broad model, data,
corruption, training, evaluation, Kaggle, or paper-claim implementation.
Benchmark spec details were tightened on 2026-06-12 for runtime proof,
selected-runtime artifact hashes, stable runtime row IDs, dataloader
measurements, paired numerical checks, gate-health module/interval semantics,
tiny-overfit smoothing/hash/clamp gates, and explicit readiness labels
(`scaffold_schema_ready`, `local_benchmark_pretest_contract_ready`,
`local_benchmark_pretest_ready`,
`model_loss_train_step_contract_ready`,
`model_loss_train_step_ready`,
`benchmark_cli_implementation_ready`, `runtime_selected`). Local CPU/laptop
dataloader pre-tests may now write measured synthetic UBC-format evidence with
`status = "local_pass"`, but must keep `full_run_eligible = false` and cannot
be selected as Kaggle runtime evidence. A follow-up clean-context audit on
2026-06-12 found the
topology-count slice itself was under-authorized and that the inventory/hash/MAC
proof schema was too weak. Spec 0001 now has a narrow `topology_count_ready`
exception, a canonical residual-block table, corrected section-level MAC split,
canonical JSON config hashing, pass-mode config guards, explicit resampling MAC
formulas, and a stronger inventory schema with shapes, branch/order metadata,
trainability, count category, and MAC formula columns.
The follow-up fix pass on 2026-06-12 resolved the concrete topology-count
findings: thin configs using `source_config` now resolve into an effective
config before model validation; `model_count.json` records raw invoked/source
file hashes plus an effective canonical-config hash; model inventory rows are
built from live module attributes plus meta-forward input/output shapes and
observed execution order; uninventoried or banned leaf modules such as nearest
upsampling fail the count proof; and `GatedScalarActivation` now keeps
scalar-gate sigmoid arithmetic in FP32 while still accepting FP16 inputs for a
local dtype-path smoke.
The local uv environment is CPU-only for PyTorch. Strict Ruff settings are
canonical in `pyproject.toml`; do not add `ruff.toml`. The no-sync quality gate
verified Python 3.12, `torch==2.12.0+cpu`, and CUDA unavailable. Strict Ruff
autofixed 14 historical formatting issues in an earlier run. Empty `main.py`
was deleted on 2026-06-12. Historical exploratory `src/nn` remains on disk by
user decision as reference material, but it is now excluded from Ruff and
BasedPyright production scopes and must not be imported by `src/eqvae`.
Spec 0002 records this production-boundary decision: active Python quality
applies to `src/eqvae`, tests, and any future explicitly production-scoped
Python helpers. After updating `scripts/python_quality.sh` to run pytest with
`PYTHONPATH=src`, the full production-scope `./scripts/python_quality.sh` gate
passes.
A final focused clean-context adversarial review on 2026-06-11 found two
benchmark-unblock doc gaps: `benchmark/model_count.json` was required in prose
but missing from CLI/output acceptance, and Kaggle push acceptance did not
explicitly require accelerator metadata validation. Both are now fixed in spec
0001, and `scripts/kaggle_kernel.sh` rejects benchmark pushes unless metadata
uses `machine_shape = "NvidiaTeslaT4"` and the launcher contains
single-visible, dual-DDP, and wrong-accelerator validation hooks.
Kaggle API read-only preflight on 2026-06-11:
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passed OAuth
token generation, kernel list/status/logs, and dataset file listing for
`maximusshtefan/patches-pre-shuffled-ubc-ocean`. It warned that
`kaggle quota -v` and `kaggle kernels files maximusshtefan/non-eq-vae -v` return
Kaggle's authentication-required message despite OAuth token generation working.
Spec 0001 now requires the API preflight before remote benchmark push, with a
Kaggle web UI quota check if the CLI quota endpoint still warns.
The user visually confirmed the Kaggle web UI quota on 2026-06-11: phone
verification is complete, identity verification is not complete, and Kaggle GPU
quota shows `00:07 / 30 hrs` used. This is enough to proceed with benchmark
implementation planning; before an actual remote benchmark push, rerun
`api-check` and confirm the UI still shows available GPU quota.

Historical selected-runtime note, superseded by the current top-level next
action: use
`runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json` as the
fallback selected runtime for selected-runtime debug/resume/tiny-overfit before
any long training launch. The compact v6 efficiency-selection follow-up tested
only the `amp_fp16_scalar_gate_relaxed` policy against v5 and did not replace
v5: the relaxed row was slower and no selected runtime was written. Do not push
another Kaggle job without explicit user approval and the guard variables. The
separate selected-runtime
benchmark/debug slice is encoded in
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` as
`runtime_matrix.selection_benchmark_slice.name =
"v8_shortlist_eager_amp_then_dual_gate"`. Remote v8 is carry-forward shortlist
evidence only, not selection evidence: it may seed eager single-visible-T4
bs8/bs12 FP32 `compile_none` branchless/indexed confirmation, with bs4 as a
fallback, then AMP follow-up only for confirmed eager rows. The benchmark must
write `benchmark/selected_runtime.json` only after its own full linked proof
passes and hash-links its artifacts; it must remain blocked while real dual-T4
train-step timing is missing, failed, or skipped. Dual-T4 timing is required,
not optional: the selection benchmark must emit dual-T4 DDP train-step rows for
per-device bs4, bs8, and bs12 FP32 eager branchless/indexed candidates, prove
two visible T4s, `world_size = 2`, `nproc_per_node = 2`, per-rank device
assignment, linked dataloader/numerical/corruption/gate evidence, and global
throughput projection. Runtime-selection v1 proved the dual-T4 timing gate but
blocked selected-runtime writing on linked-proof false negatives; v2 fixed the
wrapper artifact allow-list and those first false negatives, completed on
Kaggle, and still blocked selection because single-visible indexed candidates
lacked candidate-bound gate-health rows. The v3 local fix binds those rows from
the branchless reference gate evidence only for already-passing indexed
runtime rows. Runtime-selection v3 completed and selected the dual-T4 bs12 FP32
eager indexed-mask row. The
local v8 evidence-plumbing fix was committed as `614cd95`,
pushed as Kaggle version 8 after explicit approval, completed,
downloaded to `runs/kaggle/real_data_runtime_pretest_v8`, and inspected. It
fixed the v7 quantile blocker and produced six capped-pretest passing eager
rows, but `runtime_proof.status = pretest_incomplete`, `selection_ready =
false`, `full_run_eligible = false`, no `benchmark/selected_runtime.json`
exists, dual-T4 train-step measurement remains pending, and compiled rows
remain diagnostic/ineligible. No GitHub push or Overleaf command was run. For
paper work, continue from
`docs/specs/0004-sipaim-paper-scaffold.md` and lock the downstream WSI
classifier protocol: frozen encoder versus fine-tuning, posterior-mean
embedding extraction, patch-to-WSI aggregation, classifier capacity,
labels/splits, seeds, and metric table. Remote v5/v6/v7/v8 capped-pretest outputs
remain non-promotable selected-runtime evidence. Three clean-context
adversarial reviews found and then rechecked blockers around dataloader
thresholds, eager-reference numerical checks, fixed-batch coverage,
gate-health thresholds, child-process Dynamo counter availability, recompile
counter accounting, DDP rank/device binding, and overclaiming. The follow-up
reviews found no high blocker for a capped, non-promotable v6 evidence attempt.
Remaining v5 limits: eager rows may become eligible only if all canonical linked
proofs pass; compiled `model_forward` rows are deliberately diagnostic-only and
ineligible until full compile-settle coverage is implemented for clean
validation, DDP rank paths, final partial batches, and mask cardinalities
0/1/many/all. Remote v4
of the capped real-data runtime pretest passed the first real-data proof lane:
identity, row counts, WSI/holdout split contract, CRC, locked windows, and
clean validation loader. Artifacts are downloaded under
`runs/kaggle/real_data_runtime_pretest_v4`. Do not treat v4 as runtime
selection: `runtime_proof.status = "pretest_incomplete"`,
`linked_evidence_status = "skipped_unsupported"`, `selection_ready = false`,
`selected_runtime_written = false`, and `eligible_pass_row_count = 0`. Remote
writes still require explicit approval plus `KAGGLE_PUSH_CONFIRMED=1`; source
attachments also require `KAGGLE_FULL_DATASET_CONFIRMED=1`, and remote
reads/downloads require explicit approval plus `KAGGLE_REMOTE_CONFIRMED=1`.
The 2026-06-19 proof-lane implementation now records real-data/local identity,
SHA256 file
hashes, full-payload CRC32 validation, row-count proof, split
WSI/holdout-overlap contract proof, exact fixed spread-window proof, and a clean
validation loader/collate/normalization proof. The follow-up linked-evidence
implementation now records measured `model_forward` compile-settle/Dynamo
counters, a real dual-rank DDP launch probe when two T4s are visible,
candidate accelerator/batch dataloader throughput, same-batch eager-reference
numerical checks plus corruption checks for covered candidate paths, and
gate-health row links. Tiny local fixtures produce only `local_pass` or
`skipped_unsupported`; canonical real `pass` requires the
expected Kaggle dataset slug, exact 300000/30000 train/validation row counts,
exact 322/39 train/validation WSI counts, zero train/validation and
masked-holdout WSI overlap, 8,192/2,048 capped window totals, and the locked
train/validation spread windows. The clean-validation proof is loader-only and
explicitly does not claim corruption-RNG instrumentation; the corruption
equivalence proof also does not claim clean-validation RNG non-consumption. A
local runner/kernel/guard now exists at
`kaggle/kernels/real_data_runtime_pretest`, with exact T4 metadata, the exact
patch dataset source, `KAGGLE_FULL_DATASET_CONFIRMED=1` push guard, embedded
payload verification, upload-simulation import proof, local wrong-accelerator
artifact proof, and explicit selected-runtime rejection. It still must not be
treated as runtime selection: eager timed rows are eligible only when canonical
real-data proof, real DDP proof, matching row-specific dataloader/numerical/
corruption/gate-health evidence, and zero graph-break/recompile counts pass.
Compiled `model_forward` timed rows must remain ineligible until the full
compile-settle coverage proof passes. The first pretest
attaches only
`maximusshtefan/patches-pre-shuffled-ubc-ocean`, uses 8,192 train and 2,048
validation patches from fixed spread windows, writes blocked claims, and must
not write `benchmark/selected_runtime.json`. The approved first real-data
train-step benchmark axis includes `compile_scope = none`, `model_forward`,
`model_loss`, and `train_step_no_optimizer`, crossed first with
`amp_off_fp32` and both `branchless_all` / `indexed_masked` corruption
strategies; AMP policies may be emitted only for the exact FP32 candidates whose
runtime, dataloader, numerical, corruption, gate-health, graph-break, and
recompile evidence passes. Compile settling is locked at 5 unmeasured steps
using `torch._dynamo.utils.counters` reset per row, and must exercise all
measured code paths, including clean validation, DDP rank paths, final partial
batch path, and mask cardinalities 0/1/many/all. The v4 repeated shortlist
artifacts live under
`runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark`. Top v4 synthetic
repeat recommendations were
`dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all`
(`estimated_epoch_minutes = 1.312643`),
`single_visible_t4__bs4__amp_off_fp32__compile_off__branchless_all`
(`1.964479`), and
`single_visible_t4__bs32__amp_off_fp32__compile_off__branchless_all`
(`2.043706`). These rows seed the capped real-data pretest together with
sentinel rows; they are provenance, not candidate identity. The v4
recommendation artifact marks the repeat-shortlist gate as completed, but the
artifacts remain non-promotable loader/H2D screening evidence only; real runtime
selection still requires real-data benchmarking and the selected-runtime
debug/full-run gates.
The synthetic setup-smoke remote test passed on Kaggle as version 1 of
`maximusshtefan/eqvae-setup-smoke`; downloaded ignored evidence is at
`runs/kaggle/setup_smoke/benchmark/kaggle_setup_smoke.json`. The setup artifact
records `status = "smoke_pass"`,
`status_scope = "non_promotable_setup_smoke"`,
`benchmark_kind = "synthetic_kaggle_setup_smoke"`, no dataset slug,
`data.origin = "synthetic_or_ephemeral_path"`,
`runtime.requires_cuda_t4 = false`, `train.applied_counts = [1, 1, 0]`, and
payload provenance for clean commit `3162bececdf40b5270b06654603f1a018d5ada05`.
The real-data source delivery is migrated locally to embedded single-file
packaging with an import-only upload-simulation test, but it must be used only
when intentionally testing Kaggle dataset attachment plus UBC shard resolution.
Any future push whose metadata has nonempty `dataset_sources`,
`competition_sources`, `kernel_sources`, or `model_sources` must include both
`KAGGLE_PUSH_CONFIRMED=1` and `KAGGLE_FULL_DATASET_CONFIRMED=1` after explicit
user acceptance of source attachment/setup cost. The real-data smoke guard also
requires the known patch dataset as the only source attachment. Do not attach
the real 60 GB+ dataset for setup-only or synthetic/random timing checks. Do not
start Overleaf work, broad real training/resume, runtime selection, or paper
claims. The data/metrics, selector/dataloader, and local benchmark pre-test
contracts are recorded in spec 0001, and the local benchmark pre-test is now
`local_benchmark_pretest_ready`. The local implementation already exists under
`src/eqvae/data`, `src/eqvae/metrics`, `src/eqvae/benchmarking`, and
`src/eqvae/cli`: deterministic synthetic UBC-format patch shards, exact
`<8sIQiiii3s25x` header/CRC parsing, canonical
`file_index`/`row_index`/`sample_id` semantics, split-validation status values
that distinguish `synthetic_pass` from real-data `pass`, repo-owned FP32
MAE/MSE/PSNR/full SSIM, deterministic data-root resolution, read-only mmap
tensor-only loaders, fixed-selector schema/generation/validation, and a measured
local dataloader pre-test writer behind `eqvae.cli.benchmark_runtime
--dataloader-pretest`. In the managed sandbox, the checked-in debug pre-test
measured `num_workers = 0` train/validation rows as `local_pass`, while
worker-positive rows were explicit `fail` rows with
`failure_kind = "local_worker_transport_unavailable"`. Rerunning the same
command outside the sandbox on 2026-06-12 measured all configured local CPU
candidates successfully with `status = "local_pass"`. Focused Ruff,
BasedPyright, and pytest checks pass for the active local pre-test slice.
The model/loss train-step slice is now `model_loss_train_step_ready`. The VAE
forward API exposes explicit `eps`, raw and clamped `logvar`, sampled `z`, and
`logvar_clamp_count`; sampling and KL use clamped `logvar`; the repo-owned loss
uses exact global L1, per-image-mean SSIM loss, and global KL reductions; the
local pre-test uses identity-clean input with
`corruption_strategy = "identity_clean_no_corruption"` until the corruption
slice exists; semantic AdamW groups are implemented; and the dedicated
`--model-loss-train-step` CLI writes non-promotable
`benchmark/model_loss_train_step.json` with `status = "local_pass"` and
`full_run_eligible = false` without writing `benchmark/selected_runtime.json`.
The first beta-zero, zero-head smoke intentionally proves the final RGB head
forward/update path only; the artifact records nonzero grad/update tensor counts
and `first_step_update_scope = "zero_head_final_rgb_head_smoke"` so it is not
over-interpreted as full hidden-stack connectivity.
FSQ/data-generation inspection on 2026-06-12 confirmed that the historical
pipeline writes 64-byte-header CHW `uint8` UBC shards; final train is globally
shuffled and drops `idx`; validation keeps `idx`; and the old FSQ mmap loader is
useful for binary mechanics but not sufficient because it returns only tensors
and omits selector/sample provenance. The scaffold exists, `model_count` is now
`topology_count_ready`, `data_metrics_ready` and
`fixed_selectors_dataloader_ready` are local verified slices, and local runtime
schema smoke or local dataloader pre-test evidence still cannot be selected as a
Kaggle runtime. The HED/stain corruption local slice is now implemented with
repo-owned PyTorch HED/RGB conversion, stateless semantic RNG, config fields,
focused tests, and `/tmp/eqvae-local-stain-corruptor-qa/benchmark/stain_corruptor_qa.json`
evidence. Training integration, branchless/indexed runtime corruption checks,
and fixed real 25-patch visual QA are still pending. The remaining
implementation-relock blockers still include the future `SO(2)` count ceiling,
Kaggle T4 metadata validation/runtime proof, real fixed selector generation from
real Kaggle shards, remaining artifact protocol, and final adversarial spec
review. Kaggle
metadata was verified on
2026-06-11 by pulling `maximusshtefan/non-eq-vae`: the T4 benchmark
`machine_shape` value is `"NvidiaTeslaT4"`, and dual-DDP rows must still prove
two visible T4 devices at runtime. The branch-local
non-naive ResNet-D/anti-aliased-style residual projection/downsample policy is
now explicit in spec 0001, and the spec 0001 downsample operator is locked as a
repo-owned 5x5 separable binomial low-pass followed by decimation. Resize/area
downsampling is only a later fallback if the binomial operator fails a future
SO(2) stage-transition test. Normalization is now real-run default: standard
GroupNorm in the Conv2d baseline, repo-owned field-aware norm in the SO(2)
model, scalar bias allowed, vector additive bias forbidden. Activation uses
sigmoid gates with
learned `a` and `b`, no `gamma`, and a required gate-health benchmark before
full training; model padding defaults to zero padding with border-cropped
equivariance diagnostics. Comparable means the SO(2) model should
use less than or equal learned parameters than the Conv2d baseline and must not
blow the Kaggle memory budget. The SO(2) first-run kernel basis is now locked:
Gaussian radial shells times real angular harmonics with zero support at the
kernel center for spatial angular frequencies `m > 0`; Bessel/Fourier-Bessel is
kept only as a future fallback after disk-radius and sampled-zero risks are
locked. Also resolve the package/import policy and final clean-context
adversarial spec review.
After local verification of data/model/train/runtime code, run the short Kaggle
runtime benchmark to choose single/dual T4,
per-device/global batch, AMP, compile, precision-policy, and corruption-strategy
settings before the first 10-epoch full run. The full run stays blocked until
dataloader throughput, paired numerical checks, gate-health telemetry,
selected-runtime debug, checkpoint resume, and tiny-overfit summaries pass.

Local scaffold status from 2026-06-12:

- Added `src/eqvae` package scaffold with `eqvae.cli.model_count` and
  `eqvae.cli.benchmark_runtime`.
- Added `configs/spec0001` JSON scaffold. Kaggle/debug/full-run configs now
  source shared model/objective/corruption contract fields from
  `non_eq_vae_model_base.json`, not from the local CPU debug config, so
  local-only `benchmark`, `dataloader_pretest`, and CPU runtime fields cannot
  leak into resolved Kaggle configs. Fixed selector configs are explicitly
  placeholders with `status = "requires_real_data_generation"` and empty
  selectors until real Kaggle CSV access exists.
- Added `eqvae.models.non_equivariant_vae`,
  `eqvae.models.activations.GatedScalarActivation`, and fixed fieldwise
  downsample/upsample modules needed for the count slice.
- `eqvae.cli.model_count` now instantiates the locked Conv2d topology and writes
  `benchmark/model_count.json` with `status = "pass"`,
  `benchmark_kind = "implementation_model_count"`, `benchmark_source =
  "instantiated_model"`, `full_run_eligible = true`, layered
  `source_config` resolution, raw invoked/source file hashes, an effective
  canonical-config hash, exact observed-vs-expected counts, zero-RGB-head proof,
  stricter banned-operation proof, and `matches_spec_target = true`.
  `full_run_eligible = true` here means eligible only as a model-count
  dependency in the benchmark artifact graph; it is not a runtime/training/Kaggle
  unlock.
- `eqvae.cli.model_count` also writes `benchmark/model_inventory.csv` with
  129 rows observed from the instantiated topology by meta-forward hooks:
  43 learned convolutions, 40 GroupNorm modules, 34 learned gates, and 12 fixed
  resampling ops. The CSV includes observed input/output shapes and forward
  order.
- `eqvae.cli.benchmark_runtime` writes local CPU synthetic schema artifacts under
  `benchmark/` and `metrics/`, with `status = "schema_pass"` and
  `full_run_eligible = false` so they cannot be mistaken for Kaggle runtime
  selection. The JSON artifacts also carry
  `benchmark_source = "local_synthetic_schema_smoke"`, and the
  `dataloader_matrix.csv` schema now includes benchmark identity,
  `machine_shape`, `non_blocking_h2d`, and empty H2D timing fields for local CPU
  rows.
- `eqvae.data.roots` resolves explicit and deterministic `auto` UBC data roots;
  `eqvae.data.dataloaders.PatchTensorDataset` provides the read-only mmap
  tensor-only hot path; `eqvae.data.fixed_selectors` generates and validates
  `spec0001.fixed_selector.v1` documents; and
  `eqvae.cli.select_fixed_patches` writes selector artifacts without touching
  Kaggle remote.
- Fixed selector placeholder configs now use `spec0001.fixed_selector.v1`,
  remain `requires_real_data_generation`, and document that canonical overwrites
  require both `--validate-crc` and `--allow-tracked-config-overwrite`.
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` now records
  `dataloader.hot_path = "mmap_tensor_only_v1"`, warmup/measured batch counts,
  and the default Kaggle `dataloader.candidates` matrix over `num_workers`,
  `prefetch_factor`, `pin_memory`, `persistent_workers`, and
  `non_blocking_h2d`.
- `configs/spec0001/non_eq_vae_debug_cpu.json` now records the tiny local CPU
  synthetic dataloader pre-test matrix. That matrix is implementation guidance
  for the next slice; current schema-smoke artifacts still use `schema_pass`.
- Added `tests/test_spec0001_benchmark_scaffold.py`.
- Deleted empty `main.py`; removed `pytorch-msssim` from `pyproject.toml` and
  `uv.lock`. Historical `src/nn` remains as excluded reference material and is
  forbidden as an import source for active `src/eqvae` code.

Kaggle-specific handoff: Kaggle authentication is a user-local secret and must
remain permission-gated. Do not push the default real-data
`kaggle/kernels/non_eq_vae_debug` kernel unless intentionally testing the 60 GB+
patch dataset attachment plus UBC shard resolution, with explicit user approval
and both `KAGGLE_PUSH_CONFIRMED=1` and `KAGGLE_FULL_DATASET_CONFIRMED=1`. The
next Kaggle implementation target is local-only work for
`kaggle/kernels/synthetic_timing`: no Kaggle source attachments, generated
UBC-format shards under `/kaggle/working`, T4 GPU metadata, a dedicated
`KAGGLE_SYNTHETIC_TIMING_READY = True` guard, upload-simulation proof, and no
remote write until the user explicitly approves it.

## Settled Decisions

- The active symmetry target is continuous `SO(2)`.
- The comparable baseline must be a normal VAE, not the previous FSQ
  autoencoder.
- The paper source of record lives in `paper/sipaim2026`.
- The tracked advisor-facing PDF is `paper/sipaim2026/sipaim2026.pdf`.
- Overleaf sync must use the safe subtree workflow.
- GitHub issue images are requirements evidence and must be inspected before
  translating issue requests into deliverables.
- Adversarial clean-context subagent reviews should be used before substantial
  workflow, architecture, evaluation, or paper-claim changes when tooling is
  available.
- Before any Kaggle remote push, run every cheap local check that can catch
  writer, artifact, payload, or trivial runtime errors. For runtime-selection
  pushes, the mandatory local semantic preflight is
  `./scripts/kaggle_kernel.sh preflight-runtime-selection`; it must run before
  asking for or using remote-write approval.

Decision notes live in `docs/decisions/`.
The review process lives in `docs/agentic_review_workflow.md`.

## No Longer Active

- Old conference-deadline planning is not part of the current route.
- Discrete rotation-group implementation work is not part of the current route.
- The thesis repo is not the active editing target for this phase.

## Next Concrete Steps

1. Runtime-selection v5 is the locked selected-runtime answer. It selected AMP
   conservative dual-T4 bs12 indexed-mask at `27.381321` samples/sec and about
   30.4 projected hours for 10 epochs; v6 did not beat it.
2. Selected-runtime debug/tiny v5 completed, downloaded to
   `runs/kaggle/selected_runtime_debug_v5`, and passed strict output
   verification. It is the proven bounded gate, not the long-run launcher.
3. Full-run v1 was pushed with explicit approval, later returned
   `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`, and was downloaded to
   `runs/kaggle/selected_runtime_full_v1`. The download has checkpoints through
   `step_043750.pt` and `best_model.pt` but no `metrics/` or `benchmark/`, so
   strict full-output verification fails and v1 is not paper-promotable.
4. The local runner now has two-phase interval artifact flushing for future
   full launches. The next implementation decision is whether to restart the
   full run from scratch for complete metrics/curves or add a separately
   reviewed checkpoint-only-prefix continuation policy for `step_043750.pt` that
   remains non-paper-promotable for the missing first 43750 metric rows.
5. Do not run any Kaggle status/output/upload/push without exact explicit
   approval plus the required confirmation environment variables. Any future
   full push must use `kaggle/kernels/selected_runtime_full`, never
   `kaggle/kernels/selected_runtime_debug`.
6. Continue the shared evaluation harness, future `SO(2)` count ceiling, and
   steerable model work only after the current full-baseline recovery decision
   is settled.
7. Spec 0010 (`fixed25-equivariance-artifact-protocol`) is drafted and
   adversarially reviewed (2026-07-01); status draft, spec-only. It is the P0
   FU-040 artifact protocol and is BLOCKED on explicit user approval before any
   implementation, runner, config, test, or verifier code. Paper-promotable
   fixed-25 artifacts also need the real selector (real Kaggle data), coupling
   this to the v1-continuation decision in step 4.

## Current Blockers

- Full-run v1 is canceled and incomplete. Its checkpoint prefix is resumable in
  model-state terms, but the first 43750 metric rows are unrecoverable, so a
  naive resume would produce incomplete curves. Choose restart-from-scratch or a
  separately specified checkpoint-only-prefix continuation before any new remote
  full-run push.
- Future full launches must keep the two-phase interval flush and rank-0
  failure broadcast intact, and must pass local tests/preflights before any
  approval request.
- The exact held-out masked-WSI test shard must be generated, uploaded, and
  locked before final paper claims. The 152-image candidate pool is documented in
  `docs/data/ubc_ocean_masked_holdout_ids.csv`, and train/validation are
  available in the confirmed pre-shuffled patch dataset. Supplemental masks are
  non-exhaustive, so test generation and later supervised experiments must not
  treat unmasked regions as exhaustive negative labels.
- Strict Python quality now has a production boundary: `src/eqvae` and tests are
  strict, while historical `src/nn` is excluded as reference-only. New work must
  not add debt or import from `src.nn`.

## Latest Verification

2026-06-22 selected-runtime debug/resume local contract push:

- Commit `f9f0344` (`Add selected runtime debug proof runner`) was pushed to
  GitHub `origin/main`.
- Before the push, focused tests passed:
  `PYTHONPATH=src .venv/bin/pytest tests/test_train_cli.py tests/test_runtime_selection_benchmark.py tests/test_spec0001_benchmark_scaffold.py -q`
  (`58 passed`).
- Full quality passed: `./scripts/python_quality.sh` (`177 passed`, `0` type
  errors).
- Runtime-selection local preflight passed:
  `./scripts/kaggle_kernel.sh preflight-runtime-selection` (`35 passed`).
- Direct local synthetic v5 debug, resume, and tiny-overfit commands passed with
  `full_run_eligible = false`; the checkpoint schema is
  `spec0001.checkpoint.v4` and stores selected-runtime config hash, row id,
  policy id, and named Torch `Generator` state.
- No Kaggle/GitHub/Overleaf remote action remains pending. Kaggle was not pushed
  because no real selected-runtime debug/tiny kernel target is wired yet.

2026-06-12 topology-count implementation verification and hardening:

- `./scripts/agent_preflight.sh` passed after the topology-count implementation
  and handoff updates; it noted only the expected dirty worktree.
- `.venv/bin/ruff format src/eqvae tests/test_spec0001_benchmark_scaffold.py tests/__init__.py`
  left 22 files unchanged.
- `.venv/bin/ruff check src/eqvae tests/test_spec0001_benchmark_scaffold.py tests/__init__.py`
  passed.
- `.venv/bin/basedpyright src/eqvae tests/test_spec0001_benchmark_scaffold.py tests/__init__.py`
  passed with 0 errors.
- `env PYTHONPATH=src .venv/bin/pytest tests/test_spec0001_benchmark_scaffold.py`
  passed with 7 tests, including layered-config model count, source-config
  resolution from a non-repo cwd, raw config-hash checks, banned nearest
  upsample rejection, extra countable leaf-module rejection, and FP16 input
  smoke for the scalar gate.
- `env PYTHONPATH=src .venv/bin/python -m eqvae.cli.model_count ...` passed and
  wrote `/tmp/eqvae-model-count-final/benchmark/model_count.json` plus
  `/tmp/eqvae-model-count-final/benchmark/model_inventory.csv`; the JSON had
  `status = "pass"` and the inventory had 129 data rows.
- `env PYTHONPATH=src .venv/bin/python -m eqvae.cli.model_count --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json ...`
  passed locally and wrote `/tmp/eqvae-layered-model-count/benchmark/model_count.json`
  with `config_resolution = "source_config_deep_merge_v1"`,
  distinct raw invoked/effective hashes, `inventory_matches_expected = true`,
  `forward_order_verified = true`, and `inventory_mismatch_count = 0`. A
  regression test also verifies that an absolute invoked config resolves its
  repo-root-style `source_config` correctly when the process cwd is elsewhere.
- Clean-context adversarial subagents reviewed code, docs, and quality policy.
  Their findings were fixed in this slice: raw invoked/source config hashes now
  use file bytes, `source_config` resolution is not repo-cwd-dependent,
  milestone/status text was refreshed, and historical `src/nn` quality debt was
  documented. The later `data_metrics_ready` slice is now separately
  implemented and recorded above.
- `env PYTHONPATH=src .venv/bin/python -m eqvae.cli.benchmark_runtime ...`
  passed and wrote `/tmp/eqvae-runtime-final`; its `model_count.json` had
  `status = "pass"` while `selected_runtime.json` remained `status =
  "schema_pass"` and `full_run_eligible = false`.
- `rg -n "status|benchmark_kind|benchmark_source|full_run_eligible|model_config_hash_source|actual_implementation|matches_spec_target" ...`
  verified the current `/tmp` artifacts carry the expected pass/schema split.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock` was run with approved escalation because
  offline lock refresh could not resolve uncached packages; it removed
  `pytorch-msssim v1.0.0` from `uv.lock`.
- Earlier `./scripts/python_quality.sh` runs failed on retained historical
  `src/nn` before the production-boundary decision. That reference tree now
  remains on disk but is excluded from production Ruff/BasedPyright scopes by
  spec 0002, and active code must not import it.
- The next blocking choices before final paper claims are the exact sealed
  masked-WSI test-shard artifact, upload slug, and mount-path verification.
  The next blocking choices before the steerable model are the latent
  field/statistics policy for nontrivial `SO(2)` latents and any normalization
  ablation.

2026-06-12 `data_metrics_ready` verification and review:

- Focused active-package checks pass after the adversarial fix pass:
  `.venv/bin/ruff format src/eqvae ...` left the active data/metrics and
  scaffold test files formatted; `.venv/bin/ruff check src/eqvae ...` passed;
  `.venv/bin/basedpyright src/eqvae ...` passed with 0 errors; and
  `PYTHONPATH=src .venv/bin/pytest tests/test_patch_shards.py tests/test_split_validation.py tests/test_reconstruction_metrics.py tests/test_spec0001_benchmark_scaffold.py`
  passed 27 tests.
- `./scripts/agent_preflight.sh` passed after the `data_metrics_ready` handoff
  updates; it noted only the expected dirty worktree.
- Code review findings were fixed: real split `pass` now requires exact train
  and validation patch counts, WSI counts, a nonempty masked-holdout ID list,
  and non-TMA provenance; PSNR rejects non-image-domain inputs; blank `idx`
  values are rejected when the `idx` column exists; and metric validation now
  fails early on device mismatch or nonpositive C/H/W dimensions.
- Docs/spec review findings were fixed: spec 0001's readiness header includes
  `data_metrics_ready`, later spec sections describe the slice as locally
  verified rather than pending, the focused data/metrics verification commands
  are listed, `docs/behavior_inventory_kaggle.md` no longer asks for the
  already-implemented topology-count artifact, and `full_run_eligible = true`
  on `benchmark/model_count.json` is documented as model-count dependency
  eligibility only.
- `data_metrics_ready` remains a local slice. It does not unlock corruption,
  training, fixed real selector generation, Kaggle payload work, Kaggle remote
  execution, or paper claims.

2026-06-12 `fixed_selectors_dataloader_ready` implementation and review:

- The selector/dataloader pre-code blockers are resolved in spec 0001:
  selector schema is audit-stable `spec0001.fixed_selector.v1`; generated split
  names are canonical `train`/`validation` with `valid`/`val` only as input
  aliases; canonical selector overwrites require explicit overwrite plus full
  CRC validation; `data_root = "auto"` is env/Kaggle/repo-root only and not CWD
  dependent; and validation recomputes the deterministic selector policy rather
  than trusting internally consistent JSON.
- Implemented `src/eqvae/data/roots.py`, `src/eqvae/data/dataloaders.py`,
  `src/eqvae/data/fixed_selectors.py`, and
  `src/eqvae/cli/select_fixed_patches.py`. The hot path remains tensor-only
  read-only mmap; selector/provenance lives in selector JSON and records.
- Focused checks passed:
  `.venv/bin/ruff check src/eqvae/data/roots.py src/eqvae/data/dataloaders.py src/eqvae/data/fixed_selectors.py src/eqvae/cli/select_fixed_patches.py src/eqvae/data/__init__.py tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py`,
  `.venv/bin/basedpyright ...` on the same files with 0 errors, and
  `PYTHONPATH=src .venv/bin/pytest tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py`
  with 16 tests. The broader active-package check also passed with 43 tests.
- Full `./scripts/python_quality.sh` now passes for the production scope: Ruff
  format/check, 43 pytest tests with `PYTHONPATH=src`, and BasedPyright all
  completed successfully. Historical `src/nn` remains excluded reference-only
  code.
  `./scripts/agent_preflight.sh` passed and noted only the expected dirty
  worktree.

2026-06-12 local benchmark pre-test contract lock and adversarial review:

- Spec 0001 now has a narrow `local_benchmark_pretest_contract_ready` state.
  It allows the next local slice to measure the FSQ-derived read-only mmap
  tensor-only dataloader on tiny synthetic UBC-format shards and write
  `benchmark/dataloader_matrix.csv` rows with `status = "local_pass"`,
  `benchmark_kind = "local_synthetic_pretest"`, `benchmark_source =
  "local_cpu_synthetic_pretest"`, `accelerator_mode = "local_cpu"`,
  `machine_shape = "local_cpu"`, and `full_run_eligible = false`.
- `configs/spec0001/non_eq_vae_model_base.json` is now the shared model-only
  base for Kaggle/debug/full-run configs. This fixes the adversarial finding
  that `source_config` deep merge could otherwise carry local CPU `runtime`,
  `benchmark`, or `dataloader_pretest` fields into resolved Kaggle configs.
  A regression test verifies the resolved Kaggle runtime benchmark config has
  no local-only `benchmark`, `dataloader_pretest`, or CPU `runtime` keys.
- Local schema-smoke CSV rows now carry negative provenance consistently:
  `benchmark_kind`, `benchmark_source`, `full_run_eligible`,
  `accelerator_mode`, and `machine_shape` are present on runtime, dataloader,
  numerical-check, and gate-health rows. Local linked safety statuses are
  `schema_pass`, not real-data `pass`.
- The local CPU pre-test candidate contract is locked in spec/config:
  `num_workers = [0, 1]`, no pinned memory, no non-blocking H2D, and blank H2D
  timings locally; Kaggle benchmark candidates vary `num_workers`,
  `prefetch_factor`, `pin_memory`, `persistent_workers`, and
  `non_blocking_h2d` with real GPU H2D timing required.
- Clean-context adversarial subagents reviewed this slice. Findings were fixed:
  Kaggle config inheritance no longer leaks local CPU fields, local artifacts
  carry non-promotable provenance, readiness naming distinguishes
  `local_benchmark_pretest_contract_ready` from
  `local_benchmark_pretest_ready`, and the stale model-count example path now
  points at `non_eq_vae_model_base.json`.
- Verification after fixes: JSON validation passed for the new model base,
  debug CPU, and Kaggle runtime benchmark configs; a direct resolved-config
  check showed no local-only fields in the Kaggle runtime config; focused
  scaffold tests passed with 8 tests; full `./scripts/python_quality.sh`
  passed with Ruff, 44 pytest tests, and BasedPyright; and
  `./scripts/agent_preflight.sh` passed with only the expected dirty worktree
  note.

2026-06-12 local benchmark pre-test implementation:

- Implemented `src/eqvae/benchmarking/dataloader_pretest.py` and wired
  `eqvae.cli.benchmark_runtime --dataloader-pretest`. The writer creates tiny
  synthetic UBC-format train/validation shards under the output directory,
  measures the existing read-only mmap tensor-only `PatchTensorDataset` through
  configured `DataLoader` candidates, and overwrites
  `benchmark/dataloader_matrix.csv` with local pre-test rows.
- Successful local pre-test rows use `status = "local_pass"` and keep
  `benchmark_kind = "local_synthetic_pretest"`, `benchmark_source =
  "local_cpu_synthetic_pretest"`, `accelerator_mode = "local_cpu"`,
  `machine_shape = "local_cpu"`, and `full_run_eligible = false`.
  Host-to-device, trainer throughput, and data-wait fields are blank locally.
- Candidate failures are still emitted as rows with `status = "fail"` and a
  deterministic `failure_kind`. This matters in the managed sandbox: PyTorch
  multiprocessing tensor transport is unavailable there, so worker-positive
  candidates are recorded as `local_worker_transport_unavailable` instead of
  hanging or printing worker tracebacks.
- Verification: focused tests passed with
  `PYTHONPATH=src .venv/bin/pytest tests/test_dataloader_pretest.py tests/test_spec0001_benchmark_scaffold.py`;
  the focused tests now include a deterministic worker-transport failure-row
  regression; focused BasedPyright passed on the new module, CLI, and test; the
  checked-in debug CLI pre-test command wrote
  `/tmp/eqvae-local-dataloader-pretest-clean` in the managed sandbox with
  train/validation `num_workers = 0` rows as `local_pass` and worker-1 rows as
  explicit non-promotable failures; the same command run outside the sandbox
  wrote `/tmp/eqvae-local-dataloader-pretest-unsandboxed` with all configured
  local CPU candidates as `local_pass`; full `./scripts/python_quality.sh`
  passed at that point with 46 tests and 0 BasedPyright errors; and
  `./scripts/agent_preflight.sh` passed with only the expected dirty worktree
  note.

2026-06-12 model/loss train-step implementation:

- Implemented the narrow `model_loss_train_step_ready` slice under
  `src/eqvae`: `NonEquivariantVAE.forward()` now returns explicit `eps`, raw
  `logvar`, `logvar_clamped`, sampled `z`, and `logvar_clamp_count`; sampling
  and KL use clamped `logvar`; `eqvae.losses.vae` implements the locked
  `L1 + 0.1 * (1 - SSIM) + beta * KL` reductions and beta schedule; and
  `eqvae.training` provides semantic AdamW groups plus one identity-clean
  train-step helper.
- `eqvae.cli.benchmark_runtime --model-loss-train-step` is a dedicated local
  mode. It writes `benchmark/model_count.json`,
  `benchmark/model_inventory.csv`, and
  `benchmark/model_loss_train_step.json`; it does not write
  `benchmark/selected_runtime.json`.
- The train-step writer validates the local-only config rail before writing:
  `benchmark_kind`, `benchmark_source`, `full_run_eligible = false`,
  `required_precision_policy = "amp_off_fp32"`, and
  `corruption_strategy = "identity_clean_no_corruption"`. It also validates
  `objective.logvar_clamp` against the implementation constants and checks the
  linked `model_count.json` effective config hash, `architecture_id`, and
  `topology_version`. Invalid rail configs fail before partial benchmark
  artifacts are written.
- The local artifact is non-promotable and explicit about the first-step proof:
  the checked-in debug command wrote `status = "local_pass"`,
  `full_run_eligible = false`, `nonfinite_count = 0`, zero-head proof `pass`,
  `torch_compile.status = "local_pass"`,
  `float16_smoke.status = "local_pass"`,
  `nonzero_grad_parameter_tensor_count = 2`,
  `nonzero_update_parameter_tensor_count = 2`,
  `trainable_parameter_tensor_count = 194`, and
  `first_step_update_scope = "zero_head_final_rgb_head_smoke"`. This first
  beta-zero, zero-head smoke proves the final RGB head forward/update path; it
  is not full hidden-stack connectivity evidence.
- Adversarial review findings were fixed: strict Ruff/BasedPyright failures,
  accidental `selected_runtime.json` writing in model/loss mode, weak local rail
  validation, graph-attached metric telemetry, schema-smoke row-status wording,
  over-broad gradient/update interpretation, config/implementation logvar-clamp
  drift risk, missing model-count architecture/topology self-description, and
  weak duplicate-parameter regression coverage.
- Verification passed:
  focused Ruff format/check for the touched model/loss/training/benchmark/CLI
  and test files; focused BasedPyright with 0 errors; focused pytest with
  16 hardening tests; the checked-in debug CLI command wrote
  `/tmp/eqvae-local-model-loss-train-step/benchmark/model_loss_train_step.json`
  plus model-count artifacts and no selected-runtime artifact; and full
  `./scripts/python_quality.sh` passed with 60 tests and 0 BasedPyright errors.
- 2026-06-13 HED/stain corruptor local QA slice: added
  `src/eqvae/corruption/stain.py`, `src/eqvae/benchmarking/stain_corruptor_qa.py`,
  `--stain-corruptor-qa`, expanded corruption config fields, and
  `tests/test_stain_corruptor.py`. The focused tests compare the Torch
  channel-first HED/RGB math against scikit-image 0.26.0, verify valid
  HED-manifold identity behavior, semantic stateless RNG, clean-validation RNG
  non-consumption, public `[-1, 1]` output range, metadata, and synthetic QA
  artifact writing. Verification passed: focused Ruff, focused BasedPyright,
  focused pytest with 12 tests, CLI artifact generation at
  `/tmp/eqvae-local-stain-corruptor-qa/benchmark/stain_corruptor_qa.json`
  with `status = "local_pass"`/`full_run_eligible = false`, and full
  `./scripts/python_quality.sh` with 72 tests and 0 BasedPyright errors.
- 2026-06-13 capped Kaggle smoke prep: added
  `src/eqvae/data/training_batches.py`, `src/eqvae/benchmarking/kaggle_smoke.py`,
  `src/eqvae/cli/kaggle_smoke.py`, a smoke-only
  `kaggle/kernels/non_eq_vae_debug/run.py`, and
  `tests/test_kaggle_smoke.py`. `PatchTensorDataset` remains tensor-only for
  throughput evidence; `PatchTrainingDataset` carries metadata for corruption
  RNG and metric/artifact provenance. `TrainStepRequest` now accepts optional
  `input_batch` so the model can consume `corrupt(x_clean)` while the loss
  targets `x_clean`. The Kaggle debug config is capped at three train steps and
  one clean-validation batch with `full_run_eligible = false`. Focused Ruff,
  focused BasedPyright, focused pytest with 2 tests, `bash -n
  scripts/kaggle_kernel.sh`, metadata JSON validation, `kaggle_kernel.sh
  validate`, `kaggle_kernel.sh build`, and full `./scripts/python_quality.sh`
  passed; the full Python gate then had 74 tests and 0 BasedPyright errors.
  Before the later adversarial hardening, the built
  `kaggle/kernels/non_eq_vae_debug/run.py` entrypoint also ran locally against
  tiny synthetic 256-pixel UBC-format shards and wrote
  `/tmp/eqvae-kaggle-smoke-entry-run/benchmark/kaggle_smoke.json` with
  `status = "smoke_pass"` and `full_run_eligible = false`. After hardening, the
  same real-data launcher correctly refuses local CPU execution because
  real-data Kaggle smoke evidence must prove visible T4 CUDA; local setup-only
  entrypoint proof now belongs in a separate synthetic no-dataset smoke.
  `./scripts/agent_preflight.sh` passed before handoff, noting only the
  expected dirty worktree.
- 2026-06-13 capped Kaggle smoke remote launch: read-only
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passed for
  auth, debug-kernel status/log access, and dataset file listing, with the
  expected quota/files endpoint warnings. The first push attempt was blocked
  locally before any remote write because the push guard lowercased the
  case-sensitive `machine_shape = "NvidiaTeslaT4"` value; `scripts/kaggle_kernel.sh`
  was fixed to lowercase only boolean-like metadata values and to make
  `api-check` derive the debug kernel ID from `kernel-metadata.json` instead of
  probing the old `maximusshtefan/non-eq-vae` kernel. After
  `./scripts/kaggle_kernel.sh build`, the approved remote write
  `KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push` succeeded with
  `Kernel version 1 successfully pushed`. A later read-only status check showed
  `KernelWorkerStatus.ERROR`; logs showed `ModuleNotFoundError: No module named
  'eqvae'`, so no benchmark artifact was produced.
- 2026-06-13 remote-smoke workflow correction: setup-only Kaggle tests should
  not attach `maximusshtefan/patches-pre-shuffled-ubc-ocean`, because Kaggle may
  spend a long time preparing the 60 GB+ dataset before a capped script starts.
  The follow-up setup smoke must use empty `dataset_sources`, generate tiny
  synthetic UBC-format shards under the output directory, write distinct
  non-promotable setup evidence, and leave the real-data source-attachment guard
  intact for real-data smoke/benchmark kernels.
- 2026-06-13 adversarial smoke hardening: clean-context subagent passes found
  that the first capped smoke could report `smoke_pass` without an applied
  corruption, did not hard-enforce the three-step/one-validation cap, could pass
  real-data smoke on CPU, did not seed model initialization for reproducible
  losses, had weak stale-payload protection, and left `smoke_pass` outside the
  explicit artifact status taxonomy. Local code now hard-fails uncapped smoke
  settings, requires real-data Kaggle smoke to run on visible T4 CUDA, seeds the
  model from `global_seed`, requires at least one applied corruption plus nonzero
  input-target delta and nonzero update counts for `smoke_pass`, records seeds,
  provenance, payload manifest, data-integrity status, corruption metadata
  summaries, and update telemetry, and tightens the push guard around target ID,
  caps, and payload freshness. The failed Kaggle version predates this
  hardening and produced no evidence. Focused Ruff, focused BasedPyright, focused
  `tests/test_kaggle_smoke.py`, `bash -n scripts/kaggle_kernel.sh`,
  `./scripts/kaggle_kernel.sh validate`, and full
  `./scripts/python_quality.sh` passed; the full Python gate now has 75 tests
  and 0 BasedPyright errors.
- 2026-06-17 synthetic setup-smoke packaging: added
  `scripts/build_kaggle_embedded_kernel.py`,
  `kaggle/kernels/setup_smoke`, setup-specific guards in
  `scripts/kaggle_kernel.sh`, setup artifact naming/validation in
  `src/eqvae/benchmarking/kaggle_smoke.py`, and
  `tests/test_kaggle_embedded_kernel.py`. The setup kernel has no dataset
  sources, no GPU, no internet, and a generated ignored `run.py` that embeds a
  zipped payload. Local build passed with `./scripts/kaggle_kernel.sh build
  kaggle/kernels/setup_smoke`; focused pytest passed for
  `tests/test_kaggle_smoke.py tests/test_kaggle_embedded_kernel.py` with 7
  tests. Full `./scripts/python_quality.sh` passed with 79 tests and 0
  BasedPyright errors. `./scripts/agent_preflight.sh` passed after staging the
  new tracked setup-smoke files.
- 2026-06-17 remote setup-smoke run: after committing
  `3162bec Add synthetic Kaggle setup smoke`, rebuilt the generated setup
  kernel from a clean HEAD and pushed only `kaggle/kernels/setup_smoke` with
  `KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
  kaggle/kernels/setup_smoke`. Kaggle returned `Kernel version 1 successfully
  pushed`, `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-setup`
  progressed from `KernelWorkerStatus.RUNNING` to `KernelWorkerStatus.COMPLETE`,
  and `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-setup`
  downloaded the artifact/logs into ignored `runs/kaggle/setup_smoke/`. The
  artifact passed as non-promotable setup evidence only: no dataset slug,
  synthetic/ephemeral data origin, CPU runtime, `requires_cuda_t4 = false`, 3
  train steps, 1 clean-validation batch, 2 deterministic applied corruptions,
  and clean embedded payload provenance for commit `3162bec`.
- 2026-06-17 real-data smoke embedded packaging migration in progress:
  replaced the tracked `kaggle/kernels/non_eq_vae_debug/run.py` source with
  tracked `run_template.py` plus ignored generated `run.py`, generalized
  `scripts/build_kaggle_embedded_kernel.py` with a ready-marker option, updated
  `scripts/kaggle_kernel.sh` so the default debug kernel builds/verifies an
  embedded payload and reads capped-smoke settings from the embedded zip, and
  added an import-only upload-simulation test for the real-data kernel. Focused
  `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/pytest
  tests/test_kaggle_embedded_kernel.py` passed with 2 tests. Full
  `./scripts/python_quality.sh` passed with 80 tests and 0 BasedPyright errors.
  `./scripts/agent_preflight.sh` passed after staging the generated-file
  tracking change.
- 2026-06-18 Kaggle source-attachment push guard: after an accidental real-data
  smoke push during remote-control/timing planning, `scripts/kaggle_kernel.sh`
  now rejects any push whose metadata has nonempty `dataset_sources`,
  `competition_sources`, `kernel_sources`, or `model_sources` unless the
  command includes `KAGGLE_FULL_DATASET_CONFIRMED=1` in addition to
  `KAGGLE_PUSH_CONFIRMED=1`. The real-data smoke guard also rejects extra
  competition/kernel/model sources and allows only
  `maximusshtefan/patches-pre-shuffled-ubc-ocean` as the dataset source.
- 2026-06-18 synthetic timing adversarial check: a four-agent swarm reviewed
  the synthetic timing contract from spec/evidence, Kaggle runtime, benchmark
  design, and data-format angles. The follow-up edits fixed the all-source
  attachment guard, stale handoff text, the 30-step non-wrapping eligibility
  contract, projected real epoch-time formula, structural-only pruning language,
  required `blocked_claims`, CRC/header/file/hash integrity proof,
  active collate/normalization proof, semantic-key/sample-id proof, and fresh
  child-process row isolation requirement. The synthetic timing evidence can
  screen/order candidates but cannot select the real runtime.
- Verification for the 2026-06-18 adversarial corrections: `bash -n
  scripts/kaggle_kernel.sh`, `git diff --check`, a no-network dummy-`kaggle`
  dry push guard test, and `./scripts/agent_preflight.sh` all passed. The dummy
  guard test confirmed the default real-data kernel fails before the Kaggle CLI
  is reached unless `KAGGLE_FULL_DATASET_CONFIRMED=1` is set for its source
  attachment. `./scripts/python_quality.sh` was not rerun in this slice because
  no production Python or test files changed.
- 2026-06-18 local synthetic timing implementation: added
  `kaggle/kernels/synthetic_timing`, `src/eqvae/benchmarking/synthetic_timing.py`,
  a dedicated `KAGGLE_SYNTHETIC_TIMING_READY = True` push guard branch, and
  focused tests for no-source metadata guarding, upload simulation, generated
  UBC shard parity, non-promotable artifacts, active loader proof, and
  non-wrapping eligibility. Initial local verification before adversarial fixes
  included
  `./scripts/kaggle_kernel.sh build kaggle/kernels/synthetic_timing`,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/synthetic_timing`, focused
  pytest for synthetic timing and embedded upload simulation, focused
  BasedPyright on touched Python, and the full Python quality gate. Remote
  Kaggle push/output was not run.
- 2026-06-18 synthetic timing implementation adversarial review: the first
  replacement three-agent swarm completed after an initial usage-limit failure.
  Guard/security found no blocking issue. Artifact/claims and data/evidence
  found real blockers: bootstrap failure wrote a fifth artifact, manifest status
  could say pass when all rows were `wrong_accelerator`, non-simulation
  `/kaggle/working` confinement was not enforced, blocked-claim tests were too
  implementation-coupled, DDP ranks used `cuda:0` instead of rank-local devices,
  successful DDP could leave rank scratch JSON under `benchmark/`, and
  recommendations labeled rows without ordering them. Follow-up fixes removed
  the bootstrap artifact path, made manifest/runtime/recommendation statuses
  agree, added exact blocked-claim validation, enforced `/kaggle/working` for
  non-simulation launcher runs, moved DDP scratch files to a temporary
  auto-cleaned directory, passed local rank into the measured CUDA device,
  added projection fields and timing-row summary evidence, and sorted
  recommendations by promotability and projected real epoch time. Verification:
  focused synthetic timing/upload-simulation pytest passed with 9 tests,
  focused BasedPyright passed with 0 errors, `./scripts/kaggle_kernel.sh build
  kaggle/kernels/synthetic_timing` refreshed the ignored generated launcher,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/synthetic_timing` passed,
  `git diff --check` passed, and `./scripts/python_quality.sh` passed with 87
  tests and 0 BasedPyright errors.
- 2026-06-18 GitHub handoff: committed the synthetic timing implementation as
  `c28632c Implement synthetic timing pretest` and pushed `main` to GitHub
  origin. The push also published the preceding contract commit `dcc375d Lock
  synthetic timing pretest contract`. After staging/committing the new files,
  `./scripts/agent_preflight.sh` passed cleanly,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/synthetic_timing` passed,
  and focused synthetic timing/upload-simulation pytest passed with 9 tests.
  Local `HEAD` and `origin/main` both resolve to
  `c28632cf074548c79e827bce5399dd68f6ecdf2d`. No Kaggle push/read/output and no
  Overleaf action were run.
- 2026-06-18 Kaggle synthetic timing remote v1: with explicit user approval,
  ran the read-only Kaggle API preflight, rebuilt the ignored generated
  `kaggle/kernels/synthetic_timing/run.py` against current `HEAD`, pushed
  `maximusshtefan/eqvae-synthetic-timing` with `KAGGLE_PUSH_CONFIRMED=1`, and
  downloaded completed output to ignored `runs/kaggle/synthetic_timing`. Status
  reached `KernelWorkerStatus.COMPLETE`. The benchmark directory contains
  exactly `synthetic_timing_manifest.json`,
  `synthetic_timing_runtime_proof.json`, `synthetic_timing_matrix.csv`, and
  `synthetic_timing_recommendations.json`; no `selected_runtime.json` exists.
  Manifest/runtime/recommendations all report `synthetic_timing_pass`,
  `full_run_eligible = false`, empty Kaggle source lists, and
  `status_scope = "non_promotable_synthetic_timing"`. Matrix summary: 16 rows,
  all `pass`; 8 `single_visible_t4`, 8 `dual_t4_ddp`; 2 fit-probe-only rows.
  Historical compact profile evidence: 4096 total patches, 2048 train / 2048
  validation, 805306368 payload bytes, 2048 CSV rows per split, both shard
  files 402653248 bytes, CRC validated, semantic keys unique, and loader
  normalization range proof passed. This is not the current default profile.
  Top recommendations were
  `single_visible_t4__bs16__amp_off_fp32__compile_off__branchless_all`
  (`estimated_epoch_minutes = 1.816008`),
  `dual_t4_ddp__bs24__amp_off_fp32__compile_off__branchless_all`
  (`2.565038`), and
  `dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all`
  (`2.606373`). These are screening/order evidence only, not selected runtime
  evidence.
- 2026-06-18 Kaggle synthetic timing 2 GiB profile update and remote v2: after
  the user clarified that this benchmark should choose where/how to run later
  real training without attaching the 60 GB dataset, the default synthetic
  profile was scaled to `synthetic_binary_2gib_histology_like_v1` and committed
  as `651cc69 Scale synthetic timing profile`, then pushed to GitHub. The old
  `synthetic_binary_0p81gb_histology_like_v1` profile remains as a named
  historical profile for remote-v1 evidence lineage. The push guard now decodes
  the embedded payload and asserts the 2 GiB default/compact historical profile
  constants. The recommendation JSON explicitly records that
  `estimated_epoch_minutes` is
  `loader_collate_normalize_h2d_only_projected_to_real_train_patch_count`, with
  model forward/backward, optimizer, corruption, precision policy, and
  `torch.compile` marked unmeasured.
  Verification before the push: focused synthetic timing/upload-simulation
  pytest passed with 11 tests, focused BasedPyright passed with 0 errors,
  `./scripts/kaggle_kernel.sh build kaggle/kernels/synthetic_timing` passed,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/synthetic_timing` passed,
  `git diff --check` passed, and `./scripts/python_quality.sh` passed with 89
  tests and 0 BasedPyright errors. Remote Kaggle version 2 completed with
  `status = "synthetic_timing_pass"`, downloaded ignored benchmark artifacts at
  `runs/kaggle/synthetic_timing_2gib/benchmark`, and no
  `benchmark/selected_runtime.json`. Manifest profile evidence: 10,912 total
  patches, 5,456 train / 5,456 validation, 2,145,386,496 payload bytes,
  5,456 CSV rows per split, both split payloads 1,072,693,248 bytes, both shard
  files 1,072,693,312 bytes, CRC validated, semantic keys unique, and loader
  normalization range proof passed. Matrix summary: 16 rows, all `pass`; 8
  `single_visible_t4`, 8 `dual_t4_ddp`; 0 fit-probe rows and 0 sample reuse.
  Top recommendations were single-T4 batch sizes 8, 12, and 16 with
  estimated loader/H2D-projected epoch times `2.015909`, `2.052612`, and
  `2.090076` minutes. The output download was interrupted only after the four
  benchmark files were present to avoid downloading the generated 2 GiB raw
  synthetic data directory; an ignored partial zero-byte synthetic data file may
  remain under `runs/kaggle/synthetic_timing_2gib/synthetic_timing_data`.
- 2026-06-18 Kaggle synthetic timing remote v3: adversarial final review found
  that v2 did not preserve per-rank DDP device assignment in runtime proof and
  that rows with global batch sizes 64/128 reported padded capacity instead of
  exact `effective_samples_per_epoch = real_train_patch_count` under
  `drop_last = false`. The implementation was fixed in
  `bc25862 Strengthen synthetic timing runtime proof`: matrix rows now include
  `row_order`, child return code, DDP torchrun return code, DDP rank count/order,
  and serialized DDP rank assignments; `synthetic_timing_runtime_proof.json`
  summarizes row order, child return codes, and per-rank device assignments; and
  recommendations include an explicit 5-warmup/25-measured repeat-shortlist
  policy. Focused synthetic timing/upload-simulation pytest passed with 12
  tests, focused BasedPyright passed with 0 errors, and
  `./scripts/python_quality.sh` passed with 90 tests and 0 BasedPyright errors.
  Remote Kaggle version 3 completed with `status = "synthetic_timing_pass"`,
  downloaded ignored benchmark artifacts at
  `runs/kaggle/synthetic_timing_2gib_v3/benchmark`, and no
  `benchmark/selected_runtime.json`. Manifest profile evidence stayed at
  10,912 total patches, 5,456 train / 5,456 validation, and 2,145,386,496
  payload bytes; both splits passed CRC validation and semantic-key uniqueness.
  Matrix summary: 16 rows, all `pass`; 8 `single_visible_t4`, 8 `dual_t4_ddp`;
  0 fit-probe rows; all rows have `effective_samples_per_epoch = 300000` and
  child return code `0`; all dual rows have torchrun return code `0`, rank count
  `2`, and rank order `[0, 1]`. Top v3 recommendations were
  `dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all`
  (`estimated_epoch_minutes = 1.592481`),
  `single_visible_t4__bs32__amp_off_fp32__compile_off__branchless_all`
  (`2.385673`), and
  `single_visible_t4__bs12__amp_off_fp32__compile_off__branchless_all`
  (`2.439552`). The output download was interrupted only after the four
  benchmark files were present to avoid downloading the generated 2 GiB raw
  synthetic data directory; an ignored partial zero-byte synthetic data file may
  remain under `runs/kaggle/synthetic_timing_2gib_v3/synthetic_timing_data`.
- 2026-06-18 Kaggle synthetic timing repeat-shortlist remote v4: implemented
  explicit row specs and a `repeat_shortlist` timing phase in
  `5e3ca30 Add synthetic timing repeat shortlist`, then pushed the commit to
  GitHub and Kaggle kernel version 4 with `KAGGLE_PUSH_CONFIRMED=1`. The
  adversarial repeat-review swarm found and the implementation fixed:
  top-level `synthetic_timing_pass` masking partial row failures, repeat-phase
  recommendations still saying repeat was required, stale `run_template.py`
  launcher verification gaps, and an undocumented fourth shortlist row. The
  v4 shortlist now matches the v3 artifact top-four rows:
  `dual_t4_ddp` bs8, `single_visible_t4` bs32, `single_visible_t4` bs12, and
  `single_visible_t4` bs4. Verification before remote push:
  `PYTHONPATH=src .venv/bin/pytest tests/test_synthetic_timing.py
  tests/test_kaggle_embedded_kernel.py -q` passed with 17 tests;
  `./scripts/python_quality.sh` passed with 95 tests and 0 BasedPyright errors;
  `./scripts/kaggle_kernel.sh build kaggle/kernels/synthetic_timing` and
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/synthetic_timing`
  passed. Remote v4 completed with `status = "synthetic_timing_pass"` in all
  three JSON artifacts, downloaded ignored benchmark artifacts at
  `runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark`, and no
  `benchmark/selected_runtime.json`. Manifest profile evidence stayed at
  10,912 total patches, 5,456 train / 5,456 validation, and 2,145,386,496
  payload bytes; both splits passed CRC validation and semantic-key uniqueness.
  Matrix summary: 4 rows, all `pass`; `warmup_steps = 5`;
  `measured_steps = 25`; repeat policy `completed = true` and
  `required_before_operational_shortlist = false`; payload manifest commit
  `5e3ca30ede257fe9c03b51b41fca772875bd8c8b`; payload dirty flag `false`; and
  embedded template digest recorded for
  `kaggle/kernels/synthetic_timing/run_template.py`. Top v4 recommendations
  were `dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all`
  (`estimated_epoch_minutes = 1.312643`),
  `single_visible_t4__bs4__amp_off_fp32__compile_off__branchless_all`
  (`1.964479`), and
  `single_visible_t4__bs32__amp_off_fp32__compile_off__branchless_all`
  (`2.043706`). Dual DDP rank proof recorded rank order `[0, 1]`, rank count
  `2`, torchrun return code `0`, and one Tesla T4 per local rank. The output
  download was interrupted only after the four benchmark files were present;
  the partial raw synthetic data download was removed.
- 2026-06-18 capped real-data runtime pretest contract pass: two independent
  adversarial subagents reviewed the next benchmark design and found that a
  capped real-data run could be accidentally over-promoted, compile settling was
  still too implicit, synthetic-v4-only candidate seeding could bias selection,
  prefix caps could bias loader evidence, `indexed_masked` could change
  corruption RNG semantics, and the checked-in runtime schema/config still used
  the stale boolean compile axis. The fixes applied in this pass are
  config/schema/docs only, not a runner: `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`
  now declares the non-promotable capped real-data pretest with fixed spread
  windows, synthetic-v4 parent row provenance plus sentinel rows, staged FP32
  first / AMP follow-up policy, named compile scopes, `compile_settle_steps = 5`,
  Dynamo counter-source requirements, blocked claims, and
  `writes_selected_runtime = false`. The local schema writer now includes
  `compile_scope`, `compile_settle_steps`, `graph_break_count`,
  `recompile_count`, `benchmark/runtime_proof.json`, and
  `benchmark/corruption_checks.csv`, with the local `selected_runtime.json`
  remaining `schema_pass` / `full_run_eligible = false`. Focused verification:
  `PYTHONPATH=src .venv/bin/pytest tests/test_spec0001_benchmark_scaffold.py`
  passed with 9 tests before Ruff split the expanded schema test; final
  verification after formatting passed with 10 focused tests. A first pytest
  attempt without `PYTHONPATH=src` failed
  with `ModuleNotFoundError: No module named 'eqvae'`; this is the expected
  import-path issue avoided by the repo quality script.
  `./scripts/python_quality.sh` passed after Ruff formatting with 97 tests and
  0 BasedPyright errors. `./scripts/agent_preflight.sh` passed and noted only
  the expected dirty worktree. `python3 -m json.tool
  configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json >/dev/null` and
  `git diff --check` passed. Remaining implementation blockers: add
  `indexed_masked` corruption and equivalence checks, add the capped real-data
  train-step benchmark runner, add the dedicated Kaggle kernel and push guard
  requiring `KAGGLE_FULL_DATASET_CONFIRMED=1`, then rerun Python quality and
  preflight before any remote push.
- 2026-06-19 capped real-data runtime pretest local implementation pass:
  added the non-promotable runner/CLI/kernel/guard surface for
  `kaggle/kernels/real_data_runtime_pretest`. The kernel metadata requests T4
  GPU, disables internet, attaches only
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`, and leaves competition,
  kernel, and model sources empty. The generated ignored `run.py` embeds the
  repo payload, checks payload provenance, supports import-only upload
  simulation, calls `eqvae.cli.real_data_runtime_pretest`, and rejects
  `benchmark/selected_runtime.json`. The push guard now requires
  `KAGGLE_FULL_DATASET_CONFIRMED=1`, exact metadata/source lists, fresh embedded
  payload, config status
  `real_data_runtime_pretest_kernel_guard_ready_non_promotable`, the locked
  8,192/2,048 cap, `compile_settle_steps = 5`, non-promotable pretest
  artifacts, and explicit selected-runtime rejection. The local writer emits
  `real_data_runtime_pretest_manifest.json`, `runtime_proof.json`,
  `runtime_matrix.csv`, `dataloader_matrix.csv`, `numerical_checks.csv`,
  `corruption_checks.csv`, `metrics/gate_health.csv`,
  `gate_health_summary.json`, and
  `real_data_runtime_pretest_recommendations.json`, never
  `selected_runtime.json`; it also rejects a stale selected-runtime file before
  and after artifact writing. Local CPU rows are `wrong_accelerator`; any timed
  remote single-T4 row from this first implementation is marked `ineligible`
  with `linked_safety_evidence_pending`. Manifest/proof fields explicitly mark
  real-data identity, file hashes, row counts, WSI counts, CRC validation,
  validation-window exercise, cache/warmup policy, dataloader, numerical,
  corruption, gate-health, DDP, graph-break, and recompile evidence as pending
  rather than selection-ready. The `indexed_masked` corruptor strategy is now
  implemented and locally equivalence-tested against `branchless_all`, but it is
  still not accepted as a runtime choice because compile stability and real
  throughput evidence remain pending.
  Verification so far: focused
  `PYTHONPATH=src .venv/bin/pytest tests/test_stain_corruptor.py
  tests/test_real_data_runtime_pretest.py tests/test_kaggle_embedded_kernel.py
  tests/test_spec0001_benchmark_scaffold.py -q` passed with 39 tests;
  `./scripts/python_quality.sh` passed with 110 tests and 0 BasedPyright
  errors after Ruff fixes; `./scripts/kaggle_kernel.sh build
  kaggle/kernels/real_data_runtime_pretest`, `bash -n scripts/kaggle_kernel.sh`,
  and `./scripts/kaggle_kernel.sh validate
  kaggle/kernels/real_data_runtime_pretest` passed; `python3 -m json.tool
  configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` and
  `git diff --check` passed. The first `./scripts/agent_preflight.sh` run
  failed only because the newly required real-data pretest files were still
  untracked; after staging and committing, `./scripts/agent_preflight.sh`
  passed with a clean worktree. Two adversarial subagents reviewed the
  implementation; real findings around brittle spec-index guard text,
  contradictory config status, stale selected-runtime rejection, and
  safe-looking skipped scalar placeholders were fixed. Superseded next by the
  proof-lane pass below; after that pass, remaining real blockers are
  compile-settle/Dynamo accounting, DDP launch proof, real dataloader throughput
  matrix, paired numerical checks, corruption compile-stability checks, and
  gate-health rows. Implementation commit:
  `fc5194b` (`Add real-data runtime pretest scaffold`) and pushed to GitHub.
- 2026-06-19 capped real-data runtime pretest remote v1 plumbing check:
  after `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check`
  passed auth, kernel status/logs, and dataset file listing with the known
  quota/files endpoint warnings, the first push attempt was correctly rejected
  by the payload-freshness guard because ignored `run.py` was built before the
  commit. Rebuilding with
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`
  fixed the embedded manifest, and
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1
  ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`
  pushed Kaggle version 1. Remote status reached `KernelWorkerStatus.COMPLETE`.
  Downloaded ignored artifacts live under
  `runs/kaggle/real_data_runtime_pretest_v1`. The remote artifact contract
  worked as a plumbing test only: `runtime_proof.json` has
  `status = "pretest_incomplete"`, `status_scope =
  "non_promotable_real_data_runtime_pretest"`, `full_run_eligible = false`,
  `selection_ready = false`, `selected_runtime_written = false`,
  `eligible_pass_row_count = 0`, `row_count = 56`,
  `timed_ineligible_row_count = 6`, `skipped_unsupported_row_count = 48`, and
  no `benchmark/selected_runtime.json`. The six timed rows were single-visible
  T4 eager/FP32/no-compile rows for batch sizes 4, 8, and 12 crossed with
  `branchless_all` and `indexed_masked`, all marked `ineligible` with
  `linked_safety_evidence_pending`; single-T4 batch 32 hit OOM for both
  corruption strategies. Dual-T4/DDP rows and compile-scope rows remain
  `skipped_unsupported`. Manifest fields still mark real-data identity, file
  hashes, row counts, WSI counts, CRC validation, validation-window exercise,
  cache/warmup policy, and timed-row eligibility as pending. This remote run
  confirms Kaggle packaging, dataset attachment, artifact writing, and
  non-promotion; it does not select a runtime or support paper/training claims.
  Overleaf was untouched.
- 2026-06-19 capped real-data runtime pretest proof-lane implementation pass:
  `write_real_data_runtime_pretest` now runs a real-data/local proof lane before
  stage-1 rows. It rejects stale `benchmark/selected_runtime.json`, resolves the
  configured data root, validates both UBC binary/CSV shards through
  `PatchShard`, computes SHA256 file hashes and streaming payload CRC32, checks
  row counts, records split WSI/label/identity-window proof, enforces the exact
  locked 8,192/2,048 spread-window contract for canonical real `pass`, and
  records split/holdout overlap checks. Tiny fixture roots can only produce
  `local_pass`; canonical real `pass` requires the expected dataset slug, exact
  300000/30000 rows, exact 322/39 WSI counts, zero train/validation and
  masked-holdout WSI overlap, and the locked train/validation windows. The clean
  validation proof now exercises `PatchTrainingDataset`,
  `collate_patch_training_samples`, and `normalize_uint8_batch` over the
  validation windows, records dtype/range/sample-id/batch timing proof, and
  explicitly marks corruption RNG instrumentation as not exercised in this lane.
  Dataloader validation rows may reflect the clean loader proof but remain
  `ineligible`; timed runtime rows still require compile/DDP/real throughput/
  numerical/corruption/gate-health evidence before any eligibility.
  Verification: focused Ruff format/check, focused BasedPyright, and focused
  pytest for `src/eqvae/benchmarking/real_data_runtime_pretest.py` plus
  `tests/test_real_data_runtime_pretest.py` passed with 7 tests;
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`,
  and `bash -n scripts/kaggle_kernel.sh` passed; the full
  `./scripts/python_quality.sh` passed with 112 tests and 0 BasedPyright
  errors. A focused adversarial subagent review found and the implementation
  fixed over-broad WSI pass semantics, weak window-contract enforcement,
  overclaimed clean-validation RNG proof, and tiny-fixture overclaiming.
- 2026-06-19 capped real-data runtime pretest linked-evidence local scaffolding
  pass: the pretest writer now attaches linked evidence objects to the manifest
  and runtime proof for compile-settle, DDP launch, fixed-window dataloader
  throughput, paired branchless/indexed numerical comparison, corruption
  equivalence, and gate health. The implementation deliberately distinguishes
  local/mechanics evidence from canonical eligibility evidence: compile-settle
  and DDP lane `status` values remain `skipped_unsupported` with
  `contract_status = "local_pass"` until measured compiled rows and real
  dual-T4 ranks exist; fixed-window dataloader throughput remains
  `local_pass` only and records that candidate-row coverage is still required;
  paired numerical and corruption proof summaries are `local_pass` for one fixed
  eager single-rank batch but their candidate CSV rows stay
  `skipped_unsupported` unless the exact batch-size/accelerator/compile path is
  covered. Corruption rows no longer claim clean-validation RNG advancement
  proof. Runtime rows now read numerical status from row-specific numerical CSV
  entries instead of inheriting the broad lane status, and
  `compile_settle_policy.implemented_in_this_runner` is false until measured
  compiled rows exist. `linked_evidence_status` remains
  `skipped_unsupported` for local fixtures, row eligibility stays blocked, and
  `benchmark/selected_runtime.json` is still rejected/not written. A
  clean-context adversarial subagent review found the overclaim risks; the code
  and tests were updated accordingly. Verification: focused Ruff format/check,
  focused BasedPyright, and focused pytest for
  `src/eqvae/benchmarking/real_data_runtime_pretest.py` plus
  `tests/test_real_data_runtime_pretest.py` passed with 7 tests;
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`,
  `bash -n scripts/kaggle_kernel.sh`, and `git diff --check` passed. The full
  `./scripts/python_quality.sh` passed with 112 tests and 0 BasedPyright
  errors. `./scripts/agent_preflight.sh` passed and noted only the expected
  dirty worktree.
- 2026-06-19 capped real-data runtime pretest remote v2 result: committed the
  linked-evidence scaffolding as `53051e8` (`Add real-data linked pretest
  evidence scaffolds`), rebuilt
  `kaggle/kernels/real_data_runtime_pretest/run.py` so the ignored embedded
  payload matches that commit, and validated the kernel locally. The approved
  remote preflight
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passed
  OAuth, kernel list/status/logs, and dataset file listing, with the known
  warnings for the quota and kernels-files endpoints. The approved remote write
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1
  ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`
  pushed Kaggle version 2 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  Multiple approved status polls initially reported
  `KernelWorkerStatus.RUNNING`; a later approved status check reported
  `KernelWorkerStatus.COMPLETE`, and approved output download saved artifacts
  under `runs/kaggle/real_data_runtime_pretest_v2`. The run completed but did
  not exercise the new real-data proof lane: `runtime_proof.json` has
  `status = "pretest_incomplete"`, `linked_evidence_status =
  "skipped_unsupported"`, `eligible_pass_row_count = 0`,
  `timed_ineligible_row_count = 6`, `skipped_unsupported_row_count = 48`,
  `selection_ready = false`, and `selected_runtime_written = false`.
  `real_data_runtime_pretest_manifest.json` reports
  `real_data_proof.failure_kind = "data_root_unavailable"`, `data_root =
  "auto"`, and `resolved_data_root = ""`, so real-data identity, CRC, window,
  clean-validation dataloader, dataloader-throughput, paired numerical,
  corruption, and gate-health statuses are all `skipped_unsupported`.
  `dataloader_matrix.csv` contains only train/validation pending rows,
  `numerical_checks.csv` and `corruption_checks.csv` contain 56
  `skipped_unsupported` rows each, and `metrics/gate_health.csv` contains only
  the header. Runtime matrix behavior otherwise matches v1: six eager
  single-visible-T4 rows reached timed/ineligible shape, 48 rows were
  skipped-unsupported, and the two single-T4 batch-32 rows failed with
  `runtime_OutOfMemoryError`. No `benchmark/selected_runtime.json` was present.
  The next remote attempt should not be an identical resend; first fix or
  instrument Kaggle data-root discovery. Overleaf was untouched.
- 2026-06-19 capped real-data runtime pretest data-root diagnostics fix:
  implemented the local fix for the v2 `data_root_unavailable` blocker without
  touching Overleaf or pushing to Kaggle. `src/eqvae/data/roots.py` now performs
  bounded `/kaggle/input` discovery but only promotes complete shard roots whose
  relative path matches the expected
  `maximusshtefan/patches-pre-shuffled-ubc-ocean` mount family, including
  direct slug, owner/slug, `datasets/owner/slug`, versioned
  `datasets/owner/slug/versions/N`, and their `dataset/` children. Complete
  shard roots outside that slug family are refused for resolution and recorded
  in diagnostics as `complete_unaccepted_candidates`; diagnostics also record
  accepted candidates, missing paths, scan truncation, a bounded input snapshot,
  and only `env_value_present` rather than a raw env var value.
  `src/eqvae/benchmarking/real_data_runtime_pretest.py` now attaches
  `real_data_proof.data_root_diagnostics` on both success and failure, retries
  auto resolution up to four short attempts when `/kaggle/input` exists, raises
  a diagnostic-carrying `DataRootUnavailableError`, and emits concise stderr
  JSON probe lines with event
  `real_data_runtime_pretest_data_root_probe` so Kaggle logs reveal candidate
  counts and candidate roots. A clean-context adversarial subagent review found
  and the implementation fixed an over-broad draft that would have accepted any
  unrelated complete shard root under `/kaggle/input`. Verification: focused
  Ruff format/check, focused BasedPyright, and focused pytest for
  `src/eqvae/data/roots.py`,
  `src/eqvae/benchmarking/real_data_runtime_pretest.py`,
  `tests/test_data_roots.py`, and `tests/test_real_data_runtime_pretest.py`
  passed with 17 tests; full `./scripts/python_quality.sh` passed with 118
  tests and 0 BasedPyright errors; `bash -n scripts/kaggle_kernel.sh`,
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`,
  and `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`
  passed locally. Next action: commit, rebuild after the commit, then push a
  new Kaggle v3 only with explicit remote-write and dataset-source approval.
- 2026-06-19 capped real-data runtime pretest remote v3 plus payload fix:
  committed the data-root diagnostics fix as `8d927a5` (`Fix real-data pretest
  data-root diagnostics`), rebuilt and validated the real-data runtime pretest
  kernel locally, ran the approved Kaggle API preflight, and pushed Kaggle
  version 3 with `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`.
  Version 3 completed and downloaded ignored artifacts under
  `runs/kaggle/real_data_runtime_pretest_v3`. The v3 diagnostics confirmed the
  intended Kaggle root was present and complete:
  `/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset`
  had all four required shard files with the expected large file sizes, and
  `complete_unaccepted_candidate_count = 0`. The remaining failure was no
  longer root discovery: `real_data_proof` still reported the broad legacy
  `failure_kind = "data_root_unavailable"` only because a later proof-lane
  `FileNotFoundError` was caught by that handler. The actual missing payload
  dependency is `docs/data/ubc_ocean_masked_holdout_ids.csv`, needed by
  `_split_contract_proof` inside the embedded single-file Kaggle payload.
  The local follow-up fix now embeds that CSV, records it in the payload
  manifest/freshness checks, tests that generated `run.py` physically contains
  it, and changes non-resolver `FileNotFoundError` failures to
  `data_proof_FileNotFoundError` with a short `failure_message_excerpt`.
  Verification for the local follow-up fix: focused Ruff format/check, direct
  BasedPyright on touched files, and focused pytest for embedded-kernel plus
  real-data pretest tests passed with 12 tests; full
  `./scripts/python_quality.sh` passed with 118 tests and 0 BasedPyright
  errors; `bash -n scripts/kaggle_kernel.sh`,
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`,
  and `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`
  passed locally. Next action: commit the payload fix, rebuild after that
  commit, then push a new Kaggle v4 only with explicit remote-write and
  dataset-source approval. Overleaf was untouched.
- 2026-06-19 capped real-data runtime pretest remote v4 result: committed the
  embedded masked-holdout payload fix as `b9cc977` (`Embed holdout split proof
  data in Kaggle payload`), rebuilt and validated the real-data runtime pretest
  kernel locally, pushed approved Kaggle version 4 with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`, polled it to
  `KernelWorkerStatus.COMPLETE`, and downloaded ignored artifacts under
  `runs/kaggle/real_data_runtime_pretest_v4`. The run achieved the first
  canonical real-data proof lane: `real_data_proof.status = "pass"`,
  `identity_status = "pass"`, `row_count_status = "pass"`,
  `wsi_count_status = "pass"`, `crc_validation_status = "pass"`,
  `window_status = "pass"`, and `clean_validation_dataloader_status = "pass"`.
  It resolved data at
  `/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset`,
  confirmed train 300000 rows / 322 WSIs, validation 30000 rows / 39 WSIs,
  zero train-validation WSI overlap, zero overlap with the 152 masked-holdout
  IDs, exact locked windows, and a clean validation loader proof over 2048
  samples, 171 batches, final partial batch size 8, normalized range `[-1, 1]`,
  and `loader_samples_sec = 10271.926998`. The non-promotable runtime matrix is
  still incomplete: `runtime_proof.status = "pretest_incomplete"`,
  `linked_evidence_status = "skipped_unsupported"`, `selection_ready = false`,
  `selected_runtime_written = false`, and `eligible_pass_row_count = 0`.
  `dataloader_matrix.csv` now has two `local_pass` rows, while the 56
  numerical rows are `skipped_unsupported` with
  `compile_or_ddp_numerical_pending`, the 56 corruption rows are
  `skipped_unsupported` with `candidate_specific_corruption_pending`, and the
  timed runtime rows remain ineligible because
  `compile_settle_evidence_not_canonical_pass`; the two single-T4 batch-32
  eager rows still fail with `runtime_OutOfMemoryError`. No
  `benchmark/selected_runtime.json` was written. Next action: replace the
  local/contract-only linked scaffolds with candidate-specific canonical
  compile/DDP/dataloader/numerical/corruption/gate-health evidence before any
  row can become eligible. Overleaf was untouched.
- 2026-06-19 capped real-data runtime pretest candidate-linked evidence lane:
  implemented the next local runner slice without touching Overleaf or making
  remote Kaggle writes. `src/eqvae/benchmarking/real_data_runtime_pretest.py`
  now measures `model_forward` compile rows with 5 unmeasured settle steps,
  records per-row Dynamo counter snapshots plus post-settle graph-break and
  recompile counts, and keeps `model_loss` / `train_step_no_optimizer` as
  explicit implementation-pending scopes. Compiled rows are intentionally
  diagnostic-only/ineligible until full compile-settle coverage includes clean
  validation, DDP rank paths, final partial batches, and mask cardinalities
  0/1/many/all. The linked evidence payload now runs
  a real dual-rank `torch.distributed.run --standalone --nproc_per_node=2`
  launch probe when two T4s are visible, records rank assignments, and reports
  timeout/missing-rank-payload failures as structured proof rows. Dataloader
  throughput is measured per accelerator/world-size/per-device-batch candidate
  and must pass both the data-wait fraction and
  `loader_samples_sec >= 1.25 * trainer_samples_sec` checks. Numerical rows now
  compare each covered candidate against the same-batch eager FP32
  `branchless_all` reference across three fixed batches, corruption rows are
  emitted per fixed batch, and gate-health JSON row statuses apply saturation,
  dead-channel, output/input RMS, and parameter-threshold gates. Runtime-row
  eligibility is now decided per row from canonical data proof, DDP launch
  proof, compile status, matching dataloader rows, matching numerical/corruption
  rows, and gate-health status instead of a single global linked-evidence flag.
  Tiny fixtures still produce only `local_pass`/`skipped_unsupported`, and the
  capped pretest still never writes `benchmark/selected_runtime.json`. The first
  and follow-up adversarial review sets both ran; follow-up verdict is safe for
  a capped, non-promotable v5 attempt after commit/rebuild, not safe for
  selected-runtime promotion, full dataloader selection, paper evidence, or
  compiled-row promotion. Tests were updated for the now-implemented
  `model_forward` compile scope. Docs updated:
  `docs/specs/0001-translatable-normal-vae-baseline.md`,
  `docs/specs/README.md`, and `docs/kaggle_cli_workflow.md`. Verification:
  full `./scripts/python_quality.sh` passed with Ruff, BasedPyright, 118 tests,
  and 0 type errors; `git diff --check` passed;
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`
  passed; `./scripts/kaggle_kernel.sh validate
  kaggle/kernels/real_data_runtime_pretest` passed; dirty-worktree embedded
  payload verification passed with `--allow-dirty`; workspace
  `./agent_preflight.sh` passed. The six tracked files were committed locally,
  then the real-data runtime pretest kernel was rebuilt and revalidated from the
  clean commit; clean embedded payload verification without `--allow-dirty`
  also passed. The follow-up remote v5 launch is recorded below.
- 2026-06-19 capped real-data runtime pretest remote v5 launch:
  after explicit user approval, ran guarded remote Kaggle preflight with
  `KAGGLE_REMOTE_CONFIRMED=1`; auth, kernels list/status/logs, and patch-dataset
  file listing passed, while the quota endpoint still warned and the kernels
  files endpoint remained unavailable. Pushed
  `kaggle/kernels/real_data_runtime_pretest` with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`; Kaggle accepted
  kernel version 5 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  Status checks during the monitoring window continued to report
  `KernelWorkerStatus.RUNNING`; the direct Kaggle log read returned no useful
  lines. At that point no artifacts had been downloaded yet and no Overleaf
  work was touched. The
  polling policy was updated in `docs/kaggle_cli_workflow.md`: for
  source-attached real-data kernels, do one immediate post-push status check,
  then poll every 30 minutes or slower and record actual terminal durations
  after artifact download. The later v5 result and inspection are recorded in
  the next entry.
- 2026-06-19 capped real-data runtime pretest remote v5 result:
  a later approved status check reported `KernelWorkerStatus.COMPLETE`, and
  approved output download saved ignored artifacts under
  `runs/kaggle/real_data_runtime_pretest_v5`. No
  `benchmark/selected_runtime.json` exists. The Kaggle log shows the data-root
  probe at about 7.44 seconds and notebook result conversion around
  2355 seconds, so this source-attached capped run took roughly 39 minutes;
  v5 predates `benchmark/phase_timings.json`, so exact phase timings are not
  available. `runtime_proof.json` has `status = "pretest_incomplete"`,
  `full_run_eligible = false`, `selection_ready = false`,
  `selected_runtime_written = false`,
  `linked_evidence_status = "skipped_unsupported"`, and
  `eligible_pass_row_count = 2`. Canonical real-data identity, row count, WSI
  count, CRC, locked-window, clean-validation loader, real DDP launch,
  dataloader throughput, and gate-health proof all pass. The two passing rows
  are `single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all`
  (`samples_sec = 6.181214`, `steady_step_ms_p95 = 656.454308`) and
  `single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked`
  (`samples_sec = 6.210007`, `steady_step_ms_p95 = 650.498703`).
  `dataloader_matrix.csv` has 8/8 pass rows and `metrics/gate_health.csv` has
  68/68 pass rows. `numerical_checks.csv` and `corruption_checks.csv` each have
  12/64 pass rows: bs4 eager and bs4 `model_forward` candidates over three
  fixed batches. The eager bs8/bs12 rows are timed but ineligible because
  numerical/corruption rows are still skipped for those batch sizes; bs32 eager
  rows hit `runtime_OutOfMemoryError`; compiled rows have zero graph-break and
  recompile counts but remain intentionally ineligible because
  `compile_settle_coverage_pass = false` with missing clean-validation, DDP-rank,
  final-partial-batch, and mask-cardinality coverage. This motivated the v6
  diagnostics/coverage follow-up recorded below; keep the capped pretest
  non-promotable.
- 2026-06-19 real-data pretest phase-timing logging:
  added coarse phase instrumentation for future real-data pretest reruns after
  the already-running remote v5. The runner now emits stderr JSON lines for
  phase start/finish events, mirrors `phase_timings` into
  `runtime_proof.json` and `real_data_runtime_pretest_manifest.json`, writes
  `benchmark/phase_timings.json`, and allowlists that artifact in the runtime
  manifest, config, generated Kaggle launcher template, and shell push guard.
  Timed phases cover config resolution, output prep, real-data
  identity/clean-path proof, each stage-1 runtime row, linked-evidence sublanes
  (compile-settle/DDP/train-step/dataloader/numerical/corruption/gate-health),
  row eligibility join, schema row materialization, and artifact writing. This
  is intended to set future Kaggle polling cadence from observed durations
  instead of repeated status checks. A review-found launcher allow-list mismatch
  was fixed: local generated-launcher full simulation now validates the complete
  artifact set including `phase_timings.json`, not just import-only packaging.
  This is included in remote v6; remote v5 remains the previously downloaded
  version without phase-timing artifacts.
- 2026-06-19 real-data pretest candidate-evidence v6 diagnostics:
  after refreshing the completed remote v5 output, local follow-up changes now
  make `_paired_train_step_evidence` cover candidate targets in eager
  `compile_none` order first, then smaller per-device batch size, then row ID.
  The train-step evidence path now clears CUDA cache after paired
  branchless/indexed evidence and after candidate evidence exceptions, records
  failed candidate evidence rows with deterministic failure hashes, exposes
  `candidate_evidence_count`, `failed_candidate_evidence_count`, and
  `failed_candidate_evidence` through paired numerical and corruption
  equivalence proof objects, and mirrors quick evidence counters into
  `runtime_proof.json`. The later v6 pull did show explicit failed-candidate
  RuntimeError hashes for the bs8/bs12 evidence attempts instead of leaving
  those rows as opaque skipped evidence. Review-found gaps were fixed: if every
  candidate train-step evidence attempt fails, paired numerical, corruption,
  and gate-health proof objects now preserve failed-candidate diagnostics
  instead of collapsing to a generic linked-evidence failure; failure rows record
  strategy attempt, target corruption strategy, affected row ids, and a stable
  message hash. `scripts/kaggle_kernel.sh pull` now requires both
  `KAGGLE_REMOTE_CONFIRMED=1` and `KAGGLE_PULL_CONFIRMED=1`; real-data
  status/output aliases were added for v6 monitoring and download. Verification:
  focused `tests/test_real_data_runtime_pretest.py` and
  `tests/test_kaggle_embedded_kernel.py` passed together with 16 tests, and
  full `./scripts/python_quality.sh` passed with Ruff, BasedPyright, 122 tests,
  and 0 type errors. `git diff --check`, real-data pretest kernel
  build/validate, dirty-worktree embedded verify with `--allow-dirty`, and
  `./scripts/agent_preflight.sh` passed. The implementation was committed as
  `47437a0` (`Harden real-data pretest candidate evidence diagnostics`), then
  the real-data pretest kernel was rebuilt/revalidated from the clean commit and
  clean embedded verification passed. The remote v6 launch is recorded below.
- 2026-06-19 capped real-data runtime pretest remote v6 launch:
  after explicit user approval, rechecked the clean worktree and confirmed
  `HEAD = 47437a0`. `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh
  api-check` passed OAuth, kernel list/status/logs, and patch-dataset file
  listing; the quota endpoint still warned and the kernels files endpoint
  remained unavailable, matching the known Kaggle CLI limitation. Rebuilt and
  revalidated `kaggle/kernels/real_data_runtime_pretest`, then pushed with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`; Kaggle accepted
  kernel version 6 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  The immediate approved status read reported `KernelWorkerStatus.RUNNING` by
  2026-06-19T16:37:07-05:00. A later approved status read after the required
  wait reported `KernelWorkerStatus.COMPLETE`, and artifacts were downloaded to
  `runs/kaggle/real_data_runtime_pretest_v6`. Inspection found:
  `runtime_proof.status = "pretest_incomplete"`,
  `full_run_eligible = false`, `selection_ready = false`,
  `selected_runtime_written = false`, `eligible_pass_row_count = 2`,
  `paired_numerical_candidate_evidence_count = 2`,
  `paired_numerical_failed_candidate_evidence_count = 5`,
  `corruption_equivalence_candidate_evidence_count = 2`, and
  `corruption_equivalence_failed_candidate_evidence_count = 5`. The only pass
  rows are `single_visible_t4__bs4__amp_off_fp32__compile_none__branchless_all`
  and `single_visible_t4__bs4__amp_off_fp32__compile_none__indexed_masked`.
  Eager bs8/bs12 timed rows remain ineligible with
  `paired_numerical_evidence_not_row_pass`; their failed candidate evidence
  records `candidate_train_step_RuntimeError` with message hash
  `757ab3828da1202c080e587121c92ffa9210d9ecace6cb28842a62504733fc14`.
  Eager bs32 rows still hit `runtime_OutOfMemoryError`. Compiled
  `model_forward` rows remain diagnostic/ineligible with
  `compile_settle_or_dynamo_evidence_not_row_pass`; compile-settle proof is
  still `skipped_unsupported` because clean-validation, DDP rank, final partial
  batch, and mask-cardinality coverage are missing. Phase timings passed with
  71 recorded phases from `2026-06-19T21:36:15Z` to
  `2026-06-19T22:15:40Z`; longest phases were stage1 runtime rows
  (~1185.75s), linked evidence payload (~592.19s), real-data identity/clean
  proof (~586.85s), and linked train-step evidence (~573.67s). No
  `benchmark/selected_runtime.json` was written.
- 2026-06-19 post-v6 local observability follow-up:
  `src/eqvae/benchmarking/real_data_runtime_pretest.py` now adds bounded
  `failure_message_excerpt` fields to failed candidate train-step evidence,
  matching the existing data-root failure excerpt style while keeping the
  deterministic `failure_message_hash`. The focused regression in
  `tests/test_real_data_runtime_pretest.py` asserts excerpt propagation. This
  local change is intended to make a future v7 artifact reveal the actual
  bs8/bs12 exception instead of another hash-only failure. Verification:
  `PYTHONPATH=src .venv/bin/pytest tests/test_real_data_runtime_pretest.py -q`
  passed with 10 tests in 63.18s; after restoring exact spec-index guard tokens
  required by `scripts/kaggle_kernel.sh`, the four affected push-guard tests
  passed; full `./scripts/python_quality.sh` passed with 122 tests and 0
  BasedPyright errors. This entry is superseded by the 2026-06-20 local
  readiness entry below: kernel rebuild/validate now passes, while Kaggle API
  preflight and explicit push permission are still required before any v7
  remote push.
- 2026-06-20 v7 diagnostics local readiness:
  after full repo grounding, three read-only adversarial subagents rechecked
  the v6 artifacts, failure-excerpt code path, and Kaggle kernel guard. They
  confirmed v6 remains non-promotable, has no
  `benchmark/selected_runtime.json`, has only two capped-pretest eligible eager
  single-T4 bs4 FP32 rows, and needs v7 only to expose bounded failed-candidate
  message excerpts. Follow-up hardening in commit `cabeb89` (`Harden
  real-data pretest v7 diagnostics`) strengthens excerpt propagation tests,
  lists `metrics/gate_health.csv` in the real-data pretest manifest
  self-description, and makes
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`
  verify embedded payload freshness. Verification passed:
  `PYTHONPATH=src .venv/bin/pytest tests/test_real_data_runtime_pretest.py
  tests/test_kaggle_embedded_kernel.py -q` with 16 tests;
  `./scripts/python_quality.sh` with Ruff, 122 pytest tests, and BasedPyright;
  `bash -n scripts/kaggle_kernel.sh`; `git diff --check`;
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`;
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`;
  repo `./scripts/agent_preflight.sh`; and workspace `./agent_preflight.sh`.
  Local `main` is at `cabeb89` and ahead of local `origin/main`; the rebuilt
  generated Kaggle `run.py` is ignored and fresh. Next action is to ask before
  any Kaggle remote read or write: first
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check`, then, only
  with explicit approval and available quota,
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1
  ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`.
- 2026-06-20 v7 Kaggle preflight:
  after explicit user approval, local status was clean at `c53252d` and
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`
  passed with a fresh embedded payload. The guarded read-only preflight
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passed OAuth
  token generation, kernel list/status/logs, and dataset file listing for
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`. It still warned that the
  Kaggle accelerator quota endpoint and kernels-files endpoint failed, matching
  the known CLI limitation. No v7 push has been run yet. Next action: the user
  must confirm in the Kaggle web UI that GPU quota is available; then run the
  approved guarded push with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1
  ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`.
- 2026-06-20 v7 Kaggle push:
  the user confirmed GPU quota was still available in the Kaggle web UI after
  the CLI quota endpoint warning. Local status was clean and
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`
  passed immediately before the push. The guarded remote write
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1
  ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`
  succeeded, and Kaggle accepted kernel version 7 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  The pushed embedded payload was built from the clean local commit `fea4140`
  and should expose bounded `failure_message_excerpt` fields for failed
  candidate evidence. The immediate guarded post-push status read
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh
  status-real-data-runtime-pretest` returned `KernelWorkerStatus.RUNNING` at
  `2026-06-19T23:38:51-05:00`, and the next guarded status poll returned
  `KernelWorkerStatus.COMPLETE` at `2026-06-20T02:21:15-05:00`. Output
  download and artifact inspection are recorded in the next entry.
- 2026-06-20 v7 output download and inspection:
  after explicit user approval, the guarded remote download
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh
  output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v7`
  succeeded at `2026-06-20T09:19:52-05:00`. Downloaded artifacts include
  `benchmark/runtime_proof.json`,
  `benchmark/real_data_runtime_pretest_manifest.json`,
  `benchmark/real_data_runtime_pretest_recommendations.json`,
  `benchmark/runtime_matrix.csv`, and `metrics/gate_health.csv`. Inspection
  confirmed `runtime_proof.status = pretest_incomplete`,
  `full_run_eligible = false`, `selection_ready = false`,
  `selected_runtime_written = false`, and no `selected_runtime.json`. The matrix
  still has 56 rows with statuses `pass = 2`, `ineligible = 12`,
  `runtime_error = 2`, and `skipped_unsupported = 40`. The two pass rows are
  still single-visible-T4, bs4, FP32 eager: `indexed_masked` ranks first at
  `6.148994` samples/sec and `659.031199` p95 step ms; `branchless_all` ranks
  second at `6.112694` samples/sec and `667.591992` p95 step ms. Gate health
  passed for 68 modules, all 71 recorded phases passed, and the manifest
  allowlist now includes `metrics/gate_health.csv`. The v7 diagnostic fix
  worked: both paired-numerical and corruption failed-candidate evidence expose
  bounded `failure_message_excerpt = "quantile() input tensor is too large"`
  for the repeated `candidate_train_step_RuntimeError` hash
  `757ab3828da1202c080e587121c92ffa9210d9ecace6cb28842a62504733fc14`.
  Next action is local analysis/fix work: find why the candidate train-step
  evidence path calls `quantile()` on too large a tensor for bs8/bs12 and
  compiled candidates, then decide whether to produce a v8 diagnostics/fix
  kernel. Do not push another Kaggle version without explicit user permission
  and the required guards.
- 2026-06-20 v7 adversarial quantile review:
  three read-only subagents reviewed the downloaded v7 artifacts, the runtime
  evidence semantics, and the local code path. All agreed the surfaced
  `quantile() input tensor is too large` exception is an evidence-plumbing bug,
  not evidence that bs8/bs12 are invalid runtime choices. The likely direct
  cause is unbounded `torch.quantile(tensor.flatten(), q)` in gate-health
  `gate_p01/gate_p50/gate_p99` telemetry after full gate activations are
  captured in the candidate train-step evidence path; bs4 early gates stay
  below the PyTorch quantile limit, while bs8 reaches the large-tensor limit.
  The next local v8 fix should implement deterministic bounded/sampled
  gate-health quantiles, keep exact saturation/pass-fail checks, add tests that
  exercise the large-tensor path, and prevent lane-level gate-health success
  from overstating row-specific coverage for candidates whose evidence failed.
  v8 should prove the evidence path only; it must remain non-promotable, keep
  `full_run_eligible = false`, and never write `benchmark/selected_runtime.json`.
- 2026-06-20 local v8 evidence-plumbing fix:
  implemented the bounded gate-health quantile path in
  `src/eqvae/benchmarking/real_data_runtime_pretest.py` with
  `MAX_GATE_QUANTILE_ELEMENTS = 1_000_000`, sampled deterministically only for
  `gate_p01/gate_p50/gate_p99` telemetry. Full-tensor finite checks,
  saturation fractions, worst-channel saturation fractions, dead-channel
  checks, and row evidence semantics are preserved. Removed the unsafe
  `_row_gate_status` lane-level fallback so uncovered bs8/bs12-style rows keep
  `gate_health_status = skipped_unsupported` and cannot become eligible via a
  lane-level pass. Added focused tests for exact small quantiles, sampled
  large-tensor quantiles using a tiny monkeypatched cap, bounded
  `failure_message_excerpt` preservation, and missing row-specific gate-health
  coverage. Updated `scripts/kaggle_kernel.sh` so local real-data pretest
  `build`/`validate` can verify a generated payload against the current dirty
  worktree; the push guard still rejects dirty manifests before any remote
  write. Verification passed:
  `PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/pytest tests/test_real_data_runtime_pretest.py tests/test_kaggle_embedded_kernel.py -q`
  (`20 passed`), `./scripts/python_quality.sh` (`126 passed`, BasedPyright
  `0 errors`), `git diff --check`, `bash -n scripts/kaggle_kernel.sh`,
  `./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest`,
  `./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest`,
  `./scripts/agent_preflight.sh`, and workspace `./agent_preflight.sh`.
  The exact guarded remote sequence used for v8, and required again for any
  rerun or successor remote slice, is:
  `git status --short` and clean/commit local changes first; rebuild and
  validate locally; after explicit permission run
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check`; confirm
  Kaggle web UI GPU quota if the CLI quota endpoint still warns; after explicit
  push permission run
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`;
  monitor only with approved
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest`
  at the documented cadence; download once with
  `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v8`
  or the matching new output directory.
  Do not run any Kaggle remote action without explicit permission and required
  guards. v8 remains non-promotable unless a later spec explicitly changes that.
- 2026-06-20 v8 Kaggle push, download, and inspection:
  committed the local fix as `614cd95` (`Fix real-data pretest gate quantile
  evidence`), rebuilt/validated `kaggle/kernels/real_data_runtime_pretest` from
  a clean source state, ran the approved `KAGGLE_REMOTE_CONFIRMED=1
  ./scripts/kaggle_kernel.sh api-check` preflight, and pushed with approved
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`. Kaggle accepted
  version 8 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  Approved status reads showed `KernelWorkerStatus.RUNNING`, then
  `KernelWorkerStatus.COMPLETE`; outputs were downloaded to
  `runs/kaggle/real_data_runtime_pretest_v8`. Artifact inspection:
  `benchmark/selected_runtime.json` is absent, `runtime_proof.status =
  pretest_incomplete`, `selection_ready = false`, `selected_runtime_written =
  false`, `full_run_eligible = false`, `eligible_pass_row_count = 6`,
  `paired_numerical_candidate_evidence_count = 7`,
  `paired_numerical_failed_candidate_evidence_count = 0`,
  `corruption_equivalence_candidate_evidence_count = 7`,
  `corruption_equivalence_failed_candidate_evidence_count = 0`, and
  `gate_health_status = pass`. Passing rows are eager single-visible-T4 FP32
  `compile_none` bs4/bs8/bs12 for both `branchless_all` and
  `indexed_masked`; bs12 is fastest among passing rows in this capped pretest
  (`~7.39 samples/s` for indexed, `~7.35 samples/s` for branchless). Compiled
  `model_forward` bs4/bs8/bs12 rows now have numerical/corruption/gate-health
  evidence but remain ineligible because compile-settle/Dynamo evidence is not
  row-pass; compiled bs32 remains ineligible and eager bs32 remains
  `runtime_OutOfMemoryError`. Dual-T4 rows remain
  `dual_t4_ddp_train_step_measurement_pending`. `phase_timings.json` records
  `71` phases and `2761.447838` seconds of script elapsed time. v8 fixed the
  v7 quantile evidence-plumbing failure but is still not selected-runtime
  evidence.
- 2026-06-20 selected-runtime benchmark/debug slice decision:
  encoded the next slice in
  `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` as
  `v8_shortlist_eager_amp_then_dual_gate` and added regression coverage in
  `tests/test_spec0001_benchmark_scaffold.py`. The slice treats v8 artifacts as
  `candidate_shortlist_only`: v8 remains `pretest_incomplete`,
  `full_run_eligible = false`, and `writes_selected_runtime = false`.
- 2026-06-20 selected-runtime benchmark local implementation:
  added `src/eqvae/benchmarking/runtime_selection.py`,
  `src/eqvae/cli/runtime_selection_benchmark.py`, and
  `tests/test_runtime_selection_benchmark.py`. The local writer records v8
  artifact hashes as provenance only, writes its own runtime proof, runtime
  matrix, dataloader matrix, numerical checks, corruption checks, gate-health
  summary/rows, and model-count evidence, rewrites compiled pass rows to
  diagnostic `ineligible`, and refuses `benchmark/selected_runtime.json` unless
  dual timing and all linked proof gates pass. After adversarial review, the
  local proof gate now also requires train and validation dataloader rank
  coverage, candidate-bound gate-health row ids, child-process
  `torchrun --nproc_per_node=2` proof, scoped numerical/corruption rows, and a
  hash-linked `benchmark/stain_corruptor_qa.json`. The adversarial follow-up
  hardening now requires 25 measured dataloader batches plus wait/throughput
  thresholds, three numerical batch indices, train and validation
  corruption-check rows with clean-validation RNG unchanged, gate-health
  `candidate_row_id` binding, strict stain-QA candidate coverage, and exact
  embedded v8 payload membership. The default local CLI run is intentionally
  fail-closed because no real dual-T4 train-step evidence has been supplied.
- 2026-06-20 selected-runtime Kaggle executor/kernel implementation:
  added `src/eqvae/benchmarking/runtime_selection_executor.py`,
  `src/eqvae/cli/runtime_selection_executor.py`, and
  `kaggle/kernels/runtime_selection`. The executor reuses the v8 shortlist only
  as hashed provenance, collects its own single-visible-T4 eager FP32/AMP
  follow-up evidence, runs the real dual-T4 DDP train-step timing gate through
  a `torch.distributed.run --nproc_per_node=2` child process, and feeds the
  strict local writer so `benchmark/selected_runtime.json` remains refused if
  any linked proof is missing, failed, or skipped. The Kaggle wrapper embeds
  the required v8 provenance files, validates the output set fail-closed, and
  `scripts/kaggle_kernel.sh` now has guarded `status-runtime-selection` and
  `output-runtime-selection` actions plus exact v8 payload and specs-index
  guard checks. Adversarial review found candidate-bound gate-health,
  dataloader-depth, corruption-scope, stain-QA, numerical-depth, payload-set,
  and spec-index guard gaps; these were fixed and the follow-up review found
  only a dataloader counter shadowing bug, which was also fixed. Verification
  passed: targeted pytest for runtime-selection/Kaggle embedding tests,
  fail-closed local executor smoke, clean rebuild/validate of
  `kaggle/kernels/runtime_selection`, `./scripts/python_quality.sh` with
  `140 passed`, `git diff --check`, repo `./scripts/agent_preflight.sh`, and
  workspace `/home/maximus/Documents/Tesis/agent_preflight.sh`. Commit
  `fba9d98` was created, and Kaggle accepted version 1 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection`.
  Runtime-selection v1 later reached `KernelWorkerStatus.ERROR`; outputs were
  downloaded to `runs/kaggle/runtime_selection_v1`. Inspection found the real
  dual-T4 DDP timing gate passed and emitted the required bs4/bs8/bs12 FP32
  eager dual rows with two visible T4s, `world_size = 2`,
  `nproc_per_node = 2`, per-rank device assignment, child-process launch proof,
  and global throughput projection. The strict writer still refused
  `benchmark/selected_runtime.json` because linked single-visible proof rows
  were false-negative blocked by gate-health eligibility normalization and the
  train corruption clean-validation RNG check. The wrapper error was
  `unexpected runtime-selection benchmark artifacts:
  unexpected=['model_inventory.csv']`. The local v2 fix adds
  `model_inventory.csv` to the wrapper allow-list, scopes the corruption RNG
  requirement to validation rows, normalizes `local_pass` gate-health rows
  before computing eligibility, and keeps failed non-gate rows ineligible even
  if they carry a gate-health status. New focused regressions cover all three
  v1 failures. Two clean-context adversarial subagent reviews found no
  selected-runtime fail-open blocker; one low semantic eligibility issue was
  fixed before the v2 push. Commit `96e41f4` was then created, rebuilt,
  validated, and pushed as runtime-selection Kaggle version 2. Version 2
  completed and downloaded to `runs/kaggle/runtime_selection_v2`. It fixed the
  wrapper allow-list error and preserved the passing real dual-T4 DDP timing
  proof, but the strict writer still refused `benchmark/selected_runtime.json`
  with blocker `runtime_pass_rows_linked_proof_not_pass` because gate-health
  rows were missing for the three single-visible `indexed_masked` pass rows:
  bs4, bs8, and bs12 FP32 eager. The local v3 follow-up adds a
  runtime-selection executor helper that expands branchless single-visible
  gate-health rows to same-shape indexed candidate row ids only after the
  indexed runtime row has already passed linked evidence; a focused regression
  covers that exact v2 artifact shape. A follow-up adversarial subagent review
  found no selected-runtime fail-open blocker and recommended explicit
  FP32/eager/indexed guard rails plus negative tests; those were added. Commit
  `b6b024a` was created, the runtime-selection kernel was rebuilt/validated from
  the clean commit, and Kaggle accepted version 3. The agent stopped waiting
  after the immediate `RUNNING` status and gave the user a prompt time rather
  than idling in-turn. On resume, v3 was `KernelWorkerStatus.COMPLETE`; outputs
  were downloaded to `runs/kaggle/runtime_selection_v3`. Inspection confirmed
  `runtime_proof.status = pass`, `selection_ready = true`,
  `selected_runtime_written = true`, no write-decision blockers, dual-T4 gate
  `status = pass`, single-visible confirmation `status = pass`,
  `stain_corruptor_qa_status = pass`, and `benchmark/selected_runtime.json`
  present. The selected runtime is
  `dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`, with
  per-device batch size 12, global batch size 24, two visible Tesla T4s,
  `world_size = 2`, `nproc_per_node = 2`, FP32 eager/no compile,
  `samples_sec = 14.035497`, projected epoch time about 356.24 minutes, and
  projected 10-epoch wall time about 59.37 hours.
- 2026-06-19 GitHub issue status updates:
  posted Spanish status comments to issues #1-#6 after local grounding and
  three read-only subagent audits. Issue #2 received the substantive v6
  real-data pretest update; issues #1, #3, #4, #5, and #6 explicitly record
  that no final conference/paper/metric/visual/SO(2) deliverable changed this
  week. Posted comments:
  #1 `https://github.com/HiperMaximus/equivariant-vae/issues/1#issuecomment-4755254478`,
  #2 `https://github.com/HiperMaximus/equivariant-vae/issues/2#issuecomment-4755254390`,
  #3 `https://github.com/HiperMaximus/equivariant-vae/issues/3#issuecomment-4755254304`,
  #4 `https://github.com/HiperMaximus/equivariant-vae/issues/4#issuecomment-4755254416`,
  #5 `https://github.com/HiperMaximus/equivariant-vae/issues/5#issuecomment-4755255865`,
  and #6
  `https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-4755255877`.
- 2026-06-19 SIPAIM paper scaffold:
  added `docs/specs/0004-sipaim-paper-scaffold.md` and indexed it in
  `docs/specs/README.md`; replaced the generic SIPAIM `main.tex` template with
  a compile-safe three-page scaffold titled "Comparing Equivariant and
  Non-Equivariant VAE Representations for Semi-Supervised Histopathology WSI
  Classification"; added first IEEE BibTeX entries to
  `paper/sipaim2026/references.bib`; copied thesis figures into
  `paper/sipaim2026/figures/`; and refreshed
  `paper/sipaim2026/sipaim2026.pdf`. The scaffold uses the user's thesis
  framing: unsupervised VAE representation learning first, then a supervised
  WSI classifier on encoder embeddings. It explicitly marks selected runtime,
  full VAE runs, continuous `SO(2)` results, downstream classifier evidence,
  and sealed masked-WSI test evidence as pending. Verification:
  `./scripts/sipaim_overleaf_sync.sh compile` passed after regenerating the
  stale local `main.bbl`, the final LaTeX log has no warnings/errors, `pdfinfo`
  reports 3 pages, and rendered PDF pages were visually checked. The thesis repo
  was read-only and its worktree stayed clean. The reused method figure still
  contains Spanish thesis text and should be redrawn or relabeled before final
  submission. Final hygiene for this handoff: `git diff --check`,
  `./scripts/agent_preflight.sh`, and workspace `./agent_preflight.sh` all
  passed.
- 2026-06-19 commit and remote-sync blocker:
  committed the Kaggle diagnostics, SIPAIM paper scaffold, paper-local figures,
  bibliography entries, refreshed PDF, and handoff docs as
  `764095e` (`Scaffold SIPAIM paper and record v6 diagnostics`). Local
  `./scripts/sipaim_overleaf_sync.sh check` passed with a clean repo and the
  expected Overleaf remote/subtree. The user then asked to update paper-related
  GitHub issues and send the paper to Overleaf. Two early escalated attempts to
  run `OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push` timed
  out in the sandbox approval reviewer before the command started. A later
  guarded push attempt did start, recompiled the paper successfully, and then
  failed at Git HTTPS authentication with
  `fatal: could not read Username for 'https://git.overleaf.com': No such device or address`.
  No Overleaf remote write was performed, so the Overleaf web project can still
  appear empty even though the local paper scaffold and tracked PDF are
  committed. The configured credential helper is `cache --timeout=86400`, but no
  usable Overleaf Git credential was available to the agent shell; browser login
  does not authenticate Git HTTPS. GitHub issue updates were not attempted
  afterward because they should not claim the professor-facing Overleaf project
  is updated until the push succeeds. Next action: have the user prime the local
  Git credential helper for `https://git.overleaf.com` without pasting tokens in
  chat, or have them run the guarded push once in their own terminal:
  `OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push`. After the
  push succeeds, post Spanish paper-status comments to the paper-related GitHub
  issues, especially #5 (SIPAIM writing) and any paper-figure/evaluation issues
  such as #3/#4 as appropriate. The issue comment should mention the Overleaf web
  URL `https://www.overleaf.com/project/69c614433cbc9e46cf226d24`, the
  scaffolded `paper/sipaim2026` files, the refreshed PDF, and that the reused
  thesis method figure still needs to be redrawn or relabeled before submission.
- 2026-06-19 Overleaf token-auth correction:
  the user retried the guarded push and Overleaf returned HTTP 403 with the
  message that Git now supports only authentication tokens. Overleaf Git must
  use username `git` and an Overleaf Git authentication token as the password;
  the normal Overleaf account password/email login will fail. If a wrong
  credential was cached, clear only the `git.overleaf.com` cached credential
  with `printf 'protocol=https\nhost=git.overleaf.com\n\n' | git credential reject`,
  then rerun the guarded push. `docs/overleaf_sync_workflow.md` and
  `scripts/sipaim_overleaf_sync.sh` now record this token-auth rule.
- 2026-06-19 persistent Overleaf credential helper:
  the user installed `libsecret-1-dev`, `libsecret-tools`, and `pkg-config` so
  the Overleaf Git token can be stored in the desktop keyring instead of a
  plaintext ignored file. The Git `libsecret` helper was compiled to
  `/home/maximus/Documents/Tesis/.agent-tools/git-credential-libsecret`, and
  this repo's local Git config now resets inherited helpers for
  `https://git.overleaf.com`, uses that helper, and defaults the username to
  `git`. The token itself was entered by the user through `git credential
  approve` and stored in the desktop keyring; it was not pasted in chat and must
  never be printed or committed.
- 2026-06-19 first Overleaf push edge case:
  after the user stored the Overleaf token in the keyring, the guarded push
  authenticated and reached Overleaf but was rejected as non-fast-forward because
  Overleaf `master` already had commit `95e5ec4` with an empty tree and no
  common subtree history. The guarded `pull` fetched `overleaf/master` but
  failed with `fatal: can't squash-merge: 'paper/sipaim2026' was never added`,
  which is expected for a first sync over an unrelated empty remote. The safe
  first guarded empty-tree fallback tried `--force-with-lease`, but Overleaf
  rejected all forced pushes. The script is being updated so `push` first tries
  a normal subtree push and then, only when the current Overleaf `master` branch
  is provably an empty tree, creates a normal fast-forward commit on top of the
  observed empty Overleaf commit using the checked paper subtree tree. Spec
  `docs/specs/0005-overleaf-empty-project-initialization.md` records the narrow
  contract: master-only, exact observed empty commit, subtree split sanity check,
  and no overwrite path for nonempty Overleaf content. Do not force-push
  Overleaf.
- 2026-06-19 Overleaf sync and issue updates complete:
  after switching the first-sync fallback to a normal fast-forward commit on top
  of the empty Overleaf commit, `OVERLEAF_SYNC_CONFIRMED=1
  ./scripts/sipaim_overleaf_sync.sh push` succeeded. Overleaf `master` now points
  at `b4a8954fc4fcaa969757ac20adf84fa0fdbac6db`, and the project should show
  the `paper/sipaim2026` subtree at
  `https://www.overleaf.com/project/69c614433cbc9e46cf226d24`. Spanish status
  comments were posted without closing issues:
  #5 `https://github.com/HiperMaximus/equivariant-vae/issues/5#issuecomment-4755517582`,
  #1 `https://github.com/HiperMaximus/equivariant-vae/issues/1#issuecomment-4755518028`,
  #4 `https://github.com/HiperMaximus/equivariant-vae/issues/4#issuecomment-4755518529`,
  and #3
  `https://github.com/HiperMaximus/equivariant-vae/issues/3#issuecomment-4755518912`.
  The issue updates mention the Overleaf URL, the scaffolded paper files, the
  refreshed PDF, pending final results, and that the reused thesis
  semi-supervised figure still needs to be redrawn or relabeled before
  submission. Local `main` is ahead of `origin/main`; push to GitHub origin if
  the GitHub repo itself should show the latest paper scaffold/workflow commits.
- 2026-06-20 selected-runtime efficiency clarification:
  runtime-selection v3 is the proof-clean baseline, but not necessarily the row
  to spend 60h+ on. Before the first expensive training launch, run an
  efficiency-selection follow-up that tries AMP/FP16, stable `torch.compile`,
  channels-last layout, cuDNN benchmark/non-deterministic kernel selection, DDP
  `static_graph`/`gradient_as_bucket_view`, optimizer/zero-grad fast paths, and
  Kaggle-supported TF32/matmul precision knobs. The user explicitly accepts lost
  bitwise determinism and small numerical drift for material speedups; only
  catastrophic failures such as non-finite loss/gradients, repeated AMP skips,
  DDP instability, broken checkpoint/resume, broken artifacts, gate-health
  collapse, or clearly invalid metrics should block a faster row.
- 2026-06-21 local selected-runtime efficiency follow-up implementation:
  implemented `selected_runtime_v3_efficiency_followup` in
  `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`, the
  runtime-selection writer, and the runtime-selection executor. Runtime rows now
  carry first-class `runtime_policy_id` and fast-path metadata; dataloader,
  numerical, corruption, and gate-health proof rows bind to that policy id.
  The DDP rank path now applies channels-last, cuDNN benchmark/deterministic
  flags, DDP `static_graph`/`gradient_as_bucket_view`, AMP/FP16 with GradScaler
  and AMP skip telemetry, stable model-forward `torch.compile` settle counters,
  AdamW foreach/fused probes, zero-grad/clip fast paths, and TF32/matmul knobs.
  The writer allows stable compiled rows only with explicit policy id,
  sufficient settle steps, and zero post-settle graph breaks/recompiles; AMP
  skips and policy-mismatched linked proofs block selection. The selected
  runtime payload records relaxed determinism and keeps
  `full_training_launch_ready = false` until debug/resume/tiny-overfit proof.
  No Kaggle/GitHub/Overleaf remote action was run. Verification:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_runtime_selection_benchmark.py`,
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_kaggle_embedded_kernel.py`,
  `./scripts/python_quality.sh`, repo `./scripts/agent_preflight.sh`, and
  workspace `./agent_preflight.sh` all pass. Superseded next action:
  runtime-selection v5 is now the fallback selected runtime, and the active
  local successor is the compact `amp_fp16_scalar_gate_relaxed` comparison plus
  later debug/resume/tiny-overfit proof gates before any long run is requested.
- 2026-06-21 selected-runtime efficiency follow-up Kaggle launch and v4 result:
  after the Einstein adversarial FSQ review completed, the user approved the
  efficiency follow-up only, not the first 60h-scale real run. The first push
  attempt was blocked locally before any Kaggle write because the push guard
  rejects dirty payload manifests. The full repo quality gate then passed:
  `./scripts/python_quality.sh` reported 147 pytest tests passed plus 0 type
  errors/warnings/notes. Local commit `753c9db`
  (`Add runtime selection efficiency follow-up`) was created, the
  runtime-selection kernel was rebuilt and revalidated from the clean commit,
  and Kaggle accepted version 4 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection` with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`. The one guarded
  post-push status read with `KAGGLE_REMOTE_CONFIRMED=1` returned
  `KernelWorkerStatus.RUNNING` at 2026-06-21 00:46:21 -05. Per the long-job
  rule, the agent stopped and asked for a later `continue`. On resume, the
  guarded status read returned `KernelWorkerStatus.COMPLETE`, and outputs were
  downloaded to `runs/kaggle/runtime_selection_v4`. Version 4 failed closed:
  `runtime_proof.status = fail`, `selected_runtime_written = false`, and no
  `benchmark/selected_runtime.json` was written. The intended fastest clean row
  was
  `dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
  with `samples_sec = 25.220604`, `amp_step_skipped_count = 0`, gate health
  `pass`, and estimated 10-epoch wall time `118950.362625` seconds. The proof
  blockers were writer policy false negatives: tiny bounded numerical drift on
  the selected AMP row plus linked proof failures from nonselected rows were
  treated as global blockers.
- 2026-06-21 selected-runtime v4 proof repair and v5 launch:
  Noether adversarial review agreed that v4 failed closed rather than selecting
  an invalid row. Local commit `fc5227d`
  (`Relax runtime selection numerical drift gate`) scopes linked proof to the
  selected candidate, accepts only finite bounded small numerical drift, keeps
  AMP skips and large drift as row blockers, and lets skipped AMP rows fail
  row-local selection without globally rejecting a safe baseline or alternate
  row. Focused tests passed (`20 passed`), the full repo gate passed
  (`./scripts/python_quality.sh`: 149 pytest tests, 0 type
  errors/warnings/notes), and local replay of v4 artifacts through the patched
  writer produced `runtime_proof.status = pass` plus selected row
  `dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`.
  The runtime-selection kernel was rebuilt/validated from the clean commit, and
  Kaggle accepted version 5 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection` with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`. The one guarded
  post-push status read with `KAGGLE_REMOTE_CONFIRMED=1` returned
  `KernelWorkerStatus.RUNNING` at 2026-06-21 06:14:37 -05. Per the long-job
  rule, the agent stopped after that single poll. On the next approved status
  read, Kaggle reported `KernelWorkerStatus.COMPLETE`; outputs were downloaded
  to `runs/kaggle/runtime_selection_v5`. Version 5 wrote
  `benchmark/selected_runtime.json`, `runtime_proof.status = pass`,
  `selection_ready = true`, and `selected_runtime_written = true`. The selected
  row is
  `dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
  with `runtime_policy_id = amp_fp16_conservative`, `samples_sec = 27.381321`,
  `steady_step_ms_p50 = 876.509927`, estimated 10-epoch wall time
  `109563.740875` seconds, zero AMP skips, no OOM, gate-health pass, and strict
  local replay pass under current `main`. Selected numerical checks include one
  expected bounded `dual_t4_numerical_delta_failed` row and two pass rows; all
  selected numerical rows have `nonfinite_count = 0`, `amp_step_skipped =
  false`, and `gate_health_status = pass`. The selected payload still has
  `full_training_launch_ready = false` with blockers for missing
  selected-runtime debug, checkpoint/resume, and tiny-overfit proof.
- 2026-06-21 compact relaxed-AMP runtime-selection v6 result:
  after the user approved the compact relaxed-AMP follow-up only, the first
  guarded push attempt was blocked locally before any Kaggle write because the
  payload manifest was dirty. The local follow-up patch was committed as
  `580a844` (`Add compact relaxed AMP runtime selection follow-up`), the clean
  commit passed `./scripts/kaggle_kernel.sh preflight-runtime-selection`
  (`32 passed`), and Kaggle accepted version 6 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection` with
  `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`. The one guarded
  post-push status read with `KAGGLE_REMOTE_CONFIRMED=1` returned
  `KernelWorkerStatus.RUNNING` at 2026-06-21 14:53:39 -05. On user `continue`,
  a guarded status read returned `KernelWorkerStatus.COMPLETE`; outputs were
  downloaded to `runs/kaggle/runtime_selection_v6`. Artifact inspection found
  no `benchmark/selected_runtime.json`; `runtime_proof.status = "fail"`,
  `selection_ready = false`, `selected_runtime_written = false`, and
  `selected_runtime_write_decision.blockers` contains
  `selected_runtime_reuses_configured_baseline_no_replacement`. The relaxed row
  `dual_t4_ddp__bs12__amp_scalar_gate_relaxed__compile_none__indexed_masked__policy_amp_fp16_scalar_gate_relaxed`
  passed runtime/gate-health with zero AMP skips but reached only `25.288828`
  samples/sec against the embedded v5 baseline's `27.381321`, and one of its
  three numerical batches was the expected bounded
  `dual_t4_numerical_delta_failed`. Strict local replay to
  `/tmp/eqvae_runtime_selection_v6_replay` regenerated the same fail-closed
  proof and no selected runtime. Keep v5 as the fallback selected runtime; next
  action is selected-runtime debug, checkpoint/resume, artifact, and
  tiny-overfit implementation/proof before any long real training run.
- 2026-06-21 selected-runtime debug/resume/tiny-overfit local contract runner:
  `src/eqvae/cli/train.py` and `src/eqvae/training/debug.py` now provide a
  short synthetic proof runner that consumes
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`, writes
  `benchmark/training_summary.json`,
  `benchmark/selected_runtime_debug_summary.json`,
  `benchmark/checkpoint_resume_proof.json` when resuming,
  `benchmark/tiny_overfit_summary.json` for the fixed-32 path,
  `benchmark/artifact_manifest.json`, metrics CSVs, checkpoints, and a
  nonblank reconstruction artifact. Checkpoints now save and restore model,
  optimizer, Python RNG, explicit NumPy `Generator` state, global Torch CPU RNG
  state, and named Torch `Generator` states such as `train_data`; they also
  store selected-runtime config hash, row id, and policy id, and resume
  validates that metadata before restoring state. CUDA RNG remains a
  real-Kaggle-runner requirement. The runner rejects real `ubc-pre-shuffled`
  execution for now and marks all local proof artifacts
  `full_run_eligible = false`, so the real UBC/Kaggle debug/resume/tiny-overfit
  gate remains pending and permission-gated. A separate
  `configs/spec0001/non_eq_vae_selected_runtime_debug.json` now makes selected
  runtime consumption mandatory without changing the old capped-smoke config.
  The implementation also hardens relaxed scalar-gate AMP selection: every
  candidate-bound scalar gate row must carry gate dtype proof showing
  `gate_force_fp32 = false` and gate/input/output dtype matching the requested
  autocast dtype; a single missing or FP32 proof blocks the row. After
  adversarial review, resume is now runtime-bound before state restore, explicit
  seed `0` is preserved, and direct RNG-stream tests cover Python, NumPy
  `Generator`, global Torch CPU RNG, and named Torch `Generator` restore.
  Focused tests passed (`58 passed`), full `./scripts/python_quality.sh` passed
  (`177 passed`, 0 type errors), and
  `./scripts/kaggle_kernel.sh preflight-runtime-selection` passed (`35 passed`).
- 2026-06-22 selected-runtime debug/tiny gate contract:
  Spec 0001, Spec 0003, the spec index, and `docs/kaggle_cli_workflow.md` now
  define `selected_runtime_debug_gate_contract_ready`. The new dedicated Kaggle
  kernel directory is `kaggle/kernels/selected_runtime_debug` with metadata id
  `maximusshtefan/eqvae-selected-runtime-debug` and a generated single-file
  wrapper that embeds
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
  `scripts/build_kaggle_embedded_kernel.py` now embeds the v5 selected-runtime
  artifact for this kernel, and `scripts/kaggle_kernel.sh` has
  `preflight-selected-runtime-debug`, `status-selected-runtime-debug`, and
  `output-selected-runtime-debug` actions. The preflight is local-only: it
  builds/validates the wrapper, runs `tests/test_selected_runtime_gate.py`, and
  runs generated wrapper import-only plus full local fail-closed simulations.
  It does not use Kaggle network APIs.
  The new CLI `python -m eqvae.cli.selected_runtime_gate` writes exact
  fail-closed artifacts:
  `benchmark/selected_runtime_gate_summary.json`,
  `benchmark/training_summary.json`,
  `benchmark/selected_runtime_debug_summary.json`,
  `benchmark/checkpoint_resume_proof.json`,
  `benchmark/tiny_overfit_summary.json`,
  `benchmark/gate_health_summary.json`,
  `benchmark/artifact_manifest.json`, `metrics/train_metrics.csv`, and
  `metrics/gate_health.csv`, without writing `benchmark/selected_runtime.json`.
  It validates the v5 selected-runtime transport including top-level launch
  fields, mixed precision, dataloader, corruption, selected row snapshot, and
  safety statuses. `artifact_manifest.json` remains `status = "fail"` with
  `contract_written = true` while the real proof is absent.
  The selected-runtime debug/tiny configs now carry explicit
  `remote_pass_ready = false`, `real_train_runner_implemented = true`, and
  `fixed_32_selector_real = false`; the remaining implementation blocker is
  `selected_runtime_debug_wrapper_not_wired_to_real_runner_until_spec0008`.
  The shell push guard checks those structured flags, the embedded v5 payload,
  metadata, docs tokens, and now delegates semantic push readiness to
  `python -m eqvae.cli.selected_runtime_gate --verify-push-ready`. That
  verifier is deliberately narrower than full proof pass: it blocks on invalid
  selected-runtime transport, missing wrapper/plan capability, non-ready
  config flags, and invalid fixed-32 selector readiness, but not on remote
  debug/resume/gate/tiny artifacts that the selected-runtime debug run itself
  would have to produce.
  Fixed-32 selector readiness is now fail-closed in two layers. Fabricated
  32-row JSON fails schema parsing, and schema-valid local/synthetic selector
  replay still fails unless the selector matches the locked canonical real UBC
  train-shard fingerprints from the downloaded real-data identity proof:
  dataset slug, train split, expected filenames, 300000 rows/patches,
  256x256x3 CHW header, CRC-checked source, train CSV SHA-256, binary size, and
  header CRC. Therefore remote push remains intentionally blocked even after
  local preflight passes.
  Verification passed:
  `bash -n scripts/kaggle_kernel.sh scripts/agent_preflight.sh`,
  `PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_gate.py -q`
  (`7 passed`),
  the structured `eqvae.cli.selected_runtime_gate --verify-push-ready` check
  (expected exit 1 with only real-runner/plan, placeholder-selector, and
  config-readiness blockers),
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug` (`9 passed`),
  `./scripts/python_quality.sh` (`186 passed`, 0 type errors), and
  `git diff --check`. Repo `./scripts/agent_preflight.sh` and workspace
  `./agent_preflight.sh` passed; both noted the expected dirty worktree and
  confirmed the generated selected-runtime debug `run.py` is ignored.
  Four adversarial clean-context subagent reviews were run across the contract
  and hardening passes. Integrated fixes include: future-only push wording,
  locally coherent/wrapper-buildable preflight wording, explicit non-ready
  config flags, top-level selected-runtime transport validation, failed
  artifact manifest status, structured config readiness, structured push
  readiness, and canonical-real-UBC selector fingerprints. Remaining known
  follow-ups from review are future-pass hardening: implement the real
  UBC/DDP/AMP train/resume/tiny runner, generate the real fixed-32 selector from
  Kaggle shards, replace placeholder DDP/accelerator text hooks with structured
  real proof artifacts, and add shell-level guard tests once the real runner
  starts to unlock.
- 2026-06-22 selected-runtime local-mechanics spec review:
  The next coding slice is now locked as
  `docs/specs/0006-selected-runtime-local-mechanics.md`, while broad Spec 0001
  remains draft active. Spec 0006 splits the work into local-only sub-slices:
  shared v5 `SelectedRuntimePlan`, plan-applied proof, synthetic UBC-format
  train/validation mechanics, selected train corruption plus clean validation
  isolation, AMP skip/progress semantics, checkpoint schema v5, and structured
  local readiness aggregation. It explicitly forbids Kaggle remote action, long
  training, `full_run_eligible = true`, `benchmark/selected_runtime.json`, and
  readiness-flag flips in this slice.
  Spec 0001 now points to Spec 0006 as the implementation-ready child spec and
  clarifies that pass-capable training evidence uses `metrics/train_steps.csv`;
  the current selected-runtime gate `metrics/train_metrics.csv` remains a
  fail-closed contract/preflight artifact unless later migrated. Spec 0003 and
  the spec index now state that a future selected-runtime debug/tiny remote push
  may be requested only after `preflight-selected-runtime-debug` and the
  structured selected-runtime push-readiness gate pass, the shell push guard is
  expected to pass, real-runner and selector readiness are proven, and the user
  explicitly approves. At that time, naming Spec 0006 as the immediate next
  action was correct; the current active state above now supersedes it with
  Spec 0007 followed by Spec 0008.
  Two adversarial subagent reviews were run on the spec/memory plan. Integrated
  fixes: split the broad checklist into locked child Spec 0006, tightened
  remote-approval wording, relabeled a stale historical `Immediate next action`,
  resolved the `train_metrics.csv`/`train_steps.csv` ambiguity, and split the
  local mechanics work into atomic sub-slices. Verification for this docs-only
  review passed: `git diff --check`, repo `./scripts/agent_preflight.sh`, and
  workspace `./agent_preflight.sh`.
- 2026-06-22 selected-runtime local-mechanics implementation:
  Spec 0006 is implemented and locally verified. The new shared parser lives in
  `src/eqvae/training/selected_runtime.py` and validates the v5 selected row,
  runtime policy, launch settings, nested safety/runtime policy/compile fields,
  and linked `benchmark/runtime_proof.json` command
  `torchrun --standalone --nproc_per_node=2`. The train path in
  `src/eqvae/training/debug.py` applies the locally executable selected-runtime
  settings, writes `metrics/train_steps.csv`, emits a full plan-applied proof
  that rejects unexecuted dual-T4 CUDA AMP/DDP fields on local CPU, creates
  synthetic UBC-format shards through the canonical resolver, trains with
  `PatchTrainingDataset`/collate/normalization and selected `indexed_masked`
  train corruption, keeps validation clean without corruption RNG advancement,
  supports an integrated simulated AMP skip seam, and writes structured local
  readiness from proof/artifact status. Checkpoints now use schema
  `spec0001.checkpoint.v5` and reject missing/mismatched selected-runtime state
  before model/optimizer restore. The selected-runtime debug kernel payload now
  embeds both v5 `selected_runtime.json` and its linked `runtime_proof.json`.
  Adversarial review initially found four issues: self-fulfilling plan-applied
  proof fields, missing standalone torchrun validation, helper-only AMP skip
  semantics, and hard-coded readiness manifest status. Follow-up review still
  objected to plan proof overclaiming nonlocal fields; the proof now observes
  local CPU values and fails those full-runtime comparisons instead of marking
  them applied. Verification passed after those fixes:
  `PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_gate.py tests/test_train_cli.py tests/test_train_step.py -q`
  (`30 passed`), `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`
  (`12 passed`), `./scripts/python_quality.sh` (`192 passed`, 0 type errors),
  `git diff --check`, repo `./scripts/agent_preflight.sh`, and workspace
  `./agent_preflight.sh`.
- 2026-06-28 Spec 0008 local readiness implementation handoff:
  Spec 0008 local-first readiness is implemented and locally verified without
  Kaggle, network, remote approval, or long training. New fixed-32 readiness
  code lives in `src/eqvae/benchmarking/fixed32_selector_readiness.py` and
  `src/eqvae/cli/fixed32_selector_readiness.py`; it generates deterministic
  synthetic UBC-format fixed-32 selectors, proves schema replay, and proves
  synthetic selectors fail canonical-real UBC readiness. The selected-runtime
  gate now supports `selector_generation_mode = "remote_generate"`, structured
  local readiness, and `eqvae.cli.selected_runtime_gate --verify-output` for
  downloaded remote artifacts. The selected-runtime debug wrapper now
  remote-generates the canonical selector before training, validates it with
  strict real UBC fingerprints, passes the generated selector into
  `eqvae.cli.selected_runtime_train`, runs the bounded 4->8 resume debug proof
  plus a separate capped 128-step tiny-overfit phase, writes final gate and
  artifact-manifest summaries, and rejects failed plan/gate/manifest/tiny
  artifacts. The runner now accepts `--fixed-train-patches`, restricts training
  to the fixed selector rows, records selector path/hash/count, and reports
  DDP-safe optimizer-step counts. The shell guard now runs
  `preflight-fixed32-selector-readiness` and checks the embedded wrapper for the
  real runner, selector, resume, tiny, and output-verifier paths.
  Adversarial reviews found missing tiny-overfit execution, selector not being
  passed into the runner, missing `--verify-output`, DDP metric-row overcount,
  and static-only wrapper checks; those findings were fixed. Verification
  passed: focused Spec 0008 tests (`34 passed`), `./scripts/kaggle_kernel.sh
  preflight-fixed32-selector-readiness`, `./scripts/kaggle_kernel.sh
  preflight-selected-runtime-debug` (`24 passed` after wrapper rebuild),
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` (`43 passed`
  plus bounded dry-run), explicit `eqvae.cli.selected_runtime_gate
  --verify-push-ready --selector-generation-mode remote_generate`, full
  `./scripts/python_quality.sh` (`215 passed`, 0 type errors),
  `git diff --check`, and repo `./scripts/agent_preflight.sh`.
  Remaining work is remote-only and approval-gated: ask the user before any
  Kaggle push/read/download, run the narrow selected-runtime debug/tiny kernel,
  download artifacts, run `--verify-output`, inspect them, and only then decide
  whether the first full real selected-runtime run is the next action.
- 2026-06-28 Spec 0007 implementation handoff:
  Spec 0007 is implemented locally. New code lives in
  `src/eqvae/training/selected_runtime_runner.py` and
  `src/eqvae/cli/selected_runtime_train.py`, with tests in
  `tests/test_selected_runtime_runner.py` and local preflight wiring in
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`. The runner
  consumes v5 `SelectedRuntimePlan`, supports synthetic dry-run and real
  `ubc-pre-shuffled` roots, builds the exact
  `torchrun --standalone --nproc_per_node=2` launch proof, validates real
  distributed initialization/rank-device mapping, applies selected DDP
  `static_graph = false` and `gradient_as_bucket_view = false`, keeps rank 0
  as the only artifact writer, gathers per-rank train/gate rows, records
  selected-runtime AMP/CUDA/DDP checkpoint-state statuses, bounds AMP-skip
  retries, and blocks readiness on any AMP skip. The fail-closed debug gate now
  reports `real_train_runner_implemented = true`; its remaining local blocker
  is `selected_runtime_debug_wrapper_not_wired_to_real_runner_until_spec0008`.
  No Kaggle command, remote approval request, or long training launch was run.
  Adversarial review findings about DDP artifact races, stale checkpoint-state
  proof, wrong DDP option inference, AMP skip progress, missing distributed
  initialization validation, and stale "runner not implemented" wording were
  integrated. Final local verification passed:
  `PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_runner.py -q`
  (`4 passed`), selected-runtime gate/embedded slices (`30 passed`), exact
  `eqvae.cli.selected_runtime_train` two-step synthetic dry-run,
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`,
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`, full
  `./scripts/python_quality.sh` (`207 passed`, 0 type errors),
  `git diff --check`, repo `./scripts/agent_preflight.sh`, and workspace
  `/home/maximus/Documents/Tesis/agent_preflight.sh`. Next concrete action is
  Spec 0008 local selector/readiness implementation; remote selected-runtime
  debug/tiny still requires local readiness plus explicit user approval, and
  it is not approval for a long full run.
- 2026-06-22 real-runner and remote-debug/tiny specs:
  after commit `d02204c` (`Implement selected runtime local mechanics`), the
  next path was split into two locked specs. Spec 0007
  (`docs/specs/0007-real-ubc-ddp-amp-selected-runtime-runner.md`) covered the
  local real `ubc-pre-shuffled` selected runtime runner applying v5 exactly:
  dual-T4 DDP, standalone torchrun with `--nproc_per_node=2`, per-device batch
  12/global batch 24, AMP conservative FP16 autocast with GradScaler and FP32
  loss islands, no compile, `indexed_masked` corruption, selected dataloader
  policy, checkpoint/resume, artifact manifest, and gate-health proof writers.
  It is now implemented locally as recorded above. Spec 0008
  (`docs/specs/0008-canonical-fixed32-and-remote-debug-tiny-readiness.md`)
  follows it and covers local-first fixed-32 selector generation, canonical
  real UBC selector validation, and the narrow Kaggle selected-runtime
  debug/tiny push after local readiness and explicit user approval. The long
  full real training run remains out of scope for both specs, but it should be
  the immediate next candidate once Spec 0008 remote artifacts pass. No Kaggle
  command was run for this spec-writing update.
  Adversarial review for this docs-only spec slice first found
  selector-readiness circularity, placeholder verification commands,
  underspecified fixed-32 generation, missing remote debug/tiny quantitative
  bounds, and blurry Kaggle-wrapper ownership; fixes were integrated. A
  follow-up review found stale pre-push wording that required a pre-existing
  canonical selector despite the chosen `remote_generate` flow, plus an
  optional-vs-required `selected_runtime_train` entry point; those fixes were
  integrated too. Final verification after the docs fixes passed:
  stale-phrase sweeps for selector/runner contradictions,
  `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug` (`22 passed`),
  `./scripts/python_quality.sh` (`203 passed`, 0 type errors),
  `git diff --check`, repo `./scripts/agent_preflight.sh`, and workspace
  `/home/maximus/Documents/Tesis/agent_preflight.sh`.
  Remaining blockers after Spec 0007 are deliberately future work:
  remote-generated canonical real fixed-32 selector generation/validation, real
  gate-health/resume/tiny proofs, explicit user approval for any remote
  selected-runtime debug/tiny action, and no long real training launch until
  those gates pass.
- 2026-06-20 historical FSQ reference memory:
  the successful working FSQ Kaggle training notebook/artifact is
  `kaggle/train_runs`. It is the local reference for the broad ResNet-like
  autoencoder macro-architecture, spatial latent intuition, and runtime tactics
  such as DDP, AMP, `torch.compile`, channels-last, cuDNN benchmarking, pinned
  mmap-style data loading, static-shape loader discipline, and checkpoint/resume
  retention.
  Do not inherit FSQ quantization, codebooks, rounding, discrete latent
  telemetry, the learned quantization scale, PixelShuffle/sub-pixel upsampling,
  final `tanh` output bounding, the exact old HED corruptor implementation, or
  `rot90`-only/discrete-latent equivariance artifacts into the new continuous
  `SO(2)` equivariant VAE path. Also do not copy the old branchless validation
  behavior that computes corruption even for clean validation, or the partial
  AMP-skip behavior where scheduler/warmup logic can advance after a skipped
  optimizer step.

## Update Rule

Update this file after meaningful shifts in active work, blockers, or next
steps, and before handing work back from a partial state. Each handoff update
should make clear:

- what changed;
- what is currently in progress;
- exactly where the agent left off;
- the next concrete action;
- active blockers or decisions needed;
- verification run and remaining failures.

Delete or replace stale information instead of appending contradictory history.

## VS Code Tasks

When opening this repo in VS Code, the local workflow tasks are:

- `Agent: preflight`
- `Paper: compile SIPAIM PDF`
- `Paper: Overleaf local check`
- `Python: quality`
