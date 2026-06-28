# Spec 0007: Real UBC DDP AMP Selected-Runtime Runner

Status: implemented / locally verified
Implementation readiness: local implementation complete; remote proof remains
sequenced through Spec 0008
Owner/workstream: comparable non-equivariant VAE baseline, real selected runtime
Last updated: 2026-06-28

## Purpose

Implement the real `ubc-pre-shuffled` training runner that applies the selected
runtime from `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`
instead of only validating it locally. Spec 0006 built the local fail-closed
inspection station. This spec builds the runner path that can actually execute
the selected runtime on real Kaggle data once the narrow remote debug/tiny gate
is approved in Spec 0008.

The selected runtime remains the v5 fallback row:

```text
dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative
```

This spec must apply and prove every selected option, not reinterpret or
re-optimize them. Runtime search is already done for this stage; the job here
is to make the selected row executable, resumable, auditable, and ready for the
first real proof run.

## Non-Goals

- No Kaggle push by this spec alone.
- No first 30h-60h full training run.
- No new runtime-selection search or relaxed-AMP experiment.
- No paper result claims.
- No selector pass claim; canonical fixed-32 selector generation and remote
  debug/tiny readiness are Spec 0008.
- No acceptance of local CPU/synthetic artifacts as real DDP/AMP proof.

## Inputs And Data Contract

- Runtime config:
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
- Linked runtime proof:
  `runs/kaggle/runtime_selection_v5/benchmark/runtime_proof.json`.
- Training data:
  real Kaggle dataset `maximusshtefan/patches-pre-shuffled-ubc-ocean`, resolved
  as `ubc-pre-shuffled` with:
  - `ubc_train_shuffled.bin`;
  - `ubc_train_shuffled.csv`;
  - `ubc_ocean_valid.bin`;
  - `ubc_ocean_valid.csv`.
- Patch shape and dtype:
  `3x256x256` CHW `uint8`, normalized through the existing repo data path.
- Split policy:
  train uses pre-shuffled train shard; validation uses clean validation shard.
  No masked-WSI sealed test claims in this spec.
- Selected runtime options to apply exactly:
  - accelerator mode: `dual_t4_ddp`;
  - launcher: `torchrun --standalone --nproc_per_node=2`;
  - world size: `2`;
  - per-device batch size: `12`;
  - global batch size: `24`;
  - gradient accumulation: `1`;
  - precision policy: `amp_conservative`;
  - AMP autocast dtype: `float16`;
  - GradScaler: enabled;
  - FP32 loss islands: enabled for objective-sensitive math;
  - compile: disabled, scope `none`;
  - dataloader: selected v5 settings, including `num_workers = 0`,
    `pin_memory = false`, `persistent_workers = false`,
    `prefetch_factor = null`, and selected H2D behavior;
  - corruption: `indexed_masked`;
  - memory format: selected runtime policy, currently contiguous;
  - optimizer zero-grad: `set_to_none = true`.

## Outputs And Acceptance Artifacts

- Code artifacts:
  - dedicated `eqvae.cli.selected_runtime_train` runner CLI, allowed to reuse
    shared training internals from `eqvae.cli.train`;
  - local runner command builder for dual-rank Kaggle execution, without
    wiring it into the selected-runtime debug Kaggle wrapper yet;
  - real-data path that uses `resolve_patch_data_paths`, `PatchTrainingDataset`,
    collation, normalization, and `indexed_masked` train corruption;
  - actual AMP/GradScaler train-step path with FP32 objective islands;
  - DDP rank/device proof and rank-local artifact merge policy;
  - checkpoint/resume implementation for schema v5 on real runner state;
  - gate-health row writer for the selected runtime.
- Artifacts:
  - `benchmark/training_summary.json`;
  - `benchmark/selected_runtime_debug_summary.json`;
  - `benchmark/selected_runtime_plan_applied.json`;
  - `benchmark/checkpoint_resume_proof.json`;
  - `benchmark/gate_health_summary.json`;
  - `benchmark/artifact_manifest.json`;
  - `metrics/train_steps.csv`;
  - `metrics/gate_health.csv`.
- All local-only tests and dry-runs remain `full_run_eligible = false`.
  Real pass eligibility can only come from Spec 0008 remote debug/tiny artifacts.

## Implementation Result

Implemented locally in:

- `src/eqvae/cli/selected_runtime_train.py`;
- `src/eqvae/training/selected_runtime_runner.py`;
- `tests/test_selected_runtime_runner.py`;
- `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`.

The runner consumes the shared `SelectedRuntimePlan`, supports synthetic
UBC-format dry-runs and `ubc-pre-shuffled` roots, builds the required tokenized
`torchrun --standalone --nproc_per_node=2` command, applies selected runtime
settings, includes the AMP/GradScaler train-step path with FP32 objective
islands, writes schema-v5 checkpoint/resume proof, emits gate-health rows, and
keeps all local artifacts non-promotable. Local CPU dry-runs intentionally fail
the full dual-T4/AMP plan-applied proof and keep `remote_pass_ready = false`.
Spec 0008 remains responsible for the canonical fixed-32 selector and any
explicitly approved remote debug/tiny proof.

## Related Requirements And Evidence

- `GOAL.md`: first full run needs selected-runtime debug, checkpoint/resume,
  tiny-overfit, artifact, and gate-health proof.
- `docs/specs/0001-translatable-normal-vae-baseline.md`: broad baseline
  contract.
- `docs/specs/0003-kaggle-cli-execution-workflow.md`: remote guard workflow.
- `docs/specs/0006-selected-runtime-local-mechanics.md`: local fail-closed
  mechanics this spec must preserve.
- `docs/kaggle_cli_workflow.md`: Kaggle push/read confirmation policy.
- `docs/behavior_inventory_kaggle.md`: historical FSQ runtime behavior.

## Architecture Or Workflow Contract

1. Shared parser remains authoritative.
   The runner must consume `SelectedRuntimePlan`; do not duplicate selected
   runtime validation logic.
2. Local dry-run mode first.
   Add a local proof mode that builds the runner, resolves config, constructs
   launch arguments, exercises synthetic UBC-format batches, and writes
   non-promotable readiness artifacts without claiming DDP/AMP execution.
3. Real data path.
   Add `data = "ubc-pre-shuffled"` support for the selected-runtime runner. The
   existing generic local `synthetic` path must remain available for tests, but
   it cannot satisfy real proof.
4. Actual selected-runtime application.
   The real path must set the selected batch size, precision policy, GradScaler,
   FP32 loss islands, corruption strategy, dataloader settings, memory policy,
   and optimizer zero-grad behavior from the selected plan.
5. DDP launch and rank proof.
   The runner CLI/command builder must construct a tokenized
   `torchrun --standalone --nproc_per_node=2` command, bind two visible T4s,
   record rank/local-rank/device/current-device/world-size assignments, and
   fail closed on single GPU, wrong accelerator, or rank mismatch. Spec 0007
   owns the reusable command builder and local tests only; Spec 0008 owns
   updating `kaggle/kernels/selected_runtime_debug/run_template.py` to call the
   real runner remotely.
6. Progress and checkpoint discipline.
   Reuse schema v5 and the Spec 0006 progress semantics: batch attempts are not
   successful optimizer updates; AMP skips must not advance optimizer, beta, LR,
   checkpoint cadence, validation cadence, or tiny smoothing.
7. Resume proof.
   The runner must create a short checkpoint/resume proof that restores model,
   optimizer, scheduler/progress, AMP scaler, RNG state, selected runtime
   identity, and DDP progress before continuing. Identity mismatch must fail
   before mutable restore.
8. Gate health.
   Emit selected-runtime gate-health rows for learned gates, nonfinite counts,
   gradient/update norms, and AMP skip status. Gate-health failure blocks remote
   pass readiness.
9. No silent promotion.
   Local runner tests and dry-runs stay non-promotable. Only Spec 0008 can run
   the narrow remote debug/tiny proof and decide whether artifacts are pass.

## Config Contract

Add or update config keys under the selected-runtime debug config without
changing the selected runtime itself:

- `data = "ubc-pre-shuffled"` for the real runner path;
- `runtime_config` path to v5 selected runtime;
- `selected_runtime_required = true`;
- `selected_runtime_debug.real_train_runner_implemented`;
- `selected_runtime_debug.remote_pass_ready`;
- `selected_runtime_debug.fixed_32_selector_real`;
- `selected_runtime_debug.max_debug_optimizer_steps`;
- `selected_runtime_debug.resume_probe_enabled`;
- `selected_runtime_debug.gate_health_enabled`;
- `runtime.selected_runtime_policy_id = "amp_fp16_conservative"`;
- `runtime.torchrun_standalone = true`;
- `runtime.nproc_per_node = 2`;
- `runtime.ddp_static_graph` and `gradient_as_bucket_view` only if inherited
  safely from selected runtime or explicitly recorded as runtime-policy fields.

Defaults must fail closed: real-runner readiness flags stay false until the
local code path and tests exist, and remote readiness stays false until Spec
0008 remote artifacts pass.

## Acceptance Criteria

1. The real runner can be invoked locally in dry-run/synthetic mode and writes
   non-promotable artifacts without remote access.
2. `ubc-pre-shuffled` is a supported runner data mode, but local tests do not
   require the 60 GB dataset.
3. The runner consumes the shared `SelectedRuntimePlan` and applies every v5
   selected option listed in this spec.
4. A fabricated plan-applied proof with unexecuted DDP or AMP fails.
5. A real-runner dry-run cannot flip `remote_pass_ready` or
   `real_train_runner_implemented` unless the implementation capability tests
   and artifact writers are present.
6. DDP launch construction rejects missing `--standalone`, wrong
   `--nproc_per_node`, duplicate/conflicting nproc, wrong accelerator, single
   visible T4, and rank/device mismatch.
7. AMP-skip and checkpoint/resume behavior follows schema v5 and fails before
   restore on identity/progress mismatch.
8. Gate-health rows are required for pass readiness.
9. Existing Spec 0006 preflight remains passing.

## Tests And Verification Commands

Required before calling Spec 0007 implemented; completed locally on
2026-06-28:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_gate.py tests/test_train_cli.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_runner.py -q
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_train \
  --config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json \
  --data synthetic \
  --output-dir /tmp/eqvae-selected-runtime-runner-local \
  --run-name spec0007_local_runner_dryrun \
  --max-train-steps 2 \
  --max-val-steps 1 \
  --dry-run
./scripts/kaggle_kernel.sh preflight-selected-runtime-runner
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
./scripts/python_quality.sh
git diff --check
./scripts/agent_preflight.sh
cd /home/maximus/Documents/Tesis && ./agent_preflight.sh
```

Final pass status:

- `tests/test_selected_runtime_runner.py`: 4 passed.
- Selected-runtime gate plus embedded-kernel slice: 30 passed.
- Exact two-step `eqvae.cli.selected_runtime_train` synthetic dry-run: passed.
- `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`: passed.
- `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`: passed.
- `./scripts/python_quality.sh`: 207 passed, 0 type errors.
- `git diff --check`, repo preflight, and workspace preflight: passed.

Adversarial review fixes integrated before final verification: primary-rank
artifact writing with per-rank CSV gathers, selected DDP
`static_graph`/`gradient_as_bucket_view` values applied exactly from v5,
selected-runtime AMP/CUDA/DDP checkpoint-state statuses, bounded AMP-skip retry
handling with readiness blocking on any skip, required distributed
initialization in the rank proof, and stale fail-closed wording replaced with
the Spec 0008 debug-wrapper blocker.

The implementation must add
`./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`. It must be
local-only, must not call Kaggle, and must cover the new runner dry-run,
selected-runtime command-builder validation, AMP/progress/resume focused tests,
and fail-closed wrong-accelerator cases.

## Implementation Blockers

None for local implementation. Remote validation is intentionally sequenced to
Spec 0008 after this runner exists.

## Known Risks

- Accidentally treating local dry-run artifacts as real DDP/AMP proof.
- Applying most selected-runtime settings but silently missing AMP, GradScaler,
  FP32 loss islands, corruption, or dataloader details.
- DDP rank-local artifacts diverging or double-counting metrics.
- Checkpoint/resume passing without proving selected-runtime identity.
- Spending too long polishing local scaffolding and delaying the first real
  debug/tiny remote proof.

## Adversarial Checks

- Run with one visible GPU and verify fail-closed artifacts.
- Mutate the selected runtime to wrong batch size, precision, corruption,
  dataloader settings, or compile policy and verify the runner rejects it.
- Build a launch command with duplicate/conflicting `--nproc_per_node` and
  verify rejection.
- Force a simulated or real GradScaler skip and verify no progress counters
  advance.
- Corrupt checkpoint selected-row id, policy id, runtime hash, optimizer count,
  scheduler/progress state, AMP scaler state, or RNG state and verify restore
  fails before mutable state changes.
- Drop gate-health rows and verify pass readiness fails.

## Open Questions

None that block implementation. This spec chooses
`eqvae.cli.selected_runtime_train` as the narrow selected-runtime entry point;
it may delegate to shared training internals. The selected-runtime artifact
records DDP static-graph and bucket-view policy fields; the runner applies
those values exactly.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
- `docs/specs/0006-selected-runtime-local-mechanics.md`
- `docs/kaggle_cli_workflow.md`
- `src/eqvae/training/debug.py`
- `src/eqvae/training/selected_runtime.py`
- `src/eqvae/cli/selected_runtime_train.py`
- `src/eqvae/checkpointing.py`
- `kaggle/kernels/selected_runtime_debug/run_template.py`
