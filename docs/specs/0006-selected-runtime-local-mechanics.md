# Spec 0006: Selected-Runtime Local Mechanics

Status: implemented / locally verified
Implementation readiness: implemented and exercised by the completed remote baseline
Owner/workstream: comparable non-equivariant VAE baseline, selected runtime gate
Last updated: 2026-08-11

## Purpose

Implement the next local-first selected-runtime mechanics slice needed before
any real Kaggle selected-runtime debug/tiny push can be considered. This spec
turns the broad Spec 0001 selected-runtime checklist into an implementation
slice that can be built and verified locally without network, Kaggle remote
actions, or a long real training run.

The output of this spec is non-promotable local evidence only. It must not set
`full_run_eligible = true`, must not write `benchmark/selected_runtime.json`,
must not flip selected-runtime debug/tiny readiness flags, and must not request
or launch a remote Kaggle run.

## Implementation Note

Implemented locally on 2026-06-22. The train and gate paths share the v5
`SelectedRuntimePlan` parser, including strict linked `runtime_proof.json`
status/write-decision/rank/return-code validation and tokenized
`torchrun --standalone --nproc_per_node=2` validation. The local train path
writes a full plan-applied proof that fails locally for unexecuted dual-T4 CUDA
AMP/DDP fields, plus UBC-format mechanics, AMP/progress, checkpoint schema v5,
and structured readiness artifacts; focused tests cover corrupted/fabricated
selected runtimes, failed or misleading linked runtime proofs,
recorded-but-unapplied runtime, clean validation isolation, integrated
simulated AMP skips, checkpoint progress consistency rejection before restore,
observed local FP32/AMP-off row telemetry, and readiness checks that consume the
structured artifact instead of config booleans alone. All artifacts remain
non-promotable. Real UBC/DDP/AMP proof and canonical real fixed-32 selector
generation remain future work requiring explicit user approval before any
remote action.

## Non-Goals

- No Kaggle push, status read, output download, or API check.
- No long real training run.
- No claim that local CPU/Gloo/synthetic UBC mechanics prove dual-T4 NCCL AMP
  behavior.
- No generation of the canonical real fixed-32 selector unless the user later
  explicitly approves a real-data-gated task.
- No paper result claims.

## Inputs And Data Contract

- Selected-runtime input:
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
- Local mechanics data:
  tiny synthetic UBC-format train/validation shards with the canonical relative
  filenames `ubc_train_shuffled.*` and `ubc_ocean_valid.*`.
- Real selector boundary:
  placeholder or synthetic fixed-32 selectors remain invalid for push
  readiness. Real selector pass requires the locked canonical train-shard
  fingerprints already enforced by `eqvae.cli.selected_runtime_gate`.

## Architecture Or Workflow Contract

Implement these local-only sub-slices in order:

1. Shared selected-runtime plan parser.
   Add a single `SelectedRuntimePlan` parser/validator used by both the train
   path and selected-runtime gate path. It must validate the v5 fallback row id,
   `runtime_policy_id`, dual-T4/DDP topology, `torchrun --standalone
   --nproc_per_node=2`, per-device batch size 12, global batch size 24,
   `amp_conservative`, GradScaler, FP32 loss islands, no compile, selected
   dataloader settings, and `indexed_masked` corruption.
2. Runtime plan application proof.
   Emit a local non-promotable plan-applied proof with expected and observed
   values for batch size, AMP/scaler mode, FP32 objective islands,
   dataloader settings, compile policy, corruption strategy, DDP/topology
   status, selected-runtime artifact hash, row id, and policy id. Tests must
   reject a run that records the selected runtime without applying it.
3. Local UBC-format mechanics.
   Extend the local train path to resolve synthetic UBC-format shards through
   `resolve_patch_data_paths`, instantiate `PatchTrainingDataset`, collate
   metadata-carrying batches, normalize tensors, apply selected
   `indexed_masked` corruption only to train batches, and run clean validation
   without calling or advancing corruption RNG. Artifacts from this path must
   say `status_scope = "local_selected_runtime_mechanics"` or an equivalent
   non-promotable scope and `full_run_eligible = false`.
4. AMP skip and progress semantics.
   Add a test seam for simulated GradScaler skipped steps. Track batch attempts
   separately from successful optimizer updates. An AMP-skipped step must not
   advance optimizer-step count, beta schedule, LR scheduler, checkpoint cadence,
   validation cadence, or tiny-overfit smoothing windows.
5. Checkpoint schema v5 for selected-runtime mechanics.
   Define the selected-runtime checkpoint schema before using it. Required
   fields include model, optimizer, LR scheduler, beta/progress counters, AMP
   scaler state or explicit local-not-applicable status, Python RNG, explicit
   NumPy `Generator`, Torch CPU RNG, CUDA RNG state or explicit
   local-not-applicable status, named Torch generators, DDP sampler/progress
   state or explicit local-not-applicable status, effective config hash,
   selected-runtime artifact hash, selected row id, and runtime policy id.
   Resume must fail before restoring model/optimizer state if selected-runtime
   identity or required schema fields are missing or mismatched.
6. Readiness aggregation.
   Add a structured local readiness artifact that consumes the plan-applied,
   UBC-mechanics, AMP/progress, checkpoint/resume, selector, artifact-manifest,
   and gate-health proof statuses. `--verify-push-ready` must eventually depend
   on this structured artifact plus the canonical real fixed-32 selector and
   real runner readiness, not config booleans alone.

## Outputs And Acceptance Artifacts

- Shared selected-runtime plan parser module and tests.
- Local selected-runtime plan-applied proof artifact.
- Local UBC-format mechanics training/debug artifacts with
  `full_run_eligible = false`.
- Pass-capable training metrics must use the existing training schema
  `metrics/train_steps.csv`. The current selected-runtime gate
  `metrics/train_metrics.csv` output is a fail-closed contract artifact and
  cannot satisfy real debug/full-run training evidence unless it is renamed or
  explicitly migrated in a later patch.
- Checkpoint schema v5 and resume rejection tests.
- Structured local readiness artifact.

## Acceptance Criteria

1. A fabricated or corrupted selected-runtime JSON fails plan parsing.
2. A config/runtime mismatch fails before training begins.
3. A local synthetic UBC-format run exercises `PatchTrainingDataset`, collation,
   normalization, selected train corruption, and clean validation isolation.
4. Local mechanics artifacts are explicit non-promotable evidence and never set
   `full_run_eligible = true`.
5. Simulated AMP skips do not advance optimizer update count, beta, LR,
   checkpoint, validation, or smoothing progress.
6. Checkpoint resume rejects missing or mismatched selected-runtime state before
   restoring mutable model/optimizer state, including invalid or inconsistent
   progress counters.
7. Placeholder, fabricated, and synthetic fixed-32 selectors still fail push
   readiness.
8. `--verify-push-ready` consumes structured readiness status and cannot pass
   from config/capability booleans alone.
9. `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug` remains
   local-only and passing.

## Tests And Verification Commands

Required before handoff:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_gate.py -q
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
./scripts/python_quality.sh
git diff --check
./scripts/agent_preflight.sh
cd /home/n00b1337/Documents/Max/Tesis && ./agent_preflight.sh
```

Add focused tests for the new parser, local UBC-format mechanics, AMP skip
progress, checkpoint schema v5, and readiness aggregation.

## Implementation Blockers

None for the local mechanics slice. Real Kaggle proof, canonical real fixed-32
generation, and any remote push remain blocked on explicit user approval and
future preflight success.

## Known Risks

- Local CPU or synthetic UBC-format mechanics can be mistaken for real
  dual-T4/NCCL/AMP proof. All local artifacts must be non-promotable.
- Train and gate selected-runtime validation can drift if they do not share the
  same parser.
- Readiness flags can become declarative instead of artifact-derived if the
  readiness aggregator is skipped.
- `train_metrics.csv` and `train_steps.csv` can be confused. Use
  `train_steps.csv` for pass-capable training evidence.

## Adversarial Checks

- Try to pass by recording v5 without applying batch size, AMP, corruption, or
  dataloader settings.
- Try a fake selected-runtime JSON with the correct row id but wrong nested
  launch fields.
- Try a linked `runtime_proof.json` with matching hash but failed status,
  missing write decision, wrong rank assignment, or a command that only
  contains `torchrun --standalone --nproc_per_node=2` as a substring.
- Try a clean-validation pass that calls corruption but hides RNG consumption.
- Inject an AMP skip and verify no progress counters advance.
- Remove scheduler/scaler/DDP progress fields or corrupt progress counters in a
  checkpoint and verify resume fails before restore.
- Try a schema-valid synthetic fixed-32 selector with the real dataset slug and
  verify push readiness still fails.

## Related Files

- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
- `docs/kaggle_cli_workflow.md`
- `src/eqvae/benchmarking/selected_runtime_gate.py`
- `src/eqvae/cli/selected_runtime_gate.py`
- `src/eqvae/training/debug.py`
- `src/eqvae/checkpointing.py`
- `src/eqvae/data/fixed_selectors.py`
- `tests/test_selected_runtime_gate.py`
