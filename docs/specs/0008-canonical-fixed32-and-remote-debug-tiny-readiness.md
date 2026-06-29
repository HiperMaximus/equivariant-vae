# Spec 0008: Canonical Fixed-32 And Remote Debug Tiny Readiness

Status: local readiness implemented / locally verified; remote proof pending
explicit approval
Implementation readiness: local `remote_generate` readiness, fixed-32 selector
generation/readiness, debug/tiny wrapper wiring, and output verifier are
implemented; the narrow Kaggle debug/tiny push milestone still requires the
user to explicitly approve the remote action
Owner/workstream: comparable non-equivariant VAE baseline, pre-long-run proof
gate
Last updated: 2026-06-29

## Purpose

Generate and validate the canonical fixed-32 train selector, then use the real
runner from Spec 0007 to run the narrow selected-runtime debug/resume/artifact/
gate-health/tiny-overfit proof on Kaggle. This spec is the bridge between local
runner implementation and the first full real long run.

The priority is speed toward the first real full run without skipping the small
proof that prevents wasting a 30h-scale Kaggle job. The spec therefore has two
milestones:

1. local-first selector generator and readiness proof;
2. explicit-user-approved narrow Kaggle selected-runtime debug/tiny push and
   artifact download/inspection.

When this spec passes, the immediate next work should be preparing/requesting
the first full real selected-runtime run, not inventing another local-only
detour unless the proof artifacts expose a blocker.

## Non-Goals

- No first full real long training run inside this spec.
- No runtime re-selection.
- No paper result claims.
- No accepting synthetic fixed-32 selectors as canonical real selectors.
- No remote action without explicit user approval and the required
  confirmation environment variables.

## Inputs And Data Contract

- Runtime config:
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
- Real runner:
  implemented by Spec 0007.
- Real Kaggle dataset:
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
- Canonical real train selector target:
  `fixed_32_train_overfit_patches.json`, selector kind
  `fixed_32_train_overfit`, source split `train`, exactly 32 train patches.
- Local generator test data:
  synthetic UBC-format shards generated locally for fast schema/provenance
  replay tests.
- Selector generation mode:
  local tests generate synthetic selectors only; the approved remote
  selected-runtime debug/tiny kernel generates and validates the canonical real
  fixed-32 selector from the attached Kaggle train shard before launching the
  debug/tiny train proof. This avoids requiring a 60 GB local real-data root
  before asking for the narrow remote approval.
- Real selector pass requirements:
  - dataset slug: `maximusshtefan/patches-pre-shuffled-ubc-ocean`;
  - train filenames: `ubc_train_shuffled.bin` and `ubc_train_shuffled.csv`;
  - train CSV SHA-256: locked by `eqvae.cli.selected_runtime_gate`;
  - train binary size: locked by `eqvae.cli.selected_runtime_gate`;
  - train header CRC32: locked by `eqvae.cli.selected_runtime_gate`;
  - row/patch count: `300000`;
  - shape: `3x256x256`;
  - layout: `CHW`;
  - CRC checked: true;
  - masked holdout exclusion loaded from
    `docs/data/ubc_ocean_masked_holdout_ids.csv`.

## Outputs And Acceptance Artifacts

- Code artifacts:
  - fixed-32 selector generator CLI or function that works locally against any
    resolved UBC-format data root;
  - local synthetic selector test proving schema generation/replay;
  - selected-runtime debug kernel path updated to generate/validate the
    canonical real selector first, then use the real runner;
  - push-readiness logic that consumes structured artifacts and rejects
    synthetic/placeholder selectors.
- Local artifacts:
  - synthetic generated selector fixture/artifact, explicitly
    non-canonical-real;
  - readiness failure proving synthetic selector replay cannot pass remote
    readiness;
  - pre-push readiness pass in `remote_generate` mode only when real runner code
    exists, selector generation is wired, the Kaggle metadata attaches the real
    dataset, and remote action remains explicit-user-gated.
- Remote artifacts after approved narrow push:
  - canonical real fixed-32 selector artifact;
  - `benchmark/selected_runtime_gate_summary.json`;
  - `benchmark/training_summary.json`;
  - `benchmark/selected_runtime_debug_summary.json`;
  - `benchmark/selected_runtime_plan_applied.json`;
  - `benchmark/checkpoint_resume_proof.json`;
  - `benchmark/tiny_overfit_summary.json`;
  - `benchmark/gate_health_summary.json`;
  - `benchmark/artifact_manifest.json`;
  - `metrics/train_steps.csv`;
  - `metrics/gate_health.csv`.

## Related Requirements And Evidence

- `GOAL.md`: the first full run remains the north-star next milestone after
  debug/resume/tiny proof.
- `docs/specs/0007-real-ubc-ddp-amp-selected-runtime-runner.md`: required
  runner prerequisite.
- `docs/specs/0006-selected-runtime-local-mechanics.md`: fail-closed readiness
  baseline and fixed-selector rejection behavior.
- `docs/specs/0003-kaggle-cli-execution-workflow.md`: remote guard rules.
- `docs/kaggle_cli_workflow.md`: Kaggle confirmation variables and status
  polling policy.

## Architecture Or Workflow Contract

1. Local selector generator first.
   Implement a generator that can run quickly against synthetic UBC-format
   shards. Local tests must prove deterministic schema/provenance writing and
   replay, while also proving the synthetic selector is rejected as
   non-canonical-real.
2. Canonical real selector boundary.
   The same generator must run inside the approved selected-runtime debug/tiny
   Kaggle kernel against the real Kaggle train shard before any training step.
   The selector passes only if the selected document validates against the real
   shard and matches the locked fingerprints listed in this spec. If selector
   generation or validation fails, the kernel must write fail-closed selector
   and gate artifacts and must not train.
3. Debug/tiny readiness remains artifact-derived.
   `real_train_runner_implemented`, remote selector-generation readiness, and
   post-download `fixed_32_selector_real` may become true only from structured
   readiness artifacts, not from hand-edited booleans. In
   `remote_generate` mode, pre-push readiness may pass with
   `fixed_32_selector_real = false`; the remote kernel must generate and
   validate the canonical selector before training, and only downloaded remote
   artifacts may prove `fixed_32_selector_real = true`.
   The tiny-overfit phase still uses exactly the 32 unique canonical train
   selector rows. To avoid making selected-runtime AMP readiness depend on an
   accidental fixed-32 tail microbatch, `kaggle_tiny_overfit` repeats
   selector-order rows only at the sampler level so every selected-runtime
   microbatch is full-sized. The artifacts must record
   `train_sampler_policy = "fixed32_tiny_full_batch_repeated"`,
   `fixed_train_repeated_to_full_batch = true`,
   `train_effective_global_epoch_samples`,
   `train_effective_per_rank_epoch_samples`, and `observed_batch_sizes`; for
   the v5 dual-T4 bs12 runtime the expected effective samples are 48 global and
   24 per rank, with observed tiny batch sizes `[12]`.
4. Local preflight before remote request.
   Before asking for any Kaggle action, run local focused tests, the
   selected-runtime debug preflight, the push-readiness CLI in
   `remote_generate` mode, diff checks, and repo/workspace preflights.
5. Narrow Kaggle push after explicit approval.
   Once local readiness passes, ask the user for explicit permission for the
   selected-runtime debug/tiny push. Remote write commands must use
   `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`; remote reads or
   downloads must use `KAGGLE_REMOTE_CONFIRMED=1`.
6. Do not wait on long kernels in-turn.
   If the narrow debug/tiny kernel is still running and likely to take more
   than about five minutes, record the state in `CURRENT.md`, give the user a
   concrete local time to resume, and stop.
7. Download and inspect artifacts.
   After completion, download outputs, inspect every required artifact, and run
   strict local replay/validation. Passing remote artifacts make the first full
   real selected-runtime run the next candidate action.
8. Long run is next, not included.
   This spec should finish by saying whether the first full real run is ready,
   not by launching it.

## Config Contract

- Selector generator:
  - `selector_kind = "fixed_32_train_overfit"`;
  - `selector_seed = "20260611:tiny-overfit"` unless a later spec changes it;
  - `expected_count = 32`;
  - `source_split = "train"`;
  - `masked_holdout_exclusion =
    "docs/data/ubc_ocean_masked_holdout_ids.csv"`;
  - `validate_crc = true`.
  - selection algorithm:
    1. load all train records from the resolved train CSV/bin pair;
    2. reject duplicate semantic identities before selection;
    3. remove any record whose `wsi_id` appears in the masked-holdout file;
    4. compute `selection_key_sha256` as SHA-256 over
       `"{seed}:{wsi_id}:{label}:{x}:{y}"`;
    5. sort candidates by
       `(selection_key_sha256, wsi_id, label, x, y)`;
    6. take the first 32 records with no replacement;
    7. emit ranks `0..31` in sorted order;
    8. store `source_split`, `file_index`, `row_index`, `sample_id`, `wsi_id`,
       `label`, `x`, `y`, `selection_key_sha256`, and per-patch SHA-256 for
       each selector.
- Selected-runtime debug/tiny gate:
  - `selected_runtime_debug.remote_pass_ready`;
  - `selected_runtime_debug.real_train_runner_implemented`;
  - `selected_runtime_debug.fixed_32_selector_real`;
  - `selected_runtime_debug.selector_generation_mode = "remote_generate"`;
  - `selected_runtime_debug.remote_selector_generation_ready`;
  - tiny config mirrors gate readiness under `selected_runtime_debug_gate`;
  - readiness flags default false; local artifacts may set runner and remote
    selector-generation readiness, but `fixed_32_selector_real` becomes true
    only after downloaded remote artifacts validate the canonical real selector.
- Remote debug/tiny bounds:
  - debug/resume proof successful optimizer updates: exactly 8;
  - checkpoint/resume probe: write at successful update 4, resume, continue to
    update 8, and prove selected-runtime identity before restore;
  - validation batches during debug proof: 1 clean validation batch before and
    after resume;
  - tiny-overfit successful optimizer updates: at most 128;
  - tiny fixed-32 sampler: 32 unique canonical selector rows, repeated in
    selector order only to full selected-runtime microbatches;
  - tiny sampler evidence for v5: `train_sampler_policy =
    "fixed32_tiny_full_batch_repeated"`, `fixed_train_repeated_to_full_batch =
    true`, `train_effective_global_epoch_samples = 48`,
    `train_effective_per_rank_epoch_samples = 24`, and
    `observed_batch_sizes = [12]`;
  - selected AMP conservative GradScaler startup scale:
    `grad_scaler_init_scale = 16384.0`, recorded in debug training summary,
    tiny-overfit summary, and selected-runtime plan-applied proof as a runner
    AMP extension;
  - tiny metric-row proof: the downloaded
    `tiny_overfit_phase/metrics/train_steps.csv` must be hash-linked from the
    artifact manifest and must prove both ranks, successful updates `1..128`,
    full per-rank `batch_size = 12`, finite `grad_norm`, zero AMP skips, and
    zero nonfinite rows;
  - tiny smoothing window: 25 successful updates;
  - tiny pass threshold: final smoothed L1 and final smoothed reconstruction
    loss must each improve by at least 1 percent relative to their initial
    smoothed values;
  - AMP skip tolerance: 0 skipped optimizer updates for a pass claim;
  - nonfinite tolerance: 0 nonfinite loss/gradient rows;
  - gate-health pass: all required gate-health rows present, finite grad/update
    norms, no missing selected-runtime row ids, and no gate saturation fraction
    at or above 0.99.
- Kaggle kernel metadata remains:
  - id `maximusshtefan/eqvae-selected-runtime-debug`;
  - private script kernel;
  - GPU enabled;
  - internet disabled;
  - machine shape `NvidiaTeslaT4`;
  - dataset source exactly `maximusshtefan/patches-pre-shuffled-ubc-ocean`.

## Latest Remote Attempt

Selected-runtime debug/tiny v2 is not a passing remote proof. It did prove the
canonical fixed-32 selector boundary on Kaggle: the downloaded
`benchmark/fixed32_selector_readiness.json` reports `status = "pass"`,
`fixed_32_selector_real = true`, `remote_selector_generation_ready = true`, and
locked real UBC train shard fingerprints for 32 selected train patches. The run
then failed before writing training/checkpoint/tiny/gate artifacts because the
runner built explicit VAE epsilon from the nominal batch cap 12 while the final
single-process fixed-32 batch had 8 samples. Local follow-up fixes size epsilon
from the realized input batch and adds a fixed-32/bs12/3-step regression.

The v2 follow-up also hardened the remote path beyond the immediate crash: the
debug wrapper now launches the selected-runtime train runner through
`python -m torch.distributed.run --standalone --nproc_per_node=2 -m
eqvae.cli.selected_runtime_train` with the embedded payload on `PYTHONPATH`,
and the post-download verifier hash-links selector readiness to the downloaded
selector, replays artifact-manifest hashes, validates gate-health CSV content,
and tightens train-step CSV checks.

Selected-runtime debug/tiny v3 is not a passing remote proof. After explicit
user approval, v3 was pushed from clean commit `09b5b24`; the guarded follow-up
status read on 2026-06-29 returned `KernelWorkerStatus.ERROR`, and artifacts
were downloaded to `runs/kaggle/selected_runtime_debug_v3`. The run proved real
progress: canonical fixed-32 selector generation passed, the debug/resume phase
wrote selected-runtime plan application, checkpoint/resume, gate-health CSV,
and artifact-manifest evidence, and the tiny phase reached 128 successful
optimizer updates with strong improvement (`l1_improvement_fraction =
0.42601120821087546`, `recon_loss_improvement_fraction =
0.37984046690403395`). It still failed the remote proof because the tiny phase
observed two AMP skipped rows and 500 aggregate nonfinite gradient entries, one
per rank at optimizer step index 3 on a per-rank `batch_size = 4` tail
microbatch. The root gate summary remained `status = "fail"` with
`launch_blockers_remaining = ["tiny_overfit"]`.

The local v3 follow-up is a source fix, not a verifier workaround: the runner
keeps the debug-path partial-batch epsilon fix and regression, but
`kaggle_tiny_overfit` with the fixed-32 selector now uses the deterministic
`fixed32_tiny_full_batch_repeated` sampler described above. Runner summaries
now expose sampler policy/effective samples/full-batch-repeat fields, tiny
summaries expose observed batch sizes plus aggregate AMP skip/nonfinite counts,
training summaries aggregate nonfinite counts over all metric rows, local
readiness blocks any aggregate nonfinite metric rows, and both the embedded
wrapper and post-download verifier require the tiny sampler evidence. The
downloaded v3 artifacts predate those fields, so the hardened verifier correctly
reports the historical tiny failure plus missing-sampler-evidence blockers.

Selected-runtime debug/tiny v4 is not a passing remote proof. After explicit
user approval for the narrow rerun only, v4 was pushed from clean commit
`ce72fa0`, the immediate guarded status read returned
`KernelWorkerStatus.RUNNING`, and the guarded follow-up status read at
`2026-06-29 01:51:13 -0500` returned `KernelWorkerStatus.ERROR`. Outputs were
downloaded to `runs/kaggle/selected_runtime_debug_v4`. The run proved the v3
sampler fix: tiny uses `fixed32_tiny_full_batch_repeated`, effective epoch
samples are `48` global / `24` per rank, and observed tiny batch sizes are
`[12]`. The remaining failure is one early full-batch AMP loss-scale overflow
per rank at tiny `optimizer_step_index = 3`: `batch_size = 12`, `grad_norm =
inf`, `nonfinite_count = 125` per rank, and `amp_step_skipped = 1`. The retry
after scale reduction succeeded, and the tiny phase still reached 128
successful updates with strong improvement (`l1_improvement_fraction =
0.339919261689692`, `recon_loss_improvement_fraction =
0.29479275826312307`).

The local v4 follow-up is a source fix, not a verifier workaround: the runner
now sets an explicit conservative GradScaler init scale of `16384.0`, records
that value in training/tiny summaries and in the plan-applied runner AMP
extension, and keeps zero-tolerance AMP-skip/nonfinite checks. The embedded
wrapper and post-download verifier now require the scaler evidence and direct
row-level nested tiny metric proof. Downloaded v4 predates that evidence and
correctly fails the hardened verifier with the original tiny overflow blockers
plus scaler-proof and nested tiny-row blockers.

The first approved v5 push attempt on 2026-06-29 did not reach remote
execution: local push guards and embedded-kernel checks passed, then the Kaggle
CLI failed before upload with an authentication-required message caused by a
stale cached OAuth access token. The selected-runtime slug was correct. The
workflow wrapper now routes authenticated Kaggle calls through
`scripts/kaggle_oauth_exec.py`, which generates a fresh OAuth token from the
installed Kaggle SDK and passes it to the child CLI via a temporary 0600 token
file. The selected-runtime-specific read-only preflight
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
kaggle/kernels/selected_runtime_debug` passes through that path with only the
known quota warning. Follow-up adversarial review removed the remaining raw
`kaggle auth print-access-token` probe from `api-check`; auth is now proven by
wrapped endpoint calls. After fresh explicit approval for the narrow retry,
Kaggle accepted selected-runtime debug/tiny version 5. The immediate guarded
status read at `2026-06-29 03:27 -0500` returned
`KernelWorkerStatus.RUNNING`; the guarded follow-up status read returned
`KernelWorkerStatus.COMPLETE`. Outputs were downloaded to
`runs/kaggle/selected_runtime_debug_v5`, and strict
`eqvae.cli.selected_runtime_gate --verify-output` passed. The v5 gate summary
has no remaining launch blockers, canonical real fixed-32 selector generation
passed, tiny-overfit had zero AMP skips and zero nonfinite rows, and the nested
tiny metric CSV proved 256 rank rows over 128 optimizer steps. Spec 0008 remote
debug/tiny readiness is therefore proved. The next candidate action is the
first full real selected-runtime run, which still requires fresh explicit
approval and is not launched by this proof.

## Acceptance Criteria

1. Local synthetic selector generation is deterministic, schema-valid, and fast.
2. A schema-valid synthetic selector still fails canonical real readiness.
3. The generator can run against a real UBC-format data root and records
   fingerprints required by the selected-runtime gate.
4. Placeholder, fabricated, wrong-count, wrong-split, wrong-dataset,
   no-CRC, synthetic replay, and stale-fingerprint selectors fail push
   readiness.
5. After Spec 0007 is implemented, local `--verify-push-ready` passes only
   in `remote_generate` mode, and only when runner capability,
   generator implementation/tests, structured readiness, exact metadata source
   attachment, and remote selector-generation readiness all pass. It must not
   require a local canonical selector path in this mode.
6. The selected-runtime debug/tiny kernel push guard is expected to pass only
   after local readiness passes in `remote_generate` mode.
7. The approved remote debug/tiny push writes no `benchmark/selected_runtime.json`
   and launches only the bounded debug/resume/tiny proof, not the long full
   training run.
8. Downloaded remote artifacts prove selected-runtime plan application,
   checkpoint/resume, artifact manifest, gate health, canonical selector
   generation/validation, and tiny-overfit on real UBC/DDP/AMP, within the
   quantitative bounds in this spec.
9. If any remote artifact fails, `CURRENT.md` records the blocker and the next
   concrete fix. If all pass, `CURRENT.md` records the first full real long run
   as the next candidate action.

## Tests And Verification Commands

Required local checks before requesting remote approval:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_gate.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_runner.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_fixed_selectors.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_fixed32_selector_readiness.py -q
./scripts/kaggle_kernel.sh preflight-fixed32-selector-readiness
PYTHONPATH=src .venv/bin/python -m eqvae.cli.select_fixed_patches \
  --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
  --kind fixed_32_train_overfit \
  --data-root /tmp/eqvae-fixed32-synthetic-root \
  --masked-holdout-csv docs/data/ubc_ocean_masked_holdout_ids.csv \
  --output /tmp/eqvae-fixed32-synthetic-root/fixed_32_train_overfit_patches.json \
  --validate-crc
./scripts/kaggle_kernel.sh preflight-selected-runtime-runner
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-push-ready \
  --selector-generation-mode remote_generate \
  --debug-config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
  --tiny-config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
./scripts/python_quality.sh
git diff --check
./scripts/agent_preflight.sh
cd /home/maximus/Documents/Tesis && ./agent_preflight.sh
```

The implementation provides
`./scripts/kaggle_kernel.sh preflight-fixed32-selector-readiness`. It is
local-only, creates `/tmp/eqvae-fixed32-synthetic-root`, runs the selector CLI
on that synthetic root, proves the generated selector is deterministic and
schema-valid, and proves it still fails canonical real UBC readiness.

Expected remote sequence after explicit user approval:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_debug

KAGGLE_REMOTE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh status-selected-runtime-debug

KAGGLE_REMOTE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh output-selected-runtime-debug runs/kaggle/selected_runtime_debug_<version>

PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_<version> \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
```

The exact versioned output directory should be recorded in `CURRENT.md`.

## Implementation Blockers

- Spec 0007 real runner implementation must exist before remote readiness can
  pass.
- The canonical real selector cannot be proven until the approved remote kernel
  mounts the real Kaggle pre-shuffled train shard and generates the selector
  before training.
- Remote push/read/download requires explicit user approval at the time of the
  action.

## Known Risks

- A synthetic selector can look schema-valid and accidentally be treated as
  canonical real evidence.
- A local readiness pass can be mistaken for remote proof.
- The narrow debug/tiny push can grow into an accidental long run if caps and
  artifact expectations are not strict.
- The team can over-focus on the tiny gate and lose sight of the first full real
  run. This spec must keep the full run as the immediate next milestone after
  debug/tiny proof passes.

## Adversarial Checks

- Try to pass readiness with a synthetic selector generated from local shards.
- Mutate selector count, split, dataset slug, CRC flag, train CSV hash, binary
  size, header CRC, patch count, or masked-holdout exclusion.
- Try to flip readiness booleans without structured artifacts.
- Try to push with stale generated `run.py`, wrong metadata sources, or a
  selected-runtime debug kernel that still contains synthetic-only runner
  blockers.
- Try to let the remote debug/tiny path write `benchmark/selected_runtime.json`
  or run an uncapped long training job.
- Try to claim first-full-run readiness when checkpoint/resume, gate-health, or
  tiny-overfit artifacts are missing.

## Open Questions

None that block implementation. This spec chooses in-kernel canonical selector
generation before debug/tiny training, and the tiny-overfit proof uses the same
canonical 32 patches for train-clean and train-corrupted-fixed-seed views in
the same remote version.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0007-real-ubc-ddp-amp-selected-runtime-runner.md`
- `docs/specs/0006-selected-runtime-local-mechanics.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
- `docs/kaggle_cli_workflow.md`
- `src/eqvae/data/fixed_selectors.py`
- `src/eqvae/benchmarking/selected_runtime_gate.py`
- `src/eqvae/cli/selected_runtime_gate.py`
- `kaggle/kernels/selected_runtime_debug/run_template.py`
