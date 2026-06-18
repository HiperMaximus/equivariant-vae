# Decision 0008: Kaggle Synthetic Timing Pretest

Date: 2026-06-17
Amended: 2026-06-18

Decision: use a separate no-dataset Kaggle GPU timing pretest to screen and
order candidate runtime rows before attaching the 60 GB+ real dataset.

The pretest generates deterministic UBC-format binary+CSV shards under
`/kaggle/working` and attaches no Kaggle sources. Its default profile is
`synthetic_binary_2gib_histology_like_v1`: 10,912 total `3x256x256` CHW
`uint8` patches, split 5,456 train / 5,456 validation, with 2,145,386,496
binary payload bytes before CSV/artifacts, about 1.998 GiB. This keeps
30-step non-wrapping ranked rows eligible through global batch 128.

The earlier remote v1 Kaggle evidence used the historical compact profile
`synthetic_binary_0p81gb_histology_like_v1`: 4,096 total patches, split 2,048
train / 2,048 validation, with 805,306,368 payload bytes. That profile remains
named for evidence lineage, but it is no longer the default because it forces
global batch 128 rows into fit-probe-only status.

The generated root must mirror the real dataset filenames and format:
`dataset/ubc_train_shuffled.bin`, `dataset/ubc_train_shuffled.csv`,
`dataset/ubc_ocean_valid.bin`, and `dataset/ubc_ocean_valid.csv`, with the same
64-byte header, CRC32, CHW `uint8` payload order, CSV metadata semantics, train
CSV without `idx`, and validation CSV with `idx`. Timed reads must use the
active real loader path, not a synthetic-only loader.

Before timed rows, the pretest must prove data parity: parse and record headers,
row counts, file sizes, file hashes, CRC values, and a `validate_crc = true`
integrity pass for both splits. It must also prove that timed training batches
use `PatchTrainingDataset`, `collate_patch_training_samples`, and
`normalize_uint8_batch`, with pre/post dtype and range checks. Tensor-only rows
must prove `PatchTensorDataset`. The manifest must record uniqueness of
semantic sample keys and representative row/file/sample identifiers and hashes.

It must attempt both `single_visible_t4` and `dual_t4_ddp`. Rows are compared by
feasible global throughput and projected epoch time, not by equal per-device
batch size alone. Ranked rows use `non_wrapping_eligibility_steps = 30`; rows
where `global_batch_size * non_wrapping_eligibility_steps > split_patch_count`
are fit/VRAM probes only, even if a shorter initial fit pass can execute without
sample reuse.

Projected epoch time for the later real-data benchmark screen uses
`real_train_patch_count = 300000`, `drop_last = false`,
`global_batch_size = per_device_batch_size * world_size`,
`steps_per_epoch = ceil(real_train_patch_count / global_batch_size)`, and
`estimated_epoch_minutes = steps_per_epoch * steady_step_ms_p50 / 60000`.
Startup and compile time are recorded separately, not hidden inside steady-state
epoch time.

Boundary: synthetic timing output is non-promotable. It may recommend rows to
carry into the real-data benchmark, but it must not write
`benchmark/selected_runtime.json`, must keep `full_run_eligible = false`, and
must not claim final batch size, precision policy, corruption strategy,
dataloader settings, single-vs-dual T4 selection, convergence, paper evidence,
or full-run readiness. The artifact must include an explicit `blocked_claims`
object with those claims marked blocked.
