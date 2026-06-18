# Spec 0001: Translatable Normal VAE Baseline

Status: draft active / reopened for architecture and objective corrections
Implementation readiness: not locked for broad implementation; narrow local
benchmark scaffold is `scaffold_schema_ready`; narrow instantiated topology
count slice is implemented locally as `topology_count_ready`; narrow local
data/metrics slice is implemented as `data_metrics_ready`; narrow local
selector/dataloader slice is implemented as `fixed_selectors_dataloader_ready`;
narrow local CPU dataloader pre-test is `local_benchmark_pretest_ready`;
narrow model/loss local train-step pre-test is `model_loss_train_step_ready`;
the HED/stain corruption local correctness/QA slice is `corruption_ready`;
the narrow capped Kaggle smoke is `kaggle_smoke_ready`; the synthetic
no-dataset Kaggle setup smoke is `kaggle_setup_smoke_ready`; the no-dataset
synthetic binary Kaggle timing pretest contract is
`kaggle_synthetic_timing_contract_ready` with local implementation and
remote v1/v2/v3 non-promotable evidence
Owner/workstream: comparable non-equivariant VAE baseline
Last updated: 2026-06-18

Local scaffold exception, 2026-06-12: the user-authorized local
benchmark-unblock slice may create `src/eqvae`, `configs/spec0001`, the
local synthetic benchmark schema writer, and the first schema-only
`model_count` writer. This exception does not lock spec 0001 for model, data,
corruption, training, evaluation, Kaggle, or paper-claim implementation. The
schema-only `model_count` state has now been superseded by the
`topology_count_ready` slice below.

Topology-count exception, 2026-06-12: this spec now authorizes only the
instantiated model-count slice and its local hardening fixes:
`src/eqvae.models.non_equivariant_vae`, the activation and fixed-resampling
modules needed to instantiate that topology, shared JSON config resolution under
`src/eqvae/config.py`, benchmark helper modules under `src/eqvae/benchmarking`,
`eqvae.cli.model_count`, `benchmark/model_count.json`,
`benchmark/model_inventory.csv`, and focused tests. This exception does not
authorize data loading, corruption, training, runtime selection, Kaggle remote
execution, or paper-claim work. The slice may mark `model_count.json` with
`status = "pass"` only if it instantiates the locked model, resolves any
`source_config` overlays, verifies every expected inventory path/type, observes
inventory shapes and execution order from the instantiated model, confirms the
zero-initialized RGB head, rejects banned or uninventoried leaf modules, and
matches the analytic target exactly.

Topology-count implementation status, 2026-06-12: the local slice now
instantiates the locked Conv2d topology, resolves thin configs through
`source_config`, writes `benchmark/model_count.json` with `status = "pass"`,
and writes `benchmark/model_inventory.csv` with 129 observed rows: 43 learned
convolutions, 40 GroupNorm modules, 34 gates, and 12 fixed resampling ops.
Inventory input/output shapes and forward order are observed by meta-tensor
forward hooks rather than copied only from the analytic table. Runtime selection
remains blocked; this is model-count evidence only.

Data/metrics exception, 2026-06-12: this spec now authorizes only the narrow
`data_metrics_ready` local slice: `src/eqvae/data/synthetic.py`,
`src/eqvae/data/patch_shards.py`, `src/eqvae/data/splits.py`,
`src/eqvae/metrics/reconstruction.py`, package `__init__.py` exports, and
focused tests under `tests/test_patch_shards.py`,
`tests/test_split_validation.py`, and `tests/test_reconstruction_metrics.py`.
This exception may implement deterministic synthetic patch-shard fixtures, UBC
binary/CSV patch-shard loading, WSI/masked-holdout split validation helpers,
and repo-owned reconstruction metrics: MAE, MSE, PSNR, and standard full
SSIM. It does not authorize stain corruption, model/loss training code,
checkpointing, fixed-selector generation from real Kaggle CSVs, dashboards,
artifact writers, Kaggle payload changes, Kaggle remote execution, or paper
claims. The slice may be called `data_metrics_ready` only when local focused
tests pass, artifacts are clearly local/synthetic when applicable, and no
metric implementation imports `pytorch-msssim` or historical `src/nn` code. The
slice must lock the exact binary header/CRC contract, canonical `file_index`,
`row_index`, and `sample_id` semantics, synthetic-versus-real split validation
status semantics, explicit metric column domains, exact SSIM formula details,
and JSON-safe PSNR summary behavior before any broader benchmark CLI may depend
on it.

Selector/dataloader exception, 2026-06-12: this spec now authorizes only the
narrow `fixed_selectors_dataloader_ready` local slice:
`src/eqvae/data/roots.py`, `src/eqvae/data/dataloaders.py`,
`src/eqvae/data/fixed_selectors.py`, `eqvae.cli.select_fixed_patches`, package
exports, fixed-selector placeholder config refreshes, runtime dataloader
candidate config fields, and focused tests under `tests/test_data_roots.py`,
`tests/test_dataloaders.py`, and `tests/test_fixed_selectors.py`. This
exception may implement deterministic data-root resolution, a fast read-only
mmap tensor-only dataset, deterministic fixed-selector generation and
validation, and local synthetic selector smoke tests. It does not authorize
stain corruption, model/loss training code, evaluator/artifact writers,
Kaggle payload changes, Kaggle remote execution, or paper claims. The slice may
be called `fixed_selectors_dataloader_ready` only when the selector schema,
canonical split naming, row/file-index semantics, canonical overwrite policy,
CRC policy, `data_root = "auto"` policy, and two-rail tensor/provenance policy
are locked in this spec and pass focused local Ruff, BasedPyright, and pytest
checks.

Local benchmark pre-test contract and implementation, 2026-06-12: this spec now
authorizes the narrow local benchmark slice that measures the FSQ-derived mmap
tensor-only loader path on tiny synthetic UBC-format shards before any Kaggle
remote benchmark is attempted. The slice may write local CPU synthetic
dataloader pre-test rows with `status = "local_pass"`, `benchmark_kind =
"local_synthetic_pretest"`, `benchmark_source =
"local_cpu_synthetic_pretest"`, `accelerator_mode = "local_cpu"`,
`machine_shape = "local_cpu"`, and `full_run_eligible = false`. It does not
authorize Kaggle runtime selection, training, corruption implementation,
paper-claim work, or use of local laptop throughput as a runtime decision. The
writer and CLI flag are implemented locally. In the managed sandbox,
`num_workers = 0` rows measured successfully while worker-positive rows were
recorded as explicit non-promotable failures when multiprocessing tensor
transport was unavailable. After rerunning the same command outside the sandbox
on 2026-06-12, all configured local CPU candidates measured successfully with
`status = "local_pass"`, so this narrow slice is
`local_benchmark_pretest_ready`.

Model/loss train-step implementation, 2026-06-12: this spec now authorizes and
locally implements only the narrow `model_loss_train_step_ready` slice:
`src/eqvae/models/non_equivariant_vae.py` forward-contract updates,
`src/eqvae/losses/vae.py`, optional focused helpers under `src/eqvae/training`,
optional local pre-test code under `src/eqvae/benchmarking`, config fields for
objective/optimizer/schedule smoke, `eqvae.cli.benchmark_runtime` flags needed
to write the local evidence artifact, and focused tests such as
`tests/test_vae_loss.py`, `tests/test_train_step.py`,
`tests/test_optimizer_groups.py`, and `tests/test_compile_precision_smoke.py`.
This exception may implement the VAE forward API, explicit latent-noise
control, clamped-logvar sampling/KL semantics, `L1 + 0.1 * (1 - SSIM) + beta *
KL`, beta scheduling, semantic AdamW parameter groups, and a tiny local CPU
synthetic train-step pre-test artifact. It does not authorize HED/stain
corruption, real-data training, checkpoint/resume, evaluator/artifact writers,
Kaggle payload changes, Kaggle remote execution, runtime selection, or paper
claims. The local pre-test uses `x_in = x_clean` and
`corruption_strategy = "identity_clean_no_corruption"` until the corruption
slice exists. The local artifact now writes `status = "local_pass"`, keeps
`full_run_eligible = false`, proves the zero-head forward behavior before the
first update, emits finite loss/logvar-clamp telemetry plus explicit
first-step final-head gradient/update counts, and passes focused Ruff,
BasedPyright, pytest, and the full production-scope quality gate. It also keeps
`--model-loss-train-step` as a dedicated mode that does not write
`benchmark/selected_runtime.json`.

HED/stain corruption implementation, 2026-06-13: this spec now authorizes and
locally implements only the narrow `corruption_ready` correctness/QA slice:
`src/eqvae/corruption/stain.py`, focused corruption tests, config-schema fields,
and the non-promotable local QA artifact writer needed to prove the HED/RGB
convention, RNG determinism, output range, and synthetic visual QA. Runtime
corruption math is repo-owned PyTorch and scikit-image is a dev/test oracle only.
The local CLI writes `benchmark/stain_corruptor_qa.json` with
`status = "local_pass"` and `full_run_eligible = false`. This does not integrate
corruption into real training, write promotable Kaggle runtime evidence, push
Kaggle, touch Overleaf, or make paper claims. The first real training run must
use the locked HED corruptor after training integration and fixed real-patch
visual QA are completed; `identity_clean_no_corruption` remains valid only for
the already completed local model/loss train-step smoke.

Capped Kaggle smoke implementation, 2026-06-13: this spec now authorizes only
the narrow `kaggle_smoke_ready` remote-debug slice. It adds a metadata-carrying
training dataset/collate path, keeps the existing `PatchTensorDataset`
tensor-only for throughput evidence, lets `run_train_step` use
`input_batch = corrupt(x_clean)` with clean targets, and adds a capped smoke CLI
that runs at most three real-data train steps plus one clean-validation batch
from `configs/spec0001/non_eq_vae_kaggle_debug.json`. The smoke writes
`benchmark/kaggle_smoke.json` with `status = "smoke_pass"` and
`full_run_eligible = false`; it is not runtime selection, not convergence
evidence, not a full training run, and not paper evidence. Remote execution
still requires explicit user permission, `KAGGLE_PUSH_CONFIRMED=1` for the push,
`KAGGLE_FULL_DATASET_CONFIRMED=1` for any push that attaches Kaggle sources,
and the read-only Kaggle API preflight before/after as appropriate.
Adversarial hardening requires `smoke_pass` to prove: hard caps were enforced
(`batch_size = 1`, `1 <= max_train_steps <= 3`, `max_validation_batches = 1`,
`num_workers = 0`), at least one train sample was actually corrupted, corrupted
input differed from the clean target, optimizer updates were nonzero, real-data
Kaggle smoke ran on visible T4 CUDA, model initialization was seeded from
`global_seed`, and the artifact recorded payload provenance plus data-integrity
status. The first remote version pushed on 2026-06-13 predates this hardening and
is preliminary only if it returns.
This real-data smoke intentionally attaches
`maximusshtefan/patches-pre-shuffled-ubc-ocean`; it is therefore not the right
tool for setup-only Kaggle plumbing tests or synthetic/random training-time
efficiency benchmarks. Those should use empty Kaggle source lists, generate
tiny synthetic UBC-format shards or random batches inside `/kaggle/working`,
record a distinct synthetic/setup-only status and source, and remain
non-promotable until a benchmark spec explicitly promotes them.

Synthetic Kaggle setup-smoke implementation, 2026-06-17: the first remote
real-data smoke version ended with `ModuleNotFoundError: No module named
'eqvae'` before producing a benchmark artifact, because the Kaggle CLI script
kernel upload serialized the declared `code_file` rather than the sibling
`payload/` directory. This spec now authorizes the narrow
`kaggle_setup_smoke_ready` setup-only slice: `kaggle/kernels/setup_smoke`,
`scripts/build_kaggle_embedded_kernel.py`, setup-specific guards in
`scripts/kaggle_kernel.sh`, focused tests, and setup-smoke artifact naming. The
setup kernel has empty `dataset_sources`, `enable_gpu = "false"`,
`enable_internet = "false"`, and a generated ignored `run.py` that embeds a zip
payload containing `src/eqvae`, `configs/spec0001`, `pyproject.toml`, `uv.lock`,
and a payload manifest. At runtime it checks Python >= 3.12 before importing
active code, extracts the payload under the output directory, asserts `eqvae`
was imported from that extracted payload, clears `EQVAE_DATA_ROOT`, generates
tiny synthetic UBC-format shards under the output directory, and writes
`benchmark/kaggle_setup_smoke.json` with `status_scope =
"non_promotable_setup_smoke"`, `benchmark_kind =
"synthetic_kaggle_setup_smoke"`, `benchmark_source =
"kaggle_script_kernel_synthetic_setup_smoke"`, `full_run_eligible = false`,
and `requires_cuda_t4 = false`. It is packaging/API/import/artifact evidence
only; it is not real-data loader evidence, runtime selection, convergence
evidence, or paper evidence. The real-data smoke and future benchmarks must not
reuse the setup source strings to bypass T4 or dataset checks. Remote
setup-smoke v1 completed on 2026-06-17 with this non-promotable setup artifact
contract and clean embedded payload provenance for commit `3162bec`.

Synthetic binary Kaggle timing pretest contract and implementation, 2026-06-18:
this spec authorizes the no-dataset GPU timing pretest as non-promotable
screening evidence only. The pretest exists to screen and order candidate
real-data runtime benchmark rows before paying the 60 GB+ dataset attachment
cost. It must not write `benchmark/selected_runtime.json`, must not set
`full_run_eligible = true`, and must not claim final batch size, precision
policy, corruption strategy, dataloader settings, single-vs-dual T4 selection,
convergence, paper evidence, or full-run readiness. The pretest must use a
separate Kaggle script kernel with `dataset_sources = []`, all other Kaggle
source lists empty, `enable_gpu = "true"`, `machine_shape = "NvidiaTeslaT4"`,
`enable_internet = "false"`, and a distinct push guard marker such as
`KAGGLE_SYNTHETIC_TIMING_READY = True`. It must generate deterministic
UBC-format binary+CSV shards under `/kaggle/working` through a streaming writer,
not by materializing the whole shard in RAM. The default synthetic profile is
`synthetic_binary_2gib_histology_like_v1`: 10,912 total patches, split as
5,456 train and 5,456 validation, with `3x256x256` CHW `uint8` payloads,
the standard 64-byte `<8sIQiiii3s25x` header, CRC32, and metadata CSVs. This is
2,145,386,496 payload bytes before CSV and artifacts, about 1.998 GiB, and
supports non-wrapping 30-batch timing rows through global batch 128. The
generated root must contain the same relative filenames as the real dataset:
`dataset/ubc_train_shuffled.bin`, `dataset/ubc_train_shuffled.csv`,
`dataset/ubc_ocean_valid.bin`, and `dataset/ubc_ocean_valid.csv`. Its train CSV
must mimic the real train CSV by omitting `idx`; its validation CSV must mimic
the real validation CSV by including `idx`. Timed reads must use the active
`resolve_patch_data_paths`, `PatchTensorDataset`, and `PatchTrainingDataset`
paths with an explicit `/kaggle/working/...` data root; a synthetic-only loader
or alternate binary parser is not allowed. The payload is about 805,306,368
bytes before CSV and artifacts in the historical remote-v1 0.81 GB profile; the
current default is the 2 GiB-scale profile above. The pretest must attempt both
`single_visible_t4` and `dual_t4_ddp`, and must compare candidates by feasible
global throughput and projected epoch time, not by equal per-device batch size
alone. Batch-size rows whose
`global_batch_size * non_wrapping_eligibility_steps` exceeds the available split
size are fit/VRAM probes only and cannot rank throughput recommendations. The
default `non_wrapping_eligibility_steps` is 30, independent of the shorter
initial fit pass; the 2 GiB-scale default keeps dual-T4 per-device batch 64
non-wrapping eligible at global batch 128. Generated `/kaggle/working`
data is hot-cache-biased by construction, so dataloader timing from this pretest
is format/path/H2D plumbing evidence only; real dataloader settings remain
blocked until the real-data Kaggle benchmark writes real train/validation loader
rows.

## Purpose

Replace the historical FSQ autoencoder experiment with a normal denoising VAE
baseline whose operations can be translated to the future continuous `SO(2)`
steerable model.

Keep the broad historical FSQ architecture family: the replacement baseline is
ResNet18-like, with residual basic blocks and a spatial encoder/decoder
topology. The replacement removes FSQ quantization and non-translatable
operations, not the ResNet-like macro-architecture.

This is the first implementation target before building the full repo-owned
continuous `SO(2)` path.
The spec was reopened after adversarial review and user correction: the previous
`4x4` latent target was too compressed for the intended spatial-coherence
comparison. This spec must be relocked only after the open questions near the end
are resolved. It is not a final paper-claim contract: final claims still require
a sealed masked-WSI test shard.

## Non-Goals

- No FSQ, vector quantization, codebooks, discrete indices, or quantized latent
  telemetry.
- No FSQ bottleneck scalar scale parameter `s`, straight-through rounding, or
  tanh-bounded latent-domain trick. The Gaussian VAE bottleneck uses only `mu`,
  `logvar`, sampled `z`, and the declared KL policy.
- No final performance claims against the steerable model.
- No thesis repo update.
- No reflection, `O(2)`, or other symmetry ablation.
- No arbitrary baseline layer unless its steerable counterpart is already known
  or explicitly documented as a temporary non-comparable ablation.
- No Kaggle push until the placeholder kernel is replaced and the user gives
  explicit push permission.

## Inputs And Data Contract

First-run input contract:

- dataset: UBC-OCEAN histopathology patches from the current Kaggle pipeline;
- source behavior inventory: `docs/behavior_inventory_kaggle.md`, derived from
  `kaggle/train_runs`, `kaggle/dataset_generation`, and
  `kaggle/generate_dataset_Classification_With_Masks`;
- image size: 256x256 RGB;
- binary patch shape: `3x256x256`, CHW, `uint8`;
- binary patch header format: little-endian `struct` format
  `<8sIQiiii3s25x`, exactly 64 bytes, with fields
  `magic = b"UBC_DATA"`, CRC32 over the payload bytes, patch count, channel
  count, height, width, format version `1`, and layout `b"CHW"`;
- a loader integrity pass validates header magic, version, layout, channel
  count, image height/width, declared patch count, CSV row count, file size
  `64 + patch_count * channels * height * width`, and CRC32 when
  `validate_crc = true`;
- real benchmark data may skip full-file CRC during normal repeated loader
  construction only if a separate data-integrity artifact or manifest records a
  prior CRC pass for that exact file hash/header. Any artifact claiming
  data-integrity `pass` must state whether CRC was checked or covered by that
  manifest. Because no broader real-data integrity manifest exists yet,
  canonical fixed-selector config overwrites must run with `validate_crc = true`
  and record `crc_checked = true` before writing the tracked selector JSON;
- normalization: convert `uint8` to float in `[-1, 1]` with
  `x = image.float() / 127.5 - 1.0`;
- model output: raw normalized RGB reconstruction values in the same coordinate
  system as the target, without a final `tanh` or hard output clamp;
- image-domain projections for SSIM, PSNR, saved images, and visual artifacts
  use `clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)`;
- training input: corrupted patch `x_in = corrupt(x_clean)`;
- target: clean patch `x_clean`;
- train/validation source: confirmed Kaggle dataset
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`;
- required train files: `dataset/ubc_train_shuffled.bin` and
  `dataset/ubc_train_shuffled.csv`;
- required validation files: `dataset/ubc_ocean_valid.bin` and
  `dataset/ubc_ocean_valid.csv`;
- synthetic Kaggle timing shards must use the same relative filenames, binary
  header, payload order, CSV column semantics, `idx` asymmetry, and active
  loader APIs as the real shards. They may change only the data root and pixel
  generator. A synthetic timing artifact must prove that the generated root was
  resolved by `resolve_patch_data_paths`, that each split passed
  `PatchShardSpec` validation with CRC coverage recorded in the manifest, and
  that measured train/validation batches came from the same dataset classes used
  for real UBC shards;
- `data_root = "auto"` is deterministic and offline: explicit `--data-root`
  always wins; auto mode checks nonblank `EQVAE_DATA_ROOT`, then known Kaggle
  input mounts, then repo-root local paths such as
  `data/patches-pre-shuffled-ubc-ocean` and `dataset`. It must not use the
  process current working directory as an implicit candidate;
- canonical public split names are `train` and `validation`. Historical
  `valid` and `val` may be accepted only as input aliases; generated selector
  files, sample IDs, metrics, and artifacts must write `validation`;
- CSV schema rule: load by column name; `idx` is optional because train metadata
  has `wsi_id,label,x,y` while validation metadata has `idx,wsi_id,label,x,y`;
  if `idx` exists it is the canonical `file_index` used for binary offsets and
  must be unique, nonnegative, contiguous, and in range; if `idx` is absent,
  CSV row order is the canonical `file_index`;
- every loaded record keeps both `row_index` and `file_index`;
- hot-path training and throughput loaders use a two-rail policy:
  `PatchTensorDataset` indexes by CSV `row_index`, maps to binary `file_index`,
  uses worker-local read-only mmap, and returns only CHW `uint8` tensors;
  selector/evaluation code carries provenance separately through records and
  fixed-selector JSON. Replay of selector files must use `row_index` for
  PyTorch subset/dataset indexing after validation, or explicit direct
  `file_index` reads when bypassing row indexing;
- canonical `sample_id` is
  `{split}:{file_index:08d}:{wsi_id}:{label}:{x}:{y}` when the split is known,
  or `unknown:{file_index:08d}:{wsi_id}:{label}:{x}:{y}` inside generic loader
  helpers;
- train/validation split verification: 322 train WSIs and 39 validation WSIs,
  both non-TMA and with zero overlap with supplemental-mask image IDs;
- split validation status must distinguish `synthetic_pass` from real-data
  `pass`. Synthetic fixtures may use tiny counts and generated WSI IDs. Real
  benchmark data may claim `pass` only if exact train/validation patch counts,
  exact train/validation WSI counts, train/validation WSI non-overlap,
  supplemental-mask holdout non-overlap, and non-TMA provenance are all checked
  from official metadata or from a committed manifest tied to exact source file
  hashes. If non-TMA provenance is unavailable, the real-data status is
  `warn` or `fail`, not `pass`;
- patch CSV label mapping: `0=CC`, `1=EC`, `2=HGSC`, `3=LGSC`, `4=MC`;
- train/validation patch counts: 300000 train patches and 30000 validation
  patches;
- local tests and debug runs must support synthetic/generated patch shards so
  the laptop workflow does not require downloading the Kaggle binaries;
- synthetic/generated patch shards must use the same header format, CRC32, CSV
  column rules, CHW layout, `uint8` dtype, and split-validation code paths as
  real UBC shards, while allowing tiny image sizes and tiny expected counts for
  local fixtures.

Masked holdout contract:

- the pre-shuffled patch dataset does not contain a held-out test shard;
- exact masked holdout candidate list:
  `docs/data/ubc_ocean_masked_holdout_ids.csv`;
- candidate pool: 152 UBC-OCEAN non-TMA WSIs with supplemental masks;
- mask policy: supplemental masks are not exhaustive over each WSI; use masked
  WSIs as a held-out slide pool, but do not treat unmasked regions inside those
  WSIs as exhaustive negative labels;
- target sealed test dataset slug to create:
  `maximusshtefan/ubc-ocean-masked-holdout-patches`;
- target sealed test files:
  `dataset/ubc_ocean_test.bin`, `dataset/ubc_ocean_test.csv`,
  `dataset/ubc_ocean_test_manifest.csv`, and
  `dataset/ubc_ocean_test_provenance.json`;
- until that dataset exists and is locked, runs are train/validation-only and
  must not be used for final paper claims.

## Corruption Contract

Use the same denoising corruption policy for the baseline and future steerable
model:

- apply corruption to `x_clean` after `[-1, 1]` normalization;
- use a Tellez-style HED/optical-density stain-coordinate jitter plus mild
  image-space Gaussian noise as the first implementation;
- cite and frame this as stain-domain randomization for robust denoising, not as
  a calibrated physical scanner or section-thickness simulator;
- do not copy the historical notebook corruptor from `kaggle/train_runs`; a
  clean implementation is required because the historical CHW/HED matrix
  convention is ambiguous and likely wrong for channel-first left multiplication,
  it uses a historical linear-RGB path that is not the scikit-image stain-helper
  convention, and it relies on global RNG state.

The canonical HED convention is scikit-image-compatible, but runtime training
must use a repo-owned PyTorch implementation:

- scikit-image `rgb2hed`/`hed2rgb`, `separate_stains`, and `combine_stains` are
  the oracle for tests and documentation, not a hot-path dependency;
- the PyTorch implementation must be made of tensor operations and fixed buffers
  that are eligible for `torch.compile`;
- the public API takes and returns NCHW RGB tensors on the normalized `[-1, 1]`
  scale;
- internally, convert to RGB transmission values with `rgb = (x + 1) / 2` on the
  `[0, 1]` scale before stain separation, and convert back with
  `x = rgb * 2 - 1` before returning;
- do not apply the historical sRGB-to-linear gamma decode in this first
  convention; if a later experiment wants linear-RGB optical density, it needs a
  separate profile/spec and must not be described as scikit-compatible;
- use the scikit-image v0.26.0 HED basis:

  ```text
  rgb_from_hed =
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]]
  hed_from_rgb = inverse(rgb_from_hed)
  ```

- for channel-first tensors, document the exact left-multiplication convention
  and test it against channel-last scikit-image oracle outputs so row/column
  transposition mistakes fail fast;
- match scikit-image stain-helper optical-density semantics:
  `rgb_safe = max(rgb, 1e-6)`, `hed = log(rgb_safe) / log(1e-6) @ hed_from_rgb`,
  clamp `hed >= 0`, reconstruct with
  `rgb = exp(-(hed * -log(1e-6)) @ rgb_from_hed)`, then clamp RGB to `[0, 1]`;
- identity parameters must round-trip RGB within tolerance on valid
  nonnegative-HED-manifold fixtures before noise. Arbitrary RGB is not required
  to be losslessly round-trippable after `rgb2hed` clamps negative stain
  channels to zero under scikit-image semantics.

First-run corruption profiles:

| Profile | Use | H/E alpha | H/E beta | residual-axis alpha | residual-axis beta | noise std |
| --- | --- | --- | --- | --- | --- | --- |
| `conservative_default` | first real run default | `[0.80, 1.20]` | `[-0.05, 0.05]` | `[0.98, 1.02]` | `[-0.01, 0.01]` | per-image `Uniform(0.0, 0.05)` |
| `fsq_legacy_wide` | later benchmark/ablation only | `[0.75, 1.25]` | `[-0.10, 0.10]` | `[0.98, 1.02]` | `[-0.01, 0.01]` | per-image `Uniform(0.0, 0.05)` |

`conservative_default` is the only default for the first run. The wider
historical FSQ profile may be benchmarked, but benchmark rows and artifacts must
record the profile name so convergence evidence from different corruption
strengths is never mixed. The third HED channel is not biological DAB for this
H&E dataset; it is a tiny residual-axis jitter used to reduce an obvious
corruption signature. Paper/spec wording must not claim DAB stain variation.

Corruption RNG policy:

- sample Bernoulli corruption decisions, H/E/residual alpha-beta parameters,
  per-image Gaussian noise standard deviations, and Gaussian noise tensors from
  stateless per-sample seeds;
- derive the semantic seed from
  `corruption_seed`, `split`, `semantic_sample_key`, `corruption_step`,
  `corruption_view`, and `corruption_version`;
- the semantic sample key is `{split}:{wsi_id}:{label}:{x}:{y}`. Keep
  `file_index`, `row_index`, and the existing audit `sample_id` in metadata, but
  do not use physical file order as the semantic corruption identity;
- do not include rank in the semantic per-sample seed. Log rank/world size as
  execution context only, so the same patch receives the same corruption when it
  moves between single-GPU and DDP rows for the same step/view;
- clean validation and clean test views must not call the corruptor and must not
  consume corruption RNG. Any corrupted validation/robustness view must be named
  explicitly, for example `eval_corrupted`, with a fixed view/step seed.

Required output and metadata contract:

- preserve input shape, device, dtype, RGB channel order, and public `[-1, 1]`
  range;
- run HED/OD transforms, logarithms, exponentials, stain parameter sampling, and
  noise generation in FP32 under `torch.no_grad()`;
- clamp the final corrupted tensor to `[-1, 1]` after stain jitter and image-space
  Gaussian noise;
- preserve the clean target `x_clean`; masks and labels are not modified by the
  corruptor;
- record per-sample metadata: `applied`, semantic key, derived seed, profile name,
  H/E/residual alpha-beta values, sampled noise std, finite/range checks,
  pre-clamp min/max, final min/max, and lower/upper clamp fractions.

Required stain-corruptor implementation rules:

- store the fixed HED stain matrix with an explicit channel-first convention and
  a documented scikit-image source/version for oracle fixtures;
- add unit tests against scikit-image reference outputs on small tensors; the
  tests may use scikit-image as an oracle, but active runtime code must not call
  it in the training path;
- identity parameters must round-trip valid nonnegative-HED-manifold RGB within
  tolerance before noise;
- H/E and residual-axis perturbation parameters must affect the intended HED
  channels, not transposed mixtures;
- use stateless Torch RNG derived from the semantic seed contract above;
- do not consume stain/noise RNG in clean validation mode;
- preserve input shape and dtype, and document any memory-format conversion;
- implement `branchless_all` as the first execution strategy; keep
  `indexed_masked` as a later runtime benchmark candidate only;
- generate a fixed visual QA artifact showing clean, stain-only, Gaussian-only,
  and combined corrupted patches before the first Kaggle baseline run. Synthetic
  QA is sufficient for the first local correctness slice; fixed real-patch QA is
  required before the first Kaggle baseline run.

Relevant stain-domain references to cite in the paper/spec implementation:

- Ruifrok and Johnston, 2001, optical-density color deconvolution:
  https://pubmed.ncbi.nlm.nih.gov/11531144/;
- Macenko et al., 2009, OD/SVD stain normalization:
  https://doi.org/10.1109/ISBI.2009.5193250;
- Vahadane et al., 2016, structure-preserving stain normalization:
  https://doi.org/10.1109/TMI.2016.2529665;
- Tellez et al., 2018/2019, HED stain augmentation and HED-light/HED-strong
  benchmark settings: https://arxiv.org/abs/1808.05896 and
  https://arxiv.org/abs/1902.06543;
- RandStainNA, 2022, data-driven stain augmentation in HED/HSV/LAB spaces:
  https://arxiv.org/abs/2206.12694.

## Model Contract

The baseline must be generated from a ResNet-like layer schedule that the
equivariant model can reuse. The non-equivariant convolutions are ordinary
`torch.nn.Conv2d`; all channels are treated as scalar tensor channels, and each
convolution may freely mix all input channels. The residual macro-topology,
capacity bookkeeping, kernels, upsampling, latent shape, and gate family must
mirror the planned `SO(2)` path.

First-run fixed choices:

| Item | Value |
| --- | --- |
| Input | 256x256 RGB |
| Latent | spatial Gaussian latent `(B, 16, 32, 32)` |
| Normalization layers | baseline `GroupNorm`; future SO(2) field-aware norm |
| Stem kernel | 7x7, same padding |
| Hidden/down/up kernels | 5x5, same padding |
| VAE head kernels | 5x5, same padding |
| Padding mode | zero padding for train/model code; border-cropped metrics for equivariance diagnostics |
| Upsampling | bilinear scale factor 2 followed by convolution |
| Future SO(2) kernel basis | Gaussian radial shells plus real angular harmonics, `L <= 2` |
| Output | zero-initialized final 5x5 convolution to raw RGB, no final `tanh` |
| KL convention | mean over batch, latent channels, and latent spatial positions |

Future `SO(2)` kernel-basis policy is locked for the first implementation:

- use repo-owned analytic polar-harmonic basis construction with Gaussian radial
  shells and real angular harmonics `cos(m theta), sin(m theta)`;
- 5x5 kernels use radial shell centers `[0, 1, 2]`;
- 7x7 kernels use radial shell centers `[0, 1, 2, 3]`;
- use approximate ring widths `0.6` for interior rings, `0.4` for the outer
  ring, and a tiny origin width;
- angular frequencies `m > 0` have zero support at the kernel center; the center
  sample may only carry the `m = 0` spatial angular component, while still
  allowing legal intertwiners between compatible same-frequency input and output
  irreps;
- precompute basis buffers and learn only expansion coefficients;
- expand to dense `conv2d` inside the compiled forward path;
- allow scalar-output bias only where the representation policy permits it;
- keep Fourier-Bessel/Bessel bases as a future fallback/ablation requiring a
  separate radius, boundary, radial-order, and sampled-zero policy.

Encoder spatial schedule:

```text
256 -> 128 -> 64 -> 32
```

Channel and future field-capacity schedule:

| Spatial size | Total channels | Future scalar fields | Future F1 irrep copies | Future F2 irrep copies |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 16 | 8 | 0 |
| 128 | 48 | 16 | 12 | 4 |
| 64 | 64 | 24 | 12 | 8 |
| 32 | 96 | 32 | 20 | 12 |
| latent | 16 | 16 | 0 | 0 |

For the first non-equivariant baseline, the table is capacity bookkeeping only:
it does not restrict Conv2d mixing, activation grouping, or residual addition.
The scalar/F1/F2 columns define the planned `SO(2)` field multiplicities that a
future steerable model should mirror when reporting capacity and parameter/FLOP
differences.

The first complete run uses scalar/trivial latent fields only. Frequency-1 or
frequency-2 latent fields require a follow-up spec because `logvar`, sampling,
and KL must become representation-aware.

Encoder block pattern:

```text
Stem:
  Conv2d(3 -> 32, kernel=7, stride=1, padding=3, bias=False)
  Norm(32)
  ActivationPolicy(32)

ResBlock(in_channels, out_channels, downsample):
  main:
    Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
    ActivationPolicy(out_channels)
    if downsample: FixedBinomialLowpassDownsample2x(out_channels)
    Conv2d(out_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
  skip:
    identity if not downsample and in_channels == out_channels
    otherwise ResNet-D-style projection:
      if downsample: FixedBinomialLowpassDownsample2x(in_channels)
      Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
      Norm(out_channels)
  output:
    ActivationPolicy(out_channels)(main + skip)

Encoder stages:
  stage 256: two ResBlocks at 32 channels, first downsample=False
  stage 128: two ResBlocks at 48 channels, first downsample=True
  stage 64: two ResBlocks at 64 channels, first downsample=True
  stage 32: two ResBlocks at 96 channels, first downsample=True

VAE heads:
  mu_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2, bias=True)
  logvar_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2, bias=True)
```

Decoder mirrors the encoder:

```text
Latent projection:
  Conv2d(16 -> 96, kernel=5, stride=1, padding=2, bias=False)
  Norm(96)
  ActivationPolicy(96)

UpResBlock(in_channels, out_channels, upsample):
  main:
    optional bilinear upsample(scale_factor=2, align_corners=False)
    Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
    ActivationPolicy(out_channels)
    Conv2d(out_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
  skip:
    identity if not upsample and in_channels == out_channels
    otherwise ResNet-D-style up projection:
      if upsample: bilinear upsample(scale_factor=2, align_corners=False)
      Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
      Norm(out_channels)
  output:
    ActivationPolicy(out_channels)(main + skip)

Decoder stages:
  stage 32: two UpResBlocks at 96 channels, first upsample=False
  stage 64: two UpResBlocks to 64 channels, first upsample=True
  stage 128: two UpResBlocks to 48 channels, first upsample=True
  stage 256: two UpResBlocks to 32 channels, first upsample=True

Output:
  Conv2d(32 -> 3, kernel=5, stride=1, padding=2, bias=True)
  zero-initialize weight and bias
```

Canonical residual-block topology table:

| Block | Spatial in -> out | Channels in -> out | Resampling | Skip kind | Learned convs | Norms | Gates | Fixed resampling ops |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `enc0` | `256 -> 256` | `32 -> 32` | none | identity | 2 | 2 | 2 | 0 |
| `enc1` | `256 -> 256` | `32 -> 32` | none | identity | 2 | 2 | 2 | 0 |
| `enc2` | `256 -> 128` | `32 -> 48` | downsample | downsample then projection conv | 3 | 3 | 2 | 2 |
| `enc3` | `128 -> 128` | `48 -> 48` | none | identity | 2 | 2 | 2 | 0 |
| `enc4` | `128 -> 64` | `48 -> 64` | downsample | downsample then projection conv | 3 | 3 | 2 | 2 |
| `enc5` | `64 -> 64` | `64 -> 64` | none | identity | 2 | 2 | 2 | 0 |
| `enc6` | `64 -> 32` | `64 -> 96` | downsample | downsample then projection conv | 3 | 3 | 2 | 2 |
| `enc7` | `32 -> 32` | `96 -> 96` | none | identity | 2 | 2 | 2 | 0 |
| `dec0` | `32 -> 32` | `96 -> 96` | none | identity | 2 | 2 | 2 | 0 |
| `dec1` | `32 -> 32` | `96 -> 96` | none | identity | 2 | 2 | 2 | 0 |
| `dec2` | `32 -> 64` | `96 -> 64` | upsample | upsample then projection conv | 3 | 3 | 2 | 2 |
| `dec3` | `64 -> 64` | `64 -> 64` | none | identity | 2 | 2 | 2 | 0 |
| `dec4` | `64 -> 128` | `64 -> 48` | upsample | upsample then projection conv | 3 | 3 | 2 | 2 |
| `dec5` | `128 -> 128` | `48 -> 48` | none | identity | 2 | 2 | 2 | 0 |
| `dec6` | `128 -> 256` | `48 -> 32` | upsample | upsample then projection conv | 3 | 3 | 2 | 2 |
| `dec7` | `256 -> 256` | `32 -> 32` | none | identity | 2 | 2 | 2 | 0 |

The table above is the canonical source for residual block count verification.
It makes explicit that second blocks in every stage do not downsample/upsample,
skip projections have normalization but no gate, and each block has exactly one
post-add output gate.

Allowed first-run operations:

- odd square `Conv2d` with 7x7 only in the stem and 5x5 everywhere else;
- fixed fieldwise anti-aliased 2x downsampling for encoder stage transitions,
  used in both residual branches at their literature-consistent branch-local
  locations, followed by stride-1 odd square convolution;
- bilinear upsampling plus convolution for upsampling;
- ResNet-like residual adds with identity skips when shape/channel match;
- ResNet-D/anti-aliased-style projection skips: fixed spatial resampling first,
  then 5x5 spatial projection convolution, never 1x1 pointwise projections;
- spatial Gaussian VAE latent map;
- scalar gated activation policy defined below.

Output-head policy:

- do not apply a final `tanh`, sigmoid, or clamp in the model forward path;
- initialize the final RGB convolution weight and bias to zero, so the initial
  reconstruction is the normalized midpoint `0.0` and early training is stable;
- compute L1 against raw `x_hat` and `x_clean` in normalized `[-1, 1]`
  coordinates;
- compute SSIM, PSNR, image saving, and qualitative artifacts after projecting
  model output to image coordinates with
  `x_hat_img = clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)`;
- log output range telemetry, including `x_hat_min`, `x_hat_max`, and fraction
  of pixels below `-1` or above `1`, so the run exposes boundary behavior
  instead of hiding it behind a saturating output nonlinearity.

Residual policy:

- first-run residual connections are required and ResNet-like;
- no ReZero/Fixup/SkipInit learned residual scaling in spec 0001 unless a later
  spec explicitly adds it;
- projection skips are not naive one-shot channel adapters. They use explicit
  spatial resampling followed by odd 5x5 convolutions so they have a direct
  fixed-resampling-plus-repo-owned-SO2-convolution counterpart;
- encoder stage-transition blocks use branch-local fieldwise anti-aliased
  downsampling, following ResNet-D / BlurPool style rather than a pre-split
  downsample:
  the main branch replaces learned stride with
  `Conv5x5(stride=1) -> ActivationPolicy -> fixed_downsample_2x`, while the
  skip branch uses `fixed_downsample_2x -> Conv5x5` when spatial size changes;
- downsampling must not be hidden inside a learned stride or a one-off shortcut
  adapter;
- the fixed downsample operator is chosen from the future `SO(2)` side first:
  it must be a fieldwise spatial operator mapping a `FieldType` to the same
  `FieldType`, applying the same scalar spatial resampling to every fiber
  component and never mixing channels/frequencies;
- spec 0001 locks a repo-owned 5x5 separable binomial low-pass filter followed
  by decimation by 2:
  `kernel_1d = [1, 4, 6, 4, 1] / 16`, `kernel_2d = outer(kernel_1d, kernel_1d)`,
  zero padding `2`, and decimation by taking stride-2 samples. The Torch
  implementation may use fixed grouped `conv2d(..., groups=C, stride=2)` as an
  implementation detail, but this is a fixed fieldwise resampling operator, not
  a learned grouped/depthwise convolution;
- the fixed low-pass/downsample maps `(B, C, H, W)` to `(B, C, H/2, W/2)` for
  even `H,W`, preserves dtype/device where numerically safe, stores the filter
  as a non-trainable FP32 buffer, and applies the same scalar spatial operator
  independently to every future fiber component;
- resize/area-style scale-factor-0.5 fieldwise downsampling is moved to a later
  fallback/spike only if the locked binomial operator fails a future SO(2)
  stage-transition equivariance test. It is not a spec 0001 benchmark axis;
- FLOPs for the chosen fixed downsample are reported separately from learned
  convolutions;
- fieldwise downsampling is representation-compatible because it acts as a
  scalar spatial operator tensored with the identity on fiber components, but it
  is still a sampled-grid approximation. Future `SO(2)` stage transitions must
  include measured equivariance-error tests rather than assume perfect
  continuous-grid behavior;
- decoder up-projection skip uses bilinear upsampling before the 5x5 projection
  conv;
- parameter/FLOP counting must include all residual skip projections and fixed
  resampling operators;
- the future `SO(2)` model must mirror residual topology with matching
  `FieldType`s before addition.

Analytic Conv2d baseline count target:

The locked non-equivariant topology above has the following analytic count for a
single `256x256` RGB sample. Count MACs as multiply-accumulates; if reporting
FLOPs with the common multiply-plus-add convention, use `FLOPs = 2 * MACs`.

| Count target | Value | Notes |
| --- | ---: | --- |
| Learned convolution count | 43 | Includes skip projections, VAE heads, and RGB head |
| Normalization module count | 40 | `GroupNorm` modules with affine parameters |
| Gate module count | 34 | One learned scalar gate per hidden activation site |
| Fixed resampling op count | 12 | Six branch-local downsample ops plus six bilinear upsample ops |
| Learned convolution parameters | 3,949,539 | Includes zero-initialized RGB head bias |
| GroupNorm affine parameters | 4,800 | `weight,bias` for every norm channel |
| Learned gate parameters | 4,096 | Per-channel `a,b` for 2,048 activation-channel instances |
| Total learned parameters | 3,958,435 | Convs + norms + gates |
| Learned convolution MACs/sample | 36,471,046,144 | `36.471` GMAC/sample |
| Learned convolution FLOPs/sample | 72,942,092,288 | `72.942` GFLOP/sample with `2*MAC` convention |
| Fixed resampling MACs/sample | 85,032,960 | Conservative grouped-5x5 downsample plus 4-tap bilinear upsample |
| Fixed resampling FLOPs/sample | 170,065,920 | `0.170` GFLOP/sample with `2*MAC` convention |
| Total MACs/sample with fixed resampling | 36,556,079,104 | `36.556` GMAC/sample |

Section-level learned-convolution count:

| Section | Learned conv params | Learned conv MACs/sample |
| --- | ---: | ---: |
| Stem | 4,704 | 308,281,344 |
| Encoder residual body | 1,811,200 | 17,013,145,600 |
| VAE heads | 76,832 | 78,643,200 |
| Decoder and RGB head | 2,056,803 | 19,070,976,000 |

Activation-memory planning target:

- summing all learned-conv output tensors once gives `36,110,336` elements per
  sample;
- this rough activation-output sum is `137.75 MiB/sample` in FP32 and
  `68.88 MiB/sample` in FP16;
- largest individual hidden maps are `32x256x256 = 2,097,152` elements
  (`8 MiB` FP32), `48x128x128 = 786,432` elements (`3 MiB` FP32),
  `64x64x64 = 262,144` elements (`1 MiB` FP32), `96x32x32 = 98,304`
  elements (`0.375 MiB` FP32), and latent `16x32x32 = 16,384` elements
  (`0.0625 MiB` FP32);
- this is not a full autograd peak-memory estimate. The benchmark must still
  measure `max_vram_allocated_mb`, `max_vram_reserved_mb`, and headroom on
  Kaggle.

Implementation requirements:

- the historical first local scaffold `model_count` CLI/test was allowed to
  write `benchmark/model_count.json` from the analytic target above as a schema
  and contract smoke only;
- the current topology-count slice must instantiate the locked model topology,
  observe inventory input/output shapes and execution order with meta-forward
  hooks, and compare learned parameters, learned-conv MACs, fixed-resampling
  MACs, and activation elements against the target above before downstream
  runtime rows are eligible;
- if the fixed binomial downsample is implemented separably, report both the
  actual implementation MACs and the conservative dense grouped-5x5 equivalent;
- any topology change that moves a resampling op, adds/removes a norm, changes
  a kernel size, changes gate placement, or adds an uninventoried leaf module
  must update this count section in the same patch.

Banned first-run operations:

- FSQ or any vector-quantized bottleneck;
- PixelShuffle or sub-pixel convolution;
- nearest-neighbor upsampling in the comparable path;
- 1x1 pointwise convolutions;
- learned depthwise/grouped/MBConv/squeeze-excite/channel-attention operations;
- `BatchNorm2d`, `LayerNorm`, channel dropout, or arbitrary normalization that
  cannot be mapped to the future SO(2) field schedule. Baseline `GroupNorm` is
  required and the future SO(2) counterpart is repo-owned field-aware norm;
- arbitrary flattening, channel slicing, `.chunk()`, or tensor reshaping that
  cannot be mapped to future `GeometricTensor` field boundaries;
- FSQ-era resume sources or discrete-latent artifact requirements.

## Activation Contract

Do not use arbitrary componentwise SiLU everywhere. The baseline must use a
shared gated activation family that the future `SO(2)` model can mirror.
SiLU/Swish is treated as `x * sigmoid(x)`, not as a special unrelated
nonlinearity.

For the non-equivariant baseline, every hidden channel is a scalar tensor
channel. Apply the scalar gate componentwise to all channels, and allow the
surrounding Conv2d layers to mix channels freely.
These learned `a_i,b_i` gate parameters are intentionally added to the baseline
as well as the future `SO(2)` scalar/trivial fields. The purpose is to restore
some pointwise activation expressivity that the equivariant model loses when it
cannot use arbitrary componentwise nonlinearities, while keeping scalar-field
nonlinear expressivity matched between models.

Baseline scalar gate:

```text
gate_i = sigmoid(a_i * x_i + b_i)
out_i = gate_i * x_i
```

Rules for the baseline scalar gate:

- `a_i` and `b_i` are learned scalar parameters per channel;
- initialize to ordinary SiLU/Swish behavior where possible
  (`a=1`, `b=0`);
- do not add scalar activation parameters only to one model. The non-equivariant
  baseline and future `SO(2)` scalar/trivial fields use the same learned
  pointwise scalar gate family;
- do not tie or group baseline activation parameters by future field schedule in
  the first run. Any grouped activation tying requires a later explicit
  ablation/spec.

Future `SO(2)` counterpart:

- scalar/trivial fields use the same learned pointwise scalar gate family;
- learned additive bias is allowed only on scalar/trivial output fields;
- nontrivial 2D irrep copies use a radial gate over an invariant norm;
- learned additive vector bias is forbidden on nontrivial irrep/vector fields;
- nontrivial radial gates are implemented and tested as part of the activation
  policy, but they are not applied to fake vector pairs in the first scalar
  Conv2d baseline.

For each future 2-channel irrep copy `v = (u, w)`:

```text
r = sqrt(||v||**2 + eps) = sqrt(u**2 + w**2 + eps)
gate = sigmoid(a_i * r + b_i)
out = gate * v
```

Rules for future radial gates:

- the two components in a vector pair must share the same gate;
- initialize future vector/irrep copies to the same neutral gate convention as
  scalar fields (`a=1`, `b=0`) unless a later spec changes this;
- future vector/irrep copies may have scalar gate bias `b_i`, but must not have
  an additive learned 2D vector bias because that would break `SO(2)`
  equivariance;
- `eps` is required for stable gradients near zero vector norm. It must be large
  enough to avoid FP16 underflow/instability in AMP runs, configured explicitly,
  and tested in local/benchmark smoke. First candidate: `eps = 1e-4`;
- no learned activation amplitude `gamma` is used in spec 0001. Amplitude is
  handled by convolutions and normalization affine parameters, and `gamma` is
  reserved for a later ablation if the equivariant model is underpowered;
- gate parameters are included in trainable parameter counts and reported
  separately as a count and percentage of the model;
- implement this as an explicit `GatedScalarActivation`,
  `RadialGate`, and `ActivationPolicy` module using a central field schedule,
  not ad hoc reshaping inside model blocks;
- add a unit test that rotates synthetic vector pairs and verifies
  `activation(rho(theta) v) == rho(theta) activation(v)` within tolerance.

Normalization contract for the real run:

- the non-equivariant Conv2d baseline uses ordinary `torch.nn.GroupNorm` with
  affine parameters;
- default baseline groups: `num_groups = 8` for hidden widths
  32/48/64/96 wherever normalization is applied, including the 96-channel
  latent projection output;
- the future SO(2) model uses a repo-owned field-aware norm, not arbitrary raw
  GroupNorm over tensor channels;
- scalar/trivial fields may use additive affine bias;
- nontrivial frequency-1/frequency-2 vector fields may use invariant scalar
  scale parameters, but no additive learned vector bias;
- vector/irrep normalization uses invariant energy over whole irrep copies, for
  example RMS over `(copy, component, spatial)` groups chosen in the field
  schedule. It must never split a 2D irrep copy or group frequency-1 and
  frequency-2 components as if they were ordinary channels;
- normalization placement is after learned convolutions and before activation;
- VAE `mu_head`, `logvar_head`, and the final RGB output head do not use
  normalization;
- when a projection skip has a learned projection convolution at a location where
  the matching main branch is normalized before residual addition, normalize the
  projection branch before the add as well;
- convolution bias is disabled when immediately followed by normalization;
  scalar affine bias lives in the normalization or scalar activation. Learned
  biases remain allowed for scalar-only heads that are not followed by
  normalization.

## Objective Contract

Use a normal denoising VAE with a composite reconstruction objective:

```text
mu, logvar_raw = encode(x_in)
logvar_clamped = clamp(logvar_raw, -8.0, 4.0)
z = mu + exp(0.5 * logvar_clamped) * eps
x_hat = decode(z)
l1_loss = mean(abs(x_hat - x_clean))
ssim_loss = 1 - mean(ssim_per_image(project_for_ssim(x_hat), project_for_ssim(x_clean)))
recon_loss = l1_loss + ssim_weight * ssim_loss
kl_element = -0.5 * (1 + logvar_clamped - mu ** 2 - exp(logvar_clamped))
kl_loss = mean(kl_element)
loss = recon_loss + beta * kl_loss
```

This objective is a composite beta-VAE-style objective, not a strict Gaussian
ELBO. Keep MSE and PSNR as metrics, but do not optimize MSE in the first run.
Implement SSIM as repo-owned Torch code that runs in FP32 and can be included in
the compiled step function without internet or undeclared Kaggle dependencies.
First locked `ssim_weight`: `0.1`.

VAE forward API contract for training and benchmark code:

- `NonEquivariantVAE.forward()` is the authoritative stochastic training
  forward and must accept an optional explicit `eps` tensor. When `eps` is
  provided, the forward path must use it exactly and must not call
  `torch.randn_like`; this is required for paired numerical checks and compile
  comparisons. When `eps` is absent, the method may sample internally from the
  active Torch RNG.
- `VaeForwardOutput` must expose at least `reconstruction`, `mu`, raw
  `logvar`, `logvar_clamped`, sampled `z`, used `eps`, and
  `logvar_clamp_count`. The raw `logvar` field is model-output evidence;
  `logvar_clamped` is the tensor used for sampling and KL.
- Shapes are fixed for the first run: `reconstruction` is `(B, 3, 256, 256)`,
  while `mu`, `logvar`, `logvar_clamped`, `z`, and `eps` are
  `(B, 16, 32, 32)`.
- Posterior arithmetic, logvar clamp, latent sampling, KL, SSIM, L1, beta
  weighting, and total loss composition run in FP32 with gradients enabled.
  The reconstruction tensor may carry the surrounding model dtype, but loss code
  casts it to FP32 before arithmetic.
- Deterministic artifact code may decode `mu` directly through `decode(mu)` for
  mean-posterior reconstructions, but it must label that path separately from
  stochastic sampled reconstructions. Training loss uses sampled `z`, not
  `decode(mu)`.
- `logvar_clamp_count` counts elements where raw `logvar` differs from
  `logvar_clamped`; telemetry must also record raw and clamped logvar summaries
  when the train-step slice writes evidence.

Training-loss reduction contract:

- `l1_loss` is the global mean absolute error over all `B,C,H,W` raw normalized
  reconstruction and target elements.
- `ssim_loss` is `1 - mean(ssim_per_image(...))`, where `ssim_per_image`
  follows the standard full SSIM metric contract below on projected image-domain
  tensors.
- `kl_loss` is the global mean of `kl_element` over all batch, latent channel,
  and latent spatial elements.
- The loss API must return scalar tensors for `loss`, `recon_loss`, `l1_loss`,
  `ssim_loss`, and `kl_loss`, plus JSON/CSV-safe detached telemetry values for
  benchmark artifacts.

Reconstruction metric contract:

- metric inputs are NCHW tensors with shape `(B, C, H, W)`, matching channel
  count, spatial size, device, and finite numeric values;
- official MAE and MSE columns are computed on raw normalized `[-1, 1]`
  tensors and named `mae_norm` and `mse_norm`;
- image-domain MAE/MSE may be added later only with explicit names such as
  `mae_img` and `mse_img`, never by reusing the official normalized columns;
- official PSNR and SSIM columns are computed on image-domain tensors in
  `[0, 1]` and named `psnr_img` and `ssim_img`;
- model outputs and targets are projected for PSNR/SSIM with
  `clamp((x + 1.0) / 2.0, 0.0, 1.0)` outside the model forward path;
- all metric arithmetic runs in FP32, including when model outputs arrive as
  FP16/BF16 tensors;
- PSNR uses `data_range = 1.0` and returns `inf` for exactly zero MSE;
- standard full SSIM uses the Wang-style full-reference formula with
  `data_range = 1.0`, `K1 = 0.01`, `K2 = 0.03`, an odd `11x11` Gaussian
  window, `sigma = 1.5`, grouped per-channel convolution, reflect padding of
  five pixels, and no learned or external dependency;
- the SSIM Gaussian kernel is built in FP32 from positions `[-5, ..., 5]` with
  weights `exp(-(x**2)/(2*sigma**2))`, normalized so the 1D kernel sums to 1,
  then outer-producted into a 2D window. Local variance and covariance use
  `E[x^2] - E[x]^2` and `E[xy] - E[x]E[y]`; the SSIM map is
  `((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) /
  ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))`;
- SSIM returns per-image values by averaging the SSIM map over channel and
  spatial dimensions, and scalar summaries average those per-image values;
- SSIM requires `H >= 11` and `W >= 11`; smaller tensors must fail clearly
  rather than silently changing the window;
- every metric summary reports `n`, population mean, and population standard
  deviation (`std = 0.0` for `n == 1`);
- JSON summaries for PSNR must not emit non-standard JSON `Infinity` or `NaN`.
  A PSNR summary with one or more infinite per-image values records
  `psnr_img_inf_count`, computes `psnr_img_finite_mean` and
  `psnr_img_finite_std` over finite values only, and sets the ordinary
  `psnr_img.mean`/`psnr_img.std` fields to JSON `null` unless all values are
  finite. CSV per-image rows may write the string `inf` for exactly zero-MSE
  samples.

First beta policy:

- full epoch-based runs: linear warmup from 0 to 1 over the first full epoch,
  then keep beta fixed at 1;
- tiny step-based debug runs: linear warmup from 0 to 1 over the first 10 percent
  of configured optimizer steps;
- for step-limited debug or local pre-test runs,
  `beta_warmup_steps = max(1, ceil(0.10 * max_optimizer_steps))`;
- beta is computed from the zero-based successful optimizer-step index before
  the optimizer update:
  `beta = target_beta * min(successful_optimizer_step / beta_warmup_steps, 1.0)`;
  therefore the first successful optimizer update uses `beta = 0.0`;
- skipped AMP steps do not increment the successful optimizer-step index and do
  not advance the beta schedule;
- no cyclic beta restarts in the first locked run;
- beta value must be logged per optimizer step.

Optimizer-step indexing convention:

- `optimizer_step_index` means the zero-based successful update index used
  before the current optimizer update and before beta/LR values for that update
  are computed. The first successful optimizer update has
  `optimizer_step_index = 0`.
- `successful_optimizer_update_count` means the completed-update count after a
  successful `optimizer.step()`. After the first successful update, this count
  is `1`.
- CSV columns named `optimizer_step` are historical-short names for
  `successful_optimizer_update_count` unless a schema explicitly uses
  `optimizer_step_index`. New local train-step artifacts must include both
  fields where an update is performed.
- Skipped AMP batches do not increment either `optimizer_step_index` or
  `successful_optimizer_update_count`; they are logged only as batch attempts.

AMP and GradScaler policy:

- support both AMP and non-AMP execution; do not assume AMP is faster until the
  Kaggle runtime benchmark measures it;
- if AMP `GradScaler` detects non-finite gradients and skips
  `optimizer.step()`, call `scaler.update()`, zero gradients, log
  `amp_step_skipped = 1`, and continue to the next batch;
- do not retry the same batch after a skipped AMP step;
- do not increment `global_step`, advance LR or beta schedulers, run
  step-triggered validation/checkpointing, or count the batch as an optimizer
  update when the optimizer step was skipped.

Precision and autograd policy:

- Mirror the useful FSQ precision structure as the conservative candidate:
  allow AMP/fp16 for the main model convolutional forward when the runtime
  benchmark selects AMP, but keep numerically sensitive islands in FP32.
- Do not assume the conservative split is fastest or necessary. The Kaggle
  runtime benchmark must compare safe precision placements and select the
  fastest one that passes numerical checks.
- Run the corruption module under `torch.no_grad()` and compute HED/OD color
  transforms, logarithms, exponentials, and random stain/noise draws in FP32.
  Corruption is data augmentation, not a differentiable model component.
- Run VAE posterior arithmetic in FP32 with gradients enabled:
  `logvar` clamp, `exp(0.5 * logvar)`, latent sampling, and KL computation.
- Run SSIM, L1, KL, beta weighting, and total loss composition in FP32 outside
  autocast. SSIM buffers/constants are FP32.
- Run radial-gate norm/sigmoid arithmetic in FP32 when AMP is enabled, using the
  configured `radial_gate_eps`, then return to the surrounding model dtype if
  needed. The gate remains differentiable with respect to the input field and
  gate parameters.
- Cast the model reconstruction output to FP32 before losses and metrics.
- The narrow local model/loss train-step pre-test is restricted to
  `precision_policy = "amp_off_fp32"` for required pass/fail evidence. CPU
  `torch.compile` and CPU float16 are bounded contract smokes only; they may
  write `status = "skipped_unsupported"` when the local CPU runtime lacks
  support, provided eager FP32 evidence passes. These local smokes are not
  Kaggle AMP or throughput proof.
- Do not wrap training model forward, VAE sampling, losses, or SO(2) basis
  expansion in `torch.no_grad()`. Fixed basis buffers are non-trainable, but
  expansion coefficients require gradients.
- Use `torch.no_grad()` for metric accumulation, range/telemetry summaries,
  validation/evaluation passes, fixed-patch artifact generation, and checkpoint
  serialization helpers.
- Unlike the historical branchless FSQ validation path, `eval_clean` must not
  call the corruptor or consume corruption RNG. Deterministic corruption is used
  only in `eval_corrupted`.

Precision candidates for the Kaggle runtime benchmark:

- `amp_off_fp32`: full FP32 training step, used as the correctness and stability
  baseline.
- `amp_conservative`: main convolutional forward under AMP/fp16; corruption,
  posterior/KL, scalar/radial gate sigmoid arithmetic, SSIM, L1, and total loss
  in FP32.
- `amp_scalar_gate_relaxed`: same as `amp_conservative`, except the
  non-equivariant scalar gate sigmoid/multiply may run in the surrounding AMP
  dtype. This policy is eligible only for the scalar Conv2d baseline and only if
  paired numerical checks against `amp_off_fp32` pass. Posterior sampling,
  `logvar`, KL, SSIM/L1/loss, corruption, and future radial-gate norm/sigmoid
  arithmetic must remain FP32 in spec 0001.

Do not relax posterior/KL/loss/corruption or radial-gate norm/sigmoid numerics
in spec 0001. A broader precision ablation requires a later spec.

The current scalar Conv2d activation implementation keeps scalar-gate sigmoid
arithmetic in FP32. The `model_loss_train_step_ready` slice does not introduce
an `amp_scalar_gate_relaxed` implementation hook; that policy becomes active
only in the later Kaggle runtime benchmark slice after paired numerical checks
exist.

Corruption strategy candidates for the Kaggle runtime benchmark:

- `branchless_all`: compute corrupted images for the full batch, sample a mask,
  and select corrupted versus clean tensors with `torch.where`, matching the
  compile-friendly historical FSQ pattern. This is the only execution strategy
  required for the first local corruption correctness/QA implementation and the
  default for the first real run unless later runtime evidence selects another
  strategy.
- `indexed_masked`: sample a mask, corrupt only selected samples, and scatter
  them back into the batch. Accept this only if `torch.compile` stays stable and
  throughput improves.
- Both strategies must produce the same training distribution, support
  reproducible RNG, and preserve the validation rule that `eval_clean` consumes
  no corruption RNG.
- equivalence tests must key randomness by semantic sample key, corruption step,
  view, and corruption version, then verify that `branchless_all` and
  `indexed_masked` produce the same Bernoulli corruption decisions, the same
  HED/noise parameters and noise-field hashes for corrupted samples, unchanged
  clean samples, no RNG consumption in `eval_clean`, and stable compile behavior
  across varying mask counts. Rank/world size are logged execution context, not
  semantic per-sample seed inputs.

Until the corruption slice is implemented, the local model/loss train-step
pre-test must use identity denoising input, `x_in = x_clean`, and must write
`corruption_strategy = "identity_clean_no_corruption"`. This strategy is valid
only for local contract evidence with `full_run_eligible = false`; it is not a
Kaggle runtime matrix candidate and must not satisfy any corruption or denoising
readiness gate.

Log at minimum:

- total loss;
- reconstruction loss;
- L1 loss;
- SSIM loss and SSIM metric;
- KL loss;
- beta;
- SSIM, MAE, MSE, PSNR;
- posterior `mu` mean/std/min/max;
- posterior `logvar` mean/std/min/max/clamp count;
- learning rate;
- `event_id`, `batch_attempt`, `optimizer_step`, and `amp_step_skipped`;
- sample count `n` for every metric summary.

MSE is a metric for the first run, not a training loss.

Validation/evaluation modes:

- `eval_clean`: encoder input is `x_clean`, target is `x_clean`; do not call the
  corruptor or consume corruption RNG;
- `eval_corrupted`: encoder input is `corrupt(x_clean)` with fixed validation
  corruption seed and logged corruption config, target is `x_clean`;
- report reconstruction metrics and KL terms for both modes with separate
  `split` and `view` labels, never as one pooled number. Required `view` values
  are `eval_clean` and `eval_corrupted`.

## Training And Config Contract

All values that affect the experiment must live in versioned JSON config files,
not hidden inside model or CLI code. JSON is the first-run config format because
it can be parsed and written with the Python standard library on offline Kaggle
kernels. `uv`, `pyproject.toml`, and `uv.lock` remain the Python environment and
dependency source of truth; they are orthogonal to experiment config files. CLI
flags may override config values only when the override is recorded in the run
config snapshot.

Required seed policy:

- `global_seed = 20260610`;
- `data_seed = 20260610`;
- `corruption_seed = 20260611`;
- `latent_seed = 20260612`;
- save Python, NumPy, and Torch RNG state in checkpoints.

Required optimizer and schedule defaults:

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `1e-5` |
| Gradient clipping | global norm `1.0` |
| LR warmup | linear warmup over first 5 percent of configured train steps |
| LR schedule | cosine decay to `5e-6`, no restarts |
| Beta warmup | first epoch for epoch-based runs; first 10 percent of optimizer steps for step-limited debug runs; no cyclic restarts |
| `logvar` clamp | clamp to `[-8.0, 4.0]` before sampling and KL |

Optimizer parameter groups:

- learned convolution kernels, VAE head weights, final RGB head weights, and
  future `SO(2)` kernel expansion coefficients use the base learning rate and
  configured weight decay;
- additive biases, GroupNorm/field-norm affine parameters, and activation gate
  parameters use `weight_decay = 0.0`;
- activation gate parameters `a_i` and `b_i` use `lr_multiplier = 0.5` for the
  first run;
- every trainable parameter must appear in exactly one optimizer group, and
  tests must fail on duplicates, omissions, or a gate parameter placed in a
  decayed group;
- the first implementation should expose stable semantic group names:
  `decay`, `no_decay`, and `gate_no_decay`. `gate_no_decay` uses
  `lr = base_lr * 0.5` and `weight_decay = 0.0`;
- train-step evidence must record optimizer name, base learning rate, weight
  decay, group count, parameter coverage status, and whether every gate
  parameter is in the gate-specific no-decay group;
- this replaces the historical FSQ shape-only grouping with a semantic grouping:
  future `SO(2)` expansion coefficients may be stored as 1D tensors but still
  count as learned kernel weights and should receive weight decay unless a later
  spec overrides it;
- log gate parameter min/max and gate saturation summaries for scalar and radial
  gates.

Gate-health benchmark before the first full run:

- treat learned gate parameters as monitored capacity, not as an activation
  ablation in spec 0001;
- during the short real-data Kaggle debug/benchmark path, log gate behavior at
  fixed intervals for every gated activation module;
- use the stable module id from `model.named_modules()` with any DDP `module.`
  prefix stripped. The expected module count for the locked Conv2d baseline is
  34 gated activation modules, and every expected module must appear at every
  logged interval for the summary to pass;
- log at optimizer step 0, every 25 successful optimizer updates, and at the
  final successful optimizer update for selected-runtime debug and tiny-overfit
  runs. Runtime benchmark rows with fewer than 25 measured updates must log at
  least the first and last measured successful optimizer update;
- capture gate inputs/outputs in FP32 summary buffers during the forward pass,
  gate parameter values before `optimizer.step()`, unscaled gate gradient norms
  after `loss.backward()` and before the optimizer update, and
  `*_update_to_param_norm` from the actual parameter delta after
  `optimizer.step()`;
- write `metrics/gate_health.csv` with at least
  `run_name,optimizer_step,module,gate_kind,num_channels,num_elements,a_min,a_max,a_mean,a_std,b_min,b_max,b_mean,b_std,max_abs_a,max_abs_b,gate_mean,gate_std,gate_p01,gate_p50,gate_p99,frac_gate_lt_0_01,frac_gate_gt_0_99,worst_channel_frac_gate_lt_0_01,worst_channel_frac_gate_gt_0_99,dead_channel_count,input_rms,output_rms,output_input_rms_ratio,a_grad_norm,b_grad_norm,a_update_to_param_norm,b_update_to_param_norm,gate_health_status`;
- write `benchmark/gate_health_summary.json` with per-module worst-case
  saturation, non-finite counts, largest absolute `a`/`b`, dead-channel counts,
  zero-gradient counts, final input/output RMS ratio, and an overall
  `pass|warn|fail` status;
- compute `*_update_to_param_norm` as
  `update_norm / max(parameter_norm, 1e-8)` so zero-initialized or near-zero
  parameters have a defined denominator;
- a channel counts as dead only when it has nontrivial input
  (`input_rms >= 1e-4`) and either per-channel `output_rms < 1e-6` or
  per-channel `output_input_rms_ratio < 1e-3` for three consecutive logged
  intervals. `dead_channel_count` is the number of unique module channels that
  meet that condition;
- summary fields are worst-case reductions over all required modules, ranks,
  and logged intervals. Missing modules, missing ranks, missing intervals, or
  malformed gate rows make the summary `fail`;
- gate-health warning thresholds: `max_abs_a > 10`, `max_abs_b > 10`,
  `max(frac_gate_lt_0_01, frac_gate_gt_0_99) >= 0.80`,
  `dead_channel_count > 0`, `output_input_rms_ratio < 1e-2`, or zero
  `a_grad_norm + b_grad_norm` for three consecutive logged intervals;
- gate-health failure thresholds: any non-finite gate value, parameter, input,
  output, gradient, or update; `max_abs_a > 20` or `max_abs_b > 20`;
  `max(frac_gate_lt_0_01, frac_gate_gt_0_99) >= 0.95` for three consecutive
  logged intervals; `dead_channel_count > max(1, 0.10 * num_channels)`;
  `output_input_rms_ratio < 1e-3` for three consecutive logged intervals in a
  hidden block; or any gate-health status explicitly marked `fail`;
- do not start the first full training run unless
  `benchmark/gate_health_summary.json` has overall status `pass`. A `warn`
  status requires inspection and a spec/config update before full training;
- do not use the gate-health benchmark to choose among many nonlinearities
  unless a later spec explicitly opens an activation ablation.

Runtime benchmark requirement before the first full Kaggle run:

- the benchmark is a short decision run, not training; it must stop after fixed
  warmup/measured steps and must not tune model quality;
- use the real train and validation data loaders and the real training step;
- every benchmark artifact must carry enough state to prevent accidental
  promotion from a local schema smoke to a real Kaggle runtime decision;
- valid artifact status values are:
  - `schema_pass`: local synthetic schema smoke only; never full-run eligible;
  - `local_pass`: measured local CPU/laptop pre-test on synthetic or otherwise
    explicitly local data; never full-run eligible and never sufficient for
    runtime selection;
  - `smoke_pass`: narrow permission-gated debug smoke on a declared remote or
    setup-only path; never full-run eligible, never runtime selection evidence,
    and valid only when the artifact records its non-promotable scope, source,
    caps, data origin, integrity status, payload provenance, and linked smoke
    assertions;
  - `synthetic_timing_pass`: permission-gated Kaggle no-dataset synthetic
    binary timing pretest evidence; never full-run eligible, never runtime
    selection evidence, and valid only when the artifact records
    `status_scope = "non_promotable_synthetic_timing"`, empty Kaggle source
    lists, generated UBC-format data provenance, accelerator proof,
    fit-versus-throughput row eligibility, and explicit blocked real-data
    claims;
  - `pass`: verified on the required real runtime/data path and eligible for the
    next gate if all linked artifacts also pass;
  - `warn`: completed, but blocked from automatic promotion until inspected and
    recorded in a spec/config update;
  - `skipped_unsupported`: a configured optional local or runtime smoke was not
    supported by the observed environment, and the artifact records a
    deterministic `failure_kind`; this is never full-run eligible and cannot be
    selected as runtime evidence;
  - `fail`: completed or aborted with a failure; never eligible;
- runtime-matrix row statuses are a separate CSV row-status vocabulary scoped to
  `benchmark/runtime_matrix.csv`. Row-only statuses such as `ineligible`,
  `oom`, `compile_fail`, `ddp_fail`, `wrong_accelerator`, `nonfinite_fail`,
  `numerical_fail`, `dataloader_fail`, `gate_health_fail`,
  `amp_skipped_fail`, and `runtime_error` must not be reused as top-level JSON
  artifact statuses unless a later spec explicitly says so;
- benchmark-readiness labels used in `CURRENT.md`, the spec index, and handoff
  notes are:
  - `scaffold_schema_ready`: only local import/CLI/schema contract checks have
    passed; artifacts may use `schema_pass` and must keep
    `full_run_eligible = false`;
  - `local_benchmark_pretest_contract_ready`: the local CPU/laptop benchmark
    pre-test status, candidate matrix, and schemas are locked, but measured
    pre-test implementation has not necessarily run yet;
  - `local_benchmark_pretest_ready`: local CPU/laptop benchmark pre-tests have
    run real local code on synthetic UBC-format shards and may use
    `local_pass`, but all artifacts must keep `full_run_eligible = false` and
    must not be consumed as selected runtime evidence;
  - `model_loss_train_step_contract_ready`: the VAE forward API, explicit
    latent-noise control, clamped-logvar sampling/KL semantics, exact loss
    reductions, beta schedule, local identity-input train-step rule, semantic
    AdamW groups, and non-promotable local train-step artifact schema are
    locked, but the implementation may not have run yet;
  - `model_loss_train_step_ready`: the local model/loss train-step pre-test has
    run real local code on synthetic tensors, written
    `benchmark/model_loss_train_step.json` with `status = "local_pass"` and
    `full_run_eligible = false`, and passed focused plus full production-scope
    quality checks;
  - `corruption_contract_ready`: the scikit-compatible PyTorch HED convention,
    conservative/default and FSQ-wide profiles, residual-axis wording, semantic
    stateless RNG, branchless-all first execution strategy, metadata contract,
    and non-promotable QA artifact schema are locked, but no corruption code has
    been accepted yet;
  - `corruption_ready`: the local corruption correctness/QA implementation has
    run real local code, written `benchmark/stain_corruptor_qa.json` with
    `status = "local_pass"` and `full_run_eligible = false`, passed focused and
    full production-scope quality checks, and remains blocked from Kaggle use
    until fixed real 25-patch visual QA is generated;
  - `kaggle_smoke_ready`: the capped real-data smoke launcher has run locally on
    synthetic UBC-format shards, the Kaggle debug config caps execution at three
    train steps and one clean-validation batch, the script kernel payload can be
    built locally, and remote push remains permission-gated and non-promotable;
  - `kaggle_setup_smoke_ready`: the no-dataset setup-smoke kernel has a
    generated single-file `run.py` with an embedded payload, empty Kaggle source
    attachments, setup-specific push guards, an upload-simulation test proving
    it works without a sibling payload directory, and a non-promotable
    `benchmark/kaggle_setup_smoke.json` artifact contract;
  - `kaggle_synthetic_timing_contract_ready`: the no-dataset synthetic binary
    Kaggle timing pretest contract is locked for implementation. It may screen
    and order candidate rows for the later real-data benchmark, but it cannot
    write `benchmark/selected_runtime.json`, cannot mark
    `full_run_eligible = true`, and cannot unlock selected-runtime debug,
    tiny-overfit, full training, or paper claims;
  - `benchmark_cli_implementation_ready`: local benchmark CLIs instantiate real
    code, `benchmark/model_count.json` has `status = "pass"` from an
    instantiated model, the `data_metrics_ready`, selector/dataloader,
    local-pretest, model/loss train-step, and corruption slices are locked and
    verified, dataloader, corruption-check, numerical-check, and gate-health
    artifacts are produced by real local code, strict production Python quality
    passes, no active `src/eqvae` code imports historical `src.nn`, and no
    undeclared `pytorch-msssim` dependency remains;
  - `runtime_selected`: the permission-gated real Kaggle benchmark has produced
    `benchmark/selected_runtime.json` with `status = "pass"` and
    `full_run_eligible = true`;
- schema-only local synthetic benchmark artifacts must set `benchmark_kind =
  "local_synthetic_schema"`, `benchmark_source =
  "local_synthetic_schema_smoke"`, `status = "schema_pass"`,
  `full_run_eligible = false`, `accelerator_mode = "local_cpu"`, and
  `machine_shape = "local_cpu"`;
- measured local CPU synthetic pre-test artifacts must set `benchmark_kind =
  "local_synthetic_pretest"`, `benchmark_source =
  "local_cpu_synthetic_pretest"`, `status = "local_pass"`,
  `full_run_eligible = false`,
  `accelerator_mode = "local_cpu"`, and `machine_shape = "local_cpu"`.
  Training, selected-runtime debug, tiny-overfit, and full-run CLIs must reject
  any `--runtime-config` whose `full_run_eligible` is not `true`;
- Kaggle synthetic timing pretest artifacts must use separate names from the
  real runtime-selection artifacts:
  `benchmark/synthetic_timing_manifest.json`,
  `benchmark/synthetic_timing_runtime_proof.json`,
  `benchmark/synthetic_timing_matrix.csv`, and
  `benchmark/synthetic_timing_recommendations.json`. They must set
  `benchmark_kind = "kaggle_synthetic_timing_pretest"`,
  `benchmark_source = "kaggle_no_dataset_generated_ubc_shards"`,
  `status = "synthetic_timing_pass"` when successful,
  `status_scope = "non_promotable_synthetic_timing"`,
  `full_run_eligible = false`, `dataset_sources = []`,
  `competition_sources = []`, `kernel_sources = []`, `model_sources = []`,
  `data.origin = "/kaggle/working_generated_synthetic"`, and
  `generation_excluded_from_timing = true`. They must include
  `blocked_claims` with at least these true-valued keys:
  `final_batch_size`, `final_precision_policy`, `final_corruption_strategy`,
  `final_dataloader_settings`, `final_single_vs_dual_t4`,
  `real_data_loader_throughput`, `convergence`, `paper_evidence`, and
  `full_run_readiness`. They must never overwrite or masquerade as
  `benchmark/runtime_proof.json`, `benchmark/runtime_matrix.csv`,
  `benchmark/dataloader_matrix.csv`, `benchmark/numerical_checks.csv`, or
  `benchmark/selected_runtime.json`;
- the synthetic timing manifest must record at least: profile name, seed,
  train/validation patch counts, exact byte counts, binary paths, CSV paths,
  file SHA256 hashes, CRC32 values, generation seconds, write throughput,
  free disk before/after, cache state, data generator version,
  `crc_validated = true` from a pre-timing `validate_crc = true` integrity pass
  for both splits, parsed header fields, row counts, file sizes, per-split
  semantic-key uniqueness checks for `{split}:{wsi_id}:{label}:{x}:{y}`,
  first/last `row_index`, `file_index`, `sample_id`, semantic-key hashes from
  `PatchTrainingDataset` batches, and whether any row reused samples or
  exceeded the non-wrapping split budget;
- the default synthetic pixel profile is `histology_like_rgb_v1`, a
  deterministic non-real H&E-like RGB mixture suitable for exercising HED/OD
  corruption and image-domain losses better than uniform noise. It remains
  synthetic evidence only. A `uniform_rgb_v1` control profile may be added
  later, but it must not replace the default profile unless this spec is
  updated;
- synthetic timing recommendation rows may only classify candidates as
  `carry_to_real_benchmark`, `prune_obvious_failure`, `fit_probe_only`, or
  `needs_real_data_confirmation`. They must not classify any row as selected
  for the real run. `prune_obvious_failure` is allowed only for structural
  failures: wrong accelerator, OOM, DDP failure, compile failure, non-finite
  loss or gradients, AMP skipped steps, invalid or missing artifact proof, or
  source/loader format violations. Performance-only differences must be
  `carry_to_real_benchmark` or `needs_real_data_confirmation` unless a later
  spec adds a conservative dominance rule;
- benchmark two accelerator modes:
  `single_visible_t4` and `dual_t4_ddp`;
- `single_visible_t4` may run inside the dual-T4 Kaggle machine by setting
  visible devices to one GPU and `world_size = 1`;
- `dual_t4_ddp` must launch with two ranks, restore the historical
  `torchrun --standalone --nproc_per_node=2` behavior or an equivalent
  self-spawn implementation, and record `world_size = 2`;
- synthetic timing rows must run each accelerator mode and row in a fresh
  Python child process with `CUDA_VISIBLE_DEVICES` set before importing `torch`.
  Record launch command, environment mask, row order, subprocess exit status,
  and per-rank device assignment;
- the synthetic timing pretest must attempt both `single_visible_t4` and
  `dual_t4_ddp`. If Kaggle exposes fewer than two visible T4 devices, the
  synthetic proof must still be written and dual rows must be marked
  `wrong_accelerator` or `skipped_unsupported`; silently dropping dual rows is
  not allowed;
- the Kaggle kernel metadata must request `machine_shape = "NvidiaTeslaT4"`
  before the remote benchmark is pushed. This value was verified on 2026-06-11
  by pulling metadata for the existing `maximusshtefan/non-eq-vae` notebook that
  the Kaggle UI showed as GPU T4 x2. Because the metadata value does not encode
  visible device count, `dual_t4_ddp` rows must still verify
  `cuda_device_count == 2`, two T4 names, `world_size == 2`, and
  `nproc_per_node == 2` at runtime;
- before any runtime row is eligible, write `benchmark/runtime_proof.json`.
  This file must prove the Kaggle metadata, launch mode, visible devices,
  per-rank device names, dataset slugs, launcher command hash, and Kaggle CLI
  version used for the benchmark. If Kaggle provides the wrong accelerator, the
  proof file still must be written with `status = "fail"` and
  `failure_kind = "wrong_accelerator"`;
- for each GPU configuration, benchmark `torch.compile` off/on where the runtime
  supports it;
- within the AMP/precision axis, compare the named precision policies
  `amp_off_fp32`, `amp_conservative`, and `amp_scalar_gate_relaxed`;
- compare the corruption execution strategies `branchless_all` and
  `indexed_masked`; keep the branchless path unless masked indexing is compile
  stable, preserves RNG semantics, and is measurably faster;
- for each row, record warm steady-state samples/sec, step time, compile
  overhead, max VRAM, largest stable per-device batch, global batch,
  `amp_step_skipped` count, gate-health warning count, and any compile/DDP
  failure;
- batch size is selected from VRAM and throughput evidence for each runtime
  configuration, not hard-coded from the historical FSQ run.
- single-vs-dual comparisons must use feasible global throughput and projected
  real train-epoch time, not equal per-device batch size alone. The projection
  must record `real_train_patch_count = 300000`,
  `global_batch_size = per_device_batch_size * world_size`,
  `drop_last = false`,
  `steps_per_epoch = ceil(real_train_patch_count / global_batch_size)`,
  `effective_samples_per_epoch = real_train_patch_count`,
  `remainder_samples = real_train_patch_count % global_batch_size`,
  steady-state step-time statistics, and `estimated_epoch_minutes =
  steps_per_epoch * steady_step_ms_p50 / 60000`. Compile/startup time is
  excluded from this steady-state projection and recorded separately. This
  projection is a synthetic shortlist metric until repeated on real
  train/validation shards;
- row IDs must be stable and machine-readable:
  `{accelerator_mode}__bs{per_device_batch_size}__{precision_policy}__compile_{on|off}__{corruption_strategy}`.
  Repeated measurements of the same row may add `__repeat{n}` but must still
  point to one canonical row when selecting.
- runtime matrix coverage is one row per attempted
  `{accelerator_mode, per_device_batch_size, precision_policy,
  torch_compile_enabled, corruption_strategy}` combination. If a row is invalid
  by the valid-row table, do not emit it. If a row is valid but unsupported by
  the observed runtime, emit it with `status = "skipped_unsupported"` and a
  `failure_kind` explaining the unsupported feature.

Valid runtime matrix rows:

| AMP enabled | Precision policy | Compile | Corruption strategy |
| --- | --- | --- | --- |
| false | `amp_off_fp32` | false/true | `branchless_all` / `indexed_masked` |
| true | `amp_conservative` | false/true | `branchless_all` / `indexed_masked` |
| true | `amp_scalar_gate_relaxed` | false/true | `branchless_all` / `indexed_masked` |

Invalid rows must not be emitted, for example `amp_off_fp32` with AMP enabled or
`amp_conservative` with AMP disabled.

Allowed `benchmark/runtime_matrix.csv` row statuses:

- `schema_pass`: local synthetic schema-smoke row only; never eligible and must
  keep `benchmark_source = "local_synthetic_schema_smoke"`;
- `pass`: row ran to completion and is eligible if all linked safety artifacts
  pass;
- `ineligible`: row completed but failed a selection rule, such as insufficient
  VRAM headroom, no speedup for `indexed_masked`, or warning/fail status in a
  linked artifact;
- `skipped_unsupported`: row was configured and valid by the matrix table but
  is unsupported by the observed runtime, for example CPU-only compile or dtype
  smoke limitations in local pre-tests;
- `oom`: row hit an out-of-memory condition and must include memory telemetry if
  available;
- `compile_fail`, `ddp_fail`, `wrong_accelerator`, `nonfinite_fail`,
  `numerical_fail`, `dataloader_fail`, `gate_health_fail`,
  `amp_skipped_fail`, or `runtime_error`: row is not eligible and must record a
  `failure_kind` matching the status class plus a deterministic
  `failure_message_hash`.

The implementation must not silently drop attempted rows. Every configured row
or stopped batch-size search attempt must produce a row with a status and
failure metadata.

Benchmark budget and reset rules:

- default candidate per-device batch sizes:
  `[4, 8, 12, 16, 24, 32, 48, 64]`;
- synthetic timing pretest rows using the 2 GiB-scale default profile may attempt all
  default candidate batch sizes, but only rows satisfying
  `global_batch_size * non_wrapping_eligibility_steps <= split_patch_count` for
  both train and validation are ranked throughput rows. The default
  `non_wrapping_eligibility_steps` is 30, independent of the initial 3-warmup /
  12-measured fit pass. Other rows are fit/VRAM probes only and must not rank
  throughput recommendations. Synthetic timing rows must record
  `non_wrapping_eligibility_steps`, `non_wrapping_eligible`,
  `sample_reuse_count`, and `fit_probe_only`. Under the default profile, global
  batch 128 is the largest
  non-wrapping ranked-throughput candidate, while larger global batches are
  probe-only unless a later larger synthetic profile is explicitly accepted;
- each row starts from identical model weights, optimizer state, scaler state,
  beta/LR scheduler state, data order, and RNG seeds;
- each row uses `warmup_steps = 3`, `measured_steps = 12`, and `repeats = 1`;
  after a row is explicitly shortlisted for operational carry-forward, rerun it with
  `warmup_steps = 5`, `measured_steps = 25`, and `repeats = 1`;
- if `torch.compile` needs compilation, report compile/startup time separately
  from steady-state step time;
- OOM rows are valid failure rows: record the attempted per-device batch size,
  the exception class/message hash, max allocated/reserved memory if available,
  and continue with the next smaller candidate;
- selected rows must leave at least 10 percent VRAM headroom after warmup and
  measured steps;
- any row with non-finite loss, non-finite gradients, an AMP skipped step,
  DDP failure, compile failure, repeated graph breaks/recompiles after warmup,
  or gate-health status `fail` is ineligible;
- `indexed_masked` must improve measured steady-state samples/sec by at least
  5 percent over `branchless_all` in the same accelerator/precision/compile
  setting, or else `branchless_all` remains selected;
- a faster AMP/compile/precision row must pass paired numerical checks against
  `amp_off_fp32` eager on the same fixed batches before it is eligible.
- real-data runtime selection filters to eligible real-data rows, then sorts by:
  1. highest `samples_sec`;
  2. lower `steady_step_ms_p95`;
  3. higher `vram_headroom_fraction`;
  4. simpler execution policy in this order:
     `amp_off_fp32` before AMP when throughput differs by less than 3 percent,
     `torch_compile=false` before `true` when throughput differs by less than
     3 percent, and `branchless_all` before `indexed_masked` unless
     `indexed_masked` clears its 5 percent speedup rule;
  5. `single_visible_t4` before `dual_t4_ddp` when global throughput differs by
     less than 5 percent, because the extra DDP complexity is not justified.
  Synthetic timing recommendations instead sort ranked-throughput rows by
  lower `estimated_epoch_minutes`, then by lower `steady_step_ms_p95`, then by
  higher VRAM headroom, and may only choose carry-forward/prune/probe labels.
  Synthetic timing must not select a runtime.

Paired numerical checks:

- for every candidate that differs from
  `amp_off_fp32 + eager + branchless_all`, run three fixed-seed benchmark
  batches against that reference with identical model initialization, data
  order, corruption decisions, and latent noise. This includes compile-only FP32
  rows and corruption-strategy-only rows;
- log absolute and relative deltas for total loss, reconstruction loss, L1,
  SSIM loss, KL, gradient norm, output range, `mu` stats, `logvar` stats,
  `logvar_clamp_count`, gate-health summary, and parameter-update norm;
- compute relative deltas as
  `abs(candidate - reference) / max(abs(reference), 1e-8)` unless a metric has
  a more specific denominator in this spec;
- corruption strategy checks must also log Bernoulli corruption decisions,
  HED/noise parameters for corrupted samples, unchanged clean-sample checks, and
  clean-validation RNG non-consumption;
- default pass thresholds are: no non-finite values, no AMP skipped step,
  absolute loss/reconstruction/L1/SSIM-loss delta `<= 1e-3` or relative delta
  `<= 5e-3`, KL relative delta `<= 1e-2`, gradient-norm relative delta
  `<= 0.05`, parameter-update-norm relative delta `<= 0.05`,
  output-range absolute deltas `<= 1e-3`, `mu/logvar` mean/std absolute deltas
  `<= 1e-3`, and `logvar_clamp_count_delta == 0`;
- if a threshold is too strict for a future verified reason, update this spec
  before selecting that runtime.

Dataloader benchmark requirement:

- the dataloader benchmark starts from the historical FSQ idea that is still
  useful here: read UBC binary patch shards through a worker-local read-only
  mmap and keep the hot path tensor-only. Selector/sample provenance stays on
  the separate fixed-selector rail;
- before selecting a runtime, write `benchmark/dataloader_matrix.csv` for train
  and validation shards on real Kaggle data;
- the no-dataset Kaggle synthetic timing pretest may measure the same mmap and
  H2D code paths on generated `/kaggle/working` shards, but these measurements
  are not real dataloader evidence. They must be labeled as post-generation
  hot-cache-biased synthetic plumbing rows and cannot select `num_workers`,
  `prefetch_factor`, `pin_memory`, `persistent_workers`, or
  `non_blocking_h2d` for the real run without confirmation on real Kaggle input
  shards;
- synthetic timing train-step rows must prove they use
  `DataLoader(PatchTrainingDataset, collate_fn=collate_patch_training_samples)`
  and `normalize_uint8_batch`, with pre/post dtype and range checks. Tensor-only
  loader rows must prove they use `PatchTensorDataset`;
- before remote Kaggle execution, a local CPU/laptop pre-test may write the same
  matrix schema on tiny synthetic UBC-format shards to validate the benchmark
  mechanics and candidate expansion. These rows use `status = "local_pass"`
  when measured successfully, `accelerator_mode = "local_cpu"`,
  `machine_shape = "local_cpu"`, `benchmark_kind =
  "local_synthetic_pretest"`, `benchmark_source =
  "local_cpu_synthetic_pretest"`, and `full_run_eligible = false`. They must
  never be selected as a runtime and must not satisfy Kaggle artifact
  dependencies;
- benchmark loader settings that are runtime-relevant, at minimum
  `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`, and
  `non_blocking_h2d`;
- default local CPU pre-test loader candidates are:
  - `num_workers`: `[0, 1]`;
  - `prefetch_factor`: `null` when `num_workers = 0`, otherwise `2`;
  - `pin_memory`: `false`;
  - `persistent_workers`: `false` when `num_workers = 0`, otherwise
    `[false, true]`;
  - `non_blocking_h2d`: `false`;
  - `warmup_batches = 1` and `measured_batches = 3`;
- default Kaggle loader candidates are:
  - `num_workers`: `[1, 2, 4]`;
  - `prefetch_factor`: `[2, 4]` when `num_workers > 0`;
  - `pin_memory`: `[true, false]`;
  - `persistent_workers`: `[true, false]` when `num_workers > 0`;
  - `non_blocking_h2d`: `[true, false]`;
  - first pass uses at least `warmup_batches = 5` and
    `measured_batches = 25`;
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` must declare
  those candidates under `dataloader.candidates` and must identify the hot path
  as `mmap_tensor_only_v1` plus selector provenance as
  `fixed_selector_json_v1`;
- record at least
  `run_name,benchmark_kind,benchmark_source,full_run_eligible,accelerator_mode,machine_shape,world_size,rank,split,num_workers,prefetch_factor,pin_memory,persistent_workers,non_blocking_h2d,batch_size,batches_measured,batch_fetch_ms_p50,batch_fetch_ms_p95,h2d_ms_p50,h2d_ms_p95,loader_samples_sec,trainer_samples_sec,data_wait_fraction_p50,data_wait_fraction_p95,rank_sample_count,dropped_sample_count,status,failure_kind`;
- local CPU pre-test rows must leave `h2d_ms_p50` and `h2d_ms_p95` empty
  because there is no host-to-device transfer to benchmark. Real Kaggle GPU
  rows must time host-to-device transfer with the declared `pin_memory` and
  `non_blocking_h2d` settings;
- local CPU pre-test rows must also leave `trainer_samples_sec`,
  `data_wait_fraction_p50`, and `data_wait_fraction_p95` empty because they do
  not execute the training step;
- local CPU pre-test candidate failures must still produce rows with
  `status = "fail"` and a deterministic `failure_kind`; worker-positive
  failures on constrained local environments must not hang or disappear;
- a runtime is ineligible if validation loading is unmeasured, any rank fails,
  rank sample counts differ beyond one batch, `data_wait_fraction_p95 > 0.20`,
  or loader throughput is below `1.25 * trainer_samples_sec` for the selected
  training row.
- each loader row must measure at least 5 warmup batches followed by 25 measured
  batches per split unless the split is intentionally smaller in a local
  synthetic test. For DDP, report both per-rank rows and global aggregate rows;
  selection uses the global aggregate while failure checks inspect every rank.
- the selected runtime must copy the selected loader settings into
  `selected_runtime.json` under `dataloader` and into the resolved full-run
  config, so Kaggle debug/tiny/full runs do not quietly use a different loader.
  The copied settings include `non_blocking_h2d`.

The selected baseline runtime must be recorded in the resolved config. Use
`per_device_batch_size`, `global_batch_size`, `mixed_precision.enabled`, and
`torch_compile.enabled`, plus explicit `precision.policy` and
`corruption.strategy`; do not leave the batch-size, precision, or corruption
execution meaning ambiguous.

Benchmark artifact dependency graph:

1. `benchmark/model_count.json` must pass before runtime rows are eligible.
2. `benchmark/runtime_proof.json` must pass before any Kaggle runtime row is
   eligible.
3. Synthetic timing artifacts may recommend candidate rows to carry into the
   real-data benchmark, but they cannot satisfy `runtime_proof`,
   `runtime_matrix`, `dataloader_matrix`, `numerical_checks`, gate-health, or
   selected-runtime dependencies.
4. `benchmark/runtime_matrix.csv` can mark candidate rows as completed, but no
   row may be selected until matching `benchmark/dataloader_matrix.csv`,
   `benchmark/numerical_checks.csv`, `metrics/gate_health.csv`, and
   `benchmark/gate_health_summary.json` entries exist.
5. `benchmark/selected_runtime.json` may be written with `status = "pass"` only
   when it references one row from `runtime_matrix.csv` whose row status is
   `pass`, whose linked artifacts have `pass`, and whose accelerator proof
   matches the selected mode.
6. selected-runtime debug must consume that selected runtime and write a resume
   proof before tiny-overfit may run.
7. `benchmark/tiny_overfit_summary.json` must pass before
   `non_eq_vae_baseline.json` can be used for a first 10-epoch run.

`benchmark/model_count.json` required shape:

```json
{
  "status": "pass",
  "benchmark_kind": "implementation_model_count",
  "benchmark_source": "instantiated_model",
  "full_run_eligible": true,
  "config": "invoked config path",
  "config_resolution": "source_config_deep_merge_v1",
  "source_config_chain": [
    {
      "path": "configs/spec0001/non_eq_vae_model_base.json",
      "sha256": "sha256 hex of raw source config"
    }
  ],
  "invoked_config_hash": "sha256 hex of raw invoked config",
  "effective_config_hash": "sha256 hex of resolved effective config",
  "model_config_hash": "same value as effective_config_hash",
  "model_config_hash_source": "canonical_json_sorted_compact_effective_config",
  "count_source": "instantiated_model",
  "input_shape": [1, 3, 256, 256],
  "learned_convolution_count": 43,
  "normalization_module_count": 40,
  "gate_module_count": 34,
  "fixed_resampling_op_count": 12,
  "learned_convolution_parameters": 3949539,
  "groupnorm_affine_parameters": 4800,
  "learned_gate_parameters": 4096,
  "total_learned_parameters": 3958435,
  "learned_convolution_macs_per_sample": 36471046144,
  "fixed_resampling_macs_per_sample": 85032960,
  "total_macs_per_sample_with_fixed_resampling": 36556079104,
  "activation_output_elements_per_sample": 36110336,
  "implementation": {
    "model_factory": "eqvae.models.non_equivariant_vae",
    "instantiated_model": true,
    "uses_meta_device_or_real_cpu": "cpu",
    "zero_initialized_rgb_head_verified": true,
    "banned_operations_checked": true,
    "inventory_matches_expected": true,
    "forward_order_verified": true,
    "shape_source": "meta_forward_hooks"
  },
  "inventory_mismatch_count": 0,
  "inventory_mismatches": [],
  "expected": {
    "total_learned_parameters": 3958435,
    "learned_convolution_macs_per_sample": 36471046144,
    "fixed_resampling_macs_per_sample": 85032960
  },
  "observed": {
    "total_learned_parameters": 3958435,
    "learned_convolution_macs_per_sample": 36471046144,
    "fixed_resampling_macs_per_sample": 85032960
  },
  "resampling_macs": {
    "actual_implementation": 85032960,
    "conservative_dense_grouped_5x5_equivalent": 85032960
  },
  "module_inventory_path": "benchmark/model_inventory.csv",
  "tolerances": {
    "parameters_abs": 0,
    "macs_abs": 0,
    "activation_output_elements_abs": 0
  },
  "matches_spec_target": true
}
```

For `benchmark/model_count.json`, `full_run_eligible = true` means the artifact
is eligible only as a model-count dependency in the benchmark artifact graph. It
does not make any runtime config, selected runtime, training command, Kaggle
remote push, or paper claim eligible by itself.

The local scaffold-only form may instead use `benchmark_kind =
"local_synthetic_schema"`, `benchmark_source =
"local_synthetic_schema_smoke"`, `full_run_eligible = false`, and
`status = "schema_pass"`, but it must not satisfy Kaggle runtime or full-run
acceptance. Implementation-ready output must set `count_source =
"instantiated_model"` and `implementation.instantiated_model = true`.

Config hashing uses parsed JSON re-emitted as canonical sorted compact bytes
with separators `,`, `:`, and UTF-8 encoding. `invoked_config_hash` hashes the
raw invoked config file. `effective_config_hash` hashes the recursively resolved
config after `source_config` overlays are deep-merged, with `source_config`
treated as provenance rather than an effective runtime field.
`model_config_hash` is an alias of `effective_config_hash`, so a model-count
artifact is tied to the actual resolved run contract while still preserving the
invoked file hash. Pass-mode model-count configs must resolve to a model object
declaring `architecture_id = "spec0001_non_eq_vae_translatable"` and
`topology_version = "spec0001.count.v1"`. The model-count CLI must reject
configs whose resolved `model.implementation_status` remains
`"count_schema_only"` when writing pass-mode instantiated counts.

For `resampling_macs`, `actual_implementation` is the measured/countable
implementation formula and must never be `0` in pass mode. The first pass-mode
implementation uses dense grouped 5x5 downsampling, so actual downsample MACs
equal the conservative equivalent:
`channels * 25 * output_h * output_w`. Bilinear upsample MACs use a 4-tap
formula: `channels * 4 * output_h * output_w`. If a future separable
implementation is used, `actual_implementation` must report the separable
two-pass formula and
`conservative_dense_grouped_5x5_equivalent` must still report the dense 5x5
equivalent for comparable accounting.

`benchmark/model_inventory.csv` required columns:

```text
module_id,module_type,parent_path,stage,block,branch,op_index,observed_forward_index,input_shape,output_shape,kernel_size,stride,padding,groups,taps,trainable,learned_parameter_count,macs_per_sample,activation_output_elements,in_channels,out_channels,has_bias,followed_by_norm,gate_channels,resampling_kind,count_category,mac_formula
```

The inventory must include every learned convolution, GroupNorm, gate module,
fixed binomial downsample, and bilinear upsample that contributes to the count.
Rows are ordered by the canonical architecture order in this spec and must match
the observed forward-hook execution order: stem, encoder blocks `enc0` to
`enc7`, heads, latent projection, decoder blocks `dec0` to `dec7`, and RGB
output head. `activation_output_elements` contributes to
`activation_output_elements_per_sample` only for rows with
`count_category = "learned_convolution"`; norm, gate, and fixed-resampling rows
must set it to `0`.

`benchmark/runtime_matrix.csv` required columns:

```text
run_name,benchmark_kind,benchmark_source,full_run_eligible,row_id,accelerator_mode,machine_shape,visible_device_count,cuda_device_count,gpu_names,ddp_backend,world_size,nproc_per_node,precision_policy,amp_enabled,torch_compile_enabled,corruption_strategy,per_device_batch_size,global_batch_size,gradient_accumulation_steps,warmup_steps,measured_steps,repeats,compile_startup_sec,steady_step_ms_p50,steady_step_ms_p95,samples_sec,trainer_samples_sec,max_vram_allocated_mb,max_vram_reserved_mb,vram_headroom_fraction,amp_step_skipped_count,gate_health_status,gate_health_warning_count,numerical_check_status,data_wait_fraction_p95,oom,status,failure_kind,failure_message_hash
```

`gpu_names` is encoded as a compact JSON array string. `amp_enabled`,
`torch_compile_enabled`, and `oom` are lowercase `true|false` strings in CSV.
Timing fields use milliseconds except `compile_startup_sec`.

`benchmark/selected_runtime.json` required shape:

```json
{
  "status": "pass",
  "benchmark_kind": "kaggle_runtime_selection",
  "benchmark_source": "kaggle_runtime_benchmark",
  "full_run_eligible": true,
  "selected_row_id": "string",
  "accelerator_mode": "single_visible_t4 or dual_t4_ddp",
  "machine_shape": "NvidiaTeslaT4",
  "world_size": 2,
  "nproc_per_node": 2,
  "gpu_names": ["..."],
  "per_device_batch_size": 0,
  "global_batch_size": 0,
  "gradient_accumulation_steps": 1,
  "optimizer_updates_per_epoch": 0,
  "lr_warmup_steps": 0,
  "beta_warmup_steps": 0,
  "mixed_precision": {"enabled": false, "policy": "amp_off_fp32"},
  "torch_compile": {"enabled": false, "backend": "eager-or-inductor"},
  "corruption": {"strategy": "branchless_all"},
  "dataloader": {
    "num_workers": 1,
    "prefetch_factor": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "non_blocking_h2d": true
  },
  "throughput": {
    "samples_sec": 0.0,
    "steady_step_ms_p50": 0.0,
    "compile_startup_sec": 0.0,
    "estimated_10_epoch_wall_time_sec": 0.0
  },
  "safety": {
    "numerical_check_status": "pass",
    "gate_health_status": "pass",
    "dataloader_status": "pass",
    "amp_step_skipped_count": 0
  },
  "artifacts": {
    "runtime_matrix": "benchmark/runtime_matrix.csv",
    "runtime_matrix_sha256": "sha256 hex",
    "model_count": "benchmark/model_count.json",
    "model_count_sha256": "sha256 hex",
    "runtime_proof": "benchmark/runtime_proof.json",
    "runtime_proof_sha256": "sha256 hex",
    "dataloader_matrix": "benchmark/dataloader_matrix.csv",
    "dataloader_matrix_sha256": "sha256 hex",
    "numerical_checks": "benchmark/numerical_checks.csv",
    "numerical_checks_sha256": "sha256 hex",
    "corruption_checks": "benchmark/corruption_checks.csv",
    "corruption_checks_sha256": "sha256 hex",
    "stain_corruptor_qa": "benchmark/stain_corruptor_qa.json",
    "stain_corruptor_qa_sha256": "sha256 hex",
    "gate_health_summary": "benchmark/gate_health_summary.json",
    "gate_health_summary_sha256": "sha256 hex"
  },
  "selected_row_snapshot": {"row_id": "same as selected_row_id"},
  "resolved_full_run_config_path": "configs/spec0001/non_eq_vae_baseline.resolved.json",
  "resolved_full_run_config_sha256": "sha256 hex"
}
```

For `selected_runtime.json`, `world_size` and `nproc_per_node` are numeric:
`1` for `single_visible_t4`, `2` for `dual_t4_ddp`.

Local schema-smoke `selected_runtime.json` must use `status = "schema_pass"`,
`benchmark_kind = "local_synthetic_schema"`, `benchmark_source =
"local_synthetic_schema_smoke"`, and `full_run_eligible = false`.
Its `dataloader` object must include `non_blocking_h2d = false`, and any linked
local safety statuses must be `schema_pass`, not real-data `pass`. No training
command may accept it as a runtime config.

`benchmark/runtime_proof.json` required shape:

```json
{
  "status": "pass",
  "machine_shape": "NvidiaTeslaT4",
  "kernel_metadata_machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"],
  "kaggle_cli_version": "string",
  "launch_command_hash": "sha256 hex",
  "accelerator_modes_checked": ["single_visible_t4", "dual_t4_ddp"],
  "single_visible_t4": {
    "visible_device_count": 1,
    "cuda_device_count": 1,
    "gpu_names": ["..."],
    "world_size": 1,
    "nproc_per_node": 1,
    "cuda_visible_devices": "0-or-equivalent"
  },
  "dual_t4_ddp": {
    "visible_device_count": 2,
    "cuda_device_count": 2,
    "gpu_names": ["...", "..."],
    "world_size": 2,
    "nproc_per_node": 2,
    "ranks": [
      {"rank": 0, "local_rank": 0, "device_name": "..."},
      {"rank": 1, "local_rank": 1, "device_name": "..."}
    ]
  },
  "failure_kind": ""
}
```

`benchmark/model_loss_train_step.json` required shape for the narrow local
model/loss slice:

```json
{
  "status": "local_pass",
  "benchmark_kind": "local_synthetic_model_loss_train_step",
  "benchmark_source": "local_cpu_synthetic_train_step",
  "full_run_eligible": false,
  "run_name": "string",
  "config_path": "configs/spec0001/non_eq_vae_debug_cpu.json",
  "config_sha256": "sha256 hex",
  "effective_config_sha256": "sha256 hex",
  "architecture_id": "spec0001_non_eq_vae_translatable",
  "topology_version": "spec0001.count.v1",
  "model_count_path": "benchmark/model_count.json",
  "model_count_sha256": "sha256 hex",
  "model_count_status": "pass",
  "matches_spec_target": true,
  "accelerator_mode": "local_cpu",
  "machine_shape": "local_cpu",
  "device": "cpu",
  "precision_policy": "amp_off_fp32",
  "amp_enabled": false,
  "torch_compile": {
    "enabled": true,
    "status": "local_pass or skipped_unsupported",
    "failure_kind": ""
  },
  "float16_smoke": {
    "enabled": true,
    "status": "local_pass or skipped_unsupported",
    "failure_kind": ""
  },
  "corruption_strategy": "identity_clean_no_corruption",
  "batch_size": 2,
  "input_shape": [2, 3, 256, 256],
  "latent_shape": [2, 16, 32, 32],
  "forward_contract": {
    "explicit_eps_used": true,
    "returned_reconstruction": true,
    "returned_mu": true,
    "returned_logvar_raw": true,
    "returned_logvar_clamped": true,
    "returned_z": true,
    "returned_eps": true,
    "returned_logvar_clamp_count": true
  },
  "zero_head": {
    "weight_zero": true,
    "bias_zero": true,
    "initial_reconstruction_max_abs": 0.0,
    "status": "pass"
  },
  "loss": {
    "loss": 0.0,
    "recon_loss": 0.0,
    "l1_loss": 0.0,
    "ssim_loss": 0.0,
    "ssim_metric": 0.0,
    "kl_loss": 0.0,
    "beta": 0.0,
    "all_finite": true
  },
  "posterior": {
    "mu_mean": 0.0,
    "mu_std": 0.0,
    "logvar_raw_mean": 0.0,
    "logvar_raw_std": 0.0,
    "logvar_clamped_mean": 0.0,
    "logvar_clamped_std": 0.0,
    "logvar_clamp_count": 0,
    "logvar_clamp_fraction": 0.0
  },
  "optimizer": {
    "name": "AdamW",
    "parameter_group_count": 3,
    "all_trainable_parameters_covered_once": true,
    "gate_parameters_in_gate_no_decay_group": true,
    "base_lr": 0.0005,
    "weight_decay": 0.00001,
    "gate_lr_multiplier": 0.5
  },
  "backward_update": {
    "grad_norm": 0.0,
    "param_update_norm": 0.0,
    "nonfinite_count": 0,
    "trainable_parameter_tensor_count": 194,
    "nonzero_grad_parameter_tensor_count": 2,
    "nonzero_update_parameter_tensor_count": 2,
    "first_step_update_scope": "zero_head_final_rgb_head_smoke",
    "optimizer_step_index": 0,
    "successful_optimizer_update_count": 1,
    "beta_warmup_steps": 1
  },
  "metrics": {
    "mae_norm": 0.0,
    "mse_norm": 0.0,
    "psnr_img": 0.0,
    "ssim_img": 0.0
  },
  "failure_kind": "",
  "failure_message_hash": ""
}
```

The local model/loss train-step artifact must never update or replace
`benchmark/selected_runtime.json`. If eager FP32 evidence fails, the artifact
status is `fail`. Optional CPU compile or float16 smoke failures may be recorded
inside their nested objects as `skipped_unsupported` without failing the whole
slice, but only when eager FP32 has `status = "local_pass"` and the unsupported
feature has deterministic `failure_kind` metadata.
Because this first local smoke uses a zero-initialized RGB output head and
`beta = 0.0`, it is expected to prove the final-head forward/update path rather
than full hidden-stack gradient connectivity. The artifact must therefore record
`trainable_parameter_tensor_count`, `nonzero_grad_parameter_tensor_count`,
`nonzero_update_parameter_tensor_count`, and `first_step_update_scope =
"zero_head_final_rgb_head_smoke"` so later agents do not over-interpret the
smoke result. Broader hidden-stack connectivity and gate-health evidence belong
to later selected-runtime debug and tiny-overfit gates.
The train-step writer must run or consume the current instantiated
`benchmark/model_count.json` proof and fail unless that proof has
`status = "pass"` and `matches_spec_target = true` for the same effective config
hash, `architecture_id`, and `topology_version`.

`benchmark/stain_corruptor_qa.json` required shape for the narrow local
corruption correctness/QA slice:

```json
{
  "status": "local_pass",
  "benchmark_kind": "local_synthetic_stain_corruptor_qa",
  "benchmark_source": "local_cpu_synthetic_stain_corruptor_qa",
  "full_run_eligible": false,
  "run_name": "string",
  "config": {
    "path": "configs/spec0001/non_eq_vae_debug_cpu.json",
    "invoked_config_hash": "sha256 hex",
    "effective_config_hash": "sha256 hex",
    "source_config_chain": []
  },
  "corruption_version": "spec0001.hed_corruptor.v1",
  "profile_name": "conservative_default",
  "reference_oracle": {
    "name": "scikit-image",
    "version": "0.26.0",
    "source_url": "https://github.com/scikit-image/scikit-image/blob/v0.26.0/skimage/color/colorconv.py",
    "runtime_dependency": false,
    "runtime_code_imports_scikit_image": false,
    "fixture_source": "checked-in scikit-image 0.26.0 oracle values"
  },
  "api_contract": {
    "input_shape": [25, 3, 256, 256],
    "output_shape": [25, 3, 256, 256],
    "input_domain": "normalized_rgb_minus1_1",
    "output_domain": "normalized_rgb_minus1_1",
    "channel_order": "NCHW_RGB",
    "dtype": "torch.float32",
    "target_preservation": "x_clean_unchanged",
    "mask_handling": "masks_not_modified_by_corruptor"
  },
  "hed_convention": {
    "rgb_from_hed": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    "hed_from_rgb": [[1.8779827368521356, -1.0076786862855642, -0.5561158181996246], [-0.06590806222356334, 1.1347303724996625, -0.13552179862837116], [-0.6019073634392891, -0.4804141884970579, 1.5735880719641926]],
    "od_epsilon": 0.000001,
    "uses_srgb_gamma_decode": false,
    "channel_first_multiplication": "torch.einsum('bchw,cd->bdhw')",
    "arbitrary_rgb_roundtrip": "not_required_after_rgb2hed_clamps_negative_stain_channels"
  },
  "rng": {
    "corruption_seed": 0,
    "semantic_seed_fields": [
      "corruption_seed",
      "split",
      "semantic_sample_key",
      "corruption_step",
      "corruption_view",
      "corruption_version"
    ],
    "rank_in_semantic_seed": false,
    "corruption_step": 0,
    "corruption_view": "train_corrupted_local_qa",
    "clean_validation_consumes_rng": false
  },
  "profile": {
    "corrupt_prob": 0.3,
    "he_alpha_range": [0.8, 1.2],
    "he_beta_range": [-0.05, 0.05],
    "residual_alpha_range": [0.98, 1.02],
    "residual_beta_range": [-0.01, 0.01],
    "noise_std_range": [0.0, 0.05],
    "residual_axis_semantics": "tiny_residual_axis_jitter_not_biological_dab"
  },
  "checks": {
    "finite_pass": true,
    "output_range_pass": true,
    "shape_preserved": true,
    "dtype_preserved": true,
    "target_preserved": true,
    "clean_validation_rng_advanced": false,
    "oracle_rgb2hed_max_abs_error": 0.0,
    "oracle_hed2rgb_max_abs_error": 0.0,
    "applied_count": 0,
    "sample_count": 25
  },
  "summary": {
    "clean": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
    "corrupted": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
    "stain_only": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
    "gaussian_only": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
    "combined": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
  },
  "clamp_fractions": {
    "stain_only_low": 0.0,
    "stain_only_high": 0.0,
    "gaussian_only_low": 0.0,
    "gaussian_only_high": 0.0,
    "combined_low": 0.0,
    "combined_high": 0.0
  },
  "visual_artifacts": {
    "directory": "artifacts/stain_corruptor_qa",
    "synthetic_grid_path": "artifacts/stain_corruptor_qa/synthetic_grid.png",
    "synthetic_grid_sha256": "sha256 hex",
    "grid_order": ["clean", "stain_only", "gaussian_only", "combined"],
    "fixed_real_25_status": "requires_real_data_generation"
  }
}
```

The local `stain_corruptor_qa.json` artifact is a correctness artifact, not
runtime-selection evidence. It may use synthetic patches before real data are
available and may report `fixed_real_25_visual_qa_status =
"requires_real_data_generation"`. Before the first Kaggle baseline run, the
fixed real 25-patch visual QA must be generated from the canonical selector
provenance and must pass without changing the corruption profile.

`benchmark/corruption_checks.csv` required columns for branchless/indexed and
clean-validation RNG evidence:

```text
run_name,benchmark_kind,benchmark_source,full_run_eligible,accelerator_mode,machine_shape,row_id,reference_row_id,candidate_row_id,batch_index,corruption_version,profile_name,corruption_strategy,corruption_view,corruption_step,split,semantic_sample_key_hash,binary_sample_id_hash,rank,world_size,applied_mask_hash,stain_param_hash,noise_std_hash,noise_field_hash,clean_sample_unchanged_count,clean_validation_rng_advanced,status,failure_kind
```

`benchmark/numerical_checks.csv` required columns:

```text
run_name,benchmark_kind,benchmark_source,full_run_eligible,accelerator_mode,machine_shape,row_id,reference_row_id,candidate_row_id,batch_index,precision_policy,torch_compile_enabled,corruption_strategy,total_loss_abs_delta,total_loss_rel_delta,recon_loss_abs_delta,recon_loss_rel_delta,l1_loss_abs_delta,l1_loss_rel_delta,ssim_loss_abs_delta,ssim_loss_rel_delta,kl_loss_abs_delta,kl_loss_rel_delta,grad_norm_abs_delta,grad_norm_rel_delta,param_update_norm_abs_delta,param_update_norm_rel_delta,x_hat_min_abs_delta,x_hat_max_abs_delta,mu_mean_abs_delta,mu_std_abs_delta,logvar_mean_abs_delta,logvar_std_abs_delta,logvar_clamp_count_delta,gate_health_status,nonfinite_count,amp_step_skipped,status,failure_kind
```

`metrics/gate_health.csv` required columns:

```text
run_name,benchmark_kind,benchmark_source,full_run_eligible,accelerator_mode,machine_shape,optimizer_step,module,gate_kind,num_channels,num_elements,a_min,a_max,a_mean,a_std,b_min,b_max,b_mean,b_std,max_abs_a,max_abs_b,gate_mean,gate_std,gate_p01,gate_p50,gate_p99,frac_gate_lt_0_01,frac_gate_gt_0_99,worst_channel_frac_gate_lt_0_01,worst_channel_frac_gate_gt_0_99,dead_channel_count,input_rms,output_rms,output_input_rms_ratio,a_grad_norm,b_grad_norm,a_update_to_param_norm,b_update_to_param_norm,gate_health_status
```

`benchmark/gate_health_summary.json` required shape:

```json
{
  "status": "pass",
  "benchmark_kind": "kaggle_gate_health_benchmark",
  "benchmark_source": "kaggle_runtime_benchmark",
  "overall_status": "pass",
  "full_run_eligible": true,
  "logged_intervals": 0,
  "module_count": 34,
  "nonfinite_count": 0,
  "max_abs_a": 0.0,
  "max_abs_b": 0.0,
  "worst_frac_gate_lt_0_01": 0.0,
  "worst_frac_gate_gt_0_99": 0.0,
  "dead_channel_count": 0,
  "zero_gradient_interval_count": 0,
  "worst_output_input_rms_ratio": 0.0,
  "failing_modules": [],
  "warning_modules": []
}
```

`benchmark/tiny_overfit_summary.json` required shape:

```json
{
  "status": "pass",
  "runtime_config": "benchmark/selected_runtime.json",
  "runtime_config_sha256": "sha256 hex",
  "fixed_train_patches": "configs/spec0001/fixed_32_train_overfit_patches.json",
  "fixed_train_patches_sha256": "sha256 hex",
  "patch_count": 32,
  "optimizer_steps": 300,
  "smoothing_window_steps": 25,
  "corruption_strategy": "branchless_all",
  "eval_views": ["train_clean", "train_corrupted_fixed_seed"],
  "initial_smoothed_l1": 0.0,
  "final_smoothed_l1": 0.0,
  "initial_smoothed_recon_loss": 0.0,
  "final_smoothed_recon_loss": 0.0,
  "l1_improvement_fraction": 0.05,
  "recon_loss_improvement_fraction": 0.05,
  "zero_head_baseline_psnr": 0.0,
  "final_psnr": 0.0,
  "zero_head_baseline_ssim": 0.0,
  "final_ssim": 0.0,
  "max_logvar_clamp_fraction": 0.0,
  "max_frac_x_hat_lt_minus1": 0.0,
  "max_frac_x_hat_gt_1": 0.0,
  "gate_health_status": "pass"
}
```

Tiny-overfit passes only if `patch_count == 32`, `status == "pass"`,
finite losses/metrics are present, both L1 and reconstruction loss improve by at
least 5 percent, either PSNR or SSIM improves over the zero-head baseline,
`max_logvar_clamp_fraction <= 0.05`,
`max(max_frac_x_hat_lt_minus1, max_frac_x_hat_gt_1) <= 0.05`, and gate health
is `pass` or `warn` with an explicit note. A `warn` gate-health status still
blocks the first full run until inspected.

Tiny-overfit smoothing is fixed: `initial_smoothed_*` is the mean over the first
25 successful optimizer updates and `final_smoothed_*` is the mean over the last
25 successful optimizer updates. Skipped AMP batches do not count as successful
updates. The zero-head baseline is computed before any optimizer update on the
same 32 fixed clean patches, with PSNR/SSIM measured after the image-domain
projection used elsewhere in this spec. The corruption path uses the selected
runtime's corruption strategy, but the overfit evaluation view must also include
a fixed-seed corrupted pass so the improvement check is reproducible. The
`fixed_train_patches_sha256` is computed from canonical JSON bytes for the
selector file; changing the selector requires a spec/config update.

Required first-run budget defaults:

| Config | Batch size | Train steps | Validation interval | Checkpoint interval |
| --- | ---: | ---: | ---: | ---: |
| `non_eq_vae_debug_cpu.json` | 2 global | 8 | 4 | 4 |
| `non_eq_vae_kaggle_runtime_benchmark.json` | searched per device | short fixed benchmark steps | optional one fixed validation micro-pass | none except benchmark summary |
| `non_eq_vae_kaggle_debug.json` | benchmarked per device | 200 | 50 | half epoch or 100 steps |
| `non_eq_vae_kaggle_tiny_overfit.json` | selected runtime | 300 on 32 fixed real patches | 50 | 100 |
| `non_eq_vae_baseline.json` | benchmark-selected per device | 10 epochs | half epoch | half epoch |

The future `SO(2)` model must use the same training budget and validation
access, unless a later run spec explicitly supersedes both models together.

Checkpoint retention:

- epoch-based runs save and validate every half epoch;
- retain `best_model.pt`, the final checkpoint, and the latest four interval
  checkpoints, mirroring the useful FSQ-era retention behavior without reusing
  FSQ checkpoint formats;
- record checkpoint pruning decisions in the run manifest;
- resume must restore model, optimizer, LR scheduler, beta scheduler, AMP scaler
  when present, epoch/progress counters, config hash, and RNG state.

Required output schemas:

- `config_resolved.json`: full config after CLI overrides;
- `metrics/train_steps.csv`: one row per logged train step with at least
  `run_name,event_id,batch_attempt,optimizer_step,split,loss,recon_loss,l1_loss,ssim_loss,ssim_metric,mae_norm,mse_norm,psnr_img,ssim_img,kl_loss,beta,lr,grad_norm,batch_size,precision_policy,amp_enabled,torch_compile_enabled,corruption_strategy,amp_step_skipped,mu_mean,mu_std,mu_min,mu_max,logvar_mean,logvar_std,logvar_min,logvar_max,logvar_clamp_count,x_hat_min,x_hat_max,frac_x_hat_lt_minus1,frac_x_hat_gt_1`;
- skipped AMP rows are logged as batch-attempt events with
  `amp_step_skipped = 1`; they do not increment `optimizer_step` and do not
  trigger optimizer-step-based schedules, validation, or checkpointing;
- `metrics/validation_steps.csv`: one row per validation event with at least
  `run_name,optimizer_step,split,view,n,mse_norm_mean,mae_norm_mean,psnr_img_mean,ssim_img_mean,psnr_img_inf_count,kl_mean,mu_mean,mu_std,logvar_mean,logvar_std,logvar_clamp_count,x_hat_min,x_hat_max,frac_x_hat_lt_minus1,frac_x_hat_gt_1`;
- `eval/per_image_metrics.csv`: one row per evaluated patch with at least
  `sample_id,split,view,wsi_id,label,x,y,file_index,row_index,mse_norm,mae_norm,psnr_img,ssim_img`;
- `eval/summary.json`: mean, standard deviation, and `n` for every metric,
  grouped by `split` and `view`, with the PSNR infinite-value policy described
  in the reconstruction metric contract;
- `artifacts/manifest.json`: paths and provenance for every generated figure;
- `benchmark/model_count.json`: analytic/implementation model count comparison
  including learned parameters, learned-conv MACs, fixed-resampling MACs,
  normalization/gate parameters, and pass/fail status against this spec;
- `benchmark/model_loss_train_step.json`: local synthetic model/loss
  train-step evidence for the narrow slice, with `status = "local_pass"` only
  for eager FP32 real-code evidence and `full_run_eligible = false`;
- `benchmark/stain_corruptor_qa.json`: local synthetic and later fixed-real
  HED/stain corruptor convention, RNG, metadata, range, and visual-QA evidence
  for the narrow corruption slice, with `full_run_eligible = false`;
- `benchmark/runtime_matrix.csv`: one row per benchmarked runtime configuration;
- `benchmark/selected_runtime.json`: selected accelerator, compile, AMP, and
  batch-size decision for the first full run, including selected
  `precision.policy` and `corruption.strategy`;
- `benchmark/dataloader_matrix.csv`: real train/validation loader, transfer,
  throughput, wait-fraction, and rank-balance measurements;
- `benchmark/numerical_checks.csv`: paired fixed-batch deltas against
  `amp_off_fp32` eager for precision/compile candidates;
- `benchmark/corruption_checks.csv`: branchless/indexed corruption-equivalence
  hashes, clean-sample unchanged counts, and clean-validation RNG evidence;
- `metrics/gate_health.csv`: per-module gate parameter, saturation, RMS, and
  gradient/update telemetry from debug and benchmark runs;
- `benchmark/gate_health_summary.json`: gate-health pass/warn/fail summary used
  before the first full run;
- `benchmark/tiny_overfit_summary.json`: selected-runtime real-patch overfit
  sanity summary before the first full run;
- `checkpoints/step_*.pt`: model, optimizer, scheduler, beta scheduler, scaler
  if present, current step, config hash, and RNG state.

`logvar_clamp_count` must be logged whenever any values are clamped.

## Fixed 25-Patch Protocol

The qualitative 25-patch set must be deterministic and shared by the baseline
and future `SO(2)` model.

Selection policy for `configs/spec0001/fixed_25_validation_patches.json`:

1. Use the validation CSV from `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
2. Group rows by numeric label `0..4`.
3. For each row, compute
   `sha256("20260610:{wsi_id}:{label}:{x}:{y}")`.
4. Sort each label group by that digest, then by `wsi_id,x,y`.
5. Select the first 5 rows per label.
6. Store the ordered 25 selectors with audit-stable fields:
   `rank,source_split,file_index,row_index,sample_id,wsi_id,label,x,y,
   selection_key_sha256,patch_sha256`.
7. Store top-level source provenance: `dataset_slug`, resolved `data_root`,
   `csv_path`, `csv_sha256`, `bin_path`, `bin_file_size`, `header_sha256`,
   parsed header fields, `row_count`, `patch_count`, `idx_policy`, and
   `crc_checked`.

The artifact command may accept `--fixed-count 25`, but implementation must fail
if the fixed-patch config is missing or if the selected count is not exactly 25.
Do not silently resample a different set.

Because the Kaggle validation CSV is not committed in this repo, the first
implementation must include a deterministic selector generator:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.select_fixed_patches \
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
  --data-root auto \
  --output configs/spec0001/fixed_25_validation_patches.json \
  --validate-crc \
  --allow-tracked-config-overwrite
```

This generator requires access to the real validation CSV and is therefore a
data-access step, not a pure offline local test. Local synthetic tests may use a
separate generated synthetic selector under `runs/` but must never overwrite the
canonical fixed-25 config.

Locked selector/dataloader policy, 2026-06-12:

- selector schema is `spec0001.fixed_selector.v1` with top-level source
  provenance plus audit-stable selector rows; `wsi_id,label,x,y` alone are not
  sufficient;
- selector validation recomputes the deterministic selection policy and fails
  if a document contains internally consistent but noncanonical rows;
- canonical split names are `train` and `validation`; `valid`/`val` are input
  aliases only;
- `eqvae.cli.select_fixed_patches` may write canonical tracked selector config
  paths only when both `--allow-tracked-config-overwrite` and `--validate-crc`
  are supplied. Noncanonical local synthetic outputs should live under ignored
  `runs/` paths and may skip full CRC;
- canonical selector placeholders stay tracked with
  `status = "requires_real_data_generation"` until the real Kaggle CSV/bin
  files are available locally. Commands that consume fixed selectors must reject
  those placeholders rather than resampling.

## Fixed 32-Train Tiny-Overfit Protocol

The tiny-overfit sanity check must not reuse the fixed 25-patch validation
artifact set. It uses a separate deterministic train-patch selector so the
validation qualitative set remains a held-out visual protocol.

Selection policy for `configs/spec0001/fixed_32_train_overfit_patches.json`:

1. Use the train CSV from `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
2. Exclude any WSI listed in `docs/data/ubc_ocean_masked_holdout_ids.csv`.
3. Compute `sha256("20260611:tiny-overfit:{wsi_id}:{label}:{x}:{y}")`.
4. Sort by digest, then by `wsi_id,label,x,y`.
5. Select the first 32 rows.
6. Store the ordered 32 selectors with the same audit-stable selector row and
   source-provenance schema used by the fixed-25 validation selector.

Canonical generation command:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.select_fixed_patches \
  --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
  --data-root auto \
  --output configs/spec0001/fixed_32_train_overfit_patches.json \
  --validate-crc \
  --allow-tracked-config-overwrite
```

Tiny-overfit commands must fail if the fixed 32-train config is missing, has any
validation row, or contains a count other than 32.

## Rotated And Latent Artifact Protocol

Spec 0001 must produce baseline-compatible placeholders or outputs for the
advisor-requested rotated-input and latent visualizations without pretending the
scalar baseline has nontrivial `SO(2)` latent fields.

Baseline protocol:

- use the deterministic posterior mean `mu`, not sampled `z`, for latent
  transformation artifacts;
- rotate image inputs with the documented interpolation, padding/cropping, and
  boundary-mask policy used by the evaluator;
- for the scalar spatial latent baseline, the transformed-latent path is:
  encode clean input to `mu`, spatially rotate the 16-channel latent map with the
  same angle convention, decode with the decoder path, then compare against the
  rotated-input reconstruction;
- report boundary-masked and unmasked error maps;
- store angle list, interpolation mode, padding mode, align-corners policy, and
  mask policy in `artifacts/manifest.json`;
- future nontrivial `SO(2)` latent-field transformations require a follow-up
  spec because `mu`, `logvar`, sampling, KL, and representation action become
  irrep-aware.

## Required Implementation Artifacts

Expected package root:

```text
src/eqvae/
```

Required modules:

- `src/eqvae/data/patch_shards.py`: UBC binary/CSV patch shard dataset;
- `src/eqvae/data/synthetic.py`: tiny deterministic synthetic patch shards for
  local tests and smoke runs;
- `src/eqvae/data/splits.py`: WSI and masked-holdout split validation helpers;
- `src/eqvae/data/roots.py`: deterministic explicit/auto UBC data-root
  resolution;
- `src/eqvae/data/dataloaders.py`: fast read-only mmap tensor-only dataset and
  `uint8` batch normalization helper;
- `src/eqvae/data/fixed_selectors.py`: fixed-25/fixed-32 selector generation,
  schema loading, and drift/policy validation;
- `src/eqvae/corruption/stain.py`: Tellez-style HED/OD stain jitter and
  Gaussian noise corruption with corrected matrix convention;
- `src/eqvae/models/field_schedule.py`: tensor-channel schedule and future
  `SO(2)` field multiplicity metadata;
- `src/eqvae/models/activations.py`: gated scalar activation and future radial
  gate policy;
- `src/eqvae/models/non_equivariant_vae.py`: translatable Conv2d VAE factory;
- `src/eqvae/losses/vae.py`: reconstruction, KL, and beta schedule;
- `src/eqvae/training/`: train-step and optimizer-group helpers introduced by
  the narrow model/loss train-step slice;
- `src/eqvae/metrics/reconstruction.py`: SSIM, MAE, MSE, PSNR;
- `src/eqvae/artifacts/`: boxplots, dashboards, fixed-patch grids, rotated-input
  grids, rotated-input versus latent grids, and latent visualization helpers;
- `src/eqvae/checkpointing.py`: save/resume with RNG state;
- `src/eqvae/cli/`: `smoke`, `model_count`, `train`, `benchmark_runtime`,
  `select_fixed_patches`, `evaluate`, and `artifacts` entry points.

Required config files:

- `configs/spec0001/non_eq_vae_model_base.json`;
- `configs/spec0001/non_eq_vae_baseline.json`;
- `configs/spec0001/non_eq_vae_debug_cpu.json`;
- `configs/spec0001/non_eq_vae_kaggle_debug.json`;
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`;
- `configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json`;
- `configs/spec0001/ubc_ocean_masked_holdout_test.json`;
- `configs/spec0001/fixed_32_train_overfit_patches.json`;
- `configs/spec0001/fixed_25_validation_patches.json`.

The shared `corruption` config object is expanded in
`configs/spec0001/non_eq_vae_model_base.json` and the duplicated local debug CPU
config. It must include:

- `kind = "tellez_hed_gaussian"`;
- `implementation_status`;
- `corruption_version = "spec0001.hed_corruptor.v1"`;
- `profile_name` with one of `conservative_default` or `fsq_legacy_wide`;
- `corrupt_prob`;
- `he_alpha_range`, `he_beta_range`, `residual_alpha_range`,
  `residual_beta_range`, and `noise_std_range`;
- `hed_matrix_source`, `hed_matrix_version`, `rgb_from_hed`,
  `od_epsilon = 1e-6`, and `uses_srgb_gamma_decode = false`;
- `rng_policy = "semantic_stateless_v1"`, `corruption_seed`,
  `semantic_seed_fields`, and `rank_in_semantic_seed = false`;
- `clean_validation_consumes_rng = false`;
- output-domain fields: `input_domain`, `output_domain`, `channel_order`,
  `final_clamp_min`, `final_clamp_max`, and `metadata_schema_version`.

Scaffold status, 2026-06-12: the config files exist as JSON scaffolds.
`fixed_25_validation_patches.json` and
`fixed_32_train_overfit_patches.json` are deliberately marked
`requires_real_data_generation` with `spec0001.fixed_selector.v1` metadata and
empty selectors until the real Kaggle CSVs are available. Commands that need
canonical fixed selectors must reject those placeholder files rather than
silently resampling.

Run outputs should go under ignored `runs/` paths locally and `/kaggle/working`
on Kaggle.

Package/import policy:

- use `src/eqvae` as the implementation package root;
- local commands use `PYTHONPATH=src` until a packaging backend is explicitly
  added;
- Kaggle launchers must insert `payload/src` into `sys.path` before importing
  `eqvae`;
- adding a build backend or package-discovery metadata requires updating this
  spec and the lockfile.

Config and dependency policy:

- configs are JSON by contract for spec 0001 and must use only the Python
  standard-library `json` parser/writer;
- do not add a YAML parser solely for experiment config files in spec 0001;
- Kaggle/debug/full-run configs that use `source_config` must inherit from the
  model-only base contract, not from the local CPU debug config. Local-only keys
  such as `benchmark`, `dataloader_pretest`, and CPU `runtime.device` must not
  appear in resolved Kaggle configs;
- repo-owned Torch SSIM must be implemented under `src/eqvae`; do not import
  `pytorch-msssim` in spec 0001 code unless a later spec deliberately changes
  the offline/compiled SSIM policy;
- `pytorch-msssim` is no longer a direct dependency in `pyproject.toml` or
  `uv.lock`; the remaining `pytorch_msssim` import is inside user-retained
  historical `src/nn` reference code and must not be imported by `src/eqvae`;
- historical `src/nn` is excluded from Ruff/BasedPyright production scopes by
  spec 0002 decision and may remain as reference material; benchmark CLIs are
  not implementation-ready unless `./scripts/python_quality.sh` passes for the
  production scope and no active `src/eqvae` code imports `src.nn`.

Local CPU smoke policy:

- CPU smoke tests are shape/contract tests, not speed benchmarks;
- CPU `torch.compile` tests may use tiny synthetic batches and must have bounded
  step counts so they do not turn into long local training jobs;
- CPU float16 smoke is allowed to be a narrow dtype-path check with documented
  tolerances or explicit skips for unsupported CPU operations;
- GPU speed, AMP, and DDP behavior are decided only by the permission-gated
  Kaggle runtime benchmark.
- Local synthetic benchmark outputs may use `accelerator_mode = "local_cpu"`,
  `machine_shape = "local_cpu"`, `status = "schema_pass"` for schema-only
  smoke, or `status = "local_pass"` for measured local CPU synthetic
  pre-tests. Both local statuses require `full_run_eligible = false` and cannot
  be selected as runtime evidence.

Implementation milestones before broad coding:

1. Completed narrow scaffold/topology-count slice: `src/eqvae`,
   `configs/spec0001`, no-sync import smoke, local schema artifacts,
   instantiated model-count verification, layered JSON config resolution, and
   observed `model_inventory.csv`.
2. Remaining spec relock slice: future SO(2) count ceiling, real fixed
   validation/tiny-overfit selector generation data-access step, remaining
   artifact protocol, package/import policy, and final clean-context
   spec review.
3. Completed narrow data/metrics slice: `data_metrics_ready` now covers
   patch-shard loader, synthetic UBC-format shards, split validation, and
   MAE/MSE/PSNR/SSIM metrics under that narrow authorization. It is local
   implementation evidence only, not a benchmark, Kaggle, training, or paper
   unlock.
4. Completed narrow selector/dataloader slice:
   `fixed_selectors_dataloader_ready` now covers deterministic data-root
   resolution, tensor-only mmap loaders, fixed selector schema/generation/
   validation, and local synthetic selector tests. It is local implementation
   evidence only and does not generate the real canonical selectors until the
   real Kaggle CSV/bin files are available locally.
5. Implemented local benchmark pre-test writer:
   `local_benchmark_pretest_ready` now locks and implements how local
   CPU synthetic dataloader pre-tests report `local_pass`, the FSQ-derived mmap
   tensor-only hot path, explicit candidate knobs, candidate failure rows, and
   `full_run_eligible = false`. The checked-in debug pre-test was rerun outside
   the sandbox on 2026-06-12 and measured all configured candidates.
6. Completed model/loss slice: `model_loss_train_step_ready` now implements the
   VAE forward API, explicit `eps` control, clamped-logvar sampling/KL
   semantics, exact L1/SSIM/KL reductions, beta schedule, identity-clean local
   input rule, semantic AdamW groups, non-promotable
   `benchmark/model_loss_train_step.json`, first-step final-head update
   telemetry, and local compile/precision smoke status semantics.
7. HED/stain corruption slice: `corruption_ready` now implements PyTorch
   scikit-compatible HED/RGB conversion, conservative and FSQ-wide profiles,
   semantic stateless RNG, branchless-all execution, metadata, and
   non-promotable synthetic `benchmark/stain_corruptor_qa.json` evidence.
   Training integration, branchless/indexed runtime checks, and fixed real
   25-patch visual QA remain separate gates.
8. Train/resume slice: optimizer/scheduler, AMP skipped-step behavior,
   checkpoint save/resume, metrics schemas, retention.
9. Artifact/evaluation slice: fixed validation/tiny-overfit selectors,
   evaluator, boxplots, dashboards, rotated/latent artifacts.
10. Kaggle slice: payload build, debug launcher, local payload validation, then
   permission-gated remote benchmark/debug runs.

## Kaggle Packaging Contract

The Kaggle CLI script-kernel push serializes the declared `code_file`; the
2026-06-13 real-data smoke failed with `ModuleNotFoundError: No module named
'eqvae'` because a sibling `payload/` directory was prepared locally but was not
available to the remote script. Therefore every remote script-kernel smoke or
benchmark must be self-contained in the uploaded code file, or must use another
source-delivery mechanism explicitly proved by an upload-simulation test before
remote push.

The legacy local real-data sibling-payload layout is retained only as an ignored
historical build artifact path. The active real-data launcher now uses the same
generated single-file embedded packaging pattern as setup smoke:

```text
kaggle/kernels/non_eq_vae_debug/
  kernel-metadata.json
  run_template.py
  run.py                  # generated/ignored, embeds a zipped payload
```

The setup-smoke generated layout is:

```text
kaggle/kernels/setup_smoke/
  kernel-metadata.json
  run_template.py
  run.py                  # generated/ignored, embeds a zipped payload
```

Required local build commands:

```bash
./scripts/kaggle_kernel.sh build
./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke
```

Build rules:

- copy or embed only allowlisted implementation files;
- do not copy `.git`, `.venv`, paper files, historical notebooks, checkpoints,
  local run artifacts, credentials, or Overleaf data;
- generated setup `run.py` must decode the embedded zip, verify embedded
  SHA-256 constants, insert the extracted `src` into `sys.path` before
  importing `eqvae`, and assert `eqvae.__file__` is under the extracted payload;
- Kaggle internet stays disabled;
- first implementation must not require `pip install` or dependency resolution
  on Kaggle;
- metrics must use a repo-owned Torch SSIM implementation or another bundled
  implementation, not an undeclared network dependency;
- if a future dependency is unavailable on Kaggle, bundle a wheel under
  `payload/wheels/` and install with `--no-index --find-links`, with a separate
  spec update first.

Kaggle debug metadata must keep:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"],
"competition_sources": [],
"kernel_sources": [],
"model_sources": [],
"enable_internet": "false"
```

The historical FSQ output dataset `maximusshtefan/non-eq-vae-output` is forbidden
for spec 0001 kernels.

The setup-smoke metadata must instead keep all source lists empty, request no
GPU, and write only non-promotable setup evidence:

```json
"dataset_sources": [],
"competition_sources": [],
"kernel_sources": [],
"model_sources": [],
"enable_gpu": "false",
"enable_internet": "false"
```

## Verification Commands

The implementation is accepted only when these exact local commands exist and
pass. Local commands must not create, sync, or refresh the environment. If the
repo-local `.venv` is missing or stale, ask the user before running:

```bash
uv sync --locked --python 3.12 --group dev
```

General repo checks:

```bash
./scripts/agent_preflight.sh
./scripts/python_quality.sh
```

Spec 0001 uses the spec 0002 production boundary: historical exploratory
`src/nn` is excluded from Ruff/BasedPyright and may remain as reference
material, while production Python under `src/eqvae` and tests must pass
`./scripts/python_quality.sh`. Empty `main.py` was deleted on 2026-06-12. Do
not weaken global Ruff/BasedPyright settings and do not add global ignores.

Narrow `data_metrics_ready` local checks:

```bash
.venv/bin/ruff format src/eqvae \
  tests/test_patch_shards.py \
  tests/test_split_validation.py \
  tests/test_reconstruction_metrics.py \
  tests/test_spec0001_benchmark_scaffold.py

.venv/bin/ruff check src/eqvae \
  tests/test_patch_shards.py \
  tests/test_split_validation.py \
  tests/test_reconstruction_metrics.py \
  tests/test_spec0001_benchmark_scaffold.py \
  tests/__init__.py

.venv/bin/basedpyright src/eqvae \
  tests/test_patch_shards.py \
  tests/test_split_validation.py \
  tests/test_reconstruction_metrics.py \
  tests/test_spec0001_benchmark_scaffold.py \
  tests/__init__.py

PYTHONPATH=src .venv/bin/pytest \
  tests/test_patch_shards.py \
  tests/test_split_validation.py \
  tests/test_reconstruction_metrics.py \
  tests/test_spec0001_benchmark_scaffold.py
```

As of 2026-06-12 these focused checks pass locally. Benchmark CLIs still become
implementation-ready only when the full production-scope
`./scripts/python_quality.sh` passes and the relevant benchmark artifacts are
real-code outputs rather than schema placeholders.

Narrow `fixed_selectors_dataloader_ready` local checks:

```bash
.venv/bin/ruff format src/eqvae/data src/eqvae/cli/select_fixed_patches.py \
  tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py

.venv/bin/ruff check src/eqvae/data src/eqvae/cli/select_fixed_patches.py \
  tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py

.venv/bin/basedpyright src/eqvae/data src/eqvae/cli/select_fixed_patches.py \
  tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py

PYTHONPATH=src .venv/bin/pytest \
  tests/test_data_roots.py tests/test_dataloaders.py tests/test_fixed_selectors.py
```

As of 2026-06-12 these focused checks pass locally for the new selector and
dataloader slice. The canonical real selector JSON files remain placeholders
until real Kaggle train/validation shards are available locally and the
permission-gated canonical generation commands are run with CRC validation.

Narrow `model_loss_train_step_ready` verification checks:

```bash
.venv/bin/ruff format src/eqvae/models/non_equivariant_vae.py \
  src/eqvae/losses src/eqvae/training src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py \
  tests/test_vae_loss.py tests/test_train_step.py \
  tests/test_optimizer_groups.py tests/test_compile_precision_smoke.py

.venv/bin/ruff check src/eqvae/models/non_equivariant_vae.py \
  src/eqvae/losses src/eqvae/training src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py \
  tests/test_vae_loss.py tests/test_train_step.py \
  tests/test_optimizer_groups.py tests/test_compile_precision_smoke.py

.venv/bin/basedpyright src/eqvae/models/non_equivariant_vae.py \
  src/eqvae/losses src/eqvae/training src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py \
  tests/test_vae_loss.py tests/test_train_step.py \
  tests/test_optimizer_groups.py tests/test_compile_precision_smoke.py

PYTHONPATH=src .venv/bin/pytest \
  tests/test_vae_loss.py tests/test_train_step.py \
  tests/test_optimizer_groups.py tests/test_compile_precision_smoke.py

PYTHONPATH=src .venv/bin/python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --output-dir /tmp/eqvae-local-model-loss-train-step \
  --run-name spec0001_cpu_model_loss_train_step \
  --max-benchmark-rows 1 \
  --warmup-steps 1 \
  --measured-steps 1 \
  --model-loss-train-step
```

The final command writes
`/tmp/eqvae-local-model-loss-train-step/benchmark/model_loss_train_step.json`
with `status = "local_pass"` and `full_run_eligible = false`.

Narrow `corruption_ready` implementation checks:

```bash
.venv/bin/ruff format src/eqvae/corruption src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py tests/test_stain_corruptor.py

.venv/bin/ruff check src/eqvae/corruption src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py tests/test_stain_corruptor.py

.venv/bin/basedpyright src/eqvae/corruption src/eqvae/benchmarking \
  src/eqvae/cli/benchmark_runtime.py tests/test_stain_corruptor.py

PYTHONPATH=src .venv/bin/pytest tests/test_stain_corruptor.py

PYTHONPATH=src .venv/bin/python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --output-dir /tmp/eqvae-local-stain-corruptor-qa \
  --run-name spec0001_cpu_stain_corruptor_qa \
  --stain-corruptor-qa
```

The final command must write
`/tmp/eqvae-local-stain-corruptor-qa/benchmark/stain_corruptor_qa.json` with
`status = "local_pass"` and `full_run_eligible = false` before the corruptor is
eligible for training integration. This local QA does not replace the required
fixed real 25-patch visual QA before the first Kaggle baseline run.

Unit and contract tests:

```bash
PYTHONPATH=src uv run --locked --no-sync pytest \
  tests/test_patch_shards.py \
  tests/test_stain_corruptor.py \
  tests/test_activation_policy.py \
  tests/test_translatable_vae_shapes.py \
  tests/test_vae_loss.py \
  tests/test_metrics_artifacts.py \
  tests/test_banned_operations.py
```

Torch compile and precision smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync pytest tests/test_compile_precision_smoke.py
```

Local CPU synthetic smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.smoke \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --batch-size 2 \
  --compile inductor \
  --dtype float32
```

Local CPU float16 smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.smoke \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --batch-size 1 \
  --max-steps 1 \
  --dtype float16
```

Debug train from scratch:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --output-dir runs/local/spec0001-debug \
  --run-name spec0001_cpu_debug \
  --max-train-steps 8 \
  --max-val-steps 2 \
  --save-every-steps 4
```

Resume from midpoint checkpoint:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --resume runs/local/spec0001-debug/checkpoints/step_000004.pt \
  --output-dir runs/local/spec0001-resume \
  --run-name spec0001_cpu_resume \
  --max-train-steps 10 \
  --max-val-steps 2
```

Evaluator and summaries:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.evaluate \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --checkpoint runs/local/spec0001-debug/checkpoints/step_000008.pt \
  --data synthetic \
  --split validation \
  --max-samples 32 \
  --output-dir runs/local/spec0001-eval
```

Artifact generation:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.artifacts \
  --eval-dir runs/local/spec0001-eval \
  --fixed-count 25 \
  --angles=-90,-45,0,45,90 \
  --output-dir runs/local/spec0001-artifacts
```

Local synthetic benchmark schema smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.model_count \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --output runs/local/spec0001-runtime-benchmark/benchmark/model_count.json

PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --output-dir runs/local/spec0001-runtime-benchmark \
  --run-name spec0001_cpu_runtime_benchmark \
  --max-benchmark-rows 2 \
  --warmup-steps 1 \
  --measured-steps 2 \
  --dataloader-pretest
```

Kaggle local scaffold checks:

```bash
./scripts/kaggle_kernel.sh build
./scripts/kaggle_kernel.sh validate
bash -n scripts/kaggle_kernel.sh
python3 -m json.tool kaggle/kernels/non_eq_vae_debug/kernel-metadata.json
```

Kaggle debug command that the script kernel must run after implementation:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working \
  --run-name non_eq_vae_spec0001_kaggle_debug \
  --max-train-steps 200 \
  --max-val-steps 20 \
  --save-every-steps 100
```

Kaggle runtime benchmark command that the script kernel must run before the
first full run:

```bash
python -m eqvae.cli.model_count \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --output /kaggle/working/runtime_benchmark/benchmark/model_count.json

python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working/runtime_benchmark \
  --run-name non_eq_vae_spec0001_runtime_benchmark
```

Kaggle selected-runtime debug command that must run after
`benchmark/selected_runtime.json` is written:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
  --runtime-config /kaggle/working/runtime_benchmark/benchmark/selected_runtime.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working/selected_runtime_debug \
  --run-name non_eq_vae_spec0001_selected_runtime_debug \
  --max-train-steps 200 \
  --max-val-steps 20 \
  --save-every-steps 100
```

Kaggle tiny-overfit command that must pass before the first 10-epoch run:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
  --runtime-config /kaggle/working/runtime_benchmark/benchmark/selected_runtime.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --fixed-train-patches configs/spec0001/fixed_32_train_overfit_patches.json \
  --output-dir /kaggle/working/tiny_overfit \
  --run-name non_eq_vae_spec0001_tiny_overfit \
  --max-train-steps 300 \
  --max-val-steps 20 \
  --save-every-steps 100
```

Permission-gated remote check, not required for local implementation acceptance:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
```

Only run the remote push after all local commands pass and the user explicitly
approves the remote write.

## Narrow Model/Loss Train-Step Acceptance

The `model_loss_train_step_ready` slice is accepted only while:

1. the focused `model_loss_train_step_ready` verification commands
   above pass;
2. `./scripts/python_quality.sh` passes for the production Python scope;
3. `NonEquivariantVAE.forward()` exposes the locked fields and supports
   explicit `eps` without hidden latent RNG consumption;
4. sampling and KL use `logvar_clamped`, while telemetry records raw and clamped
   logvar summaries plus `logvar_clamp_count`, and the writer rejects
   `objective.logvar_clamp` values that drift from the implementation
   constants;
5. loss tests verify global L1, per-image-mean SSIM loss, global KL, and beta
   scheduling with the zero-based `optimizer_step_index`;
6. optimizer-group tests verify every trainable parameter appears exactly once
   and all gate parameters are in `gate_no_decay`;
7. the local train-step pre-test writes
   `benchmark/model_loss_train_step.json` with `status = "local_pass"`,
   `benchmark_kind = "local_synthetic_model_loss_train_step"`,
   `benchmark_source = "local_cpu_synthetic_train_step"`,
   `corruption_strategy = "identity_clean_no_corruption"`,
   `full_run_eligible = false`, matching effective config/model-count hashes,
   model-count `architecture_id` and `topology_version`,
   finite losses, finite gradient/update telemetry, nonzero grad/update tensor
   counts, `first_step_update_scope = "zero_head_final_rgb_head_smoke"`, and
   zero-head forward proof;
8. optional local CPU compile/fp16 smoke outcomes are either `local_pass` or
   `skipped_unsupported` with deterministic `failure_kind` metadata.
9. `--model-loss-train-step` does not write, update, or replace
   `benchmark/selected_runtime.json`.

This narrow acceptance does not require HED corruption, checkpoint/resume,
evaluator/artifact writers, Kaggle payload changes, Kaggle remote execution, or
runtime selection.

## Full Local Baseline Acceptance Criteria

The broad local baseline implementation is complete only when its later
authorized slices have been implemented and:

1. all verification commands above pass;
2. model construction is generated from the locked layer/channel schedule;
3. banned-operation checks reject FSQ, PixelShuffle, nearest upsampling, 1x1
   convs, learned grouped/depthwise convs, attention blocks, `BatchNorm2d`,
   `LayerNorm`, and arbitrary representation-breaking normalization, while
   requiring baseline `GroupNorm` in hidden/projection blocks;
4. the data loader validates binary header fields, optional or required CRC
   status, shape, dtype, patch count, file size, required CSV columns, optional
   `idx`, canonical `file_index`, `row_index`, `sample_id`, and
   train/validation WSI non-overlap;
5. the split validator checks exact train/validation patch counts, exact
   train/validation WSI counts, zero overlap with
   `docs/data/ubc_ocean_masked_holdout_ids.csv`, and non-TMA status whenever
   official `train.csv` metadata is available;
6. synthetic data tests do not require network, Kaggle, or GPU access;
7. stain-corruptor tests verify scikit-compatible HED/RGB round-trip convention,
   channel-first/channel-last oracle agreement, per-channel H/E/residual-axis
   perturbation semantics, fixed-seed reproducibility, rank-move reproducibility
   with rank excluded from the semantic seed, no RNG consumption in clean
   validation mode, finite outputs, output-domain clamp telemetry, metadata
   schema validation, synthetic visual QA artifact generation, and fixed real
   25-patch visual QA before the first Kaggle baseline run;
8. CPU smoke tests instantiate data, model, corruption, loss, optimizer,
   evaluator, and artifact writers. The narrow `kaggle_smoke_ready` path
   specifically tests metadata-carrying UBC-format batches and writes
   non-promotable `benchmark/kaggle_smoke.json` after one local synthetic train
   step and one clean-validation batch. The narrow `kaggle_setup_smoke_ready`
   path specifically tests Kaggle single-file packaging/import/artifact
   plumbing without real dataset attachment and writes non-promotable
   `benchmark/kaggle_setup_smoke.json`;
9. compile/precision smoke tests cover `torch.compile`, output shapes, and the
   configured float16 path without requiring a GPU;
10. model tests verify that the final RGB head is zero-initialized, that the
    initial reconstruction is all zeros within tolerance, and that the model
    forward path contains no final `tanh`, sigmoid, or clamp;
11. the narrow model/loss local train-step slice writes
    `benchmark/model_loss_train_step.json` with `status = "local_pass"`,
    `full_run_eligible = false`, explicit `eps` control, clamped-logvar
    sampling/KL telemetry, finite `L1 + 0.1 * (1 - SSIM) + beta * KL`
    components, semantic AdamW group coverage, nonzero finite first-step
    final-head gradient/update evidence, and identity-clean corruption
    provenance;
12. debug training completes from scratch and writes metrics, config, checkpoint,
    and RNG state;
13. resume training restores checkpoint, optimizer, scheduler/beta state, and RNG
    state;
14. AMP skipped-step behavior is tested or exercised so skipped steps do not
    advance optimizer-step counters, LR/beta schedules, validation, or
    checkpoint cadence;
15. the runtime benchmark CLI exists, runs on a tiny local synthetic budget, and
    writes `benchmark/runtime_matrix.csv`, `benchmark/selected_runtime.json`,
    `benchmark/model_count.json`, `benchmark/dataloader_matrix.csv`,
    `benchmark/numerical_checks.csv`, `metrics/gate_health.csv`, and
    `benchmark/gate_health_summary.json` with the expected schemas without
    requiring GPU or network access. Schema-only artifacts use `schema_pass`;
    measured local CPU pre-test artifacts may use `local_pass`; both must keep
    `full_run_eligible = false`;
16. checkpoint retention keeps `best_model.pt`, the final checkpoint, and the
    latest four interval checkpoints;
17. evaluator writes per-image SSIM, MAE, MSE, PSNR and summary mean/std/`n`
    separately for `eval_clean` and fixed-seed `eval_corrupted`;
18. artifact writer emits metric boxplots, dashboard, fixed 25-patch
    reconstructions, rotated-input grids, rotated-input versus latent grids, and
    latent visualization placeholders or outputs;
19. offline selector tests use synthetic fixtures; the real
    `fixed_25_validation_patches.json` and
    `fixed_32_train_overfit_patches.json` generation are data-access steps that
    must be run on the real validation/train CSVs before Kaggle debug/full runs;
20. the fixed 25-patch config contains exactly 5 validation patches per label and
    all future qualitative commands read it rather than resampling; the fixed
    32-train tiny-overfit config contains exactly 32 train patches and no
    validation rows;
21. Kaggle debug kernel runs bundled repo code through the CLI, not notebook
    source or a GitHub-linked notebook;
22. `scripts/kaggle_kernel.sh push` rejects wrong dataset slugs, historical FSQ
    output sources, internet-enabled metadata, missing payloads or stale
    embedded payloads, placeholder launchers, missing or wrong benchmark
    `machine_shape`, setup-smoke kernels with any Kaggle source attachment,
    any nonempty Kaggle source list without `KAGGLE_FULL_DATASET_CONFIRMED=1`,
    and real-data spec 0001 debug kernels with nonempty `competition_sources`,
    `kernel_sources`, or `model_sources`. The future synthetic timing kernel has
    its own guard branch: it must require empty source lists, GPU/T4 metadata,
    `KAGGLE_SYNTHETIC_TIMING_READY = True`, generated `/kaggle/working`
    synthetic data provenance, non-promotable synthetic artifact names, and
    absence of `benchmark/selected_runtime.json`;
23. runs without the sealed masked-WSI test shard are labeled
    train/validation-only and excluded from final paper claims;
24. `CURRENT.md`, `docs/specs/README.md`, and relevant workflow docs are updated
    with implementation status and verification results.

## Full Kaggle Run Acceptance Criteria

The first 10-epoch Kaggle baseline is not ready until:

1. local implementation acceptance passes;
2. the read-only Kaggle API preflight
   `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes its
   required auth/list/status/logs/dataset checks. If the quota endpoint warns,
   verify GPU quota in the Kaggle web UI before approving the remote benchmark
   push and record the warning in the run notes;
3. the user explicitly approves the remote Kaggle write/run;
4. the model-count command writes `benchmark/model_count.json` with status
   `pass` before runtime selection;
5. the short Kaggle runtime benchmark writes `benchmark/runtime_matrix.csv` and
   `benchmark/selected_runtime.json`, including AMP off/on, compile off/on,
   named precision-policy, and branchless-versus-indexed corruption evidence
   for single and dual T4;
6. the benchmark writes `benchmark/dataloader_matrix.csv` and
   `benchmark/numerical_checks.csv`; the selected row must have dataloader,
   numerical-check, and gate-health status `pass`;
7. the selected single/dual T4, per-device/global batch, AMP, compile,
   precision policy, and corruption strategy are copied into the resolved
   full-run config;
8. the gate-health benchmark writes `metrics/gate_health.csv` and
   `benchmark/gate_health_summary.json` without non-finite gate values,
   persistent near-total saturation, or unexplained near-zero hidden-block
   output/input RMS;
9. a selected-runtime real-data debug run completes 200 train steps, runs both
   `eval_clean` and fixed-seed `eval_corrupted`, writes at least one checkpoint,
   resumes once from that checkpoint, and emits nonblank fixed-patch artifacts;
10. a selected-runtime tiny-overfit run on 32 fixed real patches writes
   `benchmark/tiny_overfit_summary.json` with finite losses, final smoothed L1
   and reconstruction loss at least 5 percent below their initial smoothed
   values, PSNR or SSIM improved over the zero-head baseline, no pathological
   `logvar_clamp_count`, and gate-health status not worse than `warn`;
11. the baseline run uses the selected runtime config, validates/checkpoints every
   half epoch, and keeps the declared checkpoint retention.

## Open Questions And Gates

Implementation-relock blockers:

1. Final channel/future-field schedule: the Conv2d baseline count is now
   recorded in this spec. Before the `SO(2)` implementation is locked, the
   steerable basis/count tool must show the future field multiplicities can stay
   at or below the Conv2d baseline's learned parameter count without exceeding
   the Kaggle memory budget.
2. Data/metrics implementation status: the narrow `data_metrics_ready`
   exception is recorded and locally verified by focused patch-shard,
   split-validation, reconstruction-metric, Ruff, BasedPyright, and pytest
   checks. This is not broad benchmark readiness and does not unlock corruption,
   training, real selector generation, Kaggle remote work, or paper claims.
3. HED/stain corruption implementation status: the narrow `corruption_ready`
   local correctness/QA slice is implemented and verified with synthetic QA, but
   corruption still needs training integration, selected-runtime corruption
   checks, and fixed real 25-patch visual QA before the first Kaggle baseline
   run. The first real training run must use this corruptor.
4. Kaggle smoke status: the narrow `kaggle_smoke_ready` launcher is implemented
   for a tiny real-data smoke only, but the first remote version failed before
   import because the sibling payload directory was not uploaded. The real-data
   launcher has since been migrated locally to embedded single-file packaging
   and an upload-simulation import test, but accepted real-data smoke evidence
   still requires an intentional source-attachment push guarded by
   `KAGGLE_FULL_DATASET_CONFIRMED=1`. The narrow
   `kaggle_setup_smoke_ready` launcher is implemented for setup-only
   packaging/API/artifact checks with no dataset attachment. Both paths may be
   pushed/run only after explicit user permission and read-only API preflight;
   neither satisfies selected-runtime debug, runtime benchmark, tiny-overfit,
   fixed real visual QA, or full-run acceptance. A real-data `smoke_pass` result
   is accepted only from a hardened payload that enforces caps, actual
   corruption, T4 CUDA for real-data smoke, seeded initialization, payload
   provenance, and non-promotable artifact semantics.
5. Kaggle metadata enforcement: the workflow now records
   `machine_shape = "NvidiaTeslaT4"` for the T4 benchmark kernel. The
   implementation must enforce that metadata value before remote push and fail
   `dual_t4_ddp` benchmark rows unless runtime CUDA/DDP telemetry proves two T4
   devices and two ranks.
6. Synthetic timing pretest status: the no-dataset 2 GiB-scale generated-binary
   Kaggle pretest contract is locked for implementation only. It may rank and
   prune candidate rows for the real-data benchmark, including single-vs-dual
   T4 comparisons by projected epoch time, but cannot satisfy runtime
   selection, real dataloader, numerical-check, selected-runtime debug,
   tiny-overfit, full-run, or paper-evidence gates.
7. Final clean-context adversarial spec review must pass after the edits and
   implementation count, metadata, import, and quality routes are integrated.
8. Strict quality route must follow spec 0002's production-boundary decision:
   extract any needed behavior into `src/eqvae`, keep the removed
   `pytorch-msssim` dependency out of active dependency truth, exclude
   historical `src/nn` from production Ruff/BasedPyright scopes, and forbid
   active code from importing `src.nn`. Keep global Ruff/BasedPyright strictness
   intact.
9. JSON config/dependency policy must remain locked, or a later spec must
   explicitly justify changing config format and dependencies.
10. Package/import policy must be locked enough that the verification commands
   import `eqvae` without dependency sync.
11. Fixed selector schema/dataloader semantics are locked and locally verified
   by `fixed_selectors_dataloader_ready`; the real fixed-25 and fixed-32 JSON
   generation remains a data-access step that must be run against the real
   Kaggle train/validation shards before Kaggle debug/tiny/full runs.
   Baseline rotated/latent artifact semantics must remain exactly as specified
   above, or be revised before implementation.

Full-run blockers after implementation:

1. Runtime target: after the Kaggle benchmark matrix, should the full run use
   single GPU or dual T4 DDP, should AMP and/or `torch.compile` be enabled, and
   what are the selected precision policy, corruption execution strategy, and
   per-device/global batch size?
2. The selected runtime must be written to `benchmark/selected_runtime.json` and
   the resolved baseline config before the first 10-epoch Kaggle run.
3. Gate-health target: the short benchmark/debug path must show that learned
   gate `a,b` parameters do not create non-finite values, persistent saturation,
   or hidden-block collapse before the first full run.
4. Data/quality target: dataloader throughput, paired numerical checks,
   selected-runtime debug, checkpoint/resume, and tiny-overfit summaries must all
   pass before the first 10-epoch Kaggle run.

## Known Risks

- The future radial gate can suppress vector/irrep copies if initialized poorly.
  Initialize it with the accepted Swish-like `a=1,b=0` convention and test
  gradient flow before the `SO(2)` model depends on it.
- CPU float16 behavior can differ from Kaggle GPU float16 behavior. Local smoke
  checks are a contract test, not a replacement for Kaggle debug training.
- The target sealed test slug may need to change before upload. If it changes,
  update this spec and all configs before making final claims.
- The baseline still receives ordinary Conv2d kernels, so fairness depends on
  keeping every other degree of freedom matched when the steerable model is
  implemented.

## Adversarial Checks Before Implementation PR Completion

- Does any operation violate the future continuous `SO(2)` translation path?
- Does the VAE objective accidentally omit KL or use the wrong reduction?
- Does the training forward sample from unclamped `logvar` or hide latent noise
  so paired numerical checks cannot reuse the same `eps`?
- Does the beta schedule advance on skipped optimizer steps or use a different
  step index than the logged train-step artifact?
- Does corruption randomness differ between comparison branches?
- Does the HED PyTorch implementation match the scikit-image oracle under the
  locked RGB-domain convention without calling scikit-image in the runtime path?
- Does any code treat the third HED residual axis as biological DAB instead of a
  tiny anti-signature residual jitter?
- Does the corruption seed accidentally depend on rank, batch order, physical
  `file_index`, global Torch RNG, or clean-validation calls?
- Do metric scripts include `n` and run on the same evaluation images?
- Do qualitative artifacts use the same fixed patch IDs for both future models?
- Does any split leak WSI, patient, site, or masked-holdout information?
- Does the baseline receive more tuning or validation access than the future
  equivariant model?
- Does the Kaggle launcher import repo code instead of embedding a notebook copy?

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/behavior_inventory_kaggle.md`
- `docs/equivariant_vae_transition_plan.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/decisions/0001-continuous-so2-scope.md`
- `docs/decisions/0002-normal-vae-baseline.md`
- `docs/specs/0002-strict-python-quality-gate.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
