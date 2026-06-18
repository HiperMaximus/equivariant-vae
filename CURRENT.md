# Current Repository Status

Last updated: 2026-06-18

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE using a repo-owned,
   compile-compatible implementation, with `escnn` as a reference.

The current task is the Kaggle no-dataset synthetic timing evidence handoff for
the translatable normal VAE baseline. The local implementation is committed and
pushed to GitHub, and Kaggle remote versions 1, 2, 3, and 4 completed
successfully as non-promotable synthetic timing evidence. Version 1 used the
historical compact 0.81 GB profile; versions 2 through 4 use the current
2 GiB-scale profile. Version 3 superseded version 2 for corrected DDP runtime
proof and `drop_last = false` projection fields. Version 4 is the current
5-warmup/25-measured repeat-shortlist evidence.
Clean-context adversarial
subagent reviews were run on 2026-06-05, 2026-06-10, 2026-06-11, and a focused
scaffold-readiness check on 2026-06-12. The 2026-06-11 passes
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
RNG. The Kaggle benchmark must now select the fastest safe precision policy
among `amp_off_fp32`, `amp_conservative`, and `amp_scalar_gate_relaxed`, must
measure whether `branchless_all` or `indexed_masked` corruption is faster
without breaking compile stability or RNG semantics, must include paired
numerical checks and dataloader throughput checks, and must include gate-health
telemetry for learned gate `a,b` parameters so saturation/dead-channel behavior
is caught before full training. It is not final-paper-claim-ready until the
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

Immediate next action: design the next non-promotable benchmark slice that adds
real training-step components and/or real-data loader confirmation without
treating synthetic timing as selected-runtime evidence. The v4 repeated
shortlist artifacts live under
`runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark`. Top v4 synthetic
repeat recommendations were
`dual_t4_ddp__bs8__amp_off_fp32__compile_off__branchless_all`
(`estimated_epoch_minutes = 1.312643`),
`single_visible_t4__bs4__amp_off_fp32__compile_off__branchless_all`
(`1.964479`), and
`single_visible_t4__bs32__amp_off_fp32__compile_off__branchless_all`
(`2.043706`). The v4 recommendation artifact marks the repeat-shortlist gate
as completed, but the artifacts remain non-promotable loader/H2D screening
evidence only; real runtime selection still requires real-data benchmarking and
the selected-runtime debug/full-run gates.
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

Decision notes live in `docs/decisions/`.
The review process lives in `docs/agentic_review_workflow.md`.

## No Longer Active

- Old conference-deadline planning is not part of the current route.
- Discrete rotation-group implementation work is not part of the current route.
- The thesis repo is not the active editing target for this phase.

## Next Concrete Steps

1. Inspect `runs/kaggle/synthetic_timing/benchmark` and decide which synthetic
   timing rows should seed the later real-data benchmark candidate set.
2. Use synthetic timing only to screen/order candidate rows for the later
   real-data benchmark. Real runtime selection still requires the real-data
   benchmark, selected-runtime debug, checkpoint/resume, tiny-overfit, and
   fixed real visual QA gates.
3. Decide the next narrow implementation slice for real-data benchmark
   candidate execution; any real-data Kaggle source attachment still requires
   explicit user approval plus `KAGGLE_FULL_DATASET_CONFIRMED=1`.
4. Continue the shared evaluation harness, future `SO(2)` count ceiling, and
   steerable model work only after the benchmark plumbing gates are no longer
   blocking the first real baseline run.

## Current Blockers

- Spec 0001 is reopened and not implementation-ready for broad work. Narrow
  local scaffold, topology-count, data/metrics, selector/dataloader, local
  benchmark pre-test, model/loss train-step, HED/stain corruption, Kaggle setup
  smoke, Kaggle capped-smoke source-delivery contracts, and the no-dataset
  synthetic timing kernel/guard/remote evidence now exist. The next blocking
  implementation slice is the later real-data benchmark candidate execution,
  after deciding how to use the non-promotable synthetic timing screen.
  Remaining implementation-relock blockers include future `SO(2)` count
  ceiling, real fixed validation/tiny-overfit selector generation,
  selected-runtime debug, checkpoint/resume, full evaluation/artifact writers,
  and final adversarial spec review after those routes are integrated.
- The first full Kaggle run remains blocked until synthetic timing screens the
  candidate rows, the real-data runtime benchmark selects single/dual T4,
  per-device/global batch, AMP, compile, precision policy, dataloader settings,
  and corruption strategy, and the dataloader throughput, paired numerical
  checks, gate-health summary, selected-runtime debug, checkpoint/resume,
  tiny-overfit, and fixed real visual QA gates pass.
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
