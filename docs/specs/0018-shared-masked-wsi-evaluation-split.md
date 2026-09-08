# Spec 0018: Shared Masked-WSI Evaluation Split And Task Manifests

Status: implemented
Implementation readiness: accepted after independent feasibility, leakage, and implementation review
Owner/workstream: held-out evaluation data
Last updated: 2026-08-24

## Purpose

Turn the verified 152-WSI masked atlas into one leakage-safe WSI assignment
shared by two downstream tasks:

1. WSI ovarian-cancer subtype classification and primary VAE evaluation using
   only image-derived Otsu-selected coordinates. Primary VAE metrics use only
   the 23 sealed-test WSIs, matching the classifier test population.
2. High-purity annotated-patch tissue classification using the supplemental
   masks.

The split is chosen from labels and atlas metadata before either trained VAE is
evaluated on any of these 152 WSIs. Existing nonmasked validation/fixed-25
outputs are unrelated and predate this cohort. Patch abundance is not treated as independent evidence;
acceptance is driven first by WSI counts and then by usable patch coverage.

## Non-Goals

- No patch pixels, binary shards, embeddings, classifier, model inference, or
  Kaggle upload/run.
- No model-output-dependent split selection or protocol tuning.
- No claim that incomplete masks define background, normal tissue, exhaustive
  segmentation, or general tissue classification outside high-purity painted
  regions.
- No requirement that every diagnosis-by-tissue combination appear in every
  split. LGSC-necrosis is absent and only two MC-necrosis WSIs exist.
- No duplication or oversampling in validation or test manifests.

## Inputs And Data Contract

- Verified atlas:
  `runs/kaggle/ubc_ocean_test_atlas_v3/dataset/ubc_ocean_test_atlas.csv`.
  Expected SHA-256:
  `9258a98f9512e1a1e04e09ea8dbca28463009fd3276345daed5b2d8119692557`.
- Canonical cohort metadata:
  `docs/data/ubc_ocean_masked_holdout_ids.csv` with exactly 152 unique
  `image_id,label,is_updated_image_id` rows; expected SHA-256
  `9fa03421694f77d3964097da2388beca45b31ce3e50da0b170c13781b86dea81`.
- Frozen split SHA-256:
  `216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c`.
- Atlas labels map `CC=0, EC=1, HGSC=2, LGSC=3, MC=4` and must agree with the
  cohort metadata.
- Atlas rows are preserved in `wsi_id,y,x` order.
- Cancer/AE coordinates are atlas rows whose `selection_source` is `otsu` or
  `mask+otsu`. Mask-only coordinates are excluded because human annotation must
  not guide primary diagnosis or autoencoder sampling.
- Tissue coordinates require `annotated_fraction >= 0.10` and
  `max(tumor_fraction, stroma_fraction, necrosis_fraction) /
  annotated_fraction >= 0.95`. The largest fraction supplies the tissue label.
- The WSI assignment is deliberately mask-stratified so the shared split is
  usable for the rare tissue task. This is disclosed design-time stratification,
  not model-output leakage or a model feature. Cancer/AE coordinate inclusion
  remains mask-independent because it uses the complete image-derived Otsu set.
- Split selection used only diagnosis, technical-update status, atlas selection
  metadata, and mask-derived tissue availability/counts. It did not read images,
  patches, embeddings, checkpoints, or model outputs.
- Split sizes are 106 train, 23 validation, and 23 sealed test WSIs.
- A one-time constrained integer optimization established feasibility and
  balanced Otsu/tissue volumes. The reviewed and committed split CSV is the
  permanent authority; the production generator validates it and does not
  search for or silently replace test WSIs.

## Outputs And Acceptance Artifacts

- Readable implementation:
  `src/eqvae/cli/generate_ubc_eval_manifests.py`.
- Small canonical artifacts:
  - `docs/data/ubc_ocean_eval_wsi_split.csv`;
  - `docs/data/ubc_ocean_eval_split_audit.json`.
- Ignored, regenerable full manifests under
  `runs/local/ubc_ocean_eval_manifests/`:
  - `cancer_ae_patch_manifest.csv`;
  - `tissue_patch_manifest.csv`.
- `wsi_split.csv` columns:
  `wsi_id,diagnosis_label,diagnosis_index,is_updated_image_id,split`.
- Cancer/AE manifest columns:
  `atlas_row_index,wsi_id,diagnosis_label,diagnosis_index,x,y,split`.
  Mask metadata is deliberately absent.
- Tissue manifest columns:
  `atlas_row_index,wsi_id,diagnosis_label,diagnosis_index,x,y,split,
  tissue_label,annotated_fraction,dominant_fraction,purity,tumor_fraction,
  stroma_fraction,necrosis_fraction`.
- `atlas_row_index` is zero-based over data rows, excluding the CSV header.
- CSVs use UTF-8, RFC-4180 quoting through Python's `csv` writer, comma
  delimiters, `\n` line endings, and `.17g` for the derived purity value. JSON
  uses UTF-8, sorted keys, two-space indentation, and a final newline.
- Audit JSON records input/output hashes, row counts, split constraints, WSI
  diagnosis counts, updated-image counts, task patch totals, tissue-positive
  WSI counts, per-WSI necrosis counts, tissue patch counts,
  diagnosis-by-tissue cross-tabs, every locked floor/quota, and every acceptance
  result.

## Related Requirements And Evidence

- `GOAL.md`: compare the frozen normal and continuous-`SO(2)` VAEs on matched
  held-out evidence and downstream usefulness.
- `docs/repo_goal_and_requirements.md`: WSI/patient/site separation where
  metadata permits and final sample counts for reported metrics.
- `docs/equivariant_vae_transition_plan.md`: export embeddings and run a linear
  or small-MLP probe without WSI leakage.
- Spec 0017: authoritative atlas semantics and mask-fraction definitions.

## Workflow Contract

1. Validate the exact atlas, cohort, and canonical-split hashes, then validate
   schema, order, labels, fractions, and exact 152-WSI coverage. Structurally
   valid alternative split bytes are rejected so sealed test IDs cannot drift.
2. Aggregate Otsu and high-purity tissue availability per WSI in one streaming
   atlas pass.
3. Load the committed split and validate these exact diagnosis quotas:

   | Split | CC | EC | HGSC | LGSC | MC | Total |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | train | 23 | 24 | 40 | 11 | 8 | 106 |
   | validation | 5 | 6 | 8 | 2 | 2 | 23 |
   | test | 5 | 6 | 8 | 2 | 2 | 23 |

4. Reject the canonical split unless:
   - updated-image counts are exactly 12/3/3;
   - all 152 WSIs are tumor-positive, audited as an input invariant;
   - validation and test each contain at least 20 stroma-positive WSIs;
   - necrosis-positive WSI counts are exactly 17/5/5;
   - every validation/test necrosis-positive WSI has at least 100 qualifying
     necrosis patches;
   - validation and test each span necrosis across CC, EC, and HGSC;
   - one MC-necrosis WSI is in train and the other is in test;
   - train has at least 5,000 necrosis patches;
   - validation and test each have at least 50,000 tumor, 5,000 stroma, and
     1,000 necrosis patches.
5. Stream the atlas a second time and write both coordinate manifests in atlas
   order. Every row inherits its WSI split. No patch randomization or duplicate
   row is introduced.
6. Write both manifests to temporary files, hash and validate them, atomically
   replace their final paths, and publish the audit JSON last as the completion
   marker. The audit embeds hashes for the inputs, canonical split, generator,
   and manifests, but not a self-referential hash of itself.
7. The test assignment and both coordinate manifests are sealed before any VAE
   or classifier output on the 152 masked WSIs is inspected. Both frozen
   encoders later consume identical coordinates and order within each task.

## Config Contract

The CLI accepts explicit atlas, cohort, canonical-split, audit-output, and
manifest-output paths. Defaults are the repo paths in this spec. Locked values:

- annotated coverage: `0.10`;
- tissue purity: `0.95`;
- split names/order: `train,validation,test`;
- diagnosis quotas and acceptance floors: exactly those above.

No threshold is inferred from the observed model outputs. Changing a quota,
threshold, or constraint requires relocking this spec and regenerating every
downstream manifest.

## Acceptance Criteria

1. The canonical split contains each of the 152 masked WSIs exactly once and has
   no overlap across splits. Spec 0017 and
   `docs/behavior_inventory_kaggle.md` remain the provenance evidence that this
   canonical masked cohort has zero overlap with the existing 322/39 nonmasked
   autoencoder train/validation cohorts; their unavailable real shard CSVs are
   not pretended to be executable inputs here.
2. All diagnosis, updated-image, tissue-positive-WSI, tissue-patch, and rare-
   combination constraints in the workflow contract pass.
3. Cancer/AE rows equal the complete atlas subset whose source is in
   `{otsu, mask+otsu}`: exactly 1,750,221 rows spanning all 152 WSIs. Mask-only
   rows cannot enter that manifest and no mask-derived column is exported.
4. Tissue rows satisfy the coverage/purity rule computed from authoritative
   fractions, not only the summary `mask_label`: exactly 666,807 rows before
   any later task-specific training sampling.
5. Both manifests preserve atlas order, contain no duplicate coordinates, and
   inherit the canonical WSI split exactly.
6. Input and output SHA-256 hashes, row counts, per-split diagnosis counts,
   per-tissue `n_WSI` and `n_patch`, and diagnosis-by-tissue cross-tabs are
   recorded in the audit.
7. A second run from the identical atlas, cohort, canonical split, and generator
   produces byte-identical manifest outputs.
8. Focused tests, `./scripts/python_quality.sh`, and
   `./scripts/agent_preflight.sh` pass.

## Tests And Verification Commands

```bash
.venv/bin/python -m pytest -q tests/test_generate_ubc_eval_manifests.py
.venv/bin/python -m eqvae.cli.generate_ubc_eval_manifests
./scripts/python_quality.sh
./scripts/agent_preflight.sh
```

Tests cover invalid atlas/cohort/labels/order/fractions, quota enforcement,
acceptance-floor failures, Otsu-only cancer selection, authoritative tissue
purity, shared split inheritance, deterministic bytes, atomic publication,
hashes, and real-atlas acceptance.

## Implementation Blockers

None. Independent reviews confirmed joint numeric feasibility, the required
disclosure boundary, frozen split-byte validation, and real-atlas manifest
materialization. WSI assignment is mask-stratified, while primary cancer/AE
coordinate selection and exported features remain mask-independent.

## Verification Result

- Canonical split: 106 train, 23 validation, and 23 sealed-test WSIs; SHA-256
  `216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c`.
- Cancer/AE manifest: 1,750,221 rows; SHA-256
  `710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c`.
- Tissue manifest: 666,807 rows; SHA-256
  `7e2f61c492167129911e73515c4d82ef9291fd89b920dc191be922a44f30ac92`.
- Focused manifest suite: 13 passed, including full-atlas materialization,
  sealed output-hash verification, deterministic reruns, constraint mutations,
  and audit-last failure handling.
- Repository quality gate: formatting and Ruff passed; 846 tests passed, one
  expected GPU-only test skipped; BasedPyright reported zero errors.

## Known Risks

- Millions of correlated patches do not increase the independent WSI sample
  size. Test LGSC and MC each have only two WSIs, and validation/test necrosis
  have only five positive WSIs. Per-class conclusions remain exploratory and
  must report `n_WSI`.
- Supplemental masks are incomplete and the 95% rule deliberately excludes
  mixed/boundary morphology. The tissue task is high-purity annotated-patch
  classification, not segmentation or exhaustive tissue classification.
- Full raw patches project far beyond local and Kaggle output limits. This spec
  emits coordinates only; later extraction should stream directly to model
  evaluation/embeddings or use separately sized shards.
- The pinned atlas is currently a verified local download of private Kaggle
  kernel `maximusshtefan/eqvae-ubc-ocean-test-atlas` version 3, not a tracked
  repo file. The audit pins its byte hash and generator provenance. Before any
  downstream remote run, publish the atlas/manifests as an immutable private
  derived dataset and record its version; do not rely on an unversioned latest
  kernel output.
- Label- and mask-stratified assignment makes a balanced test set but cannot
  manufacture missing diagnosis-tissue combinations or patient/site metadata
  absent from UBC-OCEAN.

## Adversarial Checks

- Inject one mask-only row into the cancer manifest and require failure.
- Move one WSI between splits or duplicate one coordinate and require failure.
- Recompute purity from fractions and reject a forged `mask_label`.
- Concentrate necrosis in fewer WSIs while keeping patch totals high and require
  failure.
- Balance patch counts while breaking diagnosis or updated-image quotas and
  require failure.
- Change one input or output byte and require the recorded hash to disagree.
- Confirm no model checkpoint, embedding, or prediction is read anywhere in the
  split workflow.

## Open Questions

- Whether the downstream classifier protocol uses one fixed validation split or
  additional WSI-grouped cross-validation inside the 129 development WSIs is a
  later evaluation-spec decision. It does not change the sealed test WSIs or
  the shared task manifests created here.

## Related Files

- `docs/specs/0017-masked-holdout-test-dataset.md`
- `docs/data/ubc_ocean_masked_holdout_ids.csv`
- `kaggle/generate_ubc_ocean_test.py`
- `src/eqvae/data/splits.py`
