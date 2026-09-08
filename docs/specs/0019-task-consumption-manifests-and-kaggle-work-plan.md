# Spec 0019: Task Consumption Manifests And Kaggle Work Plan

Status: implemented
Implementation readiness: accepted after independent feasibility, adversarial,
and implementation review
Owner/workstream: held-out evaluation data
Last updated: 2026-08-24

## Purpose

Reduce the complete Spec 0018 candidate pools into two usable, deterministic,
WSI-leakage-safe datasets and a shared physical-read plan:

1. cancer-subtype classification and primary VAE evaluation, capped per WSI;
2. high-purity tissue classification, capped per WSI and tissue class.

The task selectors remain independent. Their union exists only to avoid reading
the same WSI coordinate twice in later Kaggle inference.

## Non-Goals

- No WSI reassignment: the frozen 106/23/23 split from Spec 0018 remains exact.
- No raw patch binaries, model inference, embeddings, classifier fitting,
  Kaggle dataset publication, or remote kernel launch.
- No claim that patch rows are independent cancer examples. Cancer diagnosis is
  WSI-level weak supervision; patch rows are inputs to later WSI aggregation.
- No embedding pooling/dtype, classifier, aggregation, equivariance-angle, or
  final metric decision. Those must be locked before test features are opened.
- No claim that five work shards have equal wall time. PNG decode cost also
  depends on WSI dimensions, compressed bytes, and deepest selected row.

## Inputs And Data Contract

- `runs/local/ubc_ocean_eval_manifests/cancer_ae_patch_manifest.csv`, exact
  Spec 0018 SHA-256
  `710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c`.
- `runs/local/ubc_ocean_eval_manifests/tissue_patch_manifest.csv`, exact SHA-256
  `7e2f61c492167129911e73515c4d82ef9291fd89b920dc191be922a44f30ac92`.
- `docs/data/ubc_ocean_eval_wsi_split.csv`, exact SHA-256
  `216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c`.
- Both input manifests are ordered by `atlas_row_index`, equivalently
  `wsi_id,y,x`, contain no duplicate coordinate, and inherit the same split.
- Cancer input contains no mask fields and remains mask-independent.
- Tissue input contains only patches meeting Spec 0018's 10% annotated and 95%
  dominant-class-purity thresholds.

## Selection Contract

### Cancer/AE

- Group by WSI and cap each group at `C=3000`.
- Use only the cancer/AE input; tissue membership cannot influence selection.
- If a group has `N <= C`, retain all rows.
- Otherwise retain the centered systematic ranks
  `index_j = ((2*j + 1) * N) // (2*C)` for `j=0..C-1`, where input rows are in
  authoritative `y,x` order and indices are zero-based.
- Final per-split counts are exactly 314,755 train, 68,045 validation, and
  67,138 test, totaling 449,938.
- The 67,138 cancer test rows are the primary VAE evaluation coordinates.

### Tissue

- Within each WSI, group independently by `tissue_label` and cap each group at
  `C=1000` using the same centered systematic-rank formula.
- Restore selected rows to authoritative atlas order after merging the three
  tissue groups.
- Final counts are:

  | Split | Tumor | Stroma | Necrosis | Total |
  | --- | ---: | ---: | ---: | ---: |
  | train | 102,188 | 37,356 | 5,671 | 145,215 |
  | validation | 22,255 | 7,809 | 1,275 | 31,339 |
  | test | 21,796 | 8,500 | 1,276 | 31,572 |

- Validation and test contain no duplication, oversampling, or resampling.
- Later training must be class- and WSI-aware; this spec does not choose that
  sampler or alter the acquired evidence.

### Spatial Coverage Evidence

The centered systematic selector is audited only for capped groups (`N>C`):
146 cancer WSI groups and 143 tissue WSI/class groups. For each group, derive
`xmin,xmax,ymin,ymax` from every candidate in that group. With `q=10`, map a
candidate coordinate to:

```text
dx = max(1, xmax - xmin + 1)
dy = max(1, ymax - ymin + 1)
bx = min(q - 1, ((x - xmin) * q) // dx)
by = min(q - 1, ((y - ymin) * q) // dy)
```

Coverage is the number of candidate-occupied `(bx,by)` cells containing at
least one selected patch divided by the number of candidate-occupied cells.
Aggregate mean is the unweighted macro mean across capped groups. The reviewed
exact results are:

- cancer mean `0.986048521387179`, minimum `70/75 = 14/15` at train WSI 37190;
- tissue mean `0.9781013992306213`, minimum `56/61` at train WSI 45185 tumor.

The audit must recompute these values and group counts. This is evidence against
raster-prefix bias, not a claim of uniform biological sampling.

## Outputs And Acceptance Artifacts

- Readable implementation:
  `src/eqvae/cli/generate_ubc_consumption_manifests.py`.
- Tests: `tests/test_generate_ubc_consumption_manifests.py`.
- Ignored regenerable output root:
  `runs/local/ubc_ocean_eval_consumption/`.
- Six task files, retaining their parent manifest schema:
  - `cancer_train.csv`, `cancer_validation.csv`, `cancer_test.csv`;
  - `tissue_train.csv`, `tissue_validation.csv`, `tissue_test.csv`.
- `union_patch_manifest.csv` columns:
  `atlas_row_index,wsi_id,diagnosis_label,diagnosis_index,x,y,split,
  cancer_ae_selected,tissue_selected,tissue_label`.
- Five `work_shards/run_XX_of_05.csv` files with the union schema.
- Small tracked artifacts:
  - `docs/data/ubc_ocean_eval_work_assignment.csv` with one row per WSI and
    columns `run_number,wsi_id,split,diagnosis_label,diagnosis_index,
    cancer_ae_patch_count,tissue_patch_count,union_patch_count,min_x,max_x,
    min_y,max_y,projected_raw_bytes`, ordered by `run_number,wsi_id`;
  - `docs/data/ubc_ocean_eval_consumption_audit.json`, written last.
- CSV and JSON byte conventions remain those in Spec 0018.

## Union And Work-Shard Contract

- Merge task selections by exact `atlas_row_index,wsi_id,x,y` identity.
- Exact overlap is 58,666 rows. Exact physical-read union is 599,398 rows:
  418,685 train, 90,311 validation, and 90,402 test.
- Tissue-only rows retain empty cancer membership; cancer consumers must reject
  them. Cancer-only rows contain no tissue label.
- Never force tissue-selected coordinates into cancer selection to increase
  overlap; that would make the cancer/VAE sample annotation-guided.
- Keep every WSI wholly in one work shard. Use the existing deterministic greedy
  contiguous whole-WSI partition rule over unique selected-row counts with five
  nonempty shards. Preserve WSI and `y,x` order inside each shard.
- Record per-shard task counts, split counts, WSI IDs, candidate bounding-box
  proxy, hashes, and projected raw bytes. Patch balance is a work allocation,
  not a runtime guarantee.
- `projected_raw_bytes` is `union_patch_count * 3 * 256 * 256`; coordinate
  bounds come from the selected union and are inclusive patch origins.

## Kaggle Launch Gate

This spec prepares work assignments but deliberately does not authorize or make
an inference kernel launch-ready. Before remote production inference:

1. publish the exact atlas/selected manifests and both final checkpoints as
   immutable private Kaggle datasets with recorded versions and hashes;
2. lock the deterministic full-map posterior-`mu` representation as
   `float32[16,32,32]`, including its container, row identity, and provenance;
3. implement and validate a dual-model worker with atomic per-WSI outputs and an
   audit completion marker;
4. run one worst-case WSI pilot measuring PNG decode, strip memory, both-model
   throughput, and output size;
5. retain five production jobs only if the pilot proves the slowest planned
   shard fits Kaggle limits; otherwise relock only the work-shard count.

Expected launch shape after that gate is one pilot, exactly five disjoint-WSI
GPU production jobs, and one global validator step. Production jobs read each union
patch once, feed identical clean batches to both frozen models, emit only the
full FP32 posterior `mu` maps plus identity/provenance audit artifacts, and do
not pool, aggregate, classify, or compute evaluation metrics.

## Acceptance Criteria

1. Exact input hashes, headers, row order, split inheritance, and unique
   coordinate identities validate before output creation.
2. Every selected group contains exactly `min(N,cap)` rows and exactly the
   specified centered systematic ranks.
3. Cancer selection is produced solely from the mask-free cancer input. Union
   logic cannot add cancer membership to a tissue-only coordinate; independent
   selector unit tests prove tissue rows are not an input to cancer selection.
4. Six task files match the exact split/class counts above and contain no row
   from another split.
5. Union overlap, per-split counts, and total equal 58,666 and 599,398 exactly.
6. Every WSI appears in exactly one of five work assignments and no shard splits
   a WSI. Concatenated shard rows equal the union exactly once.
7. Spatial-coverage diagnostics reproduce the exact capped-group counts, macro
   means, worst groups, and rational minima specified above.
8. Outputs are byte-identical across two runs. Task files, union, assignment,
   and shards use temporary files/paths; the audit publishes last.
9. Audit records input/generator/output hashes, caps/formula, cancer counts by
   split and diagnosis, tissue counts by split/diagnosis/tissue, spatial
   diagnostics, per-WSI/per-shard counts, raw-size warning, acceptance flags,
   and `launch_ready=false` with the remaining launch gates.
10. Focused tests, `./scripts/python_quality.sh`, `git diff --check`, and
    `./scripts/agent_preflight.sh` pass; clean-context implementation review has
    no blocker.

## Tests And Verification Commands

```bash
.venv/bin/python -m pytest -q tests/test_generate_ubc_consumption_manifests.py
.venv/bin/python -m eqvae.cli.generate_ubc_consumption_manifests
./scripts/python_quality.sh
./scripts/agent_preflight.sh
```

Tests must cover the midpoint formula, all-below-cap retention, interleaved
tissue classes, independent task selection, exact grouping/caps, malformed or
reordered inputs, split leakage, duplicate coordinates, union membership,
whole-WSI shards, deterministic bytes, audit hashes/audit-last failure, and the
real full-manifest hashes/counts.

## Verification Result

- Focused suite: 12 passed, including the sealed full-manifest integration and
  synthetic order, identity, split, selector-independence, deterministic-byte,
  and audit-last failures.
- Full Python quality gate: 858 passed, one expected GPU-only skip, Ruff clean,
  and BasedPyright zero errors on 2026-08-24.
- Union SHA-256:
  `f92558fa7aced13debc839c733e2c96d03c1a3df4b2a0b194b60558c69c04012`.
- Work-assignment SHA-256:
  `a5377662911d147a14e1a17714c01b0ed5c6c1f2231429342f1373a1001edeff`.
- Final clean-context implementation re-review found no local blocker. The
  separate remote inference launch gate remains closed.

## Implementation Blockers

None for consumption manifests and work assignments. The separate Kaggle
inference launch gate above remains intentionally closed.

## Known Risks

- Patch caps reduce computation and dominance, not biological correlation.
- Cancer evidence has only 23 test WSIs; LGSC and MC have two each.
- Tissue necrosis has five positive validation and five positive test WSIs.
- Raster-systematic ranks preserve broad occupied-cell coverage but are not an
  explicit diversity optimizer.
- Five patch-balanced work shards can have unequal PNG decode time.
- Raw union pixels project to about 109.75 GiB; never use these five assignments
  as raw saved-output jobs.

## Adversarial Checks

- Change one candidate row before a midpoint rank and require selected hashes to
  change or input-hash validation to fail.
- Make a tissue row cancer-selected through union logic and require failure.
- Duplicate or move a WSI across work shards and require failure.
- Preserve total rows while exceeding a per-WSI/per-class cap and require
  failure.
- Publish an audit before one shard and require the completion test to fail.
- Treat cancer patches as independent test examples and reject the metric
  contract in the later evaluation spec.

## Open Questions

- The later evaluation spec must choose compact latent representation, WSI
  aggregation, classifier protocol, and final metrics before test features are
  inspected.
- The worst-case pilot decides whether five production jobs remain sufficient.

## Related Files

- `docs/specs/0017-masked-holdout-test-dataset.md`
- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `src/eqvae/cli/generate_ubc_eval_manifests.py`
- `kaggle/generate_ubc_ocean_test.py`
