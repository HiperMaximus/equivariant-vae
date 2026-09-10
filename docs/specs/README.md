# Specs

This directory contains implementation and experiment specs.

Use these specs to define what "done" means before changing code, evaluation
logic, paper artifacts, or workflow tooling.

Read first:

1. `../spec_driven_development.md`
2. `template.md`
3. the relevant specs in the table below

## Status Values

- `draft`: useful planning text, not ready for implementation.
- `draft active`: active workstream, but blocked by open questions or missing
  verification details.
- `locked / implementation-ready`: enough contract, acceptance criteria, and
  commands exist to start coding.
- `implemented`: accepted and verified.
- `rejected`: a live exclusion contract that prevents reuse of an invalid
  candidate.

This table owns current lifecycle state. Exact spec bytes bound into accepted
artifacts remain immutable evidence contracts and are not edited merely to
change a launch-time header.

## Spec Index

| Spec | Status | Blocked By | Current Rule |
| --- | --- | --- | --- |
| `0001-translatable-normal-vae-baseline.md` | Normal-VAE control implemented and verified at update 60000; beta `0.01`, fixed-25 evidence, and final checkpoint are locked. | None for the completed comparison. | Preserve the frozen baseline and accepted evidence without retraining it. |
| `0002-strict-python-quality-gate.md` | Active; current verification state is recorded in `CURRENT.md`, not duplicated here. | None. | Run `./scripts/python_quality.sh` for Python changes with proportionate focused tests. |
| `0003-kaggle-cli-execution-workflow.md` | Implemented and proven through guarded push/status/output, resumable training, and sealed evaluations. | Remote actions require explicit approval and confirmation variables. | Preserve guarded execution and exact owner-qualified provenance. |
| `0006-selected-runtime-local-mechanics.md` | Implemented, locally verified, and exercised by the completed baseline. | None. | Maintain fail-closed runtime identity/checkpoint checks. |
| `0007-real-ubc-ddp-amp-selected-runtime-runner.md` | Implemented and remote-verified through update 60000. | None for the normal-VAE runner. | Reuse shared mechanics; change only architecture-specific `SO(2)` details. |
| `0008-canonical-fixed32-and-remote-debug-tiny-readiness.md` | Implemented; debug/resume/tiny proof complete. | None. | Retain as the short pre-long-run proof pattern. |
| `0009-first-full-selected-runtime-training-run.md` | Implemented durability and DDP lifecycle contract retained by the full-run guard; executed runtime selection is owned by Spec 0011. | None; its v1 checkpoint is not a resume base. | Preserve only the guard-compatible lifecycle contract. |
| `0004-sipaim-paper-scaffold.md` | SIPAIM route ended without submission; the scaffold remains the working manuscript. | A new venue and an explicitly requested paper update; downstream MIL and its sealed test are complete. | Do not present SIPAIM as active; integrate fixed results only under a separately scoped paper task. |
| `0005-overleaf-empty-project-initialization.md` | implemented | None for the narrow empty-project first-sync case. It is not a general conflict-resolution or force-push policy. | Use only `scripts/sipaim_overleaf_sync.sh push`; it may initialize an empty-tree Overleaf `master` with a normal fast-forward commit, but must abort for nonempty remote content. |
| `0010-fixed25-equivariance-artifact-protocol.md` | Implemented and exercised at all 20 boundaries for both completed models. | None. | Reuse its artifacts as validation evidence without substituting them for the completed sealed downstream test metrics. |
| `0011-reusable-goal-derived-runtime-and-compiled-fastpath.md` | Runtime search, both full runs, and mechanics transfer are complete. | None. | Preserve the executed runtime evidence; do not recreate a tuner. |
| `0012-continuous-so2-vae-architecture.md` | Complete: radial/F2 decision, equal-copy F01 architecture, count/init, and full-model assembly are fixed. | None. | Keep every architecture choice locked. |
| `0013-fixed-f01-architecture-probe.md` | Complete: padded-`bmm`/direct compiled mechanics pass correctness, runtime ratios, DDP/AMP/compile, and VRAM; raw CV remains diagnostic. | None. | Do not rerun or add mechanics/runtime arms. |
| `0014-fixed-f01-full-vae.md` | Implemented, locally verified, and used by the completed 60,000-update SO(2) run. | None. | Preserve the singular fixed model; do not retune architecture during evaluation. |
| `0015-fixed-so2-selected-runtime-readiness.md` | Complete and consumed by the completed real-data run. | None. | Preserve the readiness evidence; do not reinterpret its timing as evaluation quality. |
| `0016-so2-real-data-prelaunch-and-full-run.md` | Complete at update 60000 with the accepted non-clean 66/68 gate result. | None for training or the completed downstream evaluations. | Preserve the final checkpoint and caveat; do not rerun training. |
| `0017-masked-holdout-test-dataset.md` | Additive mask-plus-Otsu atlas generated and verified for all 152 masked WSIs; latent extraction consumes its coordinates directly. | None. | Preserve the verified atlas and its hash; do not create an oversized raw extraction. |
| `0018-shared-masked-wsi-evaluation-split.md` | Implemented and verified: one shared 106/23/23 WSI split, 1,750,221 Otsu-derived cancer/AE coordinates, and 666,807 high-purity tissue coordinates. | None for membership/split. | Preserve task membership and the immutable post-evaluation test boundary. |
| `0019-task-consumption-manifests-and-kaggle-work-plan.md` | Implemented: six frozen task/split manifests, one 599,398-row union, and five whole-WSI work shards. | Dataset generation is owned by Specs 0020-0021. | Preserve every hash and row identity. |
| `0020-fp32-latent-shard-format-and-io.md` | Storage core and twelve model/task/split logical views implemented; five real shards per model exist remotely and the compact global audit is complete. | None for storage. | Preserve the FP32 shard and zero-copy view contracts; do not create physical split copies. |
| `0021-dual-model-wsi-latent-inference.md` | Immutable input v2, smoke, production runs 01-05, and CPU finalizer v3 are complete and compactly validated. | None. | Keep all ten binaries remote; do not infer publication or classification authorization. |
| `0022-additive-cancer-coverage-topup.md` | Ordinary part 11 (74,033 rows/model) and WSI45630 supplement (22,649 rows/model) complete and verified. | None for extraction; consumed by completed Spec 0024 coverage audit. | Preserve remote binaries and existing quarter-coverage views; no repeat extraction. |
| `0023-matched-supervised-latent-evaluation.md` | Tissue confirmation is frozen; its gated MIL candidate did not establish sufficient learning. Specs 0026–0043 own the accepted MIL architecture, training and both sealed tests. | None. | Preserve the diagnostic boundary; capacity alone is not final-classifier evidence. |
| `0024-full-foreground-latent-completion.md` | All eight v1 jobs COMPLETE; aggregate audit and independent raw-atlas coordinate check pass: all 1,750,221 foreground patches across 152 WSIs, zero missing/duplicates. The 49,096 unencoded mask-only atlas rows are outside this target. Evidence/hash in `CURRENT.md`. | None for extraction; all launch approvals consumed. | Preserve sealed extraction bytes. Distinct full-coverage logical views/catalog are implemented by Spec 0025; no repeat pushes or latent downloads. |
| `0025-full-foreground-logical-mil-dataset.md` | Implemented / locally verified and consumed by completed matched training and sealed evaluation. | None. | Preserve its exact owner-qualified references and development/test boundaries. |
| `0026-local-global-sigmoid-mil.md` | Implemented and exercised through completed capacity, training and sealed evaluation: staged width-192, two fixed-26 local-softmax blocks, 16-REG calibrated-sigmoid summary, and final CLS-only softmax read. | None for the fixed one-seed comparison. | Preserve the 1,513,055-parameter topology; do not tune it from test results. |
| `0027-wsi45630-local-softmax-kernel-probe.md` | Executed / closed without selection. Private T4 version 1 authenticated the real graph; explicit/auto/efficient missed the elementwise K/V-gradient gate despite relative L2 below `1.047e-3`; Flash rejected the mask. | Fresh authorization plus a revised numerical-acceptance contract if another timing probe is desired. | Preserve the retrieved artifact/logs; no retry is authorized and no backend winner exists. |
| `0028-wsi45630-local-softmax-repair-probe.md` | Executed / closed. Private version 2 passed the revised gate; explicit chunk 8192 was the lowest-mean softmax implementation on the real graph. | Preserve evidence; use it only for Spec 0026's softmax backend/chunk selection. | Authority consumed; no retry. This did not compare sigmoid or test the full MIL model. |
| `0029-account-portable-kaggle-resources.md` | Implemented: actors own new launches while every source/download keeps its canonical owner/slug/version and per-file hashes. | None. | Use receipt-driven generic actions for any separately authorized resource. |
| `0030-local-global-mil-capacity-probe.md` | Rejected capacity candidate: private `maximshtefan/.../2` OOMed in local block 2 at 15.339 GB peak on a 15.636 GB T4 before any optimizer step. | None. | Preserve the negative result; do not claim that this candidate fit or learned. |
| `0031-coordinate-flexattention-probe.md` | Rejected candidate: real-graph analysis shows 7.45x/14.85x candidate-pair overcomputation for block sizes 16/32; its remote launch remains prohibited. | None. | Keep the exclusion contract; Spec 0032 uses the selected exact fixed-degree primitive. |
| `0032-amp-fixed26-mil.md` | Implemented. It defines the 1,513,055-parameter MIL mechanics: AMP, packed projections, fixed-26 local attention, calibrated sigmoid and FP32 logits/loss. | None. | Preserve the final model mechanics and accepted evidence. |
| `0033-exact-local-attention-backend-bakeoff.md` | Accepted backend selection: on the exact 32,595-node T4 graph, whole-bag fixed-25 Inductor passed oracle/gradient gates at 17.87 ms and 276 MiB peak allocated. | None. | Preserve the authenticated selection; do not relaunch the consumed probe. |
| `0034-full-compiled-fixed25-mil-probe.md` | Accepted full-model capacity gate: private version 7 passed both branches under PyTorch 2.14.0/cu130 with about 8.1 GB conservative headroom. | None. | Preserve the authenticated artifacts; capacity is not classifier performance. |
| `0035-largest-class-weighted-amp-probe.md` | Completed / Kaggle v2 accepted: both branches settled the default GradScaler from 65,536 to 16,384 on the third allowed EC attempt, then all remaining calibration and training-simulation updates committed first-attempt; boundary, weighting, compilation and memory gates passed. Its lifecycle was consumed by completed training/evaluation. | None. | Preserve the authenticated runtime evidence; do not relaunch the consumed probe. |
| `0036-local-global-mil-training.md` | Complete: learning ended by locked early stopping at updates 7420/7791; version 7 passed receipt-bound selected-best revalidation with zero optimizer updates, exact identities/predictions/discrete metrics, authenticated checkpoints and paired bootstrap. | None; Spec 0041 completed the separate sealed test. | Preserve the one-seed evidence; do not resume, retune or select from the test result. |
| `0037-tissue-fastpath-calibration-probe.md` | Implemented / accepted: authenticated private T4 probe selected `compile_max_autotune_compiled_autograd` for both static tissue shapes and proved default-scaler calibration, compiled full train steps and native fused AdamW. | None for runtime mechanics. | Preserve the runtime-only evidence; Spec 0039 owns tissue learning. |
| `0039-tissue-label-efficiency-training.md` | v2 completed under its original protocol; guarded v3 completed all ten runs from the same byte-verified input with a ten-epoch floor. Spec 0043 subsequently completed the separate sealed test. v1 remains non-evidence. | None; training and fixed test evaluation are complete. | Preserve v2/v3's distinct claims and receipts; never relaunch, tune or select from test. |
| `0038-frozen-vae-rotation-orbit-visualization.md` | Implemented but dense scientific result superseded: the run is valid provenance, yet its trajectory mixed opposite rotation conventions at cardinal angles. | Corrected validation is owned by Spec 0050. | Preserve bytes/receipt; do not reuse dense local-linearity, step-CV, PCA-orbit or continuous F1 claims. Exact-quarter controls remain valid. |
| `0040-spatial-latent-pca-coherence-visualization.md` | Implemented / locally verified: common-scale PCA-RGB maps and paired local relative-edge-RMS use all 25 SHA-pinned final posteriors; a per-tile paper-style view has no source blend; Section 6 renders a fixed-split local affine RGB probe with held-out mean R2 `0.286` normal / `0.189` SO(2). | Repo-wide gate is blocked by unrelated pre-existing packaged-kernel lint failures. | Show Section 5 and PNGs 04–05 as qualitative spatial variation; use Section 6/PNG 06 only as a held-out local-linear appearance diagnostic, never as equivariance/reconstruction evidence. |
| `0041-sealed-mil-test-evaluation.md` | Implemented / complete: one label-blind paired inference run and one receipt-bound local scoring pass evaluated the frozen checkpoints on all 23 sealed test WSIs. | None; the one-shot authority is consumed. | Preserve the authenticated outputs and report the fixed result with its one-seed, small-test-set limitations; never relaunch, tune or select on test. |
| `0042-spec0041-kaggle-slug-normalization-amendment.md` | Implemented / complete: the local scorer accepted only Kaggle's exact normalized canonical ID and immutable launch receipt without changing any scientific input or metric. | None. | Preserve the amendment and pre-score claim; do not generalize the slug exception. |
| `0043-sealed-tissue-patch-test-evaluation.md` | Implemented / complete: one label-blind private run and one receipt-bound local scoring pass evaluated all ten frozen v3 classifiers on 31,572 patches from 23 test WSIs. | None; the one-shot authority is consumed. | Preserve the fixed result and its five-necrosis-WSI/one-training-trajectory limitations; do not relaunch, tune or select on test. |
| `0044-professor-metrics-and-plots.md` | Implemented / locally verified: the hash-bound local package contains the issue-requested metrics, boxplots, dashboard and comparison views; final adversarial review found no P0/P1. | Full-population validation boxplots require a separate inference contract; broad renderer type suppressions are non-blocking debt. | Preserve `runs/local/professor_metrics_v1` and its manifest SHA `28b5d543…9a06`; Spec 0048 integrates it into the final report, while paper/Overleaf transfer remains separate. |
| `0045-frozen-vae-test-reconstruction-evaluation.md` | Implemented / complete: one frozen run and one receipt-bound local score evaluated both VAEs on all 67,138 test patches from 23 WSIs. Primary normal/SO(2) MAE is `0.0628721/0.0642299`; difference `-0.00135779 [-0.00221281,+0.00004591]`. | None; the one-shot launch and scoring authorities are consumed. | Preserve the authenticated result and its 23-WSI/training-seed limitations; do not rerun, tune or select from test. |
| `0046-spec0045-kaggle-slug-normalization-amendment.md` | Implemented and independently accepted: the exact authenticated canonical slug and receipt are allowed while every Spec 0045 scientific contract remains unchanged. | None. | Preserve the exact alias/receipt exception; do not generalize it. |
| `0047-spec0045-diagnosis-index-resume-amendment.md` | Implemented / complete: the frozen scorer remained byte-identical; an exact adapter accepted only the oracle-native redundant diagnosis numbering, and the fail-closed resume completed once. | None; independent pre/post-result review found no P0/P1. | Preserve both claims and the exact amendment; the resume authority is permanently consumed. |
| `0048-professor-final-experiment-report.md` | Implemented; its sealed and supervised sections remain accepted and Spec 0050 replaced/superseded the defective dense Figures 7/8a/8b and prose. | None for the professor report. | Preserve unaffected results and the corrected rotation section; paper, thesis and Overleaf remain separate. |
| `0049-live-documentation-cleanup.md` | Implemented: live landing docs contain current contracts only; completed backlog entries and obsolete pretest narration were removed while evidence-bound specs and verifier literals were preserved. | None. | Keep the reduced documents current; do not restart append-only status logs. |
| `0050-corrected-rotation-geometry-validation.md` | Implemented / accepted: private Kaggle v1 corrected the mixed-sign sweep and completed the fixed-25 geometry audit. H1--H4 failed; no shared action was demonstrated and H6 factorization is unresolved. The report was rebuilt from hash-verified outputs and independently reviewed. | Repo-wide gate retains only unrelated packaged-probe lint/type debt. | Preserve Spec 0038 as superseded provenance and the accepted v1 outputs; the one-shot authority is consumed. |
| `0051-decoded-latent-transform-consistency.md` | Implemented / accepted: receipt-bound private Kaggle v1 completed the fixed-25 decoder-side audit. Dense H1--H3 failed their locked thresholds, but the SO(2) decoder exactly realizes the prescribed C4 action at `rot90/180/270`; only `flip_diag` passes the reflection rule. | None; the one-shot authority is consumed. | Preserve the exact-C4 result without generalizing it to continuous SO(2), encoder equivariance, learned low-dimensional action, factorization, or complete D4/O(2); report integration is complete under Spec 0052, while issue changes remain separate. Guard: `spec0051_decoded_latent_transform_authorized`. |
| `0052-professor-report-decoded-transform-integration.md` | Implemented / complete: the professor report separates the 25/25 paired SO(2) decoded-route advantage from the unmet absolute `0.50` benchmark and limits the near-exact claim to the prescribed C4 decoder action. | None; no new inference or remote mutation occurred. | Preserve the 29-page/22-figure report, accepted hashes, and no-causal/no-continuous-generalization boundaries; paper, thesis, Overleaf, and further issue edits remain separate. |
| `0053-functional-riemannian-latent-geometry.md` | Draft active; detailed functional, Lie, decoder-metric, posterior, geodesic, three-PGA, connection/transport, and D4 roadmap independently reviewed. | The bounded 0054--0058 preflight lineage is closed; every scientific stage remains separately locked. | Use the WSI-disjoint fit/evaluation split and primary decision matrix; lock randomized-linear-algebra sketch width `s=k+p` separately from `r=128` independent scalar probes with 99% confidence; a separate future contract may use the calibrated `epsilon=.008`, but no scientific output starts from this preflight. |
| `0054-spec0053-numerical-resume-preflight.md` | Rejected / closed: its sole parent `/1` reached the eager-autodiff gate and failed before creating an output manifest. | The consumed `/1` slug and guard cannot be reused. | Preserve its receipt/log as negative evidence; no child or retry under this contract. |
| `0055-kaggle-progress-and-numerical-retry-contract.md` | Rejected / closed: private parent `maximshtefan/eqvae-functional-geometry-preflight-02a08ab5/1` failed in `frozen_state`, before branches or numerical work. | Its `/02` authority and child are consumed/prohibited; a new amendment must verify the frozen-bundle identity before any fresh user-authorized launch. | Preserve its receipt/log as negative infrastructure evidence; use 8 deterministic RMS-one Rademacher projections, primary fixed steps, FP64 aggregate-plus-worst gates, and closed-schema safe JSONL in any separately authorized successor. |
| `0056-frozen-bundle-retry-preflight.md` | Rejected / closed: private parent `maximshtefan/eqvae-functional-geometry-preflight-03a08ab5/1` passed bundle lookup, frozen model loading, and eight primary/diagnostic projections, then failed only the locked primary JVP-FD gate. | Its `/03` authority and child are consumed/prohibited; another retry needs a new numerical amendment and user authorization. | Preserve its exact log/receipt; do not reinterpret result-blind branch telemetry as a model comparison. |
| `0057-jvp-epsilon-ladder-preflight.md` | Rejected / closed: private parent `maximshtefan/eqvae-functional-geometry-preflight-04a08ab5/1` completed frozen-state loading and 8 primary plus 16 diagnostic records for one blind branch, then failed its unchanged primary JVP-FD gate (`0.0261526` aggregate, `0.0266058` worst versus `0.01`). Diagnostics pass at `0.004` (`0.00651530/0.00675288`) and `0.008` (`0.00319501/0.00330884`). | Its one-shot authority and child are consumed/prohibited; a successor must be separately amended and authorized. | Preserve the exact log/receipt; diagnostic results calibrate finite-difference step size but cannot retroactively accept the failed primary gate. |
| `0058-jvp-epsilon-grid-calibration.md` | Complete: the sole private `/1` selected `epsilon=0.008` by the locked global `0.005` aggregate-plus-worst rule (`0.00237315/0.00332016`); `.004` fails only worst case (`0.00666581`). | It is calibration evidence only, not a model comparison or full preflight. | Preserve launch/output receipts under `runs/local/{kaggle_launches,kaggle_outputs}/maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/v0001`; no binary search, retry, or child. |

Guard phrases retained for fail-closed one-shot scripts. Their
presence records consumed authority and is not permission to relaunch:

- local-softmax kernel probe authorization is `spec0027_local_softmax_probe_authorized`;
- local-softmax repair probe authorization is `spec0028_local_softmax_repair_probe_authorized`;
- local-global shared-access retry authorization is `spec0030_local_global_capacity_shared_access_retry_authorized`;
- permissive-dynamic full compiled fixed-25 retry authorization is `spec0034_permissive_dynamic_retry_v3_authorized`;
- training-effect full compiled fixed-25 retry authorization is `spec0034_training_effect_retry_v4_authorized`;
- GradScaler full compiled fixed-25 retry authorization is `spec0034_grad_scaler_retry_v5_authorized`;
- recompile-limit-3 full compiled fixed-25 retry authorization is `spec0034_recompile_limit3_retry_v6_authorized`;
- pinned-torch full compiled fixed-25 retry authorization is `spec0034_pinned_torch_retry_v7_authorized`;
- synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`;
- capped real-data runtime pretest contract is `real_data_runtime_pretest_contract_ready`;
- runtime-selection kernel contract is `runtime_selection_kernel_ready`;
- selected-runtime debug/tiny gate contract is `selected_runtime_debug_gate_contract_ready`.
- consumed fixed25 dense 360-degree population authorization is
  `spec0038_fixed25_dense360_population_authorized`; accepted result is version
  `maximshtefan/eqvae-fixed25-dense-rotation-population/1`.
- Spec 0054 parent preflight authorization is
  `spec0053_numerical_resume_preflight_authorized`; it permits only its exact
  private parent and receipt-bound singleton child continuation.
- Spec 0056 frozen-bundle retry authorization is
  `spec0056_frozen_bundle_retry_authorized`; it permits only its exact private
  parent and no automatic child launch.
- Spec 0057 JVP epsilon-ladder authorization is
  `spec0057_jvp_epsilon_ladder_authorized`; it permits only its exact private
  parent and no automatic child launch.
- Spec 0058 JVP epsilon-grid calibration authorization is
  `spec0058_jvp_epsilon_grid_calibration_authorized`; it permits only its exact
  private parent and no binary search, retry, or child.

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
