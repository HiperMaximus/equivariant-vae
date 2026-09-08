# Spec 0043: Sealed Tissue Patch Test Evaluation

Status: locked / implementation-ready after clean-context adversarial review
Implementation readiness: implement and verify the label-blind inference and
receipt-bound local scoring package
Owner/workstream: final patch-level tissue evaluation of the frozen Spec 0039
normal- and continuous-SO(2)-VAE classifiers
Last updated: 2026-09-06

Remote authorization: after confirming that the patch-level tissue test had not
been run, the user explicitly requested “ok let's do the patch tissue test
evaluation” on 2026-09-06. This authorizes one locked private input publication
and one test-only Kaggle launch after local acceptance. It does not authorize
retraining, changed checkpoints, test-driven selection, paper/thesis edits, or
an automatic scientific retry.

## Purpose And Primary Question

Evaluate all ten independently validation-selected Spec 0039 v3 best tissue
classifiers once on the same frozen 31,572-patch test split. The five fixed
training budgets are 250, 500, 1,000, 2,500 and 5,671 labelled patches per
class. Remote inference is label-blind and seals predictions before a separate
local scorer opens the tissue oracle. Test data may describe generalization;
they may never select a budget, checkpoint, hyperparameter, subset or retry.

The primary estimands are pooled-patch three-class macro-F1 for each branch at
every budget and the paired normal-minus-SO(2) difference at each budget.
Per-class F1 is computed from one confusion matrix pooled over all test patches;
macro-F1 is the unweighted arithmetic mean over the fixed classes tumor,
stroma and necrosis, so WSIs contribute in proportion to qualifying patches.
The prespecified curve summary is normalized trapezoidal area under macro-F1
against `log10(labels_per_class)` and its paired normal-minus-SO(2) difference.

## Frozen Inputs

- Local-only scoring oracle:
  `runs/local/ubc_ocean_supervised_manifests/tissue/tissue_test.csv`, SHA-256
  `ff4183da11bdee4065a061410ef7ec13f45f1fe9791cdbe7139c664721b44082`.
  It contains exactly 31,572 test patches from 23 WSIs: tumor 21,796 patches /
  23 WSIs, stroma 8,500 / 21, necrosis 1,276 / 5. The Kaggle input and kernel
  must not contain `tissue_label`, class indices, truth, target, diagnosis,
  selection ranks, or any label-dependent metric code on the executed path.
- Remote test locations are derived once from that oracle by dropping
  `tissue_label` and `selection_rank` while preserving exact row order and the
  fields `dataset_row,atlas_row_index,wsi_id,x,y,split,part,file_index`.
  The builder proves every split is `test`, row identity is unique, all 31,572
  rows are retained and the derived file contains no label-like field. The
  scorer later requires an exact one-to-one join back to the frozen oracle.
- Physical catalog: unchanged 12-row Spec 0023 supervised catalog with SHA-256
  `9bf2baa9c8dd8cfc079ff341f441207e3cf4e68843c7e926ea55934fdc0ef80e`.
  Preserve the six exact owner-qualified producer locators and version 1; do
  not infer or rewrite source owners from the active Kaggle login.
- Executable source: copy the immutable snapshot from
  `runs/local/tissue_label_efficiency_training_retry_v3/bundle/src`, bound by
  training-input contract SHA-256
  `154994e86cdc5c8b43536e2ad5cbe17390ae0591ee9484641b14084f967f14ca`.
  Dormant training/evaluation modules may remain for byte identity, but the
  tested kernel import/call graph may use only data/model/inference paths.
- Checkpoint provenance: authenticated private training kernel
  `maximshtefan/eqvae-tissue-label-efficiency-training/3`, launch-receipt
  SHA-256
  `5f69bb777f80c9926f54387cfb7e767cb623bcd55e8d77d473362d4a5795198d`
  and output-receipt SHA-256
  `fc1c68b6aea1be90ca1842dba1625da35b1098e8fe68bf42cd47c599559743e7`.
  Authenticate each full `best.pt` plus adjacent `best.json`, including branch,
  budget, config SHA-256
  `4c72e17c493b6beb2065d9b0828747caf46475af2ca75f33594014a446940b7e`,
  selected validation record and file hash. Stage only ten derived
  state-dict-only artifacts with source checkpoint, manifest and state hashes;
  never stage a full checkpoint, validation prediction or metric artifact.

## Label-Blind Inference Contract

- Private Kaggle input/output only. Use exact PyTorch `2.14.0` cu130. One
  isolated process receives one T4 and one representation. Each worker loads
  that representation's five state-only classifiers and evaluates the same
  canonical test rows with batch size 159 and a final 90-row batch. Read each
  latent batch once, then infer all five models before advancing. Use
  `TissueClassifier`, evaluation
  mode, `torch.inference_mode`, FP16 CUDA autocast and the same eager validation
  semantics used to select the checkpoints. Construct no optimizer,
  GradScaler, scheduler, augmentation, train/validation loader, loss or scorer.
  Optimizer updates are exactly zero.
- Resolve and hash-check every physical sidecar/binary through its immutable
  catalog row before reading latents. Require identical logical order, geometry
  and physical pointers across both workers and all ten prediction artifacts.
  No sampling, truncation, label balancing, row dropping or test-time
  augmentation is allowed.
- For each branch/budget, write one compressed CSV containing exactly the
  canonical 31,572 rows and fields
  `dataset_row,atlas_row_index,wsi_id,x,y,part,file_index,logit_tumor,`
  `logit_stroma,logit_necrosis,prediction`. All logits must be finite and
  `prediction` must equal argmax in canonical class order tumor, stroma,
  necrosis. Write separate runtime/access evidence with branch, budget,
  checkpoint identity, input/order hash, per-part read counts and timing. Do
  not write truth, label, loss, score, validation values or class supports.
- Each worker writes only into private scratch. The parent promotes the entire
  ten-file prediction set only after both workers pass and every row identity
  agrees. If either worker fails, remote output contains only non-predictive
  failure evidence. The top-level completion/status file is written last.

## Local Scoring And Statistics

- Only after the completed remote output is downloaded through its exact
  launch receipt and every prediction artifact is hash-verified may the scorer
  open the local oracle. It first writes an exclusive pre-score claim binding
  the immutable input/run/output receipts, ten prediction hashes, oracle hash,
  scorer hash, frozen scorer-vector hash and this spec hash.
- Join exactly by the complete logical identity, verify the fixed 31,572-row
  order and class/WSI supports, and recompute argmax. At each budget report, per
  branch: patch macro-F1, balanced accuracy, accuracy, mean unweighted cross
  entropy, three-class precision/recall/F1/support, 3x3 confusion matrix,
  `n_patch`, `n_WSI`, and overall/per-class WSI counts.
- Use 10,000 paired, tissue-support-stratified WSI-cluster bootstrap replicates
  with seed `3901`. The 23 WSIs have three pre-model strata derived only from
  true tissue support: 2 tumor-only, 16 tumor+stroma, and 5 tumor+stroma+
  necrosis. Resample 2/16/5 WSIs with replacement within those fixed strata;
  carry all patches and sampled multiplicities; reuse the exact draw for both
  representations and all five budgets. This preserves the designed support
  profile while leaving the natural patch imbalance intact and guarantees all
  three classes in every replicate. Compute metrics in fixed three-class order
  with zero division set to zero. Never redefine a replicate as a two-class
  problem or use an independent-patch bootstrap. State that interval inference
  is conditional on the observed tissue-support-stratum counts.
- Report percentile 95% intervals for every branch metric and paired
  normal-minus-SO(2) difference. Compute normalized trapezoidal log-budget AULC
  from the five raw points without smoothing or a monotone envelope. With
  `x_i=log10(b_i)` for `b=(250,500,1000,2500,5671)`, define
  `AULC=trapz(macro_f1_i,x_i)/(x_5-x_1)` using adjacent trapezoids. Treat the
  five budget differences plus paired AULC difference as one six-contrast
  primary family. For contrast `k`, define centered bootstrap error
  `e[r,k]=delta_star[r,k]-delta_hat[k]`; let `c` be NumPy's `method="linear"`
  0.95 quantile of `max_k(abs(e[r,k]))`; report simultaneous interval
  `[delta_hat[k]-c,delta_hat[k]+c]`. Pin that convention in the scorer vector.
  A positive interval is a normal-VAE advantage and a negative interval is an
  SO(2)-VAE advantage. A budget-specific claim requires its simultaneous
  interval to exclude zero; an overall curve-level claim requires the
  simultaneous AULC interval to exclude zero. Interpret AULC as average
  macro-F1 over the prespecified log-label range, not proof that one
  representation needs fewer labels. Secondary metrics are descriptive and
  may not rescue the primary result.
- Final interpretation states explicitly that inference covers sampling of
  these 23 test WSIs only, not training-seed uncertainty; all budgets have one
  optimization trajectory; nested budgets and shared test patches make curve
  points correlated; and necrosis evidence comes from only five WSIs.

## Fail-Closed Boundaries

- The builder refuses overwrite and stages only the derived label-free test
  CSV, physical catalog, immutable source and ten pruned state dicts. It rejects
  any label/diagnosis/class/truth/target/train/validation logical file, full
  checkpoint payload, prediction, metric, latest or final checkpoint.
- Runtime authenticates every staged byte, source mount, state hash, branch,
  budget, runtime identity and test-row hash. It rejects an extra/missing/
  duplicate row, split drift, nonfinite output, argmax mismatch, model mismatch
  or cross-branch identity difference.
- Freeze and test the local scorer before input publication. The remote input
  and run contract bind exact hashes of scorer, scorer vector and final spec,
  although neither labels nor scorer bytes are mounted remotely. No scorer edit
  is allowed after launch.
- The verified private-input receipt is exact-content validated and hash-bound
  into the exclusive launch and pre-score claims. The launch receipt must bind
  both source and uploaded `run.py`/metadata bytes. Generic pushes fail closed;
  only the guarded one-shot tissue-test route may upload this kernel.
- Before labels are opened, an infrastructure-only failure may be retried only
  from the byte-identical input/kernel contract with fresh explicit
  authorization; both workers rerun and no predictions from a failed attempt
  are reused. After scoring begins there is no remote retry. Any scientific or
  numerical change requires a new spec and treats this test result as observed.

## Outputs And Acceptance

- Local package root: `runs/local/tissue_test_evaluation`; remote output root:
  `runs/kaggle/tissue_test_evaluation_v0001`; local scored root:
  `runs/local/tissue_test_scored_v1`. The locked private slugs are
  `eqvae-tissue-test-inputs-v1` and
  `eqvae-label-blind-tissue-test-evaluation`; the authenticated actor qualifies
  both IDs in the build contract and guarded launch claim.
- Remote output includes ten label-free compressed prediction CSVs, per-worker
  runtime/access evidence, `run_contract.json`, `runtime.json` and a final
  `overall_status.json`. Local scored output includes the pre-score claim,
  joined predictions/metrics, `paired_bootstrap.json`, label-efficiency table,
  and completion manifest/status written last with all output SHA-256 values.
- Focused tests cover strict label-blind allow-list, exact checkpoint
  pruning/loading, source/checkpoint provenance, latent pointer restoration,
  inference-only AST/import paths, complete row/order/branch alignment,
  post-retrieval label join, metric definitions, clustered multiplicities,
  fixed support strata, paired direction, simultaneous intervals, AULC, exclusive
  claims, malformed inputs and actor-versus-source ownership.
- Run focused pytest, touched-file Ruff/BasedPyright, kernel compilation,
  `bash -n scripts/kaggle_kernel.sh`, `git diff --check`, package build/validate
  and independent clean-context code review with no unresolved P0/P1.
- Remote acceptance requires both workers and all ten aligned 31,572-row
  prediction files complete, finite and receipt-bound. Final acceptance requires
  the exact one-to-one frozen-label join and authenticated local score output.
  Do not update the paper, thesis or professor-facing issue in this workstream.

## Known Limits

This is one initialization per budget and one 23-WSI test split. Patch metrics
weight WSIs by their number of qualifying patches. Necrosis has only five
contributing WSIs. Point estimates, intervals or isolated budget behavior must
be reported without changing the models or repeatedly consulting the test set.

## Adversarial Review Result

Independent clean-context reviews found no P0 issue. The statistical review's
P1 ambiguity was resolved by fixing pooled-patch macro-F1, the exact log-budget
trapezoid, centered maximum-error construction and NumPy `linear` quantile
convention. The implementation review's three P1 findings were resolved by
making generic launch fail closed, binding exact source/upload code manifests,
and authenticating the verified private-input receipt through launch and
pre-score claims. Prediction promotion is atomic. A final re-review must confirm
no remaining P0/P1 before launch.

## Related Files

- `CURRENT.md`
- `docs/specs/0023-matched-supervised-latent-evaluation.md`
- `docs/specs/0039-tissue-label-efficiency-training.md`
- `docs/specs/0041-sealed-mil-test-evaluation.md`
- `docs/kaggle_cli_workflow.md`
