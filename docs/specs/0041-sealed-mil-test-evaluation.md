# Spec 0041: Sealed MIL Test Evaluation

Status: locked / implementation-ready after clean-context adversarial review
Implementation readiness: implement and verify the label-blind inference / post-retrieval scoring contract
Owner/workstream: final paired downstream evaluation of frozen normal- and continuous-SO(2)-VAE representations
Last updated: 2026-09-05

Remote authorization: after being told that sealed-test release and a Kaggle
launch required a separate authorization, the user explicitly replied “ok do
it” on 2026-09-05. This authorizes the one locked private input publication and
one test-only kernel launch below. It does not authorize retraining, tuning,
additional seeds, changed checkpoints, paper publication, or an automatic
scientific retry with a changed contract.

## Purpose And Primary Question

Evaluate the two independently validation-selected Spec 0036 best MIL
checkpoints once on the same frozen 23-WSI test split. Remote inference is
label-blind and hash-seals both branches' predictions before a separate local
scorer opens the frozen label oracle. The primary estimand is the paired
normal-minus-SO(2) difference in WSI macro-F1. Test data may describe
generalization only; it must never select a checkpoint, hyperparameter,
threshold, architecture, preprocessing rule, or retry.

## Frozen Inputs

- Local-only scoring oracle: `docs/data/ubc_ocean_eval_wsi_split.csv`, SHA-256
  `216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c`.
  The Kaggle input/kernel must not contain this file, diagnoses, class indices,
  or targets, and must not import or call label-dependent metrics remotely.
  Dormant metric modules inside the authenticated full source snapshot are
  permitted. After authenticated prediction retrieval, the local scorer joins
  exactly its 23 test labels: CC 5, EC 6, HGSC 8, LGSC 2, MC 2.
- Test logical locations:
  `runs/local/ubc_ocean_full_foreground_manifests/sealed_test/` bags and
  instances, SHA-256
  `8243f52abb68bea57a4a98083467a8a883999117a44908d8592627c74c160601`
  and
  `eda697c7f8a9408fef7a2d42f4aa1d74263323ae108cbe698a5c6c485d867f69`.
  Stage these original label-free rows unchanged. They contain 261,168 complete
  foreground patches in canonical `wsi_id,y,x` order and no train/validation
  row.
- Physical catalog: the unchanged 30-row Spec 0025
  `development/physical_parts.csv`, SHA-256
  `9303120aa99ab105eafdb14868bd8e1a1f785b643f149f103bb60beeabf8d92e`.
  Preserve all 15 exact owner-qualified producer locators and paired model
  rows; bind every producer to its recorded version 1 in integration-audit
  SHA-256
  `9b22ef9e3c58f59b4d88f4518749b9eee3e585b5034c2c1440dc31695e10860a`;
  do not discover or rewrite owners from the active Kaggle login.
- Executable source: copy the immutable source snapshot from
  `runs/local/ubc_ocean_mil_training_v3/bundle/src`, authenticated by training
  input contract SHA-256
  `c6b1b2ace6ed6cc8de5b1856fde74348a4895ee91eb9a5bbc1a22ae6d4a9f4f9`.
  In particular, model/candidate hashes must remain
  `9d4513c6f7d63aeb13ffc29f7586d1b336daddba3fa2daca3c2a9c45e53a7c72`
  and
  `bd16522fc5192ec330a0f46ba8f6653e6141f9c746f670c4dcbb3330d2f88a5c`.
  The snapshot may contain dormant training/scoring modules for byte identity,
  but the tested kernel import/call graph may use only inference/data/model
  paths and the remote mount contains no labels.
- Checkpoint provenance: authenticated private training kernel
  `maximshtefan/eqvae-local-global-mil-training/7`, launch-receipt SHA-256
  `e6e6e3ad366ccdc2eec7655e4576c76ed2c6c506da541ec6c421db736b674cb6`
  and output-receipt SHA-256
  `13ee0a2ccce4d9301777cd5e02bc0158d9f231843db178324137c58baf63db4a`.
  Authenticate each full `best` pointer/object locally, but never stage it.
  Stage only the derived model-state-only artifact plus this provenance:
  normal boundary 3127, source checkpoint/manifest SHA-256
  `1dcd17cbebcb9bf97ca9492681b788c1cca2c170cf4189eefcd93da0106b4806` /
  `b9e7aee6b8a99ea6b80495ec026ed59231739a09aad5c8a4c8461dcda7fd1aef`;
  SO(2) boundary 5671,
  `47e82979dc2fcf8b3b1793e1687a3efc8591b9c815f92dac5c632845e681914a` /
  `f40747bd11ecae3c47092a9f8867f28db9c8240ce2e629ebd6b4440ca530b1fd`.

## Evaluation Contract

- Private Kaggle input and output only. Use exact PyTorch `2.14.0` cu130 and
  the unchanged compiled model/runtime semantics from Spec 0036.
- One isolated process per branch and one T4 per process. The local builder
  strictly loads each best checkpoint, authenticates its
  branch/boundary/checkpoint/manifest/training-contract hashes, and saves a
  newly hash-bound artifact containing only `payload["model_state_dict"]`. Each
  worker loads its branch's model-only state into the fixed 1,513,055-parameter
  classifier, sets evaluation mode, and uses inference-only
  FP16 autocast with the existing FP32 global numerical closure. Construct no
  optimizer, GradScaler, backward pass, training dataset, validation dataset,
  scheduler, selection state, or augmentation.
- Build and hash each radius-2 graph from the released test coordinates before
  any latent read. Require exact representation-independent graph identity and
  identical WSI/order/bag identities across branches. Load complete bags;
  no sampling, truncation, padding fallback, or test-time augmentation.
- Evaluate each branch exactly once. For every WSI the remote worker atomically
  records WSI ID, five finite logits, argmax prediction, complete bag size,
  graph identity/degree summary, physical part access counts and timing. It
  records no truth, loss or score.
- Only after the completed remote output is downloaded through its exact launch
  receipt and the prediction artifacts are hash-verified does the local scorer
  open the label oracle. It joins one-to-one by WSI ID, verifies the exact class
  supports, recomputes argmax, and then reports per branch WSI macro-F1,
  balanced accuracy, accuracy, mean unweighted CE, five-class
  precision/recall/F1/support, confusion matrix and `n=23`, using canonical
  class order CC, EC, HGSC, LGSC, MC and zero for absent predictions.
- The local scorer then computes paired normal-minus-SO(2) differences for
  macro-F1, balanced accuracy, accuracy and mean CE with 10,000
  diagnosis-stratified WSI bootstrap replicates, percentile 95% intervals and
  seed 3601. Macro-F1 difference is primary; all others are secondary.
- Freeze and test scorer
  `src/eqvae/evaluation/mil_test_scoring.py` before package publication. The
  remote input/run contract binds its exact SHA-256, the final Spec 0041 hash
  and scorer test-vector hash even though scorer bytes and labels are not
  mounted remotely. No scorer edit is allowed after launch.
- Each worker writes only into private scratch. The parent promotes both
  prediction artifacts together only after both branches pass; if either branch
  fails, remote output contains only non-predictive failure evidence. Remote
  output contains no labels or metrics. The local scorer writes a pre-score
  contract binding the immutable remote prediction/output receipts and the
  local label-oracle/evaluator/test-vector hashes before opening labels. It then
  writes all scored outputs atomically and writes a completion manifest/status
  last with every output SHA-256. The scored output must explicitly state that
  the interval reflects only sampling of these 23 test WSIs, not training-seed
  uncertainty, and that two LGSC/two MC cases make class estimates unstable.

## Fail-Closed Boundaries

- The input builder refuses overwrite and stages only the exact label-free test
  view, immutable source snapshot, two derived model-state-only artifacts and
  their selected-best provenance. It rejects any label/diagnosis/class/target file,
  train/validation logical file, full checkpoint payload, or latest/final
  checkpoint pointer. Dormant source from the frozen snapshot is allowed, but
  AST/import tests prove the kernel cannot call training/scoring APIs.
- The runtime authenticates every staged byte, exact physical source, checkpoint
  branch/boundary/hash/contract hashes, runtime identity and test graph before
  inference. It refuses any extra test WSI, duplicate, missing row, split drift,
  model mismatch, nonfinite output or cross-branch identity difference. Truth,
  class support and label drift are local-scorer checks only.
- Test metrics never flow into selection or another launch. Before local labels
  are opened, an infrastructure failure may be retried only from the identical
  byte-verified input/kernel contract with a fresh explicit authorization. The
  failed attempt publishes no predictions, its scratch is discarded, and both
  branches rerun; there is no branch reuse. After scoring begins, there is no
  remote retry. A scientific or numerical contract change requires a new spec
  and must treat the released test result as already observed.

## Outputs And Acceptance

- Remote per branch: label-free `predictions.json`, runtime/access evidence and
  exact checkpoint identity. Remote root: `run_contract.json`, `runtime.json`
  and `overall_status.json` binding both child results. Local scored root:
  pre-score contract, two metrics/prediction files, `paired_bootstrap.json`,
  and completion manifest/status written last.
- Focused tests cover strict label-blind test allow-list, post-retrieval label
  join, exact checkpoint
  pruning/loading, graph/branch identity, inference-only source inspection,
  metrics/bootstrap direction, malformed/extra input rejection, package
  validation and actor-versus-source ownership.
- Run focused pytest, touched-file Ruff/BasedPyright, kernel compilation,
  `bash -n scripts/kaggle_kernel.sh`, `git diff --check`, package build/validate,
  and independent clean-context contract/code review with no unresolved P0/P1.
- Remote acceptance requires both branches complete with 23 aligned finite
  label-free prediction rows. Retrieve and hash all compact outputs before
  invoking the scorer. Final acceptance requires a one-to-one frozen-label join
  and authenticated paired output. Do not update the paper or thesis in this
  workstream.

## Known Limits

This is one initialization and one 23-WSI test split with only two LGSC and two
MC cases. A point-estimate advantage or interval crossing zero must be reported
as observed, without changing the model or repeatedly consulting the test set.

## Related Files

- `CURRENT.md`
- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0025-full-foreground-logical-mil-dataset.md`
- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/specs/0036-local-global-mil-training.md`
- `docs/kaggle_cli_workflow.md`
