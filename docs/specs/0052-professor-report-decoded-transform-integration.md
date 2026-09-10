# Spec 0052: Professor Report Decoded Transform Integration

Status: implemented / complete and verified
Implementation readiness: complete
Owner/workstream: professor-facing integration of accepted Spec 0051 evidence
Last updated: 2026-09-09

## Purpose

Update the professor-facing Spanish report from the accepted, receipt-bound
Spec 0051 fixed-validation output. The report must distinguish the comparative
normal-versus-SO(2) result from the stronger absolute decision rule that was
fixed before the remote output was viewed.

## Inputs

- `runs/local/decoded_latent_transform_v1/decoded_latent_transform_v1/decoded_latent_transform_summary.json`,
  SHA-256 `616e3a6c9caa6b8e72a0b21f41a5121d7a5baa420f8023b35775fedaf252a754`.
- `runs/local/decoded_latent_transform_v1/decoded_latent_transform_v1/manifest.json`,
  SHA-256 `a1168b419080305138f7aedca4c1ed345c3d5947e78299b0e5ea1430a75d3c50`.
- `docs/data/spec0051_decoded_transform_review_addendum.json`, SHA-256
  `b0fa0cc2eebe1207fcc2da7a00c30b9b85c40f4353e332131423f76f6c380e34`.
- The locked Spec 0051 and machine contract, with SHA-256 values
  `7a472a2de56813544506b4a9eb58e2f1f152fcee8f7d60e65ed2c2793b349679`
  and `905ef933a51c26bb3f01fcef4bb4985fa98e00f1dbecd814da3df488bd7c1018`.

## Claim Contract

The ratio denominator is the matching identity or no-action error. Therefore
`1` means no improvement over leaving the latent unchanged, and `0.50` means
that the candidate leaves at most half that error. The `0.50` cutoff is an
absolute functional benchmark, not the model-comparison criterion.

The report must state all of the following:

1. In the dense `0,5,...,355` comparison, SO(2) has lower action and inverse-
   canonicalization ratios on 25/25 patches and 16/16 WSI cluster medians. The
   medians are `0.549565` versus `0.712610` for action and `0.513238` versus
   `0.660824` for canonicalization. This is a consistent descriptive advantage
   for SO(2).
2. H1 and H2 remain `not_supported` because their conjunctive rules also
   require the SO(2) median to be at most `0.50`; both medians narrowly exceed
   it. Failure of that absolute criterion must not erase the paired comparison.
3. Exact `rot90/180/270` is a stronger positive result: aggregate action ratio
   is `0.00000218` SO(2) versus `0.709453` normal, and decoded inverse-
   canonicalization is `0.088854` versus `0.710638`. This verifies the shared,
   parameter-free, full-spatial C4 decoder action on the fixed validation set.
4. Dense end-to-end input commutation favors the normal VAE on 25/25 patches,
   with medians `0.173673` normal and `0.209678` SO(2). Do not claim general
   encoder or autoencoder equivariance.
5. Reflection conclusions remain per transform. Only `flip_diag` passes the
   locked SO(2) checkpoint rule; complete D4 or O(2) equivariance is not
   supported.
6. No learned shared low-dimensional action, local content-pose factorization,
   decoder null space, bundle trivialization, or population generalization was
   demonstrated.

## Report Changes

- Update the executive summary, Section 8, integrated synthesis, limitations,
  presentation conclusions, metric glossary, traceability, and requirements
  appendix.
- Retain all sealed reconstruction, WSI, tissue, bootstrap, and supervised-
  development results byte-for-byte in meaning and numerical content.
- Add report-derived figures from the immutable JSON: paired dense action and
  canonicalization ratios with an untruncated zero baseline and the `0.50`
  absolute reference; and exact C4/reflection medians with the same reference.
- State that qualitative targets are transformed model reconstructions, not
  original images or independent reconstruction-quality references.
- Label every new result as post-hoc fixed-validation evidence, not sealed-test
  evidence.

## Non-Goals

- No inference, training, checkpoint mutation, sealed-test access, threshold
  change, example selection, Kaggle relaunch, issue edit, paper edit, thesis
  edit, or Overleaf synchronization.
- No reinterpretation of the exact C4 decoder result as uniform continuous
  SO(2) equivariance.

## Acceptance Criteria

1. The builder rejects stale Spec 0051 hashes, identities, counts, or decisions.
2. All reported numbers are read from the accepted JSON and reproduce its
   medians and paired counts.
3. DOCX and PDF are rebuilt from `scripts/build_professor_final_report.py`.
4. Every DOCX and PDF page is rendered and visually inspected without clipping,
   overlap, broken tables, missing images, or misleading axis truncation.
5. Accessibility, image, text, syntax, Ruff, formatting, BasedPyright where
   applicable, repository preflight, workspace preflight, and `git diff
   --check` pass, apart from already documented unrelated repository-wide debt.
6. Independent clean-context reviews find no unresolved P0/P1/P2 mathematical,
   statistical, claim-scope, or layout defect.

## Verification Evidence

- The deterministic builder validates the accepted Spec 0051 summary,
  manifest, addendum, contract/spec identities, decision margins, medians,
  paired patch/WSI counts, descriptive intervals, and exact-C4 controls before
  writing the report. A focused fail-closed check confirmed that changing the
  `0.50` benchmark or the `25/25` count is rejected.
- The canonical DOCX/PDF contain 29 Letter pages and 22 inline figures. Every
  rendered page was inspected at original resolution; no clipping, overlap,
  broken table, missing image, misleading axis, or orphaned continuation
  remained.
- DOCX accessibility findings are high/medium/low `0/0/0`; image, heading,
  style, and section audits complete. The PDF is tagged, has no suspects,
  embeds its fonts, and is byte-identical to the PDF emitted by the accepted
  DOCX render.
- Builder Ruff, formatting, and syntax checks pass. The repository-wide Python
  gate reaches only the 23 pre-existing unrelated packaged-probe lint findings.
- Independent clean-context mathematics, statistics, and layout re-reviews
  found no unresolved P0/P1/P2 issue after the fixes.
- Canonical SHA-256 values: DOCX
  `c4c3cf292212830960dad6015bea518902f704a893875d407c27bc5b892ed98d`;
  PDF
  `6107d24c3964e57a1c752382f699e8c6761a4c650732e2b4e5d305d182a95e7a`.

## Related Files

- `scripts/build_professor_final_report.py`
- `reports/professor/informe_final_experimento_eqvae.docx`
- `reports/professor/informe_final_experimento_eqvae.pdf`
- `docs/specs/0048-professor-final-experiment-report.md`
- `docs/specs/0051-decoded-latent-transform-consistency.md`
- `CURRENT.md`
