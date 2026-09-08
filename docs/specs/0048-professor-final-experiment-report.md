# Spec 0048: Professor Final Experiment Report

Status: implemented / complete and verified
Implementation readiness: complete
Owner/workstream: advisor-facing synthesis of completed experiment evidence
Last updated: 2026-09-08

## Purpose

Create a presentation-ready Spanish report that answers the evaluation requests
recorded in GitHub issues #2, #3, #4, and #6. The report must synthesize the
completed normal-VAE and continuous-`SO(2)` evidence without strengthening
exploratory results into confirmatory claims. It also incorporates the
separately authorized fixed-validation dense-360 population check requested
after the initial report.

## Non-Goals

- No training, checkpoint selection, sealed-test inference, or metric change.
- The only new inference is the one Spec 0038 private T4 pass over the frozen
  checkpoints, fixed validation 25, and every integer rotation angle.
- No paper, thesis, or Overleaf update. The professor-facing issue #6 comment
  was updated under separate explicit authorization.
- No claim that either representation is universally superior.
- No use of patch-level dispersion as inferential uncertainty.
- No new WSI inference or synthetic attention map. Spatial attribution is
  described as a separately scoped post-hoc analysis because the accepted
  score artifacts retain WSI outputs, not patch-level attribution weights.

## Inputs And Data Contract

- Professor-requested fixed-25 package:
  `runs/local/professor_metrics_v1`.
- Sealed VAE reconstruction score:
  `runs/local/vae_test_reconstruction_scored_v1`.
- Sealed WSI MIL score:
  `runs/local/ubc_ocean_mil_test_scored_v1`.
- Sealed tissue label-efficiency score:
  `runs/local/tissue_test_scored_v1`.
- Rotation-orbit and latent-PCA package:
  `runs/local/frozen_vae_rotation_orbits`.
- Immutable dense-360 source:
  `maximshtefan/eqvae-fixed25-dense-rotation-population/1`, downloaded with a
  receipt under `runs/kaggle/fixed25_rotation_population_v1`.
- Canonical interpretation and limitations: `CURRENT.md`, `GOAL.md`, Specs
  0038, 0040, 0041, 0043, 0044, and 0045.

All scientific numbers and existing figures remain immutable. The report
builder reads the accepted local artifacts and embeds selected figures; it does
not recompute model outputs.

## Outputs And Acceptance Artifacts

- `reports/professor/informe_final_experimento_eqvae.docx`.
- `reports/professor/informe_final_experimento_eqvae.pdf`.
- `scripts/build_professor_final_report.py` for deterministic regeneration.
- Final Spanish issue update:
  `https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496`.
- Rendered page PNGs only as ignored QA intermediates under `.agent_tmp/`.

## Related Requirements And Evidence

- Issue #2: baseline result plots.
- Issue #3: SSIM, MAE, MSE, PSNR, standard deviation, `n`, and boxplots.
- Issue #4: fixed-25 reconstructions, quarter-turn comparisons, all metrics,
  boxplots, and EQ-VAE-style latent visualization.
- Issue #6: repeat the same evaluation for the continuous-`SO(2)` VAE.
- `docs/issue_image_inventory.md` defines the dashboard, reconstruction-grid,
  and latent-PCA presentation requirements.

## Architecture Or Workflow Contract

- Write in Spanish for an advisor audience and identify the project and author
  from the existing manuscript metadata.
- Lead with the integrated conclusion, then explain design, evaluation, results,
  limitations, and presentation conclusions.
- Distinguish fixed validation 25, full sealed reconstruction test, WSI-level
  diagnosis test, and tissue-patch test on every relevant table and caption.
- State the direction of every paired difference. Keep confirmatory MAE separate
  from secondary reconstruction endpoints and exploratory diagnostics.
- Use the existing figures at readable scale. Include both the selected-patch
  0--359-degree orbit and the all-25 population grid sampled every 1 degree.
  State that the latter supports lower raw-space local-linearity ratio for
  SO(2) in 25/25 pairs while step-size uniformity and PCA planarity do not
  consistently favor it; the comparable fixed-25 exact-quarter residual also
  does not favor SO(2).
- Generate five deterministic summary charts from accepted evidence: supervised
  development/selection with online train-loss context, overall sealed WSI
  metrics, row-normalized WSI
  confusion matrices with per-class F1, tissue macro-F1 label efficiency, and
  tissue per-class F1 label efficiency.
- Explain the supervised class breakdowns without strengthening point estimates
  into confirmatory class-wise claims. Always show the WSI support
  CC/EC/HGSC/LGSC/MC = 5/6/8/2/2 and the tissue support constraints.
- Add a concise attribution boundary: an honest WSI patch map would require a
  frozen-checkpoint post-hoc forward/attribution pass and spatial projection;
  raw attention alone is not assumed to be a faithful class explanation.
- Use Letter portrait, black titles/headings, restrained blue/orange model
  colors, inline images, visible table borders, meaningful alt text, and page
  numbers.
- Produce the PDF from the visually accepted DOCX so both deliverables match.

## Config Contract

The builder accepts `--repo-root`, `--output-docx`, and an optional
`--work-dir`. Defaults resolve to the canonical repository and report paths.

## Acceptance Criteria

1. The report includes the professor-requested reconstruction metrics with
   mean, population SD, and `n`, plus fixed-25 and sealed-test boxplots.
2. The report includes the training dashboard, 25-patch reconstruction grid,
   quarter-turn transformed-latent comparison, and latent-PCA view.
3. The sealed reconstruction table reports 67,138 patches from 23 WSIs and the
   primary normal-minus-`SO(2)` MAE interval exactly.
4. The WSI MIL and tissue label-efficiency sections reproduce the accepted
   metrics and uncertainty without implying a general winner.
5. Limitations name one training trajectory, 23 test WSIs, low LGSC/MC support,
   five necrosis-bearing tissue-test WSIs, and uncertainty scopes.
6. No text describes fixed-25 validation as sealed test evidence or a PCA image
   as a performance metric.
7. DOCX and PDF render with no clipping, overlap, broken tables, missing images,
   stale placeholders, or unreadable captions.
8. Accessibility and image audits have no unresolved high-severity findings.
9. Two independent clean-context reviewers find no P0/P1 claim or requirement
   defect after fixes.
10. The report includes the accepted dense latent orbit and explains both its
    visually regular SO(2) loop and the conflicting fixed-25 residual evidence.
11. The report includes WSI confusion/per-class F1 and tissue per-class F1
    views, with sample support and uncertainty boundaries stated locally.
12. No WSI attention or patch-attribution map is fabricated from artifacts that
    do not contain patch-level attribution values.
13. Supervised development curves clearly separate validation used for
    checkpoint selection from MIL online train cross-entropy averaged over each
    53-update window; the latter is labeled as an optimization diagnostic, not
    a fixed-checkpoint train evaluation or sealed-test evidence.
14. The report includes all 25 fixed-validation orbit pairs sampled at every
    integer degree and reports the realized raw-space medians and favorable
    counts without calling PCA appearance a confirmatory metric.

## Tests And Verification Commands

```bash
python scripts/build_professor_final_report.py
python render_docx.py reports/professor/informe_final_experimento_eqvae.docx \
  --output_dir .agent_tmp/professor_report_render --emit_pdf
python scripts/images_audit.py reports/professor/informe_final_experimento_eqvae.docx
python scripts/a11y_audit.py reports/professor/informe_final_experimento_eqvae.docx
pdfinfo reports/professor/informe_final_experimento_eqvae.pdf
pdftotext reports/professor/informe_final_experimento_eqvae.pdf -
./scripts/agent_preflight.sh
git diff --check
```

## Verification Evidence

- The dense-360 extension renders as a matching 20-page Letter DOCX/PDF with
  15 inline figures; the all-25 orbit appears as readable Figures 8a/8b.
- All 20 rendered pages were inspected at full size and by contact sheet with
  no clipping, overlap, broken table, missing image, or unreadable caption.
- The DOCX accessibility audit reports zero high, medium, or low findings; the
  image audit finds all 15 expected inline figures. The 20-page PDF is tagged,
  Letter-sized, declares `es-CO`, has a structure tree, and is byte-identical to
  the PDF emitted from the visually inspected DOCX render.
- Builder syntax and Ruff checks pass. Two independent clean-context final
  reviews found no P0/P1/P2 statistical, requirement-coverage, readability, or
  layout defect. The builder also recomputes and validates the dense-orbit
  medians, paired counts, finiteness, angle grid, patch hashes, and checkpoint
  hashes before writing claims.
- The repo-wide Python gate still stops only on the 23 known unrelated legacy
  findings in two packaged probe files; focused checks for the report builder
  pass.
- No training, checkpoint selection, sealed-test evaluation, paper edit, or
  thesis edit occurred. The separately authorized fixed-validation inference
  and issue #6 update are recorded by their exact remote identities. The issue
  remains open and the final comment embeds 12 verified attachment URLs.
  Canonical report SHA-256 values are
  `0ee1a866ed98dda1478a0b6da3568f5990f8a1c7af8319157107573369d58da1`
  (DOCX) and
  `3524c206d484b01591ac025936c44a986449ffeb6ac70de4876a1434a9653451`
  (PDF).

## Implementation Blockers

- None. All required experiment evidence is complete and local.

## Known Risks

- Combining several populations in one report can blur their statistical units.
- Visual examples can be mistaken for population evidence.
- Secondary endpoints and exploratory diagnostics can appear confirmatory if
  uncertainty labels are omitted.
- Large figures can create poor page breaks in Word and PDF.

## Adversarial Checks

- Verify every number directly against the accepted JSON or CSV artifact.
- Search for language that implies significance, proof, or general superiority.
- Check every caption for population, sample count, and interpretation boundary.
- Inspect every rendered page at full size.

## Open Questions

- None for the report artifact or issue #6 update. Paper, thesis, and Overleaf
  integration remain separate user-authorized work.

## Related Files

- `CURRENT.md`
- `GOAL.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/specs/0044-professor-metrics-and-plots.md`
- `docs/specs/0045-frozen-vae-test-reconstruction-evaluation.md`
