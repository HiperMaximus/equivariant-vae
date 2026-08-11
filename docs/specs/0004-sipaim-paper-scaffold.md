# Spec 0004: SIPAIM Paper Scaffold

Status: draft active / scaffold slice implemented
Implementation readiness: verified normal-VAE evidence is available for bounded
method/result reporting; matched comparison claims still require continuous `SO(2)` and
downstream evidence
Owner/workstream: SIPAIM 2026 paper planning
Last updated: 2026-08-11

## Purpose

Maintain the paper scaffold while the verified normal-VAE control advances to the
continuous-`SO(2)` and downstream comparison stages.

The paper should follow the user's thesis framing: first learn representations
from many unlabeled histopathology patches with an autoencoder/VAE-style model,
then train a supervised classifier on the learned embeddings for WSI labels.
The comparison is between a non-equivariant encoder and a continuous
`SO(2)`-equivariant encoder under the same data, latent, objective, and
evaluation protocol.

## Non-Goals

- Do not claim that the equivariant model improves reconstruction,
  classification, latent quality, or robustness before full evidence exists.
- Do not claim selected runtime, final batch size, AMP policy, compile policy,
  or full-run readiness from capped Kaggle pretests.
- Do not edit the thesis repo; it is a reference source only.
- Do not sync to Overleaf in this scaffold slice.
- Do not close or resolve GitHub issues.

## Inputs And Data Contract

- Paper source: `paper/sipaim2026`.
- Thesis reference source: `/home/n00b1337/Documents/Max/Tesis/Tesis/main.tex`,
  figures, and bibliography files.
- Current experiment source: `GOAL.md`, `CURRENT.md`,
  `docs/repo_goal_and_requirements.md`,
  `docs/issue_image_inventory.md`, and active specs.
- Dataset narrative: UBC-OCEAN patch shards, `256x256` RGB tiles normalized to
  `[-1, 1]`; train/validation patch evidence exists, but sealed masked-WSI
  test evidence is still pending.

## Outputs And Acceptance Artifacts

- `paper/sipaim2026/main.tex` contains a compile-safe paper outline with title,
  keywords, section headings, method placeholders, and claim boundaries.
- `paper/sipaim2026/references.bib` contains the first local bibliography
  entries needed by the scaffold.
- Paper-local figure copies live under `paper/sipaim2026/figures/` so the paper
  subtree is self-contained for GitHub and Overleaf.
- `paper/sipaim2026/sipaim2026.pdf` is refreshed if local LaTeX compilation is
  available.

## Related Requirements And Evidence

- `GOAL.md`: paper target and must-not-lose artifacts.
- `docs/repo_goal_and_requirements.md`: required metrics, figures, and tables.
- `docs/issue_image_inventory.md`: dashboard, reconstruction, rotated-input,
  and EQ-VAE-style latent visualization requirements.
- `docs/specs/0001-translatable-normal-vae-baseline.md`: model/data/runtime
  contract and current Kaggle blockers.
- Thesis reference anchors: digital pathology motivation, semi-supervised
  encoder-decoder framing, equivariance theory, and related work.

## Architecture Or Workflow Contract

The scaffold may write explanatory prose in the user's direct thesis style:
limited labels are the practical bottleneck; unlabeled patches are available;
the unsupervised encoder learns the representation; a classifier then uses the
embeddings for WSI prediction.

The first paper outline should fit a four-page IEEE conference paper:

1. Introduction.
2. Background and Related Work.
3. Proposed Semi-Supervised Comparison.
4. Experimental Protocol.
5. Expected Results and Reporting Plan.
6. Current Status and Limitations.
7. Conclusion.

The paper must mark pending evidence explicitly. Runtime pretest evidence can
be mentioned only as implementation status, not as model performance.

## Config Contract

No experiment config changes are authorized by this paper scaffold.

## Acceptance Criteria

1. The paper compiles or any compile failure is recorded in `CURRENT.md`.
2. The scaffold contains no final comparative result claims.
3. Figure paths are local to `paper/sipaim2026`.
4. The bibliography compiles with the citations used in `main.tex`.
5. `CURRENT.md` records what was scaffolded and what remains pending.

## Tests And Verification Commands

```bash
./scripts/python_quality.sh
./scripts/sipaim_overleaf_sync.sh compile
./scripts/agent_preflight.sh
```

`./scripts/python_quality.sh` is needed only when Python files changed in the
same workstream; it was already required by the Kaggle diagnostics change.

## Implementation Blockers

- Full non-equivariant VAE training and selected runtime are not complete.
- The continuous `SO(2)` model and its full comparison are not complete.
- The downstream WSI classifier protocol is not locked.
- The sealed masked-WSI test shard is not generated.
- Required result figures and tables are still placeholders.

## Known Risks

- A scaffold can accidentally read like a completed paper. Keep pending claims
  visible until evidence exists.
- Reused thesis figures may need stylistic cleanup for IEEE column width.
- Thesis references use mixed BibLaTeX/Zotero fields; paper bibliography
  entries should be normalized for IEEE BibTeX as they are added.

## Adversarial Checks

- Search for language that claims improvement before results.
- Verify no figure path points outside `paper/sipaim2026`.
- Verify no Overleaf remote command was run.
- Verify the thesis repo remains unmodified.

## Open Questions

- Which WSI classifier head will be used first: linear probe, small MLP, or
  attention/MIL aggregator?
- Will encoder embeddings be frozen for the first classifier result, or will a
  fine-tuning ablation be allowed later?
- Which thesis figures should be redrawn before final submission?

## Related Files

- `paper/sipaim2026/main.tex`
- `paper/sipaim2026/references.bib`
- `paper/sipaim2026/figures/`
- `docs/specs/README.md`
- `CURRENT.md`
