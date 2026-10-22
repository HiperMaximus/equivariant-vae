# Current Repository Status

Last updated: 2026-10-22

The two VAE checkpoints remain frozen at 60,000 updates. Scientific boundaries
and foreground limitations are in [AGENTS.md](AGENTS.md) and [GOAL.md](GOAL.md).
The 152 external WSI remain reserved for final evaluation.

## WSI classification — Spec 0054

[Scientific spec](docs/specs/0054-cross-validated-mil-dynamics-and-telemetry.md)
· [classical contract](docs/data/spec0054_classical_wsi.json).

The five-fold comparison of six fixed representations is complete. PANTHER
full16 has mean validation macro-F1 0.5545 normal / 0.5587 SO(2); mean-only
SO(2) has CE 1.0206. Normal mean/std retains its highest mean F1, 0.5612.
All PANTHER fits converged, but the train/validation gap remains substantial.
Full results, protocol and audit:
[professor report](reports/professor/spec0054_panther_20261022.md).
Published in commit `75ebc76` and
[MIDL issue update](https://github.com/HiperMaximus/equivariant-vae/issues/8#issuecomment-6350435031).

PANTHER execution: `maximshtefan/eqvae-spec0054-panther/1`, complete.
Source `45f64fe`; entry `48b8667`. Downloaded outputs:
`runs/kaggle/spec0054_panther_v1_outputs/`; inspection tables:
`runs/kaggle/spec0054_panther_v1_review/`. Both construction and classification
ran on Kaggle CPU. Use Kaggle for subsequent experiments, including CPU work.
No follow-up classical-head regularization recipe has been selected.

A new CLS + gated-ABMIL dropout/smoothing/C4 recipe is committed as `7ad0601`
in the existing model, runner and fold0 contract. It uses fresh classifier
initialization; the prior v12 states below remain the reference. Numeric values
are in `docs/data/spec0054_fold0_run.json`; the scientific definition and
comparison limits are in the Spec0054 candidate section. Submitted as
`maximshtefan/eqvae-spec0054-mil-fold0/13` on2026-10-22 at16:15 Bogota;
initial API status is RUNNING. D4 remains deferred. CPU verification covers all14 affected
tests, including direct smoothing gradients, exact C4 permutations, dropout
evaluation behavior and bitwise continuation after interrupted assessment.
Ruff, targeted BasedPyright, compilation and existing kernel build/validation
pass. GPU throughput and compiled-CUDA behavior remain to be inspected in v13 outputs.

The previous CLS + gated-ABMIL fold0 run is paused after
`maximshtefan/eqvae-spec0054-mil-fold0/12`. Resume input is
`maximshtefan/eqvae-spec0054-cls-regularized-fold0-resume/3`.

| Branch | Updates | WSI presentations | Epoch / cursor |
| --- | ---: | ---: | --- |
| normal_vae | 5228 | 20,912 | 72 / 176 |
| so2_vae | 5182 | 20,728 | 71 / 280 |

Both completed epoch70 assessment; both best-by-CE checkpoints are epoch55.
Best validation CE/F1: normal 0.96126/0.60816, SO(2) 0.85010/0.61773.
These are exploratory fold0 results, distinct from the five-fold fixed-head
comparison. Review evidence: `runs/local/spec0054_cls_fold0_regularized_v12_review/`;
[professor update](https://github.com/HiperMaximus/equivariant-vae/issues/8#issuecomment-6149719046).

The submitted `kaggle/kernels/spec0054_mil_fold0/abmil.py` starts a
fresh run pinned to `7ad06010416a789a6b96e612919203c162adc23e` with `RESUME=False`.
Only the twelve original latent inputs remain mounted; the old resume dataset
is removed. Entry commit is `ac196da`; normal/SO2 are assigned physical GPU0/1.
The shared session cutoff is2026-10-23T04:00:03-05:00, reserving15 minutes.
Inspect around2026-10-22T17:00-05:00 for startup/first-update evidence; do not
keep polling the confirmed running job. Latest states are saved every144 WSI
presentations and at the deadline;105 epochs will likely require continuation.
Previous-recipe continuation must use its original pinned source and contract.
The105-epoch horizon and five-fold MIL evaluation remain unfinished.

## Functional geometry — Spec 0053

[Scientific spec](docs/specs/0053-functional-riemannian-latent-geometry.md).
Stage A1 is accepted. Stage A2d already continued all four bridge states to
step4096; do not repeat the older step2048-to4096 plan. Source: `b630b6e`.
Outputs and saved optimizer states:
`runs/kaggle/functional_geometry_stage_a2b_v5/functional_geometry_stage_a2d/`.

All four saved best states are at step4096 with unchanged LR; traces still
descend. SO(2) bridge E/endpoint-RMS-squared is 19.002/20.986 (ranks0/12),
and decoded bottlenecks are 1.796/1.879 times the endpoint gap. Convergence and
a connected decoder fiber have not been established. No step6144 execution
was identified in the latest review; extending only the SO(2) bridges remains
a proposal. K64 reoptimization and decoded-curve stability follow convergence.

The existing restricted-chart shooting used the original 512-step sides and
an initial tangent without endpoint fitting. Its rank losses and sensitivity
to time refinement do not establish a closed geodesic cycle. Full-latent
shooting, continuous prescribed-orbit analysis and fiber continuation remain
pending; holonomy interpretation requires the relevant validated structure.

## Paper handoff

Latest published manuscript sync: `bbe8fa0`, imported main.tex v963 and
references.bib v5 on 2026-10-16. Source/PDF and current writing handoff:
[manuscript README](paper/midl2027/README.md).
The 22-page PDF compiled successfully; all 19 professor threads retain their
states (7 open, 12 resolved). Only the professor closes comments.

Read live Overleaf before further manuscript changes, preserving professor
edits. Import/build/publication procedure:
[Overleaf handoff](docs/overleaf_sync_workflow.md). The manuscript still has
pending 152-WSI results; the historical 23-WSI evaluation remains separate.
