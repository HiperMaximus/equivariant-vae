# Current Repository Status

Last updated: 2026-09-10

## Handoff

Spec 0053 is the sole active planning workstream. It is a draft roadmap for
functional latent geometry; it authorizes neither inference nor a remote launch.
The current lifecycle of every completed/closed spec is in
docs/specs/README.md. Specs whose bytes were embedded in consumed Kaggle
artifacts remain immutable; their launch-time header is historical evidence,
not live status.

The bounded numerical lineage is closed:

- Specs 0054--0057 consumed their one-shot parents and are negative evidence.
  The failure pattern established finite-difference cancellation at the former
  epsilon=.001 JVP reference; none authorizes a retry or child.
- Spec 0058 completed as
  maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/1. It selected
  epsilon=.008 from .002/.004/.008/.016: global aggregate/worst is
  .00237315/.00332016; .004 fails its worst case .00666581 > .005.
  The immutable launch receipt is
  runs/local/kaggle_launches/maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/v0001.json;
  output and receipt are under
  runs/local/kaggle_outputs/maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5/v0001/.

No Kaggle job is active. This calibration is not a model comparison, full
preflight, or scientific result.

## Frozen Models And Data

- Normal and continuous-SO(2) denoising VAEs completed 60,000 updates and have
  the matched float32[16,32,32] Gaussian posterior-mu target. Checkpoint
  SHA-256: normal
  f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075;
  SO(2) 041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7.
  Do not retrain, tune, or modify either.
- The shared masked-WSI cohort is frozen at 106/23/23 train/validation/test
  WSIs. The sealed reconstruction and tissue tests use the same 23 test WSIs.
  Specs 0017--0025 own membership, storage, and provenance details.
- Sealed-test results never drive tuning, selection, retraining, or retries.

## Accepted Evidence And Interpretation

- Full-test reconstruction is descriptively lower for normal: MAE
  .0628721/.0642299 normal/SO(2), with paired difference
  -.00135779 [-.00221281,+.00004591].
- WSI diagnosis and several low-label tissue point estimates favor SO(2), but
  their prespecified intervals do not establish a general advantage. Only the
  500-label tissue contrast excludes zero.
- Spec 0050 supersedes the defective mixed-sign dense rotation sweep. Its
  corrected one-degree direction misses the locked 10% effect margin and
  reverses at five degrees; no shared reduced action or local
  content--pose factorization was demonstrated. The raw-mu quarter-turn control
  does not favor SO(2).
- Separately, Spec 0051 verifies that the SO(2) decoder realizes the prescribed
  spatial C4 action at exact 90/180/270-degree rotations. This does not imply
  continuous SO(2), encoder equivariance, a learned low-dimensional action, or
  complete D4/O(2) equivariance.
- The advisor report is
  reports/professor/informe_final_experimento_eqvae.{docx,pdf}: 29 Letter pages
  and 22 inline figures. SHA-256: DOCX
  c4c3cf292212830960dad6015bea518902f704a893875d407c27bc5b892ed98d;
  PDF 6107d24c3964e57a1c752382f699e8c6761a4c650732e2b4e5d305d182a95e7a.
  It is visually reviewed and accessible; regenerate for content equivalence,
  not byte identity, because DOCX/PDF container timestamps vary.

## Verification State

- Focused geometry, decoded-transform, preflight, and JVP-calibration suites:
  93 passed (two third-party escnn deprecation warnings).
- Touched scientific source passes Ruff and BasedPyright.
- git diff --check, repository preflight, and workspace preflight pass.
- The full Python gate still reports 23 pre-existing lint findings in two
  packaged tissue/WSI probe files; this batch does not add one.
- Generated Kaggle run.py payloads are ignored; commit their run_template.py,
  metadata, contracts, and tests only.

## Next Authorized Boundary

Start a separate Spec 0053 experiment contract before any new remote action.
It may use epsilon=.008 only as the JVP finite-difference reference and must
lock randomized-linear-algebra policy: sketch width s=k+p, r=128 independent
full-dimensional scalar probes for trace/Frobenius certificates, a 99%
confidence interval for scalar means, and no entrywise reconstruction of J^T J
or averaging singular vectors. It needs focused tests, result-blind
observability, independent review, explicit user authorization, and a unique
one-shot guard. Do not launch it from this completed calibration.
