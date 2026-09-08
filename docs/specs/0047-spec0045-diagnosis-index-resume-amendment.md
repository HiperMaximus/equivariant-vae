# Spec 0047: Spec 0045 Diagnosis-Index Resume Amendment

Status: implemented / complete; one-use resume consumed
Owner/workstream: frozen-VAE full-test scoring provenance
Last updated: 2026-09-07

## Reason

The authenticated Spec 0045 Kaggle output completed and its exclusive
pre-score claim was created before the local diagnosis oracle was opened. The
first scoring pass then stopped before parsing reconstruction values or
producing any aggregate because the scorer required an incorrect numeric
diagnosis-index convention:

- old scorer: `HGSC=0, LGSC=1, EC=2, CC=3, MC=4`;
- frozen oracle: `CC=0, EC=1, HGSC=2, LGSC=3, MC=4`.

All 67,138 remote rows match the frozen oracle on atlas row, WSI, coordinates,
split, and diagnosis name. The numeric index is redundant output metadata; it
does not select patches, define diagnosis groups, enter a reconstruction
metric, or affect the paired bootstrap. No scored output or staging directory
exists. The user explicitly approved this narrow diagnosis-index scorer/resume
amendment on 2026-09-07.

## Immutable Evidence

- Input contract SHA-256:
  `b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1`.
- Launch receipt SHA-256:
  `bd7360a9ec6831b107b7b2235cfae37060553db6434f064d949e0129788c6f3f`.
- Output receipt SHA-256:
  `ae50e9548d1efce6c396e2102d12233181709618ed0fa50e126c701c88a00afb`.
- Remote metric SHA-256:
  `7166a668e4f7193fd3f5dddd05455b4fa9bc7fe309a92005f534fa5fb8a0c43f`.
- Frozen oracle SHA-256:
  `1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc`.
- Consumed original pre-score claim SHA-256:
  `b6fd2a34b5f21aed4c930965388a3f52db4650444c29adb5b54c7bab4d177f86`.
- Frozen scorer SHA-256, preserved unchanged:
  `55a0249661d5f972e2be42724d5afac797997f26c75f57d14988da52b01936c1`.
- Diagnosis-index adapter SHA-256:
  `6df577d3c6f2e70cb681fc97742baa3f4d35faf437ae660e85ba7f6763cf5e0c`.
- One-use resume module SHA-256:
  `493b7fcd0fe9af4837a161c4f6d79ed3cffc344b5e92797b227daa02011b77f5`.
- Frozen scorer vector SHA-256:
  `39dff52fe70da85865c7c1f9ea269cbbd7e2ab10e5fdea2631372d4ff17191bf`.
- Machine-readable amendment:
  `docs/data/spec0047_spec0045_diagnosis_index_resume_amendment.json`,
  SHA-256
  `0641f572e497f44cec51565e7f0758fbc2065e9d56b59619e2acc7ee529b1a96`.

## Contract

- Preserve the scorer byte-for-byte. A tiny adapter first requires the mapping
  already present in the immutable oracle, projects only its redundant numeric
  index to the scorer's frozen internal convention, and leaves the oracle's
  actual index in joined output. Preserve diagnosis strings and every data,
  model, reconstruction, aggregation, bootstrap, output, and limitation
  contract.
- The unchanged scorer vector must reproduce its exact expected result hash
  under the amended scorer, proving no scientific output changes.
- Preserve the original pre-score claim byte-for-byte. A separate exclusive
  Spec 0047 resume claim must be created before reopening metrics or labels.
- The resume claim binds the old claim, amendment, launch/output receipts,
  remote metric, oracle, frozen scorer, index adapter, resume module, and scorer
  vector.
- The resume route accepts no path override and may create only
  `runs/local/vae_test_reconstruction_scored_v1`. Any second resume, byte
  mutation, different path, mapping, or authority fails closed.
- No remote retry or replacement launch is authorized.

## Acceptance

Focused tests must prove the frozen oracle mapping, unchanged scorer-vector
result, exact amendment authentication, prior-claim mutation rejection,
exclusive resume-claim behavior, exact route paths, and successful synthetic
post-claim scoring mechanics. Touched Ruff, format, BasedPyright, shell syntax,
diff hygiene, both preflights, and independent clean-context review must have
no unresolved P0/P1 before invoking the one-use resume route.

## Completion

The exact resume route completed once on 2026-09-07 without any remote retry.
The external resume-claim SHA-256 is
`83ba8dafa3562fc8b20c9aa121e111de4578e308166e21744f9e7527610ea1b1`.
The atomic scored package is `runs/local/vae_test_reconstruction_scored_v1`;
its manifest SHA-256 is
`68cb479ed3b583bfde4163e1860bbf49265e5981ae05c06f4deeb79570ec572f`.
Joined output retains the oracle-native diagnosis indices. Independent audit
recomputed all headline statistics and 10,000 bootstrap draws exactly and found
no P0/P1 issue. This authority is permanently consumed.
