# Spec 0046: Spec 0045 Kaggle Slug-Normalization Amendment

Status: implemented and independently accepted administrative amendment
Owner/workstream: frozen-VAE full-test reconstruction provenance
Last updated: 2026-09-06

## Reason

Kaggle accepted the one authorized Spec 0045 launch but normalized requested
kernel ID `maximshtefan/eqvae-frozen-vae-test-reconstruction` from its title to
canonical reference
`maximshtefan/eqvae-frozen-vae-full-test-reconstruction/1`. The immutable
launch receipt SHA-256 is
`bd7360a9ec6831b107b7b2235cfae37060553db6434f064d949e0129788c6f3f`.
It proves the requested/original ID, actor, accepted version, source locators,
metadata, and complete source/upload manifests, including exact `run.py`
SHA-256
`03bf2a58a62050ec0481961ce713e326a9b2bfa3ffd935361b90bd4972f8ef94`.

No output or reconstruction metric was retrieved before this amendment. After
the mismatch was explained, the user explicitly approved this narrow
administrative amendment on 2026-09-06. No replacement launch is authorized.

## Contract

- Preserve immutable Spec 0045, input-contract SHA-256
  `b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1`,
  original remote reporter SHA-256
  `fe3d4830ea33dc9c7dd283fa60211f854ee78160068bace2457b4d35684a44d7`,
  frozen models, complete 67,138-patch population, reconstruction definitions,
  primary pooled-patch MAE, secondary metrics, paired WSI bootstrap, diagnosis
  breakdown, label boundary, and one-shot claim unchanged.
- The amended local reporter SHA-256 is
  `6b3283ec7d8be2647e577d65cd109b11f33f7163a3cd8ba4c710849407e3c302`.
  Its only semantic change is accepting the exact normalized canonical
  reference above while still requiring the requested/original ID, actor,
  version, complete source locator set, metadata hash, source/upload equality,
  frozen core kernel bytes, input contract, and output receipt.
- The machine-readable amendment is
  `docs/data/spec0046_spec0045_slug_normalization_amendment.json`, SHA-256
  `4eb160368e8eb8413171e77a6c30edd172a13693cf9f1bda625c7b60a261e147`.
  The reporter must authenticate it and persist its hash plus both reporter
  hashes in the exclusive pre-score claim before opening diagnosis labels.
- The output route accepts only the exact canonical reference and immutable
  launch-receipt hash above. Any other actor, requested ID, accepted slug,
  version, receipt, source, input, code, metric, or reporter fails closed.

## Acceptance

Focused tests must cover exact normalized-receipt acceptance, requested-ID and
version mutation rejection, original/amended reporter binding, unchanged
scorer-vector output, and exclusive pre-score persistence. Touched-file Ruff,
format, BasedPyright, shell syntax, diff hygiene, both preflights, and an
independent clean-context review must have no unresolved P0/P1 before output
retrieval.

Acceptance completed on 2026-09-06: 19 focused tests passed; touched Ruff,
format, BasedPyright, shell syntax, diff hygiene, and both preflights passed.
Independent clean-context review found no P0/P1 and inspected no Kaggle output
or metric.
