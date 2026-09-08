# Spec 0042: Spec 0041 Kaggle Slug-Normalization Amendment

Status: locked administrative amendment
Owner/workstream: sealed MIL test evaluation provenance
Last updated: 2026-09-05

## Reason

Kaggle accepted the one authorized Spec 0041 launch but normalized requested
kernel ID `maximshtefan/eqvae-local-global-mil-test-evaluation` from its title
to canonical reference
`maximshtefan/eqvae-label-blind-mil-test-evaluation/1`. The immutable launch
receipt SHA-256 is
`b1d7f1fd90900a0839c1b191afef5b825f8fcb464c16a4234c3e4864fa7fa243`.
It proves the requested/original ID, actor, version, source locators, metadata
and complete source/upload file manifests, including exact `run.py` SHA-256
`1eebd2f951a89ab691c2bf44efea5ed2c0bb5f71a8fac63b387c788b000c1bad`.

No output or prediction was retrieved before this amendment. After the agent
explained the mismatch, the user explicitly authorized only this narrow
administrative scorer amendment on 2026-09-05. No replacement launch is
authorized.

## Contract

- Preserve immutable Spec 0041, input-contract SHA-256
  `1b9d11d41e6cdb5bbfcc774b4a7a101819092327d883e4b1031f7d65c634f8e7`,
  original remote scorer SHA-256
  `89654cc6a6c6b0c174f3d073037e84406283bb2121c52568a1cbcb5a6e1d3194`,
  model/checkpoint/test geometry, metric implementation, bootstrap, labels and
  one-shot claim unchanged.
- The amended local scorer SHA-256 is
  `c78230081c3e8279343c6f2b45203dc72433c5e8c0cca6b4b10d99110e2cf640`.
  Its only semantic change is accepting the exact normalized canonical
  reference above while still requiring the requested/original ID and every
  existing byte/source/checkpoint/run-contract binding.
- The machine-readable amendment is
  `docs/data/spec0042_spec0041_slug_normalization_amendment.json`, SHA-256
  `0f63933d656cb9c8143762a6b9dabbf42d42b2b17c4670bd7f7f9a325826b3eb`.
  The scorer must authenticate it and persist its hash in the exclusive
  pre-score claim before opening labels.
- Any other actor, requested ID, accepted slug, version, receipt, code,
  metadata, input, source, checkpoint, graph, bag size or scorer change fails
  closed. Test results remain unavailable for selection or tuning.

## Acceptance

Focused tests must cover exact normalized-receipt acceptance and mutation
rejection, original remote-scorer binding, amended local-scorer binding, and
exclusive pre-score persistence. Ruff, BasedPyright, shell syntax, diff hygiene
and independent clean-context review must have no unresolved P0/P1 before
output retrieval.
