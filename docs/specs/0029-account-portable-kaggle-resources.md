# Spec 0029: Account-Portable Kaggle Resources

Status: implemented
Implementation readiness: accepted locally
Owner/workstream: Kaggle workflow portability
Last updated: 2026-09-02

## Purpose

Make new Kaggle launches work under whichever collaborator is authenticated,
without changing the identity of datasets, kernels, or versions owned by other
accounts. Separate the acting account from every resource locator.

## Non-Goals

- Do not rewrite committed historical kernel metadata, generated scripts,
  receipts, manifests, or sealed evidence.
- Do not republish completed datasets or kernels under a collaborator account.
- Do not perform a remote Kaggle read, write, or download in this workstream.
- Do not change experiment, split, model, or dataset contents.

## Inputs And Data Contract

- An authenticated Kaggle CLI identity from OAuth credentials,
  `KAGGLE_USERNAME`, or legacy `kaggle.json` credentials.
- A local kernel directory with valid `kernel-metadata.json`.
- Each source is an exact canonical locator owned by that resource:
  `owner/slug`, with a positive integer version where the API supports one.
- Source owners may differ from the authenticated actor and from one another.

## Outputs And Acceptance Artifacts

- An ephemeral upload snapshot whose kernel `id` is
  `<authenticated-owner>/<original-kernel-slug>`.
- Unchanged `dataset_sources`, `kernel_sources`, `model_sources`, and
  `competition_sources` in that snapshot.
- A local immutable launch receipt recording actor, Kaggle-returned canonical
  kernel id, required accepted version, versioned kernel reference, source
  locators, and hashes of the source and uploaded metadata.
- A retrieval receipt beside each newly downloaded kernel output, kernel source,
  or dataset, recording its resource owner/slug/version and per-file hashes.
- Unit and shell-integration tests covering two actors and mixed-owner inputs.

## Architecture Or Workflow Contract

Identity has two separate roles:

1. **Actor identity** owns a newly pushed kernel or newly created dataset. It is
   derived from the active local Kaggle credentials and is never inferred from
   any input resource.
2. **Resource identity** is the full canonical locator returned by Kaggle or
   recorded in metadata/receipts. It is never rewritten to the actor owner.

The generic push path validates and guards the original package first, then
copies it to a temporary directory and changes only the owner component of the
snapshot's top-level kernel `id`. The repository package remains unchanged.
After Kaggle explicitly returns both an accepted positive version and its
canonical `/code/owner/slug` URL, subsequent status/log/output/pull operations
use that exact versioned reference from the receipt. An exit-zero response
without both fields fails closed and creates no receipt.

Local credential discovery must emit only the username. It must never print,
copy into receipts, or expose an API key, access token, or refresh token. An
empty, malformed, or unavailable username fails closed with an authentication
hint.

Existing owner-qualified source locators are intentional provenance, including
resources from `maximusshtefan`, `maximshtefan`, `sohier`, and any future
collaborator. There is no repository-wide `KAGGLE_OWNER` for resources.
Provenance-aware downloads require a positive explicit version and a new local
directory, preventing mutable-latest aliases or mixed files from losing their
origin.

## Config Contract

- `KAGGLE_PUSH_CONFIRMED=1` remains required for a remote push.
- `KAGGLE_FULL_DATASET_CONFIRMED=1` remains required when sources are attached.
- `KAGGLE_REMOTE_CONFIRMED=1` remains required for remote reads/downloads.
- `KAGGLE_USERNAME`, when used by legacy auth, is an actor credential field,
  not a resource-owner default.
- Launch receipts default under `runs/local/kaggle_launches/`; callers may
  select another local receipt root without changing remote identity.

## Acceptance Criteria

1. OAuth, environment, and legacy credential paths return only a validated
   authenticated username; missing identity fails closed.
2. For source metadata `alice/job`, actor `bob` uploads a snapshot with id
   `bob/job`; the source directory and all source arrays are unchanged.
3. Mixed sources such as `alice/data`, `carol/checkpoint`, and
   `dave/upstream-kernel` survive snapshotting exactly and in order.
4. A successful push records the exact canonical id and accepted version from
   Kaggle; absent or ambiguous confirmation creates no receipt and cannot wait.
5. No secret value occurs in stdout, stderr, argv, snapshot metadata, or the
   receipt.
6. Historical kernel directories and receipts have no content changes from
   this implementation.
7. Dataset, kernel-output, and kernel-source downloads record the exact
   owner/slug/version plus SHA-256 and byte size for every downloaded file.

## Tests And Verification Commands

```bash
.venv/bin/pytest -q tests/test_kaggle_resources.py tests/test_kaggle_oauth_exec.py
./scripts/python_quality.sh
./agent_preflight.sh
git diff --check
```

No command in this acceptance section authorizes a Kaggle remote operation.

## Implementation Blockers

None. The actor/resource distinction and fail-closed behavior are locked.

## Verification Result

Implemented 2026-09-02 without any remote Kaggle operation. The focused
portability/OAuth suite passes 31 tests. The full repository quality gate passes
with 1,069 tests passed, one skipped, and zero type errors/warnings. Two
independent adversarial reviews found no remaining P0/P1 blocker after fixes for
exit-zero push errors, server-normalized slugs, invalid-source omission warnings,
first launches, CLI override bypasses, pre-download version validation, and
streaming multi-gigabyte hashes.

## Known Risks

- Kaggle may change local credential formats or push confirmation text.
- Generic account-portable pushes forbid CLI overrides; guarded metadata owns
  accelerator and runtime settings so `--path` cannot bypass the snapshot.
- A shared private resource can still be unreadable if the owner did not grant
  access; preserving its locator cannot create permissions.
- Versionless source locators can resolve to a newer remote version. New
  receipts should save a version whenever Kaggle exposes it.

## Adversarial Checks

- Put a source owner's name in every source list and prove none are rewritten.
- Use an actor different from both the original kernel owner and all sources.
- Inject malformed usernames, locators, versions, and credential JSON.
- Put secret-shaped sentinel values in credentials and assert they never leak.
- Compare source-tree hashes before and after portable snapshot generation.

## Open Questions

None for implementation. A future training-package spec must select exact
shared source versions before professor-facing execution.

## Related Files

- `scripts/kaggle_kernel.sh`
- `scripts/kaggle_oauth_exec.py`
- `src/eqvae/kaggle_resources.py`
- `docs/kaggle_cli_workflow.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
- `GOAL.md`
