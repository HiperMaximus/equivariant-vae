# Spec 0005: Overleaf Empty Project Initialization

Status: implemented
Implementation readiness: narrow workflow fix only
Owner/workstream: SIPAIM Overleaf sync workflow
Last updated: 2026-06-19

## Purpose

Handle the first-sync edge case where a new Overleaf project already has a
`master` commit, but that commit has an empty tree and no common history with
the local `paper/sipaim2026` subtree.

## Non-Goals

- Do not add a general force-push path.
- Do not bypass `scripts/sipaim_overleaf_sync.sh`.
- Do not push the full repo to Overleaf.
- Do not overwrite a nonempty Overleaf project.
- Do not store or print Overleaf tokens.

## Contract

`scripts/sipaim_overleaf_sync.sh push` must:

1. keep the existing explicit confirmation, remote URL, clean-subtree, and
   credential-free URL guards;
2. compile and refresh `paper/sipaim2026/sipaim2026.pdf`;
3. create a subtree split from exactly `paper/sipaim2026`;
4. sanity-check that the split contains `main.tex` and `sipaim2026.pdf`;
5. reject split trees containing repo-root paths such as `src/`, `docs/`,
   `kaggle/`, `scripts/`, `tests/`, or nested `paper/`;
6. try a normal push first;
7. if the normal push is rejected and the branch is `master`, fetch the exact
   Overleaf branch commit and prove its tree is the canonical empty tree;
8. initialize only that empty Overleaf `master` using
   `--force-with-lease=refs/heads/master:<observed-empty-commit>`;
9. abort for any nonempty Overleaf branch.

## Acceptance Criteria

- The first initialization over an empty-tree Overleaf `master` succeeds only
  through the guarded script.
- A nonempty Overleaf `master` requires pull/resolve rather than force.
- A non-`master` branch cannot use the empty-project force-with-lease path.
- The workflow docs describe this as a one-time initialization exception.

## Verification

```bash
bash -n scripts/sipaim_overleaf_sync.sh
./scripts/sipaim_overleaf_sync.sh check
./scripts/agent_preflight.sh
```

After storing the Overleaf token in the keyring, verify the real path with:

```bash
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```
