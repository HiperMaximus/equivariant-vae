# Overleaf Sync Workflow

This repo keeps the SIPAIM paper in `paper/sipaim2026` so the current text and
PDF can be visible in GitHub. The same folder is synced to Overleaf for advisor
review.

The active manuscript should not be moved to a separate local folder. The source
of record is `paper/sipaim2026`, and the professor-facing Overleaf project is a
synced copy of that subtree.

## Repositories

```text
equivariant-vae GitHub repo
  origin: https://github.com/HiperMaximus/equivariant-vae.git
  paper:  paper/sipaim2026

Overleaf project
  web:    https://www.overleaf.com/project/69c614433cbc9e46cf226d24
  remote: https://git.overleaf.com/69c614433cbc9e46cf226d24
  branch: master
```

Only `paper/sipaim2026` should go to Overleaf. The rest of the repo contains
code, experiment plans, notebooks, and implementation details that do not belong
in the Overleaf project.

## One-Time Setup

From the repo root:

```bash
./scripts/sipaim_overleaf_sync.sh setup
```

This configures a local Git remote named `overleaf`.

The script refuses embedded credentials in remote URLs. Use a Git credential
helper for Overleaf tokens.

Overleaf Git authentication is token-only. When Git prompts:

- username: `git`
- password: your Overleaf Git authentication token, not your normal Overleaf
  password

If a wrong username/password was cached, clear the cached `git.overleaf.com`
credential before retrying:

```bash
printf 'protocol=https\nhost=git.overleaf.com\n\n' | git credential reject
```

Do not paste or store Overleaf tokens in repo files, shell history, logs, or
chat.

## Persistent Local Credential Storage

For this workstation, prefer Git's `libsecret` credential helper so the
Overleaf token lives in the desktop keyring instead of a plaintext file.

One-time setup:

```bash
sudo apt-get install -y libsecret-1-dev libsecret-tools pkg-config
mkdir -p /home/maximus/Documents/Tesis/.agent-tools
gcc -o /home/maximus/Documents/Tesis/.agent-tools/git-credential-libsecret \
  /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret.c \
  $(pkg-config --cflags --libs libsecret-1)
cd /home/maximus/Documents/Tesis/equivariant-vae
git config --local --add credential.https://git.overleaf.com.helper ""
git config --local --add credential.https://git.overleaf.com.helper \
  /home/maximus/Documents/Tesis/.agent-tools/git-credential-libsecret
git config --local credential.https://git.overleaf.com.username git
```

Then store the token without echoing it:

```bash
read -rsp "Overleaf Git token: " OVERLEAF_TOKEN
printf '\n'
printf 'protocol=https\nhost=git.overleaf.com\nusername=git\npassword=%s\n\n' "$OVERLEAF_TOKEN" | git credential approve
unset OVERLEAF_TOKEN
```

Overleaf remote reads and pull/push operations require explicit user permission.
After checking status and receiving permission, run them with:

```bash
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh ls-remote
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

## Normal Local-to-Overleaf Flow

Use this when local paper edits should appear in Overleaf.

```bash
./scripts/sipaim_overleaf_sync.sh check
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull

# Edit files under paper/sipaim2026.
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Update SIPAIM paper"

OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

Why this order:

- `pull` first reduces conflicts with advisor edits from Overleaf.
- `compile` creates the advisor-facing `sipaim2026.pdf`.
- `commit` is required because subtree push exports committed history, not
  unsaved working-tree changes.
- `push` exports only `paper/sipaim2026` to Overleaf.

First sync edge case: a newly created Overleaf project may already have a
`master` commit whose tree is empty, even though the web editor shows no files.
The safe script handles that one case by first trying a normal subtree push and
then, only if the observed Overleaf branch is an empty tree, initializing it with
a normal fast-forward commit on top of the exact observed empty commit. This
exception is `master`-only, validates that the pushed split contains paper files
rather than repo-root paths, and must not overwrite nonempty Overleaf content.

## Normal Overleaf-to-Local Flow

Use this after the professor edits in Overleaf.

```bash
./scripts/sipaim_overleaf_sync.sh check
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Pull Overleaf paper edits"
```

If the pull creates merge conflicts, resolve only files under
`paper/sipaim2026`, compile again, then commit.

## Important Warnings

- Never run `git push overleaf` from this repo.
- Never set `overleaf` as `origin`.
- Never embed Overleaf tokens in Git remote URLs.
- Never use the normal Overleaf account password for Git; use username `git`
  and an Overleaf Git authentication token.
- Never force-push Overleaf manually. The scripted first-sync initialization of
  an observed empty-tree `master` branch uses a normal fast-forward commit, not a
  force push.
- Never run Overleaf remote reads, pull, or push as an agent without asking the
  user first.
- Never push notebooks, training code, or experiment folders to Overleaf.
- Avoid concurrent editing: if the professor is actively editing in Overleaf,
  pull before making local edits and push only after checking with them.
- Overleaf uses `master`; do not expect normal branch workflows there.
- Overleaf comments/track changes can be awkward with Git pushes. Prefer normal
  text edits in Overleaf when using this workflow.

## Generated Files

Track:

```text
paper/sipaim2026/sipaim2026.pdf
```

This PDF is intentionally tracked so the latest compiled paper is visible in the
GitHub repo and in Overleaf. Refresh it with:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

Ignore:

```text
paper/sipaim2026/main.pdf
paper/sipaim2026/*.aux
paper/sipaim2026/*.log
paper/sipaim2026/*.bbl
paper/sipaim2026/*.fls
paper/sipaim2026/*.fdb_latexmk
```

The `sipaim2026.pdf` file is tracked because it is useful for GitHub/advisor
visibility. Build artifacts remain ignored.
