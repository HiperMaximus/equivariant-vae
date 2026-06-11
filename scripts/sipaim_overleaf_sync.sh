#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="overleaf"
REMOTE_URL="https://git.overleaf.com/69c614433cbc9e46cf226d24"
REMOTE_BRANCH="${OVERLEAF_BRANCH:-master}"
PREFIX="paper/sipaim2026"
TRACKED_PDF="${PREFIX}/sipaim2026.pdf"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/sipaim_overleaf_sync.sh <command>

Commands:
  check       Show repo, remote, subtree, and paper status.
  setup       Add or repair the local Overleaf remote.
  ls-remote   Check the Overleaf remote refs.
  compile     Build main.tex and refresh paper/sipaim2026/sipaim2026.pdf.
  pull        Pull Overleaf edits into paper/sipaim2026 with git subtree.
  push        Push committed paper/sipaim2026 changes to Overleaf with git subtree.

Rules:
  - Never run plain `git push overleaf` from this repo.
  - Commit local paper changes before `push`.
  - Run `compile` before committing paper changes.
  - Pull Overleaf edits before starting local paper edits when advisor edits are possible.
  - Commands that access or change Overleaf remote state require:
      OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh ls-remote
      OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull
      OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
EOF
}

repo_root() {
  git rev-parse --show-toplevel
}

enter_repo() {
  local root
  root="$(repo_root)"
  cd "$root"

  if [[ "$(basename "$root")" != "equivariant-vae" ]]; then
    echo "Refusing to run outside the equivariant-vae repo: $root" >&2
    exit 1
  fi

  if [[ ! -d "$PREFIX" ]]; then
    echo "Missing expected paper subtree: $PREFIX" >&2
    exit 1
  fi
}

sanitize_url() {
  local url="$1"
  printf '%s\n' "$url" | sed -E 's#^([[:alpha:]][[:alnum:].+-]*://)[^/@]+@#\1***@#'
}

url_has_credentials() {
  local url="$1"
  [[ "$url" =~ ^[[:alpha:]][[:alnum:].+-]*://[^/@]+@ ]]
}

validate_remote_url() {
  local remote="$1"
  local url="$2"

  if url_has_credentials "$url"; then
    echo "Refusing to use remote '$remote' because its URL contains credentials." >&2
    echo "Use a Git credential helper instead of embedding tokens in remote URLs." >&2
    exit 1
  fi
}

validate_remote_safety() {
  local remote
  local url

  while read -r remote; do
    url="$(git remote get-url "$remote")"
    validate_remote_url "$remote" "$url"

    if [[ "$remote" == "origin" && "$url" == *"git.overleaf.com"* ]]; then
      echo "Refusing to continue: origin points at Overleaf." >&2
      exit 1
    fi

    if [[ "$remote" != "$REMOTE_NAME" && "$url" == *"git.overleaf.com"* ]]; then
      echo "Refusing to continue: remote '$remote' points at Overleaf." >&2
      echo "Only the '$REMOTE_NAME' remote may use the Overleaf URL." >&2
      exit 1
    fi
  done < <(git remote)
}

print_sanitized_remotes() {
  local remote
  local url

  while read -r remote; do
    url="$(git remote get-url "$remote")"
    echo "${remote} $(sanitize_url "$url")"
  done < <(git remote)
}

require_overleaf_remote_exact() {
  local current_url

  if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    echo "Missing Overleaf remote '$REMOTE_NAME'. Run setup first." >&2
    exit 1
  fi

  current_url="$(git remote get-url "$REMOTE_NAME")"
  validate_remote_url "$REMOTE_NAME" "$current_url"

  if [[ "$current_url" != "$REMOTE_URL" ]]; then
    echo "Overleaf remote has an unexpected URL." >&2
    echo "  expected: $(sanitize_url "$REMOTE_URL")" >&2
    echo "  actual:   $(sanitize_url "$current_url")" >&2
    echo "Run setup to repair it." >&2
    exit 1
  fi
}

ensure_overleaf_remote() {
  validate_remote_safety

  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    local current_url
    current_url="$(git remote get-url "$REMOTE_NAME")"
    if [[ "$current_url" != "$REMOTE_URL" ]]; then
      echo "Updating $REMOTE_NAME remote URL:"
      echo "  old: $(sanitize_url "$current_url")"
      echo "  new: $(sanitize_url "$REMOTE_URL")"
      git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
    fi
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
  fi

  validate_remote_safety
  require_overleaf_remote_exact
}

require_clean_worktree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit or stash changes before this command." >&2
    git status --short >&2
    exit 1
  fi
}

require_clean_paper_subtree() {
  if [[ -n "$(git status --porcelain -- "$PREFIX")" ]]; then
    echo "Paper subtree has uncommitted changes. Commit them before pushing." >&2
    git status --short -- "$PREFIX" >&2
    exit 1
  fi
}

require_remote_confirmation() {
  local command="$1"

  if [[ "${OVERLEAF_SYNC_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<EOF
Refusing to run Overleaf '$command' without explicit confirmation.

Agents must check status and ask the user before Overleaf pull/push operations.
After permission, rerun as:

  OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh $command
EOF
    exit 1
  fi
}

cmd_check() {
  validate_remote_safety
  echo "Repo root: $(repo_root)"
  echo
  echo "Remotes:"
  print_sanitized_remotes
  echo
  echo "Paper subtree status:"
  git status --short -- "$PREFIX"
  echo
  echo "Full repo status:"
  git status --short
  echo
  echo "Expected Overleaf remote:"
  echo "  ${REMOTE_NAME} $(sanitize_url "$REMOTE_URL")"
  echo "Expected subtree:"
  echo "  ${PREFIX}"

  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    require_overleaf_remote_exact
  else
    echo "Overleaf remote is not configured. Run setup when sync is needed." >&2
  fi
}

cmd_setup() {
  ensure_overleaf_remote
  echo "Configured Overleaf remote:"
  print_sanitized_remotes | grep "^${REMOTE_NAME}[[:space:]]" || true
}

cmd_ls_remote() {
  require_remote_confirmation "ls-remote"
  validate_remote_safety
  require_overleaf_remote_exact
  git ls-remote "$REMOTE_NAME"
}

cmd_compile() {
  if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk is not installed or not on PATH." >&2
    exit 1
  fi

  (
    cd "$PREFIX"
    latexmk -pdf main.tex
  )

  cp "${PREFIX}/main.pdf" "$TRACKED_PDF"
  echo "Updated $TRACKED_PDF"
}

cmd_pull() {
  require_remote_confirmation "pull"
  ensure_overleaf_remote
  require_clean_worktree
  git status --short
  git subtree pull --prefix "$PREFIX" "$REMOTE_NAME" "$REMOTE_BRANCH" --squash
}

cmd_push() {
  require_remote_confirmation "push"
  ensure_overleaf_remote
  cmd_compile
  require_clean_paper_subtree
  git status --short
  git subtree push --prefix "$PREFIX" "$REMOTE_NAME" "$REMOTE_BRANCH"
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 2
  fi

  enter_repo

  case "$1" in
    check) cmd_check ;;
    setup) cmd_setup ;;
    ls-remote) cmd_ls_remote ;;
    compile) cmd_compile ;;
    pull) cmd_pull ;;
    push) cmd_push ;;
    help|-h|--help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
