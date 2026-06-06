#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

repo_root="$(git rev-parse --show-toplevel)"
expected_name="equivariant-vae"
expected_overleaf_url="https://git.overleaf.com/69c614433cbc9e46cf226d24"

echo "Repo preflight"
echo "=============="
echo "Repo: $repo_root"
echo

if [[ "$(basename "$repo_root")" != "$expected_name" ]]; then
  echo "error: expected to run inside $expected_name"
  exit 1
fi

cd "$repo_root"

required_files=(
  "AGENTS.md"
  "CLAUDE.md"
  "CURRENT.md"
  "GOAL.md"
  "README.md"
  "docs/repo_goal_and_requirements.md"
  "docs/issue_image_inventory.md"
  "docs/equivariant_vae_transition_plan.md"
  "docs/overleaf_sync_workflow.md"
  "docs/agentic_review_workflow.md"
  "docs/decisions/README.md"
  "scripts/sipaim_overleaf_sync.sh"
  "paper/sipaim2026/main.tex"
  "paper/sipaim2026/sipaim2026.pdf"
)

missing=0
tracked_problem=0

sanitize_url() {
  local url="$1"
  printf '%s\n' "$url" | sed -E 's#^([[:alpha:]][[:alnum:].+-]*://)[^/@]+@#\1***@#'
}

url_has_credentials() {
  local url="$1"
  [[ "$url" =~ ^[[:alpha:]][[:alnum:].+-]*://[^/@]+@ ]]
}

echo "Required files"
for path in "${required_files[@]}"; do
  if [[ -f "$path" ]]; then
    echo "ok: $path"
  else
    echo "missing: $path"
    missing=1
  fi
done

echo
echo "Required files tracked by Git"
for path in "${required_files[@]}"; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "tracked: $path"
  else
    echo "untracked: $path"
    tracked_problem=1
  fi
done

echo
echo "Git status"
git status --short

echo
echo "Overleaf remote"
if git remote get-url origin >/dev/null 2>&1; then
  origin_url="$(git remote get-url origin)"
  if url_has_credentials "$origin_url"; then
    echo "error: origin URL contains credentials"
    missing=1
  fi
  if [[ "$origin_url" == *"git.overleaf.com"* ]]; then
    echo "error: origin points at Overleaf"
    missing=1
  fi
fi

while read -r remote; do
  remote_url="$(git remote get-url "$remote")"
  if url_has_credentials "$remote_url"; then
    echo "error: remote '$remote' URL contains credentials"
    missing=1
  fi
  if [[ "$remote" != "overleaf" && "$remote_url" == *"git.overleaf.com"* ]]; then
    echo "error: remote '$remote' points at Overleaf"
    missing=1
  fi
done < <(git remote)

if git remote get-url overleaf >/dev/null 2>&1; then
  overleaf_url="$(git remote get-url overleaf)"
  echo "$(sanitize_url "$overleaf_url")"
  if [[ "$overleaf_url" != "$expected_overleaf_url" ]]; then
    echo "error: overleaf remote URL is not the expected project"
    missing=1
  fi
else
  echo "missing: overleaf remote"
  missing=1
fi

echo
echo "Stale planning-term check"
stale_pattern='\b(MAPI4?|Springer|D4)\b'
if ! command -v rg >/dev/null 2>&1; then
  echo "missing: rg; cannot run stale planning-term check"
  missing=1
elif rg -n --ignore-case "$stale_pattern" AGENTS.md CLAUDE.md CURRENT.md GOAL.md README.md docs >/tmp/equivariant_vae_preflight_stale_terms.txt; then
  cat /tmp/equivariant_vae_preflight_stale_terms.txt
  echo "warning: stale planning terms found in human-authored docs"
  missing=1
else
  echo "ok: no stale planning terms found"
fi

echo
echo "Read before substantial work"
printf '%s\n' \
  "1. AGENTS.md" \
  "2. CURRENT.md" \
  "3. GOAL.md" \
  "4. docs/repo_goal_and_requirements.md" \
  "5. docs/issue_image_inventory.md" \
  "6. docs/equivariant_vae_transition_plan.md" \
  "7. docs/overleaf_sync_workflow.md" \
  "8. docs/agentic_review_workflow.md" \
  "9. docs/decisions/README.md"

if [[ "$missing" -ne 0 || "$tracked_problem" -ne 0 ]]; then
  exit 1
fi
