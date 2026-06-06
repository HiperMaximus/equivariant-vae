#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh push [kernel_dir] [extra kaggle args...]
  ./scripts/kaggle_kernel.sh status [kernel_id]
  ./scripts/kaggle_kernel.sh output [kernel_id] [output_dir]
  ./scripts/kaggle_kernel.sh pull [kernel_id] [kernel_dir]

Remote writes require KAGGLE_PUSH_CONFIRMED=1.
Remote reads/downloads require KAGGLE_REMOTE_CONFIRMED=1.
Remote pulls require KAGGLE_PULL_CONFIRMED=1.
EOF
}

require_kaggle_cli() {
  if ! command -v kaggle >/dev/null 2>&1; then
    cat >&2 <<'EOF'
missing: kaggle

Install and authenticate the Kaggle CLI only after explicit user permission.
Do not commit Kaggle credentials.
EOF
    exit 1
  fi
}

require_remote_confirmed() {
  if [[ "${KAGGLE_REMOTE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_REMOTE_CONFIRMED=1 after explicit user permission" >&2
    exit 1
  fi
}

metadata_path() {
  local kernel_dir="${1:-$default_kernel_dir}"
  printf '%s/kernel-metadata.json\n' "$kernel_dir"
}

json_field() {
  local metadata="$1"
  local field="$2"
  python3 - "$metadata" "$field" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
field = sys.argv[2]
value = json.loads(metadata.read_text(encoding="utf-8")).get(field, "")
if isinstance(value, str):
    print(value)
PY
}

validate_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local code_file
  metadata="$(metadata_path "$kernel_dir")"

  if [[ ! -d "$kernel_dir" ]]; then
    echo "missing: $kernel_dir" >&2
    exit 1
  fi

  if [[ ! -f "$metadata" ]]; then
    echo "missing: $metadata" >&2
    exit 1
  fi

  python3 -m json.tool "$metadata" >/dev/null
  code_file="$(json_field "$metadata" code_file)"

  if [[ -z "$code_file" ]]; then
    echo "error: metadata code_file is empty" >&2
    exit 1
  fi

  if [[ ! -f "$kernel_dir/$code_file" ]]; then
    echo "missing: $kernel_dir/$code_file" >&2
    exit 1
  fi

  echo "ok: $kernel_dir"
  echo "ok: $metadata"
  echo "ok: $kernel_dir/$code_file"
}

kernel_id_from_metadata() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  json_field "$metadata" id
}

guard_push_ready() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local code_file
  metadata="$(metadata_path "$kernel_dir")"
  code_file="$(json_field "$metadata" code_file)"

  if grep -q "NOT_IMPLEMENTATION_READY" "$kernel_dir/$code_file"; then
    cat >&2 <<'EOF'
error: kernel scaffold is not implementation-ready.

Write docs/behavior_inventory_kaggle.md, lock spec 0001, implement the real
launcher, and remove the NOT_IMPLEMENTATION_READY guard before pushing.
EOF
    exit 1
  fi

  if [[ ! -f "docs/behavior_inventory_kaggle.md" ]]; then
    echo "error: missing docs/behavior_inventory_kaggle.md" >&2
    exit 1
  fi

  if ! grep -Eq '^Implementation readiness: (locked / implementation-ready|implementation-ready|ready)$' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 is not locked as implementation-ready" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "kernel_type": "script",
    "is_private": "true",
    "enable_internet": "false",
}

for key, expected in required_values.items():
    if str(data.get(key, "")).lower() != expected:
        errors.append(f"{key} must be {expected!r}")

dataset_sources = data.get("dataset_sources")
if not isinstance(dataset_sources, list) or not dataset_sources:
    errors.append("dataset_sources must contain at least one confirmed slug")
else:
    for source in dataset_sources:
        if not isinstance(source, str) or "/" not in source or source == "owner/dataset-slug":
            errors.append(f"invalid dataset source slug: {source!r}")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

guard_clean_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  if [[ -n "$(git status --short -- "$kernel_dir")" ]]; then
    cat >&2 <<'EOF'
error: local kernel directory has uncommitted changes.

Commit/stash/reconcile local changes before pulling from Kaggle, or pull into a
separate temporary directory manually.
EOF
    exit 1
  fi
}

action="${1:-}"
case "$action" in
  validate)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    ;;
  check)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    require_kaggle_cli
    kaggle --version
    ;;
  push)
    kernel_dir="${2:-$default_kernel_dir}"
    if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    validate_kernel_dir "$kernel_dir"
    guard_push_ready "$kernel_dir"
    require_kaggle_cli
    shift 2 || true
    kaggle kernels push -p "$kernel_dir" "$@"
    ;;
  status)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    require_remote_confirmed
    require_kaggle_cli
    kaggle kernels status "$kernel_id"
    ;;
  output)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    output_dir="${3:-$default_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle kernels output "$kernel_id" -p "$output_dir"
    ;;
  pull)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    kernel_dir="${3:-$default_kernel_dir}"
    if [[ "${KAGGLE_PULL_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PULL_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    guard_clean_kernel_dir "$kernel_dir"
    require_kaggle_cli
    kaggle kernels pull "$kernel_id" -p "$kernel_dir"
    ;;
  *)
    usage
    exit 1
    ;;
esac
