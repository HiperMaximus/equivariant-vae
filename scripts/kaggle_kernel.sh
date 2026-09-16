#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

default_kernel_dir="kaggle/kernels/functional_geometry_stage_a2_calibration"
default_output_dir="runs/kaggle/functional_geometry_stage_a2_calibration"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh identity
  ./scripts/kaggle_kernel.sh push [kernel_dir]
  ./scripts/kaggle_kernel.sh status [owner/slug[/version]]
  ./scripts/kaggle_kernel.sh logs [owner/slug[/version]]
  ./scripts/kaggle_kernel.sh output owner/slug/version [new_output_dir]
  ./scripts/kaggle_kernel.sh dataset-download owner/slug/version new_output_dir
EOF
}

require_kaggle_cli() {
  command -v kaggle >/dev/null 2>&1 || {
    echo "error: Kaggle CLI is not installed" >&2
    exit 1
  }
}

kaggle_tool_python() {
  local kaggle_bin shebang interpreter command_name executable_name env_name
  kaggle_bin="$(command -v kaggle)"
  IFS= read -r shebang <"$kaggle_bin" || return 1
  [[ "$shebang" == '#!'* ]] || return 1
  interpreter="${shebang#\#!}"
  command_name="${interpreter%% *}"
  executable_name="$(basename "$command_name")"
  if [[ -x "$command_name" && "$executable_name" == python* ]]; then
    printf '%s\n' "$command_name"
    return
  fi
  if [[ "$interpreter" == /usr/bin/env\ * ]]; then
    env_name="${interpreter#/usr/bin/env }"
    [[ "$env_name" == -S\ * ]] && env_name="${env_name#-S }"
    env_name="${env_name%% *}"
    if [[ "$(basename "$env_name")" == python* ]] \
      && command -v "$env_name" >/dev/null 2>&1; then
      command -v "$env_name"
      return
    fi
  fi
  return 1
}

kaggle_api() {
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    local kaggle_python
    kaggle_python="$(kaggle_tool_python)" || {
      echo "error: cannot resolve the Kaggle CLI Python interpreter" >&2
      exit 1
    }
    "$kaggle_python" scripts/kaggle_oauth_exec.py "$@"
  else
    kaggle "$@"
  fi
}

kaggle_username() {
  require_kaggle_cli
  local kaggle_python
  kaggle_python="$(kaggle_tool_python)" || {
    echo "error: cannot resolve the Kaggle CLI Python interpreter" >&2
    exit 1
  }
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    "$kaggle_python" scripts/kaggle_oauth_exec.py --print-oauth-username
  else
    "$kaggle_python" scripts/kaggle_oauth_exec.py --print-legacy-username
  fi
}

metadata_value() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get(sys.argv[2])
if not isinstance(value, str) or not value:
    raise SystemExit(f"metadata {sys.argv[2]} must be a nonempty string")
print(value)
PY
}

validate_kernel() {
  local kernel_dir="${1:-$default_kernel_dir}" metadata code_file
  metadata="$kernel_dir/kernel-metadata.json"
  [[ -f "$metadata" ]] || { echo "missing: $metadata" >&2; exit 1; }
  code_file="$(metadata_value "$metadata" code_file)"
  [[ "$(basename "$code_file")" == "$code_file" ]] || {
    echo "error: code_file must be one local filename" >&2
    exit 1
  }
  [[ -f "$kernel_dir/$code_file" ]] || {
    echo "missing: $kernel_dir/$code_file" >&2
    exit 1
  }
  python3 -m py_compile "$kernel_dir/$code_file"
  python3 -m json.tool "$metadata" >/dev/null
  echo "ok: $kernel_dir"
}

require_versioned_reference() {
  [[ "$1" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[1-9][0-9]*$ ]] || {
    echo "error: expected owner/slug/version" >&2
    exit 1
  }
}

action="${1:-}"
case "$action" in
  build|validate) validate_kernel "${2:-$default_kernel_dir}" ;;
  check)
    validate_kernel "${2:-$default_kernel_dir}"
    require_kaggle_cli
    kaggle --version
    ;;
  identity) kaggle_username ;;
  push)
    kernel_dir="${2:-$default_kernel_dir}"
    validate_kernel "$kernel_dir"
    require_kaggle_cli
    metadata="$kernel_dir/kernel-metadata.json"
    kernel_id="$(metadata_value "$metadata" id)"
    actor="$(kaggle_username)"
    [[ "${kernel_id%%/*}" == "$actor" ]] || {
      echo "error: metadata owner ${kernel_id%%/*} != authenticated user $actor" >&2
      exit 1
    }
    upload_dir="$(mktemp -d)"
    trap 'rm -rf "$upload_dir"' EXIT
    code_file="$(metadata_value "$metadata" code_file)"
    cp -- "$metadata" "$upload_dir/kernel-metadata.json"
    cp -- "$kernel_dir/$code_file" "$upload_dir/$code_file"
    kaggle_api kernels push -p "$upload_dir"
    ;;
  status)
    require_kaggle_cli
    reference="${2:-$(metadata_value "$default_kernel_dir/kernel-metadata.json" id)}"
    kaggle_api kernels status "$reference"
    ;;
  logs)
    require_kaggle_cli
    reference="${2:-$(metadata_value "$default_kernel_dir/kernel-metadata.json" id)}"
    kaggle_api kernels logs "$reference"
    ;;
  output)
    require_kaggle_cli
    reference="${2:?missing owner/slug/version}"
    require_versioned_reference "$reference"
    output_dir="${3:-$default_output_dir}"
    [[ ! -e "$output_dir" ]] || { echo "error: output exists: $output_dir" >&2; exit 1; }
    mkdir -p "$output_dir"
    kaggle_api kernels output "$reference" -p "$output_dir"
    ;;
  dataset-download)
    require_kaggle_cli
    reference="${2:?missing owner/slug/version}"
    require_versioned_reference "$reference"
    output_dir="${3:?missing output directory}"
    [[ ! -e "$output_dir" ]] || { echo "error: output exists: $output_dir" >&2; exit 1; }
    mkdir -p "$output_dir"
    kaggle_api datasets download "$reference" -p "$output_dir" --unzip
    ;;
  *) usage; exit 1 ;;
esac
