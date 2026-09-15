#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="$PWD/runs/local_tmp/kaggle_kernel_$$"
  trap 'rm -rf "$TMPDIR"' EXIT
fi
mkdir -p "$TMPDIR"

build_python="${PYTHON:-.venv/bin/python}"
default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh api-check [kernel_dir]
  ./scripts/kaggle_kernel.sh identity
  ./scripts/kaggle_kernel.sh push [kernel_dir] [--wait]
      [--wait-interval seconds] [--wait-max polls] [--wait-queued seconds]
  ./scripts/kaggle_kernel.sh status [owner/slug[/version]]
  ./scripts/kaggle_kernel.sh status-launch launch-receipt.json
  ./scripts/kaggle_kernel.sh logs [owner/slug[/version]]
  ./scripts/kaggle_kernel.sh logs-launch launch-receipt.json
  ./scripts/kaggle_kernel.sh wait owner/slug[/version]
      [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh output owner/slug/version [new_output_dir]
  ./scripts/kaggle_kernel.sh output-launch launch-receipt.json new_output_dir
  ./scripts/kaggle_kernel.sh dataset-download owner/dataset/version new_output_dir
  ./scripts/kaggle_kernel.sh pull owner/slug[/version] [clean_kernel_dir]
  ./scripts/kaggle_kernel.sh pull-launch launch-receipt.json new_kernel_dir

EOF
}

require_build_python() {
  if ! "$build_python" -c 'import eqvae.benchmarking' >/dev/null 2>&1; then
    echo "error: kernel tooling needs eqvae and torch in $build_python" >&2
    echo "hint: uv sync --locked --python 3.12 --group dev" >&2
    exit 1
  fi
}

require_kaggle_cli() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: Kaggle CLI is not installed" >&2
    exit 1
  fi
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
    return
  fi
  kaggle "$@"
}

kaggle_authenticated_username() {
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

metadata_path() {
  printf '%s/kernel-metadata.json\n' "${1:-$default_kernel_dir}"
}

kernel_id_from_metadata() {
  python3 - "$(metadata_path "$1")" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("id")
if not isinstance(value, str) or value.count("/") != 1:
    raise SystemExit("kernel metadata id must be owner/slug")
print(value)
PY
}

validate_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata code_file
  metadata="$(metadata_path "$kernel_dir")"
  [[ -d "$kernel_dir" ]] || { echo "missing: $kernel_dir" >&2; exit 1; }
  [[ -f "$metadata" ]] || { echo "missing: $metadata" >&2; exit 1; }
  code_file="$(python3 - "$metadata" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
kernel_id = data.get("id")
code_file = data.get("code_file")
if not isinstance(kernel_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", kernel_id) is None:
    raise SystemExit("metadata id must be owner/slug")
if not isinstance(code_file, str) or not code_file or Path(code_file).name != code_file:
    raise SystemExit("metadata code_file must be one local filename")
for field in ("dataset_sources", "kernel_sources", "model_sources", "competition_sources"):
    values = data.get(field, [])
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise SystemExit(f"metadata {field} must be a list of nonempty strings")
print(code_file)
PY
)"
  [[ -f "$kernel_dir/$code_file" ]] || {
    echo "missing: $kernel_dir/$code_file" >&2
    exit 1
  }
  python3 -m py_compile "$kernel_dir/$code_file"
  echo "ok: metadata $metadata"
  echo "ok: Python $kernel_dir/$code_file"
}

build_kernel() {
  local kernel_dir="${1:-$default_kernel_dir}"
  if [[ -f "$kernel_dir/run_template.py" ]]; then
    require_build_python
    "$build_python" scripts/build_kaggle_embedded_kernel.py \
      --kernel-dir "$kernel_dir" --allow-dirty
  fi
  validate_kernel_dir "$kernel_dir"
}

verify_embedded_kernel() {
  local kernel_dir="${1:-$default_kernel_dir}"
  validate_kernel_dir "$kernel_dir"
  if [[ -f "$kernel_dir/run_template.py" ]]; then
    require_build_python
    "$build_python" scripts/build_kaggle_embedded_kernel.py \
      --kernel-dir "$kernel_dir" --verify-only --allow-dirty
  fi
}

portable_snapshot() {
  local kernel_dir="$1" actor="$2" stage_root source_dir upload_dir code_file
  stage_root="$(mktemp -d "$TMPDIR/kaggle_upload.XXXXXX")"
  source_dir="$stage_root/source"
  upload_dir="$stage_root/kernel"
  mkdir -p "$source_dir"
  code_file="$(python3 - "$(metadata_path "$kernel_dir")" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["code_file"])
PY
)"
  cp -- "$kernel_dir/kernel-metadata.json" "$source_dir/kernel-metadata.json"
  cp -- "$kernel_dir/$code_file" "$source_dir/$code_file"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources snapshot \
    --source-dir "$source_dir" --destination-dir "$upload_dir" --actor "$actor" \
    >/dev/null
  printf '%s\n' "$upload_dir"
}

confirmed_kernel_reference() {
  require_build_python
  printf '%s\n' "$1" | "$build_python" -m eqvae.kaggle_resources confirmation
}

record_launch() {
  local source_dir="$1" upload_dir="$2" reference="$3"
  local receipt_root="${EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT:-runs/local/kaggle_launches}"
  require_build_python
  "$build_python" -m eqvae.kaggle_resources receipt \
    --source-dir "$source_dir" --upload-dir "$upload_dir" \
    --receipt-root "$receipt_root" --accepted-reference "$reference"
}

reference_from_receipt() {
  require_build_python
  "$build_python" - "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reference = payload.get("kernel_reference")
version = payload.get("accepted_version")
kernel_id = payload.get("kernel_id")
if payload.get("schema_version") != "eqvae.kaggle_kernel_launch.v1":
    raise SystemExit("invalid Kaggle launch receipt schema")
if not isinstance(version, int) or isinstance(version, bool) or version < 1:
    raise SystemExit("invalid Kaggle launch receipt version")
if not isinstance(kernel_id, str) or reference != f"{kernel_id}/{version}":
    raise SystemExit("invalid Kaggle launch receipt reference")
print(reference)
PY
}

versioned_reference() {
  require_build_python
  "$build_python" -m eqvae.kaggle_resources validate-versioned-reference \
    --reference "$1"
}

record_download() {
  require_build_python
  "$build_python" -m eqvae.kaggle_resources download-receipt \
    --resource-kind "$1" --resource-reference "$2" --download-dir "$3" \
    --receipt-name "$4"
}

wait_kernel_until_settled() {
  local kernel_id="$1" poll_interval="${2:-300}" max_polls="${3:-180}"
  local max_queued_seconds="${4:-300}" running_polls=0 queued_elapsed=0
  local status status_line queued_interval=30
  ((poll_interval >= 10)) || poll_interval=10
  ((queued_interval <= poll_interval)) || queued_interval="$poll_interval"
  while :; do
    status_line="$(kaggle_api kernels status "$kernel_id" 2>&1)" || true
    status="$(printf '%s\n' "$status_line" | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1 || true)"
    status="${status#KernelWorkerStatus.}"
    case "$status" in
      QUEUED)
        echo "wait: QUEUED ${queued_elapsed}s/${max_queued_seconds}s"
        ((queued_elapsed >= max_queued_seconds)) && {
          echo "WAIT_SETTLED_STATUS=QUEUED_TIMEOUT"
          return 3
        }
        sleep "$queued_interval"
        queued_elapsed=$((queued_elapsed + queued_interval))
        ;;
      RUNNING|'')
        running_polls=$((running_polls + 1))
        [[ -n "$status" ]] \
          && echo "wait: RUNNING poll ${running_polls}/${max_polls}" \
          || printf 'wait: unparseable status (%s/%s): %s\n' \
            "$running_polls" "$max_polls" "$status_line" >&2
        ((running_polls < max_polls)) || {
          echo "WAIT_SETTLED_STATUS=TIMEOUT_STILL_PENDING"
          return 2
        }
        sleep "$poll_interval"
        ;;
      *)
        echo "WAIT_SETTLED_STATUS=$status"
        return
        ;;
    esac
  done
}

api_check() {
  local kernel_dir="${1:-$default_kernel_dir}" actor original_id kernel_id source_rows
  verify_embedded_kernel "$kernel_dir"
  require_kaggle_cli
  actor="$(kaggle_authenticated_username)"
  original_id="$(kernel_id_from_metadata "$kernel_dir")"
  kernel_id="${actor}/${original_id#*/}"
  echo "ok: Kaggle CLI $(kaggle --version 2>&1)"
  echo "ok: authenticated actor $actor"
  echo "check: kernel $kernel_id"
  kaggle_api kernels list --mine --search "${kernel_id#*/}" --csv >/dev/null
  source_rows="$(python3 - "$(metadata_path "$kernel_dir")" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
    for value in data.get(field, []):
        print(f"{field}\t{value}")
PY
)"
  while [[ -n "$source_rows" ]] && IFS=$'\t' read -r kind reference; do
    case "$kind" in
      dataset_sources) kaggle_api datasets files "$reference" -v >/dev/null ;;
      competition_sources) kaggle_api competitions files "$reference" -v >/dev/null ;;
      kernel_sources) kaggle_api kernels files "$reference" -v >/dev/null ;;
      model_sources) kaggle_api models instances versions files "$reference" -v >/dev/null ;;
    esac
    echo "ok: $kind $reference"
  done <<<"$source_rows"
}

guard_clean_kernel_dir() {
  [[ -z "$(git status --short -- "$1")" ]] || {
    echo "error: local kernel directory has uncommitted changes: $1" >&2
    exit 1
  }
}

action="${1:-}"
case "$action" in
  build) build_kernel "${2:-$default_kernel_dir}" ;;
  validate) validate_kernel_dir "${2:-$default_kernel_dir}" ;;
  check)
    verify_embedded_kernel "${2:-$default_kernel_dir}"
    require_kaggle_cli
    kaggle --version
    ;;
  api-check) api_check "${2:-$default_kernel_dir}" ;;
  identity) kaggle_authenticated_username ;;
  push)
    kernel_dir="${2:-$default_kernel_dir}"
    shift $(( $# >= 2 ? 2 : 1 ))
    push_wait=0
    push_wait_interval=300
    push_wait_max=180
    push_wait_queued=300
    while (($#)); do
      case "$1" in
        --wait) push_wait=1 ;;
        --wait-interval) push_wait_interval="${2:?missing seconds}"; shift ;;
        --wait-max) push_wait_max="${2:?missing polls}"; shift ;;
        --wait-queued) push_wait_queued="${2:?missing seconds}"; shift ;;
        *) echo "error: unknown push option: $1" >&2; exit 1 ;;
      esac
      shift
    done
    verify_embedded_kernel "$kernel_dir"
    require_kaggle_cli
    actor="$(kaggle_authenticated_username)"
    upload_dir="$(portable_snapshot "$kernel_dir" "$actor")"
    echo "push: actor=$actor kernel=$(kernel_id_from_metadata "$upload_dir")"
    if ! push_response="$(kaggle_api kernels push -p "$upload_dir" 2>&1)"; then
      printf '%s\n' "$push_response" >&2
      exit 1
    fi
    printf '%s\n' "$push_response"
    reference="$(confirmed_kernel_reference "$push_response")"
    receipt="$(record_launch "$kernel_dir" "$upload_dir" "$reference")"
    echo "ok: launch receipt $receipt"
    ((push_wait == 0)) || wait_kernel_until_settled \
      "$reference" "$push_wait_interval" "$push_wait_max" "$push_wait_queued"
    ;;
  status)
    require_kaggle_cli
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    kaggle_api kernels status "$kernel_id"
    ;;
  status-launch)
    require_kaggle_cli
    kaggle_api kernels status "$(reference_from_receipt "${2:?missing receipt}")"
    ;;
  logs)
    require_kaggle_cli
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    kaggle_api kernels logs "$kernel_id"
    ;;
  logs-launch)
    require_kaggle_cli
    kaggle_api kernels logs "$(reference_from_receipt "${2:?missing receipt}")"
    ;;
  wait)
    require_kaggle_cli
    wait_kernel_until_settled "${2:?missing kernel reference}" "${3:-300}" "${4:-180}" "${5:-300}"
    ;;
  output)
    require_kaggle_cli
    reference="$(versioned_reference "${2:?missing owner/slug/version}")"
    output_dir="${3:-$default_output_dir}"
    [[ ! -e "$output_dir" ]] || { echo "error: output directory exists: $output_dir" >&2; exit 1; }
    mkdir -p "$output_dir"
    kaggle_api kernels output "$reference" -p "$output_dir"
    record_download kernel "$reference" "$output_dir" kaggle_output_receipt.json
    ;;
  output-launch)
    require_kaggle_cli
    reference="$(reference_from_receipt "${2:?missing receipt}")"
    output_dir="${3:?missing output directory}"
    [[ ! -e "$output_dir" ]] || { echo "error: output directory exists: $output_dir" >&2; exit 1; }
    mkdir -p "$output_dir"
    kaggle_api kernels output "$reference" -p "$output_dir"
    record_download kernel "$reference" "$output_dir" kaggle_output_receipt.json
    ;;
  dataset-download)
    require_kaggle_cli
    reference="$(versioned_reference "${2:?missing owner/dataset/version}")"
    output_dir="${3:?missing output directory}"
    [[ ! -e "$output_dir" ]] || { echo "error: output directory exists: $output_dir" >&2; exit 1; }
    mkdir -p "$output_dir"
    kaggle_api datasets download "$reference" -p "$output_dir" --unzip
    record_download dataset "$reference" "$output_dir" kaggle_dataset_receipt.json
    ;;
  pull)
    require_kaggle_cli
    kernel_id="${2:?missing kernel reference}"
    kernel_dir="${3:-$default_kernel_dir}"
    guard_clean_kernel_dir "$kernel_dir"
    kaggle_api kernels pull "$kernel_id" -p "$kernel_dir"
    ;;
  pull-launch)
    require_kaggle_cli
    reference="$(reference_from_receipt "${2:?missing receipt}")"
    kernel_dir="${3:?missing kernel directory}"
    [[ ! -e "$kernel_dir" ]] || { echo "error: kernel directory exists: $kernel_dir" >&2; exit 1; }
    kaggle_api kernels pull "$reference" -p "$kernel_dir"
    record_download kernel "$reference" "$kernel_dir" kaggle_kernel_pull_receipt.json
    ;;
  *) usage; exit 1 ;;
esac
