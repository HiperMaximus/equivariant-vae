#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh api-check
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

build_kernel_payload() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local payload_dir="$kernel_dir/payload"

  validate_kernel_dir "$kernel_dir"

  if [[ ! -d "src/eqvae" ]]; then
    echo "error: missing src/eqvae; implement spec 0001 before building Kaggle payload" >&2
    exit 1
  fi

  if [[ ! -d "configs/spec0001" ]]; then
    echo "error: missing configs/spec0001; implement spec 0001 before building Kaggle payload" >&2
    exit 1
  fi

  python3 - "$payload_dir" <<'PY'
import shutil
import sys
from pathlib import Path

payload = Path(sys.argv[1])
if payload.exists():
    shutil.rmtree(payload)
(payload / "src").mkdir(parents=True)
(payload / "configs").mkdir(parents=True)
shutil.copytree("src/eqvae", payload / "src" / "eqvae")
shutil.copytree("configs/spec0001", payload / "configs" / "spec0001")
shutil.copy2("pyproject.toml", payload / "pyproject.toml")
shutil.copy2("uv.lock", payload / "uv.lock")
PY

  echo "ok: built $payload_dir"
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

Use docs/behavior_inventory_kaggle.md and spec 0001, implement the real launcher,
and remove the NOT_IMPLEMENTATION_READY guard before pushing.
EOF
    exit 1
  fi

  if [[ ! -f "docs/behavior_inventory_kaggle.md" ]]; then
    echo "error: missing docs/behavior_inventory_kaggle.md" >&2
    exit 1
  fi

  if [[ ! -d "$kernel_dir/payload/src/eqvae" ]]; then
    echo "error: missing bundled payload src/eqvae in $kernel_dir" >&2
    exit 1
  fi

  if [[ ! -d "$kernel_dir/payload/configs/spec0001" ]]; then
    echo "error: missing bundled payload configs/spec0001 in $kernel_dir" >&2
    exit 1
  fi

  if grep -q "KAGGLE_SMOKE_READY = True" "$kernel_dir/$code_file"; then
    if ! grep -q 'kaggle_smoke_ready' \
      "docs/specs/0001-translatable-normal-vae-baseline.md"; then
      echo "error: spec 0001 does not authorize the narrow Kaggle smoke" >&2
      exit 1
    fi
    if ! grep -Eq '^\| `0001-translatable-normal-vae-baseline\.md` \|[^|]*kaggle smoke is `kaggle_smoke_ready`' \
      "docs/specs/README.md"; then
      echo "error: spec index does not authorize the narrow Kaggle smoke" >&2
      exit 1
    fi
  else
    if ! grep -Eq '^Implementation readiness: (locked / implementation-ready|implementation-ready|ready)$' \
      "docs/specs/0001-translatable-normal-vae-baseline.md"; then
      echo "error: spec 0001 is not locked as implementation-ready" >&2
      exit 1
    fi

    if ! grep -Eq '^\| `0001-translatable-normal-vae-baseline\.md` \|[^|]*locked / implementation-ready' \
      "docs/specs/README.md"; then
      echo "error: spec 0001 is not locked as implementation-ready in docs/specs/README.md" >&2
      exit 1
    fi
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
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    if str(data.get(key, "")).lower() != expected:
        errors.append(f"{key} must be {expected!r}")

dataset_sources = data.get("dataset_sources")
expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
forbidden_sources = {"maximusshtefan/non-eq-vae-output"}

if dataset_sources != expected_dataset_sources:
    errors.append(
        "dataset_sources must be exactly "
        f"{expected_dataset_sources!r} for the spec 0001 debug kernel"
    )

for source_group in (
    data.get("dataset_sources"),
    data.get("competition_sources"),
    data.get("kernel_sources"),
    data.get("model_sources"),
):
    if isinstance(source_group, list):
        for source in source_group:
            if source in forbidden_sources:
                errors.append(f"forbidden historical FSQ source: {source!r}")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  for required_hook in single_visible_t4 dual_t4_ddp wrong_accelerator; do
    if ! grep -q "$required_hook" "$kernel_dir/$code_file"; then
      echo "error: launcher must include $required_hook runtime validation hook" >&2
      exit 1
    fi
  done
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

api_check() {
  require_remote_confirmed
  require_kaggle_cli

  echo "Kaggle API read-only preflight"
  echo "=============================="
  kaggle --version

  kaggle auth print-access-token >/dev/null
  echo "ok: OAuth access token can be generated"

  kaggle kernels list --mine --search non-eq-vae --csv >/dev/null
  echo "ok: kernels list can see non-eq-vae"

  kaggle kernels status maximusshtefan/non-eq-vae >/dev/null
  echo "ok: kernels status works for maximusshtefan/non-eq-vae"

  kaggle kernels logs maximusshtefan/non-eq-vae >/dev/null
  echo "ok: kernels logs works for maximusshtefan/non-eq-vae"

  kaggle datasets files maximusshtefan/patches-pre-shuffled-ubc-ocean -v >/dev/null
  echo "ok: dataset file listing works for patches-pre-shuffled-ubc-ocean"

  if kaggle quota -v >/dev/null 2>&1; then
    echo "ok: accelerator quota endpoint works"
  else
    echo "warn: accelerator quota endpoint failed; verify quota in Kaggle UI before remote benchmark push" >&2
  fi

  if kaggle kernels files maximusshtefan/non-eq-vae -v >/dev/null 2>&1; then
    echo "ok: kernels files endpoint works"
  else
    echo "warn: kernels files endpoint failed; status/logs still work, but source-file introspection is unavailable" >&2
  fi
}

action="${1:-}"
case "$action" in
  build)
    build_kernel_payload "${2:-$default_kernel_dir}"
    ;;
  validate)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    ;;
  check)
    validate_kernel_dir "${2:-$default_kernel_dir}"
    require_kaggle_cli
    kaggle --version
    ;;
  api-check)
    api_check
    ;;
  push)
    kernel_dir="${2:-$default_kernel_dir}"
    if [[ "$#" -ge 2 ]]; then
      shift 2
    else
      shift 1
    fi
    if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    validate_kernel_dir "$kernel_dir"
    guard_push_ready "$kernel_dir"
    require_kaggle_cli
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
