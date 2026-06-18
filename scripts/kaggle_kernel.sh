#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"
setup_kernel_dir="kaggle/kernels/setup_smoke"
setup_output_dir="runs/kaggle/setup_smoke"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh api-check
  ./scripts/kaggle_kernel.sh push [kernel_dir] [extra kaggle args...]
  ./scripts/kaggle_kernel.sh status [kernel_id]
  ./scripts/kaggle_kernel.sh status-setup
  ./scripts/kaggle_kernel.sh output [kernel_id] [output_dir]
  ./scripts/kaggle_kernel.sh output-setup [output_dir]
  ./scripts/kaggle_kernel.sh pull [kernel_id] [kernel_dir]

Remote writes require KAGGLE_PUSH_CONFIRMED=1.
Remote writes with Kaggle source attachments also require
KAGGLE_FULL_DATASET_CONFIRMED=1.
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

require_kaggle_sources_confirmed() {
  local metadata="$1"
  local source_summary
  source_summary="$(python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_fields = (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
)
nonempty = {}
for field in source_fields:
    sources = data.get(field)
    if sources is None:
        continue
    if not isinstance(sources, list):
        print(f"error: {field} must be a list", file=sys.stderr)
        raise SystemExit(2)
    if sources:
        nonempty[field] = sources
if nonempty:
    print(json.dumps(nonempty, sort_keys=True))
PY
)"

  if [[ -n "$source_summary" && "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<EOF
error: $metadata declares Kaggle source attachments: $source_summary

Kaggle source attachments can make Kaggle prepare external datasets, kernels,
models, or competition inputs before the script starts. Set
KAGGLE_FULL_DATASET_CONFIRMED=1 only after explicitly deciding to attach those
sources for this push. Use a no-dataset
synthetic/random benchmark kernel for setup or timing-plumbing tests.
EOF
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
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

payload = Path(sys.argv[1])
if payload.exists():
    shutil.rmtree(payload)
(payload / "src").mkdir(parents=True)
(payload / "configs").mkdir(parents=True)
ignore_generated = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
shutil.copytree("src/eqvae", payload / "src" / "eqvae", ignore=ignore_generated)
shutil.copytree(
    "configs/spec0001",
    payload / "configs" / "spec0001",
    ignore=ignore_generated,
)
shutil.copy2("pyproject.toml", payload / "pyproject.toml")
shutil.copy2("uv.lock", payload / "uv.lock")


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(digest_file(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


manifest = {
    "schema_version": "spec0001.kaggle_payload_manifest.v1",
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_dirty": bool(git_output("status", "--short")),
    "entries": {
        "src/eqvae": digest_tree(payload / "src" / "eqvae"),
        "configs/spec0001": digest_tree(payload / "configs" / "spec0001"),
        "pyproject.toml": digest_file(payload / "pyproject.toml"),
        "uv.lock": digest_file(payload / "uv.lock"),
    },
}

(payload / "payload_manifest.json").write_text(
    f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
    encoding="utf-8",
)
PY

  echo "ok: built $payload_dir"
}

is_setup_kernel_dir() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  if [[ ! -f "$metadata" ]]; then
    return 1
  fi
  [[ "$(json_field "$metadata" id)" == "maximusshtefan/eqvae-setup-smoke" ]]
}

build_embedded_setup_kernel() {
  local kernel_dir="${1:-$setup_kernel_dir}"
  build_embedded_kernel "$kernel_dir"
}

build_embedded_kernel() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  local ready_marker
  metadata="$(metadata_path "$kernel_dir")"
  ready_marker="$(embedded_ready_marker "$kernel_dir")"

  if [[ ! -f "$metadata" ]]; then
    echo "missing: $metadata" >&2
    exit 1
  fi
  if [[ ! -f "$kernel_dir/run_template.py" ]]; then
    echo "missing: $kernel_dir/run_template.py" >&2
    exit 1
  fi

  python3 -m json.tool "$metadata" >/dev/null
  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "$ready_marker" \
    --allow-dirty
  validate_kernel_dir "$kernel_dir"
}

embedded_ready_marker() {
  local kernel_dir="${1:-$default_kernel_dir}"
  local metadata
  metadata="$(metadata_path "$kernel_dir")"
  case "$(json_field "$metadata" id)" in
    maximusshtefan/eqvae-setup-smoke)
      printf '%s\n' "KAGGLE_SETUP_SMOKE_READY = True"
      ;;
    maximusshtefan/non-eq-vae-debug)
      printf '%s\n' "KAGGLE_SMOKE_READY = True"
      ;;
    *)
      printf '%s\n' "KAGGLE_SMOKE_READY = True"
      ;;
  esac
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

  if grep -q "KAGGLE_SETUP_SMOKE_READY = True" "$kernel_dir/$code_file"; then
    guard_setup_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if [[ ! -f "docs/behavior_inventory_kaggle.md" ]]; then
    echo "error: missing docs/behavior_inventory_kaggle.md" >&2
    exit 1
  fi

  if grep -q "KAGGLE_SMOKE_READY = True" "$kernel_dir/$code_file"; then
    guard_real_smoke_push_ready "$kernel_dir" "$metadata"
    return
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
}

guard_real_smoke_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

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

  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SMOKE_READY = True" \
    --verify-only

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/non-eq-vae-debug",
    "title": "non-eq-VAE debug",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

dataset_sources = data.get("dataset_sources")
expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
forbidden_sources = {"maximusshtefan/non-eq-vae-output"}

if dataset_sources != expected_dataset_sources:
    errors.append(
        "dataset_sources must be exactly "
        f"{expected_dataset_sources!r} for the spec 0001 debug kernel"
    )

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for the spec 0001 debug kernel")

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

  python3 - "$kernel_dir/run.py" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile

run_text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: generated run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)
zip_bytes = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
    config = json.loads(
        archive.read("configs/spec0001/non_eq_vae_kaggle_debug.json"),
    )

smoke = config.get("kaggle_smoke")
errors: list[str] = []
if not isinstance(smoke, dict):
    errors.append("payload config must contain kaggle_smoke object")
else:
    expected = {
        "full_run_eligible": False,
        "batch_size": 1,
        "max_validation_batches": 1,
        "num_workers": 0,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            errors.append(f"kaggle_smoke.{key} must be {value!r}")
    max_train_steps = smoke.get("max_train_steps")
    if not isinstance(max_train_steps, int) or not 1 <= max_train_steps <= 3:
        errors.append("kaggle_smoke.max_train_steps must be an integer from 1 to 3")
    if smoke.get("benchmark_source") != "kaggle_script_kernel_capped_smoke":
        errors.append("kaggle_smoke.benchmark_source must identify capped smoke")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  for required_hook in single_visible_t4 dual_t4_ddp wrong_accelerator; do
    if ! grep -q "$required_hook" "$kernel_dir/run.py"; then
      echo "error: launcher must include $required_hook runtime validation hook" >&2
      exit 1
    fi
  done
}

guard_setup_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: setup smoke must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if ! grep -q 'kaggle_setup_smoke_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not authorize the synthetic setup smoke" >&2
    exit 1
  fi

  if ! grep -q 'synthetic no-dataset setup smoke' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe setup-smoke evidence" >&2
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
    "id": "maximusshtefan/eqvae-setup-smoke",
    "title": "eqvae setup smoke",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "false",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

if data.get("machine_shape") not in (None, "", "None"):
    errors.append("setup smoke machine_shape must be absent or empty")

for source_field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for setup smoke")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  if ! grep -q "synthetic_kaggle_setup_smoke" "$kernel_dir/run.py"; then
    echo "error: setup run.py must declare synthetic_kaggle_setup_smoke" >&2
    exit 1
  fi

  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SETUP_SMOKE_READY = True" \
    --verify-only
}

validate_payload_freshness() {
  local payload_dir="$1"
  python3 - "$payload_dir" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

payload = Path(sys.argv[1])
manifest_path = payload / "payload_manifest.json"
if not manifest_path.exists():
    print("error: missing payload_manifest.json; rebuild kernel payload", file=sys.stderr)
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors: list[str] = []


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(digest_file(item).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if manifest.get("schema_version") != "spec0001.kaggle_payload_manifest.v1":
    errors.append("payload manifest has an unexpected schema_version")
if manifest.get("git_dirty") is not False:
    errors.append("payload was built from a dirty git worktree")
if manifest.get("git_commit") != git_output("rev-parse", "HEAD"):
    errors.append("payload git_commit does not match current HEAD; rebuild payload")

entries = manifest.get("entries")
expected_entries = {
    "src/eqvae": digest_tree(Path("src/eqvae")),
    "configs/spec0001": digest_tree(Path("configs/spec0001")),
    "pyproject.toml": digest_file(Path("pyproject.toml")),
    "uv.lock": digest_file(Path("uv.lock")),
}
if not isinstance(entries, dict):
    errors.append("payload manifest entries must be an object")
else:
    for key, expected in expected_entries.items():
        if entries.get(key) != expected:
            errors.append(f"payload entry {key!r} is stale; rebuild payload")

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

api_check() {
  require_remote_confirmed
  require_kaggle_cli
  local kernel_id
  kernel_id="$(kernel_id_from_metadata "$default_kernel_dir")"

  echo "Kaggle API read-only preflight"
  echo "=============================="
  kaggle --version

  kaggle auth print-access-token >/dev/null
  echo "ok: OAuth access token can be generated"

  kaggle kernels list --mine --search "${kernel_id#*/}" --csv >/dev/null
  echo "ok: kernels list can see $kernel_id"

  kaggle kernels status "$kernel_id" >/dev/null
  echo "ok: kernels status works for $kernel_id"

  kaggle kernels logs "$kernel_id" >/dev/null
  echo "ok: kernels logs works for $kernel_id"

  kaggle datasets files maximusshtefan/patches-pre-shuffled-ubc-ocean -v >/dev/null
  echo "ok: dataset file listing works for patches-pre-shuffled-ubc-ocean"

  if kaggle quota -v >/dev/null 2>&1; then
    echo "ok: accelerator quota endpoint works"
  else
    echo "warn: accelerator quota endpoint failed; verify quota in Kaggle UI before remote benchmark push" >&2
  fi

  if kaggle kernels files "$kernel_id" -v >/dev/null 2>&1; then
    echo "ok: kernels files endpoint works"
  else
    echo "warn: kernels files endpoint failed; status/logs still work, but source-file introspection is unavailable" >&2
  fi
}

action="${1:-}"
case "$action" in
  build)
    kernel_dir="${2:-$default_kernel_dir}"
    if [[ -f "$kernel_dir/run_template.py" ]]; then
      build_embedded_kernel "$kernel_dir"
    elif is_setup_kernel_dir "$kernel_dir"; then
      build_embedded_setup_kernel "$kernel_dir"
    else
      build_kernel_payload "$kernel_dir"
    fi
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
    require_kaggle_sources_confirmed "$(metadata_path "$kernel_dir")"
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
  status-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
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
  output-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
    output_dir="${2:-$setup_output_dir}"
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
