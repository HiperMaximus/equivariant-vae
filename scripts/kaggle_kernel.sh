#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"
setup_kernel_dir="kaggle/kernels/setup_smoke"
setup_output_dir="runs/kaggle/setup_smoke"
synthetic_timing_kernel_dir="kaggle/kernels/synthetic_timing"
real_data_runtime_pretest_kernel_dir="kaggle/kernels/real_data_runtime_pretest"
real_data_runtime_pretest_output_dir="runs/kaggle/real_data_runtime_pretest"
runtime_selection_kernel_dir="kaggle/kernels/runtime_selection"
runtime_selection_output_dir="runs/kaggle/runtime_selection"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh preflight-runtime-selection
  ./scripts/kaggle_kernel.sh api-check
  ./scripts/kaggle_kernel.sh push [kernel_dir] [extra kaggle args...]
  ./scripts/kaggle_kernel.sh status [kernel_id]
  ./scripts/kaggle_kernel.sh status-setup
  ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
  ./scripts/kaggle_kernel.sh status-runtime-selection
  ./scripts/kaggle_kernel.sh output [kernel_id] [output_dir]
  ./scripts/kaggle_kernel.sh output-setup [output_dir]
  ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest [output_dir]
  ./scripts/kaggle_kernel.sh output-runtime-selection [output_dir]
  ./scripts/kaggle_kernel.sh pull [kernel_id] [kernel_dir]

Remote writes require KAGGLE_PUSH_CONFIRMED=1.
Remote writes with Kaggle source attachments also require
KAGGLE_FULL_DATASET_CONFIRMED=1.
Remote reads/downloads require KAGGLE_REMOTE_CONFIRMED=1.
Remote pulls require both KAGGLE_REMOTE_CONFIRMED=1 and
KAGGLE_PULL_CONFIRMED=1.
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

  if [[ "$kernel_dir" == "$real_data_runtime_pretest_kernel_dir" ]]; then
    python3 scripts/build_kaggle_embedded_kernel.py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: real-data runtime pretest embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$runtime_selection_kernel_dir" ]]; then
    python3 scripts/build_kaggle_embedded_kernel.py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_RUNTIME_SELECTION_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: runtime-selection embedded payload matches current worktree"
  fi
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
    maximusshtefan/eqvae-synthetic-timing)
      printf '%s\n' "KAGGLE_SYNTHETIC_TIMING_READY = True"
      ;;
    maximusshtefan/eqvae-real-data-runtime-pretest)
      printf '%s\n' "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True"
      ;;
    maximusshtefan/eqvae-runtime-selection)
      printf '%s\n' "KAGGLE_RUNTIME_SELECTION_READY = True"
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

  if grep -q "KAGGLE_SYNTHETIC_TIMING_READY = True" "$kernel_dir/$code_file"; then
    guard_synthetic_timing_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" "$kernel_dir/$code_file"; then
    guard_real_data_runtime_pretest_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_RUNTIME_SELECTION_READY = True" "$kernel_dir/$code_file"; then
    guard_runtime_selection_push_ready "$kernel_dir" "$metadata"
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

guard_synthetic_timing_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: synthetic timing must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not set KAGGLE_FULL_DATASET_CONFIRMED=1 for no-dataset synthetic timing" >&2
    exit 1
  fi

  if ! grep -q 'kaggle_synthetic_timing_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not authorize the synthetic timing contract" >&2
    exit 1
  fi
  if ! grep -q 'synthetic binary timing pretest contract is `kaggle_synthetic_timing_contract_ready`' \
    "docs/specs/README.md"; then
    echo "error: spec index does not authorize the synthetic timing contract" >&2
    exit 1
  fi
  if ! grep -q 'The synthetic binary timing pretest workflow becomes Kaggle-push-ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not describe synthetic timing push readiness" >&2
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
    "id": "maximusshtefan/eqvae-synthetic-timing",
    "title": "eqvae synthetic timing",
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

for source_field in (
    "dataset_sources",
    "competition_sources",
    "kernel_sources",
    "model_sources",
):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for synthetic timing")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SYNTHETIC_TIMING_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: synthetic timing run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/synthetic_timing.py",
        ).decode("utf-8")
    except KeyError:
        print(
            "error: synthetic timing payload is missing synthetic_timing.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

required_source_text = (
    'DEFAULT_PROFILE_NAME = "synthetic_binary_2gib_histology_like_v1"',
    "DEFAULT_TOTAL_PATCHES = 10_912",
    "DEFAULT_SPLIT_PATCHES = 5_456",
    'COMPACT_PROFILE_NAME = "synthetic_binary_0p81gb_histology_like_v1"',
    "COMPACT_TOTAL_PATCHES = 4_096",
    "COMPACT_SPLIT_PATCHES = 2_048",
    "def compact_synthetic_timing_profile()",
    "REPEAT_SHORTLIST_WARMUP_STEPS = 5",
    "REPEAT_SHORTLIST_MEASURED_STEPS = 25",
    "def repeat_shortlist_row_specs()",
)
missing = [text for text in required_source_text if text not in source]
if missing:
    for text in missing:
        print(
            f"error: synthetic timing embedded source missing required text: {text}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY

  if grep -q "selected_runtime" "$run_file"; then
    echo "error: synthetic timing launcher must not reference selected runtime artifacts" >&2
    exit 1
  fi

  for required_text in \
    "synthetic_timing_manifest.json" \
    "synthetic_timing_runtime_proof.json" \
    "synthetic_timing_matrix.csv" \
    "synthetic_timing_recommendations.json" \
    "non_promotable_synthetic_timing" \
    "kaggle_synthetic_timing_pretest" \
    "kaggle_no_dataset_generated_ubc_shards" \
    "blocked_claims" \
    "/kaggle/working" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "eqvae_synthetic_timing_repeat_shortlist" \
    "repeat_shortlist_row_specs" \
    "SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST" \
    "wrong_accelerator"; do
    if ! grep -q "$required_text" "$run_file"; then
      echo "error: synthetic timing run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_real_data_runtime_pretest_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: real-data runtime pretest must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the real-data runtime pretest.
EOF
    exit 1
  fi

  if ! grep -q 'real_data_runtime_pretest_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not authorize the real-data runtime pretest contract" >&2
    exit 1
  fi
  if ! grep -q 'real_data_runtime_pretest_contract_ready' \
    "docs/specs/README.md"; then
    echo "error: spec index does not authorize the real-data runtime pretest contract" >&2
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
    "id": "maximusshtefan/eqvae-real-data-runtime-pretest",
    "title": "eqvae real data runtime pretest",
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

expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if data.get("dataset_sources") != expected_dataset_sources:
    errors.append(f"dataset_sources must be exactly {expected_dataset_sources!r}")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for real-data pretest")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: real-data pretest run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/real_data_runtime_pretest.py",
        ).decode("utf-8")
        config = json.loads(
            archive.read(
                "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
            ),
        )
    except KeyError as error:
        print(f"error: real-data pretest payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

errors: list[str] = []
if "write_synthetic_benchmark_artifacts" in source:
    errors.append("real-data pretest payload must not call schema selected-runtime writer")
if "selected_runtime_path" in source:
    errors.append("real-data pretest payload must not define selected_runtime_path")
if "def _reject_selected_runtime_artifact" not in source:
    errors.append("real-data pretest payload must reject stale selected_runtime artifacts")
if source.count("_reject_selected_runtime_artifact(") < 3:
    errors.append("real-data pretest payload must check selected_runtime before and after writes")
if re.search(r"write_json\s*\([^)]*selected_runtime", source, flags=re.DOTALL):
    errors.append("real-data pretest payload must not write selected_runtime artifacts")

data = config.get("data")
runtime = config.get("runtime_matrix")
pretest = config.get("runtime_pretest")
if config.get("status") != "real_data_runtime_pretest_kernel_guard_ready_non_promotable":
    errors.append(
        "config.status must be real_data_runtime_pretest_kernel_guard_ready_non_promotable",
    )
if not isinstance(data, dict):
    errors.append("config.data must be an object")
else:
    if data.get("dataset_slug") != "maximusshtefan/patches-pre-shuffled-ubc-ocean":
        errors.append("config.data.dataset_slug must be the pre-shuffled patch dataset")
    cap = data.get("benchmark_cap")
    if not isinstance(cap, dict):
        errors.append("config.data.benchmark_cap must be an object")
    else:
        if cap.get("train_patch_count") != 8192:
            errors.append("benchmark_cap.train_patch_count must be 8192")
        if cap.get("validation_patch_count") != 2048:
            errors.append("benchmark_cap.validation_patch_count must be 2048")
        if cap.get("full_epoch_allowed") is not False:
            errors.append("benchmark_cap.full_epoch_allowed must be false")
if not isinstance(runtime, dict):
    errors.append("config.runtime_matrix must be an object")
else:
    settle = runtime.get("compile_settle_policy")
    if not isinstance(settle, dict) or settle.get("compile_settle_steps") != 5:
        errors.append("compile_settle_steps must be 5")
if not isinstance(pretest, dict):
    errors.append("config.runtime_pretest must be an object")
else:
    if pretest.get("full_run_eligible") is not False:
        errors.append("runtime_pretest.full_run_eligible must be false")
    if pretest.get("writes_selected_runtime") is not False:
        errors.append("runtime_pretest.writes_selected_runtime must be false")
    artifacts = pretest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("runtime_pretest.artifacts must be an object")
    elif "selected_runtime" in artifacts:
        errors.append("runtime_pretest.artifacts must not include selected_runtime")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  for required_text in \
    "real_data_runtime_pretest_manifest.json" \
    "runtime_proof.json" \
    "runtime_matrix.csv" \
    "dataloader_matrix.csv" \
    "numerical_checks.csv" \
    "corruption_checks.csv" \
    "gate_health_summary.json" \
    "real_data_runtime_pretest_recommendations.json" \
    "phase_timings.json" \
    "non_promotable_real_data_runtime_pretest" \
    "real_data_runtime_pretest" \
    "blocked_claims" \
    "selected_runtime.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "wrong_accelerator"; do
    if ! grep -q "$required_text" "$run_file"; then
      echo "error: real-data runtime pretest run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_runtime_selection_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: runtime selection must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime benchmark.
EOF
    exit 1
  fi

  if ! grep -q 'v8_shortlist_eager_amp_then_dual_gate' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not describe the v8 selected-runtime slice" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not authorize runtime-selection kernel push readiness" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe runtime-selection kernel push readiness" >&2
    exit 1
  fi
  if ! grep -q 'runtime_selection_kernel_ready' \
    "docs/specs/README.md"; then
    echo "error: specs index does not describe runtime-selection kernel push readiness" >&2
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
    "id": "maximusshtefan/eqvae-runtime-selection",
    "title": "eqvae runtime selection",
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

expected_dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if data.get("dataset_sources") != expected_dataset_sources:
    errors.append(f"dataset_sources must be exactly {expected_dataset_sources!r}")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list for runtime selection")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  python3 scripts/build_kaggle_embedded_kernel.py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_RUNTIME_SELECTION_READY = True" \
    --verify-only

  local run_file="$kernel_dir/run.py"
  python3 - "$run_file" <<'PY'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: runtime-selection run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_files = {
        "src/eqvae/benchmarking/runtime_selection.py",
        "src/eqvae/benchmarking/runtime_selection_executor.py",
        "src/eqvae/cli/runtime_selection_executor.py",
        "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
        "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_proof.json",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/runtime_matrix.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/dataloader_matrix.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/numerical_checks.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/corruption_checks.csv",
        "runs/kaggle/real_data_runtime_pretest_v8/benchmark/gate_health_summary.json",
        "runs/kaggle/real_data_runtime_pretest_v8/metrics/gate_health.csv",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    unexpected_v8 = sorted(
        name for name in names
        if name.startswith("runs/kaggle/real_data_runtime_pretest_v8/")
        and name not in required_files
    )
    if unexpected_v8:
        errors.append(f"embedded payload has unexpected v8 files: {unexpected_v8!r}")
    try:
        executor_source = archive.read(
            "src/eqvae/benchmarking/runtime_selection_executor.py",
        ).decode("utf-8")
        writer_source = archive.read(
            "src/eqvae/benchmarking/runtime_selection.py",
        ).decode("utf-8")
        config = json.loads(
            archive.read(
                "configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json",
            ),
        )
        baseline = json.loads(
            archive.read(
                "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json",
            ),
        )
    except KeyError as error:
        print(f"error: runtime-selection payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

    required_executor_text = (
        "RuntimeSelectionEvidence",
        "write_runtime_selection_benchmark",
        "--nproc_per_node=2",
        "torch.distributed.run",
        "DistributedDataParallel",
        "stain_corruptor_qa",
    )
    for text in required_executor_text:
        if text not in executor_source:
            errors.append(f"runtime-selection executor missing required text: {text}")
    required_writer_text = (
        "do_not_write_selected_runtime_if_missing_failed_or_skipped",
        "v8_hash_provenance_not_pass",
        "compiled_rows_diagnostic_only",
    )
    for text in required_writer_text:
        if text not in writer_source:
            errors.append(f"runtime-selection writer missing required text: {text}")
    runtime = config.get("runtime_matrix")
    if not isinstance(runtime, dict):
        errors.append("config.runtime_matrix must be an object")
    else:
        selection = runtime.get("selection_benchmark_slice")
        if not isinstance(selection, dict):
            errors.append("config.runtime_matrix.selection_benchmark_slice must be an object")
        elif selection.get("name") != "v8_shortlist_eager_amp_then_dual_gate":
            errors.append("selection slice must be v8_shortlist_eager_amp_then_dual_gate")
        else:
            efficiency = selection.get("efficiency_followup")
            if not isinstance(efficiency, dict):
                errors.append("selection efficiency_followup must be an object")
            else:
                expected_row = efficiency.get("baseline_row_id")
                expected_policy = efficiency.get("baseline_runtime_policy_id")
                snapshot = baseline.get("selected_row_snapshot")
                if not isinstance(snapshot, dict):
                    errors.append("baseline selected runtime must contain a snapshot")
                elif (
                    baseline.get("status") != "pass"
                    or baseline.get("selected_row_id") != expected_row
                    or baseline.get("runtime_policy_id") != expected_policy
                    or snapshot.get("row_id") != expected_row
                    or snapshot.get("runtime_policy_id") != expected_policy
                    or snapshot.get("status") != "pass"
                ):
                    errors.append("baseline selected runtime does not match config")
        carry = runtime.get("v8_carry_forward")
        if not isinstance(carry, dict):
            errors.append("config.runtime_matrix.v8_carry_forward must be an object")
        elif carry.get("full_run_eligible") is not False:
            errors.append("v8 carry-forward artifacts must remain non-promotable")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PY

  for required_text in \
    "KAGGLE_RUNTIME_SELECTION_READY = True" \
    "v8_shortlist_eager_amp_then_dual_gate" \
    "runtime_selection_executor" \
    "selected_runtime.json" \
    "stain_corruptor_qa.json" \
    "runtime_proof.json" \
    "runtime_matrix.csv" \
    "dataloader_matrix.csv" \
    "numerical_checks.csv" \
    "corruption_checks.csv" \
    "gate_health_summary.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "torchrun" \
    "--nproc_per_node=2" \
    "wrong_accelerator"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: runtime-selection run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
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
    "docs/data/ubc_ocean_masked_holdout_ids.csv": digest_file(
        Path("docs/data/ubc_ocean_masked_holdout_ids.csv")
    ),
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

preflight_runtime_selection() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  build_embedded_kernel "$runtime_selection_kernel_dir"
  validate_kernel_dir "$runtime_selection_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_runtime_selection_benchmark.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_runtime_selection_kernel_import_simulation \
    tests/test_kaggle_embedded_kernel.py::test_embedded_runtime_selection_kernel_full_local_fail_closed_simulation \
    -q
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
  preflight-runtime-selection)
    preflight_runtime_selection
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
    require_kaggle_sources_confirmed "$(metadata_path "$kernel_dir")"
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
  status-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle kernels status "$kernel_id"
    ;;
  status-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
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
  output-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    output_dir="${2:-$real_data_runtime_pretest_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
    output_dir="${2:-$runtime_selection_output_dir}"
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
    require_remote_confirmed
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
