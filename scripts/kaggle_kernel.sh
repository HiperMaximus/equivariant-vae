#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="$PWD/runs/local_tmp/kaggle_kernel_$$"
  trap 'rm -rf "$TMPDIR"' EXIT
fi
mkdir -p "$TMPDIR"

# The kernel build imports eqvae (editable-installed into .venv by `uv sync`) to reuse the
# single-sourced schedule helper + patch count, so it MUST run on the venv interpreter.
# Bare python3 is the system interpreter: it has neither torch nor eqvae, and using it is
# what forced the old load-a-module-by-file-path workaround. Fail closed with an
# actionable message rather than falling back and dying deep inside a build.
build_python="${PYTHON:-.venv/bin/python}"

build_kernel_py() {
  # Probe EXACTLY what the build imports, not something weaker. Two traps here:
  #   -x only proves a file is executable, and a stale venv (built before the project had
  #     a [build-system], so eqvae was never installed) passes that trivially.
  #   `import eqvae` only proves src/ is reachable: src/eqvae/__init__.py is a 162-byte
  #     docstring that imports nothing, so `PYTHONPATH=src /usr/bin/python3` passes it and
  #     the build then dies at `from torch import Tensor` with a raw traceback -- the exact
  #     failure this guard exists to replace with a hint.
  # eqvae.benchmarking.__init__ pulls torch, so this probe fails closed on a torch-less
  # interpreter AND on one that cannot see eqvae at all.
  if ! "$build_python" -c 'import eqvae.benchmarking' >/dev/null 2>&1; then
    echo "error: kernel build needs a venv interpreter with eqvae AND torch importable" >&2
    echo "       tried: $build_python" >&2
    echo "hint:  uv sync --locked --python 3.12 --group dev" >&2
    exit 1
  fi
  "$build_python" scripts/build_kaggle_embedded_kernel.py "$@"
}

default_kernel_dir="kaggle/kernels/non_eq_vae_debug"
default_output_dir="runs/kaggle/non_eq_vae_debug"
setup_kernel_dir="kaggle/kernels/setup_smoke"
setup_output_dir="runs/kaggle/setup_smoke"
synthetic_timing_kernel_dir="kaggle/kernels/synthetic_timing"
real_data_runtime_pretest_kernel_dir="kaggle/kernels/real_data_runtime_pretest"
real_data_runtime_pretest_output_dir="runs/kaggle/real_data_runtime_pretest"
runtime_selection_kernel_dir="kaggle/kernels/runtime_selection"
runtime_selection_output_dir="runs/kaggle/runtime_selection"
selected_runtime_debug_kernel_dir="kaggle/kernels/selected_runtime_debug"
selected_runtime_debug_output_dir="runs/kaggle/selected_runtime_debug"
selected_runtime_lr_range_kernel_dir="kaggle/kernels/selected_runtime_lr_range"
selected_runtime_lr_range_output_dir="runs/kaggle/selected_runtime_lr_range"
selected_runtime_full_kernel_dir="kaggle/kernels/selected_runtime_full"
selected_runtime_full_output_dir="runs/kaggle/selected_runtime_full"
fixed25_selector_kernel_dir="kaggle/kernels/fixed25_selector"
fixed25_selector_output_dir="runs/kaggle/fixed25_selector"
selected_runtime_compile_probe_kernel_dir="kaggle/kernels/selected_runtime_compile_probe"
so2_architecture_probe_kernel_dir="kaggle/kernels/so2_architecture_probe"
so2_architecture_probe_output_dir="runs/kaggle/so2_architecture_probe_v3"
so2_runtime_readiness_kernel_dir="kaggle/kernels/so2_runtime_readiness"
so2_runtime_readiness_output_dir="runs/kaggle/so2_runtime_readiness_v1"
so2_prelaunch_kernel_dir="kaggle/kernels/so2_prelaunch"
so2_prelaunch_output_dir="runs/kaggle/so2_prelaunch"
so2_full_kernel_dir="kaggle/kernels/so2_selected_runtime_full"
so2_full_output_dir="runs/kaggle/so2_selected_runtime_full"
so2_full_session1_output_dir="runs/kaggle/so2_selected_runtime_full_v1_session1"
so2_full_resume_authority_dir="runs/kaggle/so2_selected_runtime_full_v3_session3"
so2_full_resume_dataset_dir="runs/kaggle/so2_session3_resume_dataset"
so2_full_resume_dataset_slug="maximusshtefan/eqvae-so2-session3-step27000"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/kaggle_kernel.sh build [kernel_dir]
  ./scripts/kaggle_kernel.sh validate [kernel_dir]
  ./scripts/kaggle_kernel.sh check [kernel_dir]
  ./scripts/kaggle_kernel.sh preflight-runtime-selection
  ./scripts/kaggle_kernel.sh preflight-fixed32-selector-readiness
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-runner
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-lr-range
  ./scripts/kaggle_kernel.sh preflight-selected-runtime-full
  ./scripts/kaggle_kernel.sh preflight-fixed25-selector
  ./scripts/kaggle_kernel.sh preflight-so2-architecture-probe
  ./scripts/kaggle_kernel.sh preflight-so2-runtime-readiness
  ./scripts/kaggle_kernel.sh preflight-so2-prelaunch
  ./scripts/kaggle_kernel.sh preflight-so2-selected-runtime-full
  ./scripts/kaggle_kernel.sh api-check [kernel_dir]
  ./scripts/kaggle_kernel.sh push [kernel_dir] [--wait [--wait-interval N] [--wait-max N] [--wait-queued N]] [extra kaggle args...]
  ./scripts/kaggle_kernel.sh status [kernel_id]
  ./scripts/kaggle_kernel.sh status-setup
  ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
  ./scripts/kaggle_kernel.sh status-runtime-selection
  ./scripts/kaggle_kernel.sh status-selected-runtime-debug
  ./scripts/kaggle_kernel.sh status-selected-runtime-lr-range
  ./scripts/kaggle_kernel.sh status-selected-runtime-full
  ./scripts/kaggle_kernel.sh status-fixed25-selector
  ./scripts/kaggle_kernel.sh status-so2-architecture-probe
  ./scripts/kaggle_kernel.sh status-so2-runtime-readiness
  ./scripts/kaggle_kernel.sh status-so2-prelaunch
  ./scripts/kaggle_kernel.sh status-so2-selected-runtime-full
  ./scripts/kaggle_kernel.sh wait [kernel_id] [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-fixed25-selector [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-selected-runtime-full [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-so2-architecture-probe [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh wait-so2-runtime-readiness [poll_seconds] [max_polls] [max_queued_seconds]
  ./scripts/kaggle_kernel.sh output [kernel_id] [output_dir]
  ./scripts/kaggle_kernel.sh output-setup [output_dir]
  ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest [output_dir]
  ./scripts/kaggle_kernel.sh output-runtime-selection [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-debug [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-lr-range [output_dir]
  ./scripts/kaggle_kernel.sh output-selected-runtime-full [output_dir]
  ./scripts/kaggle_kernel.sh output-fixed25-selector [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-architecture-probe [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-runtime-readiness [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-prelaunch [output_dir]
  ./scripts/kaggle_kernel.sh output-so2-selected-runtime-full [output_dir]
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

kaggle_tool_python() {
  local kaggle_bin
  local shebang
  local interpreter
  local interpreter_command
  local interpreter_name
  local env_interpreter
  kaggle_bin="$(command -v kaggle)"
  if ! IFS= read -r shebang <"$kaggle_bin"; then
    return 1
  fi
  if [[ "$shebang" != '#!'* ]]; then
    return 1
  fi
  interpreter="${shebang#\#!}"
  interpreter_command="${interpreter%% *}"
  interpreter_name="$(basename "$interpreter_command")"
  if [[ -x "$interpreter_command" && "$interpreter_name" == python* ]]; then
    printf '%s\n' "$interpreter_command"
    return 0
  fi
  if [[ "$interpreter" == /usr/bin/env\ * ]]; then
    env_interpreter="${interpreter#/usr/bin/env }"
    if [[ "$env_interpreter" == -S\ * ]]; then
      env_interpreter="${env_interpreter#-S }"
    fi
    env_interpreter="${env_interpreter%% *}"
    if [[ "$(basename "$env_interpreter")" == python* ]] \
      && command -v "$env_interpreter" >/dev/null 2>&1; then
      command -v "$env_interpreter"
      return 0
    fi
  fi
  return 1
}

kaggle_api() {
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    local kaggle_python
    if kaggle_python="$(kaggle_tool_python)"; then
      "$kaggle_python" scripts/kaggle_oauth_exec.py "$@"
      return
    fi
    cat >&2 <<'EOF'
error: Kaggle OAuth credentials are present, but the Kaggle CLI Python
interpreter could not be resolved for the fresh-token wrapper.
Set KAGGLE_DISABLE_FRESH_OAUTH=1 only when intentionally debugging raw Kaggle
CLI authentication.
EOF
    exit 1
  fi

  kaggle "$@"
}

kaggle_auth_path_message() {
  if [[ "${KAGGLE_DISABLE_FRESH_OAUTH:-}" != "1" \
    && -f "${HOME}/.kaggle/credentials.json" ]]; then
    if kaggle_tool_python >/dev/null; then
      echo "ok: fresh OAuth wrapper selected for authenticated Kaggle calls"
      return
    fi
    cat >&2 <<'EOF'
error: Kaggle OAuth credentials are present, but the Kaggle CLI Python
interpreter could not be resolved for the fresh-token wrapper.
Set KAGGLE_DISABLE_FRESH_OAUTH=1 only when intentionally debugging raw Kaggle
CLI authentication.
EOF
    exit 1
  fi

  echo "ok: raw Kaggle auth path selected for authenticated Kaggle calls"
}

require_remote_confirmed() {
  if [[ "${KAGGLE_REMOTE_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_REMOTE_CONFIRMED=1 after explicit user permission" >&2
    exit 1
  fi
}

wait_kernel_until_settled() {
  # Poll a kernel's status until it leaves the actively-pending states, then
  # print WAIT_SETTLED_STATUS=<outcome> and return so the caller is woken instead
  # of hanging. The two pending states are bounded separately:
  #   * RUNNING -- polled at the slow steady cadence (poll_interval) and bounded
  #     by max_polls; on exhaustion it prints TIMEOUT_STILL_PENDING (return 2).
  #   * QUEUED  -- polled faster and bounded by a shorter budget
  #     (max_queued_seconds); a kernel that never gets a compute slot is
  #     abandoned early with QUEUED_TIMEOUT (return 3) rather than tying the
  #     watcher up for the full multi-hour running backstop.
  # Every other status settles and returns 0: COMPLETE, ERROR, a cancellation
  # (CANCEL_REQUESTED / CANCEL_ACKNOWLEDGED, e.g. the Kaggle session time limit
  # killing the run), or any unrecognized status. Each poll goes through
  # kaggle_api, which mints a fresh OAuth token, so multi-hour waits stay
  # authenticated. A transient unparseable reply is retried against max_polls.
  # Every path wakes the caller.
  local kernel_id="$1"
  # Default to a 5-minute steady cadence: most watched kernels are multi-hour
  # runs, so a slow poll is plenty and stays far clear of API rate limits.
  local poll_interval="${2:-300}"
  local max_polls="${3:-180}"
  # Abandon a kernel stuck in QUEUED after this many seconds (default 5 min).
  local max_queued_seconds="${4:-300}"
  # Hard floor so no argument can hammer the Kaggle API into rate limiting.
  if ((poll_interval < 10)); then
    echo "wait: clamping poll interval to the 10s minimum (was ${poll_interval}s)" >&2
    poll_interval=10
  fi
  # Poll QUEUED faster than the steady cadence so the shorter queue budget is
  # enforced with useful granularity, but never below the floor or above the
  # steady interval.
  local queued_interval=30
  if ((queued_interval > poll_interval)); then
    queued_interval="$poll_interval"
  fi
  if ((queued_interval < 10)); then
    queued_interval=10
  fi
  local running_polls=0 queued_elapsed=0 status status_line
  while :; do
    status_line="$(kaggle_api kernels status "$kernel_id" 2>&1)" || true
    status="$(printf '%s\n' "$status_line" \
      | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1 || true)"
    status="${status#KernelWorkerStatus.}"
    case "$status" in
    QUEUED)
      echo "wait: QUEUED ${queued_elapsed}s/${max_queued_seconds}s"
      if ((queued_elapsed >= max_queued_seconds)); then
        echo "WAIT_SETTLED_STATUS=QUEUED_TIMEOUT"
        return 3
      fi
      sleep "$queued_interval"
      queued_elapsed=$((queued_elapsed + queued_interval))
      ;;
    RUNNING)
      queued_elapsed=0
      running_polls=$((running_polls + 1))
      echo "wait: RUNNING poll ${running_polls}/${max_polls}"
      if ((running_polls >= max_polls)); then
        echo "WAIT_SETTLED_STATUS=TIMEOUT_STILL_PENDING"
        return 2
      fi
      sleep "$poll_interval"
      ;;
    "")
      running_polls=$((running_polls + 1))
      printf 'wait: unparseable status (%s/%s); raw output follows\n%s\n' \
        "$running_polls" "$max_polls" "$status_line" >&2
      if ((running_polls >= max_polls)); then
        echo "WAIT_SETTLED_STATUS=TIMEOUT_STILL_PENDING"
        return 2
      fi
      sleep "$poll_interval"
      ;;
    *)
      echo "WAIT_SETTLED_STATUS=${status}"
      return 0
      ;;
    esac
  done
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
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: real-data runtime pretest embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$runtime_selection_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_RUNTIME_SELECTION_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: runtime-selection embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_debug_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime debug embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_lr_range_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime LR-range embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$selected_runtime_full_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SELECTED_RUNTIME_FULL_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: selected-runtime full embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$so2_prelaunch_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SO2_PRELAUNCH_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: SO2 prelaunch embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$so2_full_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: SO2 full embedded payload matches current worktree"
  fi

  if [[ "$kernel_dir" == "$fixed25_selector_kernel_dir" ]]; then
    build_kernel_py \
      --kernel-dir "$kernel_dir" \
      --ready-marker "KAGGLE_FIXED25_SELECTOR_READY = True" \
      --verify-only \
      --allow-dirty
    echo "ok: fixed25-selector embedded payload matches current worktree"
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
  build_kernel_py \
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
    maximusshtefan/eqvae-selected-runtime-debug)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-lr-range)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-full)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_FULL_READY = True"
      ;;
    maximusshtefan/eqvae-fixed25-selector)
      printf '%s\n' "KAGGLE_FIXED25_SELECTOR_READY = True"
      ;;
    maximusshtefan/eqvae-selected-runtime-compile-probe)
      printf '%s\n' "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True"
      ;;
    maximusshtefan/eqvae-so2-architecture-probe)
      printf '%s\n' "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True"
      ;;
    maximusshtefan/eqvae-so2-runtime-readiness)
      printf '%s\n' "KAGGLE_SO2_RUNTIME_READINESS_READY = True"
      ;;
    maximusshtefan/eqvae-so2-prelaunch)
      printf '%s\n' "KAGGLE_SO2_PRELAUNCH_READY = True"
      ;;
    maximusshtefan/eqvae-so2-selected-runtime-full)
      printf '%s\n' "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True"
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

  if grep -q "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_debug_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_lr_range_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_FULL_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_full_push_ready "$kernel_dir" "$metadata" "push"
    return
  fi

  if grep -q "KAGGLE_FIXED25_SELECTOR_READY = True" "$kernel_dir/$code_file"; then
    guard_fixed25_selector_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True" "$kernel_dir/$code_file"; then
    guard_selected_runtime_compile_probe_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_architecture_probe_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_RUNTIME_READINESS_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_runtime_readiness_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_PRELAUNCH_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_prelaunch_push_ready "$kernel_dir" "$metadata"
    return
  fi

  if grep -q "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True" "$kernel_dir/$code_file"; then
    guard_so2_full_push_ready "$kernel_dir" "$metadata"
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

  build_kernel_py \
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
    "enable_internet": "true",
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
    "enable_internet": "true",
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

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SETUP_SMOKE_READY = True" \
    --verify-only
}

guard_fixed25_selector_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: fixed25 selector must be a single generated run.py, not a payload" >&2
    exit 1
  fi

  if ! grep -q 'fixed25_selector_kernel_ready' \
    "docs/specs/0010-fixed25-equivariance-artifact-protocol.md"; then
    echo "error: spec 0010 does not authorize the fixed25 selector kernel" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_FULL_DATASET_CONFIRMED=1 to attach the UBC dataset for selector generation" >&2
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
    "id": "maximusshtefan/eqvae-fixed25-selector",
    "title": "eqvae fixed25 selector",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "true",
}

for key, expected in required_values.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")

if data.get("machine_shape") not in (None, "", "None"):
    errors.append("fixed25 selector machine_shape must be absent or empty (CPU-only)")

expected_datasets = [
    "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    "maximusshtefan/eqvae-baseline-session1-step15000",
]
if data.get("dataset_sources") != expected_datasets:
    errors.append("dataset_sources must attach the exact UBC and session-1 datasets")

for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be an empty list")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  local code_file
  code_file="$(json_field "$metadata" code_file)"
  local required_text
  for required_text in \
    "KAGGLE_FIXED25_SELECTOR_READY = True" \
    "fixed_25_validation" \
    "select_fixed_patches" \
    "fixed25_originals" \
    "originals.pt" \
    "originals.png" \
    "--validate-crc"; do
    if ! grep -q -- "$required_text" "$kernel_dir/$code_file"; then
      echo "error: fixed25 selector run.py missing required text: $required_text" >&2
      exit 1
    fi
  done

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_FIXED25_SELECTOR_READY = True" \
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
    "enable_internet": "true",
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

  build_kernel_py \
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

guard_selected_runtime_compile_probe_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: compile probe must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not set KAGGLE_FULL_DATASET_CONFIRMED=1 for the no-dataset compile probe" >&2
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
    "id": "maximusshtefan/eqvae-selected-runtime-compile-probe",
    "title": "eqvae selected runtime compile probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
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
        errors.append(f"{source_field} must be an empty list for the compile probe")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_COMPILE_PROBE_READY = True" \
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
    print("error: compile probe run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    try:
        source = archive.read(
            "src/eqvae/benchmarking/compiled_fastpath_probe.py",
        ).decode("utf-8")
    except KeyError:
        print(
            "error: compile probe payload is missing compiled_fastpath_probe.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

required_source_text = (
    'COMPILED_FASTPATH_PROBE_KIND = "kaggle_compiled_fastpath_probe"',
    'COMPILED_FASTPATH_PROBE_STATUS_SCOPE = "non_promotable_compiled_fastpath_probe"',
    'RECIPE_PYTHON_REDUCER = "python_reducer_whole_step"',
    'RECIPE_DDP_OPTIMIZER = "ddp_optimizer_whole_step"',
    "def run_compiled_fastpath_probe(",
    "def run_negative_control_desync(",
)
missing = [text for text in required_source_text if text not in source]
if missing:
    for text in missing:
        print(
            f"error: compile probe embedded source missing required text: {text}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY

  for required_text in \
    "compiled_fastpath_probe_proof.json" \
    "compiled_fastpath_probe_matrix.csv" \
    "compiled_fastpath_probe_manifest.json" \
    "non_promotable_compiled_fastpath_probe" \
    "kaggle_compiled_fastpath_probe" \
    "kaggle_no_dataset_synthetic_compiled_fastpath" \
    "blocked_claims" \
    "/kaggle/working" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.compiled_fastpath_probe"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: compile probe run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_so2_architecture_probe_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: SO(2) probe must be one generated run.py, not a sibling payload" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not attach a dataset to the generated-tensor SO(2) probe" >&2
    exit 1
  fi

  python3 - "$metadata" <<'PYSO2METADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "id": "maximusshtefan/eqvae-so2-architecture-probe",
    "title": "eqvae so2 architecture probe",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
errors = [
    f"{key} must be {value!r}"
    for key, value in expected.items()
    if (
        str(data.get(key, "")).lower()
        if value in {"true", "false"}
        else str(data.get(key, ""))
    ) != value
]
for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
    if data.get(field) != []:
        errors.append(f"{field} must be empty")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2METADATA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_ARCHITECTURE_PROBE_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO(2) probe guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  for required_text in \
    "spec0013_so2_dual_t4_probe.json" \
    "spec0013.so2_dual_t4_final.v1" \
    "locked_so2_architecture_mechanics_final" \
    "padded_bmm_direct" \
    "compile_step_python_reducer_fp16_channels_last" \
    "e9e998fd161f0955959c64aed7cd7ddbdfcb55a271b9ce05805903c97c93efb8" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.so2_architecture_probe" \
    "graph_breaks,recompiles"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: SO(2) probe run.py missing required text: $required_text" >&2
      exit 1
    fi
  done

  python3 - "$run_file" <<'PYSO2PAYLOAD'
import base64
import io
import re
import sys
import zipfile
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""', text, re.DOTALL)
if match is None:
    raise SystemExit("error: SO(2) probe has no embedded payload")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group("payload")))) as archive:
    source = archive.read("src/eqvae/benchmarking/so2_architecture_probe.py").decode()
required = (
    "PER_DEVICE_BATCH: Final = 4",
    "SETTLED_UPDATES: Final = 32",
    "WARMUP_UPDATES: Final = 20",
    "TIMED_WINDOW_UPDATES: Final = 50",
    'RUNTIME_BUNDLE_ID: Final = "compile_step_python_reducer_fp16_channels_last"',
    "SO2LargestDDConv",
    "def _gradient_mean_check(",
    "def _check_buffers_across_ranks(",
)
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("\n".join(f"error: embedded SO(2) source missing {item}" for item in missing))
for forbidden in ("four_mm_three_cat", "four_mm_direct", "_follow_up_verdict"):
    if forbidden in source:
        raise SystemExit(f"error: embedded final SO(2) source retains {forbidden}")
PYSO2PAYLOAD
}

preflight_so2_architecture_probe() {
  build_embedded_kernel "$so2_architecture_probe_kernel_dir"
  guard_so2_architecture_probe_push_ready \
    "$so2_architecture_probe_kernel_dir" \
    "$(metadata_path "$so2_architecture_probe_kernel_dir")" \
    "local_preflight"
  echo "ok: Spec 0013 dual-T4 probe is built and locally guarded; no remote write performed"
}

guard_so2_runtime_readiness_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: SO(2) readiness must be one generated run.py" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" == "1" ]]; then
    echo "error: do not attach a dataset to SO(2) readiness" >&2
    exit 1
  fi
  python3 - "$metadata" <<'PYSO2READINESSMETADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "id": "maximusshtefan/eqvae-so2-runtime-readiness",
    "title": "eqvae so2 runtime readiness",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
errors = [
    f"{key} must be {value!r}"
    for key, value in expected.items()
    if (str(data.get(key, "")).lower() if value in {"true", "false"} else str(data.get(key, ""))) != value
]
for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
    if data.get(field) != []:
        errors.append(f"{field} must be empty")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2READINESSMETADATA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_RUNTIME_READINESS_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO(2) readiness guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  for required_text in \
    "spec0015_so2_runtime_readiness.json" \
    "spec0015_so2_gate_health.csv" \
    "spec0015.so2_selected_runtime_readiness.v1" \
    "compile_step_python_reducer_fp16_channels_last" \
    "generated_device_resident" \
    "GATE_ROW_COUNT = 68" \
    "torch.distributed.run" \
    "--nproc_per_node=2" \
    "eqvae.benchmarking.so2_runtime_readiness" \
    "graph_breaks,recompiles"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: SO(2) readiness run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

preflight_so2_runtime_readiness() {
  build_embedded_kernel "$so2_runtime_readiness_kernel_dir"
  guard_so2_runtime_readiness_push_ready \
    "$so2_runtime_readiness_kernel_dir" \
    "$(metadata_path "$so2_runtime_readiness_kernel_dir")" \
    "local_preflight"
  echo "ok: Spec 0015 SO(2) readiness is built and guarded; no remote write performed"
}

guard_so2_training_metadata() {
  local metadata="$1"
  local expected_id="$2"
  local resume_dataset_slug="${3:-}"
  python3 - "$metadata" "$expected_id" "$resume_dataset_slug" <<'PYSO2TRAINMETADATA'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_id = sys.argv[2]
resume_dataset_slug = sys.argv[3]
dataset_sources = ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
if resume_dataset_slug:
    dataset_sources.append(resume_dataset_slug)
expected = {
    "id": expected_id,
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": dataset_sources,
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
errors = [f"{key} must be {value!r}" for key, value in expected.items() if data.get(key) != value]
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYSO2TRAINMETADATA
}

guard_so2_prelaunch_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  guard_so2_training_metadata "$metadata" "maximusshtefan/eqvae-so2-prelaunch"
  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SO2_PRELAUNCH_READY = True" \
    --verify-only
}

guard_so2_full_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local mode="${3:-push}"
  guard_so2_training_metadata \
    "$metadata" \
    "maximusshtefan/eqvae-so2-selected-runtime-full" \
    "$so2_full_resume_dataset_slug"
  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SO2_SELECTED_RUNTIME_FULL_READY = True"
    --verify-only
  )
  if [[ "$mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  elif [[ "$mode" != "push" ]]; then
    echo "error: unsupported SO2 full guard mode: $mode" >&2
    exit 1
  fi
  build_kernel_py "${verify_args[@]}"
  if [[ "${KAGGLE_SO2_FULL_COST_CONFIRMED:-}" != "1" ]]; then
    echo "error: set KAGGLE_SO2_FULL_COST_CONFIRMED=1 after accepting measured prelaunch cost" >&2
    exit 1
  fi
  local verdict="runs/kaggle/so2_prelaunch/benchmark/so2_prelaunch_verdict.json"
  if [[ ! -f "$verdict" ]]; then
    echo "error: missing downloaded SO2 prelaunch verdict: $verdict" >&2
    exit 1
  fi
  PYTHONPATH=src .venv/bin/python - \
    "$verdict" \
    "$so2_full_session1_output_dir/embedded_payload" \
    "$so2_full_resume_authority_dir/embedded_payload" \
    "$so2_full_resume_dataset_dir" <<'PYSO2FULLVERDICT'
import hashlib
import json
import sys
from pathlib import Path
from eqvae.benchmarking.so2_prelaunch import (
    execution_identity,
    validate_prelaunch_artifacts,
)
from eqvae.checkpointing import read_training_checkpoint_metadata
from eqvae.config import resolve_json_config

EXPECTED_PRELAUNCH_COMMIT = "4aaf614f2cdbf1bc628e13858eb6c4e08300266b"
EXPECTED_RESUME_COMMIT = "d251175609c7ecfcac8d34d88556828f16e72386"
EXPECTED_DATASET_SLUG = "maximusshtefan/eqvae-so2-session3-step27000"
EXPECTED_CHECKPOINT_SHA256 = (
    "7adfea7850ee7ab620f0363ca4a8fe9e41fd67160feeaeae1f07ff291a0bf6ba"
)
EXPECTED_CONTINUATION_WRAPPER_SHA256 = (
    "cac998a6497cdb74e7092dc3919e89ee170841752ea35173dbd734928154bdd8"
)
EXPECTED_STEP = 27000
ALLOWED_CONTINUATION_CHANGES = {
    "kaggle/kernels/so2_selected_runtime_full/kernel-metadata.json",
    "kaggle/kernels/so2_selected_runtime_full/run_template.py",
}

verdict = Path(sys.argv[1])
prelaunch_authority = Path(sys.argv[2])
resume_authority = Path(sys.argv[3])
dataset_dir = Path(sys.argv[4])
repo = Path.cwd()

blockers = list(
    validate_prelaunch_artifacts(
        verdict.parents[1],
        repo_root=prelaunch_authority,
        expected_source_commit=EXPECTED_PRELAUNCH_COMMIT,
    ),
)
resume_manifest = json.loads(
    (resume_authority / "payload_manifest.json").read_text(encoding="utf-8")
)
if (
    resume_manifest.get("git_commit") != EXPECTED_RESUME_COMMIT
    or resume_manifest.get("git_dirty") is not False
):
    blockers.append("so2_continuation_resume_authority_commit_mismatch")
authority_identity = execution_identity(resume_authority)
current_identity = execution_identity(repo)
continuation_wrapper = (
    repo / "kaggle/kernels/so2_selected_runtime_full/run_template.py"
)
if hashlib.sha256(continuation_wrapper.read_bytes()).hexdigest() != (
    EXPECTED_CONTINUATION_WRAPPER_SHA256
):
    blockers.append("so2_continuation_wrapper_sha256_mismatch")
for name, expected in authority_identity.items():
    if (
        name not in ALLOWED_CONTINUATION_CHANGES
        and current_identity.get(name) != expected
    ):
        blockers.append(f"so2_continuation_execution_core_changed:{name}")

expected_files = {"dataset-metadata.json", "step_027000.pt"}
observed_files = (
    {path.name for path in dataset_dir.iterdir()}
    if dataset_dir.is_dir()
    else set()
)
if observed_files != expected_files:
    blockers.append("so2_continuation_dataset_files_mismatch")
checkpoint = dataset_dir / "step_027000.pt"
if checkpoint.is_file():
    observed_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if observed_sha256 != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("so2_continuation_checkpoint_sha256_mismatch")
    metadata = read_training_checkpoint_metadata(path=checkpoint)
    if (
        metadata.optimizer_step != EXPECTED_STEP
        or metadata.successful_optimizer_update_count != EXPECTED_STEP
    ):
        blockers.append("so2_continuation_checkpoint_step_mismatch")
    runtime = repo / "configs/spec0001/non_eq_vae_selected_runtime.json"
    runtime_sha256 = hashlib.sha256(runtime.read_bytes()).hexdigest()
    effective_sha256 = resolve_json_config(
        repo / "configs/spec0016/so2_selected_runtime_full.json",
    ).effective_config_hash
    if metadata.runtime_config_sha256 != runtime_sha256:
        blockers.append("so2_continuation_checkpoint_runtime_mismatch")
    if metadata.effective_config_sha256 != effective_sha256:
        blockers.append("so2_continuation_checkpoint_config_mismatch")
else:
    blockers.append("so2_continuation_checkpoint_missing")

dataset_metadata = dataset_dir / "dataset-metadata.json"
if dataset_metadata.is_file():
    payload = json.loads(dataset_metadata.read_text(encoding="utf-8"))
    if (
        payload.get("id") != EXPECTED_DATASET_SLUG
        or EXPECTED_CHECKPOINT_SHA256 not in payload.get("description", "")
    ):
        blockers.append("so2_continuation_dataset_metadata_mismatch")
else:
    blockers.append("so2_continuation_dataset_metadata_missing")

if blockers:
    raise SystemExit("\n".join(f"error: {blocker}" for blocker in blockers))
PYSO2FULLVERDICT
}

preflight_so2_prelaunch() {
  build_embedded_kernel "$so2_prelaunch_kernel_dir"
  guard_so2_training_metadata \
    "$(metadata_path "$so2_prelaunch_kernel_dir")" \
    "maximusshtefan/eqvae-so2-prelaunch"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q \
    tests/test_so2_prelaunch.py tests/test_so2_full_run.py
}

preflight_so2_full() {
  build_embedded_kernel "$so2_full_kernel_dir"
  KAGGLE_SO2_FULL_COST_CONFIRMED=1 guard_so2_full_push_ready \
    "$so2_full_kernel_dir" \
    "$(metadata_path "$so2_full_kernel_dir")" \
    "local_preflight"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q \
    tests/test_so2_prelaunch.py tests/test_so2_full_run.py
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
    "enable_internet": "true",
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

  build_kernel_py \
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
    "enable_internet": "true",
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

  build_kernel_py \
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
        "runs/kaggle/runtime_selection_v5/benchmark/runtime_proof.json",
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

guard_selected_runtime_lr_range_push_ready() {
  local kernel_dir="$1"
  local _metadata="$2"
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: LR-range kernel must be one generated run.py, not a sibling payload" >&2
    exit 1
  fi
  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime LR-range run.
EOF
    exit 1
  fi
  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin" >&2
    exit 1
  fi

  PYTHONPATH=src "$python_bin" - <<'PYLRCONFIG'
from pathlib import Path

from eqvae.config import resolve_json_config
from eqvae.training.selected_runtime import parse_selected_runtime_plan

plan = parse_selected_runtime_plan(
    Path("configs/spec0001/non_eq_vae_selected_runtime.json"),
)
config = resolve_json_config(
    Path("configs/spec0001/non_eq_vae_selected_runtime_lr_range.json"),
).effective_config
sweep = config.get("learning_rate_range")
training = config.get("training")
errors = []
if plan.per_device_batch_size != 25 or plan.global_batch_size != 50:
    errors.append("selected runtime must be the measured bs25/global50 winner")
if not isinstance(sweep, dict) or sweep.get("start") != 0.00002 \
        or sweep.get("end") != 0.003 or sweep.get("successful_updates") != 192:
    errors.append("LR range must be the bounded 2e-5..3e-3, 192-update sweep")
if not isinstance(training, dict) or training.get("max_train_steps") != 192:
    errors.append("LR range training.max_train_steps must be 192")
if errors:
    raise SystemExit("\n".join(f"error: {error}" for error in errors))
PYLRCONFIG

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_LR_RANGE_READY = True" \
    --verify-only \
    --allow-dirty
}

guard_selected_runtime_debug_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: selected-runtime debug gate must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOF'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime debug/tiny gate.
EOF
    exit 1
  fi

  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/0001-translatable-normal-vae-baseline.md"; then
    echo "error: spec 0001 does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/0003-kaggle-cli-execution-workflow.md"; then
    echo "error: spec 0003 does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/kaggle_cli_workflow.md"; then
    echo "error: Kaggle workflow doc does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_debug_gate_contract_ready' \
    "docs/specs/README.md"; then
    echo "error: specs index does not describe the selected-runtime debug gate contract" >&2
    exit 1
  fi

  local python_bin="${PYTHON:-.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
  fi
  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-push-ready \
    --selector-generation-mode remote_generate \
    --debug-config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
    --tiny-config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json \
    --fixed-train-patches configs/spec0001/fixed_32_train_overfit_patches.json
  preflight_fixed32_selector_readiness

  python3 - "$metadata" <<'PY'
import json
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []

required_values = {
    "id": "maximusshtefan/eqvae-selected-runtime-debug",
    "title": "eqvae selected runtime debug",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
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
        errors.append(f"{source_field} must be an empty list for selected-runtime debug")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

  build_kernel_py \
    --kernel-dir "$kernel_dir" \
    --ready-marker "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
    --verify-only \
    --allow-dirty

  local run_file="$kernel_dir/run.py"
  PYTHONPATH=src "$python_bin" - "$run_file" <<'PYDEBUGPAYLOAD'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from eqvae.training.selected_runtime import selected_runtime_plan_errors

run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""',
    run_text,
    flags=re.DOTALL,
)
if match is None:
    print("error: selected-runtime debug run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)

payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_run_text = (
        "selector_generation.get(\"status\") == \"pass\"",
        "_generate_remote_fixed32_selector(",
        "_fixed32_selector_status_from_payload_cwd(",
        "fixed32_selector_status(selector_path, data_root=data_root)",
        "_run_real_selected_runtime_debug(",
        "_run_real_selected_runtime_tiny_overfit(",
        "_write_real_gate_summary(",
        "_run_selected_runtime_train_torchrun(",
        "_selected_runtime_train_torchrun_command(",
        "\"torch.distributed.run\"",
        "\"--standalone\"",
        "\"--nproc_per_node=2\"",
        "\"eqvae.cli.selected_runtime_train\"",
        "\"--fixed-train-patches\"",
        "DEBUG_RESUME_STEP = 4",
        "DEBUG_FINAL_STEP = 8",
        "TINY_MAX_STEP = 128",
        "\"--resume\"",
        "step_{DEBUG_RESUME_STEP:06d}.pt",
        "_validate_real_runner_artifacts(output_dir=output_dir)",
    )
    for text in required_run_text:
        if text not in run_text:
            errors.append(f"selected-runtime debug run.py missing required source text: {text}")
    required_files = {
        "src/eqvae/benchmarking/fixed32_selector_readiness.py",
        "src/eqvae/benchmarking/selected_runtime_gate.py",
        "src/eqvae/cli/fixed32_selector_readiness.py",
        "src/eqvae/cli/select_fixed_patches.py",
        "src/eqvae/cli/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_train.py",
        "src/eqvae/cli/train.py",
        "src/eqvae/training/debug.py",
        "src/eqvae/training/selected_runtime_runner.py",
        "configs/spec0001/non_eq_vae_selected_runtime_debug.json",
        "configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json",
        "configs/spec0001/fixed_32_train_overfit_patches.json",
        "configs/spec0001/non_eq_vae_selected_runtime.json",
        "configs/spec0001/non_eq_vae_runtime_winner.json",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    try:
        gate_source = archive.read(
            "src/eqvae/benchmarking/selected_runtime_gate.py",
        ).decode("utf-8")
        runner_source = archive.read(
            "src/eqvae/training/selected_runtime_runner.py",
        ).decode("utf-8")
        debug_config = json.loads(
            archive.read("configs/spec0001/non_eq_vae_selected_runtime_debug.json"),
        )
        tiny_config = json.loads(
            archive.read("configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json"),
        )
        fixed_selector = json.loads(
            archive.read("configs/spec0001/fixed_32_train_overfit_patches.json"),
        )
        selected_runtime = json.loads(
            archive.read("configs/spec0001/non_eq_vae_selected_runtime.json"),
        )
    except KeyError as error:
        print(f"error: selected-runtime debug payload missing {error}", file=sys.stderr)
        raise SystemExit(1) from None

    # Spec 0011 S17b-3: delegate the selected-runtime identity/recipe/snapshot/batch
    # validation to the single-source parser instead of mirroring its frozen eager
    # literals here. This accepts a re-measured compiled winner (amp-off whole-step
    # compile, any exact batch) while keeping every hardware/topology anchor -- the
    # parser pins accelerator/machine_shape/world_size/nproc/grad-accum even more
    # explicitly than the old identity literal did (which carried them only
    # incidentally). selected_runtime_path is None: the launch parse re-checks the
    # runtime proof; this push guard only needs the proof file present (required_files
    # above). Byte-identical acceptance on the committed v5 plan.
    if not isinstance(selected_runtime, dict):
        errors.append("selected runtime must be a JSON object")
    else:
        errors.extend(
            selected_runtime_plan_errors(
                selected_runtime,
                selected_runtime_path=None,
            ),
        )

    debug_gate = debug_config.get("selected_runtime_debug")
    tiny_gate = tiny_config.get("selected_runtime_debug_gate")
    for name, gate in (
        ("selected_runtime_debug", debug_gate),
        ("selected_runtime_debug_gate", tiny_gate),
    ):
        if not isinstance(gate, dict):
            errors.append(f"{name} must be an object")
            continue
        expected_gate_values = {
            "remote_pass_ready": False,
            "real_train_runner_implemented": True,
            "selector_generation_mode": "remote_generate",
            "remote_selector_generation_ready": True,
            "fixed_32_selector_real": False,
        }
        for key, expected in expected_gate_values.items():
            if gate.get(key) != expected:
                errors.append(f"{name}.{key} must be {expected!r} before remote push")

    stale_blockers = (
        "real_ubc_selected_runtime_train_runner_not_implemented",
        "selected_runtime_debug_wrapper_not_wired_to_real_runner_until_spec0008",
    )
    for stale in stale_blockers:
        if stale in gate_source:
            errors.append(f"selected-runtime debug gate contains stale blocker {stale}")
    if "Only data='synthetic' is implemented" in runner_source:
        errors.append("train runner still rejects data='ubc-pre-shuffled'")
    required_runner_text = (
        "_loaded_checkpoint_resume_proof",
        "loaded_successful_optimizer_update_count",
        "additional_optimizer_steps",
        "ubc-pre-shuffled",
    )
    for text in required_runner_text:
        if text not in runner_source:
            errors.append(f"selected-runtime runner missing required source text: {text}")
    if fixed_selector.get("status") != "requires_real_data_generation":
        errors.append("embedded fixed_32 selector should remain a remote-generate placeholder")
    selectors = fixed_selector.get("selectors")
    if not isinstance(selectors, list) or selectors:
        errors.append("embedded fixed_32 placeholder must not contain local selectors")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PYDEBUGPAYLOAD

  for required_text in \
    "KAGGLE_SELECTED_RUNTIME_DEBUG_READY = True" \
    "selected_runtime_debug_gate_contract_ready" \
    "remote_generate" \
    "select_fixed_patches" \
    "fixed_32_train_overfit" \
    "fixed32_selector_readiness" \
    "selected_runtime_gate" \
    "selected_runtime_train" \
    "selected_runtime_gate_summary.json" \
    "selected_runtime_debug_summary.json" \
    "selected_runtime_plan_applied.json" \
    "local_selected_runtime_readiness.json" \
    "checkpoint_resume_proof.json" \
    "tiny_overfit_summary.json" \
    "artifact_manifest.json" \
    "gate_health_summary.json" \
    "selected_runtime.json" \
    "single_visible_t4" \
    "dual_t4_ddp" \
    "torchrun" \
    "--nproc_per_node=2" \
    "wrong_accelerator"; do
    if ! grep -q -- "$required_text" "$run_file"; then
      echo "error: selected-runtime debug run.py missing required text: $required_text" >&2
      exit 1
    fi
  done
}

guard_selected_runtime_full_push_ready() {
  local kernel_dir="$1"
  local metadata="$2"
  local guard_mode="${3:-push}"

  case "$guard_mode" in
    push|local_preflight)
      ;;
    *)
      echo "error: unknown selected-runtime full guard mode: $guard_mode" >&2
      exit 1
      ;;
  esac

  if [[ "$guard_mode" != "local_preflight" ]] && \
    [[ "${EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY:-}" == "1" ]]; then
    cat >&2 <<'EOFGUARD'
error: EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY is only valid
inside preflight-selected-runtime-full; unset it before a real push guard.
EOFGUARD
    exit 1
  fi

  if [[ -d "$kernel_dir/payload" ]]; then
    echo "error: selected-runtime full run must be a single generated run.py, not a sibling payload" >&2
    exit 1
  fi

  if [[ "${KAGGLE_FULL_DATASET_CONFIRMED:-}" != "1" ]]; then
    cat >&2 <<'EOFGUARD'
error: set KAGGLE_FULL_DATASET_CONFIRMED=1 only after accepting the real
patch dataset attachment/setup cost for the selected-runtime full training run.
EOFGUARD
    exit 1
  fi

  if ! grep -q 'selected_runtime_full_run_contract_ready' \
    "configs/spec0001/non_eq_vae_selected_runtime_full.json"; then
    echo "error: full config does not carry the selected-runtime full contract token" >&2
    exit 1
  fi
  if ! grep -q 'selected_runtime_full_run_contract_ready' \
    "$kernel_dir/run_template.py"; then
    echo "error: full kernel template does not carry the selected-runtime full contract token" >&2
    exit 1
  fi
  if ! grep -q '0009-first-full-selected-runtime-training-run.md' \
    "docs/specs/README.md"; then
    echo "error: specs index does not list spec 0009" >&2
    exit 1
  fi

  local python_bin="${PYTHON:-.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="python3"
  fi
  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-output \
    --output-dir runs/kaggle/selected_runtime_debug \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json

  python3 - "$metadata" <<'PYFULLMETA'
import json
import sys
from pathlib import Path
metadata = Path(sys.argv[1])
data = json.loads(metadata.read_text(encoding="utf-8"))
errors: list[str] = []
required = {
    "id": "maximusshtefan/eqvae-selected-runtime-full",
    "title": "eqvae selected runtime full",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
}
for key, expected in required.items():
    actual = str(data.get(key, ""))
    comparable = actual.lower() if expected in {"true", "false"} else actual
    if comparable != expected:
        errors.append(f"{key} must be {expected!r}")
expected_datasets = [
    "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    "maximusshtefan/eqvae-baseline-session2-step45000",
]
if data.get("dataset_sources") != expected_datasets:
    errors.append("dataset_sources must attach the exact UBC and session-2 datasets")
for source_field in ("competition_sources", "kernel_sources", "model_sources"):
    if data.get(source_field) != []:
        errors.append(f"{source_field} must be empty")
if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PYFULLMETA

  local verify_args=(
    --kernel-dir "$kernel_dir"
    --ready-marker "KAGGLE_SELECTED_RUNTIME_FULL_READY = True"
    --verify-only
  )
  if [[ "$guard_mode" == "local_preflight" ]]; then
    verify_args+=(--allow-dirty)
  fi
  build_kernel_py "${verify_args[@]}"

  local run_file="$kernel_dir/run.py"
  PYTHONPATH=src "$python_bin" - "$run_file" <<'PYFULLPAYLOAD'
import base64
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from eqvae.benchmarking.schedule import training_steps_per_epoch
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT


def _positive_int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


run_text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'EMBEDDED_PAYLOAD_B64 = """\n(?P<payload>.*?)\n"""', run_text, flags=re.DOTALL)
if match is None:
    print("error: selected-runtime full run.py has no embedded payload", file=sys.stderr)
    raise SystemExit(1)
payload = base64.b64decode(match.group("payload").encode("ascii"))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = set(archive.namelist())
    errors: list[str] = []
    required_files = {
        "src/eqvae/benchmarking/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_gate.py",
        "src/eqvae/cli/selected_runtime_train.py",
        "src/eqvae/training/selected_runtime.py",
        "src/eqvae/training/selected_runtime_runner.py",
        "configs/spec0001/non_eq_vae_selected_runtime_full.json",
        "configs/spec0001/non_eq_vae_selected_runtime.json",
        "configs/spec0001/non_eq_vae_runtime_winner.json",
    }
    missing = sorted(required_files - names)
    if missing:
        errors.append(f"embedded payload missing required files: {missing!r}")
    full_config = json.loads(archive.read("configs/spec0001/non_eq_vae_selected_runtime_full.json"))
    selected_runtime = json.loads(archive.read("configs/spec0001/non_eq_vae_selected_runtime.json"))
    training = full_config.get("training") if isinstance(full_config, dict) else None
    objective = full_config.get("objective") if isinstance(full_config, dict) else None
    beta = objective.get("beta") if isinstance(objective, dict) else None
    if not isinstance(beta, dict) or beta.get("target") != 0.01:
        errors.append("full config objective.beta.target must be locked to 0.01")
    # Spec 0011 S8: derive the schedule from the plan's measured global batch and the
    # single-sourced patch count instead of pinning the reference literals. At the
    # reference global batch 24 these reproduce 12500/125000/6250 exactly, so the built
    # kernel is unchanged; a re-measured non-24 plan is validated by relationship.
    per_device_batch = _positive_int_or_none(selected_runtime.get("per_device_batch_size"))
    world_size = _positive_int_or_none(selected_runtime.get("world_size"))
    global_batch = _positive_int_or_none(selected_runtime.get("global_batch_size"))
    epochs = _int_or_none(training.get("epochs")) if isinstance(training, dict) else None
    if isinstance(training, dict) and "epochs" in training and epochs is None:
        # A present-but-non-int epochs (e.g. JSON 10.0) must fail closed here: it would
        # otherwise pass the ``!= 10`` anchor pin yet null the derivation, silently
        # skipping the FULL_TARGET_UPDATES/FULL_HALF_EPOCH_INTERVAL run.py token check.
        errors.append("full config training.epochs must be an integer")
    if global_batch is None:
        errors.append("selected runtime global_batch_size must be a positive integer")
        derived_updates = None
    else:
        derived_updates = training_steps_per_epoch(
            real_train_patch_count=REAL_TRAIN_PATCH_COUNT,
            global_batch_size=global_batch,
        )
    if global_batch is not None and (
        per_device_batch is None
        or world_size is None
        or global_batch != per_device_batch * world_size
    ):
        errors.append(
            "selected runtime global_batch_size must equal "
            "per_device_batch_size * world_size"
        )
    if derived_updates is None or epochs is None:
        derived_target = None
        derived_half = None
    else:
        derived_target = epochs * derived_updates
        derived_half = derived_updates // 2
    if not isinstance(training, dict):
        errors.append("full config training must be an object")
    else:
        expected_training = {
            "epochs": 10,
            "train_reparameterization": "stochastic_seeded",
            "checkpoint_retention": "best_final_latest_four_interval",
            "resume_supported": True,
        }
        for key, expected in expected_training.items():
            if training.get(key) != expected:
                errors.append(f"full config training.{key} must be {expected!r}")
        for key in (
            "optimizer_updates_per_epoch",
            "max_train_steps",
            "half_epoch_interval_steps",
            "save_every_steps",
        ):
            if key in training:
                errors.append(
                    f"full config training must not re-freeze {key}; "
                    "schedule is runner-derived (Spec 0011)"
                )
    recorded_updates = selected_runtime.get("optimizer_updates_per_epoch")
    if derived_updates is not None and (
        not isinstance(recorded_updates, int)
        or isinstance(recorded_updates, bool)
        or recorded_updates != derived_updates
    ):
        errors.append(
            f"selected runtime optimizer_updates_per_epoch must be {derived_updates!r}"
        )
    forbidden = (
        "selected_runtime_debug",
        "DEBUG_FINAL_STEP",
        "TINY_MAX_STEP",
        "non_eq_vae_selected_runtime_debug.json",
    )
    required_text = (
        "KAGGLE_SELECTED_RUNTIME_FULL_READY = True",
        "selected_runtime_full_run_contract_ready",
        "non_eq_vae_selected_runtime_full.json",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "eqvae.cli.selected_runtime_train",
        "--resume",
        "maximusshtefan/eqvae-baseline-session2-step45000",
        "/kaggle/input/eqvae-baseline-session2-step45000/step_045000.pt",
        "e7a0f05e013bff4f7a5bfbfd4442f3c9a6d19cf261c42f54a6d04391be76e88b",
        "dual_t4_ddp",
    )
    for token in required_text:
        if token not in run_text:
            errors.append(f"full run.py missing required text: {token}")
    if derived_target is not None and derived_half is not None:
        for token in (
            f"FULL_TARGET_UPDATES = {derived_target}",
            f"FULL_HALF_EPOCH_INTERVAL = {derived_half}",
        ):
            if token not in run_text:
                errors.append(f"full run.py missing required text: {token}")
    forbidden_command_token = '"--max-train-steps"'
    command_builder = "_selected_runtime_full_torchrun_command"
    command_block_match = re.search(
        rf"def {command_builder}\(.*?(?=\ndef _)",
        run_text,
        flags=re.DOTALL,
    )
    if command_block_match is None:
        errors.append("full run.py missing selected-runtime full torchrun builder")
    else:
        command_block = command_block_match.group(0)
        for token in forbidden:
            if token in command_block:
                errors.append(f"full torchrun command contains debug/tiny token: {token}")
        if forbidden_command_token in command_block:
            errors.append("full torchrun command must not contain --max-train-steps")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
PYFULLPAYLOAD
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
  local kernel_dir="${1:-$default_kernel_dir}"
  require_remote_confirmed
  require_kaggle_cli
  local kernel_id
  kernel_id="$(kernel_id_from_metadata "$kernel_dir")"

  echo "Kaggle API read-only preflight"
  echo "=============================="
  kaggle --version
  kaggle_auth_path_message

  kaggle_api kernels list --mine --search "${kernel_id#*/}" --csv >/dev/null
  echo "ok: kernels list can see $kernel_id"

  kaggle_api kernels status "$kernel_id" >/dev/null
  echo "ok: kernels status works for $kernel_id"

  kaggle_api kernels logs "$kernel_id" >/dev/null
  echo "ok: kernels logs works for $kernel_id"

  kaggle_api datasets files maximusshtefan/patches-pre-shuffled-ubc-ocean -v >/dev/null
  echo "ok: dataset file listing works for patches-pre-shuffled-ubc-ocean"

  if kaggle_api quota -v >/dev/null 2>&1; then
    echo "ok: accelerator quota endpoint works"
  else
    echo "warn: accelerator quota endpoint failed; verify quota in Kaggle UI before remote benchmark push" >&2
  fi

  if kaggle_api kernels files "$kernel_id" -v >/dev/null 2>&1; then
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

preflight_fixed25_selector() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  build_embedded_kernel "$fixed25_selector_kernel_dir"
  validate_kernel_dir "$fixed25_selector_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_fixed_selectors.py \
    tests/test_fixed25_equivariance_artifacts.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_fixed25_selector_kernel_import_simulation \
    -q
}

preflight_fixed32_selector_readiness() {
  local python_bin="${PYTHON:-.venv/bin/python}"
  local synthetic_root="/tmp/eqvae-fixed32-synthetic-root"
  local output_dir="$synthetic_root/readiness"
  local selector_output="$synthetic_root/fixed_32_train_overfit_patches.json"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_fixed_selectors.py \
    tests/test_fixed32_selector_readiness.py \
    -q

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.fixed32_selector_readiness \
    --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --synthetic-root "$synthetic_root" \
    --output-dir "$output_dir" \
    --masked-holdout-csv docs/data/ubc_ocean_masked_holdout_ids.csv

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.select_fixed_patches \
    --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
    --kind fixed_32_train_overfit \
    --data-root "$synthetic_root" \
    --masked-holdout-csv docs/data/ubc_ocean_masked_holdout_ids.csv \
    --output "$selector_output" \
    --validate-crc

  "$python_bin" - "$output_dir" "$selector_output" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
selector_output = Path(sys.argv[2])
readiness_path = output_dir / "benchmark" / "fixed32_selector_readiness.json"
errors = []
if not selector_output.exists():
    errors.append(f"selector output missing: {selector_output}")
if not readiness_path.exists():
    errors.append(f"readiness artifact missing: {readiness_path}")
else:
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    synthetic_status = readiness.get("synthetic_selector_status")
    if not isinstance(synthetic_status, dict):
        errors.append("synthetic_selector_status must be an object")
        synthetic_status = {}
    expected = {
        "status": "pass",
        "selector_generation_mode": "remote_generate",
        "remote_selector_generation_ready": True,
        "fixed_32_selector_real": False,
        "synthetic_selector_deterministic": True,
        "synthetic_selector_canonical_real_rejected": True,
    }
    for key, value in expected.items():
        if readiness.get(key) != value:
            errors.append(f"{key} must be {value!r}")
    if synthetic_status.get("failure_kind") != "fixed_32_selector_not_canonical_real_ubc":
        errors.append("synthetic selector must fail canonical-real readiness")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

preflight_selected_runtime_debug() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  preflight_fixed32_selector_readiness
  build_embedded_kernel "$selected_runtime_debug_kernel_dir"
  validate_kernel_dir "$selected_runtime_debug_kernel_dir"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_gate.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_debug_kernel_import_simulation \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_debug_kernel_full_local_fail_closed_simulation \
    -q
}

preflight_selected_runtime_lr_range() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi
  build_embedded_kernel "$selected_runtime_lr_range_kernel_dir"
  validate_kernel_dir "$selected_runtime_lr_range_kernel_dir"
  KAGGLE_FULL_DATASET_CONFIRMED=1 guard_selected_runtime_lr_range_push_ready \
    "$selected_runtime_lr_range_kernel_dir" \
    "$selected_runtime_lr_range_kernel_dir/kernel-metadata.json"
  "$python_bin" -m pytest -q tests/test_selected_runtime_full_run.py \
    -k 'spec0011_checked_in_winner_plan or spec0011_winner_plan_rejects or lr_range'
}

preflight_selected_runtime_full() {
  local python_bin="${PYTHON:-.venv/bin/python}"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi

  PYTHONPATH=src "$python_bin" -m eqvae.cli.selected_runtime_gate \
    --verify-output \
    --output-dir runs/kaggle/selected_runtime_debug \
    --runtime-config configs/spec0001/non_eq_vae_selected_runtime.json

  build_embedded_kernel "$selected_runtime_full_kernel_dir"
  validate_kernel_dir "$selected_runtime_full_kernel_dir"
  EQVAE_SELECTED_RUNTIME_FULL_LOCAL_PREFLIGHT_ALLOW_DIRTY=1 \
    KAGGLE_FULL_DATASET_CONFIRMED=1 guard_selected_runtime_full_push_ready \
    "$selected_runtime_full_kernel_dir" \
    "$selected_runtime_full_kernel_dir/kernel-metadata.json" \
    "local_preflight"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_full_run.py \
    tests/test_kaggle_embedded_kernel.py::test_embedded_selected_runtime_full_kernel_import_simulation \
    -q
}


preflight_selected_runtime_runner() {
  local python_bin="${PYTHON:-.venv/bin/python}"
  local output_dir="$TMPDIR/selected_runtime_runner_preflight_$$"

  if [[ ! -x "$python_bin" ]]; then
    echo "error: missing executable $python_bin; run repo setup before preflight" >&2
    exit 1
  fi
  rm -rf "$output_dir"

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m pytest \
    tests/test_selected_runtime_gate.py \
    tests/test_train_cli.py \
    tests/test_selected_runtime_runner.py \
    -q

  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$python_bin" -m eqvae.cli.selected_runtime_train \
    --config configs/spec0001/non_eq_vae_selected_runtime_debug.json \
    --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json \
    --data synthetic \
    --output-dir "$output_dir" \
    --run-name spec0007_local_runner_dryrun \
    --max-train-steps 2 \
    --max-val-steps 1 \
    --dry-run

  "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
benchmark = output_dir / "benchmark"
metrics = output_dir / "metrics"
required = [
    benchmark / "training_summary.json",
    benchmark / "selected_runtime_debug_summary.json",
    benchmark / "selected_runtime_plan_applied.json",
    benchmark / "checkpoint_resume_proof.json",
    benchmark / "gate_health_summary.json",
    benchmark / "artifact_manifest.json",
    metrics / "train_steps.csv",
    metrics / "gate_health.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print(f"error: selected-runtime runner preflight missing artifacts: {missing}", file=sys.stderr)
    raise SystemExit(1)

summary = json.loads((benchmark / "training_summary.json").read_text(encoding="utf-8"))
debug = json.loads((benchmark / "selected_runtime_debug_summary.json").read_text(encoding="utf-8"))
plan = json.loads((benchmark / "selected_runtime_plan_applied.json").read_text(encoding="utf-8"))
manifest = json.loads((benchmark / "artifact_manifest.json").read_text(encoding="utf-8"))
readiness = json.loads((benchmark / "local_selected_runtime_readiness.json").read_text(encoding="utf-8"))

errors = []
if summary.get("full_run_eligible") is not False:
    errors.append("training summary must remain non-promotable")
if debug.get("real_train_runner_implemented") is not True:
    errors.append("runner readiness must prove real_train_runner_implemented=true")
if debug.get("remote_pass_ready") is not False:
    errors.append("runner dry-run must not claim remote_pass_ready")
if plan.get("status") != "fail" or plan.get("plan_applied") is not False:
    errors.append("local dry-run must fail full dual-T4/AMP plan application")
if readiness.get("remote_pass_ready") is not False:
    errors.append("local readiness must keep remote_pass_ready=false")
if manifest.get("status") != "local_pass":
    errors.append("artifact manifest must pass locally")
if (benchmark / "selected_runtime.json").exists():
    errors.append("runner preflight must not write benchmark/selected_runtime.json")

if errors:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
  rm -rf "$output_dir"
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
    api_check "${2:-$default_kernel_dir}"
    ;;
  preflight-runtime-selection)
    preflight_runtime_selection
    ;;
  preflight-fixed32-selector-readiness)
    preflight_fixed32_selector_readiness
    ;;
  preflight-selected-runtime-runner)
    preflight_selected_runtime_runner
    ;;
  preflight-selected-runtime-debug)
    preflight_selected_runtime_debug
    ;;
  preflight-selected-runtime-lr-range)
    preflight_selected_runtime_lr_range
    ;;
  preflight-selected-runtime-full)
    preflight_selected_runtime_full
    ;;
  preflight-fixed25-selector)
    preflight_fixed25_selector
    ;;
  preflight-so2-architecture-probe)
    preflight_so2_architecture_probe
    ;;
  preflight-so2-runtime-readiness)
    preflight_so2_runtime_readiness
    ;;
  preflight-so2-prelaunch)
    preflight_so2_prelaunch
    ;;
  preflight-so2-selected-runtime-full)
    preflight_so2_full
    ;;
  push)
    # Only treat the first token as kernel_dir when it is a real path, not an option
    # flag, so `push --wait ...` still falls back to the default kernel_dir instead of
    # swallowing `--wait` as the directory.
    if [[ -n "${2:-}" && "$2" != --* ]]; then
      kernel_dir="$2"
      shift 2
    else
      kernel_dir="$default_kernel_dir"
      shift 1
    fi
    # --wait blocks after a successful push until the kernel settles, so a caller
    # can push-and-be-woken in a single backgrounded command. --wait-interval /
    # --wait-max / --wait-queued tune the RUNNING cadence, the RUNNING backstop,
    # and the QUEUED budget; all wait flags are consumed here and never forwarded
    # to the Kaggle CLI. Everything else passes through unchanged.
    push_wait=0
    push_wait_interval=300
    push_wait_max=180
    push_wait_queued=300
    push_passthrough=()
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
      --wait)
        push_wait=1
        ;;
      --wait-interval)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-interval requires a value" >&2
          exit 1
        fi
        push_wait_interval="$2"
        shift
        ;;
      --wait-max)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-max requires a value" >&2
          exit 1
        fi
        push_wait_max="$2"
        shift
        ;;
      --wait-queued)
        if [[ "$#" -lt 2 ]]; then
          echo "error: --wait-queued requires a value" >&2
          exit 1
        fi
        push_wait_queued="$2"
        shift
        ;;
      *)
        push_passthrough+=("$1")
        ;;
      esac
      shift
    done
    if [[ "${KAGGLE_PUSH_CONFIRMED:-}" != "1" ]]; then
      echo "error: set KAGGLE_PUSH_CONFIRMED=1 after explicit user permission" >&2
      exit 1
    fi
    if [[ "$push_wait" == "1" ]]; then
      require_remote_confirmed
    fi
    validate_kernel_dir "$kernel_dir"
    guard_push_ready "$kernel_dir"
    require_kaggle_sources_confirmed "$(metadata_path "$kernel_dir")"
    require_kaggle_cli
    kaggle_api kernels push -p "$kernel_dir" \
      "${push_passthrough[@]+"${push_passthrough[@]}"}"
    if [[ "$push_wait" == "1" ]]; then
      push_kernel_id="$(kernel_id_from_metadata "$kernel_dir")"
      echo "push: waiting for ${push_kernel_id} to settle..."
      wait_kernel_until_settled \
        "$push_kernel_id" "$push_wait_interval" "$push_wait_max" \
        "$push_wait_queued"
    fi
    ;;
  status)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-debug)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_debug_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-lr-range)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_lr_range_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-prelaunch)
    kernel_id="$(kernel_id_from_metadata "$so2_prelaunch_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  status-so2-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$so2_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    kaggle_api kernels status "$kernel_id"
    ;;
  wait)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${3:-300}" "${4:-180}" "${5:-300}"
    ;;
  wait-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-180}" "${4:-300}"
    ;;
  wait-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-480}" "${4:-600}"
    ;;
  wait-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-36}" "${4:-600}"
    ;;
  wait-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    require_remote_confirmed
    require_kaggle_cli
    wait_kernel_until_settled "$kernel_id" "${2:-300}" "${3:-36}" "${4:-600}"
    ;;
  output)
    kernel_id="${2:-$(kernel_id_from_metadata "$default_kernel_dir")}"
    output_dir="${3:-$default_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-real-data-runtime-pretest)
    kernel_id="$(kernel_id_from_metadata "$real_data_runtime_pretest_kernel_dir")"
    output_dir="${2:-$real_data_runtime_pretest_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-runtime-selection)
    kernel_id="$(kernel_id_from_metadata "$runtime_selection_kernel_dir")"
    output_dir="${2:-$runtime_selection_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-debug)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_debug_kernel_dir")"
    output_dir="${2:-$selected_runtime_debug_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-lr-range)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_lr_range_kernel_dir")"
    output_dir="${2:-$selected_runtime_lr_range_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$selected_runtime_full_kernel_dir")"
    output_dir="${2:-$selected_runtime_full_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-fixed25-selector)
    kernel_id="$(kernel_id_from_metadata "$fixed25_selector_kernel_dir")"
    output_dir="${2:-$fixed25_selector_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-architecture-probe)
    kernel_id="$(kernel_id_from_metadata "$so2_architecture_probe_kernel_dir")"
    output_dir="${2:-$so2_architecture_probe_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-runtime-readiness)
    kernel_id="$(kernel_id_from_metadata "$so2_runtime_readiness_kernel_dir")"
    output_dir="${2:-$so2_runtime_readiness_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-prelaunch)
    kernel_id="$(kernel_id_from_metadata "$so2_prelaunch_kernel_dir")"
    output_dir="${2:-$so2_prelaunch_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-so2-selected-runtime-full)
    kernel_id="$(kernel_id_from_metadata "$so2_full_kernel_dir")"
    output_dir="${2:-$so2_full_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
    ;;
  output-setup)
    kernel_id="$(kernel_id_from_metadata "$setup_kernel_dir")"
    output_dir="${2:-$setup_output_dir}"
    require_remote_confirmed
    require_kaggle_cli
    mkdir -p "$output_dir"
    kaggle_api kernels output "$kernel_id" -p "$output_dir"
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
    kaggle_api kernels pull "$kernel_id" -p "$kernel_dir"
    ;;
  *)
    usage
    exit 1
    ;;
esac
