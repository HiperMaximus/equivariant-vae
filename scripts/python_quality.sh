#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

venv_bin=".venv/bin"
if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="$PWD/runs/local_tmp/python_quality_$$"
  trap 'rm -rf "$TMPDIR"' EXIT
fi
mkdir -p "$TMPDIR"
missing=0

for tool in python ruff pytest basedpyright; do
  if [[ ! -x "$venv_bin/$tool" ]]; then
    echo "missing: $venv_bin/$tool" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  cat >&2 <<'EOF'
Repo-local Python tools are missing.

This quality gate intentionally does not run uv sync or download dependencies.
After explicit user permission for dependency sync, prepare the environment with:

  uv sync --locked --python 3.12 --group dev

Then rerun:

  ./scripts/python_quality.sh
EOF
  exit 1
fi

echo "Python quality checks"
echo "====================="
echo "Using repo-local .venv without dependency sync"
echo

"$venv_bin/python" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version.split()[0]}")

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required in the repo-local quality environment") from exc

if platform.system() == "Linux" and "+cpu" not in torch.__version__:
    raise SystemExit(f"expected CPU-only Linux PyTorch wheel, got {torch.__version__}")

if torch.cuda.is_available():
    raise SystemExit("local quality environment must not expose CUDA")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print("cuda_available=False")
PY

"$venv_bin/ruff" format .
"$venv_bin/ruff" check --fix .
"$venv_bin/ruff" check .

if find tests -type f -name '*.py' -print -quit | grep -q .; then
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" "$venv_bin/pytest" tests
else
  echo "Skipping pytest: tests/ has no Python tests yet."
fi

"$venv_bin/basedpyright"
