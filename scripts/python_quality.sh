#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for Python quality checks." >&2
  exit 1
fi

python_version="${UV_PYTHON:-3.12}"

echo "Python quality checks"
echo "====================="
echo "Using uv Python target: ${python_version}"
echo

uv run --python "$python_version" ruff format .
uv run --python "$python_version" ruff check --fix .
uv run --python "$python_version" ruff check .
uv run --python "$python_version" basedpyright
