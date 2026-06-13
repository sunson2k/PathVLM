#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/Scripts/python.exe"
else
  PYTHON="python"
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/run_pipeline.py" "$@"
