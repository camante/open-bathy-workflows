#!/usr/bin/env bash
set -euo pipefail

# Offline verification runner: compile + smoke tests (no network).
# Usage: ./verify_repo.sh

PY="${PYTHON:-python}"

TMP_PYCACHE="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_PYCACHE" "${PYTHONPYCACHEPREFIX:-}" tests/__pycache__ __pycache__ || true
  find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$TMP_PYCACHE"

echo "[verify] python: $PY"
echo "[verify] python version: $($PY -V)"

echo "[verify] pycacheprefix: $PYTHONPYCACHEPREFIX"

# Always start clean (avoid stale artifacts affecting results)
cleanup

echo "[verify] py_compile"
"$PY" - <<'PYVERIFY'
from pathlib import Path
import py_compile
for path in Path('.').rglob('*.py'):
    if '.git' in path.parts:
        continue
    py_compile.compile(str(path), doraise=True)
PYVERIFY

if [ -f "./run_smoke.sh" ]; then
  echo "[verify] smoke tests"
  bash ./run_smoke.sh
else
  echo "[verify] WARNING: run_smoke.sh not found; skipping smoke tests"
fi

echo "[verify] OK"
