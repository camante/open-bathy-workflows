#!/usr/bin/env bash
set -euo pipefail

# Offline verification runner: compile + smoke tests (no network).
# Usage: ./verify_repo.sh

PY="${PYTHON:-python}"

echo "[verify] python: $PY"
echo "[verify] python version: $($PY -V)"

# Always start clean (avoid stale artifacts affecting results)
rm -rf tests/_smoke_out* tests/__pycache__ __pycache__ || true

echo "[verify] compileall"
$PY -m compileall -q .

# Remove bytecode outputs so the working tree stays clean after verification.
rm -rf tests/__pycache__ __pycache__ || true

if [ -f "./run_smoke.sh" ]; then
  echo "[verify] smoke tests"
  bash ./run_smoke.sh
else
  echo "[verify] WARNING: run_smoke.sh not found; skipping smoke tests"
fi

echo "[verify] OK"
