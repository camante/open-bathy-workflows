#!/usr/bin/env bash
# run_regression.sh — Full unit/regression test suite.
#
# Runs all tests in tests/ directory. Use before merge/release.
# For fast pre-delivery checks, use run_smoke.sh instead.
set -euo pipefail

PY="${PYTHON:-python}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(mktemp -d)}"

echo "[regression] python: $PY ($($PY -V 2>&1))"

# Run smoke first (fast fail)
echo "[regression] Running smoke checks first..."
bash run_smoke.sh

# Full test suite
echo "[regression] Running full test suite..."
$PY -m unittest discover -s tests -p "test*.py" -v

echo "[regression] ALL PASSED"
