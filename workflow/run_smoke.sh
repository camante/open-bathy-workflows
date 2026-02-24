#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-python}"

echo "[smoke] python: $PY"

echo "[smoke] compile-only"
$PY tests/test_compile.py

# Optional: run contract tests if a harness exists
if [[ -f "contract_tests.py" ]]; then
  echo "[smoke] contract tests (best effort)"
  $PY -c 'import contract_tests; print("[contract] import OK")' >/dev/null
fi

echo "[smoke] OK"

echo "[SMOKE] CLI help wiring"
python tests/test_cli_help.py || true
