#!/usr/bin/env bash
# run_smoke.sh — Fast pre-delivery gate (~5 seconds).
#
# Checks:  1) All .py files compile  2) Key imports work  3) Contract suite loads
#          4) CLI help wiring
#
# Does NOT run full test suite — use run_regression.sh for that.
set -euo pipefail

PY="${PYTHON:-python}"
echo "[smoke] python: $PY ($($PY -V 2>&1))"

# 1. Compile sweep
echo "[smoke] 1/4 py_compile"
$PY -c '
import glob, py_compile, sys
files = sorted(glob.glob("**/*.py", recursive=True))
errs = []
for f in files:
    try: py_compile.compile(f, doraise=True)
    except Exception as e: errs.append((f, str(e)))
if errs:
    for f, e in errs: print(f"  FAIL: {f}: {e}", file=sys.stderr)
    sys.exit(1)
print(f"[smoke] py_compile OK ({len(files)} files)")
'

# 2. Key imports (catches missing deps / circular imports)
echo "[smoke] 2/4 key imports"
$PY -c '
import sys
modules = [
    "pipeline.aoi", "sdb_context", "train_context", "sdb_tier",
    "errors_scientific", "constants", "errors", "fusion",
]
failed = []
for m in modules:
    try:
        __import__(m)
    except Exception as e:
        failed.append(f"{m}: {type(e).__name__}: {e}")
if failed:
    for f in failed: print(f"  FAIL: {f}", file=sys.stderr)
    sys.exit(1)
print(f"[smoke] {len(modules)} key imports OK")
'

# 3. Contract suite structural check
echo "[smoke] 3/4 contract suite"
if [[ -f "contract_tests.py" ]]; then
  $PY -c '
from contract_tests import ContractTestSuite
suite = ContractTestSuite()
suite.add_standard_tests()
assert len(suite.tests) > 0, "No tests registered"
print("[smoke] contract suite OK: %d tests registered" % len(suite.tests))
'
fi

# 4. CLI help wiring
echo "[smoke] 4/4 CLI help"
$PY tests/test_cli_help.py

echo "[smoke] ALL PASSED"
