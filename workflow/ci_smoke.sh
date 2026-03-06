#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Prevent creation of __pycache__ in the repo during checks.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(mktemp -d)}"

echo "[ci_smoke] python: $($PYTHON_BIN -V)"
echo "[ci_smoke] pycacheprefix: ${PYTHONPYCACHEPREFIX}"

check_slop () {
  local bad_dirs bad_files
  bad_dirs=$(find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) || true)
  bad_files=$(find . -type f \( -name "*.pyc" -o -name "*.pyo" \) || true)
  if [[ -n "${bad_dirs}" || -n "${bad_files}" ]]; then
    echo "[ci_smoke][ERROR] Cache artifacts found in repo tree:"
    [[ -n "${bad_dirs}" ]] && echo "${bad_dirs}"
    [[ -n "${bad_files}" ]] && echo "${bad_files}"
    exit 1
  fi
}

# Fail fast on committed/unwanted cache artifacts
check_slop

# Compile all Python files (without writing .pyc into repo)
$PYTHON_BIN - <<'PY'
import glob, py_compile, sys
files = sorted(glob.glob('**/*.py', recursive=True))
errs = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        errs.append((f, str(e)))
if errs:
    print("[ci_smoke][ERROR] py_compile failed:")
    for f, e in errs[:50]:
        print("  -", f, e)
    sys.exit(1)
print(f"[ci_smoke] py_compile OK ({len(files)} files)")
PY

# Run unit tests
$PYTHON_BIN -m unittest discover -s tests -p "test*.py" -v

# Lightweight "receipt hooks" assertions (no pipeline execution)
$PYTHON_BIN - <<'PY'
from pathlib import Path
import sys

req = [
    (Path("bathy_main.py"), "input_receipt.json"),
    (Path("xs_infer_bathy_raster.py"), "energy_solver_receipt.json"),
    (Path("seam_metrics.py"), "compute_mask_boundary_seam_metrics"),
    (Path("CHECKLIST_A_GRADE.md"), "workflow checklist"),
]
missing = []
for p, token in req:
    if not p.exists():
        missing.append(f"missing file: {p}")
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    if token not in text:
        missing.append(f"{p}: missing token '{token}'")
if missing:
    print("[ci_smoke][ERROR] Receipt hooks check failed:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
print("[ci_smoke] receipt hooks OK")
PY

# Ensure checks did not create slop
check_slop

echo "[ci_smoke] OK"
