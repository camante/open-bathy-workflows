#!/usr/bin/env bash
# ci_smoke.sh — CI-friendly smoke test.
#
# Three tiers of verification:
#   ci_smoke.sh       — Fast gate (~5s): compile, imports, contracts, CLI help
#   run_regression.sh — Full unit tests (~30s): all 400+ tests
#   (heavy validation is manual / pre-release only)
#
# This script runs ONLY the smoke tier for fast CI feedback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$(mktemp -d)}"

cleanup_slop () {
  rm -rf "${PYTHONPYCACHEPREFIX:-}" 2>/dev/null || true
  find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
}
trap cleanup_slop EXIT

# Check for committed cache artifacts
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

echo "[ci_smoke] python: $($PYTHON_BIN -V)"
check_slop

$PYTHON_BIN repo_contract_checks.py --root .

# Run the lightweight smoke checks
PYTHON="$PYTHON_BIN" bash run_smoke.sh

# Receipt hooks (lightweight structural assertions)
$PYTHON_BIN - <<'PY'
from pathlib import Path
import sys

req = [
    (Path("repo_runtime_modes.py"), "ACTIVE_RIVER_METHOD"),
    (Path("repo_contract_checks.py"), "validate_runtime_contract"),
    (Path("xs_infer_bathy_raster.py"), "energy_solver_receipt.json"),
    (Path("validation/seam_metrics.py"), "compute_mask_boundary_seam_metrics"),
    (Path("archive/design_notes/CHECKLIST_A_GRADE.md"), "workflow checklist"),
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

check_slop
echo "[ci_smoke] OK"
