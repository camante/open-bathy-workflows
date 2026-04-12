#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 REPORT_JSON OUT_JSON [extra args...]" >&2
  exit 2
fi

REPORT_JSON="$1"
OUT_JSON="$2"
shift 2

python ./pre_hydraulic_baseline_gate.py \
  --report-json "$REPORT_JSON" \
  --out "$OUT_JSON" \
  --fail-on-regression \
  "$@"
