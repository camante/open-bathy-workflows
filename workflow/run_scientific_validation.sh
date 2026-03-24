#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
pytest -q \
  tests/test_phase_j_scientific_validation.py \
  tests/test_validation_invariance_framework.py \
  tests/test_validation_runner.py
