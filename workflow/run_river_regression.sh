#!/usr/bin/env bash
set -euo pipefail
pytest -q \
  tests/test_phase_a_contracts.py \
  tests/test_phase_b_xs_inference.py \
  tests/test_phase_c_hybrid_merge.py \
  tests/test_phase_d_hybrid_stage.py \
  tests/test_phase_e_river_mask_stage.py \
  tests/test_phase_f_river_end_to_end.py
