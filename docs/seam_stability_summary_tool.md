# Seam Stability Summary Tool

`compute_seam_stability_summary.py` aggregates seam metrics (coastal/river/fusion/transition)
and optional bank drift into a single JSON and applies readiness gates.

Default gates:
- Coastal seam: |bias|≤0.15, rmse≤0.50, p95≤1.0, overlap≥0.10
- River seam:   |bias|≤0.15, rmse≤0.50, overlap≥0.10
- Transition:   |bias|≤0.25, rmse≤0.75, overlap≥0.05
- Convergence (if provided): probe_p95_abs_delta_m≤0.25 for last 3 updates

Example:
python seam_stability_tools/scripts/compute_seam_stability_summary.py \
  --coastal-seam sdb_seam.csv --river-seam river_seam.csv --transition transition.csv \
  --bank-drift bank_drift.csv \
  --out-json seam_stability_summary.json --out-csv seam_gate.csv
