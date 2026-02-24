# Model Bank Validation Plan (Operational & Publication-Grade)

Purpose:
Ensure SDB model bank produces:
- tile-to-tile continuity
- convergent updates
- accuracy paired with coverage
- explicit DOA behavior

## Required outputs per run
- unified_bathy_report.json (bank id/hash, update_index, sources, depth limits, DOA policy)
- training_source_summary.csv
- validation_summary.csv (spatial CV / leave-tile-out required)
- seam_metrics.csv
- bank_drift.csv

## Coverage accounting (required)
Report fractions for:
optical finite, clear water, water mask, DOA pass, final predicted.

## Bank convergence (required)
Probe-set deltas:
p95(|Δ_k|) should drop below 0.25 m for 3 consecutive updates.

---
End of plan.
