# Model Bank Validation Runbook

1) Run AOI1 (SDB)
2) Run adjacent AOI2 (SDB)
3) Compute seam metrics:
   compute_raster_seam_metrics.py --a AOI1_sdb.tif --b AOI2_sdb.tif --buffer-m 200
4) Record drift (bank_drift.csv)
5) Gate readiness using compute_seam_stability_summary.py

---
End of runbook.
