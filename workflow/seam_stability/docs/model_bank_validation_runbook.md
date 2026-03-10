# Model Bank Validation Runbook

1. Run AOI A with the intended bank settings.
2. Run adjacent AOI B with the same bank settings.
3. Save each run's `io_manifest.json`, `unified_bathy_report.json`, and `artifacts_sdb.json`.
4. Compute seam metrics with `seam_stability/seam_stability_tools/scripts/compute_raster_seam_metrics.py`.
5. If you are tracking repeated bank updates, compute your probe-set drift or bank-drift artifact.
6. Summarize the results with `seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py`.
