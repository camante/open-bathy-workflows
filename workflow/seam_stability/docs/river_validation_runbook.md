# River Bathymetry Validation Runbook

1. Run river AOI A.
2. Run adjacent or downstream/upstream river AOI B if seam behavior is part of the review.
3. Save each run's `io_manifest.json`, `bathy_report.json`, and `unified_bathy_report.json`.
4. Compute raster seam metrics with `seam_stability/seam_stability_tools/scripts/compute_raster_seam_metrics.py`.
5. Compute longitudinal metrics with `seam_stability/seam_stability_tools/scripts/compute_longitudinal_seam_metrics.py` when centerline review is needed.
6. Aggregate results with `seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py` if you want a gateable summary artifact.
