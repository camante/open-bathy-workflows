# River Bathymetry Validation Runbook

1) Run AOI1 (river)
2) Run adjacent AOI2 (river)
3) Seam metrics (with masks):
   compute_raster_seam_metrics.py --a riverA.tif --b riverB.tif --mask-a maskA.tif --mask-b maskB.tif
4) Longitudinal metrics:
   compute_longitudinal_seam_metrics.py --bed-elev bed_navd88.tif --centerline centerline.gpkg
5) Gate readiness using compute_seam_stability_summary.py

---
End of runbook.
