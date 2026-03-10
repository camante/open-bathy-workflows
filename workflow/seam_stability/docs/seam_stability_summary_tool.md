# Seam Stability Summary Tool

`seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py` combines seam outputs into one summary JSON and optional gate result.

Use it after you have already generated one or more of the following:

- raster seam metrics
- transition metrics
- longitudinal seam metrics
- bank-drift or probe-set drift artifacts

Typical use:

```bash
python seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py \
  --coastal-seam sdb_seam.csv \
  --river-seam river_seam.csv \
  --transition transition.csv \
  --out-json seam_stability_summary.json
```

Treat the summary as a review aid, not a replacement for inspecting the underlying metrics and manifests.
