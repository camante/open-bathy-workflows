# Seam Stability Tools

These scripts provide explicit seam-analysis utilities that sit beside, rather than replace, the lightweight seam-comparison hooks already available in `bathy_main.py`.

## Included assets

- architecture diagram in SVG and PNG form
- seam-metric scripts for raster overlap, transitions, and longitudinal behavior
- summary/gating script for combining metrics into a single readiness result

## Scripts

### `seam_stability/seam_stability_tools/scripts/compute_raster_seam_metrics.py`
Compute overlap-based seam metrics between two rasters, optionally with masks.

### `seam_stability/seam_stability_tools/scripts/compute_transition_metrics.py`
Measure transition behavior where two methods or domains meet.

### `seam_stability/seam_stability_tools/scripts/compute_longitudinal_seam_metrics.py`
Evaluate longitudinal behavior along a centerline or river corridor.

### `seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py`
Aggregate seam metrics and optional bank-drift inputs into a single summary JSON and gate output.

## Typical workflow

1. produce or identify the rasters/manifests you want to compare
2. compute seam metrics with the appropriate script
3. aggregate the resulting metrics with `seam_stability/seam_stability_tools/scripts/compute_seam_stability_summary.py`
4. archive the resulting JSON/CSV beside the run reports that generated the compared rasters
