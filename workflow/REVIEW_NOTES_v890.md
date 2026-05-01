# Workflow v890 Review Notes

## Purpose

This is a targeted corrective release for the Bundle A active river workflow naming layer.

## Fix

`workflow_v889` introduced the public `run_active_river_workflow()` wrapper but accidentally called `run_river_workflow_direct()` without forwarding the required bathy-main callbacks. That caused runs to fail immediately with:

```text
RuntimeError: river_runner_missing_detect_working_srs_callback
```

`workflow_v890` preserves the Bundle A active naming layer but forwards the same explicit callbacks that the old bathy-main `run_river()` wrapper supplied:

- `logger`
- `script_dir`
- `ensure_dir_fn`
- `detect_working_srs_fn`
- `estimate_raster_pixel_size_m_for_dst_crs_fn`
- `resolve_linear_shared_source_artifacts_fn`
- `prepare_linear_canonical_source_bundle_fn`
- `linear_inputs_override=None`
- `return_linear_details=True`

This is a wiring fix only. It does not change the river science stages, canonical parent/export logic, or final DEM materialization contract.

## Compare script

The configurable AOI-count compare behavior from v889 is retained:

```bash
./compare.sh merrimack 890 2
./compare.sh merrimack 890 3
./compare.sh merrimack 890 --aoi-count 4
./compare.sh merrimack v890 --aoi-count 5
```

## Verification

Targeted syntax/static verification was run after the patch, including the active workflow modules and compare scripts.
