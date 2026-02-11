# Archived scripts (likely unused by `bathy_main.py`)

This folder holds scripts that are **not in the current execution path** when running the end-to-end workflow via:

```bash
python bathy_main.py ...
```

They were moved out of the repo root to reduce clutter and make it clearer which scripts are part of the supported pipeline.

## What’s here

The archived scripts are mostly:

- **Old/legacy entrypoints** from earlier workflow iterations
- **One-off utilities** for QA/QC, plotting, or packaging
- **Experimental prototypes** (e.g., alternative interpolation or adjustment approaches)

These scripts may still be useful, but they are **not imported or called** by the default pipeline (as of v1.2).

## How to use an archived script

If you want to run an archived script directly:

```bash
python archive/likely_unused/<script_name>.py --help
```

If you want to reintegrate one into the pipeline, move it back to the repo root (or a proper module directory) and update any imports/CLI calls accordingly.

## Contents

Currently archived:

- `CHANGELOG_v0.7.2.py`
- `adaptive_spatial_cv.py`
- `apply_patches.py`
- `bank_mask_from_xs.py`
- `bathy_interp_cli.py`
- `cudem_river_burn_taper.py`
- `cudem_river_fill_monotonic.py`
- `diagnose_bands.py`
- `generate_script_reference.py`
- `hyperparameter_tuning.py`
- `idw_gpu.py`
- `make_channel_mask.py`
- `measured_mask_from_points.py`
- `optical_utils.py`
- `prediction_validation.py`
- `river_bathy.py`
- `river_skeleton.py`
- `setup.py`
- `swot_adjust.py`
- `tidal.py`
- `tile_blending.py`
- `unified_bathy_report.py`
- `waffles_bathy_interp.py`
- `xs_adjust_monotonic.py`
