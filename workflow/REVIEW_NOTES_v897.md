# Workflow v897 review notes

Bundle G was applied on top of v896.

## Main invariant

The active river workflow remains one canonical parent solve followed by AOI export and final DEM materialization. This pass does not change the science behavior; it makes the active implementation easier to trace and removes broad exception masking from active river construction files.

## Changes

- Added numbered active stage import modules under `pipeline/river_workflow/`:
  - `stage_01_solve_domain.py`
  - `stage_02_grids.py`
  - `stage_03_authoritative_source.py`
  - `stage_04_centerline.py`
  - `stage_05_wse_proxy.py`
  - `stage_06_authoritative_bed.py`
  - `stage_07_observed_offset.py`
  - `stage_08_modeled_offset.py`
  - `stage_09_bed_backbone.py`
  - `stage_10_river_corridor.py`
  - `stage_11_primary_surface.py`
  - `stage_12_authoritative_lock.py`
  - `stage_13_aoi_export.py`
  - `stage_14_final_products.py`
- Updated `river_workflow_pipeline.py` so the active orchestrator imports through those numbered stage modules.
- Added `river_workflow_stage_errors.py` as the common typed stage error location for later stage-level cleanup.
- Replaced broad `except Exception` / bare `except:` handlers in active `pipeline/river_workflow`, `river_workflow*.py`, `active_pipeline.py`, and `river_runner.py` with typed catches or explicit diagnostic-only handling.
- Added `tests/test_bundle_g_stage_split_static.py`.

## Limitations intentionally left

- This pass adds numbered stage import modules but does not yet physically move all implementation bodies into those files. The next cleanup can move the code bodies once a real v897 run confirms behavior is stable.
- The active execution still has 14 current implementation stages because grid and corridor are separate implementation artifacts. The conceptual target remains the simpler final river workflow, but this pass avoids collapsing artifacts without a real run.

## Recommended local run

```bash
./compare.sh merrimack 897 2
```
