# workflow_v896 review notes

Bundle F was applied after Bundles A through E.

## Error fix / integration check

The current active code was checked for stale active imports from `pipeline.river_linear`. The final DEM materializer still imported the old contract package, which would fail after the physical package rename. This pass updates final-route/materialization imports to the new active package.

## Bundle F changes

- Added active implementation package `pipeline/river_workflow/`.
- Renamed active implementation files from `river_linear_*` to `river_workflow_*`.
- Updated active imports in `active_pipeline.py`, `river_runner.py`, `river_workflow_final_route.py`, `pipeline/final_dem_materialization.py`, and `pipeline/river_shared_solve/*`.
- Changed active run work directory from `<out_dir>/river_linear/` to `<out_dir>/river_workflow/`.
- Moved the old implementation package to `legacy/river/pipeline_river_linear_reference/` as reference-only code.
- Removed the active `active_linear` mirror and deprecated `river_linear_*` compatibility output keys from the active wrapper layer.
- Added `tests/test_bundle_f_physical_rename_static.py`.
- Updated `docs/ACTIVE_RIVER_WORKFLOW_CONTRACT.md` to describe the new active package and runtime folder.

## Remaining known limitations

- Internal data-object names such as `linear_inputs` still exist in the renamed package and should be cleaned later when the stage split is done.
- The active implementation is still not split into the final twelve conceptual modules.
- Some method aliases such as `linear_v1` still exist in runtime normalization for backward compatibility; removing those belongs to the later method-removal bundle.
