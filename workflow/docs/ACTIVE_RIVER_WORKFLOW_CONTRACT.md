# Active River Workflow Contract

This document records the active river workflow contract after Bundles A through F.
The active route is the canonical river parent/export route, not an older `linear_v1`, `simple_v2`, or user-selectable river method.

## Active entrypoints

- `river_workflow_entry.execute_river_workflow_entry(...)`
- `active_pipeline.run_active_river_workflow(ctx)`
- `active_pipeline.finalize_active_river_workflow(ctx, workflow_result)`
- `river_runner.run_river_workflow_direct(...)`

## Active implementation package

The active implementation package is now:

- `pipeline/river_workflow/`

The historical `pipeline/river_linear/` package has been moved to:

- `legacy/river/pipeline_river_linear_reference/`

That legacy directory is retained only as reference during cleanup and must not be imported by active workflow code.

## Active runtime work directory

New runs write active river workflow artifacts under:

- `<out_dir>/river_workflow/`

The active final DEM materialization route remains:

```text
canonical_parent_dem -> aoi_export_dem -> combined/DEM_enhanced.tif
```

## Required final-route invariant

The final user DEM must be copy-materialized from the AOI export DEM only. It must not be produced by AOI-local recomputation, blending, interpolation, reprojection, or a second final DEM writer.

## Transitional items intentionally left for later bundles

- Some internal variable names still use `linear_inputs` to refer to the canonical source bundle; those are data-object names, not active route selectors.
- Deeper stage modules are renamed into `pipeline/river_workflow/`, but they have not yet been split into the final twelve conceptual stage files.
- Method-style aliases in runtime-mode normalization remain for backward compatibility and should be removed in a later method-removal bundle.
