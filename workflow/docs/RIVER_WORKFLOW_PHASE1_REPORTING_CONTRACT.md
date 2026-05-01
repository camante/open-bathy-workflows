# River workflow Phase 1 reporting contract

## Goal

Make the active river workflow boring to debug.  A completed run should leave one clear evidence chain for the canonical parent/export/final route, without requiring a developer to infer which route wrote the final DEM.

## Required evidence chain

The active river route is:

1. `canonical_parent_dem`
2. `aoi_export_dem`
3. `final_user_dem`

The Phase 1 reporting contract requires these retained products/receipts to be visible in the run summary:

- canonical parent DEM
- AOI export DEM
- final user DEM
- retained canonical manifest
- AOI export identity receipt
- river workflow receipt
- final output receipt
- run summary JSON
- run summary text
- workflow actual trace, written later in finalization

## Important behavior

The run summary is written before the final workflow trace is emitted.  Therefore `workflow_actual_trace.json` is listed as `expected_after_summary_write` instead of being treated as a failed missing product at summary-write time.

Read-only science/support checks must never show empty skipped details.  If a science metric cannot be reported, the summary must say whether the stage science receipt was not retained, not available to the AOI export, or whether support/composition counts were unavailable.

## What this does not do

This phase does not rerun science stages, scan rasters, modify the final DEM, add fallback routes, or change the canonical parent/export architecture.  It only makes the existing evidence chain explicit and auditable.
