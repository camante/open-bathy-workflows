# Active Linear Phase 8: seam identity and nested-AOI contract checks

This phase locks the solve-once/export-many rule at the contract layer.

## Rule
If two AOIs resolve to the same canonical river solve identity, they must reuse the same shared canonical solve cache root.

## Current contract checks
- same `river_system_id` + same `canonical_solve_aoi` + same `projected_crs` + same `target_resolution_m`
  -> same `canonical_solve_cache_key`
- the canonical solve root must live under the shared workflow cache root, not the per-run derived cache root
- the canonical solve contract records the shared cache key explicitly

These checks do not replace a full raster seam comparison, but they prevent the workflow from silently solving the same river system into separate per-run cache trees.
