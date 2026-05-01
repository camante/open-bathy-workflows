# REVIEW_NOTES_v895

## Bundle E: final route hard guard

Main invariant: the active river final DEM must follow one route only:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

This pass adds an explicit guard receipt that validates the final route after AOI identity and final materialization are complete. The guard does not repair, reroute, reproject, resample, interpolate, blend, or apply authoritative locking. It only checks the route and fails if a hard invariant is violated.

## Changes

- Added `river_workflow_final_route.py`.
- Added `final/river_final_route_guard_receipt.json` and `.txt` outputs.
- Wired the guard into `active_pipeline.py` immediately after:
  - AOI export is materialized into `combined/DEM_enhanced.tif`;
  - AOI export identity is checked against the canonical parent;
  - retained AOI export identity receipt is written.
- Added the guard to the active stage chain as `final_route_guard`.
- Added the guard receipt to run-summary receipt references.
- Added guard status to the consolidated river workflow receipt.
- Tightened `pipeline/final_dem_materialization.py` with:
  - named `write_final_dem_from_aoi_export(...)` entrypoint;
  - `src.is_file()` source check;
  - source/destination SHA256 fields;
  - explicit source/destination hash-match field.
- Added `tests/test_bundle_e_final_route_guard_static.py`.

## Guard checks

The guard requires:

- canonical parent exists;
- AOI export exists;
- final user DEM exists at `combined/DEM_enhanced.tif`;
- materialization receipt exists;
- materialization source path is the named AOI export;
- materialization destination path is the final user DEM;
- source role is `aoi_export_dem`;
- destination role is `final_user_dem`;
- writer role is `final_dem_materializer`;
- materialization is copy-only;
- no resampling;
- no reprojection;
- no pixel modification;
- no interpolation;
- no blending;
- no authoritative locking inside final materialization;
- no AOI-local final solve;
- touch-log actions are materializer-only;
- AOI export identity was checked against the canonical parent;
- AOI export identity passed at zero tolerance;
- mismatch pixels are zero;
- max absolute difference is zero;
- final hash equals AOI export hash.

## Remaining limitation

This bundle hardens the final route, but it does not yet rename the physical `river_workflow/` output folder or split the deeper `pipeline/river_workflow/` implementation into final stage modules. Those remain later cleanup work.
