# River workflow Phase 3: bathy_main boundary

Phase 3 keeps `bathy_main.py` as the command/config entrypoint and continues moving active river finalization details into named modules.

## Active boundary after this phase

`bathy_main.py` is allowed to:

- parse command-line configuration;
- prepare the run report and top-level config fields;
- call `execute_river_workflow_entry(...)`;
- provide explicit callbacks for source resolution and shared final reporting.

`bathy_main.py` should not own detailed river final DEM identity logic.  The final DEM identity/touch-log implementation now lives in:

- `pipeline/final_dem_identity.py`

That module owns these verify-only functions:

- `verify_dem_enhanced_single_source_of_truth(...)`
- `record_dem_enhanced_touch(...)`
- `replace_with_symlink_or_copy(...)`
- `file_sha256(...)`

## Why this matters

The river workflow is now easier to debug because the active chain stays explicit:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

and final DEM identity verification is a named module, not embedded inside the large CLI orchestrator.  This same boundary should be reused when the SDB workflow is cleaned: keep the CLI entrypoint thin, and put source/export/final identity checks into small modules with one responsibility.

## Non-goals

This phase does not change river science, canonical solve behavior, AOI export behavior, or final DEM values.  It is a responsibility-boundary cleanup only.
