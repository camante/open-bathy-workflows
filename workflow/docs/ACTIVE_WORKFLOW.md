# Active Workflow

This is the authoritative high-level description of the active workflow story for this repository.

## Scope

The repository supports coastal SDB, river guidance, and fused final products. Within that broader system, the **active river workflow** is the explicit linear path run through `bathy_main.py` with:


Older river methods still exist in legacy code, but the active repo contract is now one built-in river workflow (`river_workflow`). User-facing control comes from `--aoi`, optional `--solve-domain`, and a small set of explicit science/runtime knobs.

## Active river path

The active river path is:

`entry bundle -> solve_domain -> grids -> authoritative -> centerline -> wse_proxy -> authoritative_bed -> observed_offset -> modeled_offset -> backbone -> corridor -> surface -> lock -> export -> final_dem`

Each stage is intended to have one clear job and one explicit handoff into the next stage. The active package surface for this path now lives under `pipeline/river_shared_solve/`, with `pipeline/river_shared_solve/river_pipeline.py` as the main package-level orchestrator and `river_runner.py (generic active runner)` as the built-in active runner entrypoint. During the transition, that active package still delegates to the mature stage implementation under `pipeline/river_workflow/`.

## Stage meanings

### 1. entry bundle
Resolve the run-level inputs once:
- export AOI
- broader solve AOI
- projected CRS
- resolution
- canonical network source
- authoritative source paths
- export baseline source path

This stage exists so downstream stages do not rediscover inputs independently.

### 2. solve_domain
Register and validate the prepared canonical solve-domain bundle for the run.

### 3. grids
Build the solve-grid and export-grid templates that the downstream raster stages will use.

### 4. authoritative
Consume explicit authoritative source rasters, align them to the solve/export grids, and build authoritative support masks.

### 5. centerline
Build the explicit river centerline point framework used by the linear river path.

### 6. wse_proxy
Estimate centerline water-surface proxy values.

### 7. authoritative_bed
Sample authoritative river-bed evidence where it exists.

### 8. observed_offset
Compute observed bed offsets from WSE proxy to authoritative bed where direct support exists.

### 9. modeled_offset
Construct the modeled centerline offset field used in unsupported or weakly supported reaches.

### 10. backbone
Compute the centerline bed backbone.

### 11. corridor
Spread the backbone through the river corridor support domain.

### 12. surface
Build the solve-domain river primary surface.

### 13. lock
Authoritative-lock the river surface before export.

### 14. export
Move the locked river guidance and masks onto the export grid and prepare the final DEM handoff bundle.

### 15. final_dem
Build the canonical parent DEM on the canonical solve grid, export the user AOI DEM as an exact parent-grid subset, and then materialize `combined/DEM_enhanced.tif` from that AOI export. This stage must preserve the linear route:

`canonical_parent_dem -> aoi_export_dem -> final_user_dem`

The final materialization step does not perform terrain interpolation, blending, or authoritative locking.

## Invariants

The active workflow should satisfy these repo-level rules:

- one explicit upstream input bundle
- one clear stage handoff into the next stage
- no hidden source rediscovery inside downstream stages
- authoritative data remain hard control where they exist
- final user DEM is materialized only from the AOI export artifact
- normal runs keep outputs small and readable
- detailed receipts/manifests exist only for intermediate/debug runs

## What to inspect after a run

For a normal run, inspect in this order:

1. `RUN_OVERVIEW.txt`
2. `reports/README_FIRST.txt`
3. `reports/WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt`
4. `reports/WORKFLOW_INPUT_OUTPUT_TRACE.txt`
5. `reports/river_active_summary.json`
6. `bathy_report.json` only when you need machine-readable run/output detail
7. `io_manifest.json` only when you need exact file paths
8. screen log if needed

That is the primary operator path.

## Legacy note

Legacy river methods still exist in the codebase, including structured, hybrid, v1, and v2-era paths. They are not the main documentation path for the repo and should be treated as legacy or comparison methods unless explicitly needed.


## Supporting namespaces

The active linear path is still supported by grouped secondary namespaces:
- `validation/` for seam, benchmark, nested-AOI, and scientific validation helpers
- `reporting/` for run/final/provenance reporting writers
- `tools/debug/` for the active curated diagnostics helpers
- `legacy/debug/` for older debug-bundle and consolidated river-debug helpers kept only for comparison/troubleshooting

These are subordinate to the active river stage chain above; they are not alternate workflow stories.


The final route single-writer handoff and final DEM contract helpers now live under `pipeline/final_route/` and `pipeline/final_dem/`, keeping the active river path package, the final output route, and the final DEM policy/contract logic in clearly separated active namespaces.
