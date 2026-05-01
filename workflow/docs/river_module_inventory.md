# River module inventory — Bundle A

This inventory separates the active public workflow name from retained implementation internals.
Bundle A is behavior-preserving: it does not delete legacy code or rename on-disk product folders yet.

| Module/file | Bundle A status | Notes |
|---|---|---|
| `river_workflow.py` | ACTIVE_PUBLIC_ENTRY | Stable public façade for the single active river workflow. |
| `river_workflow_entry.py` | ACTIVE_ENTRY | Calls `run_active_river_workflow()` / `finalize_active_river_workflow()`. |
| `active_pipeline.py` | ACTIVE_ORCHESTRATION | Still contains compatibility aliases and internal `active_river` names. |
| `river_runner.py` | ACTIVE_RUNNER | Runs canonical parent/AOI export path; still delegates to historical `pipeline.river_workflow` internals. |
| `pipeline/river_shared_solve/river_pipeline.py` | ACTIVE_PIPELINE | Current detailed canonical parent/AOI export implementation. |
| `pipeline/river_workflow/*` | ACTIVE_INTERNAL_TRANSITION | Retained implementation internals; should be renamed/split in later bundles only after tests lock behavior. |
| `legacy/river/archive_root_scripts/*` | LEGACY_REFERENCE | Archived historical root scripts; must not participate in active routing. |
| `xs_builder.py` and `xs_*` root modules | LEGACY_OR_SUPPORT_REVIEW | Not removed in Bundle A; classify before deletion/quarantine in later bundles. |

## Bundle A boundary

Bundle A only changes the active naming layer and comparison script ergonomics.  It does not change river science,
canonical solve construction, raster math, or final DEM writing behavior.
