# Active River Module Boundary

This document records the active river workflow boundary and the Phase 0–9 cleanup policy. It is a cleanup/guard artifact only. It does **not** change DEM construction, river science, AOI export, final materialization, or cache behavior.

## Core invariant

The active river workflow is:

```text
shared solve domain
-> canonical parent river solution
-> exact AOI export/window from the canonical parent
-> combined/DEM_enhanced.tif materialized once from the AOI export
-> final review products and verify-only comparison
```

The saved development rule applies here:

```text
one input artifact -> one transformation -> one output artifact -> one receipt/check
```

For failures:

```text
do not patch the final DEM
do not add a fallback
do not add another route
find the first wrong artifact
fix the stage that created that artifact
```

## Active orchestration and entrypoint bridge files

These files are allowed to connect the monolithic CLI/runtime environment to the active river route:

- `bathy_main.py`
- `river_workflow_entry.py`
- `active_pipeline.py`
- `river_runner.py`
- `river_runner_contract.py`

`river_runner.py` and `river_runner_contract.py` are active entrypoint bridge files. They are not construction stages, but they are part of the active route because `bathy_main.py` calls `run_river_workflow_direct(...)` through this bridge. They may register products and receipts and call the canonical parent/export pipeline, but they must not implement an alternate river workflow.

`bathy_main.py` is still broad and imports older/general modules because it remains the top-level workflow CLI. That is transitional orchestration complexity, not the river construction contract.

## Active river core

The active construction/export implementation is the canonical shared-solve wrapper plus the mature linear river stage implementation:

- `pipeline/river_shared_solve/river_pipeline.py`
- `pipeline/river_workflow/river_workflow_pipeline.py`
- `pipeline/river_workflow/river_workflow_context.py`
- `pipeline/river_workflow/river_workflow_contract.py`
- `pipeline/river_workflow/river_workflow_execution_contract.py`
- `pipeline/river_workflow/river_workflow_paths.py`
- `pipeline/river_workflow/river_workflow_receipts.py`
- `pipeline/river_workflow/river_workflow_canonical_manifest.py`
- `pipeline/river_workflow/river_workflow_stage_solve_domain.py`
- `pipeline/river_workflow/river_workflow_stage_grids.py`
- `pipeline/river_workflow/river_workflow_stage_authoritative.py`
- `pipeline/river_workflow/river_workflow_stage_centerline.py`
- `pipeline/river_workflow/river_workflow_stage_wse_proxy.py`
- `pipeline/river_workflow/river_workflow_stage_authoritative_bed.py`
- `pipeline/river_workflow/river_workflow_stage_observed_offset.py`
- `pipeline/river_workflow/river_workflow_stage_modeled_offset.py`
- `pipeline/river_workflow/river_workflow_stage_backbone.py`
- `pipeline/river_workflow/river_workflow_stage_corridor.py`
- `pipeline/river_workflow/river_workflow_stage_surface.py`
- `pipeline/river_workflow/river_workflow_stage_lock.py`
- `pipeline/river_workflow/river_workflow_stage_export.py`
- `pipeline/river_workflow/river_workflow_stage_final_dem.py`


## Phase 3 parent/export function boundary

The active river pipeline now exposes the architectural boundary as named top-level functions in `pipeline/river_workflow/river_workflow_pipeline.py`:

- `run_river_workflow_pipeline(ctx)` remains the backward-compatible entry point.
- `run_river_parent_export_workflow(ctx)` owns parent/export routing.
- `resolve_canonical_parent_handoff(ctx)` resolves an existing canonical parent manifest + parent DEM for export-only routing.
- `run_aoi_export_from_canonical_parent(ctx, handoff)` is the AOI export-only path and is not allowed to run construction stages; it consumes a `CanonicalParentHandoff`, not cache/source branches.
- `run_canonical_river_build(ctx)` is the construction path and is the only path allowed to run the canonical build stages before AOI export.

This is a code-clarity boundary only. It does not add a new route, fallback, repair step, or DEM source.

## Active parent/export/final identity helpers

These helpers support canonical-parent identity, AOI export identity, and final DEM materialization. They should remain subordinate to the active chain and should not create alternate DEM routes:

- `canonical_river_solve_cache.py`
- `canonical_river_solve_contract.py`
- `canonical_river_solve_materialization.py`
- `canonical_river_identity_guard.py`
- `canonical_river_export_identity.py`
- `canonical_river_seam_identity.py`
- `pipeline/aoi.py`
- `pipeline/aoi_identity.py`
- `pipeline/final_dem_materialization.py`

## Active verify/reporting boundary files

These files are active, but they are not construction stages or parent/export identity helpers. They may read receipts and summarize results, but must not repair, reroute, or rewrite construction products:

- `pipeline/run_summary.py`
- `reporting/final_reporting.py`
- `tools/compare_aoi_exports.py`

## Verify-only tools

These may read outputs and receipts, but must not repair, reroute, or rewrite construction products:

- `tools/compare_aoi_exports.py`
- `tests/test_active_river_phase0_contract_static.py`
- `tests/test_active_river_module_boundary_static.py`

## Active transitional utilities

The current active centerline stage still imports selected helper functions from:

- `river_structured_scaffold.py`

This module is therefore not quarantined in Phase 0/1. A later cleanup can extract the small helper surface actually used by `pipeline/river_workflow/river_workflow_stage_centerline.py` into an active utility module, then quarantine the remaining historical scaffold code.

## Legacy/quarantine candidates

These files may still exist and may contain useful historical reference or utility logic, but they should not be treated as active river workflow paths. They should be quarantined or refactored only after import tracing proves what the active route still needs.

- `legacy/river/archive_root_scripts/legacy_linear_v1_runner.py`
- `legacy/river/archive_root_scripts/legacy_linear_v1_runner_contract.py`
- `pipeline/river_shared_solve/river_context.py`
- `pipeline/river_shared_solve/river_contract.py`
- `pipeline/river_shared_solve/river_paths.py`
- `legacy/river/archive_root_scripts/river_primary_surface_rebuild.py`
- `legacy/river/archive_root_scripts/river_channel_scaffold.py`
- `legacy/river/archive_root_scripts/river_channel_surface.py`
- `legacy/river/archive_root_scripts/river_channel_template.py`
- `legacy/river/archive_root_scripts/river_graph_backbone_solver.py`
- `legacy/river/archive_root_scripts/river_xs_realism.py`
- `legacy/river/archive_root_scripts/river_bank_guidance.py`
- `legacy/river/archive_root_scripts/river_bank_longitudinal_fit.py`
- `legacy/river/`

## Phase 0–9 cleanup constraint

These phases intentionally do not delete active modules or change construction behavior. They add documented boundaries, static guards, route clarity, parent/export handoff clarity, receipt-purpose clarity, legacy import guards, route-specific reporting, first-wrong-artifact guidance, and validation-noise policy so future cleanup can continue without destabilizing the stable canonical-parent/AOI-export behavior.

## Phase 2 route ownership reporting boundary

Phase 2 clarifies method/route ownership in logs and summaries without changing construction behavior. Canonical river runs should report:

```text
requested_methods = sdb,river,fuse
active_route = canonical_river_parent_export
active_methods = river
inactive_methods = sdb,fuse
final_dem_owner = river
```

This is reporting-only. It must not select alternate DEM sources, reroute outputs, repair seams, or change cache/export behavior.

## Phase 4 canonical parent handoff boundary

The active route now treats cache reuse, an existing canonical solution manifest,
and identity-only adoption of an older parent raster as different ways to resolve
the same object: `CanonicalParentHandoff`.

The architectural rule is:

```text
resolve_or_build canonical parent -> CanonicalParentHandoff -> AOI export
```

AOI export should not branch on whether the parent came from a fresh build, a
cached solve, an existing manifest, or an adopted parent. Those details are
reported in the handoff summary and receipts, but they do not change export
behavior. Cache is therefore kept as an implementation detail rather than a
second architecture.


## Phase 5 receipt-purpose boundary

The active route now writes one receipt-purpose manifest:

- runtime `river_workflow/manifests/receipt_purpose_manifest.json` under each run output

This file is an index, not a new workflow stage. It identifies the primary receipts for each purpose:

- canonical parent manifest
- AOI export identity receipt
- final output receipt
- comparison report
- human run summary
- one ordered receipt per active construction/export stage

Older/internal receipts remain available as diagnostic receipts, but they are explicitly subordinate to the primary receipts. The receipt-purpose manifest must remain verify-only: it must not drive routing, select DEM sources, repair seams, or patch final outputs.

## Phase 6 legacy-boundary guard

Phase 6 adds a hard static boundary around old river workflow modules without moving or deleting them yet. Phase 2 corrected the active boundary so `river_runner.py` and `river_runner_contract.py` are explicitly active entrypoint bridge files, not hard-quarantine legacy modules. The boundary is defined in:

- `legacy_river_quarantine_manifest.json`

This is a cleanup guard only. It does not change DEM construction, river science, cache behavior, AOI export, final materialization, or comparison behavior.

Phase 6 also separates construction files from verify/reporting files in `legacy_river_quarantine_manifest.json`.  Comparison and summary modules are active, but they are not construction stages; they remain verify-only and must not import hard-quarantine river workflow modules.

The current rule is:

```text
active construction files and verify/reporting files must not import hard-quarantine river workflow modules
```

Hard-quarantine modules include older compatibility/surface/template/backbone/XS/bank workflow modules such as:

- `legacy/river/archive_root_scripts/legacy_linear_v1_runner.py`
- `legacy/river/archive_root_scripts/river_primary_surface_rebuild.py`
- `legacy/river/archive_root_scripts/river_channel_scaffold.py`
- `legacy/river/archive_root_scripts/river_channel_surface.py`
- `legacy/river/archive_root_scripts/river_channel_template.py`
- `legacy/river/archive_root_scripts/river_graph_backbone_solver.py`
- `legacy/river/archive_root_scripts/river_xs_realism.py`
- `legacy/river/archive_root_scripts/river_bank_guidance.py`
- `legacy/river/archive_root_scripts/river_bank_longitudinal_fit.py`
- `legacy/river/`

This phase deliberately does **not** delete those files. It first prevents the active canonical-parent/AOI-export route from depending on them. Once the guard stays green, later cleanup can move or remove legacy modules safely.

`river_structured_scaffold.py` remains an explicit active transitional utility because `pipeline/river_workflow/river_workflow_stage_centerline.py` still imports `build_centerline_points` from it. The next legacy cleanup for that file should be helper extraction, not deletion.

## Phase 7 river-route reporting boundary

Phase 7 simplifies the human-facing run summary for canonical river runs. It is reporting-only and does not change DEM construction, river science, cache behavior, AOI export, final materialization, or comparison behavior.

For `canonical_river_parent_export` / `canonical_parent_plus_aoi_export` runs, the summary must read as a debugging map for the active route:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

The summary should identify the primary receipts and artifacts first:

- canonical parent manifest
- AOI export identity receipt
- river workflow receipt
- final output receipt
- final user DEM

The canonical river summary must not present inactive SDB/fusion products as if they participated in DEM construction. Any SDB/fusion entries for this route should remain inactive-method context only.

## Phase 8 first-wrong-artifact stage guides

Phase 8 adds read-only first-wrong-artifact guidance to active stage receipts and trace entries. This is diagnostic metadata only and does not change DEM construction, river science, cache behavior, AOI export, final materialization, or comparison behavior.

Each active stage receipt now carries a `first_wrong_artifact_guide` with:

- the first explicit input artifact listed by the stage;
- the first explicit output artifact listed by the stage;
- the stage's critical invariant;
- the first validation/check name for that stage;
- a concise hint for what to inspect if that stage is the first suspicious artifact.

The guide is constrained by this policy:

```text
first_wrong_artifact_guide is verify-only
first_wrong_artifact_guide must not reroute
first_wrong_artifact_guide must not repair outputs
first_wrong_artifact_guide must not select another DEM source
```

This phase supports the saved debugging rule: find the first wrong artifact and fix the stage that created it, rather than patching the final DEM or adding another route.

## Phase 9 validation-noise policy

Phase 9 narrows validation/reporting to checks that answer real river debugging questions. It is reporting/validation cleanup only and does not change DEM construction, river science, cache behavior, AOI export, final materialization, or comparison behavior.

The allowed river validation questions are:

- Did AOI export exactly match the canonical parent window?
- Did final DEM equal AOI export?
- Did authoritative lock preserve measured cells?
- Did WSE/backbone behave physically enough to inspect?
- Did north/south use the same parent?
- What was the first failing artifact?

Optional placeholder validation stages are suppressed by default when their inputs are not configured:

- benchmark holdout validation;
- postrun adjacent/nested regression validation;
- external validation/invariance case evaluation;
- external scientific validation case evaluation.

Suppression is recorded under `optional_validation` when those helper functions are invoked, but skipped placeholder products should not be written as primary outputs. This keeps validation verify-only and prevents generic skipped reports from looking like active river workflow stages.

The policy is defined by:

- `validation/river_validation_policy.py`
- `validation/manifests/river_validation_policy_manifest.json`

Configured external validation remains possible, but unconfigured placeholder validation must not add workflow noise, reroute outputs, repair rasters, or select alternate DEM sources.
