# River Workflow Post-v886 Legacy Archive Cleanup

## Goal

After v886 passed the canonical parent/export identity checks and read-only river science diagnostics, this cleanup freezes the active river workflow and moves hard-quarantined root-level legacy river scripts out of the root workflow namespace.

The active invariant remains:

```text
shared solve domain -> canonical parent river solution -> exact AOI export -> combined/DEM_enhanced.tif written once -> verify-only comparison
```

No river science, canonical solve construction, AOI export, final DEM materialization, or SDB/fusion routing behavior was intentionally changed in this cleanup.

## Active river path retained

The active river construction path remains under:

```text
pipeline/river_workflow/
pipeline/river_shared_solve/
```

with the active entrypoint bridge:

```text
river_workflow_entry.py
active_pipeline.py
river_runner.py
river_runner_contract.py
```

and canonical identity/export/final reporting helpers:

```text
canonical_river_solve_cache.py
canonical_river_solve_contract.py
canonical_river_solve_materialization.py
canonical_river_identity_guard.py
canonical_river_export_identity.py
canonical_river_seam_identity.py
pipeline/final_dem_identity.py
pipeline/final_dem_materialization.py
pipeline/run_summary.py
tools/compare_aoi_exports.py
```

## Archived root-level legacy river scripts

These files were moved to:

```text
legacy/river/archive_root_scripts/
```

Moved files:

```text
linear_v1_runner.py
linear_v1_runner_contract.py
river_bank_guidance.py
river_bank_longitudinal_fit.py
river_channel_scaffold.py
river_channel_surface.py
river_channel_template.py
river_graph_backbone_solver.py
river_primary_surface_rebuild.py
river_xs_realism.py
```

They represent inactive compatibility, v1/v2/channel/template/XS/graph-bank logic and are not part of the active canonical parent/export river route.

## Transitional utility not moved

`river_structured_scaffold.py` remains in the root namespace for now because the active river centerline stage still imports `build_centerline_points` from it. The next optional cleanup should extract the needed centerline-only helper surface into `pipeline/river_workflow/river_workflow_centerline_logic.py` and then archive the remaining legacy content.

## Compatibility references

Stale compatibility imports in non-active/legacy helper surfaces were redirected to the archive package so accidental use does not depend on root-level legacy files being present. The active route should still not call those archived modules.

## SDB template implication

This cleanup leaves the river workflow as the template for SDB:

```text
canonical/source-domain guidance product
-> exact AOI export/subset
-> final materialization from named export artifact
-> identity receipts
-> read-only parent science summary
-> single-writer final contract
```
