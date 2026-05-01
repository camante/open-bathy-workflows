# River workflow Phase 6: bed-backbone downstream-rise cleanup

## Goal

Keep the passing canonical parent/export/final architecture unchanged, while reducing physically implausible downstream rises in the centerline bed backbone.

The active architecture remains:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

## First wrong artifact

The latest passing comparison reported a warning in the retained canonical parent science receipts:

```text
backbone_downstream_rise: warn
```

WSE monotonicity passed and modeled-offset support was present, so this phase is limited to the bed-backbone stage.

## Change

`pipeline/river_workflow/river_workflow_stage_backbone.py` now uses a support-aware downstream-rise limiter:

- guidance/interpolated/scaffolded points may be damped to the allowed local downstream-rise step;
- observed offset anchors remain capped, so true conflicts with observed support remain visible in receipts;
- no whole-profile isotonic projection is reintroduced;
- the raw formula field `bed_backbone_raw_z_m = wse_proxy_z_m - offset_modeled_m` remains retained for audit.

## Cache behavior

The canonical cache contract version is bumped to force rebuilding the parent solution with the updated backbone algorithm. This avoids silently reusing the old cached parent that produced the downstream-rise warning.

## What this does not change

- AOI export identity
- final DEM single-writer contract
- final-folder materialization
- SDB/fusion routing
- WSE construction
- observed-offset construction
- modeled-offset construction
