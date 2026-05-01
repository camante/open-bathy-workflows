# River workflow Phase 4 — canonical science receipts as parent artifacts

## Goal

The active river workflow should remain:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

AOI export runs must not recompute WSE, observed offsets, modeled offsets, backbone, primary surface, or authoritative lock stages. They should summarize science/support diagnostics only by reading the retained canonical parent manifest and its immutable stage receipts.

## Implemented contract

The canonical parent manifest now carries:

- `canonical_construction_stage_receipts`: audit records for parent construction receipts.
- `canonical_science_summary`: compact WSE/offset/backbone/authoritative-lock diagnostics copied from parent stage receipts.
- `canonical_science_summary_policy`: explicit policy that AOI exports are read-only and may not recompute science diagnostics.

The run summary now reads `canonical_science_summary` first, then falls back to retained stage receipt paths. This keeps AOI export reports useful even when the run only exported from an existing parent.

## Debugging rule

If the AOI summary reports a bad WSE/offset/backbone diagnostic, the first wrong artifact is in the canonical parent chain, not the AOI export. Fix the parent construction stage that produced the receipt; do not patch the AOI DEM or recompute diagnostics in the export path.
