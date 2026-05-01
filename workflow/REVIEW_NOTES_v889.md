# REVIEW_NOTES_v889

Bundle A was applied as a behavior-preserving active-path naming pass.

## Main invariant protected

The active river workflow remains canonical parent solve -> AOI export -> final DEM materialization.
The final DEM must still come from the named AOI export artifact, and the compare script still checks AOI exports
against the canonical parent reference.

## Changes

- Added `river_workflow.py` as the public active river workflow façade.
- Added `river_workflow_context.py`, `river_workflow_paths.py`, `river_workflow_receipts.py`,
  `river_workflow_validation.py`, and `river_workflow_contract.py` as Bundle A naming/contract modules.
- Updated `river_workflow_entry.py` to call `run_active_river_workflow()` and `finalize_active_river_workflow()`.
- Updated `active_pipeline.py` to expose those active river function names, while retaining compatibility aliases
  for older imports.
- Updated stage names visible in the active workflow from `run_river_workflow` / `register_linear_outputs` to
  `run_river_workflow` / `register_river_outputs`.
- Updated `compare.sh`, `final_compare.sh`, and `compare_5aois_union_final_products_parent_ref_v12.sh` so users can
  run 2, 3, 4, or 5 AOIs with either positional count or `--aoi-count`.

## Compare usage

```bash
./compare.sh merrimack 889 2
./compare.sh merrimack v889 --aoi-count 5
./compare.sh merrimack 889 --aois 3 --river-dem-res-m 3
```

The fixed AOI order is: north, south, contained, downstream_overlap, upstream_overlap.

## Intentional remaining transitional items

- `pipeline/river_workflow/*` remains active internal implementation code for now.
- `active_river` report mirrors remain for backward compatibility.
- The on-disk work folder remains `river_workflow/` for this bundle to avoid breaking existing receipts and compare paths.

Those should be addressed in later bundles after the Bundle A naming layer is stable.
