# workflow_v888 review notes

Target invariant: one canonical river parent for the shared solve domain, AOI outputs as exact parent-window exports, final DEM copied once from the AOI export, and final comparison products built as review sidecars.

Changes in this package:

1. Replaced `compare.sh` with the five-AOI final comparison script.
2. Added the same script under `final_compare.sh` and retained the source name `compare_5aois_union_final_products_parent_ref_v12.sh`.
3. Patched the compare script to accept either `887` or `v887` style version tags without producing `vv887` output folders.
4. Tightened `tools/compare_aoi_exports.py` so the final DEM source check fails if `final/river_workflow_receipt.json` does not explicitly record `final_dem_source: aoi_export_dem`.

Verification performed here:

- `bash -n compare.sh`
- `bash -n final_compare.sh`
- `bash -n compare_5aois_union_final_products_parent_ref_v12.sh`
- AST syntax parse of modified/inspected Python river workflow modules.

Full runtime validation still needs to be run in the CUDEM environment because it depends on GDAL/rasterio/perspecto/CUDEM data and local caches.
