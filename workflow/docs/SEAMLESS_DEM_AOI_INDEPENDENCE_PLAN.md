# Seamless DEM AOI-Independence Plan

## Core invariant

Build one canonical parent DEM for the river/coastal guidance system, then export each user AOI as an exact deterministic subset of that canonical parent. Adjacent or nested AOIs must not independently solve, blend, condition, crop, or mutate the final DEM in a way that can change overlapping pixels.

## Product roles

1. `canonical_parent_dem` is the scientific solve product. It is built on the canonical solve grid and is independent of the requested user AOI export window.
2. `aoi_export_dem` is an exact export/subset from `canonical_parent_dem`. It is not a new solve.
3. `final_user_dem` is the stable user-facing materialized file at `combined/DEM_enhanced.tif`.
4. `combined/DEM_enhanced.tif` must come from `aoi_export_dem` through the final DEM materializer.
5. Any display-only crop, visualization raster, reprojection, or convenience copy must use a different role/name and must not be treated as the scientific final DEM.

## Allowed final route

```text
canonical_parent_dem -> aoi_export_dem -> combined/DEM_enhanced.tif
```

Bundle A only locks the final materialization contract. Until the later parent/export split is implemented, the active river output is treated as the named AOI export artifact and must still pass through the final materializer before it becomes `combined/DEM_enhanced.tif`.

## Disallowed final-route behavior

- Do not write `combined/DEM_enhanced.tif` directly from AOI-local terrain interpolation.
- Do not write `combined/DEM_enhanced.tif` directly from blending/conditioning stages.
- Do not silently return an internal river raster as the final user DEM.
- Do not suppress missing-source or failed-materialization errors.
- Do not use final-output path names that make an internal diagnostic raster look like the final scientific DEM.

## Receipts required by Bundle A

The final materialization receipt must state:

```json
{
  "stage": "final_dem_materialization",
  "writer_role": "final_dem_materializer",
  "source_role": "aoi_export_dem",
  "destination_role": "final_user_dem",
  "aoi_local_solve": false,
  "terrain_interpolation": false,
  "blending": false,
  "authoritative_locking": false
}
```

## Later bundle targets

- Resolve a stable canonical system identity before scientific raster generation.
- Build or reuse one canonical parent DEM for the full river/coastal guidance system.
- Export AOIs as exact parent-grid subsets.
- Compare overlapping AOI exports in parent-grid coordinates.
- Enforce a single writer for `combined/DEM_enhanced.tif`.
- Reject stale canonical parent cache products whose receipts do not match the current canonical identity and source fingerprints.
- Produce one consolidated run summary that reports canonical identity, parent/export/final paths, identity checks, and single-writer status.

## Bundle B update — canonical system identity receipt

Bundle B adds a first-class canonical system identity before downstream river raster stages. The identity is intended to be stable for adjacent or nested AOIs that belong to the same canonical river solve. The user AOI is retained in receipts only as export metadata, not as an ingredient in the canonical identity hash.

The canonical identity receipt is written to:

```text
river_workflow/receipts/canonical_system_identity_receipt.json
```

The receipt records:

```text
canonical_system_id
source_system_id when available
user_aoi as export-only metadata
canonical_domain_bounds
river_network_fingerprint
authoritative_source_fingerprint
canonical_trace_distance_km
projected_crs
target_resolution_m
included_user_aoi_in_id=false
included_out_dir_in_id=false
included_timestamp_in_id=false
included_temporary_directory_in_id=false
included_run_id_in_id=false
```

The intended invariant after Bundle B is:

```text
No scientific river raster stage should run before the canonical system identity has been resolved and recorded.
```

## Bundle C implementation note: parent/export split

Bundle C separates the river final DEM stage into two explicit products:

1. `canonical_parent_dem` is the scientific parent product on the canonical solve grid. It is built from the canonical authoritative support, baseline background, authoritative-locked river guidance, and canonical take mask. It is not cropped to the user AOI and it does not write `combined/DEM_enhanced.tif`.
2. `aoi_export_dem` is the user AOI export from the canonical parent DEM. It is written by exact parent-grid subset logic and records `pixel_values_modified=false`, `resampled=false`, and `reprojected=false` in its receipt.
3. `combined/DEM_enhanced.tif` remains the materialized user-facing DEM from Bundle A and must be sourced from `aoi_export_dem`.

The intended route is therefore:

```text
canonical_parent_dem -> aoi_export_dem -> combined/DEM_enhanced.tif
```

No AOI-local interpolation, blending, or authoritative re-locking should happen after `aoi_export_dem` is produced.

## Bundle D implementation note: AOI identity and single-writer enforcement

Bundle D turns the parent/export route into a testable invariant.

The AOI identity check compares `aoi_export_dem` back to the exact parent-grid window inside `canonical_parent_dem`. It does not reproject, resample, or compare by visual appearance. It requires CRS, resolution, and transform alignment, then compares pixel values directly in parent-grid coordinates.

The identity receipt is written to:

```text
river_workflow/receipts/aoi_identity_receipt.json
```

The receipt records:

```text
stage=aoi_identity
check=export_vs_parent
checked_against_parent=true
parent_dem
export_dem
export_window
grid_aligned
mismatch_pixels
max_abs_diff
first_mismatch
passed
```

The final DEM single-writer check uses the materialization receipt for `combined/DEM_enhanced.tif` and requires exactly one final writer:

```text
writer_role=final_dem_materializer
source_role=aoi_export_dem
destination_role=final_user_dem
```

The intended invariant after Bundle D is:

```text
If an AOI export is shifted, resampled, locally modified, or written by more than one final route, the workflow should fail clearly instead of producing a visually detected seam later.
```

## Bundle F — Consolidated run summary and trace roles

Bundle F adds one consolidated summary for the seamless DEM route so a run can be checked from a single report instead of scattered diagnostic files. The summary is role-based and must report the exact route:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

The summary files are:

```text
reports/run_summary.json
reports/run_summary.txt
```

The summary records canonical system identity, canonical parent DEM, AOI export DEM, final user DEM, final DEM source role, cache status, export-vs-parent identity status, final DEM single-writer status, authoritative lock status when available, and warnings/errors for missing or failed seamless-route checks.

## Bundle G — River science receipts for WSE, offset, and backbone evidence

Bundle G strengthens the canonical parent DEM evidence chain without changing AOI export behavior. The AOI export remains an exact subset of the canonical parent, while the river stages now record compact science evidence in their existing stage receipts and consolidated run summary.

The receipt evidence covers:

1. WSE proxy: candidate downstream direction, expected WSE trend, raw/proxy WSE ranges, monotone violation count, monotone adjustment count, largest adjustment, flat segment count, and quality-flag counts.
2. Observed offset: authoritative bed sample count, valid observed offset count, observed offset min/max/median, rejected sample count, source formula, and explicit confirmation that bank samples are not treated as authoritative bed.
3. Modeled offset: modeled offset range/median, global observed median, global prior, support mode (`observed_supported`, `sparse_supported`, or `low_support_inferred`), support-class counts, source counts, and unsupported/global-prior point count.
4. Backbone: formula `bed_backbone_z_m = wse_proxy_z_m - offset_modeled_m`, bed range/median, downstream trend violations, abrupt step count, largest step, adjustment count, largest adjustment, and support-class counts.

This is deliberately receipt/summary evidence, not a new diagnostic tree. The purpose is to make the canonical parent DEM scientifically reviewable from one run summary while preserving the simple parent/export/final route.

## Bundle H — route cleanup and AOI-local mutation retirement

Bundle H keeps the active route narrow and removes method-specific ambiguity from the user-facing workflow story.

The supported active route is:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

The final user DEM remains:

```text
combined/DEM_enhanced.tif
```

The only active writer allowed for that file is the final DEM materializer. Terrain interpolation, final conditioning, comparison-folder creation, legacy/v2 compatibility helpers, and display-product generation must not overwrite or locally mutate `combined/DEM_enhanced.tif`.

Historical labels such as `linear_v1`, `v1`, `simple_v2`, `v2`, and `shared_solve` are compatibility aliases only. They normalize to the single built-in `river_workflow` route and must not select different runtime code paths.

Product roles are explicitly separated in `pipeline/output_products.py`:

- `canonical_parent_dem`: scientific parent product on the canonical solve grid.
- `aoi_export_dem`: exact AOI subset of the canonical parent product.
- `final_user_dem`: stable materialized AOI export at `combined/DEM_enhanced.tif`.
- `display_dem`: optional visual/comparison product, not the scientific final DEM.
- `diagnostic_dem`: troubleshooting product only.
- `internal_conditioning_dem`: internal terrain-generation product only.

Any future output registration should use these roles rather than ambiguous names such as “final,” “enhanced,” or method-specific river labels.

## Active final-product reprojection rule

`combined/DEM_enhanced.tif` is the scientific AOI export materialized from `aoi_export_dem`. It must not be reprojected or resampled in place. Any alternate-SRS, hillshade, comparison, or display product must be written as a sidecar/display product and must not replace `combined/DEM_enhanced.tif`.

## Phase 1 materialization fix

The active route must not pass `final_out_srs`, CRS-match callbacks, or warp callbacks into `materialize_final_dem_from_aoi_export()`. The scientific final DEM is a copy-only materialization of the AOI export. If a comparison or display raster is needed in another SRS, it must be generated by a separate sidecar step after `combined/DEM_enhanced.tif` exists and must not participate in AOI identity checks.

## Phase 1b integration fix: materializer traceability contract

The final DEM traceability contract must recognize the new single-writer route:

```text
AOI export DEM -> final_dem_materializer -> combined/DEM_enhanced.tif
```

`combined/DEM_enhanced.tif` is still copy-only and must not be reprojected or resampled. The materializer now writes the final DEM touch-log action `final_dem_materializer_write`, and post-run identity verification may record `active_river_existing_output_verify` as a verify-only action. Legacy `debug_final_route_*` files are optional for this route because the old final-route writer is no longer the owner of the active river final DEM.

## Phase 2 identity cleanup

Canonical river-system identity is now intentionally path-free. It excludes user AOI, export AOI, output directory, run ID, temporary directories, AOI-local authoritative/base raster paths, and baseline paths. Resolution is snapped to the canonical resolution policy before identity hashing so neighboring AOIs with slightly different raw source-grid resolution estimates still resolve to the same canonical river system when they share the same canonical solve domain.

Authoritative raster content remains part of parent DEM cache validation, not the river-system ID. This preserves the separation between “which river system is being solved” and “which measured authoritative source was used to build the parent DEM.”

## Phase 3 comparison contract

`compare.sh` is part of the debugging contract. After the north and south AOI runs finish, it must compare the role-based run summaries rather than infer success from legacy stage summaries. The comparison is now explicit and reports:

- whether north and south resolved the same `canonical_system_id`;
- whether north and south used the same `canonical_cache_key`;
- whether each final DEM came from `aoi_export_dem`;
- whether each AOI export passed export-vs-parent identity;
- whether each final DEM passed the single-writer check;
- whether the final user DEMs match across their common parent-grid overlap.

The command should end with one readable comparison report under `output/<name>_v<version>/comparison.txt` and a machine-readable JSON report under `output/<name>_v<version>/comparison.json`. This avoids inspecting many individual receipts to answer the basic AOI-independence question.
