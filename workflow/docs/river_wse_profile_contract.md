# River WSE Profile Contract (One-Path Implementation)

## Main invariant

The WSE stage is a canonical-parent longitudinal profile stage, not a local interpolation stage.

```
noisy measured bank-edge WSE support
-> one robust downstream-decreasing canonical WSE profile
-> centerline_wse_proxy_points.gpkg
```

Bed structure should come primarily from modeled offsets and the bed backbone:

```
wse_proxy_z_m - offset_modeled_m = bed_backbone_z_m
```

## Active WSE construction path

The active path is intentionally one-way:

1. Read canonical centerline points.
2. Read the canonical measured-only authoritative raster.
3. Extract measured bank-edge support from the canonical river/channel polygon boundary.
4. Project support to canonical centerline station.
5. Resolve downstream station orientation per component.
6. Build robust upstream/downstream WSE anchors from support near the component ends.
7. Build a simple decreasing base WSE trend between those anchors.
8. Fit a bounded, station-binned residual correction from noisy bank-edge support.
9. Enforce one final non-increasing downstream profile with PAVA.
10. Write `centerline_wse_proxy_points.gpkg`.

There is no separate interpolation/flatness/repair chain in the active WSE path; the old Phase-0 repair scaffold has been removed so future changes stay aligned with this one-path contract.

## Allowed WSE inputs

- canonical centerline points
- canonical river/channel polygon boundary
- canonical measured authoritative raster
- canonical solve grid metadata

## Forbidden WSE support sources

- local bed samples used as water-surface evidence
- broad centerline annulus samples that can capture upland/valley-side terrain
- AOI-local products
- final DEM or conditioned surfaces
- interpolated baseline values treated as WSE evidence

## Expected behavior

The WSE profile must be finite and non-increasing downstream. It may be smooth, simple, flat, or low-gradient when the bank-edge support does not defensibly support a larger drop. Local support can refine the broad trend through bounded residuals, but it cannot create an upstream-rising WSE profile.

## Required output columns

`centerline_wse_proxy_points.gpkg` must include:

- `point_id`
- `station_m`
- `station_downstream_m`
- `wse_support_z_m`
- `support_distance_m`
- `wse_profile_base_z_m`
- `wse_profile_residual_z_m`
- `wse_proxy_z_m`
- `wse_profile_method`
- `wse_confidence`
- `geometry`

## Validation failures

The WSE stage should fail only for construction-contract problems:

- missing canonical measured-only authoritative input
- no finite station values
- insufficient bank-edge support for a component
- invalid upstream/downstream anchors
- all-NaN WSE output
- downstream-increasing WSE after monotone enforcement
- missing required output columns

It should not fail just because WSE support is noisy, low-gradient, or because the final WSE profile is simple.

## Debugging receipt

`wse_proxy_receipt.json` summarizes:

- support source
- profile contract version
- support counts and ranges
- direction method per component
- anchor elevations and drop
- base slope
- residual method
- monotone violation counts before and after enforcement
- output finite WSE count

This receipt is the preferred WSE debugging artifact.

## Phase D handoff contract

The WSE proxy GPKG must now carry the downstream handoff columns used by the
offset/backbone stages:

- `station_downstream_m`
- `wse_proxy_z_m`
- `wse_profile_z_m`
- `wse_support_count`
- `wse_support_distance_m`
- `wse_profile_method`
- `wse_direction`
- `wse_confidence`

The validator treats the WSE profile as a reference surface: flat or
low-gradient profiles are allowed, but downstream increases are not. Bed
structure remains owned by the modeled-offset and backbone stages through
`observed_offset_m = wse_proxy_z_m - authoritative_bed_z_m` and
`bed_backbone_z_m = wse_proxy_z_m - offset_modeled_m`.

## Phase v848 clarification: broad canonical WSE profile grouping

The WSE profile is fit on the broad canonical river solution, not independently on every short reach. Per-reach fitting can make sparse bank-edge support appear insufficient even when the parent component/levelpath has enough support to define the intended simple downstream WSE profile. The active grouping for the WSE profile is therefore `component_id + levelpath_id` when available, with reach-level identifiers retained only as point attributes for downstream traceability.
