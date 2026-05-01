# River primary surface contract

The river primary surface stage is a one-path handoff from the canonical bed backbone to the canonical solve-grid surface.

## Invariant

The stage must preserve the centerline/backbone bed profile at seed pixels, spread that profile only inside the canonical river corridor, taper laterally toward the WSE reference surface, and leave authoritative locking to the next stage.

## Inputs

- `river_centerline_bed_backbone_points.gpkg`
- `river_corridor_solve.tif`
- canonical solve grid template

## Construction

1. Rasterize centerline bed/WSE backbone points to seed pixels.
2. Build a continuous corridor surface from the centerline bed and WSE reference values.
3. Use lateral distance-to-centerline and distance-to-bank only to taper from backbone bed toward WSE.
4. Write exactly one `river_primary_surface_solve.tif`.

## Contract checks

The stage writes `river_primary_surface_contract.json` and fails if:

- the corridor is empty,
- the surface is empty,
- finite surface pixels exist outside the canonical corridor,
- no backbone seed pixel falls inside the corridor,
- in-corridor backbone seed pixels are changed by the surface construction.

This stage does not apply authoritative locking. The authoritative lock remains the next explicit stage.

## Corridor source rule

The active workflow must use canonical `linear_polygons` for the river corridor. Centerline-buffer corridor fallback is intentionally disabled so the active route remains one-path and polygon-bounded.
