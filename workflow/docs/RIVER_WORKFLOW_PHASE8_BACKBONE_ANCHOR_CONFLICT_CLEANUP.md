# River workflow phase 8: backbone anchor-conflict cleanup

## Goal

Keep the canonical parent/export/final architecture unchanged while making the final centerline bed backbone physically smoother downstream.

## Problem fixed

The prior backbone-rise limiter still allowed large downstream rises at observed-offset-supported points because those points were capped to a maximum adjustment. That kept observed-anchor conflicts visible, but it also preserved large local jumps in the guidance backbone. The final authoritative-lock stage already protects hard measured cells, so the backbone itself should remain smooth guidance rather than carrying large local discontinuities.

## Active behavior

The backbone stage now limits all final backbone points to the allowed local downstream rise before authoritative locking. If an observed-offset-supported point must be adjusted, the adjustment is not hidden: the point retains the raw formula value, final adjusted value, adjustment magnitude, support class, and an explicit `observed_offset_anchor_adjusted_by_backbone_rise_limiter` adjustment reason.

## Invariant preserved

This change does not alter the canonical parent/export/final route. It only changes the stage that creates `centerline_bed_backbone_points.gpkg`; AOI exports still subset from the canonical parent and final DEM materialization still uses the named AOI export artifact.
