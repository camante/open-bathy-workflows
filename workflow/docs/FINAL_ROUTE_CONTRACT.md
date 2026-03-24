# Final DEM Route Contract

The final DEM route is frozen around the following structural inputs only:

- authoritative aligned base
- authoritative gap / eligible fill masks
- support classes
- structured river guidance artifacts
- SDB guidance artifacts
- deterministic terrain interpolation
- optional plausible bounds and regime masks
- explicit provenance / support / confidence outputs

Forbidden structural inputs in the final route:

- legacy fused candidate raster
- dense river depth raster as peer surface
- dense SDB depth raster as peer surface
- weighted-overlap blended bathymetry as a structural input

River and SDB guidance manifests now self-describe which artifacts may structurally enter the final route and which are diagnostic-only.
