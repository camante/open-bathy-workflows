River Guidance Products
=======================

River outputs are guidance-first artifacts that shape downstream terrain generation in fluvial gaps under authoritative control. Dense river depth and bed-elevation rasters remain diagnostic helpers and must not be treated as peer authoritative terrain.

Primary guidance artifacts
--------------------------
- river_guidance_weight.tif: soft influence weight inside the trusted export interior, zero at authoritative anchors and outside trusted export region
- river_trusted_interior.tif: trusted export region from the halo-domain river solve
- river_soft_guidance_domain.tif: broader soft-guidance region before removing authoritative anchors
- river_admissibility.tif: anchor-excluded subset admissible for soft river guidance
- river_guide_points.gpkg: sparse pseudo-soundings for confidence-weighted interpolation guidance
- river_authoritative_support.tif: rasterized authoritative anchor support
- river_authoritative_support_depth.tif: anchor depth used for exact overwrite
- river_corridor_mask.tif: river corridor guidance domain
- river_scaffold_domains.json: scaffold/solve/export AOI metadata
- river_scaffold_manifest.json: canonical scaffold product/cache lineage
- river_guidance_manifest.json: explicit manifest describing the guidance-first artifact family

Diagnostic-only products
------------------------
- river_depth.tif
- river_bed_elev.tif

These dense rasters may support diagnostics or later artifact derivation, but they are not peer terrain products to blend against authoritative data.
