# SDB Guidance Products

SDB outputs are guidance-first artifacts for support-aware terrain generation.
The dense SDB depth raster may still be produced for diagnostics, but it is not a peer authoritative terrain surface.

Primary SDB guidance artifacts:
- `*_guide_points.gpkg` — spatially thinned pseudo-soundings with confidence/guidance weights
- `*_guidance_weight.tif` — soft guidance weight
- `*_trusted_interior.tif` — trusted export region for SDB guidance
- `*_admissibility.tif` — optical soft-guidance domain where SDB may influence interpolation
- `*_lower_bound.tif` — uncertainty-derived plausible lower bound
- `*_upper_bound.tif` — uncertainty-derived plausible upper bound
- `*_confidence.tif` — per-pixel confidence
- `*_provenance.tif` — diagnostic provenance

The dense `*_depth*.tif` SDB raster should be treated as `diagnostic_only` in workflow reasoning and reporting.
