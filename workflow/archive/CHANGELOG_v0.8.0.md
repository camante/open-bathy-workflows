# Changelog v0.8.0 (integration patch)

## Added
- `waffles_bathy_interp.py`: waffles module wrapper (`bathy-interp`) that supports:
  - standalone SDB/river output generation (no authoritative datalist), and
  - intelligent gap-fill constrained to an authoritative waffles stack (only fills gaps).
- `WAFFLES_INTEGRATION.md`: step-by-step CUDEM waffles integration notes.
- `bathy_interp_cli.py`: convenience wrapper for standalone pipeline + optional gapfill using an authoritative raster.

## Notes
- The gap-fill wrapper uses `gapfill_intelligent.gapfill_depth_raster()` and then performs an explicit
  *authoritative clamp* as a final guard to ensure strict agreement where authoritative values exist.
