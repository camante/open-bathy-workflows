# Cleanup v856 notes

This pass fixes end-of-run finalization issues after the canonical river workflow has already produced valid AOI exports.

Changes:

- Replace the placeholder traceability manifest with an active canonical-river export validator.
- Treat `combined/DEM_enhanced.tif` as the required final-route artifact for the active river path.
- Use retained `aoi_export_identity.json` when available to confirm canonical-parent AOI export identity.
- Allow `workflow_run_overview.write_run_overview()` to support the active caller form: `out_dir=..., report=...`.

This pass does not change WSE, offset, backbone, surface, authoritative lock, or AOI export science logic.
