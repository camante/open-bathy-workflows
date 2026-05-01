# REVIEW NOTES v900 — Bundle J

Bundle J adds an explicit execution-mode receipt for the active river workflow.

## Goal

Make the canonical-build versus AOI-export-only distinction explicit and auditable. Cache reuse is treated as an implementation detail; the architecture remains canonical parent plus exact AOI export.

## Changes

- Added `pipeline/river_workflow/river_workflow_execution_receipt.py`.
- The canonical-build route writes `river_execution_mode_receipt.json/txt` with `effective_mode=canonical_build_then_export`.
- The AOI-export-only route writes the same receipt with `effective_mode=aoi_export_only` and fails if construction-stage receipts are present.
- The receipt is copied into both `river_workflow/manifests/` and top-level `reports/`.
- Added synthetic tests for final DEM copy-only materialization and execution-mode receipt validation.

## Remaining work

The next cleanup bundle should focus on final deletion/hygiene: remove old `linear` manifest filenames, finish renaming remaining compatibility filenames, and run the final ambiguity grep pass.
