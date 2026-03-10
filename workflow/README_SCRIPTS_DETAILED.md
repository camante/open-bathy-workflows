# Open Bathy Workflows — Script and Module Index

This file is a current grouped index of the repository modules.
It is intentionally organized by responsibility instead of trying to preserve outdated version labels or hard-coded output filenames.

## 1) Main entrypoints

### `bathy_main.py`
Unified coastal + river bathymetry orchestrator.
This is the main CLI for end-to-end runs and is the place where method selection, reporting, fusion, final domain clipping, seam comparison, and run-summary generation come together.

### `sdb_main.py`
Top-level SDB driver.
Builds or reuses optical inputs, ingests training points, trains the SDB model, predicts rasters, and writes SDB-side manifests and reports.

### `river_skeleton_bathy.py`
Channel-skeleton river bathymetry CLI.
Implements the non-XS river path and also contributes to the hybrid river workflow.

### `xs_builder.py`
Build cross-sections from the prepared river network.
Includes overlap trimming, densification, junction skipping, and deconfliction controls.

### `xs_infer_bathy_raster.py`
Infer river bathymetry from cross-sections and write a DEM-aligned bed raster.
Supports soundings, optional hydraulic priors, and auditable receipts for some constraint paths.

### `bathy_fusion.py`
Standalone multi-source fusion CLI for combining bathymetry rasters.

### `fusion.py`
Training-point fusion for ATL03, ATL24, and external XYZ on the SDB side.
This is different from raster fusion in `bathy_fusion.py`.

## 2) SDB modules

### `s2_optics.py`
Sentinel-2 discovery, filtering, compositing, mask support, and optical feature generation.

### `atl.py`
ICESat-2 acquisition, cache-first reuse, photon/bathymetry extraction, and extra XYZ ingestion utilities.

### `train.py`
Model training, metadata writing, and training diagnostics for the SDB stack.

### `predict.py`
Main SDB raster prediction engine.

### `predict_parallel.py`
Parallel tile processing for SDB prediction.

### `predict_chunked.py`
Chunked prediction helper for memory-constrained cases.

### `vis.py`
Visual QC utilities for ATL03 tracks, training points, and prediction artifacts.

### `alignment.py`
Post-prediction residual alignment of an SDB raster to reference tie points.

### `sdb_uncertainty.py`
Uncertainty-aware SDB utilities and supporting calculations.

### `bottom_physics.py`
Physics-based SDB helpers and inversion routines.

### `physics_integration.py`
Integration layer for physics-informed SDB enhancements.

### `kd_estimation.py`
Water-clarity and depth-penetration estimation helpers.

### `training_diversity.py`
Analyze and improve spatial and spectral diversity of training data.

### `spatial_sampling.py`
Adaptive spatial sampling logic for SDB training-point selection.

### `spatial_cv.py`
Spatial cross-validation utilities.

## 3) River modules

### `river_network.py`
Acquire and prepare river hydrography for downstream bathymetry work.

### `river_domain_mask.py`
Construct the river/channel domain raster used to restrict river deliverables.

### `river_wse.py`
Fit or smooth longitudinal water-surface elevation profiles along river stationing.

### `manning_inversion.py`
Manning-based depth estimation and prior logic.

### `usgs_nwis.py`
USGS NWIS metadata and discharge helpers.

### `swot_riversp_fetch.py`
SWOT RiverSP fetch helper.

### `xyz_constraints.py`
Helper logic for external XYZ constraints.

### `river_diagnostics.py`
River-specific diagnostics and unified report generation.

### `river_report.py`
Structured river report object for recording river-stage inputs, outputs, diagnostics, and errors.

### `regression_metrics.py`
Metrics utilities for reviewing river outputs across runs.

## 4) Fusion, seams, and diagnostics

### `seam_metrics.py`
Compute seam metrics between adjacent raster tiles.

### `seam_compare.py`
Wrapper/CLI for seam comparison workflows.

### `run_summary.py`
Generate machine, technical, scientific, and human-readable run summaries.

### `flight_recorder.py`
Structured execution trace logging for whole runs.

### `log_report.py`
Lightweight SDB-side reporting utilities.

## 5) Shared infrastructure and utilities

### `constants.py`
Centralized constants and shared metadata defaults.

### `cache_utils.py`
Deterministic cache fingerprinting and metadata helpers.

### `checkpoints.py`
Checkpoint/resume support.

### `chunked_processing.py`
Memory-efficient raster chunk processing helpers.

### `validation.py`
Centralized validation logic for configuration and inputs.

### `region_resolver.py`
Resolve AOI to region/state-specific defaults or coefficients.

### `process_utils.py`
Small subprocess wrappers.

### `logging_config.py`
Centralized logging setup.

### `plot_utils.py`
Lazy matplotlib setup for headless-safe plotting.

### `deps.py`
Dependency import helpers and clear runtime requirement checks.

### `errors.py`
Custom error hierarchy.

### `contract_tests.py`
Contract checks for expected behavior and structure.

### `gapfill_intelligent.py`
Gap-fill helpers for post-fusion refinement.

### `vdatum_utils.py`
Vertical datum conversion helpers.

### `compat_pandas.py`
Compatibility shim for pandas / GeoPandas version mismatches.

## 6) Internal packages

### `core/`
Small shared helpers for command construction, execution, hashing, JSON I/O, and path handling.

### `geo/`
Shared raster operations and geospatial helpers used across the workflow.

### `pipeline/`
AOI and orchestration-oriented helpers.

## 7) Shell helpers and tests

### `verify_repo.sh`
Standard offline verification runner.

### `ci_smoke.sh`
Stricter repository hygiene and unit-test gate.

### `run_smoke.sh`
Minimal smoke test wrapper used by `verify_repo.sh`.

### `smoke_test.py`
Small smoke-test helper module.

### `tests/`
Current lightweight unit and smoke tests for:

- compile checks
- CLI `--help` wiring
- energy-solver gating and fallback behavior
- effective Manning-n region mapping

## 8) Seam-stability package

### `seam_stability/`
Documentation, architecture figures, and standalone seam-analysis scripts used to evaluate cross-tile and transition stability beyond the basic in-run seam comparison hooks.

## 9) Reading order for new users

If you are new to the repo, start with:

1. `README.md`
2. `README_GENERAL_WORKFLOW.md`
3. `README_WORKFLOW_TECHNICAL.md`
4. `README_WORKFLOW_DETAILED.md`
5. the main entrypoints listed in section 1

If you are debugging a specific run, inspect the run reports and manifests first before diving into script internals.
