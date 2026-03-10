# Open Bathy Workflows — Detailed Workflow Guide

This document walks through the current end-to-end behavior of the workflow and is intended to stay aligned with the actual repository state rather than older filename conventions.

## 0) First principles

Three rules matter throughout this repository:

1. **Use explicit manifests and reports.**
   Exact output filenames are not assumed to be canonical across all run modes.

2. **Treat final domain clipping as a safety invariant.**
   Coastal and river deliverables should not survive outside their intended water domain.

3. **Interpret final depth rasters as negative-down unless the product explicitly says otherwise.**

## 1) Main entrypoint

The orchestration entrypoint is `bathy_main.py`.
Its current default behavior is to run:

- SDB
- river bathymetry
- fusion

with `--river-method=hybrid` unless the user overrides it.

Example:

```bash
python bathy_main.py \
  --aoi="-71.14/-71.10/42.75/42.78" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --methods=sdb,river,fuse \
  --river-method=hybrid \
  --out-dir=output/example_run \
  --cache-root=cache
```

## 2) Repository layout

The repo is organized around a few major layers:

- orchestration and fusion: `bathy_main.py`, `bathy_fusion.py`, `fusion.py`
- SDB processing: `sdb_main.py`, `s2_optics.py`, `atl.py`, `train.py`, `predict.py`, `predict_parallel.py`, and `predict_chunked.py`, `vis.py`
- river processing: `river_network.py`, `river_domain_mask.py`, `xs_builder.py`, `xs_infer_bathy_raster.py`, `river_skeleton_bathy.py`
- hydraulics and constraints: `manning_inversion.py`, `river_wse.py`, `usgs_nwis.py`, `swot_riversp_fetch.py`, `xyz_constraints.py`
- reporting and diagnostics: `run_summary.py`, `river_diagnostics.py`, `river_report.py`, `seam_metrics.py`, `seam_compare.py`, `flight_recorder.py`
- shared infrastructure: `core/`, `geo/`, `pipeline/`, `cache_utils.py`, `logging_config.py`, `process_utils.py`

## 3) Stage-by-stage flow

### Stage A — argument parsing, AOI handling, and run scaffolding

`bathy_main.py` parses the AOI, method list, cache/output locations, and optional river/SDB controls.
It then creates run directories, logging, and flight-recorder state.

Important current behavior:

- explicit AOI helpers live under `pipeline/aoi.py`
- output retention is handled later so the run can prune non-deliverables from `--out-dir`
- run summaries and manifests are written at the end based on actual executed steps and observed outputs

### Stage B — SDB pipeline

If `sdb` is enabled, `bathy_main.py` calls into `sdb_main.py`.
That pipeline performs the optical side of the workflow:

1. prepare or reuse masks and Sentinel-2 composites
2. ingest ATL03 / ATL24 and optional extra XYZ
3. normalize and fuse training data
4. train the model and optional uncertainty helpers
5. predict SDB rasters
6. write `run_report.json` and `artifacts_sdb.json`

Supporting modules include:

- `s2_optics.py` for Sentinel-2 acquisition and compositing
- `atl.py` for ICESat-2 acquisition and point extraction
- `fusion.py` for training-point fusion
- `train.py`, `predict.py`, `predict_parallel.py`, and `predict_chunked.py` for model training and raster prediction
- `alignment.py`, `sdb_uncertainty.py`, `bottom_physics.py`, and `physics_integration.py` for optional refinement paths

### Stage C — river network and domain construction

If `river` is enabled, the workflow prepares the river side before choosing a bathymetry method.
Key steps include:

1. hydrography acquisition and preparation with `river_network.py`
2. river/channel domain raster creation with `river_domain_mask.py`
3. DEM preparation or auto-acquisition if configured
4. optional setup of soundings, drainage area, SWOT WSE, and hydraulic priors

The river/channel domain is operationally important because later deliverables are clipped to it.

### Stage D — river method execution

#### Hybrid method (current default)

The hybrid path focuses explicit XS inference on the mainstem and lets the skeleton method fill the broader river domain.
This is the workflow's current operational default because it balances continuity and geometric robustness better than using XS everywhere.

#### XS method

The explicit XS path uses:

- `xs_builder.py` to generate and deconflict cross-sections
- `xs_infer_bathy_raster.py` to infer the river bed and rasterize a DEM-aligned patch

This path supports optional soundings, Manning/drainage-area priors, and optional 1D energy-solver logic.
It can also write auditable receipts when those constraints are used.

#### Skeleton method

The skeleton path uses `river_skeleton_bathy.py` to derive a channel-bottom surface without XS generation.
It supports WSE smoothing, soundings, SWOT residual anchoring, optional authoritative-bed blending, and diagnostic receipts.

### Stage E — fusion

If `fuse` is enabled, `bathy_main.py` fuses the available SDB and river rasters under `out_dir/combined/`.
Current fusion behavior includes:

- fast-path handling when only one source is available
- optional sanitization of problematic SDB zero-valued regions before fusion
- alignment to a template grid
- domain-aware source separation so river and non-river areas do not bleed across the wrong mask
- provenance output alongside the fused raster where supported

### Stage F — final domain policy and post-processing

Near the end of the run, `bathy_main.py` applies final domain clipping rules.
This is intentionally late in the pipeline so it can act as a final safety net even if an upstream stage produced a broader raster than intended.

Optional additional late-stage tasks may include:

- SDB vertical-datum conversion when explicitly requested
- seam comparison against neighboring `io_manifest.json` files
- transition and seam receipts where those checks are enabled

### Stage G — reports, manifests, and summaries

The workflow writes multiple layers of metadata after execution:

- `bathy_report.json` — main orchestrator report
- `unified_bathy_report.json` and `.md` — whole-run summary
- `io_manifest.json` and `.md` — explicit paths observed during the run
- `run_logs/run_summary_*` — machine, technical, scientific, and human summaries
- `run_logs/flight_recorder_*.jsonl` — structured execution trace

These files are intended to replace old habits of inferring output locations from memory.

## 4) Output retention behavior

By default, the workflow keeps final deliverables and run metadata in `--out-dir` and removes or relocates non-deliverable intermediates.
If `--save-intermediates` is enabled, the run retains more artifacts under the configured intermediates directory.

This means the output directory is intentionally not a dump of every temporary file by default.
The manifest and reports are the stable way to know what survived the retention policy.

## 5) Important receipts and diagnostics

Depending on the exact path taken, the workflow may write:

- `input_receipt.json`
- `soundings_channel_receipt.json`
- `energy_solver_receipt.json`
- `*.reproject_receipt.json`
- `seam_comparisons.json`

These are best-effort audit artifacts and are especially useful when verifying a bugfix or reviewing a questionable result.

## 6) Verification and packaging

Before packaging a zip or committing changes, run:

```bash
./verify_repo.sh
./ci_smoke.sh
```

The second script is the stronger hygiene gate and will fail if the repo tree contains committed cache artifacts such as `__pycache__`, `.pytest_cache`, or `*.pyc` files.

## 7) Practical debugging order

When something is wrong, check in this order:

1. `bathy_report.json`
2. `unified_bathy_report.json`
3. `io_manifest.json`
4. `run_logs/`
5. component-specific receipts in `sdb/`, `river/`, or the cache tree
6. source-specific manifests such as `artifacts_sdb.json`
