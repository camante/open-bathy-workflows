# Open Bathy Workflows — General Workflow Guide

> **Authoritative file list:** every run writes `io_manifest.json` and `io_manifest.md` into `--out-dir`.
> Use those manifests, together with `bathy_report.json` and `unified_bathy_report.json`, as the source of truth for exact inputs, outputs, and paths.

This repository produces coastal and river bathymetry from a single orchestration layer.
The code is centered on `bathy_main.py`, which can run three families of work:

- **SDB** from Sentinel-2, ICESat-2, and optional external XYZ soundings
- **River bathymetry** from hydrography, DEMs, masks, cross-sections, skeleton interpolation, and optional hydraulic constraints
- **Fusion** into a combined deliverable raster

## What is current in this snapshot

The current workflow is no longer best described as separate experimental pieces.
It behaves as an integrated production-style pipeline with these important defaults and guardrails:

- `--methods=sdb,river,fuse` by default
- `--river-method=hybrid` by default, meaning XS inference is focused on the mainstem while the skeleton method fills the broader river domain
- explicit artifact tracking through `bathy_report.json`, `unified_bathy_report.json`, and `io_manifest.json`
- bounded SDB model-bank support for cross-tile consistency
- deterministic final domain clipping so deliverables do not persist outside the intended coastal and/or river water mask
- optional seam comparison against neighboring `io_manifest.json` files
- default output retention that keeps final deliverables plus run metadata while removing or relocating non-deliverable intermediates

## Core products

### 1) SDB

The SDB path uses `sdb_main.py` and related modules to:

- build or reuse optical composites
- assemble land/water masks
- ingest ATL03, ATL24, and optional external XYZ
- train or refresh the SDB model
- predict raster depth products
- write an SDB artifact manifest (`artifacts_sdb.json`) for downstream discovery

### 2) River bathymetry

The river path begins with hydrography preparation and domain masking, then runs one of three modes:

- `hybrid` — current default; uses mainstem XS inference plus skeleton fill elsewhere
- `skeleton` — channel-skeleton distance-transform method without XS generation
- `xs` — cross-sections everywhere; most flexible, but generally more artifact-prone at junctions

River products are constrained to the river/channel domain and can incorporate:

- external soundings
- optional authoritative bed blending
- optional 1D energy-solver priors
- optional SWOT RiverSP water-surface elevation anchoring
- optional drainage-area and Manning-based priors

### 3) Fusion

Fusion combines SDB and river results onto a common grid and records the decision path in the run report.
The workflow can sanitize problematic SDB zeros before fusion, enforce river-domain separation, and then clip final deliverables to the intended domain as a final safety step.

## Run template

```bash
python bathy_main.py \
  --aoi="-71.14/-71.10/42.75/42.78" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --methods=sdb,river,fuse \
  --river-method=hybrid \
  --out-dir=output/merrimack_example \
  --cache-root=cache \
  --extra-xyz-cudem=hydronos,ehydro
```

## External dependencies

A typical working environment includes:

- Python with the scientific/geospatial stack used by the repo
- GDAL / rasterio / PROJ support
- GeoPandas / Shapely / PyProj
- scikit-learn, NumPy, pandas, SciPy
- CUDEM command-line tools where relevant to your run
- network-enabled dependencies when fetching remote data products

Some CLIs support `--help` without all heavy runtime dependencies, but real pipeline execution still requires the full environment.

## Reports and metadata written by the workflow

At the end of a successful run, the most important metadata artifacts are:

- `bathy_report.json` — main orchestrator report with step status and explicit outputs
- `unified_bathy_report.json` and `unified_bathy_report.md` — compact whole-run summary
- `io_manifest.json` and `io_manifest.md` — explicit input/output path manifest
- `run_logs/flight_recorder_*.jsonl` — structured execution trace
- `run_logs/run_summary_*` — machine, technical, scientific, and human summaries

Additional receipts may be written when those components run, including input receipts, soundings/channel receipts, reprojection receipts, and energy-solver receipts.

## Sign conventions and datums

Use the actual product metadata and report context when interpreting a raster.
The current repo documentation assumes:

- final depth rasters are generally **negative-down**
- bed rasters are elevations in the working DEM vertical reference unless an explicit conversion step is requested
- optional SDB vertical-datum conversion is explicit and recorded in reports and manifests

## Troubleshooting priorities

When a run looks wrong, inspect in this order:

1. `bathy_report.json` for step failures or missing outputs
2. `io_manifest.json` to confirm what files were actually used and written
3. `run_logs/` for flight-recorder and run-summary context
4. river or SDB subdirectories for component-specific receipts and diagnostics

## Verification before packaging changes

```bash
./verify_repo.sh
./ci_smoke.sh
```

The second script is stricter and is intended to catch committed cache/slop artifacts in addition to compile and unit-test failures.
