# Open Bathy Workflows (v1.3)

A Python workflow that generates bathymetry by combining:
- **Satellite‑Derived Bathymetry (SDB)** from Sentinel‑2 + ICESat‑2 / in‑situ training points
- **River bathymetry** from DEM + NHD river network cross‑sections / skeleton methods
- **Fusion** to produce a single combined depth product

> This repo version keeps scripts in `workflow/`. Most runs are executed from that directory.

## Quickstart

From the repo root:
```bash
cd workflow
python bathy_main.py \
  --aoi="-74.52/-74.23/40.23/40.52" \
  --start 2024-01-01 \
  --end   2026-01-01 \
  --out-dir output/nyc \
  --extra-xyz-cudem=hydronos,ehydro
```

### External dependencies
The workflow assumes you have:
- A working Python environment (often the `cudem` conda env)
- **CUDEM** CLIs available on PATH (e.g., `fetches`, `dlim`)
- **waffles** available on PATH (used for coastline / land‑water masks)
- GDAL/PROJ set up (vertical CRS transforms are optional but supported)

## What the pipeline does

### 1) Orchestration (`bathy_main.py`)
Parses arguments, validates configuration, sets up cache + output directories, and runs:
- **SDB pipeline** via `sdb_main.py`
- **River pipeline** via `river_network.py`, `xs_builder.py`, and `xs_infer_bathy_raster.py` (or `river_skeleton_bathy.py`)
- **Fusion** via `bathy_fusion.py` / `fusion.py`
It also writes a JSON run report (e.g., `output/<name>/bathy_report.json`).

### 2) SDB (`sdb_main.py`)
High‑level flow:
1. Builds/loads a land‑water mask (typically **waffles coastline**)
2. Acquires Sentinel‑2 scenes (STAC), applies date/cloud QC, and builds composites
3. Loads ICESat‑2 (ATL03/ATL24) + optional **extra XYZ** points, aligns depths if needed
4. Samples training points (spatial sampling + optional spatial CV)
5. Trains the model (Random Forest + uncertainty helpers)
6. Predicts SDB depth tiles and writes outputs to `output/<name>/sdb/`

### 3) River bathymetry
There are two main river approaches in this codebase:
- **Cross‑section approach**: build XS lines, infer depths/beds from DEM + optional WSE/Manning constraints.
- **Skeleton approach**: a lighter‑weight, morphology‑driven estimator (`river_skeleton_bathy.py`).
Outputs generally land in `output/<name>/river/`.

### 4) Fusion (`bathy_fusion.py` / `fusion.py`)
Combines SDB and river rasters into a single product, optionally patching gaps and enforcing masks.
Final merged outputs are written to `output/<name>/combined/`.

## Outputs

- `output/<name>/sdb/` – SDB depth rasters, masks, QC plots, run_report JSON
- `output/<name>/river/` – river depth/bed rasters, GPKGs (network + XS) and diagnostics
- `output/<name>/combined/` – fused depth product (e.g. `bathy_depth_final_epsg4269.tif`)
- `output/<name>/bathy_report.json` – top‑level pipeline report

## Troubleshooting

- **S2 acquisition keeps failing**: try widening the date range, lowering `--cloud`, or clearing the S2 cache.
- **CRS / vertical datum errors**: make sure PROJ grid files are available; consider running without vertical transforms first.
- **Mask surprises (land vs water)**: waffles coastline masks typically use **water=0, land=1**; the workflow expects that convention.
- **Cross‑section artifacts**: intersecting cross‑sections can cause interpolation artifacts; fix XS generation parameters or post‑filter XS intersections.

## Where the scripts live

All scripts in this zip are intended to live under `workflow/`. The detailed per‑script reference is in `README_SCRIPTS_DETAILED.md`.
