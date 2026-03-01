# Open Bathy Workflows – Technical Workflow Guide


> **Authoritative file list:** Each run writes `io_manifest.json` and `io_manifest.md` into your `--out-dir`.
> Use those manifests (and `unified_bathy_report.json`) as the *only* source of truth for exact input/output filenames and paths.
> Do **not** rely on any “canonical” filenames in docs; outputs can vary by enabled methods and configuration.

This repository runs an end-to-end, **reproducible** bathymetry workflow that can generate:

- **Coastal / nearshore bathymetry from Satellite-Derived Bathymetry (SDB)** (Sentinel‑2 optical + calibration/constraints)
- **River channel bathymetry** using a **skeleton / width‑proxy** approach with optional constraints (soundings, slope/curvature, WSE)
- **A fused/combined product** when both modes run

The design goal is **hydrologically safe, domain‑restricted outputs** (no bathy outside intended water domains), with deterministic masking and conservative post‑processing.

---

## Coordinate systems and sign conventions

- Depth products are **positive down** (meters) where applicable.
- Bed elevation products are **NAVD88-referenced** where explicitly named `*_navd88_*`.
- Final deliverable filenames are not guaranteed/canonical. Treat `io_manifest.json` and `unified_bathy_report.json` as the source of truth for exact output paths.

> Tip: Always verify the input DEM vertical datum and units before interpreting output elevations.

---

## High-level pipeline (bathy_main.py)

`bathy_main.py` is the orchestrator. It runs one or more methods based on `--methods`:

- `--methods=sdb` runs SDB only
- `--methods=river` runs river only
- `--methods=sdb,river` runs both, then fuses outputs

Each stage writes intermediate products into `cache_root/...` and final products into `out_dir/...`.

### Stage A — AOI and region resolution
- Parses `--aoi W/E/S/N`
- Sets template grid / resolution
- Creates run folder structure and logging

### Stage B — Masks / domains (critical)
Two domains are maintained intentionally:

1) **SDB domain (coastal/ocean water)**  
   Derived from **waffles coastline masks**. Convention:
   - water = 0
   - land = 1

2) **River domain (rivers only)**  
   Derived from **NHDArea polygons** rasterized to a channel mask:
   - inside river polygons = 1
   - outside = 0

#### Final domain clipping policy (mode-dependent)

After the pipeline completes, `bathy_main.py` applies a deterministic clipping policy:

- **SDB only** → clip final combined outputs to **waffles coastline (ocean-only)**
- **River only** → clip river outputs to **NHDArea channel mask**
- **SDB + River** → clip combined + river outputs to **waffles coastline with NHD**

This is implemented by `_apply_final_domain_policy(...)` in `bathy_main.py`.

This policy is specifically intended to prevent:
- stray bathymetry in lakes/land
- outputs outside the intended AOI
- inconsistencies when only one mode runs

### Stage C — River bathymetry (river_skeleton_bathy.py)
The river method:
- builds a **river channel mask** (`river_domain_mask.py`)
- computes a **channel skeleton**
- uses **width proxy** and WSE smoothing controls to avoid junction artifacts
- optionally incorporates:
  - soundings (as constraints / priors)
  - bed profile constraints (max slope/curvature)

Outputs include:
- River depth output path (see `io_manifest.json` for exact filename)

### Stage D — SDB bathymetry (sdb_main.py)
SDB uses:
- Sentinel‑2 reflectance / water column signal (see `s2_optics.py`)
- training & prediction utilities (`train.py`, `predict.py`, `predict_chunked.py`)
- optional uncertainty characterization (`sdb_uncertainty.py`)
- optional ICESat‑2 constraints (ATL utilities in `atl.py`) depending on your run configuration

Primary outputs are fed into fusion when both modes run.

### Stage E — Fusion (fusion.py / bathy_fusion.py)
When both modes are enabled, the workflow creates a combined depth surface. The current design prioritizes:
- correct nodata handling
- domain clipping by the policy above
- consistent warping to final CRS

---

## Key final outputs to inspect

Depending on the run mode(s), the main rasters you should inspect:

### Always (combined folder)
- `combined/` (see `io_manifest.json` for exact filenames)  
  Final bathymetry depth (meters, +down), warped to EPSG:4269.

If produced:
- `combined/` (see `io_manifest.json` for exact filenames)  
  Final bed elevation (NAVD88), warped to EPSG:4269.

### River outputs (river folder)
- River depth output path (see `io_manifest.json`)

---

## Metrics / regression runs

`regression_metrics.py` computes a per-run `metrics_summary.csv`. It expects a channel mask and now:
- skips non-run folders like `logs/`
- can read `bathy_report.json` to locate cached channel masks when not present in the run directory

---

## What was verified in this review

This review confirmed (by code inspection and executable checks):

- All `*.py` files compile (no indentation/syntax errors).
- `bathy_main.py`, `river_skeleton_bathy.py`, and `regression_metrics.py` support `--help` successfully.
- `sdb_main.py --help` was fixed (it previously crashed due to a stray `%` in a help string).
- Mode-dependent domain clipping is implemented by `_apply_final_domain_policy(...)` and invoked after `_ensure_output_contract(...)`.

> Note: Full numerical/physics validation requires running on real data. The checks above confirm **control flow, argument parsing, file naming, and clipping logic** behave as documented.

---

## Reproducible run template

Example (both modes):

```bash
PYTHONUNBUFFERED=1 python -u bathy_main.py \
  --aoi="-71.25/-71.00/42.75/43.00" \
  --start="2025-01-01" --end="2026-01-01" \
  --methods=sdb,river \
  --out-dir="output/my_run" \
  --cache-root="cache/my_run"
```

---

## Troubleshooting checklist

1) **Outputs outside domain**  
   Confirm the correct mask exists:
   - waffles coastline masks in `.../masks/`
   - `river_channel_mask.tif` exists in the cache river folder
   Then confirm `--methods` matches the clipping rule you expect.

2) **Metrics summary “No channel mask found”**  
   Ensure each run folder has `bathy_report.json` or that the cached mask still exists.

3) **Junction artifacts**  
   River junction handling is controlled in `river_skeleton_bathy.py` (width-proxy mainstem preservation + WSE-only junction smoothing). Check the debug masks (if enabled) to verify mainstem corridor coverage.
