# Open Bathy Workflows: Detailed Workflow Guide (v2.0.42)

This document explains **what each stage does**, **what inputs it consumes**, **what it writes**, and the **guardrails** that enforce the intended scientific behavior.

The orchestration entrypoint is:

- `bathy_main.py` (run from the `workflow/` directory)

It can run:

- **SDB** (Sentinel‑2 + ICESat‑2 / extra XYZ)
- **River** bathymetry (cross‑sections *or* skeleton)
- **Fusion** (combine SDB + river into a single depth raster)

---

## 0) Conventions

### AOI
Most commands use an AOI bounding box:

- `--aoi="W/E/S/N"` (lon/lat degrees)

### Sign convention
Depth products are generally written **negative‑down** (more negative = deeper). Bed/elevation products are in the chosen vertical datum (typically NAVD88 or whatever your DEM is).

### Nodata
River rasters use `--river-nodata` (default `-9999`). Final river outputs are **hard‑clipped** so pixels outside the river/channel domain are nodata.

---

## 1) Repository layout

Typical top-level files (in this zip, everything lives in the same folder as `bathy_main.py`):

- `bathy_main.py` – orchestrator; writes `bathy_report.json`
- `sdb_main.py` – SDB driver
- `river_network.py` – hydrography/network acquisition & preprocessing
- `xs_builder.py` – builds cross-sections along river network
- `xs_infer_bathy_raster.py` – infers bed/depth raster from XS + DEM (+ optional WSE constraints)
- `river_skeleton_bathy.py` – alternative river method (morphology-driven)
- `river_domain_mask.py` – constructs the river/channel domain raster mask
- `bathy_fusion.py` – fuses SDB + river rasters

---

## 2) Running the workflow

### Minimal run (SDB + River + Fusion)

```bash
cd workflow
python bathy_main.py \
  --aoi="-74.52/-74.23/40.23/40.52" \
  --start 2024-01-01 \
  --end   2026-01-01 \
  --out-dir output/nyc \
  --methods sdb,river,fusion
```

### River-only run

```bash
cd workflow
python bathy_main.py \
  --aoi="-74.52/-74.23/40.23/40.52" \
  --start 2024-01-01 \
  --end   2026-01-01 \
  --out-dir output/nyc \
  --methods river
```

### Choose river method

- Cross‑sections: `--river-method xs` (default in many configs)
- Skeleton: `--river-method skeleton`

---

## 3) Outputs (what you get)

Within `--out-dir`, the workflow writes:

- `bathy_report.json` – machine-readable run report (commands, rc, tails)
- `sdb/` – SDB outputs (when run)
- `river/` – river outputs
- `combined/` – fused outputs (when run)

### River outputs (guaranteed river-only)

Inside `output/<name>/river/`:

- `river_bottom_navd88_patch.tif` – river bed elevation raster (same grid as working DEM)
- `river_depth_terrain_patch.tif` – river depth raster (negative-down) computed from bed and DEM
- `river_network.gpkg` – processed hydrography network
- `cross_sections.gpkg` – XS vectors (XS method)
- `river_channel_mask.tif` – river/channel domain mask raster

**Hard guarantee:** both `river_bottom_navd88_patch.tif` and `river_depth_terrain_patch.tif` are **hard‑clipped** to `river_channel_mask.tif` at the end of the river stage. Pixels outside the channel domain are nodata.

---

## 4) Stage-by-stage behavior

### 4.1 Orchestration (`bathy_main.py`)

Responsibilities:

1. Parse CLI args; build a config object
2. Create cache & output subfolders
3. Run selected stages (`--methods`)
4. Capture return codes and output tails in `bathy_report.json`

Key logic that matters scientifically:

- Ensures the **river-only** postcondition by clipping final river rasters to the channel domain mask.
- Passes the channel mask into the XS inference stage when available.

---

### 4.2 SDB stage (`sdb_main.py`)

High-level steps:

1. Acquire Sentinel‑2 scenes over AOI/time window
2. Apply cloud/QC filtering
3. Build reflectance composites
4. Load training points:
   - ICESat‑2 photons/products (ATL03/ATL24) depending on settings
   - optional extra XYZ (user-provided)
5. Train a model (typically Random Forest)
6. Predict raster tiles and write outputs

Outputs land in `output/<name>/sdb/`.

Masks:

- Uses waffles coastline / land-water masks to limit where SDB is allowed.

---

### 4.3 River stage overview

The river stage begins with **hydrography/network** creation and **DEM prep**, then runs either the **XS** method or the **skeleton** method.

Common prerequisites:

1. `river_network.py` generates a network GPKG (and optionally NHDArea polygons/layers).
2. A river DEM is prepared (user-provided via `--river-dem` or auto-downloaded if enabled).
3. `river_domain_mask.py` generates a **river/channel domain mask** raster (`river_channel_mask.tif`).

#### River/channel domain mask (`river_domain_mask.py`)

This mask defines **where river bathymetry is allowed**.

Depending on `--river-channel-source`, it uses one of:

- `nhdarea` (polygons from NHDArea)
- `corridor` (buffer around centerline / corridor estimate)
- `auto` (best available)

Optional connectivity filter:

- When `--river-connectivity-filter` is enabled, the mask can be filtered to keep only connected components that represent the main connected water network (helpful for removing isolated puddles).

**Downstream contract:** any river bed/depth raster written by this workflow must be nodata outside this mask.

---

### 4.4 River: Cross-section method

#### Step A: Build cross-sections (`xs_builder.py`)

Inputs:

- `river_network.gpkg` (centerlines/reaches)
- `river_dem.tif` (working CRS/resolution)

Outputs:

- `cross_sections.gpkg`

Important controls:

- spacing, half-width, smoothing
- overlap trimming / global deconfliction
- junction skipping/snap/buffer parameters

These exist because **intersecting or overly dense XS** are a primary cause of interpolation artifacts.

#### Step B: Infer bed raster from XS (`xs_infer_bathy_raster.py`)

Inputs:

- `cross_sections.gpkg`
- `river_network.gpkg`
- `river_dem.tif`
- optional `--channel-mask-raster=river_channel_mask.tif`

Outputs:

- `river_bottom_navd88_patch.tif` (bed)

Key scientific guardrails implemented here:

1. **Main‑stem priority at confluences**
   - When multiple branches contribute control points in a local neighborhood, the interpolation prefers the **highest‑priority branch** (typically the larger stream order) so small tributaries do not imprint circular “bullseye” artifacts into the main channel.

2. **Thalweg spine construction uses thalweg points**
   - The longitudinal spine used for anisotropic interpolation is constructed from one representative thalweg point per XS (deepest predicted within that XS), not all interior points. This avoids spurious cross-channel spurs near junctions.

3. **Channel mask enforcement**
   - If `--channel-mask-raster` is supplied, inference is constrained to that domain.

After inference, `bathy_main.py` still performs a final **hard clip** to guarantee river-only outputs.

---

### 4.5 River: Skeleton method (`river_skeleton_bathy.py`)

This alternative produces a river bed estimate using a longitudinal skeleton approach. It is designed to be simpler and more stable where XS construction is difficult.

Inputs:

- `river_network.gpkg`
- `river_dem.tif`
- optional soundings XYZ

Outputs:

- `river_bottom_navd88_patch.tif`

It also produces/uses `river_channel_mask.tif`, and `bathy_main.py` hard-clips the final products.

---

## 5) SWOT WSE usage (what it does and does not do)

SWOT is used only as a **Water Surface Elevation (WSE) constraint** to stabilize/anchor the stage profile.

- **It is not used as a direct bathymetry predictor.**

Where it’s applied:

- XS method: `xs_infer_bathy_raster.py` can incorporate WSE observations when available.
- Skeleton method: `river_skeleton_bathy.py` can apply a smooth residual correction between a modeled WSE profile and SWOT WSE.

Robustness improvements in this repo:

- Uses neighborhood aggregation (KDTree) instead of a single nearest point when multiple SWOT WSE samples exist within `--swot-max-dist-m`.
- Applies robust MAD-based outlier rejection before computing a representative WSE.

If no valid SWOT WSE is available, the workflow continues without applying SWOT.

---

## 6) Fusion stage (`bathy_fusion.py`)

Fusion combines SDB and river rasters into a single product.

Typical behavior:

- Uses masks to prevent SDB from populating river-only regions and vice versa.
- Gap-fills where one method has nodata and the other has data.

Outputs land in `output/<name>/combined/`.

---

## 7) Verifying that outputs match the described behavior

These checks are designed to confirm that the code is doing what this README claims.

### 7.1 River-only guarantee

1) Confirm the mask exists:

```bash
ls -lh output/<name>/river/river_channel_mask.tif
```

2) Confirm river rasters are nodata outside mask:

```bash
gdal_calc.py \
  -A output/<name>/river/river_depth_terrain_patch.tif \
  -B output/<name>/river/river_channel_mask.tif \
  --calc="(B==1)*(A!= -9999)" \
  --NoDataValue=0 \
  --outfile=/tmp/river_outside_check.tif
```

If the workflow is behaving correctly, pixels where `B!=1` should not contain valid river values.

### 7.2 Confluence artifact suppression

Inspect several junctions at common scales and confirm:

- the main stem is continuous through the confluence
- tributary influence does not produce circular “rings” in the main stem

If rings persist, the first thing to revisit is XS generation (spacing, overlap trimming, junction skipping).

---

## 8) Practical run patterns

### River XS with conservative parameters

```bash
python bathy_main.py \
  --methods river \
  --river-method xs \
  --xs-spacing-m 250 \
  --xs-length-m 120 \
  --xs-deconflict-tol-m 3 \
  --xs-junction-snap-m 30 \
  --xs-junction-buffer-m 120 \
  --out-dir output/test_river
```

### Skeleton river with SWOT WSE (RiverSP vectors)

```bash
python bathy_main.py \
  --methods river \
  --river-method skeleton \
  --river-swot-riversp /path/to/riversp_points.gpkg \
  --river-swot-wse-field wse \
  --river-swot-max-dist-m 1500 \
  --out-dir output/test_swot
```

---

## 9) Troubleshooting by stage

- **Hydrography fetch fails**: check `--river-hydrography-source` and network access.
- **XS artifacts**: check `xs_builder` parameters first (overlap trimming, spacing, junction skipping).
- **SWOT does nothing**: confirm you provided a RiverSP vector with a numeric WSE field; check `bathy_report.json` for whether SWOT points were found.
- **Fusion has river outside rivers**: this should not happen if fusion consumes the clipped `river_depth_terrain_patch.tif`. If it does, verify fusion is using the correct river raster path.

---

## 10) Smoke tests (offline sanity checks)

This repo includes lightweight **offline** smoke tests intended to catch obvious breakages (syntax, basic geometry/math),
without requiring any external downloads or real rasters.

Run:

```bash
./verify_repo.sh
# or
./run_smoke.sh
# (compat wrapper)
./tests/run_smoke.sh
```

What is tested:

- **Compile-only:** compiles all `*.py` files (`tests/test_compile.py`)
- **River XS deconflict:** verifies intersecting cross-sections are dropped deterministically (`tests/smoke_test_river_xs.py`)
- **WSE profile fitting (SWOT-like):** verifies monotone-smoothed WSE fitting on a noisy synthetic series (`tests/smoke_test_river_skeleton_swot.py`)

These tests do **not** validate full data-dependent river inference or SDB alignment correctness.
