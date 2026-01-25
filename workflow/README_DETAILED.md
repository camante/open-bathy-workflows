# Unified SDB + River Bathymetry + Intelligent Gap-Fill (v0.7.8)

## What this is for

This repository implements a unified workflow to generate **gap-filling bathymetry** products that complement a CUDEM topobathy DEM stack.

- In the final CUDEM stack, **authoritative measurements** (topo lidar, bathy lidar, sonar/BAG/CSB, NOS hydro surveys, etc.) should be treated as the primary truth surfaces.
- This workflow is intended to **fill the remaining gaps**:
  - **Shallow, optically clear nearshore/estuarine water** where SDB works well.
  - **River channels** where no sonar or bathy lidar exists, using geometry + hydraulics + calibration anchors.
  - **Optional intelligent interpolation** that corrects model priors using high-quality point measurements.

The workflow is orchestrated by `bathy_main.py` and can run:

- **SDB only**
- **River only**
- **SDB + river with fusion**
- **SDB + river + gap-fill**

---

## Quick start

### SDB + river (fused)

```bash
python bathy_main.py   --aoi "-88/-87.75/30.5/30.75"   --start 2025-01-01 --end 2026-01-01   --out-dir output/mobile_bay   --methods sdb,river   --priority sdb   --extra-xyz-cudem=hydronos,ehydro
```

### River only (requires a topo DEM)

```bash
python bathy_main.py   --aoi "-74.5/-74.25/40.25/40.5"   --start 2025-01-01 --end 2026-01-01   --out-dir output/raritan   --methods river   --river-dem USACE_2012_NCMP_Lidar_DEM.tif
```

### Enable Manning + regional curves

```bash
python bathy_main.py   --aoi "-88/-87.75/30.5/30.75"   --start 2025-01-01 --end 2026-01-01   --out-dir output/mobile_bay   --methods sdb,river   --river-prior-mode multivariate   --river-manning-enabled   --river-manning-region auto   --river-regional-curve-enabled
```

### Enable intelligent gap-fill (Tier 1/2)

Provide high-quality bathymetry points (sonar/bathy lidar) in lon/lat with a depth column:

```bash
python bathy_main.py   --aoi "-88/-87.75/30.5/30.75"   --start 2025-01-01 --end 2026-01-01   --out-dir output/mobile_bay   --methods sdb,river   --priority sdb   --gapfill-enabled   --gapfill-hq xyz/sonar_navd88.xyz xyz/bathy_lidar_navd88.xyz
```

---

## Outputs

The run output directory will contain per-method subfolders and combined products:

- `output/<run>/sdb/`  
  SDB model, diagnostics, and predicted depth products.

- `output/<run>/river/`  
  Cross-section GeoPackages, inferred bed/depth rasters, and river diagnostics.

- `output/<run>/combined/`  
  Fused surfaces and (if enabled) intelligent gap-filled surfaces.

### Common rasters

- `*_depth.tif`: Depth raster, **negative down** convention (e.g., -2.3 m).
- `*_bed_elev.tif`: Bed elevation raster (same horizontal CRS as the workflow outputs).
- `*_sigma.tif`: A simple uncertainty proxy (meters). Not a full Bayesian posterior.
- `*_provenance.tif`: Encodes which method produced the pixel.

---

## Workflow overview

### 1) Orchestration (`bathy_main.py`)

`bathy_main.py` is the top-level driver. It:

1. Runs **SDB** (`sdb_main.py`) if `--methods` includes `sdb`.
2. Runs **river inference** (cross-section generation + inference + rasterization) if `--methods` includes `river`.
3. Runs **fusion** (`bathy_fusion.py`) when both methods are requested.
4. Optionally runs **gap-fill** (`gapfill_intelligent.py`) to correct the fused prior using high-quality measurements.

All subprocess calls are logged and the workflow writes a run report under the output folder.

---

## SDB pipeline (Sentinel-2 + ICESat-2)

### Sentinel-2 retrieval and masking

Sentinel-2 L2A surface reflectance imagery is fetched for the AOI and date range. Scenes are filtered by cloud percentage.

A conservative land/water mask is applied to prevent training collapse in complex estuaries:

1. **waffles coastline** mask aligned to the Sentinel-2 grid
2. Union with **hydrography water**:
   - `NHDWaterbody` polygons
   - optional `NHDArea` polygons
   - buffered `NHDFlowline` (conservative river/estuary corridor)
3. If the hydrography union is empty, use **NDWI fallback** (`(B03 - B08) / (B03 + B08)` > 0) as a safety valve.

The pipeline logs:
- hydrography feature counts
- approximate water coverage
- a warning when masks look suspicious (e.g., "nearly all land")

### Training

Training samples are built from:

- ICESat-2 photon-derived depths (ATL03/ATL24 filtering)
- optional extra XYZ soundings

A Random Forest model is trained to predict depth from optical features.
If the environment filter removes all samples, the run fails fast with an explicit log message.

### Prediction

The trained model predicts depth over valid water pixels. Depth-of-applicability (DOA) checks and max-depth guards are applied.

---

## River pipeline (cross-sections + priors + anchors)

### Inputs

Minimum:
- A **topo DEM** covering the river corridor (`--river-dem`)
- River flowlines (downloaded via TNM/NHD by default)

Optional but valuable:
- Drainage area / slope attributes (NHDPlus HR attributes if available)
- USGS gage metadata or discharge measurement anchors
- Regional curve coefficients (Drainage Area → bankfull depth)

### Steps

1) **River network extraction** (`river_network.py`)
- Builds river features inside the AOI.
- Preserves reach attributes when present.

2) **Cross-section generation** (`xs_builder.py`)
- Generates cross-sections at spacing `--xs-spacing-m`.
- Samples bank elevations and estimates a bank-to-bank width.

3) **Depth and bed inference** (`xs_infer_bathy_raster.py`)
- Computes an initial Dmax prior from width (powerlaw or multivariate).
- Blends optional secondary priors and anchors:
  - **USGS discharge-measurement anchors (Option A)**
  - **Regional curves** (Drainage Area → bankfull depth)
  - **Manning inversion** (Q + width + slope → depth), with tidal/backwater guards
  - **Width–stage inversion** if you provide time series
- Converts depth to a bed surface using the DEM-derived local WSE proxy.

4) **Reach-consistent WSE profile fitting** (`river_wse.py`)

Per `river_id`, noisy per-cross-section WSE proxies are smoothed along stationing:
- rolling median/mean
- optional isotonic (monotone) enforcement using PAVA

The stabilized profile provides:
- `wse_fit_m`
- `slope_wse_mpm` (preferred slope for Manning and multivariate priors when reach slope is missing)

5) **Rasterization**
- The inferred cross-section bed points are interpolated to a raster aligned to the target grid.

---

## Intelligent gap-fill (Tier 1/2)

`gapfill_intelligent.py` performs a conservative, barrier-aware residual correction:

1. Start from a **prior depth raster** (typically fused SDB+river depth).
2. Load **high-quality (HQ) depth points** (sonar, bathy lidar, curated XYZ).
3. Sample prior depth at HQ points and compute residuals:

   `r = depth_obs - depth_prior(sampled)`

4. Interpolate residuals and add back to the prior:

   `depth_gapfilled = depth_prior + r_interp`

### Barrier-aware residual interpolation

When SciPy is available, the workflow:
- Labels connected components in a **water mask**
- Fits/evaluates residuals **per component** to avoid bleeding across land barriers

If SciPy is not available, it falls back safely to IDW-like behavior with clear logging.

### Output products

The gap-fill step writes:
- `combined/bathy_combined_depth_gapfill.tif`
- `combined/bathy_combined_depth_gapfill_sigma.tif`
- `combined/bathy_combined_depth_gapfill_provenance.tif`

---

## Scientific limitations and guards

- **SDB** depends strongly on water clarity and bottom type. The workflow uses conservative masking and DOA guards, but you should expect failures in turbid rivers and highly variable bottoms.
- **River priors** infer plausible bathymetry, not guaranteed thalweg truth. Backwater/tidal reaches are guarded; Manning inversion is down-weighted or suppressed when slope is not meaningful.
- **Gap-fill** is a residual smoother, not a hydrodynamic inversion. It is designed to correct a physically motivated prior using HQ measurements, but it does not enforce momentum/continuity equations.

---

## File map

- `bathy_main.py`: top-level orchestrator
- `sdb_main.py` + `sdb/*.py`: SDB pipeline
- `xs_builder.py`, `xs_infer_bathy_raster.py`: river cross-sections + inference
- `river_wse.py`: reach-consistent WSE profile fitting
- `gapfill_intelligent.py`: residual interpolation gap-fill
- `bathy_fusion.py`: fusion logic
- `nss_services.py`: StreamStats/NSS helpers + caching
