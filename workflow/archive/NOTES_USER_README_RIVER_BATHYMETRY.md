# River Bathymetry Estimation Pipeline v0.7.7

A comprehensive Python pipeline for estimating river bathymetry from multiple data sources, including satellite imagery, DEM analysis, USGS gage data, and hydraulic geometry relationships.

## Overview

This pipeline provides **multi-method depth estimation** for rivers where direct measurements (sonar, bathymetric lidar) are unavailable. It combines:

1. **Hydraulic Geometry** - Width-depth power laws (Leopold & Maddock, 1953)
2. **Multivariate Priors** - Drainage area and slope-adjusted estimates
3. **Manning Inversion** - Q + width + slope → depth (NEW in v0.7.7)
4. **USGS Measurement Integration** - Width/area from discharge measurements
5. **Width-Stage Relationships** - At-a-station hydraulic geometry
6. **Soundings Calibration** - Direct measurements where available

## New in v0.7.7: Manning Inversion with Backwater/Tidal Guard

### Manning's Equation Inversion

The pipeline now supports estimating depth by inverting Manning's equation:

```
Q = (1/n) × W × D^(5/3) × S^(1/2)

Solving for D:
D = (Q × n / (W × S^(1/2)))^(3/5)
```

Where:
- **Q** = Discharge (estimated from drainage area using StreamStats-style regional regression)
- **n** = Manning's roughness coefficient
- **W** = Channel width (from cross-section analysis)
- **S** = Water surface slope (from DEM or reach attributes)

### Backwater/Tidal Guard

Manning's equation assumes steady, uniform flow. In tidal or backwater zones, this assumption fails. The pipeline implements a **confidence guard** that automatically reduces Manning's influence when:

| Condition | Guard Effect |
|-----------|--------------|
| Slope < 0.0001 m/m | Near-zero confidence (tidal zone) |
| Slope < 0.0003 m/m | Reduced confidence (possible backwater) |
| Distance to tide < 5 km | Significantly reduced confidence |
| Distance to tide 5-15 km | Moderately reduced confidence |
| Elevation < 3 m ASL | Additional reduction |

### Usage

Enable Manning prior with:

```bash
python xs_infer_bathy_raster.py \
    --xs-gpkg river_xs.gpkg \
    --out-gpkg river_bathy.gpkg \
    --dem cudem_dem.tif \
    --prior-mode multivariate \
    --manning-enabled \
    --manning-n 0.035 \
    --manning-region coastal_plain
```

Or via the main pipeline:

```bash
python bathy_main.py \
    --aoi="-88/-87.75/30.5/30.75" \
    --start 2025-01-01 --end 2026-01-01 \
    --out-dir output/mobile_bay \
    --methods river \
    --river-manning-enabled \
    --river-manning-region coastal_plain
```

### Regional Q2 Estimation

The pipeline estimates bankfull discharge (Q2) from drainage area using regional regression equations:

| Region | Equation Form | Uncertainty |
|--------|---------------|-------------|
| Coastal Plain | Q2 = 15.0 × A^0.75 | ±40% |
| Piedmont | Q2 = 20.0 × A^0.78 | ±35% |
| Appalachian | Q2 = 25.0 × A^0.80 | ±35% |
| Great Plains | Q2 = 8.0 × A^0.70 | ±50% |
| Rocky Mountain | Q2 = 15.0 × A^0.75 | ±45% |

---

## Complete Workflow

### Step 1: Run the Main Pipeline

```bash
python bathy_main.py \
    --aoi="-77.5/-77.0/38.5/39.0" \
    --start 2025-01-01 \
    --end 2026-01-01 \
    --out-dir output/potomac \
    --methods sdb,river \
    --priority sdb \
    --river-manning-enabled \
    --river-prior-mode multivariate
```

### Step 2: Manual River-Only Processing

```bash
# Build river network
python river_network.py \
    --aoi "-77.5/-77.0/38.5/39.0" \
    --out-gpkg river_network.gpkg

# Build cross-sections
python xs_builder.py \
    --river-gpkg river_network.gpkg \
    --dem cudem_dem.tif \
    --out-gpkg river_xs.gpkg \
    --spacing-m 200 \
    --half-width-m 150

# Infer bathymetry with Manning
python xs_infer_bathy_raster.py \
    --xs-gpkg river_xs.gpkg \
    --out-gpkg river_bathy.gpkg \
    --dem cudem_dem.tif \
    --template-raster cudem_dem.tif \
    --out-bathy-raster river_bed.tif \
    --prior-mode multivariate \
    --manning-enabled \
    --continuous walid
```

---

## Depth Estimation Methods

### 1. Power-Law Prior (Default)

```
D_max = a × W^b
```

Default: a=0.18, b=0.50 (Leopold & Maddock, 1953)

### 2. Multivariate Prior

```
D_max = a₀ × W^bw × (A + ε_a)^ba × (S + ε_s)^bs
```

### 3. Manning Inversion (NEW)

Uses discharge-width-slope relationship with backwater/tidal guard.

### 4. USGS Gage Calibration

Uses USGS discharge measurement records.

### 5. Width-Stage Inversion

Uses historical width observations at different water levels.

### 6. Soundings Calibration

Direct depth measurements where available.

---

## Configuration Reference

### Manning Inversion Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--manning-enabled` | False | Enable Manning inversion prior |
| `--manning-n` | 0.035 | Manning's roughness coefficient |
| `--manning-region` | default | Region for Q2 estimation |
| `--manning-min-confidence` | 0.3 | Minimum confidence to include in blend |
| `--manning-blend-weight` | 0.5 | Maximum weight when blending |

### Backwater/Tidal Guard Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-backwater-guard` | False | Disable slope-based guard |
| `--no-tidal-guard` | False | Disable distance-to-tide guard |
| `--slope-tidal-threshold` | 0.0001 | Slope below which flow is tidal |
| `--slope-backwater-threshold` | 0.0003 | Slope below which backwater possible |

---

## Output Files

### GeoPackage Layers

| Layer | Contents |
|-------|----------|
| `xs_params` | Cross-section parameters including depth estimates |
| `xs_bathy_points` | Bed elevation points along each cross-section |
| `xs_lines` | Cross-section line geometries |

### Key Columns in xs_params

| Column | Description |
|--------|-------------|
| `dmax_prior_m` | Prior depth estimate |
| `dmax_raw_m` | Final depth estimate |
| `calib_src` | Calibration source |
| `manning_depth_m` | Manning inversion estimate (if enabled) |
| `manning_conf` | Manning estimate confidence |
| `manning_backwater` | Backwater flag |
| `manning_tidal` | Tidal flag |

### Raster Outputs

| File | Contents |
|------|----------|
| `*_bed.tif` | Predicted bed elevation (m) |
| `*_mask.tif` | Binary mask (1 = has data) |
| `*_uncert.tif` | Uncertainty raster (m) |

---

## Scientific References

### Hydraulic Geometry
- Leopold, L.B. & Maddock, T. (1953). The hydraulic geometry of stream channels. USGS Professional Paper 252.

### Manning's Equation
- Chow, V.T. (1959). Open-Channel Hydraulics. McGraw-Hill.
- Barnes, H.H. (1967). Roughness characteristics of natural channels. USGS Water-Supply Paper 1849.

### Regional Regression (StreamStats)
- Ries, K.G., et al. (2017). StreamStats, version 4. USGS Fact Sheet 2017-3046.

---

## Module Structure

```
├── bathy_main.py              # Main pipeline orchestrator
├── xs_infer_bathy_raster.py   # Depth inference engine
├── manning_inversion.py       # Manning equation inversion (NEW)
├── river_network.py           # River network acquisition
├── xs_builder.py              # Cross-section construction
├── xs_adjust_monotonic.py     # Monotonic bed adjustment
├── usgs_nwis.py               # USGS data fetching
├── constants.py               # Centralized constants
└── errors.py                  # Custom exceptions
```

---

## Version History

### v0.7.7 (Current)
- Added Manning inversion prior
- Added backwater/tidal guard
- Added StreamStats Q2 regional regression
- Fixed duplicate argument bug in bathy_main.py

### v0.7.6
- Multivariate prior with drainage area and slope
- USGS measurement integration
