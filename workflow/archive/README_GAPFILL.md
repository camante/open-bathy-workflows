# Intelligent Gap-Fill Module

## Overview

The intelligent gap-fill module (`gapfill_intelligent.py`) implements physics-informed gap-filling using a **prior + residual interpolation** framework. This approach is designed for seamless integration with CUDEM workflows where authoritative bathymetric data (sonar, bathy lidar) constrains the SDB/river model.

## Architecture

The module is designed with future waffles integration in mind:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BathyInterpolator Class                      │
│  (Core interpolation - can become waffles gridding module)      │
├─────────────────────────────────────────────────────────────────┤
│  • fit(obs_xy, obs_z, prior_z, uncertainty)                     │
│  • predict(query_xy, prior_z) → (z_pred, uncertainty)           │
│  • Coordinate scaling (geographic → local meters)               │
│  • Multiple interpolation backends (RBF, GP, IDW)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              gapfill_depth_raster() Function                    │
│  (Raster workflow wrapper - standalone usage)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Raster I/O                                                   │
│  • Connected component processing                               │
│  • Bank constraint enforcement                                  │
│  • CUDEM XYZ output                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concept

The key equation:

```
z_final(x) = z_prior(x) + r_interp(x)
```

Where:
- **z_prior**: SDB + river hydraulics merged surface (the "physics prior")
- **r_interp**: Interpolated residuals from high-quality measurements
- **z_final**: Seamless bathymetry that honors measurements and uses physics in gaps

This approach ensures:
- **No seams** - transitions are continuous by construction
- **Physics in gaps** - prior provides plausible shape where no HQ data exists
- **Measurements honored** - HQ data pulls the prior toward ground truth

## Key Improvements in v0.8.0

1. **Coordinate Scaling**: Properly handles geographic (lat/lon) coordinates by converting to local meters before interpolation. This fixes distance calculations for RBF/GP/IDW.

2. **BathyInterpolator Class**: Cleanly separated core interpolation logic that can be used standalone or integrated as a waffles module.

3. **Robust Sign Detection**: Improved auto-detection of depth sign conventions (positive-down vs negative-down).

4. **Proper Uncertainty Propagation**: Bayesian combination of prior and residual uncertainties.

## Usage

### Command Line

```bash
# Basic usage
python gapfill_intelligent.py \
    --prior bathy_combined.tif \
    --hq sonar_points.xyz lidar_bathy.gpkg \
    --out gapfilled.tif \
    --sigma gapfilled_sigma.tif \
    --prov gapfilled_prov.tif

# With river smoothing and CUDEM output
python gapfill_intelligent.py \
    --prior bathy_combined.tif \
    --hq sonar_points.xyz \
    --out gapfilled.tif \
    --sigma gapfilled_sigma.tif \
    --prov gapfilled_prov.tif \
    --xs-gpkg river_xs_params.gpkg \
    --river-mask river_depth.tif \
    --method rbf \
    --river-smooth-sigma 500 \
    --cudem-xyz
```

### Via bathy_main.py

```bash
python bathy_main.py \
    --aoi "-76.5/-76.0/38.5/39.0" \
    --start 2024-01-01 \
    --methods sdb river \
    --gapfill-enabled \
    --gapfill-hq sonar_soundings.xyz \
    --gapfill-method rbf \
    --gapfill-river-smooth-sigma 500 \
    --gapfill-cudem-xyz \
    --out-dir ./output
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prior` | *required* | Prior depth raster (SDB + river fusion) |
| `--hq` | *required* | High-quality point files (CSV/XYZ/GPKG) |
| `--out` | *required* | Output gap-filled raster |
| `--sigma` | *required* | Output uncertainty raster |
| `--prov` | *required* | Output provenance raster |
| `--method` | `rbf` | Interpolation: `rbf`, `gp`, `idw` |
| `--rbf-function` | `thin_plate` | RBF kernel type |
| `--water-mask` | auto | Water mask (1=water) |
| `--river-mask` | None | River corridor mask |
| `--xs-gpkg` | None | XS parameters for anisotropic smoothing |
| `--bank-elev` | None | Bank elevation for constraints |
| `--river-smooth-sigma` | 500 | Along-channel smoothing (m) |
| `--cudem-xyz` | False | Output CUDEM-compatible XYZ |

## Input Formats

### High-Quality Points

Supports:
- **CSV/TXT/XYZ**: Columns with `x/lon`, `y/lat`, `z/depth` headers
- **GeoPackage/Shapefile**: Point geometry with depth attribute
- Optional uncertainty column: `uncertainty`, `sigma`, `error`

Example CSV:
```csv
lon,lat,depth,uncertainty
-76.234,38.567,-5.23,0.15
-76.235,38.568,-4.87,0.12
```

### Prior Raster

- GeoTIFF with depth/elevation values
- Negative values = below reference datum (typical convention)
- NoData for land/invalid pixels

## Interpolation Methods

### 1. RBF (Thin-Plate Spline) - Default

Radial basis function interpolation. Fast, smooth, works well with moderate point density.

```python
cfg = GapfillConfig(
    interpolation_method="rbf",
    rbf_function="thin_plate",  # or 'multiquadric', 'cubic'
    rbf_smoothing=0.0,  # 0 = exact interpolation
)
```

### 2. Gaussian Process

Provides uncertainty estimates. Best when point density varies.

```python
cfg = GapfillConfig(
    interpolation_method="gp",
    gp_length_scale=500.0,  # meters
    gp_nu=2.5,  # Matern smoothness
    gp_max_points=2000,  # subsamples if exceeded
)
```

### 3. IDW (Fallback)

Simple inverse distance weighting. Used when RBF/GP fails.

```python
cfg = GapfillConfig(
    interpolation_method="idw",
    idw_power=2.0,
    idw_neighbors=12,
)
```

## Key Features

### 1. Barrier-Constrained Interpolation

Interpolation respects land barriers. Each connected water component is processed independently, preventing smoothing across peninsulas or islands.

### 2. Anisotropic River Smoothing

In river corridors, residuals are smoothed along the channel centerline:

- **Along-channel**: Smooth (sigma = 500m typical)
- **Cross-channel**: Sharp (preserves bank transitions)

This prevents classic artifacts like bulging across meanders.

### 3. Bank Constraint Enforcement

Optional constraint ensures bed stays below bank elevation:

```
z_bed ≤ z_bank - clearance
```

Default clearance is 0.3m.

### 4. Uncertainty Propagation

Output uncertainty combines:
1. Prior uncertainty (from SDB/river)
2. Residual RMSE at HQ points
3. Distance-based growth away from measurements

## Output Files

| File | Description |
|------|-------------|
| `*_gapfill.tif` | Gap-filled depth raster |
| `*_gapfill_sigma.tif` | Uncertainty raster (m) |
| `*_gapfill_provenance.tif` | Provenance codes (see below) |
| `*_gapfill.xyz` | CUDEM-compatible XYZ (if enabled) |

### Provenance Codes

| Code | Meaning |
|------|---------|
| 0 | NoData |
| 1 | Prior only (no HQ nearby) |
| 2 | Prior + residual correction |
| 3 | At/near HQ measurement |
| 4 | River smoothed (anisotropic) |
| 5 | Bank constrained |

## CUDEM Integration

The gap-fill module is designed for integration with CUDEM's waffles gridding workflow.

### Option 1: As a Pre-Processor

Run gap-fill on your SDB/river output, then use the result as a datalist entry:

```bash
# Generate gap-filled surface
python gapfill_intelligent.py \
    --prior sdb_river_fused.tif \
    --hq lidar_bathy.xyz sonar.xyz \
    --out gapfilled.tif \
    --sigma gapfilled_sigma.tif \
    --prov gapfilled_prov.tif \
    --cudem-xyz

# Use in waffles with uncertainty
waffles -R ... -E .11111111s -O dem \
    gapfilled.xyz,-250,.3,.5 \
    lidar_bathy,-201,.6,.1 \
    ...
```

### Option 2: Direct CUDEM XYZ Output

The `--cudem-xyz` flag outputs a datalist-compatible file:

```
x y z weight uncertainty
-76.234 38.567 -5.23 0.8 0.35
...
```

Weight is derived from uncertainty (inverse relationship).

## Scientific Background

### Prior + Residual Framework

This approach is mathematically equivalent to **regression kriging** (Hengl et al., 2007):

1. Fit a trend model (our physics prior)
2. Krige the residuals
3. Add trend + kriged residuals

The physics prior captures:
- SDB: optical depth relationships
- River: hydraulic geometry, Manning's equation
- Channel: cross-section shape constraints

The residual interpolation:
- Corrects systematic biases in the prior
- Honors HQ measurements exactly (or near-exactly)
- Smoothly transitions to prior-only in gaps

### Why Residuals?

Interpolating residuals instead of raw depths:
- Residuals are smaller magnitude → more stable numerics
- Residuals are often smoother → better interpolation
- Near HQ data, residual → 0, so prior is corrected
- Far from HQ data, residual → 0, so prior dominates

## Troubleshooting

### "RBF interpolation failed"

- Try increasing `rbf_min_points` 
- Check for duplicate points in HQ data
- Try `--method idw` as fallback

### "Insufficient HQ points"

- Need at least `component_min_points` (default 10) per water body
- Check that HQ points overlap with water mask

### Large residual RMSE

- Check depth sign convention (negative down vs positive down)
- The module auto-detects and flips if needed
- Verify HQ data is in same vertical datum as prior

### Seams at component boundaries

- This shouldn't happen with barrier-constrained interpolation
- Check water mask connectivity
- May need to manually merge disconnected water bodies

## References

- Hengl, T., et al. (2007). "About regression-kriging: From equations to case studies." *Computers & Geosciences*, 33(10), 1301-1315.
- Merwade, V., et al. (2008). "Anisotropic considerations while interpolating river channel bathymetry." *Journal of Hydrology*, 351(1-2), 95-103.

## Future: Waffles Integration Path

The `BathyInterpolator` class is designed to eventually become a waffles gridding module (e.g., `WafflesBathyInterp` or `WafflesSDBRiver`). The integration would follow the existing waffles pattern:

```python
# Future waffles module (conceptual)
class WafflesBathyInterp(Waffle):
    """Physics-informed bathymetry interpolation with prior + residual framework."""
    
    def __init__(self, prior_datalist=None, **kwargs):
        super().__init__(**kwargs)
        self.prior_datalist = prior_datalist
        self.interpolator = BathyInterpolator(cfg=self.cfg)
    
    def run(self):
        # 1. Grid the prior (SDB/river) using existing waffles methods
        prior_grid = self._grid_prior()
        
        # 2. Fit interpolator with HQ data and prior
        self.interpolator.fit(hq_xy, hq_z, prior_at_hq)
        
        # 3. Predict at output grid locations
        z_pred, unc = self.interpolator.predict(grid_xy, prior_grid)
        
        # 4. Yield results or write to output
        self.dem.z = z_pred
        self.dem.u = unc
```

This would allow usage like:
```bash
waffles -R ... -E .11111111s -O dem \
    -M bathy_interp:prior=sdb_river.datalist \
    lidar_bathy,-201,.6,.1 \
    multibeam,-202,.1,.1
```

The key design decisions for this integration:
1. Prior is specified as a datalist (SDB, river, or combined)
2. HQ data comes from the main datalist entries with weights
3. Output is the gap-filled surface with uncertainty
4. Respects waffles' region, resolution, and projection settings

---

*Module version: 0.8.0*
