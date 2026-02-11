# SDB Pipeline v0.7.8 - Satellite-Derived Bathymetry + River Integration

A production-ready pipeline for deriving bathymetry from **Sentinel-2 imagery** using **ICESat-2 ATL03/ATL24** altimetry as training data.

> For an end-to-end description of both SDB and the river workflow (including Option A USGS anchors), see **README_DETAILED.md** (recommended) or **README_WORKFLOW.md**.

## Script inventory

For a generated inventory of **all scripts** in this folder (with one-line summaries and a CLI flag), see **README_SCRIPTS.md**.

## Quick Start

```bash
# Basic SDB run
python sdb_main.py \
    --aoi "-88.1/-87.9/30.2/30.4" \
    --start 2024-01-01 \
    --out-dir ./output

# With extra soundings + NAVD88 output
dlim -R=-88.1/-87.9/30.2/30.4 hydronos -P epsg:4269+5714 > soundings.xyz
python sdb_main.py \
    --aoi "-88.1/-87.9/30.2/30.4" \
    --start 2024-01-01 \
    --extra-xyz soundings.xyz \
    --convert-sdb-to-navd88 \
    --out-dir ./output
```

## Vertical Datum Handling

### SDB Output Values

**SDB outputs are ELEVATION values relative to MSL** (Mean Sea Level):

| Pixel Value | Meaning |
|-------------|---------|
| -5.0m | Seabed is at -5.0m MSL (5m below Mean Sea Level) |
| -10.0m | Seabed is at -10.0m MSL (10m below Mean Sea Level) |

This is **NOT** "depth below instantaneous water surface" - it's elevation relative to a fixed vertical datum (MSL).

### Why MSL?

MSL is the reference because:
- **Sentinel-2 compositing**: Multi-temporal median averages out tidal variations
- **ICESat-2 training**: Uses EGM2008 orthometric heights, which approximate MSL
- **Multiple passes**: Different tidal phases average toward MSL

### Converting to NAVD88

Use `--convert-sdb-to-navd88` to convert from MSL to NAVD88:

```bash
python sdb_main.py --aoi "..." --start "..." --convert-sdb-to-navd88
```

The conversion applies the spatially-varying MSL-NAVD88 separation using VDatum grids via CUDEM dlim:

```
elev_NAVD88 = elev_MSL + (NAVD88 - MSL separation)

Example (if MSL is 0.3m above NAVD88 locally):
- Input:  -5.0m MSL
- Output: -5.0 + 0.3 = -4.7m NAVD88
```

### Output Files

| File | Vertical Datum | Description |
|------|----------------|-------------|
| `SDB_Prediction_10m.tif` | MSL | Original output (elevations relative to MSL) |
| `SDB_Prediction_10m_NAVD88.tif` | NAVD88 | Converted output (if --convert-sdb-to-navd88) |

## Input Data Requirements

### Extra XYZ Soundings

When using `--extra-xyz`, Z values should be **elevations relative to MSL**:

```bash
# Download NOAA soundings converted to MSL (from MLLW)
dlim -R=W/E/S/N hydronos -P epsg:4269+5714 > soundings_msl.xyz
```

The `-P epsg:4269+5714` tells dlim to output in NAD83 + MSL.

## CLI Reference

### Core Arguments

| Argument | Description |
|----------|-------------|
| `--aoi` | Bounding box: W/E/S/N |
| `--start` | Start date: YYYY-MM-DD |
| `--end` | End date (default: today) |
| `--out-dir` | Output directory |

### Vertical Datum

| Argument | Description | Default |
|----------|-------------|---------|
| `--convert-sdb-to-navd88` | Convert from MSL to NAVD88 | False |
| `--sdb-source-vdatum` | Source CRS | epsg:4269+5714 (NAD83+MSL) |
| `--sdb-target-vdatum` | Target CRS | epsg:4269+5703 (NAD83+NAVD88) |

### Data Sources

| Argument | Description |
|----------|-------------|
| `--icesat` | ICESat-2 products: atl03, atl24, all_atl |
| `--extra-xyz` | Additional XYZ files (should be MSL-referenced) |

## Installation

```bash
pip install numpy pandas scikit-learn rasterio geopandas pyproj joblib
pip install sliderule-python h5py
pip install cudem  # For dlim (datum transformations)
```

## Troubleshooting

### "dlim not found"
```bash
pip install cudem
which dlim  # Verify installation
```

### "No ICESat-2 data found"
- Expand date range
- Check AOI format (W/E/S/N, not S/N/W/E)
- Try `--icesat atl03`

### "Zero pixels predicted"
- Lower `--cw-min` threshold (try 0.05)
- Try `--no-doa` to disable DOA filtering
- Check cloud cover in imagery

## References

- Stumpf, R.P., et al. (2003). Determination of water depth with high-resolution satellite imagery. *Limnology and Oceanography*, 48(1), 547-556.
- Parrish, C.E., et al. (2019). Validation of ICESat-2 ATLAS bathymetry. *Remote Sensing*, 11(14), 1634.
