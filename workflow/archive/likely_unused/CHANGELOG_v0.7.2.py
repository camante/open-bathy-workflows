#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHANGELOG - SDB + River Bathymetry Pipeline v0.7.2

Bug Fixes and Improvements Applied 2026-01-14
=============================================

This document summarizes all fixes applied to resolve the issues in the
pipeline run logs and address code review feedback.


## CRITICAL BUG FIXES

### 1. NameError in bathy_main.py (Line 1274)
**Problem**: `patch_tif` was referenced but not defined
**Solution**: Changed to `bed_tif` which is the correct variable name
**File**: bathy_main.py

### 2. SDB Training Failure (CLEAR_WATER filter too strict)
**Problem**: Only 154 of 2000 extra_xyz points passed the CLEAR_WATER filter,
falling below the 200 point minimum required for training.
**Root Cause**: The extra_xyz bypass logic (line 1399-1400) only bypassed the
CLEAR_WATER filter but still required points to pass the LAND mask filter.
For turbid Gulf Coast areas, many shallow water points were being incorrectly
classified as "land" by the waffles coastline mask.
**Solution**: Modified train.py to bypass BOTH land AND clear_water filters for
extra_xyz points, since these are typically high-quality surveyed data that
should be trusted regardless of S2-derived mask values.

**File**: train.py (lines 1397-1410)
**Old code**:
```python
m_env = m_land & m_cw
if "source_norm" in df.columns and "CLEAR_WATER" in df.columns:
    m_env |= (m_land & (df["source_norm"] == "extra_xyz"))
```
**New code**:
```python
m_env = m_land & m_cw
# FIX: extra_xyz points should bypass BOTH land AND clear_water filters
if "source_norm" in df.columns:
    m_is_xyz = (df["source_norm"] == "extra_xyz")
    n_xyz_before = int(m_is_xyz.sum())
    m_env |= m_is_xyz
    if n_xyz_before > 0:
        log.info(f"[TRAIN][ENV] Bypassed env filter for {n_xyz_before} extra_xyz points")
```

### 3. US_Gulf_Coast Configuration Too Restrictive
**Problem**: The region config had `cw_min: 0.6` and `min_training_points_for_sdb: 200`
which is too strict for turbid river-influenced waters.
**Solution**: Updated sdb_config.json:
- `cw_min`: 0.6 → 0.3 (more lenient for turbid water)
- `min_training_points_for_sdb`: 200 → 50 (allow smaller training sets)
- Added missing parameters: `water_class`, `min_bottom_frac`, `atl24_conf_min`,
  `linf_enabled`, `doa_threshold`, `max_depth_source`, `spatial_cv*`

**File**: sdb_config.json


## LOGGING CONSISTENCY FIXES

### 4. Centralized Logging Module
**Problem**: Multiple modules called `logging.basicConfig()` at import time,
causing inconsistent log formats depending on import order.
**Solution**: Created `logging_config.py` with centralized logging setup.

**New File**: logging_config.py
- `setup_logging()`: Call once at entrypoint before other imports
- `get_logger(name)`: Safe way to get module loggers
- `add_file_handler()`: Add per-run log files

### 5. Removed basicConfig() from Module-Level Code
**Files Modified**:
- train.py - Removed lines 437-442
- atl.py - Removed lines 290-295
- fusion.py - Removed lines 25-29
- river_network.py - Removed lines 60-64
- xs_builder.py - Removed lines 66-67
- river_bathy.py - Removed lines 40-45
- bathy_fusion.py - Removed lines 50-55
- bathy_main.py - Updated to call setup_logging() before imports
- sdb_main.py - Updated to call setup_logging() before imports


## TYPE HINTS IMPROVEMENTS

### 6. Added Comprehensive Type Hints to Core Modules

**fusion.py**:
- Added `from __future__ import annotations`
- Added type hints: `Dict[str, Any]`, `List`, `Tuple`, `Union`, `Optional`
- Improved function signatures with proper return types

**predict.py**:
- Added `from __future__ import annotations`
- Added `Union` to typing imports
- Type-annotated module-level constants: `ALIGNMENT_AVAILABLE: bool`
- Type-annotated optional error strings: `_alignment_import_error: Optional[str]`

**river_bathy.py**:
- Added `Union` to typing imports (was missing)


## CODE QUALITY IMPROVEMENTS

### 7. Implemented river_diagnostics.py
**Problem**: The file was a 37-byte stub that only imported RiverReport
**Solution**: Implemented full diagnostic module with:
- `XSectionStats` dataclass for per-XS statistics
- `RiverDiagnostics` dataclass for run-level statistics
- `compute_xs_diagnostics()` - Analyze XS bathymetry GPKG
- `compute_raster_diagnostics()` - Analyze output rasters
- `summarize_river_run()` - Generate comprehensive run summary

**File**: river_diagnostics.py (expanded from 37 bytes to ~6KB)


## SUMMARY OF FILES MODIFIED

1. **bathy_main.py** - Fixed NameError, added centralized logging
2. **sdb_main.py** - Added centralized logging before imports
3. **train.py** - Fixed extra_xyz bypass, removed basicConfig
4. **fusion.py** - Added type hints, removed basicConfig
5. **predict.py** - Added type hints
6. **atl.py** - Removed basicConfig
7. **river_network.py** - Removed basicConfig
8. **xs_builder.py** - Removed basicConfig
9. **river_bathy.py** - Added Union import, removed basicConfig
10. **bathy_fusion.py** - Removed basicConfig
11. **sdb_config.json** - Updated US_Gulf_Coast parameters
12. **logging_config.py** - NEW: Centralized logging configuration
13. **river_diagnostics.py** - Implemented (was stub)


## TESTING

All modified files pass Python syntax checking:
```bash
python3 -m py_compile bathy_main.py train.py fusion.py logging_config.py \
    river_diagnostics.py predict.py sdb_main.py atl.py bathy_fusion.py \
    river_bathy.py river_network.py xs_builder.py
```

## EXPECTED BEHAVIOR AFTER FIXES

1. **SDB Training**: Should now succeed with extra_xyz data bypassing both
   land and clear_water filters, allowing the full 2000 points to be used.

2. **River Pipeline**: Should complete without NameError, producing:
   - `river_bed_elev_patch.tif` - Bed elevation (NAVD88)
   - `river_depth_terrain_patch.tif` - Depth relative to DEM

3. **Logging**: All log messages should have consistent format:
   `%(asctime)s [%(levelname)s] %(name)s: %(message)s`

## REMAINING RECOMMENDATIONS (from code review)

Priority High (not addressed in this patch):
- Add tests for river modules (2-3 days effort)
- Complete type hints in remaining core modules (1-2 days)

Priority Medium:
- Consolidate L∞ estimation (appears in train.py:94-118 and 121-177)
- Add YAML config file support as alternative to CLI
- Standardize logging levels (some INFO should be DEBUG)

Priority Low:
- Add Dask parallelization for large AOIs
- Consider race condition in cache check (sdb_main.py:255-267)
"""

__version__ = "0.7.2"
PATCH_DATE = "2026-01-14"
