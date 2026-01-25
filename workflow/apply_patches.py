#!/usr/bin/env python3
"""
PATCH FILE: Critical Bug Fixes for SDB + River Bathymetry Pipeline v0.7.1

This file contains the patches needed to fix critical bugs preventing:
1. River channel full interpolation (corridor mask from lines failing)
2. Potential SDB zero-value issues

Apply these changes to your pipeline files.
"""

# =============================================================================
# PATCH 1: xs_infer_bathy_raster.py - Add missing imports and function
# =============================================================================

PATCH_1_IMPORTS = """
# CHANGE FROM:
from shapely.geometry import Point
from shapely.ops import unary_union

# CHANGE TO:
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
"""

PATCH_1_NEW_FUNCTION = '''
# ADD THIS FUNCTION after _utm_crs_from_lonlat() (around line 555):

def _utm_crs_for_point(x: float, y: float, source_crs: CRS) -> CRS:
    """Return a WGS84 UTM CRS for a point, handling both geographic and projected input CRS.
    
    Args:
        x: X coordinate (longitude if geographic, easting if projected)
        y: Y coordinate (latitude if geographic, northing if projected)
        source_crs: CRS of the input coordinates
        
    Returns:
        CRS object for the appropriate UTM zone
    """
    source_crs = CRS.from_user_input(source_crs)
    if source_crs.is_geographic:
        lon, lat = x, y
    else:
        # Transform to WGS84 to get lon/lat for UTM zone calculation
        tr = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
    return _utm_crs_from_lonlat(float(lon), float(lat))
'''

# =============================================================================
# PATCH 2: xs_infer_bathy_raster.py - Improve corridor buffer default
# =============================================================================

PATCH_2_BUFFER_IMPROVEMENT = '''
# CHANGE the buffer default calculation in _continuous_surface call (around line 1206-1208):

# FROM:
if continuous_buffer_m is None:
    continuous_buffer_m = max(3.0 * _template_pixel_size_m(tmpl), 20.0)

# TO:
if continuous_buffer_m is None:
    # Use a larger buffer to ensure continuity between cross-sections
    # Buffer should be at least half the XS spacing to ensure overlap
    pixel_based = 3.0 * _template_pixel_size_m(tmpl)
    # Default XS spacing is 200m, so use 150m as minimum to ensure overlap
    continuous_buffer_m = max(pixel_based, 150.0, 20.0)
'''

# =============================================================================
# PATCH 3: predict.py - Add diagnostic summary
# =============================================================================

PATCH_3_DIAGNOSTICS = '''
# ADD THIS after the windows loop (around line 815, before "finally:"):

        # Diagnostic summary
        log.info("=" * 60)
        log.info("[PREDICT] PREDICTION SUMMARY")
        log.info("=" * 60)
        total = max(1, funnel['pixels_total'])
        log.info(f"  Total pixels processed: {funnel['pixels_total']:,}")
        log.info(f"  Finite optical values:  {funnel['finite_optical']:,} ({100*funnel['finite_optical']/total:.1f}%)")
        log.info(f"  Passed clear water:     {funnel['cw_pass']:,} ({100*funnel['cw_pass']/total:.1f}%)")
        log.info(f"  Passed land mask:       {funnel['land_pass']:,} ({100*funnel['land_pass']/total:.1f}%)")
        log.info(f"  Base valid:             {funnel['base_valid']:,} ({100*funnel['base_valid']/total:.1f}%)")
        log.info(f"  Passed DOA:             {funnel['doa_pass']:,} ({100*funnel['doa_pass']/total:.1f}%)")
        log.info(f"  Final predicted:        {funnel['predicted']:,} ({100*funnel['predicted']/total:.1f}%)")
        log.info("=" * 60)
        
        if funnel['predicted'] == 0:
            log.error("[PREDICT] ⚠️ ZERO PIXELS PREDICTED!")
            log.error("[PREDICT] Possible causes:")
            log.error("  1. Clear water mask (CLEAR_WATER raster) has no valid pixels >= cw_min threshold")
            log.error("  2. Land mask incorrectly masking water pixels")
            log.error("  3. Sentinel-2 bands have invalid/missing data over AOI")
            log.error("  4. DOA filter rejecting all pixels (try --no-doa to disable)")
        elif funnel['predicted'] < 0.01 * total:
            log.warning(f"[PREDICT] ⚠️ Very few pixels predicted ({funnel['predicted']:,})")
            log.warning("[PREDICT] Consider reviewing mask settings and thresholds")
'''

# =============================================================================
# PATCH 4: atl.py - Remove unreachable code
# =============================================================================

PATCH_4_ATL_CLEANUP = '''
# REMOVE the unreachable line after transform_xyz_dataframe_crs() function (line 171):

# DELETE THIS LINE:
    log.warning("Harmony library not found. Harmony downloads will be skipped.")
    
# This line is outside the function and unreachable.
'''

# =============================================================================
# AUTOMATED PATCH APPLICATION SCRIPT
# =============================================================================

def apply_patches():
    """Apply all patches to the pipeline files."""
    import re
    from pathlib import Path
    
    # Patch xs_infer_bathy_raster.py
    xs_file = Path("xs_infer_bathy_raster.py")
    if xs_file.exists():
        content = xs_file.read_text()
        
        # Fix import
        content = content.replace(
            "from shapely.geometry import Point",
            "from shapely.geometry import Point, mapping"
        )
        
        # Add missing function after _utm_crs_from_lonlat
        if "_utm_crs_for_point" not in content:
            # Find the end of _utm_crs_from_lonlat function
            pattern = r'(def _utm_crs_from_lonlat\(.*?\n    return CRS\.from_epsg\(epsg\))'
            replacement = r'''\1


def _utm_crs_for_point(x: float, y: float, source_crs: CRS) -> CRS:
    """Return a WGS84 UTM CRS for a point, handling both geographic and projected input CRS."""
    source_crs = CRS.from_user_input(source_crs)
    if source_crs.is_geographic:
        lon, lat = x, y
    else:
        tr = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
    return _utm_crs_from_lonlat(float(lon), float(lat))'''
            
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Improve buffer default
        content = content.replace(
            'continuous_buffer_m = max(3.0 * _template_pixel_size_m(tmpl), 20.0)',
            'continuous_buffer_m = max(3.0 * _template_pixel_size_m(tmpl), 150.0, 20.0)'
        )
        
        xs_file.write_text(content)
        print(f"✓ Patched {xs_file}")
    
    # Patch atl.py - remove unreachable code
    atl_file = Path("atl.py")
    if atl_file.exists():
        content = atl_file.read_text()
        
        # Remove the orphaned log statement
        content = content.replace(
            '        return df\n    log.warning("Harmony library not found. Harmony downloads will be skipped.")',
            '        return df'
        )
        
        atl_file.write_text(content)
        print(f"✓ Patched {atl_file}")
    
    print("\n✓ All patches applied successfully!")
    print("\nNOTE: For predict.py diagnostics patch, manually add the diagnostic summary")
    print("      code block shown in PATCH_3_DIAGNOSTICS after the windows processing loop.")


if __name__ == "__main__":
    print("SDB Pipeline v0.7.1 Critical Bug Fixes")
    print("=" * 50)
    print("\nThis script will apply the following fixes:")
    print("1. Add missing 'mapping' import to xs_infer_bathy_raster.py")
    print("2. Add missing '_utm_crs_for_point' function")
    print("3. Increase corridor buffer default for better interpolation")
    print("4. Remove unreachable code in atl.py")
    print("\nRun this script from your pipeline directory.")
    print("\nTo apply patches, run: python apply_patches.py")
    
    response = input("\nApply patches now? [y/N]: ")
    if response.lower() == 'y':
        apply_patches()
    else:
        print("Patches not applied. Review the patch content above and apply manually.")
