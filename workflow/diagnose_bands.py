#!/usr/bin/env python3
"""
Diagnostic script to check S2 band values and Stumpf relationship.
Compares composite vs single best date if both are available.

Run from your S2 cache directory:
    python diagnose_bands.py
    
Or specify path:
    python diagnose_bands.py /path/to/s2/cache/
"""
import numpy as np
import sys
from pathlib import Path

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


def analyze_bands(b02_path, b03_path, label=""):
    """Analyze B02/B03 relationship for a given pair of rasters."""
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*60}")
    
    if not b02_path.exists():
        print(f"  ERROR: {b02_path} not found")
        return None
    if not b03_path.exists():
        print(f"  ERROR: {b03_path} not found")
        return None
    
    with rasterio.open(b02_path) as ds02:
        b02 = ds02.read(1)
        print(f"\nB02 (Blue): {b02_path.name}")
        print(f"  Shape: {b02.shape}")
        
    with rasterio.open(b03_path) as ds03:
        b03 = ds03.read(1)
        print(f"B03 (Green): {b03_path.name}")
        print(f"  Shape: {b03.shape}")
    
    # Get valid water pixels (low values, finite)
    valid = (np.isfinite(b02) & np.isfinite(b03) & 
             (b02 > 0.01) & (b02 < 0.3) & 
             (b03 > 0.01) & (b03 < 0.3))
    
    n_valid = valid.sum()
    print(f"\nValid water pixels: {n_valid:,} / {b02.size:,} ({100*n_valid/b02.size:.1f}%)")
    
    if n_valid < 100:
        print("  ERROR: Too few valid pixels!")
        return None
    
    b02_v = b02[valid]
    b03_v = b03[valid]
    
    print(f"\n--- Reflectance Statistics ---")
    print(f"B02 (Blue):  min={b02_v.min():.4f}, p10={np.percentile(b02_v, 10):.4f}, "
          f"p50={np.median(b02_v):.4f}, p90={np.percentile(b02_v, 90):.4f}, max={b02_v.max():.4f}")
    print(f"B03 (Green): min={b03_v.min():.4f}, p10={np.percentile(b03_v, 10):.4f}, "
          f"p50={np.median(b03_v):.4f}, p90={np.percentile(b03_v, 90):.4f}, max={b03_v.max():.4f}")
    
    ratio = np.median(b02_v) / np.median(b03_v)
    print(f"\nMedian B02/B03 ratio: {ratio:.3f}")
    
    if ratio > 1.0:
        print("  ✓ Blue > Green (expected for clear water)")
    else:
        print("  ✗ Green > Blue (seagrass/CDOM-rich water)")
    
    # Compute Stumpf index
    eps = 1e-6
    log_b02 = np.log(np.maximum(b02_v, eps))
    log_b03 = np.log(np.maximum(b03_v, eps))
    stumpf_idx = log_b02 / log_b03
    
    print(f"\n--- Stumpf Index ---")
    print(f"stumpf_idx: min={np.min(stumpf_idx):.4f}, p10={np.percentile(stumpf_idx, 10):.4f}, "
          f"p50={np.median(stumpf_idx):.4f}, p90={np.percentile(stumpf_idx, 90):.4f}, max={np.max(stumpf_idx):.4f}")
    
    # Dynamic range
    stumpf_range = np.percentile(stumpf_idx, 90) - np.percentile(stumpf_idx, 10)
    print(f"  Dynamic range (p90-p10): {stumpf_range:.4f}")
    if stumpf_range > 0.1:
        print("  ✓ Good dynamic range for depth discrimination")
    else:
        print("  ✗ Low dynamic range - Stumpf may not discriminate depth well")
    
    # Sample pixels by brightness (proxy for depth)
    print(f"\n--- Depth Proxy Analysis (brightness) ---")
    brightness = b02_v + b03_v
    
    # Brightest (shallowest)
    bright_mask = brightness > np.percentile(brightness, 90)
    dark_mask = brightness < np.percentile(brightness, 10)
    
    bright_stumpf = np.median(stumpf_idx[bright_mask])
    dark_stumpf = np.median(stumpf_idx[dark_mask])
    
    print(f"Bright pixels (shallow, top 10%): median stumpf_idx = {bright_stumpf:.4f}")
    print(f"Dark pixels (deep, bottom 10%):   median stumpf_idx = {dark_stumpf:.4f}")
    print(f"  Difference: {dark_stumpf - bright_stumpf:.4f}")
    
    if dark_stumpf > bright_stumpf:
        print("  ✓ Stumpf INCREASES with depth (standard relationship)")
        relationship = "positive"
    else:
        print("  ✗ Stumpf DECREASES with depth (inverted relationship)")
        relationship = "negative"
    
    # Band attenuation check
    print(f"\n--- Band Attenuation Check ---")
    bright_ratio = np.median(b02_v[bright_mask]) / np.median(b03_v[bright_mask])
    dark_ratio = np.median(b02_v[dark_mask]) / np.median(b03_v[dark_mask])
    
    print(f"Bright pixels (shallow): B02/B03 = {bright_ratio:.4f}")
    print(f"Dark pixels (deep):      B02/B03 = {dark_ratio:.4f}")
    
    if dark_ratio < bright_ratio:
        print("  ✓ Blue attenuates faster than green (standard clear water)")
    else:
        print("  ✗ Green attenuates faster than blue (CDOM/seagrass effect)")
    
    return {
        "label": label,
        "n_valid": n_valid,
        "b02_median": np.median(b02_v),
        "b03_median": np.median(b03_v),
        "ratio": ratio,
        "stumpf_median": np.median(stumpf_idx),
        "stumpf_range": stumpf_range,
        "bright_stumpf": bright_stumpf,
        "dark_stumpf": dark_stumpf,
        "relationship": relationship,
    }


def main():
    # Find directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(".")
    
    # Look for band files
    composite_b02 = base_dir / "B02_10m.tif"
    composite_b03 = base_dir / "B03_10m.tif"
    best_b02 = base_dir / "B02_10m_best_date.tif"
    best_b03 = base_dir / "B03_10m_best_date.tif"
    
    print("="*60)
    print("SENTINEL-2 BAND DIAGNOSTIC")
    print("="*60)
    print(f"Directory: {base_dir.absolute()}")
    
    results = []
    
    # Analyze composite
    if composite_b02.exists() and composite_b03.exists():
        r = analyze_bands(composite_b02, composite_b03, "COMPOSITE (temporal median)")
        if r:
            results.append(r)
    else:
        print(f"\n[SKIP] Composite bands not found")
    
    # Analyze best date
    if best_b02.exists() and best_b03.exists():
        r = analyze_bands(best_b02, best_b03, "SINGLE BEST DATE")
        if r:
            results.append(r)
    else:
        print(f"\n[SKIP] Best date bands not found")
    
    # Comparison summary
    if len(results) == 2:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        comp = results[0]
        best = results[1]
        
        print(f"\n{'Metric':<30} {'Composite':>15} {'Best Date':>15} {'Winner':>12}")
        print("-"*75)
        
        # Stumpf range (higher is better)
        winner = "Best Date" if best["stumpf_range"] > comp["stumpf_range"] else "Composite"
        print(f"{'Stumpf dynamic range':<30} {comp['stumpf_range']:>15.4f} {best['stumpf_range']:>15.4f} {winner:>12}")
        
        # Depth discrimination (larger difference is better)
        comp_diff = abs(comp["dark_stumpf"] - comp["bright_stumpf"])
        best_diff = abs(best["dark_stumpf"] - best["bright_stumpf"])
        winner = "Best Date" if best_diff > comp_diff else "Composite"
        print(f"{'Depth discrimination':<30} {comp_diff:>15.4f} {best_diff:>15.4f} {winner:>12}")
        
        # Valid pixels (higher is better for coverage)
        winner = "Best Date" if best["n_valid"] > comp["n_valid"] else "Composite"
        print(f"{'Valid pixels':<30} {comp['n_valid']:>15,} {best['n_valid']:>15,} {winner:>12}")
        
        # Relationship
        print(f"{'Stumpf-depth relationship':<30} {comp['relationship']:>15} {best['relationship']:>15}")
        
        print("\n" + "-"*75)
        if best["stumpf_range"] > comp["stumpf_range"] * 1.1:
            print("RECOMMENDATION: Use --single-best-date for better spectral contrast")
        elif comp["n_valid"] > best["n_valid"] * 1.2:
            print("RECOMMENDATION: Use composite for better spatial coverage")
        else:
            print("RECOMMENDATION: Both are similar; composite provides more robustness")
    
    elif len(results) == 1:
        print(f"\n[INFO] Only one dataset available for analysis")
    else:
        print(f"\n[ERROR] No valid band data found!")
        print(f"  Looking for: B02_10m.tif, B03_10m.tif")
        print(f"           or: B02_10m_best_date.tif, B03_10m_best_date.tif")


if __name__ == "__main__":
    main()
