#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared support-point loading utilities.

This module centralizes CSV/Parquet/vector loading for authoritative support,
extra XYZ soundings, and river guidance anchors so orchestration code does not
need to import `atl.py` just to read support points.
"""

from pathlib import Path
from typing import List
import logging
import re

import numpy as np
import pandas as pd
from pyproj import Transformer

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401

log = logging.getLogger("support_points")


def _canonical_support_source(value: object, fallback_path: Path) -> str:
    """Normalize support-point provenance tags into stable, readable source labels."""
    s = str(value).strip() if value is not None else ""
    if not s or s.lower() in {"nan", "none", "unknown"}:
        s = fallback_path.stem
    # Collapse filesystem paths to a stable stem so reports do not depend on host-specific prefixes.
    if "/" in s or "\\" in s:
        s = Path(s).stem
    s = re.sub(r"[^a-zA-Z0-9_:\-]+", "_", s).strip("_")
    return s or fallback_path.stem

def load_extra_xyz_points(xyz_files: List[str], crs: str, aoi_str: str) -> pd.DataFrame:
    """
    Load external XYZ bathymetry data from multiple files.

    Supports CSV, TXT, Parquet, and vector formats. Reprojects from ``crs`` to EPSG:4326
    and clips to ``aoi_str`` (W/E/S/N). If an input already carries a meaningful
    ``source`` column, that provenance is preserved; otherwise rows are labelled as
    ``extra_xyz:<stem>``.

    Args:
        xyz_files: List of file paths
        crs: Input CRS (e.g., "EPSG:4326", "EPSG:32617")
        aoi_str: AOI string "W/E/S/N" in EPSG:4326

    Returns:
        DataFrame with columns: longitude, latitude, depth_m, source
    """
    from pyproj import Transformer
    import geopandas as gpd
    import re

    dfs = []
    W, E, S, N = [float(x) for x in aoi_str.split("/")]

    log.info("Loading XYZ data from %s file(s)", len(xyz_files))
    log.info("Target AOI: W=%.4f, E=%.4f, S=%.4f, N=%.4f", W, E, S, N)
    log.info("Input CRS: %s", crs)

    # Setup coordinate transformation
    transformer = None
    if crs.upper() != "EPSG:4326":
        try:
            # Create transformer from input CRS to WGS84
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            log.info("Created CRS transformer: %s → EPSG:4326", crs)
        except Exception as e:
            log.error("Failed to create CRS transformer: %s", e, exc_info=True)
            log.error("Assuming data is already in EPSG:4326")
            transformer = None

    for i, f in enumerate(xyz_files):
        try:
            filepath = Path(f)
            log.info("Processing file %s/%s: %s", i+1, len(xyz_files), filepath.name)

            # Try different file formats
            tmp = None

            # Try GeoPackage/Shapefile first
            if filepath.suffix.lower() in ['.gpkg', '.shp', '.geojson']:
                try:
                    gdf = gpd.read_file(f)
                    # Extract coordinates
                    tmp = pd.DataFrame({
                        'longitude': gdf.geometry.x,
                        'latitude': gdf.geometry.y
                    })
                    # Look for depth column
                    depth_cols = [c for c in gdf.columns if 'depth' in c.lower() or 'z' in c.lower()]
                    if depth_cols:
                        tmp['depth_m'] = gdf[depth_cols[0]]
                    log.info("Loaded as vector file: %s points", len(tmp))
                except Exception as e:
                    log.debug("Not a vector file: %s", e)

            # Try Parquet first when explicitly requested. This is important for
            # river guidance support, where bathy_main wires authoritative
            # soundings subsets through a .parquet cache for performance.
            if tmp is None and filepath.suffix.lower() == '.parquet':
                try:
                    tmp = pd.read_parquet(f)
                    log.info("Loaded as Parquet: %s points, %s columns", len(tmp), len(tmp.columns))
                except Exception as e:
                    log.error("Failed to read parquet file %s: %s", filepath.name, e)
                    # Do NOT fall through to CSV for .parquet files — binary files
                    # will always fail CSV parsing with confusing codec errors.
                    raise

            # Try CSV/TXT (only for non-parquet, non-vector files)
            if tmp is None:
                tmp = pd.read_csv(f)

                # Try whitespace delimiter if comma fails
                if len(tmp.columns) == 1:
                    tmp = pd.read_csv(f, sep=r'\s+', header=None,
                                     names=['x', 'y', 'z'])

                log.info("Loaded as CSV/TXT: %s points, %s columns", len(tmp), len(tmp.columns))

            # Normalize column names
            tmp.columns = [c.lower() for c in tmp.columns]

            # Resolve already-standard parquet columns before generic renaming.
            if 'longitude' in tmp.columns and 'x_orig' not in tmp.columns:
                tmp['x_orig'] = pd.to_numeric(tmp['longitude'], errors='coerce')
            if 'latitude' in tmp.columns and 'y_orig' not in tmp.columns:
                tmp['y_orig'] = pd.to_numeric(tmp['latitude'], errors='coerce')
            if 'depth_m' in tmp.columns and isinstance(tmp['depth_m'], pd.DataFrame):
                tmp['depth_m'] = tmp['depth_m'].bfill(axis=1).iloc[:, 0]

            # Map columns to standard names with case-insensitive matching
            rename_map = {}
            for c in tmp.columns:
                cl = str(c).lower().strip()

                # Exact matches first (most reliable)
                if cl in ('lon', 'longitude', 'x'):
                    rename_map[c] = 'x_orig'
                elif cl in ('lat', 'latitude', 'y'):
                    rename_map[c] = 'y_orig'
                elif cl in ('z', 'depth', 'depth_m', 'z_m', 'bathy', 'bathymetry'):
                    rename_map[c] = 'depth_m'
                # Substring matching as fallback
                elif 'lon' in cl and 'x_orig' not in rename_map.values():
                    rename_map[c] = 'x_orig'
                elif 'lat' in cl and 'y_orig' not in rename_map.values():
                    rename_map[c] = 'y_orig'
                elif ('depth' in cl or 'bathy' in cl) and 'depth_m' not in rename_map.values():
                    rename_map[c] = 'depth_m'

            tmp = tmp.rename(columns=rename_map)

            # Duplicate standardized columns can happen for parquet subsets that already
            # carry both x/y/z and longitude/latitude/depth_m. Coalesce duplicates explicitly
            # instead of silently keeping whichever duplicate happened to come first.
            for canon in ('x_orig', 'y_orig', 'depth_m'):
                idx = np.where(tmp.columns == canon)[0]
                if len(idx) > 1:
                    cols = tmp.iloc[:, idx].apply(pd.to_numeric, errors='coerce')
                    coalesced = cols.bfill(axis=1).iloc[:, 0]
                    keep_mask = np.ones(len(tmp.columns), dtype=bool)
                    keep_mask[idx] = False
                    tmp = tmp.iloc[:, keep_mask].copy()
                    tmp[canon] = coalesced
                    log.info("Coalesced %d duplicate '%s' columns in %s", len(idx), canon, filepath.name)

            # If any duplicates remain after coalescing, keep the first copy deterministically.
            if tmp.columns.duplicated().any():
                dup_cols = list(tmp.columns[tmp.columns.duplicated()])
                log.info("Dropping remaining duplicate columns: %s", dup_cols)
                tmp = tmp.loc[:, ~tmp.columns.duplicated(keep='first')]

            # Check required columns
            if 'x_orig' not in tmp.columns or 'y_orig' not in tmp.columns:
                log.warning("Skipping %s: missing x/y columns", filepath.name)
                log.warning("Available columns: %s", list(tmp.columns))
                continue

            if 'depth_m' not in tmp.columns:
                # Safety: do not silently treat elevations as depths. If the file appears to contain
                # elevation/height (e.g., NAVD88 orthometric heights), you must convert to *depth below surface*
                # before using it as training data for a depth model.
                elev_like = any("elev" in str(c).lower() or "height" in str(c).lower() for c in tmp.columns)
                if elev_like:
                    log.warning("Skipping %s: appears to contain elevation/height but no depth column.", filepath.name)
                    log.warning("For depth modeling, supply a depth column (below-surface) or pre-convert elevations to depth.")
                else:
                    log.warning("Skipping %s: missing depth column", filepath.name)
                log.warning("Available columns: %s", list(tmp.columns))
                continue


            # Coerce depth_m to numeric early (protect against object/string depths)
            # This prevents downstream np.isfinite / sign logic failures.
            tmp['depth_m'] = pd.to_numeric(tmp['depth_m'], errors='coerce')
            n_bad = int(tmp['depth_m'].isna().sum())
            if n_bad:
                log.warning("Dropping %s rows with non-numeric depth_m in %s", n_bad, filepath.name)
                tmp = tmp.loc[tmp['depth_m'].notna()].copy()
            if len(tmp) == 0:
                log.warning("No valid depth rows remain after coercion in %s", filepath.name)
                continue

            # Drop obvious nodata/sentinel depths before sign inference or QC statistics.
            # River parquet subsets can legitimately carry exported nodata placeholders from
            # upstream raster sampling (for example -999999), which should never participate
            # in sign detection, MAD clipping, or guidance export.
            sentinel_mask = np.isfinite(tmp['depth_m'].values) & (
                np.isin(tmp['depth_m'].values, [-999999.0, -99999.0, -9999.0, 9999.0, 99999.0, 999999.0])
                | (np.abs(tmp['depth_m'].values) >= 1.0e5)
            )
            n_sentinel = int(sentinel_mask.sum())
            if n_sentinel:
                log.warning("Dropping %s sentinel/nodata depth rows in %s before QC", n_sentinel, filepath.name)
                tmp = tmp.loc[~sentinel_mask].copy()
            if len(tmp) == 0:
                log.warning("No valid depth rows remain after dropping sentinel depths in %s", filepath.name)
                continue

            # Apply CRS transformation if needed
            if transformer is not None:
                log.info("Transforming coordinates...")
                x_in = tmp['x_orig'].values
                y_in = tmp['y_orig'].values

                # Transform
                lon, lat = transformer.transform(x_in, y_in)
                tmp['longitude'] = lon
                tmp['latitude'] = lat

                # Sanity check: log range before/after
                log.info("Input range: X=[%.2f, %.2f], Y=[%.2f, %.2f]",
                        x_in.min(), x_in.max(), y_in.min(), y_in.max())
                log.info("Output range: Lon=[%.4f, %.4f], Lat=[%.4f, %.4f]",
                        lon.min(), lon.max(), lat.min(), lat.max())
            else:
                # No transformation needed
                tmp['longitude'] = tmp['x_orig']
                tmp['latitude'] = tmp['y_orig']

            # Ensure depths are negative (below surface)
            depth_vals = tmp['depth_m'].values
            finite_depths = depth_vals[np.isfinite(depth_vals)]

            if len(finite_depths) == 0:
                log.warning("No finite depth values in %s", filepath.name)
                continue

            pct_positive = (finite_depths > 0).sum() / len(finite_depths)
            depth_min, depth_max = finite_depths.min(), finite_depths.max()

            log.info("Depth statistics: min=%.2f, max=%.2f, pct_positive=%.1f%%",
                    depth_min, depth_max, pct_positive * 100)

            # Infer depth sign from distribution and convert to negative-down
            if pct_positive >= 0.9:
                # Mostly positive -> assume depths below surface, convert to negative
                log.info("Converting depths to negative (below surface)")
                tmp['depth_m'] = -np.abs(tmp['depth_m'])
            elif pct_positive <= 0.1:
                # Mostly negative -> already in correct convention
                log.info("Depths already negative (correct convention)")
            else:
                # Mixed signs -> ambiguous, warn user
                log.warning(
                    "[load_extra_xyz]   Mixed depth signs detected (%.1f%% positive). "
                    "Not auto-converting. Please verify depth convention or add --xyz-depth-convention flag.",
                    pct_positive * 100,
                )
                log.warning("Depth range: [%.2f, %.2f] m", depth_min, depth_max)

            # MAD-based outlier guard before high-weight ingestion
            depth_vals_clean = tmp['depth_m'].values
            depth_vals_clean = depth_vals_clean[np.isfinite(depth_vals_clean)]

            if len(depth_vals_clean) > 0:
                # MAD-based outlier detection (robust to extreme values)
                median_depth = np.median(depth_vals_clean)
                # Robust MAD without scipy (1.4826 scales MAD to ~sigma for normal)
                mad_raw = np.median(np.abs(depth_vals_clean - median_depth))
                mad = 1.4826 * mad_raw

                # Define reasonable depth range (configurable)
                # Default: typical bathymetry 0-50m depth
                expected_min_depth = -60.0  # 60m max depth
                expected_max_depth = 5.0    # 5m above surface (for tidal variation)

                # Check for gross outliers
                n_too_deep = (depth_vals_clean < expected_min_depth).sum()
                n_too_shallow = (depth_vals_clean > expected_max_depth).sum()

                if n_too_deep > 0 or n_too_shallow > 0:
                    log.warning(
                        "[load_extra_xyz] QC WARNING: Potential outliers detected in %s",
                        filepath.name,
                    )
                    if n_too_deep > 0:
                        log.warning("  %s points deeper than %sm", n_too_deep, expected_min_depth)
                    if n_too_shallow > 0:
                        log.warning("  %s points shallower than %sm", n_too_shallow, expected_max_depth)

                # MAD-based outlier clipping (5-sigma equivalent)
                if mad > 0:
                    threshold = 5.0 * mad
                    lower_bound = median_depth - threshold
                    upper_bound = median_depth + threshold

                    outlier_mask = (tmp['depth_m'] < lower_bound) | (tmp['depth_m'] > upper_bound)
                    n_outliers = outlier_mask.sum()

                    if n_outliers > 0:
                        pct_outliers = 100.0 * n_outliers / len(tmp)
                        log.warning(
                            "[load_extra_xyz] QC: Removing %d outliers "
                            "(%.1f%%) beyond 12×MAD "
                            "(median=%.2f, MAD=%.2f)",
                            n_outliers, pct_outliers, median_depth, mad,
                        )
                        tmp = tmp[~outlier_mask].copy()

                # Sanity check: AOI coordinate range
                lon_range = tmp['longitude'].max() - tmp['longitude'].min()
                lat_range = tmp['latitude'].max() - tmp['latitude'].min()

                if lon_range > 10.0 or lat_range > 10.0:
                    log.warning(
                        "[load_extra_xyz] QC WARNING: Very large coordinate range "
                        "(lon_range=%.2f deg, lat_range=%.2f deg). "
                        "Check CRS transformation!",
                        lon_range, lat_range,
                    )

            # Clip to AOI
            n_before = len(tmp)
            tmp = tmp[
                (tmp.longitude >= W) & (tmp.longitude <= E) &
                (tmp.latitude >= S) & (tmp.latitude <= N)
            ]
            n_after = len(tmp)

            log.info("AOI clipping: %s → %s points (%.1f%%)", n_before, n_after, n_after/max(n_before,1)*100)

            if not tmp.empty:
                # Preserve per-file provenance so authoritative subsets can be tiered differently.
                # If the input already has a meaningful 'source' column, keep it. Otherwise tag by filename.
                if "source" in tmp.columns and tmp["source"].notna().any():
                    tmp["source"] = tmp["source"].map(lambda v: _canonical_support_source(v, filepath))
                elif "_src_file" in tmp.columns and tmp["_src_file"].notna().any():
                    tmp["source"] = tmp["_src_file"].map(lambda v: _canonical_support_source(v, filepath))
                else:
                    tag = _canonical_support_source(filepath.stem.lower().strip(), filepath)
                    tmp["source"] = f"extra_xyz:{tag}"

                # Keep only required columns
                tmp = tmp[['longitude', 'latitude', 'depth_m', 'source']].copy()

                # Log depth statistics
                depth_stats = tmp['depth_m'].describe()
                log.info("Depth stats: min=%.2f, median=%.2f, max=%.2f m",
                        depth_stats['min'], depth_stats['50%'], depth_stats['max'])

                dfs.append(tmp)
            else:
                log.warning("No points within AOI after clipping")

        except Exception as e:
            log.error("Failed to load %s: %s", f, e)
            import traceback
            log.debug(traceback.format_exc())

    if not dfs:
        log.warning("No XYZ data loaded from any file")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    log.info("XYZ data loaded: %d points | lon [%.4f, %.4f] | lat [%.4f, %.4f] | depth [%.2f, %.2f] m",
             len(result), result.longitude.min(), result.longitude.max(),
             result.latitude.min(), result.latitude.max(),
             result.depth_m.min(), result.depth_m.max())

    return result
