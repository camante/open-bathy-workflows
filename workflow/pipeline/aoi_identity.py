from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


def _same_crs(a: Any, b: Any) -> bool:
    return str(a) == str(b)


def _nearly_equal(a: float, b: float, tol: float = 1.0e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _grid_alignment(parent, child) -> tuple[bool, int, int, str | None]:
    if not _same_crs(parent.crs, child.crs):
        return False, 0, 0, "crs_mismatch"
    pt = parent.transform
    ct = child.transform
    if not (_nearly_equal(pt.a, ct.a) and _nearly_equal(pt.e, ct.e) and _nearly_equal(pt.b, ct.b) and _nearly_equal(pt.d, ct.d)):
        return False, 0, 0, "resolution_or_rotation_mismatch"
    if not (_nearly_equal(pt.b, 0.0) and _nearly_equal(pt.d, 0.0)):
        return False, 0, 0, "rotated_grid_not_supported"
    col_float = (ct.c - pt.c) / pt.a if pt.a else float("nan")
    row_float = (ct.f - pt.f) / pt.e if pt.e else float("nan")
    col_off = int(round(col_float))
    row_off = int(round(row_float))
    if not (_nearly_equal(col_float, col_off) and _nearly_equal(row_float, row_off)):
        return False, row_off, col_off, "grid_not_parent_aligned"
    if row_off < 0 or col_off < 0 or row_off + child.height > parent.height or col_off + child.width > parent.width:
        return False, row_off, col_off, "child_outside_parent_extent"
    return True, row_off, col_off, None


def _arrays_mismatch(a: np.ndarray, b: np.ndarray, *, tolerance: float, nodata_a: Any, nodata_b: Any) -> tuple[int, float, tuple[int, int] | None]:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    finite_a = np.isfinite(a_arr)
    finite_b = np.isfinite(b_arr)
    if nodata_a is not None:
        finite_a &= a_arr != nodata_a
    if nodata_b is not None:
        finite_b &= b_arr != nodata_b
    valid_both = finite_a & finite_b
    nodata_mismatch = finite_a ^ finite_b
    value_mismatch = np.zeros(a_arr.shape, dtype=bool)
    max_abs_diff = 0.0
    if np.any(valid_both):
        diffs = np.abs(a_arr[valid_both].astype("float64") - b_arr[valid_both].astype("float64"))
        if diffs.size:
            max_abs_diff = float(np.nanmax(diffs))
        value_mismatch[valid_both] = diffs > float(tolerance)
    mismatch = nodata_mismatch | value_mismatch
    mismatch_count = int(np.count_nonzero(mismatch))
    first = None
    if mismatch_count:
        first_idx = np.argwhere(mismatch)[0]
        first = (int(first_idx[0]), int(first_idx[1]))
    return mismatch_count, max_abs_diff, first


def _compare_aligned_windows(parent, child, *, row_off: int, col_off: int, tolerance: float) -> dict[str, Any]:
    total_pixels = int(child.width * child.height)
    mismatch_pixels = 0
    max_abs_diff = 0.0
    first_mismatch: dict[str, int] | None = None
    for _, child_window in child.block_windows(1):
        parent_window = Window(
            col_off=col_off + child_window.col_off,
            row_off=row_off + child_window.row_off,
            width=child_window.width,
            height=child_window.height,
        )
        child_data = child.read(1, window=child_window, masked=False)
        parent_data = parent.read(1, window=parent_window, masked=False)
        count, block_max, first = _arrays_mismatch(
            parent_data,
            child_data,
            tolerance=tolerance,
            nodata_a=parent.nodata,
            nodata_b=child.nodata,
        )
        mismatch_pixels += count
        max_abs_diff = max(max_abs_diff, block_max)
        if first is not None and first_mismatch is None:
            first_mismatch = {
                "export_row": int(child_window.row_off + first[0]),
                "export_col": int(child_window.col_off + first[1]),
                "parent_row": int(row_off + child_window.row_off + first[0]),
                "parent_col": int(col_off + child_window.col_off + first[1]),
            }
    return {
        "overlap_pixels": total_pixels,
        "mismatch_pixels": int(mismatch_pixels),
        "max_abs_diff": float(max_abs_diff),
        "first_mismatch": first_mismatch,
        "passed": bool(mismatch_pixels == 0 and max_abs_diff <= float(tolerance)),
    }


def compare_export_to_parent(
    *,
    parent_dem: Path,
    export_dem: Path,
    export_receipt: Path | None = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare an AOI export against the exact parent-grid window it claims to represent."""
    parent_path = Path(parent_dem)
    export_path = Path(export_dem)
    if not parent_path.exists():
        raise FileNotFoundError(f"aoi_identity_parent_missing:{parent_path}")
    if not export_path.exists():
        raise FileNotFoundError(f"aoi_identity_export_missing:{export_path}")

    with rasterio.open(parent_path) as parent, rasterio.open(export_path) as export:
        aligned, row_off, col_off, reason = _grid_alignment(parent, export)
        result: dict[str, Any] = {
            "stage": "aoi_identity",
            "check": "export_vs_parent",
            "checked_against_parent": True,
            "parent_dem": str(parent_path),
            "export_dem": str(export_path),
            "export_receipt": str(export_receipt) if export_receipt is not None else None,
            "parent_crs": str(parent.crs),
            "export_crs": str(export.crs),
            "parent_shape": [int(parent.height), int(parent.width)],
            "export_shape": [int(export.height), int(export.width)],
            "export_window": {
                "row_off": int(row_off),
                "col_off": int(col_off),
                "height": int(export.height),
                "width": int(export.width),
            },
            "tolerance": float(tolerance),
            "grid_aligned": bool(aligned),
            "failure_reason": reason,
        }
        if not aligned:
            result.update({
                "overlap_pixels": 0,
                "mismatch_pixels": None,
                "max_abs_diff": None,
                "first_mismatch": None,
                "passed": False,
            })
            return result
        result.update(_compare_aligned_windows(parent, export, row_off=row_off, col_off=col_off, tolerance=tolerance))
        return result


def compare_overlap_pixels(
    *,
    dem_a: Path,
    dem_b: Path,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare overlapping pixels for two rasters on the same unrotated grid."""
    a_path = Path(dem_a)
    b_path = Path(dem_b)
    if not a_path.exists():
        raise FileNotFoundError(f"aoi_identity_dem_a_missing:{a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"aoi_identity_dem_b_missing:{b_path}")

    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        aligned, row_off, col_off, reason = _grid_alignment(a, b)
        result: dict[str, Any] = {
            "stage": "aoi_identity",
            "check": "overlap_pixels",
            "dem_a": str(a_path),
            "dem_b": str(b_path),
            "tolerance": float(tolerance),
            "grid_aligned": bool(aligned),
            "failure_reason": reason,
        }
        if aligned:
            result.update(_compare_aligned_windows(a, b, row_off=row_off, col_off=col_off, tolerance=tolerance))
            return result

        # If neither raster contains the other, still require the same grid geometry and CRS.
        if not _same_crs(a.crs, b.crs):
            result.update({"overlap_pixels": 0, "mismatch_pixels": None, "max_abs_diff": None, "first_mismatch": None, "passed": False})
            return result
        at = a.transform
        bt = b.transform
        if not (_nearly_equal(at.a, bt.a) and _nearly_equal(at.e, bt.e) and _nearly_equal(at.b, 0.0) and _nearly_equal(at.d, 0.0) and _nearly_equal(bt.b, 0.0) and _nearly_equal(bt.d, 0.0)):
            result.update({"overlap_pixels": 0, "mismatch_pixels": None, "max_abs_diff": None, "first_mismatch": None, "passed": False})
            return result
        col_delta = (bt.c - at.c) / at.a if at.a else float("nan")
        row_delta = (bt.f - at.f) / at.e if at.e else float("nan")
        col_b_in_a = int(round(col_delta))
        row_b_in_a = int(round(row_delta))
        if not (_nearly_equal(col_delta, col_b_in_a) and _nearly_equal(row_delta, row_b_in_a)):
            result.update({"overlap_pixels": 0, "mismatch_pixels": None, "max_abs_diff": None, "first_mismatch": None, "passed": False})
            return result
        a_row0 = max(0, row_b_in_a)
        a_col0 = max(0, col_b_in_a)
        b_row0 = max(0, -row_b_in_a)
        b_col0 = max(0, -col_b_in_a)
        height = min(a.height - a_row0, b.height - b_row0)
        width = min(a.width - a_col0, b.width - b_col0)
        if height <= 0 or width <= 0:
            result.update({"overlap_pixels": 0, "mismatch_pixels": None, "max_abs_diff": None, "first_mismatch": None, "passed": False, "failure_reason": "no_overlap"})
            return result
        a_win = Window(a_col0, a_row0, width, height)
        b_win = Window(b_col0, b_row0, width, height)
        a_data = a.read(1, window=a_win, masked=False)
        b_data = b.read(1, window=b_win, masked=False)
        count, block_max, first = _arrays_mismatch(a_data, b_data, tolerance=tolerance, nodata_a=a.nodata, nodata_b=b.nodata)
        result.update({
            "overlap_pixels": int(width * height),
            "mismatch_pixels": int(count),
            "max_abs_diff": float(block_max),
            "first_mismatch": ({"dem_a_row": int(a_row0 + first[0]), "dem_a_col": int(a_col0 + first[1]), "dem_b_row": int(b_row0 + first[0]), "dem_b_col": int(b_col0 + first[1])} if first is not None else None),
            "passed": bool(count == 0 and block_max <= float(tolerance)),
            "failure_reason": None if count == 0 else "pixel_mismatch",
        })
        return result


__all__ = ["compare_export_to_parent", "compare_overlap_pixels"]
