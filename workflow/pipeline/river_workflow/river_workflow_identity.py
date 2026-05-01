from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def sha256_file(path: Path | str | None) -> str | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_identity(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p)}
    if p.exists():
        with rasterio.open(p) as ds:
            out.update({
                "width": int(ds.width),
                "height": int(ds.height),
                "crs": str(ds.crs) if ds.crs is not None else None,
                "transform": [float(x) for x in tuple(ds.transform)[:6]],
                "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                "nodata": None if ds.nodata is None else float(ds.nodata),
                "dtype": str(ds.dtypes[0]) if ds.count else None,
            })
    return out


def export_window_identity(parent_path: Path | str, export_template_path: Path | str) -> dict[str, Any]:
    parent_path = Path(parent_path)
    export_template_path = Path(export_template_path)
    with rasterio.open(parent_path) as parent, rasterio.open(export_template_path) as tmpl:
        if str(parent.crs) != str(tmpl.crs):
            raise ValueError(f"export_template_crs_mismatch:parent={parent.crs}:template={tmpl.crs}")
        window = from_bounds(
            tmpl.bounds.left,
            tmpl.bounds.bottom,
            tmpl.bounds.right,
            tmpl.bounds.top,
            transform=parent.transform,
        ).round_offsets().round_lengths()
        return {
            "parent_path": str(parent_path),
            "export_template_path": str(export_template_path),
            "row_off": int(window.row_off),
            "col_off": int(window.col_off),
            "height": int(window.height),
            "width": int(window.width),
            "parent_width": int(parent.width),
            "parent_height": int(parent.height),
            "export_width": int(tmpl.width),
            "export_height": int(tmpl.height),
            "export_bounds": [float(tmpl.bounds.left), float(tmpl.bounds.bottom), float(tmpl.bounds.right), float(tmpl.bounds.top)],
            "parent_transform": [float(x) for x in tuple(parent.transform)[:6]],
            "export_transform": [float(x) for x in tuple(tmpl.transform)[:6]],
        }


def compare_parent_window_to_export(
    *,
    parent_path: Path | str,
    export_template_path: Path | str,
    export_path: Path | str,
) -> dict[str, Any]:
    parent_path = Path(parent_path)
    export_template_path = Path(export_template_path)
    export_path = Path(export_path)
    window_info = export_window_identity(parent_path, export_template_path)
    with rasterio.open(parent_path) as parent, rasterio.open(export_template_path) as tmpl, rasterio.open(export_path) as export:
        if str(export.crs) != str(tmpl.crs):
            raise ValueError(f"export_dem_crs_mismatch:export={export.crs}:template={tmpl.crs}")
        if export.width != tmpl.width or export.height != tmpl.height:
            raise ValueError(
                f"export_dem_shape_mismatch:export={export.width}x{export.height}:template={tmpl.width}x{tmpl.height}"
            )
        if tuple(export.transform)[:6] != tuple(tmpl.transform)[:6]:
            raise ValueError("export_dem_transform_mismatch_template")
        if window_info["width"] != export.width or window_info["height"] != export.height:
            raise ValueError(
                "parent_window_shape_mismatch_export:"
                f"window={window_info['width']}x{window_info['height']}:export={export.width}x{export.height}"
            )
        parent_window = rasterio.windows.Window(
            col_off=window_info["col_off"],
            row_off=window_info["row_off"],
            width=window_info["width"],
            height=window_info["height"],
        )
        parent_arr = parent.read(1, window=parent_window)
        export_arr = export.read(1)

    equal = np.isclose(parent_arr, export_arr, rtol=0.0, atol=0.0, equal_nan=True)
    mismatch = ~equal
    mismatch_count = int(np.count_nonzero(mismatch))
    max_abs_diff = 0.0
    if mismatch_count:
        diffs = np.abs(parent_arr[mismatch].astype(np.float64) - export_arr[mismatch].astype(np.float64))
        finite = diffs[np.isfinite(diffs)]
        max_abs_diff = float(np.max(finite)) if finite.size else None
    status = "PASS" if mismatch_count == 0 else "FAIL"
    return {
        "status": status,
        "parent_path": str(parent_path),
        "export_template_path": str(export_template_path),
        "export_path": str(export_path),
        "parent_hash": sha256_file(parent_path),
        "export_hash": sha256_file(export_path),
        "export_window": window_info,
        "max_abs_diff_m": max_abs_diff,
        "mismatch_count": mismatch_count,
        "pixel_count": int(export_arr.size),
        "exact_subset_identity": mismatch_count == 0,
    }


def assert_parent_window_matches_export(*, parent_path: Path | str, export_template_path: Path | str, export_path: Path | str) -> dict[str, Any]:
    result = compare_parent_window_to_export(
        parent_path=parent_path,
        export_template_path=export_template_path,
        export_path=export_path,
    )
    if result.get("status") != "PASS":
        raise ValueError(
            "aoi_export_not_exact_parent_subset:"
            f"mismatch_count={result.get('mismatch_count')}:max_abs_diff_m={result.get('max_abs_diff_m')}"
        )
    return result


__all__ = [
    "assert_parent_window_matches_export",
    "compare_parent_window_to_export",
    "export_window_identity",
    "raster_identity",
    "sha256_file",
]
