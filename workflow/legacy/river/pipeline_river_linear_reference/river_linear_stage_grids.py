from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from pyproj import Transformer
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds, transform as window_transform

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_stage_solve_domain import SolveDomainStageResult
from pipeline.river_linear.river_linear_validation import (
    validate_export_aoi_contained_by_canonical_solve_aoi,
    validate_grid_alignment,
    validate_grid_template,
    validate_parent_grid_contains_bounds,
    validate_solve_contains_export,
)


@dataclass(frozen=True)
class GridStageResult:
    solve_grid_template_path: Path
    export_grid_template_path: Path
    solve_shape: tuple[int, int]
    export_shape: tuple[int, int]
    resolution_m: float
    crs: str
    export_aoi_contained_in_canonical_solve: bool = True
    receipt_path: Path | None = None



def _parse_aoi(aoi: str) -> tuple[float, float, float, float]:
    west, east, south, north = [float(part) for part in str(aoi).split("/")]
    return west, east, south, north



def _project_bounds(aoi: str, dst_crs: str) -> tuple[float, float, float, float]:
    west, east, south, north = _parse_aoi(aoi)
    tx = Transformer.from_crs("EPSG:4269", dst_crs, always_xy=True)
    xs = [west, east, west, east]
    ys = [south, south, north, north]
    px, py = tx.transform(xs, ys)
    return float(min(px)), float(min(py)), float(max(px)), float(max(py))





def _union_bounds(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def _aligned_bounds(bounds: tuple[float, float, float, float], resolution_m: float) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    res = float(resolution_m)
    aw = math.floor(west / res) * res
    as_ = math.floor(south / res) * res
    ae = math.ceil(east / res) * res
    an = math.ceil(north / res) * res
    return aw, as_, ae, an



def _write_template(path: Path, *, bounds: tuple[float, float, float, float], crs: str, resolution_m: float) -> tuple[int, int]:
    west, south, east, north = _aligned_bounds(bounds, resolution_m)
    width = max(1, int(round((east - west) / float(resolution_m))))
    height = max(1, int(round((north - south) / float(resolution_m))))
    transform = from_origin(west, north, float(resolution_m), float(resolution_m))
    arr = np.full((height, width), -9999.0, dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
    ) as ds:
        ds.write(arr, 1)
    return height, width


def _round_window(window: Window) -> Window:
    return Window(
        col_off=int(math.floor(float(window.col_off))),
        row_off=int(math.floor(float(window.row_off))),
        width=int(math.ceil(float(window.width))),
        height=int(math.ceil(float(window.height))),
    )


def _write_subset_template_from_parent(path: Path, *, parent_grid_path: Path, bounds: tuple[float, float, float, float]) -> tuple[int, int]:
    validate_parent_grid_contains_bounds(parent_grid_path=parent_grid_path, requested_bounds=bounds, label='export')
    with rasterio.open(parent_grid_path) as parent_ds:
        window = _round_window(from_bounds(*bounds, transform=parent_ds.transform))
        if int(window.col_off) < 0 or int(window.row_off) < 0:
            raise ValueError('linear_export_window_outside_parent_grid_negative_offset')
        if int(window.col_off) + int(window.width) > int(parent_ds.width) or int(window.row_off) + int(window.height) > int(parent_ds.height):
            raise ValueError('linear_export_window_outside_parent_grid_extent')
        transform = window_transform(window, parent_ds.transform)
        profile = parent_ds.profile.copy()
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(
            width=int(window.width),
            height=int(window.height),
            count=1,
            dtype="float32",
            nodata=-9999.0,
            transform=transform,
            compress="deflate",
        )
        arr = np.full((int(window.height), int(window.width)), -9999.0, dtype="float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(arr, 1)
    return int(window.height), int(window.width)



def run_grid_stage(ctx: RiverLinearContext, solve_result: SolveDomainStageResult) -> GridStageResult:
    validate_export_aoi_contained_by_canonical_solve_aoi(
        export_aoi=ctx.export_aoi,
        canonical_solve_aoi=solve_result.canonical_solve_aoi,
    )
    canonical_grid_path = (Path(ctx.linear_inputs.canonical_solve_grid_template_path) if ctx.linear_inputs is not None and ctx.linear_inputs.canonical_solve_grid_template_path is not None else None)
    if canonical_grid_path is not None and canonical_grid_path.exists():
        solve_grid_path = canonical_grid_path
        with rasterio.open(solve_grid_path) as ds:
            solve_shape = (int(ds.height), int(ds.width))
    else:
        solve_bounds = _project_bounds(solve_result.canonical_solve_aoi, ctx.projected_crs)
        solve_grid_path = ctx.paths.solve_grid_template
        solve_shape = _write_template(
            solve_grid_path,
            bounds=solve_bounds,
            crs=ctx.projected_crs,
            resolution_m=ctx.target_resolution_m,
        )
    export_shape = _write_subset_template_from_parent(
        ctx.paths.export_grid_template,
        parent_grid_path=solve_grid_path,
        bounds=_project_bounds(ctx.export_aoi, ctx.projected_crs),
    )
    validate_grid_template(solve_grid_path, expected_crs=ctx.projected_crs, expected_resolution_m=ctx.target_resolution_m)
    validate_grid_template(ctx.paths.export_grid_template, expected_crs=ctx.projected_crs, expected_resolution_m=ctx.target_resolution_m)
    validate_solve_contains_export(solve_grid_path=solve_grid_path, export_grid_path=ctx.paths.export_grid_template)
    validate_grid_alignment(solve_grid_path=solve_grid_path, export_grid_path=ctx.paths.export_grid_template)
    return GridStageResult(
        solve_grid_template_path=solve_grid_path,
        export_grid_template_path=ctx.paths.export_grid_template,
        solve_shape=solve_shape,
        export_shape=export_shape,
        resolution_m=float(ctx.target_resolution_m),
        crs=str(ctx.projected_crs),
        export_aoi_contained_in_canonical_solve=True,
    )


__all__ = ["GridStageResult", "run_grid_stage"]
