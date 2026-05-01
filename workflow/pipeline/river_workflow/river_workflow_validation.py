from __future__ import annotations

import json
from math import isclose
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box


def _parse_aoi(aoi: str) -> tuple[float, float, float, float]:
    west, east, south, north = [float(part) for part in str(aoi).split("/")]
    if not (west < east and south < north):
        raise ValueError("invalid_aoi")
    return west, east, south, north




def validate_export_aoi_contained_by_canonical_solve_aoi(*, export_aoi: str, canonical_solve_aoi: str, tolerance: float = 1e-12) -> None:
    e_west, e_east, e_south, e_north = _parse_aoi(export_aoi)
    s_west, s_east, s_south, s_north = _parse_aoi(canonical_solve_aoi)
    if (e_west < s_west - tolerance or e_east > s_east + tolerance or e_south < s_south - tolerance or e_north > s_north + tolerance):
        raise ValueError(
            'linear_export_aoi_outside_canonical_solve:'
            f'export_aoi={export_aoi};canonical_solve_aoi={canonical_solve_aoi}'
        )


def validate_parent_grid_contains_bounds(*, parent_grid_path: Path, requested_bounds: tuple[float, float, float, float], label: str = 'export') -> None:
    with rasterio.open(parent_grid_path) as parent_ds:
        left, bottom, right, top = parent_ds.bounds
    req_left, req_bottom, req_right, req_top = requested_bounds
    tol = 1e-9
    if req_left < left - tol or req_right > right + tol or req_bottom < bottom - tol or req_top > top + tol:
        raise ValueError(
            f'linear_{label}_bounds_outside_parent_grid:'
            f'requested_bounds={requested_bounds};parent_bounds={(left, bottom, right, top)}'
        )

def validate_network_gpkg_exists_and_nonempty(path: Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(str(path))
    layers = gpd.list_layers(path)
    if layers.empty:
        raise ValueError("linear_network_no_layers")
    first = str(layers.iloc[0]["name"])
    gdf = gpd.read_file(path, layer=first)
    if gdf.empty:
        raise ValueError("linear_network_empty")




def validate_linear_canonical_network_gpkg(path: Path) -> None:
    validate_network_gpkg_exists_and_nonempty(path)
    layers = gpd.list_layers(path)
    names = set(layers["name"].tolist())
    if "linear_flows" not in names:
        raise ValueError("linear_canonical_network_missing_linear_flows")
    flows = gpd.read_file(path, layer="linear_flows")
    if flows.empty:
        raise ValueError("linear_canonical_network_empty_flows")

def validate_solve_aoi_json(path: Path, *, export_aoi: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    solve = payload.get("canonical_solve_aoi") or payload.get("solve_aoi")
    if not isinstance(solve, str):
        raise ValueError("linear_missing_canonical_solve_aoi")
    _parse_aoi(solve)
    _parse_aoi(export_aoi)


def validate_grid_template(path: Path, *, expected_crs: str, expected_resolution_m: float) -> None:
    with rasterio.open(path) as ds:
        if ds.width <= 0 or ds.height <= 0:
            raise ValueError("linear_invalid_grid_shape")
        if ds.crs is None or ds.crs.to_string() != str(expected_crs):
            raise ValueError("linear_invalid_grid_crs")
        xres, yres = ds.res
        if not isclose(float(xres), float(expected_resolution_m), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("linear_invalid_grid_xres")
        if not isclose(abs(float(yres)), float(expected_resolution_m), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("linear_invalid_grid_yres")


def validate_solve_contains_export(*, solve_grid_path: Path, export_grid_path: Path) -> None:
    with rasterio.open(solve_grid_path) as solve_ds, rasterio.open(export_grid_path) as export_ds:
        s_left, s_bottom, s_right, s_top = solve_ds.bounds
        e_left, e_bottom, e_right, e_top = export_ds.bounds
        if s_left > e_left or s_right < e_right or s_bottom > e_bottom or s_top < e_top:
            raise ValueError("linear_solve_grid_does_not_contain_export_grid")


def validate_grid_alignment(*, solve_grid_path: Path, export_grid_path: Path) -> None:
    with rasterio.open(solve_grid_path) as solve_ds, rasterio.open(export_grid_path) as export_ds:
        if solve_ds.crs != export_ds.crs:
            raise ValueError("linear_grid_crs_mismatch")
        sx, sy = solve_ds.res
        ex, ey = export_ds.res
        if not (isclose(float(sx), float(ex), abs_tol=1e-6) and isclose(abs(float(sy)), abs(float(ey)), abs_tol=1e-6)):
            raise ValueError("linear_grid_resolution_mismatch")
        s0x, s0y = solve_ds.transform.c, solve_ds.transform.f
        e0x, e0y = export_ds.transform.c, export_ds.transform.f
        if not isclose((e0x - s0x) / float(sx), round((e0x - s0x) / float(sx)), abs_tol=1e-6):
            raise ValueError("linear_grid_x_alignment_mismatch")
        if not isclose((s0y - e0y) / abs(float(sy)), round((s0y - e0y) / abs(float(sy))), abs_tol=1e-6):
            raise ValueError("linear_grid_y_alignment_mismatch")


def validate_raster_matches_template(*, raster_path: Path, template_path: Path) -> None:
    with rasterio.open(raster_path) as ds, rasterio.open(template_path) as template_ds:
        if ds.crs != template_ds.crs:
            raise ValueError("linear_raster_template_crs_mismatch")
        if ds.transform != template_ds.transform:
            raise ValueError("linear_raster_template_transform_mismatch")
        if ds.width != template_ds.width or ds.height != template_ds.height:
            raise ValueError("linear_raster_template_shape_mismatch")


def validate_support_mask_matches_measured_only(*, measured_path: Path, support_mask_path: Path) -> None:
    with rasterio.open(measured_path) as measured_ds, rasterio.open(support_mask_path) as mask_ds:
        measured = measured_ds.read(1)
        mask = mask_ds.read(1)
        nodata = measured_ds.nodata
        valid = np.isfinite(measured)
        if nodata is not None:
            valid &= measured != nodata
        mask_bool = mask > 0
        if valid.shape != mask_bool.shape:
            raise ValueError("linear_support_mask_shape_mismatch")
        if not np.array_equal(valid, mask_bool):
            raise ValueError("linear_support_mask_not_equal_to_measured_validity")


def validate_background_covers_measured_only(*, measured_path: Path, background_path: Path) -> None:
    with rasterio.open(measured_path) as measured_ds, rasterio.open(background_path) as background_ds:
        measured = measured_ds.read(1)
        background = background_ds.read(1)
        measured_nodata = measured_ds.nodata
        background_nodata = background_ds.nodata
        measured_valid = np.isfinite(measured)
        background_valid = np.isfinite(background)
        if measured_nodata is not None:
            measured_valid &= measured != measured_nodata
        if background_nodata is not None:
            background_valid &= background != background_nodata
        if np.any(measured_valid & ~background_valid):
            raise ValueError("linear_background_missing_measured_cells")


def validate_centerline_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {"point_id", "station_m", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_centerline_missing_fields:{sorted(missing)}")
    if gdf.empty:
        raise ValueError("linear_centerline_empty")




def validate_centerline_points_overlap_template(*, path: Path, template_path: Path, min_points: int = 10) -> None:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError('linear_centerline_empty')
    with rasterio.open(template_path) as ds:
        bounds = ds.bounds
        bounds_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    intersects = np.asarray(gdf.geometry.intersects(bounds_poly), dtype=bool)
    overlap_count = int(intersects.sum())
    if overlap_count <= 0:
        raise ValueError('linear_centerline_no_export_overlap')
    if overlap_count < int(max(min_points, 1)):
        raise ValueError(f'linear_centerline_insufficient_export_overlap:{overlap_count}')

def validate_wse_proxy_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {
        "point_id", "station_m", "station_downstream_m",
        "wse_proxy_z_m", "wse_profile_z_m",
        "wse_support_count", "wse_support_distance_m",
        "wse_profile_method", "wse_direction", "wse_confidence",
        "geometry",
    }
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_wse_proxy_missing_fields:{sorted(missing)}")
    vals = np.asarray(pd.to_numeric(gdf["wse_proxy_z_m"], errors="coerce"), dtype=float)
    if not np.isfinite(vals).any():
        raise ValueError("linear_wse_proxy_all_nan")

    group_fields = [c for c in ("component_id", "levelpath_id", "reach_id", "source_reach_key") if c in gdf.columns]
    if group_fields:
        group_keys = gdf[group_fields].astype(str).agg("|".join, axis=1)
    else:
        group_keys = pd.Series(["all"] * len(gdf), index=gdf.index, dtype=object)
    for _, idx in group_keys.groupby(group_keys).groups.items():
        grp = gdf.loc[list(idx)].copy()
        grp = grp.sort_values("station_downstream_m", kind="mergesort")
        wse = np.asarray(pd.to_numeric(grp["wse_proxy_z_m"], errors="coerce"), dtype=float)
        finite = np.isfinite(wse)
        if np.count_nonzero(finite) <= 1:
            continue
        diffs = np.diff(wse[finite])
        if np.any(diffs > 1.0e-6):
            raise ValueError("linear_wse_proxy_increases_downstream")

def validate_authoritative_bed_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {"point_id", "station_m", "authoritative_bed_z_m", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_authoritative_bed_missing_fields:{sorted(missing)}")
    if len(gdf) == 0:
        return
    vals = np.asarray(pd.to_numeric(gdf["authoritative_bed_z_m"], errors="coerce"), dtype=float)
    if not np.isfinite(vals).any():
        raise ValueError("linear_authoritative_bed_all_nan")


def validate_observed_offset_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {"point_id", "station_m", "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_observed_offset_missing_fields:{sorted(missing)}")
    vals = np.asarray(pd.to_numeric(gdf["observed_offset_m"], errors="coerce"), dtype=float)
    if not np.isfinite(vals).any():
        raise ValueError("linear_observed_offset_all_nan")



def validate_modeled_offset_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {"point_id", "station_m", "wse_proxy_z_m", "offset_modeled_m", "offset_source", "offset_raw_bed_z_m", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_modeled_offset_missing_fields:{sorted(missing)}")
    vals = np.asarray(pd.to_numeric(gdf["offset_modeled_m"], errors="coerce"), dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("linear_modeled_offset_nonfinite")
    if np.any(vals <= 0.0):
        raise ValueError("linear_modeled_offset_nonpositive")


def validate_backbone_points_gpkg(path: Path) -> None:
    gdf = gpd.read_file(path)
    required = {"point_id", "station_m", "wse_proxy_z_m", "offset_modeled_m", "offset_raw_bed_z_m", "bed_backbone_z_m", "bed_backbone_raw_z_m", "backbone_formula_guard_status", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"linear_backbone_missing_fields:{sorted(missing)}")
    vals = np.asarray(pd.to_numeric(gdf["bed_backbone_z_m"], errors="coerce"), dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("linear_backbone_nonfinite")


def validate_surface_raster(*, raster_path: Path, template_path: Path) -> None:
    validate_raster_matches_template(raster_path=raster_path, template_path=template_path)
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        finite = np.isfinite(arr)
        if nodata is not None:
            finite &= arr != nodata
        if not np.any(finite):
            raise ValueError("linear_surface_all_nodata")


def validate_locked_surface_raster(*, locked_raster_path: Path, unlocked_raster_path: Path, measured_path: Path, support_mask_path: Path, template_path: Path) -> None:
    validate_raster_matches_template(raster_path=locked_raster_path, template_path=template_path)
    validate_raster_matches_template(raster_path=unlocked_raster_path, template_path=template_path)
    with rasterio.open(locked_raster_path) as locked_ds, rasterio.open(unlocked_raster_path) as unlocked_ds, rasterio.open(measured_path) as measured_ds, rasterio.open(support_mask_path) as support_ds:
        locked = locked_ds.read(1)
        unlocked = unlocked_ds.read(1)
        measured = measured_ds.read(1)
        support = support_ds.read(1) > 0
        measured_nodata = measured_ds.nodata
        measured_valid = np.isfinite(measured)
        if measured_nodata is not None:
            measured_valid &= measured != measured_nodata
        lock_mask = support & measured_valid
        if np.any(lock_mask):
            if not np.allclose(locked[lock_mask], measured[lock_mask], equal_nan=False):
                raise ValueError("linear_locked_surface_does_not_match_measured_support")
        unchanged_mask = ~lock_mask
        if np.any(unchanged_mask):
            if not np.allclose(locked[unchanged_mask], unlocked[unchanged_mask], equal_nan=True):
                raise ValueError("linear_locked_surface_changed_outside_support")


def validate_mask_raster(*, raster_path: Path, template_path: Path) -> None:
    validate_raster_matches_template(raster_path=raster_path, template_path=template_path)
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1)
        vals = np.unique(arr)
        allowed = {0, 1}
        if not set(int(v) for v in vals.tolist()).issubset(allowed):
            raise ValueError("linear_mask_not_binary")




def validate_final_dem_raster(*, final_dem_path: Path, measured_path: Path, support_mask_path: Path, background_path: Path, guidance_path: Path, take_mask_path: Path, template_path: Path) -> None:
    validate_raster_matches_template(raster_path=final_dem_path, template_path=template_path)
    validate_mask_raster(raster_path=support_mask_path, template_path=template_path)
    validate_mask_raster(raster_path=take_mask_path, template_path=template_path)
    with rasterio.open(final_dem_path) as final_ds, rasterio.open(measured_path) as measured_ds, rasterio.open(support_mask_path) as support_ds, rasterio.open(background_path) as background_ds, rasterio.open(guidance_path) as guidance_ds, rasterio.open(take_mask_path) as take_ds:
        final_arr = final_ds.read(1)
        measured = measured_ds.read(1)
        support = support_ds.read(1) > 0
        background = background_ds.read(1)
        guidance = guidance_ds.read(1)
        take = take_ds.read(1) > 0
        final_nodata = final_ds.nodata
        measured_nodata = measured_ds.nodata
        background_nodata = background_ds.nodata
        guidance_nodata = guidance_ds.nodata

        measured_valid = np.isfinite(measured)
        if measured_nodata is not None:
            measured_valid &= measured != measured_nodata
        guidance_valid = np.isfinite(guidance)
        if guidance_nodata is not None:
            guidance_valid &= guidance != guidance_nodata
        background_valid = np.isfinite(background)
        if background_nodata is not None:
            background_valid &= background != background_nodata

        support_apply = support & measured_valid
        guidance_apply = take & guidance_valid & ~support_apply
        background_apply = ~(support_apply | guidance_apply)

        if np.any(support_apply) and not np.allclose(final_arr[support_apply], measured[support_apply], equal_nan=False):
            raise ValueError('linear_final_dem_does_not_match_measured_support')
        if np.any(guidance_apply) and not np.allclose(final_arr[guidance_apply], guidance[guidance_apply], equal_nan=False):
            raise ValueError('linear_final_dem_does_not_match_river_guidance_take_mask')
        if np.any(background_apply) and not np.allclose(final_arr[background_apply], background[background_apply], equal_nan=True):
            raise ValueError('linear_final_dem_does_not_match_background_elsewhere')

        final_valid = np.isfinite(final_arr)
        if final_nodata is not None:
            final_valid &= final_arr != final_nodata
        expected_valid = support_apply | guidance_apply | background_valid
        if not np.array_equal(final_valid, expected_valid):
            raise ValueError('linear_final_dem_validity_mismatch')

def validate_take_mask_raster(*, take_mask_path: Path, corridor_mask_path: Path, support_mask_path: Path, template_path: Path) -> None:
    validate_mask_raster(raster_path=take_mask_path, template_path=template_path)
    validate_mask_raster(raster_path=corridor_mask_path, template_path=template_path)
    validate_mask_raster(raster_path=support_mask_path, template_path=template_path)
    with rasterio.open(take_mask_path) as take_ds, rasterio.open(corridor_mask_path) as corridor_ds, rasterio.open(support_mask_path) as support_ds:
        take = take_ds.read(1) > 0
        corridor = corridor_ds.read(1) > 0
        support = support_ds.read(1) > 0
        expected = corridor & ~support
        if not np.array_equal(take, expected):
            raise ValueError("linear_take_mask_not_equal_to_corridor_minus_support")


__all__ = [
    "validate_grid_alignment",
    "validate_grid_template",
    "validate_network_gpkg_exists_and_nonempty",
    "validate_raster_matches_template",
    "validate_solve_aoi_json",
    "validate_support_mask_matches_measured_only",
    "validate_solve_contains_export",
    "validate_export_aoi_contained_by_canonical_solve_aoi",
    "validate_parent_grid_contains_bounds",
    "validate_background_covers_measured_only",
    "validate_centerline_points_gpkg",
    "validate_centerline_points_overlap_template",
    "validate_wse_proxy_points_gpkg",
    "validate_authoritative_bed_points_gpkg",
    "validate_observed_offset_points_gpkg",
    "validate_modeled_offset_points_gpkg",
    "validate_backbone_points_gpkg",
    "validate_surface_raster",
    "validate_locked_surface_raster",
    "validate_mask_raster",
    "validate_take_mask_raster",
    "validate_final_dem_raster",
]
