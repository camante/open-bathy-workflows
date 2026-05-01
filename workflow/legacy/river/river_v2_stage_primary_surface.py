from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult, STAGE_RIVER_PRIMARY_SURFACE
from legacy.river.river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt
from legacy.river.river_v2_validation import ensure_existing_path, validate_primary_surface_raster


_REQUIRED_FIELDS = ("point_id", "station_m", "bed_backbone_z_m", "geometry")
_GROUP_COLS = ("component_id", "levelpath_id", "reach_id", "source_reach_key")


class GridSpec:
    def __init__(self, width: int, height: int, transform, crs, nodata: float, dtype: str):
        self.width = int(width)
        self.height = int(height)
        self.transform = transform
        self.crs = crs
        self.nodata = float(nodata)
        self.dtype = str(dtype)


def resolve_primary_surface_target_grid(ctx: RiverV2Context, domain_mask_path: Path) -> GridSpec:
    with rasterio.open(domain_mask_path) as ds:
        nodata = float(ds.nodata) if ds.nodata is not None else 0.0
        return GridSpec(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs, nodata=nodata, dtype=ds.dtypes[0])


def compute_distance_to_seed_cells(valid_mask: np.ndarray, seed_presence: np.ndarray, transform) -> np.ndarray:
    if valid_mask.shape != seed_presence.shape:
        raise RuntimeError("river_v2_primary_support_shape_mismatch")
    seed_count = int(np.count_nonzero(seed_presence))
    if seed_count == 0:
        raise RuntimeError("river_v2_primary_support_no_backbone_cells_on_grid")
    in_domain_seed_count = int(np.count_nonzero(seed_presence & valid_mask))
    # Be tolerant when the backbone seed cells land just outside the active domain mask.
    # This can happen when the solve/support domain extends slightly beyond the final
    # river-channel mask or where rasterization semantics differ by one pixel. In that
    # case we still build the distance field from the nearest backbone cells anywhere on
    # the grid and then mask the final surface back to the valid river domain.
    seed_support = seed_presence & valid_mask if in_domain_seed_count > 0 else seed_presence
    sampling = (abs(float(transform.e)), abs(float(transform.a)))
    background = ~seed_support
    dist = distance_transform_edt(background, sampling=sampling).astype("float32")
    dist[~valid_mask] = np.nan
    dist[seed_support] = 0.0
    return dist


def write_primary_surface_domain_raster(*, primary_surface_domain: np.ndarray, grid_spec: GridSpec, domain_path: Path) -> Path:
    domain_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid_spec.width,
        "height": grid_spec.height,
        "count": 1,
        "crs": grid_spec.crs,
        "transform": grid_spec.transform,
        "compress": "deflate",
        "tiled": False,
    }
    with rasterio.open(domain_path, "w", dtype="uint8", nodata=0, **profile) as ds:
        ds.write(primary_surface_domain.astype("uint8"), 1)
    if not domain_path.exists():
        raise RuntimeError("river_v2_primary_support_write_failed")
    return domain_path


def validate_primary_surface_domain(domain_path: Path, *, expected_shape: tuple[int, int], expected_transform, expected_crs) -> dict[str, Any]:
    with rasterio.open(domain_path) as ds:
        domain = ds.read(1)
        valid = np.isfinite(domain) & (domain > 0)
        return {
            "valid": bool(ds.shape == expected_shape and ds.transform == expected_transform and ds.crs == expected_crs and np.count_nonzero(valid) > 0),
            "grid_shape": [int(ds.height), int(ds.width)],
            "shape_match": bool(ds.shape == expected_shape),
            "transform_match": bool(ds.transform == expected_transform),
            "crs_match": bool(ds.crs == expected_crs),
            "domain_cell_count": int(np.count_nonzero(valid)),
        }


def _numeric(values: pd.Series | Any) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def load_backbone_points(backbone_points_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(backbone_points_path)
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    if missing:
        raise RuntimeError(f"river_v2_primary_surface_missing_fields:{missing}")
    if len(gdf) == 0:
        raise RuntimeError("river_v2_primary_surface_no_backbone_points")
    gdf = gdf.copy()
    gdf["point_id"] = gdf["point_id"].astype(str)
    gdf["station_m"] = pd.to_numeric(gdf["station_m"], errors="coerce")
    gdf["bed_backbone_z_m"] = pd.to_numeric(gdf["bed_backbone_z_m"], errors="coerce")
    gdf = gdf.sort_values(["station_m", "point_id"], kind="mergesort").reset_index(drop=True)
    return gdf


def load_centerline_points(centerline_points_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(centerline_points_path)
    required = ("point_id", "station_m", "geometry")
    missing = [f for f in required if f not in gdf.columns]
    if missing:
        raise RuntimeError(f"river_v2_primary_surface_centerline_missing_fields:{missing}")
    if len(gdf) == 0:
        raise RuntimeError("river_v2_primary_surface_centerline_empty")
    gdf = gdf.copy()
    gdf["point_id"] = gdf["point_id"].astype(str)
    gdf["station_m"] = pd.to_numeric(gdf["station_m"], errors="coerce")
    keep = [c for c in ("point_id", "station_m", "geometry") if c in gdf.columns]
    gdf = gpd.GeoDataFrame(gdf[keep].copy(), geometry="geometry", crs=gdf.crs)
    gdf = gdf.sort_values(["station_m", "point_id"], kind="mergesort").reset_index(drop=True)
    return gdf


def prepare_primary_support_points(*, backbone_points_path: Path, centerline_points_path: Path | None) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    backbone = load_backbone_points(Path(backbone_points_path))
    diagnostics: dict[str, Any] = {
        "geometry_source": "backbone_points",
        "backbone_input_crs": str(backbone.crs) if backbone.crs is not None else None,
        "backbone_input_count": int(len(backbone)),
    }
    if centerline_points_path is None or not Path(centerline_points_path).exists():
        return backbone, diagnostics
    centerline = load_centerline_points(Path(centerline_points_path))
    diagnostics["centerline_input_crs"] = str(centerline.crs) if centerline.crs is not None else None
    diagnostics["centerline_input_count"] = int(len(centerline))
    merged = centerline.merge(
        pd.DataFrame(backbone.drop(columns="geometry")),
        on=["point_id", "station_m"],
        how="inner",
        suffixes=("_centerline", "_backbone"),
    )
    if len(merged) == 0:
        diagnostics["centerline_backbone_join_count"] = 0
        diagnostics["geometry_source"] = "backbone_points_fallback_empty_join"
        return backbone, diagnostics
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=centerline.crs)
    merged_gdf["bed_backbone_z_m"] = pd.to_numeric(merged_gdf["bed_backbone_z_m"], errors="coerce")
    merged_gdf = merged_gdf.sort_values(["station_m", "point_id"], kind="mergesort").reset_index(drop=True)
    diagnostics["centerline_backbone_join_count"] = int(len(merged_gdf))
    diagnostics["geometry_source"] = "centerline_points_joined_with_backbone_values"
    return merged_gdf, diagnostics




def _group_cols_for_transfer(backbone_gdf: gpd.GeoDataFrame, centerline_gdf: gpd.GeoDataFrame) -> list[str]:
    return [c for c in _GROUP_COLS if c in backbone_gdf.columns and c in centerline_gdf.columns]


def _station_transfer_backbone_to_centerline(backbone_gdf: gpd.GeoDataFrame, centerline_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    groups = _group_cols_for_transfer(backbone_gdf, centerline_gdf)
    if groups:
        backbone_groups = {tuple(k if isinstance(k, tuple) else (k,)): grp for k, grp in backbone_gdf.groupby(groups, dropna=False, sort=False)}
        center_iter = centerline_gdf.groupby(groups, dropna=False, sort=False)
    else:
        backbone_groups = {tuple(): backbone_gdf}
        center_iter = [(tuple(), centerline_gdf)]
    rows: list[dict[str, Any]] = []
    matched_groups = 0
    for key, center_grp in center_iter:
        key_t = tuple(key) if isinstance(key, tuple) else tuple([key]) if key is not None else tuple()
        backbone_grp = backbone_groups.get(key_t)
        if backbone_grp is None:
            if len(backbone_groups) == 1:
                backbone_grp = next(iter(backbone_groups.values()))
            else:
                continue
        anchor_station = pd.to_numeric(backbone_grp.get("station_m"), errors="coerce").to_numpy(dtype=float)
        anchor_vals = pd.to_numeric(backbone_grp.get("bed_backbone_z_m"), errors="coerce").to_numpy(dtype=float)
        finite_anchor = np.isfinite(anchor_station) & np.isfinite(anchor_vals)
        if np.count_nonzero(finite_anchor) == 0:
            continue
        anchor_station = anchor_station[finite_anchor]
        anchor_vals = anchor_vals[finite_anchor]
        order = np.argsort(anchor_station, kind="mergesort")
        anchor_station = anchor_station[order]
        anchor_vals = anchor_vals[order]
        center_station = pd.to_numeric(center_grp.get("station_m"), errors="coerce").to_numpy(dtype=float)
        finite_center = np.isfinite(center_station)
        if np.count_nonzero(finite_center) == 0:
            continue
        mapped = np.interp(center_station[finite_center], anchor_station, anchor_vals, left=float(anchor_vals[0]), right=float(anchor_vals[-1]))
        valid_idx = np.where(finite_center)[0]
        matched_groups += 1
        for out_i, center_i in enumerate(valid_idx):
            row = center_grp.iloc[int(center_i)]
            new_row = {c: row[c] for c in center_grp.columns if c != "geometry"}
            new_row["bed_backbone_z_m"] = float(mapped[out_i])
            new_row["geometry"] = row.geometry
            rows.append(new_row)
    if not rows:
        return backbone_gdf.iloc[0:0].copy(), {"transfer_row_count": 0, "transfer_group_count": 0}
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=centerline_gdf.crs)
    out["point_id"] = out["point_id"].astype(str)
    out["station_m"] = pd.to_numeric(out["station_m"], errors="coerce")
    out["bed_backbone_z_m"] = pd.to_numeric(out["bed_backbone_z_m"], errors="coerce")
    out = out.sort_values([c for c in (*groups, "station_m", "point_id") if c in out.columns], kind="mergesort").reset_index(drop=True)
    return out, {"transfer_row_count": int(len(out)), "transfer_group_count": int(matched_groups)}


def fallback_export_primary_support_points(*, backbone_points_path: Path, export_centerline_points_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    backbone = load_backbone_points(Path(backbone_points_path))
    export_centerline = load_centerline_points(Path(export_centerline_points_path))
    diagnostics: dict[str, Any] = {
        "geometry_source": "export_centerline_station_transfer",
        "backbone_input_crs": str(backbone.crs) if backbone.crs is not None else None,
        "export_centerline_input_crs": str(export_centerline.crs) if export_centerline.crs is not None else None,
        "backbone_input_count": int(len(backbone)),
        "export_centerline_input_count": int(len(export_centerline)),
    }
    merged = export_centerline.merge(
        pd.DataFrame(backbone.drop(columns="geometry")),
        on=["point_id", "station_m"],
        how="inner",
    )
    if len(merged) > 0:
        merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=export_centerline.crs)
        merged_gdf["bed_backbone_z_m"] = pd.to_numeric(merged_gdf["bed_backbone_z_m"], errors="coerce")
        diagnostics["centerline_backbone_join_count"] = int(len(merged_gdf))
        diagnostics["geometry_source"] = "export_centerline_exact_join_with_backbone_values"
        return merged_gdf, diagnostics
    transferred, transfer_diag = _station_transfer_backbone_to_centerline(backbone, export_centerline)
    diagnostics.update(transfer_diag)
    return transferred, diagnostics

def align_backbone_to_target_grid(backbone_gdf: gpd.GeoDataFrame, *, target_crs) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    work = backbone_gdf.copy()
    diagnostics = {
        "source_crs": str(backbone_gdf.crs) if backbone_gdf.crs is not None else None,
        "target_crs": str(target_crs) if target_crs is not None else None,
        "reprojected": False,
    }
    if target_crs is None or work.crs is None:
        return work, diagnostics
    source_crs_str = str(work.crs)
    target_crs_str = str(target_crs)
    if source_crs_str == target_crs_str:
        return work, diagnostics
    transformer = Transformer.from_crs(source_crs_str, target_crs_str, always_xy=True)
    work = work.copy()
    work["geometry"] = work.geometry.apply(lambda geom: None if geom is None or geom.is_empty else shapely_transform(transformer.transform, geom))
    work = gpd.GeoDataFrame(work, geometry="geometry", crs=target_crs_str)
    diagnostics["reprojected"] = True
    return work, diagnostics


def count_points_on_grid(backbone_gdf: gpd.GeoDataFrame, *, out_shape, transform) -> int:
    left = float(transform.c)
    top = float(transform.f)
    px_w = float(transform.a)
    px_h = float(transform.e)
    right = left + px_w * int(out_shape[1])
    bottom = top + px_h * int(out_shape[0])
    minx, maxx = (left, right) if left <= right else (right, left)
    miny, maxy = (bottom, top) if bottom <= top else (top, bottom)
    count = 0
    for geom in backbone_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        x = float(geom.x)
        y = float(geom.y)
        if minx <= x <= maxx and miny <= y <= maxy:
            count += 1
    return int(count)


def rasterize_backbone_to_grid(backbone_gdf: gpd.GeoDataFrame, out_shape, transform) -> np.ndarray:
    values = []
    shapes = []
    for _, row in backbone_gdf.iterrows():
        geom = row.geometry
        z = row.get("bed_backbone_z_m")
        if geom is None or geom.is_empty or not np.isfinite(z):
            continue
        shapes.append(geom)
        values.append(float(z))
    if not shapes:
        raise RuntimeError("river_v2_primary_surface_no_finite_backbone_values")
    out = rasterize(
        list(zip(shapes, values)),
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        all_touched=True,
        dtype="float32",
    )
    return out.astype("float32")


def interpolate_longitudinal_backbone_surface(backbone_raster: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    finite_any = np.isfinite(backbone_raster)
    finite = finite_any & valid_mask
    if np.count_nonzero(finite_any) == 0:
        raise RuntimeError("river_v2_primary_surface_no_seed_cells_on_grid")
    # Allow seeds just outside the valid river-domain mask to contribute to the
    # nearest-fill surface when they are still on the same target grid. This
    # avoids brittle failures from one-pixel mask/support mismatches while the
    # final surface is still clipped back to the valid domain below.
    seed_support = finite if np.count_nonzero(finite) > 0 else finite_any
    indices = distance_transform_edt(~seed_support, return_distances=False, return_indices=True)
    filled = backbone_raster[tuple(indices)]
    filled[~valid_mask] = np.nan
    return filled.astype("float32")


def _distance_to_bank(valid_mask: np.ndarray, transform) -> np.ndarray:
    sx = abs(float(getattr(transform, "a", 1.0) or 1.0))
    sy = abs(float(getattr(transform, "e", 1.0) or 1.0))
    dist = distance_transform_edt(valid_mask, sampling=(sy, sx)).astype("float32")
    # Make the first in-channel pixel at the NHD edge behave like distance 0 so
    # the taper can actually reach the bank, rather than stopping one pixel short.
    edge_step = float(min(sx, sy))
    if np.isfinite(edge_step) and edge_step > 0.0:
        dist = np.maximum(dist - edge_step, 0.0)
    dist[~valid_mask] = np.nan
    return dist.astype("float32")


def _load_aligned_reference(reference_path: Path | None, out_shape, transform, crs) -> np.ndarray | None:
    if reference_path is None or not Path(reference_path).exists():
        return None
    import rasterio.warp
    with rasterio.open(reference_path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            arr[np.isclose(arr, np.float32(nod))] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        if src.shape == tuple(out_shape) and tuple(src.transform) == tuple(transform) and str(src.crs) == str(crs):
            return arr
        dst = np.full(out_shape, np.nan, dtype="float32")
        rasterio.warp.reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.warp.Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return dst




def _resolve_bank_taper_reference_path(ctx: RiverV2Context) -> Path | None:
    candidates = [
        getattr(ctx, "baseline_interpolated_path", None),
        getattr(ctx, "aligned_authoritative_base_path", None),
        getattr(ctx, "river_dem_path", None),
    ]
    for cand in candidates:
        if cand is None:
            continue
        path = Path(cand)
        if path.exists():
            return path
    fallback = ctx.out_dir.parent / "final" / "authoritative_base_aligned.tif"
    return fallback if fallback.exists() else None

def _apply_bank_taper(centerline_surface: np.ndarray, valid_mask: np.ndarray, distance_to_centerline_m: np.ndarray, distance_to_bank_m: np.ndarray, reference_surface: np.ndarray | None) -> tuple[np.ndarray, int]:
    if reference_surface is None:
        return centerline_surface.astype("float32"), 0
    out = centerline_surface.astype("float32").copy()
    d_center = np.asarray(distance_to_centerline_m, dtype="float32")
    d_bank = np.asarray(distance_to_bank_m, dtype="float32")
    denom = d_center + d_bank
    lateral_fraction = np.zeros_like(out, dtype="float32")
    finite = np.isfinite(d_center) & np.isfinite(d_bank) & (denom > 0.0)
    lateral_fraction[finite] = d_center[finite] / denom[finite]
    # Use a smooth full-width taper so the surface transitions gradually from
    # the centerline toward the bank reference across the entire channel width,
    # and reaches the baseline exactly at the NHD river edge.
    edge_weight = np.clip(lateral_fraction, 0.0, 1.0) ** 1.25
    blend_mask = valid_mask & np.isfinite(out) & np.isfinite(reference_surface) & (edge_weight > 0.0)
    if not np.any(blend_mask):
        return out, 0
    out[blend_mask] = ((1.0 - edge_weight[blend_mask]) * out[blend_mask]) + (edge_weight[blend_mask] * reference_surface[blend_mask])
    return out, int(np.count_nonzero(blend_mask))


def apply_valid_domain(centerline_surface: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = centerline_surface.astype("float32").copy()
    out[~valid_mask] = np.nan
    return out


def build_river_primary_surface(
    ctx: RiverV2Context,
    *,
    backbone_points_path: Path,
    channel_mask_path: Path,
    centerline_points_path: Path | None = None,
    output_path: Path | None = None,
    receipt_path: Path | None = None,
    surface_domain_role: str = "solve_domain",
) -> RiverV2StageResult:
    backbone_points_path = ensure_existing_path(backbone_points_path, "river_v2_primary_surface_backbone")
    channel_mask_path = ensure_existing_path(channel_mask_path, "river_v2_primary_surface_channel_mask")
    backbone, support_point_diag = prepare_primary_support_points(
        backbone_points_path=backbone_points_path,
        centerline_points_path=centerline_points_path,
    )
    grid_spec = resolve_primary_surface_target_grid(ctx, channel_mask_path)
    with rasterio.open(channel_mask_path) as domain_ds:
        valid_mask = domain_ds.read(1)
        profile = domain_ds.profile.copy()
        out_shape = (domain_ds.height, domain_ds.width)
        transform = domain_ds.transform
    valid_mask = np.isfinite(valid_mask) & (valid_mask > 0)
    backbone, alignment_diag = align_backbone_to_target_grid(backbone, target_crs=grid_spec.crs)
    on_grid_point_count = count_points_on_grid(backbone, out_shape=out_shape, transform=transform)
    export_fallback_diag: dict[str, Any] | None = None
    if on_grid_point_count == 0 and ctx.paths.river_centerline_points_export.exists():
        backbone_export, export_support_diag = fallback_export_primary_support_points(
            backbone_points_path=backbone_points_path,
            export_centerline_points_path=ctx.paths.river_centerline_points_export,
        )
        backbone_export, export_alignment_diag = align_backbone_to_target_grid(backbone_export, target_crs=grid_spec.crs)
        export_on_grid_point_count = count_points_on_grid(backbone_export, out_shape=out_shape, transform=transform)
        export_fallback_diag = {
            **{f"export_{k}": v for k, v in export_support_diag.items()},
            **{f"export_{k}": v for k, v in export_alignment_diag.items()},
            "export_on_grid_point_count": int(export_on_grid_point_count),
        }
        if export_on_grid_point_count > 0:
            backbone = backbone_export
            support_point_diag = {**support_point_diag, "fallback_used": "export_centerline_station_transfer"}
            alignment_diag = export_alignment_diag
            on_grid_point_count = export_on_grid_point_count
    if on_grid_point_count == 0:
        diag = {**support_point_diag, **alignment_diag, **(export_fallback_diag or {})}
        raise RuntimeError(f"river_v2_primary_support_no_backbone_points_on_target_grid:{diag!r}")
    # Use the already-rasterized finite backbone cells as the centerline seed support.
    # This keeps the stage linear and avoids a second, mismatched rasterization of the
    # centerline points that can disagree with the actual backbone cells feeding the surface.
    backbone_raster = rasterize_backbone_to_grid(backbone, out_shape, transform)
    seed_presence = np.isfinite(backbone_raster)
    distance_to_centerline = compute_distance_to_seed_cells(valid_mask, seed_presence, transform)
    distance_to_bank = _distance_to_bank(valid_mask, transform)
    max_distance = None
    primary_surface_domain = valid_mask.copy()
    domain_path = write_primary_surface_domain_raster(primary_surface_domain=primary_surface_domain, grid_spec=grid_spec, domain_path=ctx.paths.river_primary_surface_domain)
    domain_validation = validate_primary_surface_domain(domain_path, expected_shape=out_shape, expected_transform=transform, expected_crs=grid_spec.crs)
    if not domain_validation.get("valid"):
        raise RuntimeError(f"river_v2_primary_surface_domain_invalid:{domain_validation}")
    centerline_surface = interpolate_longitudinal_backbone_surface(backbone_raster, primary_surface_domain)
    bank_reference_path = _resolve_bank_taper_reference_path(ctx)
    bank_reference = _load_aligned_reference(bank_reference_path, out_shape, transform, grid_spec.crs)
    tapered_surface, tapered_cell_count = _apply_bank_taper(
        centerline_surface,
        primary_surface_domain,
        distance_to_centerline,
        distance_to_bank,
        bank_reference,
    )
    primary_surface = apply_valid_domain(tapered_surface, primary_surface_domain)
    out_path = Path(output_path) if output_path is not None else ctx.paths.river_primary_surface_solve_domain
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("BLOCKXSIZE", None)
    profile.pop("BLOCKYSIZE", None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=False)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(primary_surface.astype("float32"), 1)
    validation = validate_primary_surface_raster(out_path, domain_path, domain_path)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_primary_surface_invalid:{validation}")
    receipt = build_river_v2_raster_receipt(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE,
        output_artifacts=[str(out_path)],
        input_artifacts=ctx.direct_stage_input_artifacts(
        backbone_points_path,
        channel_mask_path,
        centerline_points_path if centerline_points_path is not None else None,
    ),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="primary_surface_from_dense_backbone_point_rasterization_then_nearest_fill_within_full_nhd_channel_domain_followed_by_full_width_lateral_taper_to_nhd_edge",
        validation=validation,
        grid_shape=out_shape,
        stats={
            "backbone_geometry_source": support_point_diag.get("geometry_source"),
            "backbone_input_crs": support_point_diag.get("backbone_input_crs"),
            "centerline_input_crs": support_point_diag.get("centerline_input_crs"),
            "centerline_backbone_join_count": support_point_diag.get("centerline_backbone_join_count"),
            "backbone_source_crs": alignment_diag.get("source_crs"),
            "target_grid_crs": alignment_diag.get("target_crs"),
            "backbone_reprojected_to_target_grid": bool(alignment_diag.get("reprojected")),
            "backbone_point_count_on_target_grid": int(on_grid_point_count),
            "seed_cell_count": int(np.count_nonzero(np.isfinite(backbone_raster))),
            "seed_cell_count_in_domain": int(np.count_nonzero(np.isfinite(backbone_raster) & primary_surface_domain)),
            "domain_cell_count": int(np.count_nonzero(primary_surface_domain)),
            "valid_cell_count": int(np.count_nonzero(np.isfinite(primary_surface) & primary_surface_domain)),
            "max_distance_to_centerline_m": float(np.nanmax(distance_to_centerline[primary_surface_domain])) if np.count_nonzero(primary_surface_domain) else None,
            "max_distance_to_bank_m": float(np.nanmax(distance_to_bank[primary_surface_domain])) if np.count_nonzero(primary_surface_domain) else None,
            "bank_taper_reference_path": str(bank_reference_path) if bank_reference_path is not None else None,
            "bank_taper_blended_cell_count": int(tapered_cell_count),
            "domain_distance_limit_m": max_distance,
            "surface_domain_role": str(surface_domain_role),
            "channel_mask_path": str(channel_mask_path),
            "output_path": str(out_path),
        },
    )
    receipt_path = write_river_v2_receipt(receipt, Path(receipt_path) if receipt_path is not None else ctx.paths.river_primary_surface_solve_domain_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(np.count_nonzero(np.isfinite(primary_surface) & primary_surface_domain)),
        validation=validation,
        warnings=[],
        aux_outputs={"primary_surface_domain": str(domain_path)},
    )
