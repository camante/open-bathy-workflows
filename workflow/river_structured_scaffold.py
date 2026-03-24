from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import logging
import math
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RiverScaffoldSelection:
    flows: object
    polygons: object
    metadata: dict


def _find_col(gdf, names: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in gdf.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _safe_to_crs(gdf, target_crs):
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    if target_crs is None:
        return gdf
    if gdf.crs is None:
        raise RuntimeError("Cannot reproject GeoDataFrame with undefined CRS")
    if str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _geometry_union(geoms):
    if geoms is None:
        return None
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    if hasattr(geoms, "unary_union"):
        return geoms.unary_union
    from shapely.ops import unary_union
    return unary_union(list(geoms))


def select_retained_river_features(
    river_gpkg: str | Path,
    *,
    target_crs=None,
    min_stream_order: int = 3,
    min_length_km: float = 0.25,
    keep_top_components: int = 6,
    nhdarea_layer: str = "nhdarea_clip",
    flows_layer: str = "rivers_clip",
):
    import geopandas as gpd

    gpkg = Path(river_gpkg)
    flows = gpd.read_file(gpkg, layer=flows_layer)
    flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
    flows = _safe_to_crs(flows, target_crs)

    polygons = None
    layer_names = set(gpd.list_layers(gpkg)["name"].tolist())
    if nhdarea_layer in layer_names:
        polygons = gpd.read_file(gpkg, layer=nhdarea_layer)
        polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
        polygons = _safe_to_crs(polygons, target_crs)
    else:
        polygons = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(flows, "crs", None))

    flow_count0 = int(len(flows))
    col_order = _find_col(flows, ["streamorde", "streamorder", "streamord", "streamleve"])
    col_len = _find_col(flows, ["lengthkm", "length_km", "len_km"])
    col_comp = _find_col(flows, ["component_id"])

    keep = np.ones(len(flows), dtype=bool)
    if col_len:
        keep &= np.nan_to_num(np.asarray(flows[col_len], dtype=float), nan=0.0) >= float(min_length_km)
    if col_order:
        keep &= np.nan_to_num(np.asarray(flows[col_order], dtype=float), nan=-1.0) >= float(min_stream_order)
    flows_attr = flows.loc[keep].copy()

    component_lengths = {}
    if col_comp and not flows.empty:
        if col_len:
            tmp = flows[[col_comp, col_len]].copy()
            tmp[col_len] = np.nan_to_num(np.asarray(tmp[col_len], dtype=float), nan=0.0)
            component_lengths = tmp.groupby(col_comp)[col_len].sum().to_dict()
        else:
            component_lengths = flows.groupby(col_comp).size().to_dict()

    top_components = set()
    if component_lengths:
        top_components = set([k for k, _ in sorted(component_lengths.items(), key=lambda kv: kv[1], reverse=True)[: max(int(keep_top_components), 1)]])

    if flows_attr.empty and top_components:
        flows_attr = flows[flows[col_comp].isin(list(top_components))].copy() if col_comp else flows.copy()

    flows_attr = flows_attr[flows_attr.geometry.notnull() & ~flows_attr.geometry.is_empty].copy()

    # Restrict polygons to those intersecting retained flowlines when available.
    retained_poly_count = 0
    if polygons is not None and not polygons.empty and not flows_attr.empty:
        merged = _geometry_union([geom.buffer(max(5.0, 1.5), cap_style=2) for geom in flows_attr.geometry if geom is not None and not geom.is_empty])
        polygons = polygons[polygons.geometry.intersects(merged)].copy()
        retained_poly_count = int(len(polygons))

    meta = {
        "input_flow_count": flow_count0,
        "retained_flow_count": int(len(flows_attr)),
        "retained_polygon_count": retained_poly_count,
        "min_stream_order": int(min_stream_order),
        "min_length_km": float(min_length_km),
        "keep_top_components": int(keep_top_components),
        "top_components": [str(x) for x in list(top_components)[: max(int(keep_top_components), 1)]],
        "selection_mode": "mainstem_and_key_tributaries",
        "dropped_feature_families": ["lakes", "ponds", "minor_streams", "small_disconnected_waterbodies"],
    }
    return RiverScaffoldSelection(flows=flows_attr, polygons=polygons, metadata=meta)


def _sample_points_along_line(line, spacing_m: float):
    if line is None or line.is_empty:
        return []
    try:
        if line.geom_type == "MultiLineString":
            pts = []
            for geom in line.geoms:
                pts.extend(_sample_points_along_line(geom, spacing_m))
            return pts
        L = float(line.length)
        if L <= 0:
            return []
        step = max(float(spacing_m), 1.0)
        dists = list(np.arange(0.0, L, step, dtype=float))
        if not dists or (L - dists[-1]) > (0.35 * step):
            dists.append(L)
        return [line.interpolate(float(d)) for d in dists]
    except Exception:
        log.debug("_sample_points_along_line: suppressed exception", exc_info=True)
        return []


def _sample_raster_values(raster_path: str | Path, points_gdf, field: str = "z_m"):
    import rasterio

    out = np.full(len(points_gdf), np.nan, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True):
        points_gdf[field] = out
        return points_gdf
    with rasterio.open(raster_path) as ds:
        pts = list(zip(points_gdf.geometry.x.to_numpy(dtype=float), points_gdf.geometry.y.to_numpy(dtype=float)))
        vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            vals.append(val)
        out = np.asarray(vals, dtype=np.float32)
    points_gdf[field] = out
    return points_gdf


def _mask_points_to_allowed(points_gdf, allowed_mask: Optional[np.ndarray], transform):
    if points_gdf is None or getattr(points_gdf, "empty", True) or allowed_mask is None or transform is None:
        return points_gdf
    import rasterio.transform

    keep = np.zeros(len(points_gdf), dtype=bool)
    try:
        xs = points_gdf.geometry.x.to_numpy(dtype=float)
        ys = points_gdf.geometry.y.to_numpy(dtype=float)
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        ok = (rows >= 0) & (rows < allowed_mask.shape[0]) & (cols >= 0) & (cols < allowed_mask.shape[1])
        keep[ok] = np.asarray(allowed_mask[rows[ok], cols[ok]], dtype=bool)
    except Exception:
        log.debug("_mask_points_to_allowed: suppressed exception", exc_info=True)
        return points_gdf
    return points_gdf.loc[keep].copy()




def _clip_gdf_to_allowed_mask(gdf, allowed_mask: Optional[np.ndarray], transform):
    if gdf is None or getattr(gdf, "empty", True) or allowed_mask is None or transform is None:
        return gdf
    from rasterio.features import shapes
    from shapely.geometry import shape

    mask_bool = np.asarray(allowed_mask, dtype=bool)
    if not np.any(mask_bool):
        return gdf.iloc[0:0].copy()
    try:
        geoms = [shape(geom) for geom, value in shapes(mask_bool.astype("uint8"), transform=transform) if int(value) == 1]
    except Exception:
        log.debug("_clip_gdf_to_allowed_mask: suppressed exception", exc_info=True)
        geoms = []
    if not geoms:
        return gdf.iloc[0:0].copy()
    try:
        allowed_geom = _geometry_union(geoms)
    except Exception:
        log.debug("_clip_gdf_to_allowed_mask: suppressed exception", exc_info=True)
        allowed_geom = geoms[0]
        for geom in geoms[1:]:
            allowed_geom = allowed_geom.union(geom)
    clipped = gdf.copy()
    clipped["geometry"] = [geom.intersection(allowed_geom) if geom is not None else None for geom in clipped.geometry]
    clipped = clipped[clipped.geometry.notnull() & ~clipped.geometry.is_empty].copy()
    return clipped


def _sample_raster_point(ds, x: float, y: float) -> float:
    try:
        v = next(ds.sample([(float(x), float(y))]))
    except Exception:
        log.debug("_sample_raster_point: suppressed exception", exc_info=True)
        return float("nan")
    val = float(v[0]) if np.size(v) else np.nan
    if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
        return float("nan")
    return val


def _estimate_line_tangent(line, distance: float, eps: float = 1.0):
    try:
        L = float(line.length)
        d0 = max(0.0, min(L, float(distance) - float(eps)))
        d1 = max(0.0, min(L, float(distance) + float(eps)))
        p0 = line.interpolate(d0)
        p1 = line.interpolate(d1)
        dx = float(p1.x - p0.x)
        dy = float(p1.y - p0.y)
        n = math.hypot(dx, dy)
        if not np.isfinite(n) or n <= 0.0:
            return None
        return (dx / n, dy / n)
    except Exception:
        log.debug("_estimate_line_tangent: suppressed exception", exc_info=True)
        return None


def _normal_search_value(ds, x: float, y: float, tangent, max_offset_m: float = 8.0, step_m: float = 1.0):
    if tangent is None:
        return float("nan"), float("inf"), "missing"
    tx, ty = tangent
    nx, ny = -ty, tx
    best_val = float("nan")
    best_dist = float("inf")
    best_status = "missing"
    for sign, status in ((1.0, "normal_search_outward"), (-1.0, "normal_search_inward")):
        offset = max(float(step_m), 0.5)
        while offset <= float(max_offset_m) + 1e-6:
            xx = float(x + sign * nx * offset)
            yy = float(y + sign * ny * offset)
            val = _sample_raster_point(ds, xx, yy)
            if np.isfinite(val):
                if offset < best_dist:
                    best_val = val
                    best_dist = float(offset)
                    best_status = status
                break
            offset += max(float(step_m), 0.5)
    return best_val, best_dist, best_status


def _fill_along_bank(points_gdf, value_field: str, status_field: str, distance_field: str, confidence_field: str):
    if points_gdf is None or getattr(points_gdf, "empty", True) or value_field not in points_gdf.columns:
        return points_gdf
    points_gdf = points_gdf.copy()
    if "bank_group" not in points_gdf.columns:
        points_gdf["bank_group"] = 0
    if "bank_chain_m" not in points_gdf.columns:
        points_gdf["bank_chain_m"] = np.arange(len(points_gdf), dtype=float)
    if confidence_field in points_gdf.columns:
        points_gdf[confidence_field] = pd.to_numeric(points_gdf[confidence_field], errors="coerce").astype(np.float32)
    for _, idx in points_gdf.groupby("bank_group", sort=False).groups.items():
        ids = list(idx)
        vals = np.asarray(points_gdf.loc[ids, value_field], dtype=float)
        chain = np.asarray(points_gdf.loc[ids, "bank_chain_m"], dtype=float)
        valid = np.isfinite(vals)
        if np.count_nonzero(valid) < 2:
            continue
        interp = np.interp(chain, chain[valid], vals[valid])
        missing = ~valid
        if np.any(missing):
            sel = np.asarray(ids)[missing]
            points_gdf.loc[sel, value_field] = interp[missing].astype(np.float32)
            points_gdf.loc[sel, status_field] = "along_bank_interp"
            points_gdf.loc[sel, distance_field] = np.nan
            points_gdf.loc[sel, confidence_field] = np.float32(0.45)
    return points_gdf


def build_dense_bank_points(
    polygons_gdf,
    *,
    spacing_m: float = 40.0,
    raster_path: str | Path | None = None,
    role: str = "bank_boundary_control",
    normal_search_max_m: float = 8.0,
    normal_search_step_m: float = 1.0,
    xs_bank_points=None,
    allowed_mask: Optional[np.ndarray] = None,
    transform=None,
):
    import geopandas as gpd
    from shapely.geometry import Point
    import rasterio

    if polygons_gdf is None or getattr(polygons_gdf, "empty", True):
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(polygons_gdf, "crs", None))

    rows = []
    for idx, geom in enumerate(polygons_gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        boundaries = []
        try:
            if geom.geom_type == "Polygon":
                boundaries = [geom.exterior] + list(geom.interiors)
            elif geom.geom_type == "MultiPolygon":
                for g in geom.geoms:
                    if g is None or g.is_empty:
                        continue
                    boundaries.append(g.exterior)
                    boundaries.extend(list(g.interiors))
        except Exception:
            log.debug("build_dense_bank_points: suppressed exception", exc_info=True)
            boundaries = []
        for b in boundaries:
            pts = _sample_points_along_line(b, spacing_m)
            for j, pt in enumerate(pts):
                chain = float(j) * float(max(spacing_m, 1.0))
                rows.append(
                    {
                        "artifact_role": role,
                        "bank_group": int(idx),
                        "bank_chain_m": chain,
                        "bank_sample_status": "missing",
                        "bank_sample_distance_m": np.nan,
                        "bank_confidence": 0.0,
                        "geometry": Point(pt.x, pt.y),
                        "_line_ref": b,
                        "_line_dist": chain,
                    }
                )
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=polygons_gdf.crs)
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    if raster_path is None or out.empty:
        return out.drop(columns=[c for c in ["_line_ref", "_line_dist"] if c in out.columns])


    vals = []
    statuses = []
    dists = []
    confs = []
    with rasterio.open(raster_path) as ds:
        xs = out.geometry.x.to_numpy(dtype=float)
        ys = out.geometry.y.to_numpy(dtype=float)
        pts = list(zip(xs, ys))
        exact_vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            exact_vals.append(val)
        exact_vals = np.asarray(exact_vals, dtype=float)
        for i in range(len(out)):
            rec = out.iloc[i]
            x = float(xs[i])
            y = float(ys[i])
            val = float(exact_vals[i]) if i < exact_vals.size else np.nan
            status = "exact"
            dist = 0.0
            conf = 1.0
            if not np.isfinite(val):
                tangent = _estimate_line_tangent(rec["_line_ref"], rec["_line_dist"], eps=max(float(spacing_m) * 0.25, 1.0))
                val, dist, status = _normal_search_value(ds, x, y, tangent, max_offset_m=normal_search_max_m, step_m=normal_search_step_m)
                conf = 0.8 if np.isfinite(val) else 0.0
            vals.append(val)
            statuses.append(status)
            dists.append(dist if np.isfinite(val) else np.nan)
            confs.append(conf if np.isfinite(val) else 0.0)

    out["bank_z_m"] = np.asarray(vals, dtype=np.float32)
    out["bank_sample_status"] = statuses
    out["bank_sample_distance_m"] = np.asarray(dists, dtype=np.float32)
    out["bank_confidence"] = np.asarray(confs, dtype=np.float32)
    out = _fill_along_bank(out, "bank_z_m", "bank_sample_status", "bank_sample_distance_m", "bank_confidence")
    return out.drop(columns=[c for c in ["_line_ref", "_line_dist"] if c in out.columns])


def build_centerline_points(flows_gdf, polygons_gdf=None, *, spacing_m: float = 60.0, raster_path: str | Path | None = None, role: str = "longitudinal_control", allowed_mask: Optional[np.ndarray] = None, transform=None, logger=None):
    import geopandas as gpd
    import time

    if flows_gdf is None or getattr(flows_gdf, "empty", True):
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(flows_gdf, "crs", None))
    t0 = time.perf_counter()
    flows = flows_gdf.copy()
    poly_union = None
    clip_buffer = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            polygons_gdf = _safe_to_crs(polygons_gdf, flows.crs)
            poly_union = _geometry_union(polygons_gdf.geometry)
            if poly_union is not None and not poly_union.is_empty:
                clip_buffer = poly_union.buffer(1.0)
                intersects = flows.geometry.intersects(clip_buffer)
                flows = flows.loc[np.asarray(intersects, dtype=bool)].copy()
                flows["geometry"] = [geom.intersection(clip_buffer) if geom is not None else None for geom in flows.geometry]
                flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
        except Exception:
            log.debug("build_centerline_points: suppressed exception", exc_info=True)
            poly_union = None
            clip_buffer = None
    t_union = time.perf_counter()
    rows = []
    n_sampled = 0
    keep_cols = [c for c in ("component_id", "streamorde", "streamorder", "levelpathi", "levelpathid", "s_m_from", "s_m_to", "s_m_min", "s_m_max") if c in flows.columns]
    for idx, rec in flows.iterrows():
        line = rec.geometry
        pts = _sample_points_along_line(line, spacing_m)
        if not pts:
            continue
        n_sampled += int(len(pts))
        base = {"artifact_role": role, "flow_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
        for col in keep_cols:
            base[col] = rec[col]
        line_len = float(getattr(line, "length", 0.0) or 0.0)
        s0 = pd.to_numeric(rec.get("s_m_from", rec.get("s_m_min", np.nan)), errors="coerce")
        s1 = pd.to_numeric(rec.get("s_m_to", rec.get("s_m_max", np.nan)), errors="coerce")
        use_station = np.isfinite(s0) and np.isfinite(s1) and line_len > 0.0
        for pt in pts:
            row = dict(base)
            row["geometry"] = pt
            proj = float(line.project(pt)) if line_len > 0.0 else 0.0
            if use_station:
                frac = float(proj / line_len) if line_len > 0.0 else 0.0
                row["station_m"] = float(s0 + ((s1 - s0) * frac))
            else:
                # Fallback to local along-line distance so downstream anisotropic river routing
                # still has a usable station coordinate even when absolute station fields are absent.
                row["station_m"] = proj
            rows.append(row)
    t_sample = time.perf_counter()
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=flows_gdf.crs)
    n_before_allowed = int(len(out))
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    t_mask = time.perf_counter()
    if raster_path is not None and not out.empty:
        out = _sample_raster_values(raster_path, out, field="centerline_z_m")
    t_raster = time.perf_counter()
    if logger is not None:
        logger.info(
            "[RIVER][GUIDANCE][DETAIL] Centerline point build: flows=%d flows_clipped=%d sampled=%d kept_pre_allowed=%d kept_final=%d union=%.2fs sample=%.2fs mask=%.2fs raster=%.2fs total=%.2fs",
            0 if flows_gdf is None else int(len(flows_gdf)),
            0 if flows is None else int(len(flows)),
            int(n_sampled),
            int(n_before_allowed),
            int(len(out)),
            t_union - t0,
            t_sample - t_union,
            t_mask - t_sample,
            t_raster - t_mask,
            t_raster - t0,
        )
    return out


def build_xs_support_points(xs_gpkg: str | Path, polygons_gdf=None, *, spacing_fraction: float = 0.25, spacing_m: Optional[float] = None, raster_path: str | Path | None = None, role: str = "cross_stream_control", allowed_mask: Optional[np.ndarray] = None, transform=None, logger=None):
    import geopandas as gpd
    import time

    gpkg = Path(xs_gpkg)
    try:
        layers = gpd.list_layers(gpkg)
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect XS GeoPackage layers for {gpkg}: {exc}") from exc
    layer_names = [str(v) for v in getattr(layers, "name", layers)]
    if "xs_lines" not in layer_names:
        raise RuntimeError(f"XS GeoPackage must contain layer 'xs_lines': {gpkg} | available_layers={layer_names}")
    try:
        xs = gpd.read_file(gpkg, layer="xs_lines")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not read xs_lines layer from {gpkg}: {exc}") from exc
    if xs.empty:
        raise RuntimeError(f"XS GeoPackage layer 'xs_lines' is empty: {gpkg}")
    xs_input_count = int(len(xs))
    poly_union = None
    clip_buffer = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            polygons_gdf = _safe_to_crs(polygons_gdf, xs.crs)
            poly_union = _geometry_union(polygons_gdf.geometry)
            if poly_union is not None and not poly_union.is_empty:
                clip_buffer = poly_union.buffer(1.0)
                intersects = xs.geometry.intersects(clip_buffer)
                xs = xs.loc[np.asarray(intersects, dtype=bool)].copy()
                xs["geometry"] = [geom.intersection(clip_buffer) if geom is not None else None for geom in xs.geometry]
                xs = xs[xs.geometry.notnull() & ~xs.geometry.is_empty].copy()
        except Exception:
            log.debug("build_xs_support_points: suppressed exception", exc_info=True)
            poly_union = None
            clip_buffer = None
    t0 = time.perf_counter()
    rows = []
    frac = float(np.clip(spacing_fraction, 0.1, 0.45))
    n_sampled = 0
    for idx, rec in xs.iterrows():
        line = rec.geometry
        if line is None or line.is_empty:
            continue
        try:
            L = float(line.length)
            if spacing_m is not None and float(spacing_m) > 0.0:
                distances = list(np.arange(max(float(spacing_m), 1.0), max(L - float(spacing_m), 0.0) + 1e-6, max(float(spacing_m), 1.0), dtype=float))
                if len(distances) < 3:
                    distances = [L * frac, L * 0.5, L * (1.0 - frac)]
            else:
                distances = [L * frac, L * 0.5, L * (1.0 - frac)]
            for i, d in enumerate(distances):
                pt = line.interpolate(float(d))
                n_sampled += 1
                rows.append({"artifact_role": role, "xs_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "xs_node": int(i), "geometry": pt})
        except Exception:
            log.debug("build_xs_support_points: suppressed exception", exc_info=True)
            continue
    t_sample = time.perf_counter()
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=xs.crs)
    n_before_allowed = int(len(out))
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    t_mask = time.perf_counter()
    if raster_path is not None and not out.empty:
        out = _sample_raster_values(raster_path, out, field="xs_z_m")
    t_raster = time.perf_counter()
    if logger is not None:
        logger.info(
            "[RIVER][GUIDANCE][DETAIL] XS support point build: xs=%d xs_clipped=%d sampled=%d kept_pre_allowed=%d kept_final=%d sample=%.2fs mask=%.2fs raster=%.2fs total=%.2fs",
            int(xs_input_count),
            int(len(xs)),
            int(n_sampled),
            int(n_before_allowed),
            int(len(out)),
            t_sample - t0,
            t_mask - t_sample,
            t_raster - t_mask,
            t_raster - t0,
        )
    return out


def _nearest_surface_from_points(*, shape, transform, domain_mask: np.ndarray, points_gdf, value_field: str, max_distance_m: Optional[float] = None):
    import rasterio.transform
    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy.spatial.cKDTree is required for river structured scaffold nearest-surface generation") from exc

    out = np.full(shape, np.nan, dtype=np.float32)
    influence = np.zeros(shape, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return out, influence
    vals = np.asarray(points_gdf.get(value_field, np.full(len(points_gdf), np.nan)), dtype=np.float32)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return out, influence
    pts = np.column_stack([points_gdf.geometry.x.to_numpy(dtype=float)[valid], points_gdf.geometry.y.to_numpy(dtype=float)[valid]])
    vals = vals[valid]
    tree = _cKDTree(pts)
    rows, cols = np.where(np.asarray(domain_mask, dtype=bool))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    dist, idx = tree.query(q, k=1)
    sampled = vals[np.asarray(idx, dtype=int)]
    out[rows, cols] = sampled.astype(np.float32)
    if max_distance_m is None:
        max_distance_m = np.nanpercentile(np.asarray(dist, dtype=float), 80) if len(dist) else 0.0
    cutoff = max(float(max_distance_m), 1.0)
    infl = np.clip(1.0 - (np.asarray(dist, dtype=np.float32) / cutoff), 0.0, 1.0)
    out_mask = np.asarray(dist, dtype=float) > cutoff
    if np.any(out_mask):
        sampled = sampled.copy()
        sampled[out_mask] = np.nan
        infl[out_mask] = 0.0
        out[rows, cols] = sampled.astype(np.float32)
    influence[rows, cols] = infl.astype(np.float32)
    return out, influence
