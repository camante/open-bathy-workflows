from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import math
import numpy as np


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
    try:
        if gdf.crs is not None and str(gdf.crs) == str(target_crs):
            return gdf
        return gdf.to_crs(target_crs)
    except Exception:
        return gdf


def _geometry_union(geoms):
    if geoms is None:
        return None
    try:
        return geoms.union_all()
    except Exception:
        try:
            return geoms.unary_union
        except Exception:
            return None


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
    try:
        polygons = gpd.read_file(gpkg, layer=nhdarea_layer)
        polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
        polygons = _safe_to_crs(polygons, target_crs)
    except Exception:
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
        try:
            merged = _geometry_union(flows_attr.geometry.buffer(max(5.0, 1.5), cap_style=2))
            polygons = polygons[polygons.geometry.intersects(merged)].copy()
        except Exception:
            pass
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
        return points_gdf
    return points_gdf.loc[keep].copy()


def _sample_raster_point(ds, x: float, y: float) -> float:
    try:
        v = next(ds.sample([(float(x), float(y))]))
    except Exception:
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
            points_gdf.loc[sel, confidence_field] = 0.45
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
    from scipy.spatial import cKDTree
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
                boundaries = [geom.exterior]
            elif geom.geom_type == "MultiPolygon":
                boundaries = [g.exterior for g in geom.geoms if g is not None and not g.is_empty]
        except Exception:
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

    xs_tree = None
    xs_vals = None
    if xs_bank_points is not None and not getattr(xs_bank_points, "empty", True):
        val_col = None
        for cand in ("bank_z_m", "z_m", "elevation_m"):
            if cand in xs_bank_points.columns:
                val_col = cand
                break
        if val_col is not None:
            xsv = np.asarray(xs_bank_points[val_col], dtype=float)
            valid = np.isfinite(xsv)
            if np.any(valid):
                xs_tree = cKDTree(np.column_stack([xs_bank_points.geometry.x.to_numpy(dtype=float)[valid], xs_bank_points.geometry.y.to_numpy(dtype=float)[valid]]))
                xs_vals = xsv[valid]

    vals = []
    statuses = []
    dists = []
    confs = []
    with rasterio.open(raster_path) as ds:
        for _, rec in out.iterrows():
            x = float(rec.geometry.x)
            y = float(rec.geometry.y)
            val = _sample_raster_point(ds, x, y)
            status = "exact"
            dist = 0.0
            conf = 1.0
            if not np.isfinite(val):
                tangent = _estimate_line_tangent(rec["_line_ref"], rec["_line_dist"], eps=max(float(spacing_m) * 0.25, 1.0))
                val, dist, status = _normal_search_value(ds, x, y, tangent, max_offset_m=normal_search_max_m, step_m=normal_search_step_m)
                conf = 0.8 if np.isfinite(val) else 0.0
            if (not np.isfinite(val)) and xs_tree is not None and xs_vals is not None:
                qdist, qidx = xs_tree.query([[x, y]], k=1)
                qd = float(np.ravel(qdist)[0])
                if np.isfinite(qd) and qd <= max(float(normal_search_max_m) * 4.0, float(spacing_m) * 2.0):
                    val = float(xs_vals[int(np.ravel(qidx)[0])])
                    status = "xs_fallback"
                    dist = qd
                    conf = 0.55
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


def build_centerline_points(flows_gdf, polygons_gdf=None, *, spacing_m: float = 60.0, raster_path: str | Path | None = None, role: str = "longitudinal_control", allowed_mask: Optional[np.ndarray] = None, transform=None):
    import geopandas as gpd
    from shapely.geometry import Point

    if flows_gdf is None or getattr(flows_gdf, "empty", True):
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(flows_gdf, "crs", None))
    poly_union = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            poly_union = _geometry_union(polygons_gdf.geometry)
        except Exception:
            poly_union = None
    rows = []
    for idx, rec in flows_gdf.iterrows():
        line = rec.geometry
        for pt in _sample_points_along_line(line, spacing_m):
            p = Point(pt.x, pt.y)
            if poly_union is not None:
                try:
                    if not poly_union.buffer(1.0).contains(p):
                        continue
                except Exception:
                    pass
            row = {"artifact_role": role, "flow_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "geometry": p}
            for col in ("component_id", "streamorde", "streamorder", "levelpathi", "levelpathid"):
                if col in rec.index:
                    row[col] = rec[col]
            rows.append(row)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=flows_gdf.crs)
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    if raster_path is not None and not out.empty:
        out = _sample_raster_values(raster_path, out, field="centerline_z_m")
    return out


def build_xs_support_points(xs_gpkg: str | Path, polygons_gdf=None, *, spacing_fraction: float = 0.25, spacing_m: Optional[float] = None, raster_path: str | Path | None = None, role: str = "cross_stream_control", allowed_mask: Optional[np.ndarray] = None, transform=None):
    import geopandas as gpd
    from shapely.geometry import Point

    try:
        xs = gpd.read_file(xs_gpkg)
    except Exception:
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(polygons_gdf, "crs", None))
    if xs.empty:
        return xs
    poly_union = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            polygons_gdf = _safe_to_crs(polygons_gdf, xs.crs)
            poly_union = _geometry_union(polygons_gdf.geometry)
        except Exception:
            poly_union = None
    rows = []
    frac = float(np.clip(spacing_fraction, 0.1, 0.45))
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
                p = Point(pt.x, pt.y)
                if poly_union is not None:
                    try:
                        if not poly_union.buffer(1.0).contains(p):
                            continue
                    except Exception:
                        pass
                rows.append({"artifact_role": role, "xs_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "xs_node": int(i), "geometry": p})
        except Exception:
            continue
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=xs.crs)
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    if raster_path is not None and not out.empty:
        out = _sample_raster_values(raster_path, out, field="xs_z_m")
    return out


def _nearest_surface_from_points(*, shape, transform, domain_mask: np.ndarray, points_gdf, value_field: str, max_distance_m: Optional[float] = None):
    from scipy.spatial import cKDTree
    import rasterio.transform

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
    tree = cKDTree(pts)
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
