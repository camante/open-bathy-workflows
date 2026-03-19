#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xs_builder.py – Build river cross-sections (XS) from a river network and DEM/topo rasters.

Keeps the largest connected components, applies rolling-window orientation smoothing,
and trims intersecting XS lines to prevent zipper artifacts.

Inputs:  river_network.gpkg, --dem (required), --topo-lidar (optional)
Outputs: GPKG with 'xs_lines' and 'xs_points' layers.
"""


import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set, Any

import numpy as np
import pandas as pd

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import linemerge, split
from shapely.strtree import STRtree
from pyproj import CRS, Transformer

# Use centralized logging
log = logging.getLogger("xs_builder")


@dataclass
class XSConfig:
    spacing_m: float = 200.0
    half_width_m: float = 150.0
    sample_step_m: float = 2.0
    bank_search_m: float = 40.0
    bank_edge_refine_m: float = 12.0
    bank_quantile: float = 0.85
    bank_smooth_window_m: float = 8.0
    min_centerline_len_m: float = 50.0
    max_xs_per_reach: int = 2000
    # New options for overlap handling
    smoothing_window_m: float = 0.0  # 0.0 means "auto" (use spacing)
    trim_overlaps: bool = True

    # Global intersection handling (within a reach)
    global_deconflict: bool = True     # drop XS that intersect non-adjacent XS
    deconflict_tol_m: float = 2.0      # endpoint tolerance for "touching" intersections


    # Conservative global intersection handling (across the entire AOI)
    # If enabled, drop any XS that intersects a higher-score kept XS, regardless of reach id.
    # This reduces interpolation artifacts at tight meanders / reach boundaries at the cost of fewer XS.
    global_deconflict_all: bool = True
    # Junction handling (avoid XS near confluences where geometry is ambiguous)
    skip_junctions: bool = True
    junction_snap_m: float = 30.0      # snapping scale for junction detection from reach endpoints (m)
    junction_buffer_m: float = 120.0   # skip XS within this distance of junction nodes (m)

    # Centerline preprocessing
    densify_step_m: float = 20.0       # densify centerlines to this vertex spacing (m) before tangents


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def _read_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.empty:
        raise RuntimeError(f"Layer '{layer}' is empty in {gpkg}")
    if gdf.crs is None:
        raise RuntimeError(f"Layer '{layer}' has no CRS: {gpkg}")
    return gdf


def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy()
    return gdf


def _ensure_single_linestring(geom) -> Optional[LineString]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        try:
            merged = linemerge(geom)
            if merged.geom_type == "LineString":
                return merged
            if merged.geom_type == "MultiLineString":
                parts = list(merged.geoms)
                parts = sorted(parts, key=lambda g: g.length, reverse=True)
                return parts[0] if parts else None
        except Exception:
            parts = list(geom.geoms)
            parts = sorted(parts, key=lambda g: g.length, reverse=True)
            return parts[0] if parts else None
    return None


def _find_col(gdf: gpd.GeoDataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


# --------------------------------------------------------------------------------------
# Component-first pruning (recommended for coastal/tidal networks)
# --------------------------------------------------------------------------------------

def _ensure_edges_length_m(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edges = edges.copy()
    if "length_m" not in edges.columns:
        edges["length_m"] = edges.geometry.length
    return edges


def compute_component_lengths(edges: gpd.GeoDataFrame) -> pd.Series:
    if "component_id" not in edges.columns:
        raise RuntimeError("graph_edges is missing 'component_id'. Re-run river_network.py topology step.")
    edges = _ensure_edges_length_m(edges)
    comp_len = edges.groupby("component_id")["length_m"].sum().sort_values(ascending=False)
    return comp_len


def _attach_component_id_to_rivers(rivers: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "component_id" in rivers.columns:
        return rivers

    if "river_id" in rivers.columns and "river_id" in edges.columns and "component_id" in edges.columns:
        mapping = edges[["river_id", "component_id"]].dropna().drop_duplicates()
        rivers2 = rivers.merge(mapping, on="river_id", how="left")
        if "component_id" in rivers2.columns:
            return rivers2

    return rivers


def filter_by_component_length(
    rivers: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    keep_top_components: int = 1,
) -> gpd.GeoDataFrame:
    rivers = rivers.copy()
    edges = edges.copy()

    if "component_id" not in edges.columns:
        log.warning("graph_edges has no component_id; skipping component pruning.")
        return rivers

    rivers = _attach_component_id_to_rivers(rivers, edges)
    if "component_id" not in rivers.columns:
        log.warning("rivers_clip has no component_id and could not be joined; skipping component pruning.")
        return rivers

    comp_len = compute_component_lengths(edges)
    if comp_len.empty:
        log.warning("No component lengths computed; skipping component pruning.")
        return rivers

    keep_top_components = max(1, int(keep_top_components))
    keep_ids: Set[int] = set(comp_len.head(keep_top_components).index.tolist())

    before = len(rivers)
    rivers = rivers[rivers["component_id"].isin(keep_ids)].copy()
    log.info(
        "[COMPONENT] Kept top %d component(s) by total length: %s | rivers %d → %d",
        keep_top_components,
        sorted(list(keep_ids)),
        before,
        len(rivers),
    )
    return rivers


# --------------------------------------------------------------------------------------
# Attribute filtering (secondary pass)
# --------------------------------------------------------------------------------------

def filter_centerlines(
    rivers: gpd.GeoDataFrame,
    min_stream_order: int = 3,
    min_length_km: float = 0.05,
    ftype_allow: Optional[List[int]] = None,
    include_artificial_path: bool = False,
) -> gpd.GeoDataFrame:
    gdf = rivers.copy()
    n0 = len(gdf)

    col_order = _find_col(gdf, ["streamorde", "streamorder", "streamord"])
    col_len = _find_col(gdf, ["lengthkm", "length_km", "len_km"])
    col_ftype = _find_col(gdf, ["ftype"])

    if ftype_allow is None or len(ftype_allow) == 0:
        ftype_allow = [460]

    if include_artificial_path and 558 not in ftype_allow:
        ftype_allow = list(ftype_allow) + [558]

    if col_ftype:
        before = len(gdf)
        gdf = gdf[gdf[col_ftype].astype("float64").isin([float(x) for x in ftype_allow])]
        log.info("ftype allow=%s (%s): %d → %d", ftype_allow, col_ftype, before, len(gdf))
    else:
        log.info("No ftype field found; skipping ftype filter.")

    if col_order:
        before = len(gdf)
        gdf = gdf[pd.to_numeric(gdf[col_order], errors="coerce").fillna(-1) >= int(min_stream_order)]
        log.info("min_stream_order=%d (%s): %d → %d", int(min_stream_order), col_order, before, len(gdf))
    else:
        log.info("No stream order field found; skipping stream order filter.")

    if col_len:
        before = len(gdf)
        gdf = gdf[pd.to_numeric(gdf[col_len], errors="coerce").fillna(0.0) >= float(min_length_km)]
        log.info("min_length_km=%.3f (%s): %d → %d", float(min_length_km), col_len, before, len(gdf))
    else:
        log.info("No lengthkm field found; skipping length filter.")

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    log.info("centerlines kept: %d / %d", len(gdf), n0)
    return gdf


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------

def _unit_perp(dx: float, dy: float) -> Tuple[float, float]:
    px, py = -dy, dx
    n = (px**2 + py**2) ** 0.5
    if n == 0:
        return 0.0, 0.0
    return px / n, py / n


def _line_tangent(line: LineString, s: float, eps: float) -> Tuple[float, float]:
    """Calculate unit tangent vector of line at distance s, using a window of +/- eps."""
    L = line.length
    # Ensure window is valid
    s0 = max(0.0, min(L, s - eps))
    s1 = max(0.0, min(L, s + eps))
    
    # If segment is too small (start/end of line), bias the window inward
    if abs(s1 - s0) < 1e-3:
        if s0 == 0.0:
            s1 = min(L, s0 + 1.0) # Look ahead
        elif s1 == L:
            s0 = max(0.0, s1 - 1.0) # Look back

    p0 = line.interpolate(s0)
    p1 = line.interpolate(s1)
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    n = (dx**2 + dy**2) ** 0.5
    if n == 0:
        return 0.0, 0.0
    return dx / n, dy / n


def build_xs_line(center_pt: Point, tan: Tuple[float, float], half_width_m: float) -> LineString:
    dx, dy = tan
    px, py = _unit_perp(dx, dy)
    x0 = center_pt.x - px * half_width_m
    y0 = center_pt.y - py * half_width_m
    x1 = center_pt.x + px * half_width_m
    y1 = center_pt.y + py * half_width_m
    return LineString([(x0, y0), (x1, y1)])


def _trim_line_at_intersection(line_geom: LineString, intersect_pt: Point) -> LineString:
    """Trim a XS line at its intersection point, keeping the half containing the line midpoint."""
    d_int = line_geom.project(intersect_pt)
    d_center = line_geom.length * 0.5
    if d_int < d_center:
        return LineString([intersect_pt, line_geom.coords[-1]])
    return LineString([line_geom.coords[0], intersect_pt])


def _trim_overlapping_xs(xs_list: List[Dict]) -> List[Dict]:
    """
    Check adjacent cross-sections for intersection. If they cross, clip them at the intersection point.
    xs_list must be sorted by s_center_m.
    """
    if len(xs_list) < 2:
        return xs_list

    modified = 0
    # Iterate through adjacent pairs
    # Note: We assume xs_list is sorted by station
    for i in range(len(xs_list) - 1):
        curr = xs_list[i]
        next_xs = xs_list[i+1]
        
        g1 = curr['geometry']
        g2 = next_xs['geometry']
        
        if not g1.intersects(g2):
            continue
            
        pt = g1.intersection(g2)
        
        # We only handle single point intersections (standard crossing)
        if pt.geom_type != 'Point':
            continue
            
        # Strategy: The intersection usually happens on the "inside" of the bend.
        # We want to keep the segment of the line that connects to the centerline.
        # Since we construct lines as [Left, Right] centered on the centerline,
        # Apply trimming using the module-level helper (no inner def, no unused param).
        try:
            new_g1 = _trim_line_at_intersection(g1, pt)
            new_g2 = _trim_line_at_intersection(g2, pt)

            if not new_g1.is_empty and new_g1.length > 1.0:
                curr["geometry"] = new_g1
            if not new_g2.is_empty and new_g2.length > 1.0:
                next_xs["geometry"] = new_g2

            modified += 1
        except Exception:
            log.debug("XS trim failed for pair %s/%s; skipping.", curr.get("xs_id"), next_xs.get("xs_id"), exc_info=True)

    if modified > 0:
        log.info("Trimmed %d intersecting cross-section pairs.", modified)
        
    return xs_list



def _densify_linestring(line: LineString, step_m: float) -> LineString:
    """Densify a LineString so that no segment is longer than step_m.

    Improves tangent estimation and XS orientation stability on coarse/segmented inputs.
    """
    if line is None or line.is_empty:
        return line
    if not step_m or step_m <= 0:
        return line
    L = float(line.length)
    if L <= step_m:
        return line
    # Always include endpoints
    dists = list(np.arange(0.0, L, float(step_m)))
    if dists[-1] < L:
        dists.append(L)
    pts = [line.interpolate(d) for d in dists]
    return LineString([(p.x, p.y) for p in pts])


def _compute_junction_points(lines: List[LineString], snap_m: float, min_degree: int = 3) -> List[Point]:
    """Approximate junction points by counting snapped reach endpoints.

    This is a fast, topology-light proxy that works well when reach endpoints are already
    snapped by upstream preprocessing (e.g., river_network snap_m).
    """
    if not lines:
        return []
    snap_m = float(snap_m) if snap_m and snap_m > 0 else 30.0

    def _key(pt: Point) -> tuple[int, int]:
        return (int(round(pt.x / snap_m)), int(round(pt.y / snap_m)))

    counts: Dict[tuple[int, int], int] = {}
    reps: Dict[tuple[int, int], Point] = {}
    for ln in lines:
        if ln is None or ln.is_empty:
            continue
        try:
            c0 = Point(ln.coords[0])
            c1 = Point(ln.coords[-1])
        except Exception:
            continue
        for pt in (c0, c1):
            k = _key(pt)
            counts[k] = counts.get(k, 0) + 1
            if k not in reps:
                reps[k] = pt

    out = [reps[k] for k, n in counts.items() if n >= int(min_degree)]
    return out


def _is_harmful_intersection(g1: LineString, g2: LineString, tol_m: float) -> bool:
    """Return True if g1 and g2 intersect in a way likely to cause interpolation artifacts."""
    if not g1.intersects(g2):
        return False
    inter = g1.intersection(g2)
    if inter.is_empty:
        return False

    # For multi-intersections, treat as harmful (rare but typically messy).
    pts: List[Point] = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type == "MultiPoint":
        pts = list(inter.geoms)
    else:
        return True

    a0 = Point(g1.coords[0]); a1 = Point(g1.coords[-1])
    b0 = Point(g2.coords[0]); b1 = Point(g2.coords[-1])
    tol_m = float(tol_m) if tol_m is not None else 0.0

    for p in pts:
        # If the intersection is very close to an endpoint on either line, treat as benign.
        if min(p.distance(a0), p.distance(a1)) <= tol_m:
            continue
        if min(p.distance(b0), p.distance(b1)) <= tol_m:
            continue
        return True

    return False





def _global_deconflict_xs_all(xs_lines_records: List[Dict], tol_m: float) -> List[Dict]:
    """Conservatively drop cross-sections that intersect across the entire AOI.

    Policy (deterministic):
      - Score each XS by (xs_len_m, then xs_id) and keep higher-score first.
      - If a candidate intersects any already-kept XS in a *harmful* way, drop it.

    Notes:
      - "Harmful" intersections exclude benign endpoint touches within tol_m.
      - This is intentionally conservative to reduce interpolation artifacts where
        XS from adjacent reaches or tight meanders intersect.
    """
    if len(xs_lines_records) < 2:
        return xs_lines_records

    geoms: List[LineString] = []
    idx_map: List[int] = []
    for i, rec in enumerate(xs_lines_records):
        g = rec.get("geometry")
        if g is None or getattr(g, "is_empty", True):
            continue
        if getattr(g, "geom_type", None) != "LineString":
            continue
        geoms.append(g)
        idx_map.append(i)

    if len(geoms) < 2:
        return xs_lines_records

    def _score(rec: Dict) -> tuple[float, str]:
        L = float(rec.get("xs_len_m", 0.0) or 0.0)
        xid = str(rec.get("xs_id", ""))
        return (L, xid)

    scores = {i: _score(xs_lines_records[i]) for i in idx_map}

    tree = STRtree(geoms)

    # record-index <-> geometry-index mapping
    geom_to_rec = {g_i: rec_i for g_i, rec_i in enumerate(idx_map)}
    rec_to_geom = {rec_i: g_i for g_i, rec_i in enumerate(idx_map)}

    sorted_recs = sorted(idx_map, key=lambda i: scores[i], reverse=True)

    kept: Set[int] = set()
    dropped: Set[int] = set()

    for rec_i in sorted_recs:
        if rec_i in dropped:
            continue
        g_i = rec_to_geom.get(rec_i)
        if g_i is None:
            continue
        g = geoms[g_i]

        try:
            hits = tree.query(g)
        except Exception:
            hits = []

        hit_geom_indices: List[int] = []
        if hits is None or len(hits) == 0:
            hit_geom_indices = []
        else:
            # Shapely 2 returns indices; Shapely 1 may return geometries
            if isinstance(hits[0], (int, np.integer)):
                hit_geom_indices = [int(h) for h in hits]
            else:
                # Build id-based map for fallback
                id_map = {id(gg): ii for ii, gg in enumerate(geoms)}
                for hg in hits:
                    ii = id_map.get(id(hg))
                    if ii is not None:
                        hit_geom_indices.append(ii)

        conflict = False
        for hg_i in hit_geom_indices:
            if hg_i == g_i:
                continue
            other_rec_i = geom_to_rec.get(hg_i)
            if other_rec_i is None or other_rec_i not in kept:
                continue  # only compare to already-kept higher-score XS
            try:
                if _is_harmful_intersection(g, geoms[hg_i], tol_m=tol_m):
                    conflict = True
                    break
            except Exception:
                continue

        if conflict:
            dropped.add(rec_i)
        else:
            kept.add(rec_i)

    if dropped:
        log.info("Global deconflict (all) dropped %d intersecting XS across AOI.", len(dropped))

    # Return records in original order for stability
    out: List[Dict] = []
    for i, rec in enumerate(xs_lines_records):
        if i in idx_map:
            if i in kept:
                out.append(rec)
        else:
            out.append(rec)
    return out
def _global_deconflict_xs(xs_list: List[Dict], tol_m: float) -> List[Dict]:
    """Drop cross-sections that intersect other cross-sections within a reach.

    Strategy (deterministic):
      - Iterate in station order.
      - Keep the first XS in any conflicting group; drop later ones.
    """
    if len(xs_list) < 2:
        return xs_list

    kept: List[Dict] = []
    for rec in xs_list:
        g = rec.get("geometry")
        if g is None or g.is_empty:
            continue
        conflict = False
        for prev in kept:
            g2 = prev.get("geometry")
            if g2 is None or g2.is_empty:
                continue
            if _is_harmful_intersection(g, g2, tol_m=tol_m):
                conflict = True
                break
        if not conflict:
            kept.append(rec)

    dropped = len(xs_list) - len(kept)
    if dropped > 0:
        log.info("Global deconflict dropped %d intersecting XS within reach.", dropped)
    return kept


# --------------------------------------------------------------------------------------
# Raster sampling with CRS transforms + nodata handling
# --------------------------------------------------------------------------------------

def _transform_coords(coords_xy: List[Tuple[float, float]], transformer: Optional[Transformer]) -> List[Tuple[float, float]]:
    if transformer is None:
        return coords_xy
    xs = [c[0] for c in coords_xy]
    ys = [c[1] for c in coords_xy]
    x2, y2 = transformer.transform(xs, ys)
    return list(zip(x2, y2))


def _sample_dataset(ds: rasterio.DatasetReader, coords_in_ds_crs: List[Tuple[float, float]]) -> np.ndarray:
    nodata = ds.nodata
    vals: List[float] = []
    for v in ds.sample(coords_in_ds_crs):
        if v is None or len(v) == 0:
            vals.append(np.nan)
            continue
        z = float(v[0])
        if nodata is not None and np.isfinite(nodata) and z == float(nodata):
            vals.append(np.nan)
            continue
        if z <= -1e20 or z >= 1e20:
            vals.append(np.nan)
            continue
        vals.append(z)
    return np.asarray(vals, dtype="float64")


def sample_rasters_along_line(
    line: LineString,
    dem_ds: rasterio.DatasetReader,
    topo_ds: Optional[rasterio.DatasetReader],
    step_m: float,
    xform_to_dem: Optional[Transformer],
    xform_to_topo: Optional[Transformer],
) -> pd.DataFrame:
    length = float(line.length)
    if length == 0:
        return pd.DataFrame(columns=["x", "y", "dist_m", "z_dem", "z_topo"])

    n = int(np.floor(length / step_m)) + 1
    dists = np.linspace(0.0, length, n)
    pts = [line.interpolate(d) for d in dists]
    coords_river = [(p.x, p.y) for p in pts]

    coords_dem = _transform_coords(coords_river, xform_to_dem)
    dem_vals = _sample_dataset(dem_ds, coords_dem)

    if topo_ds is not None:
        coords_topo = _transform_coords(coords_river, xform_to_topo)
        topo_vals = _sample_dataset(topo_ds, coords_topo)
    else:
        topo_vals = np.full_like(dem_vals, np.nan, dtype="float64")

    return pd.DataFrame(
        {"x": [c[0] for c in coords_river],
         "y": [c[1] for c in coords_river],
         "dist_m": dists,
         "z_dem": dem_vals,
         "z_topo": topo_vals}
    )


# --------------------------------------------------------------------------------------
# Bank picking (corridor-aware refinement with endpoint fallback)
# --------------------------------------------------------------------------------------

def _rolling_nanmedian(arr: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(arr, dtype="float64")
    if arr.size == 0 or window_samples <= 1:
        return arr.copy()
    half = max(1, int(window_samples) // 2)
    out = np.full(arr.shape, np.nan, dtype="float64")
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        chunk = arr[lo:hi]
        if np.isfinite(chunk).any():
            out[i] = float(np.nanmedian(chunk))
    return out


def _distance_on_line(line: LineString, geom) -> Optional[float]:
    try:
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return float(line.project(geom))
        if geom.geom_type == "MultiPoint":
            vals = [float(line.project(g)) for g in geom.geoms if g is not None and (not g.is_empty)]
            return vals[0] if vals else None
        if geom.geom_type in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
            if not coords:
                return None
            mid = Point(coords[len(coords) // 2])
            return float(line.project(mid))
        if hasattr(geom, "geoms"):
            vals = []
            for g in geom.geoms:
                v = _distance_on_line(line, g)
                if v is not None:
                    vals.append(v)
            if vals:
                vals.sort()
                return vals[0]
    except Exception:
        log.debug("ignored", exc_info=True)
    return None


def estimate_bank_edge_distances(
    xs_line: LineString,
    center_pt: Optional[Point],
    bank_domain_geom,
) -> Tuple[Optional[float], Optional[float]]:
    if xs_line is None or xs_line.is_empty or bank_domain_geom is None or getattr(bank_domain_geom, "is_empty", True):
        return None, None
    try:
        inter = xs_line.intersection(bank_domain_geom)
    except Exception:
        log.debug("ignored", exc_info=True)
        return None, None
    if inter is None or inter.is_empty:
        return None, None

    center_dist = float(xs_line.length) * 0.5
    if center_pt is not None and not center_pt.is_empty:
        try:
            center_dist = float(xs_line.project(center_pt))
        except Exception:
            log.debug("ignored", exc_info=True)

    segments: List[Tuple[float, float]] = []

    def _add_segment(g):
        if g is None or g.is_empty:
            return
        if g.geom_type == "Point":
            d = float(xs_line.project(g))
            segments.append((d, d))
            return
        if g.geom_type == "MultiPoint":
            ds = sorted(float(xs_line.project(pt)) for pt in g.geoms if pt is not None and (not pt.is_empty))
            if len(ds) >= 2:
                for a, b in zip(ds[:-1:2], ds[1::2]):
                    segments.append((a, b))
            elif ds:
                segments.append((ds[0], ds[0]))
            return
        if g.geom_type in {"LineString", "LinearRing"}:
            coords = list(g.coords)
            if coords:
                ds = sorted(float(xs_line.project(Point(c))) for c in (coords[0], coords[-1]))
                segments.append((ds[0], ds[-1]))
            return
        if hasattr(g, "geoms"):
            for sub in g.geoms:
                _add_segment(sub)

    _add_segment(inter)
    if not segments:
        return None, None

    chosen = None
    best_score = None
    for a, b in segments:
        lo, hi = min(a, b), max(a, b)
        contains_center = (lo - 1e-6) <= center_dist <= (hi + 1e-6)
        score = (0 if contains_center else 1, abs(((lo + hi) * 0.5) - center_dist), -(hi - lo))
        if best_score is None or score < best_score:
            best_score = score
            chosen = (lo, hi)
    if chosen is None:
        return None, None
    return chosen[0], chosen[1]


def _pick_bank_in_window(
    z: np.ndarray,
    z_smooth: np.ndarray,
    d: np.ndarray,
    expected_dist: float,
    refine_m: float,
    quantile: float,
) -> Optional[int]:
    if d.size == 0:
        return None
    mask = np.abs(d - float(expected_dist)) <= max(float(refine_m), 1e-6)
    if not np.any(mask):
        return None
    idxs = np.where(mask)[0]
    zc = z[idxs]
    zsc = z_smooth[idxs]
    valid = np.isfinite(zc)
    if not np.any(valid):
        return None
    idxs = idxs[valid]
    zc = zc[valid]
    zsc = zsc[valid]
    quality = np.where(np.isfinite(zsc), zsc, zc)
    finite_quality = quality[np.isfinite(quality)]
    if finite_quality.size == 0:
        return None
    q = float(np.nanquantile(finite_quality, min(max(float(quantile), 0.5), 0.99)))
    keep = quality >= q
    cand = idxs[keep] if np.any(keep) else idxs
    if cand.size == 0:
        return None
    distances = np.abs(d[cand] - float(expected_dist))
    if cand.size > 1:
        qual_cand = np.where(np.isfinite(z_smooth[cand]), z_smooth[cand], z[cand])
        order = np.lexsort((-qual_cand, distances))
        return int(cand[order[0]])
    return int(cand[0])


def pick_banks(
    profile: pd.DataFrame,
    bank_search_m: float,
    prefer_topo: bool = True,
    expected_left_dist_m: Optional[float] = None,
    expected_right_dist_m: Optional[float] = None,
    bank_edge_refine_m: float = 12.0,
    bank_quantile: float = 0.85,
    bank_smooth_window_m: float = 8.0,
) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "method": "endpoint_peak_fallback",
        "expected_left_dist_m": float(expected_left_dist_m) if expected_left_dist_m is not None else np.nan,
        "expected_right_dist_m": float(expected_right_dist_m) if expected_right_dist_m is not None else np.nan,
    }
    if profile is None or profile.empty:
        return None, None, meta

    use_topo = prefer_topo and profile["z_topo"].notna().any()
    z = profile["z_topo"].to_numpy() if use_topo else profile["z_dem"].to_numpy()
    d = profile["dist_m"].to_numpy()
    L = float(d[-1]) if len(d) else 0.0
    if L <= 0:
        return None, None, meta

    step = float(np.nanmedian(np.diff(d))) if len(d) > 1 and np.isfinite(np.diff(d)).any() else 1.0
    smooth_samples = max(1, int(round(max(float(bank_smooth_window_m), step) / max(step, 1e-6))))
    z_smooth = _rolling_nanmedian(z, smooth_samples)

    idx_left = None
    idx_right = None
    refined = False

    if expected_left_dist_m is not None:
        idx_left = _pick_bank_in_window(z, z_smooth, d, float(expected_left_dist_m), bank_edge_refine_m, bank_quantile)
        refined = refined or (idx_left is not None)
    if expected_right_dist_m is not None:
        idx_right = _pick_bank_in_window(z, z_smooth, d, float(expected_right_dist_m), bank_edge_refine_m, bank_quantile)
        refined = refined or (idx_right is not None)

    left_mask = d <= min(bank_search_m, L)
    right_mask = d >= max(0.0, L - bank_search_m)

    if idx_left is None and np.any(left_mask):
        zl = z[left_mask]
        if np.isfinite(zl).any():
            j = int(np.nanargmax(zl))
            idx_left = int(np.where(left_mask)[0][j])
    if idx_right is None and np.any(right_mask):
        zr = z[right_mask]
        if np.isfinite(zr).any():
            j = int(np.nanargmax(zr))
            idx_right = int(np.where(right_mask)[0][j])

    if refined and (idx_left is not None or idx_right is not None):
        meta["method"] = "corridor_edge_refined"
    meta["source"] = "topo" if use_topo else "dem"
    return idx_left, idx_right, meta

# --------------------------------------------------------------------------------------
# Build XS
# --------------------------------------------------------------------------------------

def build_xs_for_river(
    rivers_clip: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    cfg: XSConfig,
    dem_path: Path,
    topo_path: Optional[Path],
    out_gpkg: Path,
    out_csv: Optional[Path],
    enable_component_prune: bool,
    keep_top_components: int,
    min_stream_order: int,
    min_length_km: float,
    ftype_allow: List[int],
    include_artificial_path: bool,
    bank_domain_gdf: Optional[gpd.GeoDataFrame] = None,
) -> None:
    rivers_clip = _explode_lines(rivers_clip)
    edges = edges.copy()

    bank_domain_geom = None
    if bank_domain_gdf is not None and not bank_domain_gdf.empty:
        try:
            bank_domain_gdf = bank_domain_gdf.copy()
            bank_domain_gdf = bank_domain_gdf[bank_domain_gdf.geometry.notnull() & ~bank_domain_gdf.geometry.is_empty]
            if not bank_domain_gdf.empty:
                if bank_domain_gdf.crs != rivers_clip.crs:
                    bank_domain_gdf = bank_domain_gdf.to_crs(rivers_clip.crs)
                bank_domain_geom = bank_domain_gdf.geometry.union_all() if hasattr(bank_domain_gdf.geometry, "union_all") else bank_domain_gdf.unary_union
        except Exception:
            bank_domain_geom = None
            log.debug("ignored", exc_info=True)

    if enable_component_prune:
        rivers_clip = filter_by_component_length(rivers_clip, edges, keep_top_components=keep_top_components)

    rivers_clip = filter_centerlines(
        rivers_clip,
        min_stream_order=min_stream_order,
        min_length_km=min_length_km,
        ftype_allow=ftype_allow,
        include_artificial_path=include_artificial_path,
    )

    comp_map: Dict[str, int] = {}
    if "river_id" in edges.columns and "component_id" in edges.columns:
        for rid, cid in zip(edges["river_id"].astype(str), edges["component_id"].astype(int)):
            comp_map[rid] = int(cid)

    rivers_crs = CRS.from_user_input(rivers_clip.crs)

    with rasterio.open(dem_path) as dem_ds:
        if dem_ds.crs is None:
            raise RuntimeError(f"DEM has no CRS: {dem_path}")
        dem_crs = CRS.from_user_input(dem_ds.crs)

        topo_ds = topo_ds_ctx = rasterio.open(topo_path) if topo_path else None
        topo_crs = CRS.from_user_input(topo_ds.crs) if topo_ds is not None else None

        xform_to_dem = None if rivers_crs == dem_crs else Transformer.from_crs(rivers_crs, dem_crs, always_xy=True)
        xform_to_topo = None
        if topo_ds is not None:
            if topo_crs is None:
                raise RuntimeError(f"Topo raster has no CRS: {topo_path}")
            xform_to_topo = None if rivers_crs == topo_crs else Transformer.from_crs(rivers_crs, topo_crs, always_xy=True)

        log.info("rivers=%s | dem=%s | topo=%s", rivers_crs.to_string(), dem_crs.to_string(), topo_crs.to_string() if topo_crs else "<none>")
        log.info("nodata=%s | bounds=%s", str(dem_ds.nodata), str(dem_ds.bounds))

        xs_lines_records = []
        xs_points_records = []
        xs_id_counter = 1

        # Pre-compute junction nodes from snapped reach endpoints (fast proxy for confluences).
        junction_pts: List[Point] = []
        junction_tree = None
        if bool(cfg.skip_junctions) and float(cfg.junction_buffer_m) > 0:
            try:
                lines: List[LineString] = []
                for _, r in rivers_clip.iterrows():
                    g = _ensure_single_linestring(r.geometry)
                    if g is None or g.is_empty:
                        continue
                    g = _densify_linestring(g, float(cfg.densify_step_m))
                    lines.append(g)
                junction_pts = _compute_junction_points(lines, snap_m=float(cfg.junction_snap_m), min_degree=3)
                if junction_pts:
                    junction_tree = STRtree(junction_pts)
                    log.info("Detected %d junction node(s) from reach endpoints.", len(junction_pts))
            except Exception as e:
                log.debug("Junction detection failed: %s", e)
                junction_tree = None
        # Determine smoothing window (use spacing if not explicit)
        smoothing_eps = (cfg.smoothing_window_m / 2.0) if cfg.smoothing_window_m > 0 else (cfg.spacing_m / 2.0)
        smoothing_eps = max(smoothing_eps, 0.5)

        for i, row in rivers_clip.iterrows():
            geom = _ensure_single_linestring(row.geometry)
            if geom is None:
                continue

            # Densify to stabilize tangents and reduce spurious XS crossings on coarse centerlines
            geom = _densify_linestring(geom, float(cfg.densify_step_m))

            L = float(geom.length)
            if L < cfg.min_centerline_len_m:
                continue

            river_id = str(row.get("river_id", f"river_{i}"))
            component_id = int(row.get("component_id", comp_map.get(river_id, -1)))

            n_xs = int(np.floor(L / cfg.spacing_m)) + 1
            n_xs = min(n_xs, cfg.max_xs_per_reach)

            # 1. Generate XS Geometries first (pre-sampling)
            xs_batch = []
            
            for k in range(n_xs):
                s_center = min(L, k * cfg.spacing_m)
                center_pt = geom.interpolate(s_center)

                # Skip XS too close to a junction/confluence (reduces self-intersection artifacts)
                if junction_tree is not None:
                    try:
                        buf = center_pt.buffer(float(cfg.junction_buffer_m))
                        hits = junction_tree.query(buf)
                        # Shapely 2 returns indices; Shapely 1 may return geometries
                        if len(hits) > 0:
                            if isinstance(hits[0], (int, np.integer)):
                                _hit_jpts = [junction_pts[int(h)] for h in hits]
                            else:
                                _hit_jpts = list(hits)
                            if any(center_pt.distance(pt) <= float(cfg.junction_buffer_m) for pt in _hit_jpts):
                                continue
                    except Exception:
                        log.debug("Junction proximity check failed; skipping.", exc_info=True)

                # Use smoothed tangent for orientation
                tan = _line_tangent(geom, s_center, eps=smoothing_eps)
                if tan == (0.0, 0.0):
                    continue

                xs_line = build_xs_line(center_pt, tan, cfg.half_width_m)
                
                xs_id = f"xs_{xs_id_counter:08d}"
                xs_id_counter += 1
                
                xs_batch.append({
                    "xs_id": xs_id,
                    "river_id": river_id,
                    "component_id": component_id,
                    "s_center_m": float(s_center),
                    "geometry": xs_line,
                    "center_pt": center_pt
                })
            
            # 2. Trim overlapping XS if requested (adjacent pairs)
            if cfg.trim_overlaps and len(xs_batch) > 1:
                xs_batch = _trim_overlapping_xs(xs_batch)

            # 2b. Global deconflict within reach: drop XS that still intersect after trimming
            if bool(cfg.global_deconflict) and len(xs_batch) > 1:
                xs_batch = _global_deconflict_xs(xs_batch, tol_m=float(cfg.deconflict_tol_m))
            
            # 3. Sample Rasters along final geometries
            for rec in xs_batch:
                xs_line = rec['geometry']
                # Skip if trimming made it too short
                if xs_line.length < cfg.bank_search_m:
                    continue
                    
                prof = sample_rasters_along_line(
                    xs_line,
                    dem_ds=dem_ds,
                    topo_ds=topo_ds,
                    step_m=cfg.sample_step_m,
                    xform_to_dem=xform_to_dem,
                    xform_to_topo=xform_to_topo,
                )

                expected_left_dist_m = None
                expected_right_dist_m = None
                if bank_domain_geom is not None:
                    expected_left_dist_m, expected_right_dist_m = estimate_bank_edge_distances(
                        xs_line,
                        rec.get("center_pt"),
                        bank_domain_geom,
                    )

                idx_l, idx_r, bank_pick_meta = pick_banks(
                    prof,
                    bank_search_m=cfg.bank_search_m,
                    prefer_topo=True,
                    expected_left_dist_m=expected_left_dist_m,
                    expected_right_dist_m=expected_right_dist_m,
                    bank_edge_refine_m=cfg.bank_edge_refine_m,
                    bank_quantile=cfg.bank_quantile,
                    bank_smooth_window_m=cfg.bank_smooth_window_m,
                )

                def _bank_z(idx: Optional[int]) -> float:
                    if idx is None:
                        return np.nan
                    zt = prof.loc[idx, "z_topo"]
                    if pd.notna(zt):
                        return float(zt)
                    zd = prof.loc[idx, "z_dem"]
                    return float(zd) if pd.notna(zd) else np.nan
                
                # Add to lines result
                rec.update({
                    "xs_len_m": float(xs_line.length),
                    "bank_left_dist_m": float(prof.loc[idx_l, "dist_m"]) if idx_l is not None else np.nan,
                    "bank_right_dist_m": float(prof.loc[idx_r, "dist_m"]) if idx_r is not None else np.nan,
                    "bank_left_z_m": _bank_z(idx_l),
                    "bank_right_z_m": _bank_z(idx_r),
                    "bank_left_expected_dist_m": float(expected_left_dist_m) if expected_left_dist_m is not None else np.nan,
                    "bank_right_expected_dist_m": float(expected_right_dist_m) if expected_right_dist_m is not None else np.nan,
                    "bank_pick_method": str(bank_pick_meta.get("method", "unknown")),
                    "bank_pick_source": str(bank_pick_meta.get("source", "unknown")),
                })
                # Remove temp key
                if 'center_pt' in rec: del rec['center_pt']
                
                xs_lines_records.append(rec)

                # Add to points result
                prof = prof.copy()
                prof["xs_id"] = rec["xs_id"]
                prof["river_id"] = rec["river_id"]
                prof["component_id"] = rec["component_id"]
                prof["is_bank_left"] = False
                prof["is_bank_right"] = False
                prof["bank_pick_method"] = str(bank_pick_meta.get("method", "unknown"))
                if idx_l is not None:
                    prof.loc[idx_l, "is_bank_left"] = True
                if idx_r is not None:
                    prof.loc[idx_r, "is_bank_right"] = True

                for _, pr in prof.iterrows():
                    xs_points_records.append(
                        {
                            "xs_id": pr["xs_id"],
                            "river_id": pr["river_id"],
                            "component_id": int(pr["component_id"]),
                            "dist_m": float(pr["dist_m"]),
                            "z_dem": float(pr["z_dem"]) if pd.notna(pr["z_dem"]) else np.nan,
                            "z_topo": float(pr["z_topo"]) if pd.notna(pr["z_topo"]) else np.nan,
                            "is_bank_left": bool(pr["is_bank_left"]),
                            "is_bank_right": bool(pr["is_bank_right"]),
                            "geometry": Point(float(pr["x"]), float(pr["y"])),
                        }
                    )

        out_gpkg = Path(out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)

        if not xs_lines_records:
            log.warning("No cross-sections were generated. Check river filtering or DEM coverage.")
            return

        # Conservative: drop intersecting XS across the entire AOI to reduce artifacts at tight meanders
        # and reach boundaries (at the cost of fewer XS).
        if cfg.global_deconflict_all and len(xs_lines_records) > 1:
            before = len(xs_lines_records)
            xs_lines_records = _global_deconflict_xs_all(xs_lines_records, tol_m=float(cfg.deconflict_tol_m))
            kept_ids = {r.get('xs_id') for r in xs_lines_records}
            xs_points_records = [r for r in xs_points_records if r.get('xs_id') in kept_ids]
            dropped = before - len(xs_lines_records)

        xs_lines_gdf = gpd.GeoDataFrame(xs_lines_records, crs=rivers_clip.crs)
        xs_pts_gdf = gpd.GeoDataFrame(xs_points_records, crs=rivers_clip.crs)

        log.info("%s (xs_lines=%d, xs_points=%d)", out_gpkg, len(xs_lines_gdf), len(xs_pts_gdf))
        xs_lines_gdf.to_file(out_gpkg, layer="xs_lines", driver="GPKG")
        xs_pts_gdf.to_file(out_gpkg, layer="xs_points", driver="GPKG")

        if out_csv:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_csv = xs_pts_gdf.drop(columns=["geometry"]).copy()
            df_csv.to_csv(out_csv, index=False)
            log.info("%s", out_csv)

        if topo_ds_ctx is not None:
            try:
                topo_ds_ctx.close()
            except Exception:
                log.debug("ignored", exc_info=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "xs_builder.py – build cross-sections from river network + DEM + topo lidar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--river-gpkg", required=True, help="river_network.gpkg produced by river_network.py")
    p.add_argument("--dem", required=True, help="DEM raster (GeoTIFF) used for elevations/banks")
    p.add_argument("--topo-lidar", default=None, help="Optional topo lidar raster. If omitted, banks use --dem.")
    p.add_argument("--out-gpkg", required=True, help="Output GeoPackage (xs_lines, xs_points)")
    p.add_argument("--out-csv", default=None, help="Optional CSV output (xs_points attributes)")

    p.add_argument("--rivers-layer", default="rivers_clip", help="Centerline layer in river_gpkg")
    p.add_argument("--edges-layer", default="graph_edges", help="Graph edges layer in river_gpkg")

    p.add_argument("--spacing-m", type=float, default=200.0, help="Spacing between XS along centerline (m)")
    p.add_argument("--half-width-m", type=float, default=150.0, help="Half-width of each XS (m)")
    p.add_argument("--sample-step-m", type=float, default=2.0, help="Sampling step along XS (m)")
    p.add_argument("--bank-search-m", type=float, default=40.0, help="Search window near each XS end for fallback bank peak search (m)")
    p.add_argument("--bank-edge-refine-m", type=float, default=12.0, help="Half-width of corridor-edge refinement window around expected bank edge (m)")
    p.add_argument("--bank-quantile", type=float, default=0.85, help="Upper quantile used inside the corridor-edge refinement window before choosing nearest bank candidate")
    p.add_argument("--bank-smooth-window-m", type=float, default=8.0, help="Rolling-median smoothing scale (m) used before bank picking")
    p.add_argument("--min-centerline-len-m", type=float, default=50.0, help="Skip centerlines shorter than this (m)")
    
    # Overlap and smoothing options
    p.add_argument("--smoothing-window-m", type=float, default=0.0, 
                   help="Window size for calculating tangent/orientation. 0 = auto (uses spacing-m). Larger values smooth out XS direction at bends.")
    p.add_argument("--trim-overlaps", action="store_true", default=True, 
                   help="Trim intersecting cross-sections (default True).")
    p.add_argument("--no-trim-overlaps", dest="trim_overlaps", action="store_false", help="Disable overlap trimming.")

    p.add_argument("--deconflict-tol-m", type=float, default=2.0,
                   help="Endpoint tolerance (m) when identifying intersecting cross-sections.")
    p.add_argument("--no-global-deconflict", dest="global_deconflict", action="store_false",
                   help="Disable dropping XS that intersect non-adjacent XS within a reach.")
    p.set_defaults(global_deconflict=True)

    p.add_argument("--no-global-deconflict-all", dest="global_deconflict_all", action="store_false",
                   help="Disable conservative AOI-wide dropping of intersecting cross-sections.")
    p.set_defaults(global_deconflict_all=True)

    p.add_argument("--no-skip-junctions", dest="skip_junctions", action="store_false",
                   help="Do not skip XS near confluences/junctions.")
    p.set_defaults(skip_junctions=True)
    p.add_argument("--junction-snap-m", type=float, default=30.0,
                   help="Snapping scale (m) for junction detection from reach endpoints.")
    p.add_argument("--junction-buffer-m", type=float, default=120.0,
                   help="Skip XS within this distance (m) of junction nodes.")
    p.add_argument("--densify-step-m", type=float, default=20.0,
                   help="Densify centerlines to this vertex spacing (m) before computing tangents.")

    # Component pruning (ON by default)
    p.add_argument("--disable-component-prune", action="store_true", help="Disable component-first pruning.")
    p.add_argument("--keep-top-components", type=int, default=1, help="Keep the N largest components by total length.")

    # Attribute filters (secondary)
    p.add_argument("--min-stream-order", type=int, default=3, help="Minimum stream order to keep (if field exists)")
    p.add_argument("--min-length-km", type=float, default=0.05, help="Minimum centerline length (km) to keep (if field exists)")
    p.add_argument(
        "--ftype-allow",
        default="460,558",
        help="Comma-separated allowed FType values. Default '460,558' keeps Stream/River + Artificial Path.",
    )
    p.add_argument(
        "--include-artificial-path",
        action="store_true",
        help="Ensure FType=558 is included even if not in --ftype-allow.",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = XSConfig(
        spacing_m=float(args.spacing_m),
        half_width_m=float(args.half_width_m),
        sample_step_m=float(args.sample_step_m),
        bank_search_m=float(args.bank_search_m),
        bank_edge_refine_m=float(args.bank_edge_refine_m),
        bank_quantile=float(args.bank_quantile),
        bank_smooth_window_m=float(args.bank_smooth_window_m),
        min_centerline_len_m=float(args.min_centerline_len_m),
        smoothing_window_m=float(args.smoothing_window_m),
        trim_overlaps=bool(args.trim_overlaps),
        global_deconflict=bool(getattr(args, "global_deconflict", True)),
        global_deconflict_all=bool(getattr(args, "global_deconflict_all", True)),
        deconflict_tol_m=float(getattr(args, "deconflict_tol_m", 2.0)),
        skip_junctions=bool(getattr(args, "skip_junctions", True)),
        junction_snap_m=float(getattr(args, "junction_snap_m", 30.0)),
        junction_buffer_m=float(getattr(args, "junction_buffer_m", 120.0)),
        densify_step_m=float(getattr(args, "densify_step_m", 20.0)),
    )

    river_gpkg = Path(args.river_gpkg)
    rivers = _read_layer(river_gpkg, args.rivers_layer)
    edges = _read_layer(river_gpkg, args.edges_layer)
    bank_domain_gdf = None
    for layer_name in ("nhdarea_clip", "nhdarea_aoi"):
        try:
            bank_domain_gdf = _read_layer(river_gpkg, layer_name)
            if bank_domain_gdf is not None and not bank_domain_gdf.empty:
                log.info("Using %s as corridor-aware bank domain for XS bank picking.", layer_name)
                break
        except Exception:
            bank_domain_gdf = None
            log.debug("No optional bank domain layer %s available.", layer_name, exc_info=True)

    if CRS.from_user_input(rivers.crs).is_geographic:
        raise RuntimeError(
            f"rivers layer CRS looks geographic ({rivers.crs}). "
            "Re-run river_network.py so it outputs projected CRS (auto-UTM), or pass --out-crs there."
        )

    ftype_allow: List[int] = []
    for s in str(args.ftype_allow).split(","):
        s = s.strip()
        if not s:
            continue
        try:
            ftype_allow.append(int(float(s)))
        except Exception:
            log.debug("Could not parse ftype value %r; skipping.", s, exc_info=True)
    if not ftype_allow:
        ftype_allow = [460, 558]

    build_xs_for_river(
        rivers_clip=rivers,
        edges=edges,
        cfg=cfg,
        dem_path=Path(args.dem),
        topo_path=Path(args.topo_lidar) if args.topo_lidar else None,
        out_gpkg=Path(args.out_gpkg),
        out_csv=Path(args.out_csv) if args.out_csv else None,
        enable_component_prune=not bool(args.disable_component_prune),
        keep_top_components=int(args.keep_top_components),
        min_stream_order=int(args.min_stream_order),
        min_length_km=float(args.min_length_km),
        ftype_allow=ftype_allow,
        include_artificial_path=bool(args.include_artificial_path),
        bank_domain_gdf=bank_domain_gdf,
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
