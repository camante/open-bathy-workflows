#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xs_builder.py

Build river cross-sections (XS) for CUDEM-style coastal river bathymetry workflows.

What this version fixes
-----------------------
1) "Too many minor streams":
   - Adds a *component-first* pruning option (recommended for tidal/coastal rivers):
       keep the largest connected component(s) by total network length.
   - Then applies attribute filters (stream order / lengthkm / ftype) as a second pass.

   This avoids the common coastal failure mode where the main trunk is encoded as
   FType=558 "Artificial Path" or has low/odd StreamOrde, and gets filtered out.

2) DEM sampling returning -9999 / NULL:
   - Cross-sections are built in the projected CRS of rivers_clip (meters).
   - Rasters may be in a different CRS (e.g., EPSG:4269). We now reproject sample
     coordinates into each raster's CRS before sampling.
   - Nodata values (e.g., -9999) are converted to NaN.

Inputs
------
- river_network.gpkg (from river_network.py), layers:
    - rivers_clip   : LineString centerlines clipped to AOI
    - graph_edges   : edges with component_id and lengths (or geometry)

- Rasters:
    - --dem         : DEM used to sample elevations (often your bank/topo DEM)
    - --topo-lidar  : optional higher-quality topo raster; if omitted, banks use --dem

Outputs
-------
- Output GeoPackage with layers:
    - xs_lines   : cross-section LineStrings + metadata
    - xs_points  : sampled points along XS with z_dem/z_topo + bank flags

- Optional CSV:
    - xs_profiles.csv : long table of xs_points attributes (geometry dropped)

Dependencies
------------
pip install geopandas rasterio shapely numpy pandas pyproj
"""

from __future__ import annotations

import argparse
import logging


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from pyproj import CRS, Transformer


# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("xs_builder")


@dataclass
class XSConfig:
    spacing_m: float = 200.0
    half_width_m: float = 150.0
    sample_step_m: float = 2.0
    bank_search_m: float = 40.0
    min_centerline_len_m: float = 50.0
    max_xs_per_reach: int = 2000


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
        log.warning("[COMPONENT] graph_edges has no component_id; skipping component pruning.")
        return rivers

    rivers = _attach_component_id_to_rivers(rivers, edges)
    if "component_id" not in rivers.columns:
        log.warning("[COMPONENT] rivers_clip has no component_id and could not be joined; skipping component pruning.")
        return rivers

    comp_len = compute_component_lengths(edges)
    if comp_len.empty:
        log.warning("[COMPONENT] No component lengths computed; skipping component pruning.")
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
        log.info("[FILTER] ftype allow=%s (%s): %d → %d", ftype_allow, col_ftype, before, len(gdf))
    else:
        log.info("[FILTER] No ftype field found; skipping ftype filter.")

    if col_order:
        before = len(gdf)
        gdf = gdf[pd.to_numeric(gdf[col_order], errors="coerce").fillna(-1) >= int(min_stream_order)]
        log.info("[FILTER] min_stream_order=%d (%s): %d → %d", int(min_stream_order), col_order, before, len(gdf))
    else:
        log.info("[FILTER] No stream order field found; skipping stream order filter.")

    if col_len:
        before = len(gdf)
        gdf = gdf[pd.to_numeric(gdf[col_len], errors="coerce").fillna(0.0) >= float(min_length_km)]
        log.info("[FILTER] min_length_km=%.3f (%s): %d → %d", float(min_length_km), col_len, before, len(gdf))
    else:
        log.info("[FILTER] No lengthkm field found; skipping length filter.")

    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    log.info("[FILTER] centerlines kept: %d / %d", len(gdf), n0)
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
    L = line.length
    s0 = max(0.0, min(L, s - eps))
    s1 = max(0.0, min(L, s + eps))
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
# Bank picking (minimal heuristic)
# --------------------------------------------------------------------------------------

def pick_banks(profile: pd.DataFrame, bank_search_m: float, prefer_topo: bool = True) -> Tuple[Optional[int], Optional[int]]:
    if profile is None or profile.empty:
        return None, None

    use_topo = prefer_topo and profile["z_topo"].notna().any()
    z = profile["z_topo"].to_numpy() if use_topo else profile["z_dem"].to_numpy()
    d = profile["dist_m"].to_numpy()
    L = float(d[-1]) if len(d) else 0.0
    if L <= 0:
        return None, None

    left_mask = d <= min(bank_search_m, L)
    right_mask = d >= max(0.0, L - bank_search_m)

    idx_left = None
    idx_right = None

    if np.any(left_mask):
        zl = z[left_mask]
        if np.isfinite(zl).any():
            j = int(np.nanargmax(zl))
            idx_left = int(np.where(left_mask)[0][j])

    if np.any(right_mask):
        zr = z[right_mask]
        if np.isfinite(zr).any():
            j = int(np.nanargmax(zr))
            idx_right = int(np.where(right_mask)[0][j])

    return idx_left, idx_right


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
) -> None:
    rivers_clip = _explode_lines(rivers_clip)
    edges = edges.copy()

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

        topo_ds = rasterio.open(topo_path) if topo_path else None
        topo_crs = CRS.from_user_input(topo_ds.crs) if topo_ds is not None else None

        xform_to_dem = None if rivers_crs == dem_crs else Transformer.from_crs(rivers_crs, dem_crs, always_xy=True)
        xform_to_topo = None
        if topo_ds is not None:
            if topo_crs is None:
                raise RuntimeError(f"Topo raster has no CRS: {topo_path}")
            xform_to_topo = None if rivers_crs == topo_crs else Transformer.from_crs(rivers_crs, topo_crs, always_xy=True)

        log.info("[CRS] rivers=%s | dem=%s | topo=%s", rivers_crs.to_string(), dem_crs.to_string(), topo_crs.to_string() if topo_crs else "<none>")
        log.info("[DEM] nodata=%s | bounds=%s", str(dem_ds.nodata), str(dem_ds.bounds))

        xs_lines_records = []
        xs_points_records = []
        xs_id_counter = 1

        for i, row in rivers_clip.iterrows():
            geom = _ensure_single_linestring(row.geometry)
            if geom is None:
                continue

            L = float(geom.length)
            if L < cfg.min_centerline_len_m:
                continue

            river_id = str(row.get("river_id", f"river_{i}"))
            component_id = int(row.get("component_id", comp_map.get(river_id, -1)))

            n_xs = int(np.floor(L / cfg.spacing_m)) + 1
            n_xs = min(n_xs, cfg.max_xs_per_reach)

            for k in range(n_xs):
                s_center = min(L, k * cfg.spacing_m)
                center_pt = geom.interpolate(s_center)

                eps = max(0.5, 0.01 * cfg.spacing_m)
                tan = _line_tangent(geom, s_center, eps=eps)
                if tan == (0.0, 0.0):
                    continue

                xs_line = build_xs_line(center_pt, tan, cfg.half_width_m)

                prof = sample_rasters_along_line(
                    xs_line,
                    dem_ds=dem_ds,
                    topo_ds=topo_ds,
                    step_m=cfg.sample_step_m,
                    xform_to_dem=xform_to_dem,
                    xform_to_topo=xform_to_topo,
                )

                idx_l, idx_r = pick_banks(prof, bank_search_m=cfg.bank_search_m, prefer_topo=True)

                xs_id = f"xs_{xs_id_counter:08d}"
                xs_id_counter += 1

                def _bank_z(idx: Optional[int]) -> float:
                    if idx is None:
                        return np.nan
                    zt = prof.loc[idx, "z_topo"]
                    if pd.notna(zt):
                        return float(zt)
                    zd = prof.loc[idx, "z_dem"]
                    return float(zd) if pd.notna(zd) else np.nan

                xs_lines_records.append(
                    {
                        "xs_id": xs_id,
                        "river_id": river_id,
                        "component_id": component_id,
                        "s_center_m": float(s_center),
                        "xs_len_m": float(xs_line.length),
                        "bank_left_dist_m": float(prof.loc[idx_l, "dist_m"]) if idx_l is not None else np.nan,
                        "bank_right_dist_m": float(prof.loc[idx_r, "dist_m"]) if idx_r is not None else np.nan,
                        "bank_left_z_m": _bank_z(idx_l),
                        "bank_right_z_m": _bank_z(idx_r),
                        "geometry": xs_line,
                    }
                )

                prof = prof.copy()
                prof["xs_id"] = xs_id
                prof["river_id"] = river_id
                prof["component_id"] = component_id
                prof["is_bank_left"] = False
                prof["is_bank_right"] = False
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

        xs_lines_gdf = gpd.GeoDataFrame(xs_lines_records, crs=rivers_clip.crs)
        xs_pts_gdf = gpd.GeoDataFrame(xs_points_records, crs=rivers_clip.crs)

        log.info("[WRITE] %s (xs_lines=%d, xs_points=%d)", out_gpkg, len(xs_lines_gdf), len(xs_pts_gdf))
        xs_lines_gdf.to_file(out_gpkg, layer="xs_lines", driver="GPKG")
        xs_pts_gdf.to_file(out_gpkg, layer="xs_points", driver="GPKG")

        if out_csv:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_csv = xs_pts_gdf.drop(columns=["geometry"]).copy()
            df_csv.to_csv(out_csv, index=False)
            log.info("[WRITE] %s", out_csv)

        if topo_ds is not None:
            topo_ds.close()


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
    p.add_argument("--bank-search-m", type=float, default=40.0, help="Search window near each XS end for bank peak (m)")
    p.add_argument("--min-centerline-len-m", type=float, default=50.0, help="Skip centerlines shorter than this (m)")

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
        min_centerline_len_m=float(args.min_centerline_len_m),
    )

    river_gpkg = Path(args.river_gpkg)
    rivers = _read_layer(river_gpkg, args.rivers_layer)
    edges = _read_layer(river_gpkg, args.edges_layer)

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
            pass
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
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
