#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_network.py – Download/prepare a river network for cross-section generation (TNM/NHD with HydroRIVERS fallback)

One-stop script: download (TNM/NHD) → ingest → AOI subset/clip → reach graph/topology.

NEW in this version
-------------------
By default the script ONLY attempts TNMAccess (NHD/NHDPlus-ish) acquisition.
HydroRIVERS is downloaded ONLY if no NHD flowlines can be obtained.

Download sources:
- TNMAccess (USGS The National Map API) for NHD-ish hydrography.
- HydroSHEDS / HydroRIVERS download server (regional shapefile zips).

Outputs (GeoPackage layers)
---------------------------
- rivers_aoi   : fast subset by AOI intersects filter (no clipping)
- rivers_clip  : clipped to AOI boundary
- graph_nodes  : unique endpoint nodes (snapped)
- graph_edges  : reach edges with from/to node ids, length_m, and pragmatic stationing

Stationing (s_m) is computed per connected component from a stable “root”:
  choose a degree-1 node with minimum Y (south-most) as root, else minimum Y node.
This is a pragmatic default for smoothing/plotting; you can replace with true routing
if you later use NHDPlus connectivity attributes.

Dependencies
------------
- requests
- geopandas
- shapely
- fiona
- pyproj
- numpy"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

import re
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box, Point
from pyproj import CRS
import fiona


# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("river_network")

TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASETS_URL = "https://tnmaccess.nationalmap.gov/api/v1/datasets"

# HydroRIVERS regional zip pattern (continent extracts; v10)
# Examples include ..._eu_shp.zip, ..._na_shp.zip, etc.
HYDRORIVERS_ZIP_TEMPLATE = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip"
HYDRORIVERS_REGIONS = {"af", "as", "au", "eu", "na", "sa"}


# --------------------------------------------------------------------------------------
# ArcGIS REST fallback (NHD / NHDPlus HR feature services)
# --------------------------------------------------------------------------------------

NHD_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
NHDPLUS_HR_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"

def arcgis_query_layer_geojson(
    base_url: str,
    layer_id: int,
    bbox_lonlat: Tuple[float, float, float, float],
    where: str = "1=1",
    out_fields: str = "*",
    timeout_s: int = 120,
    max_records: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Query an ArcGIS MapServer layer for features intersecting the AOI bbox.

    Uses pagination via resultOffset/resultRecordCount and requests GeoJSON output.

    Notes
    -----
    NHD and NHDPlus_HR MapServers publish layers like Flowline / NetworkNHDFlowline.
    """
    lonmin, lonmax, latmin, latmax = bbox_lonlat

    # ArcGIS expects envelope as xmin,ymin,xmax,ymax
    geom = f"{lonmin},{latmin},{lonmax},{latmax}"

    url = f"{base_url.rstrip('/')}/{int(layer_id)}/query"
    all_frames = []
    offset = 0

    while True:
        params = {
            "where": where,
            "geometry": geom,
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
            "resultRecordCount": int(max_records),
            "resultOffset": int(offset),
        }
        r = requests.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()

        # Some ArcGIS servers may return JSON with an "error" key
        js = r.json()
        if isinstance(js, dict) and "error" in js:
            raise RuntimeError(f"ArcGIS query error: {js.get('error')}")

        # Read GeoJSON into GeoDataFrame
        gdf = gpd.read_file(json.dumps(js), driver="GeoJSON")
        if gdf is None or gdf.empty:
            break

        all_frames.append(gdf)
        if len(gdf) < max_records:
            break
        offset += max_records

        # Safety stop: avoid runaway downloads on huge AOIs
        if offset > 200000:
            log.warning("[ARCGIS] Stopping pagination after %d features (safety cap).", offset)
            break

    if not all_frames:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    out = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True), crs="EPSG:4326")
    return out


def try_arcgis_nhd_flowlines(
    aoi: Tuple[float, float, float, float],
    out_crs: Optional[str],
    timeout_s: int = 120,
) -> gpd.GeoDataFrame:
    """
    Try to fetch flowlines via ArcGIS REST as a fallback when TNMAccess download returns 0.

    Priority:
      1) NHDPlus_HR NetworkNHDFlowline (layer 3)
      2) NHD Flowline - Large Scale (layer 6)

    Returns GeoDataFrame in projected CRS (out_crs or auto-UTM).
    """
    # 1) NHDPlus HR
    try:
        log.info("[ARCGIS] Querying NHDPlus_HR NetworkNHDFlowline (layer 3) ...")
        gdf_ll = arcgis_query_layer_geojson(NHDPLUS_HR_MAPSERVER, 3, aoi, timeout_s=timeout_s)
        if gdf_ll is not None and not gdf_ll.empty:
            # Normalize id
            if "COMID" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["COMID"].astype(str)
            elif "NHDPlusID" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["NHDPlusID"].astype(str)
            else:
                gdf_ll["river_id"] = [f"reach_{i}" for i in range(len(gdf_ll))]
            # Project
            lonc, latc = _aoi_center(aoi)
            crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))
            gdfp = gdf_ll.to_crs(crs_out)
            gdfp = _clean_lines(gdfp)
            gdfp["length_m"] = gdfp.geometry.length.astype("float64")
            gdfp["source"] = "arcgis_nhdplus_hr"
            return gdfp
    except Exception as e:
        log.warning("[ARCGIS] NHDPlus_HR query failed: %s", str(e))

    # 2) NHD Flowline Large Scale
    try:
        log.info("[ARCGIS] Querying NHD Flowline - Large Scale (layer 6) ...")
        gdf_ll = arcgis_query_layer_geojson(NHD_MAPSERVER, 6, aoi, timeout_s=timeout_s)
        if gdf_ll is not None and not gdf_ll.empty:
            if "Permanent_Identifier" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["Permanent_Identifier"].astype(str)
            elif "ReachCode" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["ReachCode"].astype(str)
            else:
                gdf_ll["river_id"] = [f"reach_{i}" for i in range(len(gdf_ll))]
            lonc, latc = _aoi_center(aoi)
            crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))
            gdfp = gdf_ll.to_crs(crs_out)
            gdfp = _clean_lines(gdfp)
            gdfp["length_m"] = gdfp.geometry.length.astype("float64")
            gdfp["source"] = "arcgis_nhd"
            return gdfp
    except Exception as e:
        log.warning("[ARCGIS] NHD query failed: %s", str(e))

    return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")



# --------------------------------------------------------------------------------------
# Optional: ArcGIS REST NHDArea polygons (river channel polygons)
# --------------------------------------------------------------------------------------

_ARCGIS_LAYER_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def arcgis_list_layers(service_url: str, timeout_s: int = 60) -> List[Dict[str, Any]]:
    """Return ArcGIS MapServer layer metadata (id+name) for a service URL."""
    if service_url in _ARCGIS_LAYER_CACHE:
        return _ARCGIS_LAYER_CACHE[service_url]
    try:
        r = requests.get(service_url, params={"f": "pjson"}, timeout=timeout_s)
        r.raise_for_status()
        j = r.json()
        layers = j.get("layers", []) or []
        # keep minimal, stable fields
        out = [{"id": int(L.get("id")), "name": str(L.get("name", ""))} for L in layers if "id" in L]
        _ARCGIS_LAYER_CACHE[service_url] = out
        return out
    except Exception:
        return []

def arcgis_find_layer_id(service_url: str, name_patterns: List[str], timeout_s: int = 60) -> Optional[int]:
    """Find a layer id whose name matches any of the provided substrings (case-insensitive)."""
    layers = arcgis_list_layers(service_url, timeout_s=timeout_s)
    if not layers:
        return None
    for pat in name_patterns:
        pl = pat.lower()
        for L in layers:
            if pl in (L.get("name", "").lower()):
                return int(L["id"])
    return None

def _clean_polys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        return gdf
    if (~gdf.is_valid).any():
        log.info("[CLEAN] Fixing invalid polygon geometries with buffer(0) where possible.")
        fixed = gdf.geometry.buffer(0)
        ok = fixed.geom_type.isin(["Polygon", "MultiPolygon"])
        gdf.loc[ok, "geometry"] = fixed[ok].values
        gdf = gdf[gdf.is_valid]
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gdf

def clip_polys_to_aoi(gdf_polys: gpd.GeoDataFrame, aoi_poly_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    poly = aoi_poly_proj.iloc[0].geometry
    gdf = gdf_polys.copy()
    gdf["geometry"] = gdf.geometry.intersection(poly)
    gdf = _clean_polys(gdf)
    return gdf

def try_arcgis_nhdarea_polygons(
    aoi: Tuple[float, float, float, float],
    out_crs: Optional[str],
    timeout_s: int = 120,
) -> gpd.GeoDataFrame:
    """Fetch NHDArea polygons via ArcGIS REST (best-effort).

    This is used downstream to constrain river bathymetry predictions to polygonal channel areas
    when those features exist (e.g., wide rivers, braided channels, tidal channels).

    Returns GeoDataFrame in projected CRS (out_crs or auto-UTM). Empty on failure.
    """
    lonc, latc = _aoi_center(aoi)
    crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))

    # Try NHD MapServer first (most common place for NHDArea)
    layer_id = arcgis_find_layer_id(
        NHD_MAPSERVER,
        name_patterns=["NHDArea", "NHD Area"],
        timeout_s=min(60, timeout_s),
    )
    if layer_id is not None:
        try:
            log.info("[ARCGIS] Querying NHDArea polygons (%s layer %s) ...", NHD_MAPSERVER, layer_id)
            gdf_ll = arcgis_query_layer_geojson(NHD_MAPSERVER, layer_id, aoi, timeout_s=timeout_s)
            if gdf_ll is not None and not gdf_ll.empty:
                gdfp = gdf_ll.to_crs(crs_out)
                gdfp = _clean_polys(gdfp)
                if not gdfp.empty:
                    gdfp["source"] = "arcgis_nhdarea"
                    return gdfp
        except Exception as e:
            log.warning("[ARCGIS] NHDArea query failed: %s", str(e))

    # Fallback: some services expose similar polygons under waterbody/area names.
    layer_id = arcgis_find_layer_id(
        NHD_MAPSERVER,
        name_patterns=["NHD Waterbody", "NHDWaterbody", "Waterbody"],
        timeout_s=min(60, timeout_s),
    )
    if layer_id is not None:
        try:
            log.info("[ARCGIS] Querying NHD waterbody polygons (%s layer %s) ...", NHD_MAPSERVER, layer_id)
            gdf_ll = arcgis_query_layer_geojson(NHD_MAPSERVER, layer_id, aoi, timeout_s=timeout_s)
            if gdf_ll is not None and not gdf_ll.empty:
                gdfp = gdf_ll.to_crs(crs_out)
                gdfp = _clean_polys(gdfp)
                if not gdfp.empty:
                    gdfp["source"] = "arcgis_waterbody"
                    return gdfp
        except Exception as e:
            log.warning("[ARCGIS] Waterbody polygon query failed: %s", str(e))

    return gpd.GeoDataFrame(columns=["geometry"], crs=crs_out)

# --------------------------------------------------------------------------------------
# AOI / CRS helpers
# --------------------------------------------------------------------------------------

def _parse_aoi(aoi_str: str) -> Tuple[float, float, float, float]:
    parts = aoi_str.strip().split("/")
    if len(parts) != 4:
        raise ValueError("AOI must be 'lonmin/lonmax/latmin/latmax'")
    lonmin, lonmax, latmin, latmax = map(float, parts)
    if lonmin >= lonmax or latmin >= latmax:
        raise ValueError("Invalid AOI ordering; expected lonmin<lonmax and latmin<latmax.")
    return lonmin, lonmax, latmin, latmax


def _auto_utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _aoi_hash(aoi: Tuple[float, float, float, float]) -> str:
    s = f"{aoi[0]:.6f}_{aoi[1]:.6f}_{aoi[2]:.6f}_{aoi[3]:.6f}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _aoi_center(aoi: Tuple[float, float, float, float]) -> Tuple[float, float]:
    lonmin, lonmax, latmin, latmax = aoi
    return 0.5 * (lonmin + lonmax), 0.5 * (latmin + latmax)


def _guess_hydrorivers_region(aoi: Tuple[float, float, float, float]) -> str:
    """
    Coarse continent guess for selecting a HydroRIVERS regional extract.

    Returns one of: af, as, au, eu, na, sa

    Notes:
      - This is intentionally simple and pragmatic; for edge cases (e.g., Russia/Turkey),
        you can override with --hydrorivers-region.
    """
    lonc, latc = _aoi_center(aoi)

    # Australia / Oceania
    if 110 <= lonc <= 180 and -50 <= latc <= 0:
        return "au"

    # South America
    if -92 <= lonc <= -30 and -60 <= latc <= 15:
        return "sa"

    # North America (incl. Central America + Caribbean)
    if -170 <= lonc <= -10 and 5 <= latc <= 85:
        return "na"
    if -170 <= lonc <= -10 and -10 <= latc < 5:
        # Central America / Caribbean tends to be included in NA in many products
        return "na"

    # Europe (very coarse)
    if -25 <= lonc <= 60 and 35 <= latc <= 75:
        return "eu"

    # Africa
    if -25 <= lonc <= 60 and -40 <= latc <= 35:
        return "af"

    # Asia (fallback)
    return "as"


# --------------------------------------------------------------------------------------
# TNMAccess helpers
# --------------------------------------------------------------------------------------

def tnm_list_datasets(timeout_s: int = 30) -> List[Dict[str, Any]]:
    r = requests.get(TNM_DATASETS_URL, timeout=timeout_s)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict):
        for key in ("data", "items", "datasets"):
            if key in js and isinstance(js[key], list):
                return js[key]
    if isinstance(js, list):
        return js
    return []


def tnm_pick_hydro_datasets(user_dataset_hint: Optional[str] = None) -> List[str]:
    try:
        datasets = tnm_list_datasets()
    except Exception as e:
        log.warning("[TNM] Failed to list datasets (%s). Using common guesses.", str(e))
        datasets = []

    names: List[str] = []
    for d in datasets:
        if isinstance(d, dict):
            nm = d.get("name") or d.get("title") or d.get("datasetName")
            if nm:
                names.append(str(nm))

    if user_dataset_hint:
        return [user_dataset_hint] + [n for n in names if n != user_dataset_hint]

    def rank(n: str) -> int:
        s = n.lower()
        if "nhdplus" in s and "high" in s:
            return 0
        if "nhdplus" in s:
            return 10
        if "national hydrography dataset" in s and "high" in s:
            return 20
        if "nhd" in s:
            return 50
        if "hydrography" in s:
            return 100
        return 1000

    if not names:
        return [
            "NHDPlus High Resolution",
            "National Hydrography Dataset (NHD) - High Resolution",
            "National Hydrography Dataset (NHD)",
        ]

    ranked = sorted(names, key=rank)
    out = []
    for n in ranked:
        s = n.lower()
        if ("nhd" in s) or ("hydrography" in s):
            out.append(n)
        if len(out) >= 8:
            break
    return out or ranked[:5]


def tnm_search_products(
    bbox_lonlat: Tuple[float, float, float, float],
    datasets: List[str],
    prod_formats: Optional[List[str]] = None,
    max_items: int = 200,
    timeout_s: int = 60,
) -> List[Dict[str, Any]]:
    lonmin, lonmax, latmin, latmax = bbox_lonlat
    bbox = f"{lonmin},{latmin},{lonmax},{latmax}"

    items: List[Dict[str, Any]] = []

    for ds in datasets:
        params = {"bbox": bbox, "datasets": ds}
        if prod_formats:
            params["prodFormats"] = ",".join(prod_formats)

        log.info("[TNM] Searching products datasets='%s' bbox=%s", ds, bbox)
        try:
            r = requests.get(TNM_PRODUCTS_URL, params=params, timeout=timeout_s)
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            log.warning("[TNM] Search failed for dataset '%s': %s", ds, str(e))
            continue

        ds_items: List[Dict[str, Any]] = []
        if isinstance(js, dict):
            for key in ("items", "data", "products", "results"):
                if key in js and isinstance(js[key], list):
                    ds_items = js[key]
                    break
        elif isinstance(js, list):
            ds_items = js

        if not ds_items:
            tot = js.get("total", 0) if isinstance(js, dict) else 0
            log.info("[TNM] No items for '%s' (total=%s).", ds, str(tot))
            continue

        items.extend([x for x in ds_items if isinstance(x, dict)])
        log.info("[TNM] Found %d items for '%s'.", len(ds_items), ds)

        if len(items) >= max_items:
            items = items[:max_items]
            break

    # De-dup by download URL (or id)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in items:
        url = it.get("downloadURL") or it.get("downloadUrl") or it.get("url")
        key = url or it.get("id") or json.dumps(it, sort_keys=True)[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq[:max_items]


def _safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name[:180] if name else "product"


def tnm_download_products(items: List[Dict[str, Any]], out_dir: Path, timeout_s: int = 120) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []

    for i, it in enumerate(items, start=1):
        url = it.get("downloadURL") or it.get("downloadUrl") or it.get("url")
        if not url:
            continue

        title = it.get("title") or it.get("name") or it.get("productName") or f"product_{i:03d}"
        fn = _safe_filename(title)

        ext = Path(url.split("?")[0]).suffix
        if not ext:
            ext = ".zip"
        dst = out_dir / f"{fn}{ext}"

        if dst.exists() and dst.stat().st_size > 0:
            log.info("[TNM] Exists, skipping: %s", dst.name)
            downloaded.append(dst)
            continue

        log.info("[TNM] Downloading %d/%d: %s", i, len(items), url)
        try:
            with requests.get(url, stream=True, timeout=timeout_s) as r:
                r.raise_for_status()
                tmp = dst.with_suffix(dst.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dst)
            downloaded.append(dst)
        except Exception as e:
            log.warning("[TNM] Download failed: %s (%s)", url, str(e))
            try:
                part = dst.with_suffix(dst.suffix + ".part")
                if part.exists():
                    part.unlink()
            except Exception:
                pass

    return downloaded


def extract_archives(downloads: List[Path], work_dir: Path) -> List[Path]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    roots: List[Path] = []

    for p in downloads:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix.lower() == ".zip":
            sub = work_dir / p.stem
            if sub.exists() and any(sub.iterdir()):
                roots.append(sub)
                continue
            sub.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(p, "r") as z:
                    z.extractall(sub)
                roots.append(sub)
            except Exception as e:
                log.warning("[TNM] Failed to extract %s: %s", p.name, str(e))
        else:
            roots.append(p)

    return roots


def find_best_flowline_source(extract_roots: List[Path]) -> Tuple[Optional[Path], Optional[str]]:
    candidates: List[Path] = []
    for root in extract_roots:
        root = Path(root)
        if root.is_file():
            candidates.append(root)
            continue
        candidates.extend(root.rglob("*.gpkg"))
        candidates.extend([p for p in root.rglob("*.gdb") if p.is_dir()])
        candidates.extend(root.rglob("*.shp"))

    if not candidates:
        return None, None

    layer_pref = ["NHDFlowline", "NHDFlowline_Network", "Flowline", "Flowlines"]

    def score(path: Path, layer: Optional[str]) -> int:
        p = str(path).lower()
        s = 1000
        if "nhd" in p:
            s -= 200
        if "flowline" in p:
            s -= 200
        if path.suffix.lower() == ".gpkg":
            s -= 50
        if path.suffix.lower() == ".gdb":
            s -= 25
        if path.suffix.lower() == ".shp":
            s -= 10
        if layer:
            l = layer.lower()
            if l == "nhdflowline":
                s -= 500
            elif "flowline" in l:
                s -= 200
        return s

    best_path = None
    best_layer = None
    best_score = 10**9

    for c in candidates:
        try:
            if c.is_dir() and c.suffix.lower() == ".gdb":
                layers = fiona.listlayers(str(c))
                chosen = None
                for lp in layer_pref:
                    if lp in layers:
                        chosen = lp
                        break
                if chosen is None:
                    for l in layers:
                        if "flowline" in l.lower():
                            chosen = l
                            break
                if chosen is None:
                    continue
                sc = score(c, chosen)
                if sc < best_score:
                    best_score, best_path, best_layer = sc, c, chosen
            else:
                layers = []
                try:
                    layers = fiona.listlayers(str(c))
                except Exception:
                    layers = []
                chosen = None
                if layers:
                    for lp in layer_pref:
                        if lp in layers:
                            chosen = lp
                            break
                    if chosen is None:
                        for l in layers:
                            if "flowline" in l.lower():
                                chosen = l
                                break
                sc = score(c, chosen)
                if sc < best_score:
                    best_score, best_path, best_layer = sc, c, chosen
        except Exception:
            continue

    return best_path, best_layer


# --------------------------------------------------------------------------------------
# HydroRIVERS auto-download
# --------------------------------------------------------------------------------------

def download_hydrorivers_zip(region: str, out_dir: Path, timeout_s: int = 600) -> Path:
    """
    Download the HydroRIVERS regional shapefile zip to out_dir and return the zip path.
    """
    region = region.lower().strip()
    if region not in HYDRORIVERS_REGIONS:
        raise ValueError(f"Invalid HydroRIVERS region '{region}'. Must be one of {sorted(HYDRORIVERS_REGIONS)}")

    url = HYDRORIVERS_ZIP_TEMPLATE.format(region=region)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"HydroRIVERS_v10_{region}_shp.zip"

    if dst.exists() and dst.stat().st_size > 0:
        log.info("[HYRIV] Exists, skipping download: %s", dst.name)
        return dst

    log.info("[HYRIV] Downloading HydroRIVERS region='%s' → %s", region, dst)
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dst)
    except Exception as e:
        log.error("[HYRIV] Download failed (%s): %s", url, str(e))
        try:
            part = dst.with_suffix(dst.suffix + ".part")
            if part.exists():
                part.unlink()
        except Exception:
            pass
        raise

    return dst


def extract_hydrorivers(zip_path: Path, work_dir: Path) -> Path:
    """
    Extract HydroRIVERS zip to work_dir/<zip_stem>/ and return extracted root.
    """
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sub = work_dir / zip_path.stem
    if sub.exists() and any(sub.iterdir()):
        return sub
    sub.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(sub)
    return sub


def find_hydrorivers_shp(extracted_root: Path) -> Path:
    """
    Find the primary HydroRIVERS shapefile under extracted_root.
    """
    extracted_root = Path(extracted_root)
    shps = list(extracted_root.rglob("HydroRIVERS_v10_*.shp"))
    if not shps:
        # Fallback: any .shp with hydrorivers in name
        shps = [p for p in extracted_root.rglob("*.shp") if "hydrorivers" in p.name.lower()]
    if not shps:
        raise RuntimeError(f"No HydroRIVERS shapefile found under {extracted_root}")
    # Prefer shortest path (top-level) and largest size (often the main file)
    shps = sorted(shps, key=lambda p: (len(str(p)), -p.stat().st_size))
    return shps[0]


# --------------------------------------------------------------------------------------
# Ingest / clip / graph
# --------------------------------------------------------------------------------------

def _clean_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if (~gdf.is_valid).any():
        log.info("[CLEAN] Fixing invalid geometries with buffer(0) where possible.")
        fixed = gdf.geometry.buffer(0)
        ok = fixed.geom_type.isin(["LineString", "MultiLineString"])
        gdf.loc[ok, "geometry"] = fixed[ok].values
        gdf = gdf[gdf.is_valid]
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    return gdf


def ingest_flowlines(
    flowlines_path: Path,
    layer: Optional[str],
    aoi: Tuple[float, float, float, float],
    out_crs: Optional[str] = None,
    id_field: str = "COMID",
    keep_fields: Optional[List[str]] = None,
    simplify_m: float = 0.0,
) -> gpd.GeoDataFrame:
    flowlines_path = Path(flowlines_path)
    if not flowlines_path.exists():
        raise FileNotFoundError(flowlines_path)

    log.info("[READ] %s (layer=%s)", flowlines_path, layer or "<default>")
    gdf = gpd.read_file(flowlines_path, layer=layer) if layer else gpd.read_file(flowlines_path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError("Input flowlines have no CRS. Assign CRS before running.")

    lonmin, lonmax, latmin, latmax = aoi
    bbox_ll = box(lonmin, latmin, lonmax, latmax)
    gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs.to_string() != "EPSG:4326" else gdf
    gdf_ll = gdf_ll[gdf_ll.geometry.intersects(bbox_ll)]
    if gdf_ll.empty:
        return gdf_ll

    if out_crs:
        crs_out = CRS.from_user_input(out_crs)
    else:
        lon_c = 0.5 * (lonmin + lonmax)
        lat_c = 0.5 * (latmin + latmax)
        crs_out = CRS.from_epsg(_auto_utm_epsg_from_lonlat(lon_c, lat_c))

    gdfp = gdf_ll.to_crs(crs_out)
    gdfp = _clean_lines(gdfp)

    if simplify_m and simplify_m > 0:
        gdfp["geometry"] = gdfp.geometry.simplify(float(simplify_m), preserve_topology=True)
        gdfp = _clean_lines(gdfp)

    if keep_fields:
        keep = [c for c in keep_fields if c in gdfp.columns]
        if id_field in gdfp.columns and id_field not in keep:
            keep.append(id_field)
        gdfp = gdfp[keep + ["geometry"]].copy()

    if id_field in gdfp.columns:
        gdfp["river_id"] = gdfp[id_field].astype(str)
    elif "Permanent_Identifier" in gdfp.columns:
        gdfp["river_id"] = gdfp["Permanent_Identifier"].astype(str)
    elif "HYRIV_ID" in gdfp.columns:
        gdfp["river_id"] = gdfp["HYRIV_ID"].astype(str)
    else:
        gdfp["river_id"] = [f"reach_{i}" for i in range(len(gdfp))]

    gdfp["length_m"] = gdfp.geometry.length.astype("float64")
    return gdfp


def build_aoi_polygon_projected(aoi: Tuple[float, float, float, float], out_crs: CRS) -> gpd.GeoDataFrame:
    lonmin, lonmax, latmin, latmax = aoi
    poly_ll = box(lonmin, latmin, lonmax, latmax)
    return gpd.GeoDataFrame([{"geometry": poly_ll}], crs="EPSG:4326").to_crs(out_crs)


def clip_lines_to_aoi(gdf_lines: gpd.GeoDataFrame, aoi_poly_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    poly = aoi_poly_proj.iloc[0].geometry
    gdf = gdf_lines.copy()
    gdf["geometry"] = gdf.geometry.intersection(poly)
    gdf = _clean_lines(gdf)
    gdf["length_m"] = gdf.geometry.length.astype("float64")
    return gdf


def _snap_key(x: float, y: float, snap_m: float) -> Tuple[int, int]:
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def build_reach_graph(gdf_lines: gpd.GeoDataFrame, snap_m: float = 5.0) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if gdf_lines.empty:
        return (
            gpd.GeoDataFrame(columns=["node_id", "degree", "geometry"], crs=gdf_lines.crs),
            gpd.GeoDataFrame(columns=["edge_id", "from_node", "to_node", "length_m", "geometry"], crs=gdf_lines.crs),
        )

    node_map: Dict[Tuple[int, int], int] = {}
    node_xy: Dict[int, Tuple[float, float]] = {}
    node_degree: Dict[int, int] = {}

    def get_node_id(pt: Point) -> int:
        key = _snap_key(pt.x, pt.y, snap_m)
        if key in node_map:
            nid = node_map[key]
        else:
            nid = len(node_map) + 1
            node_map[key] = nid
            node_xy[nid] = (pt.x, pt.y)
            node_degree[nid] = 0
        return nid

    edges = []
    for idx, row in gdf_lines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            if part.length == 0:
                continue
            p0 = Point(part.coords[0])
            p1 = Point(part.coords[-1])
            n0 = get_node_id(p0)
            n1 = get_node_id(p1)
            node_degree[n0] += 1
            node_degree[n1] += 1
            edges.append(
                {
                    "edge_id": f"e_{len(edges)+1:08d}",
                    "river_id": row.get("river_id", str(idx)),
                    "from_node": n0,
                    "to_node": n1,
                    "length_m": float(part.length),
                    "geometry": part,
                }
            )

    nodes = [{"node_id": nid, "degree": int(node_degree[nid]), "geometry": Point(node_xy[nid])} for nid in sorted(node_xy)]
    return gpd.GeoDataFrame(nodes, crs=gdf_lines.crs), gpd.GeoDataFrame(edges, crs=gdf_lines.crs)


def station_edges_from_root(nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if nodes_gdf.empty or edges_gdf.empty:
        return edges_gdf

    adj: Dict[int, List[Tuple[int, float]]] = {}
    for _, e in edges_gdf.iterrows():
        a = int(e["from_node"])
        b = int(e["to_node"])
        w = float(e["length_m"])
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))

    def dijkstra(root: int) -> Dict[int, float]:
        import heapq
        dist = {root: 0.0}
        pq = [(0.0, root)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, None):
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    node_ids = nodes_gdf["node_id"].astype(int).tolist()
    seen = set()
    comps: List[List[int]] = []
    neighbors = {nid: [v for v, _ in adj.get(nid, [])] for nid in node_ids}

    for nid in node_ids:
        if nid in seen:
            continue
        stack = [nid]
        comp = []
        seen.add(nid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)

    node_y = {int(r["node_id"]): float(r.geometry.y) for _, r in nodes_gdf.iterrows()}
    node_deg = {int(r["node_id"]): int(r.get("degree", 0)) for _, r in nodes_gdf.iterrows()}

    node_station: Dict[int, float] = {}
    node_component: Dict[int, int] = {}
    comp_root: Dict[int, int] = {}

    for ci, comp in enumerate(comps, start=1):
        deg1 = [n for n in comp if node_deg.get(n, 0) == 1]
        root = min(deg1, key=lambda n: node_y.get(n, 0.0)) if deg1 else min(comp, key=lambda n: node_y.get(n, 0.0))
        comp_root[ci] = root
        dist = dijkstra(root)
        for n in comp:
            node_station[n] = float(dist.get(n, float("nan")))
            node_component[n] = ci

    edges = edges_gdf.copy()
    edges["component_id"] = edges.apply(lambda r: node_component.get(int(r["from_node"]), -1), axis=1)
    edges["root_node"] = edges["component_id"].map(comp_root)
    edges["s_m_from"] = edges["from_node"].map(node_station)
    edges["s_m_to"] = edges["to_node"].map(node_station)
    edges["s_m_min"] = np.minimum(edges["s_m_from"], edges["s_m_to"])
    edges["s_m_max"] = np.maximum(edges["s_m_from"], edges["s_m_to"])
    return edges


# --------------------------------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "river_network.py – download→ingest centerlines + AOI clip + reach graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--aoi", required=True, help="AOI bbox lonmin/lonmax/latmin/latmax")
    p.add_argument("--out-gpkg", required=True, help="Output GeoPackage path")

    p.add_argument("--cache-dir", default="cache/hydrography", help="Cache root for downloads/extracts.")
    p.add_argument("--out-crs", default=None, help="Output CRS (e.g., EPSG:32618). Default: auto UTM.")
    p.add_argument("--simplify-m", type=float, default=0.0, help="Optional simplify tolerance (m)")

    # Local NHD
    p.add_argument("--nhd-flowlines", default=None, help="Path to NHD/NHDPlus flowlines dataset (GPKG/GDB/SHP).")
    p.add_argument("--layer", default=None, help="Layer name if applicable.")
    p.add_argument("--id-field", default="COMID", help="ID field when available (COMID typical for NHDPlus).")
    p.add_argument("--keep-fields", default=None, help="Comma-separated list of fields to keep (optional).")

    # TNM auto-download (US) — ON by default (so the script "just works")
    p.add_argument("--tnm-enable", dest="tnm_enable", action="store_true",
                   default=True,
                   help="Enable TNMAccess auto-download (default: enabled).")
    p.add_argument("--no-tnm", dest="tnm_enable", action="store_false",
                   help="Disable TNMAccess auto-download.")
    p.add_argument("--tnm-dataset", default=None, help="Optional dataset name override for TNMAccess search.")
    p.add_argument("--tnm-formats", default="FileGDB,Shapefile,GeoPackage", help="Comma-separated preferred product formats.")
    p.add_argument("--tnm-max-items", type=int, default=50, help="Max TNM products to download.")
    p.add_argument("--tnm-timeout", type=int, default=120, help="HTTP timeout seconds for TNM search/download.")
    p.add_argument("--arcgis-timeout", type=int, default=120, help="HTTP timeout seconds for ArcGIS REST flowline fallback.")

    # Optional NHDArea polygons (constrains river bathymetry domain when available)
    p.add_argument("--no-nhdarea-polygons", dest="nhdarea_enable", action="store_false", default=True,
                   help="Disable ArcGIS REST fetch of NHDArea polygons.")
    p.add_argument("--nhdarea-layer-name", default="nhdarea_clip",
                   help="Output layer name for NHDArea polygons (default nhdarea_clip).")

    p.add_argument("--tnm-dry-run", action="store_true", help="Search but do not download.")

    # HydroRIVERS fallback (local and/or auto-download)
    p.add_argument("--hydrorivers", default=None,
                   help="Path to HydroRIVERS lines (optional; if omitted can auto-download when NHD is unavailable).")
    p.add_argument("--hydrorivers-auto-download", dest="hydrorivers_auto_download",
                   action="store_true", default=True,
                   help="Auto-download HydroRIVERS ONLY if NHD is unavailable (default: enabled).")
    p.add_argument("--no-hydrorivers-fallback", dest="hydrorivers_auto_download",
                   action="store_false",
                   help="Disable HydroRIVERS fallback (script will error if no NHD found).")
    p.add_argument("--hydrorivers-region", default=None,
                   help="Override HydroRIVERS region: af, as, au, eu, na, sa. Default: guessed from AOI.")
    p.add_argument("--hydrorivers-timeout", type=int, default=600,
                   help="HTTP timeout seconds for HydroRIVERS download.")

    # Graph output
    p.add_argument("--snap-m", type=float, default=5.0, help="Endpoint snap tolerance for node merging (m).")
    p.add_argument(
        "--write-layers",
        default="rivers_aoi,rivers_clip,graph_nodes,graph_edges,nhdarea_aoi,nhdarea_clip",
        help="Comma-separated output layers to write.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    aoi = _parse_aoi(args.aoi)

    keep_fields = None
    if args.keep_fields:
        keep_fields = [s.strip() for s in args.keep_fields.split(",") if s.strip()]

    gdf = gpd.GeoDataFrame()

    # 1) Local NHD ingest
    if args.nhd_flowlines:
        gdf = ingest_flowlines(
            flowlines_path=Path(args.nhd_flowlines),
            layer=args.layer,
            aoi=aoi,
            out_crs=args.out_crs,
            id_field=args.id_field,
            keep_fields=keep_fields,
            simplify_m=float(args.simplify_m),
        )
        if gdf is not None and not gdf.empty:
            gdf["source"] = "nhd_local"
            log.info("[OK] Loaded %d reaches from --nhd-flowlines.", len(gdf))

    # 2) TNM auto-download for NHD (US)
    if (gdf is None or gdf.empty) and args.tnm_enable and (not args.nhd_flowlines):
        cache_root = Path(args.cache_dir) / f"tnm_nhd_{_aoi_hash(aoi)}"
        dl_dir = cache_root / "downloads"
        wk_dir = cache_root / "work"

        ds_candidates = tnm_pick_hydro_datasets(args.tnm_dataset)
        formats = [s.strip() for s in (args.tnm_formats or "").split(",") if s.strip()]

        items = tnm_search_products(
            bbox_lonlat=aoi,
            datasets=ds_candidates,
            prod_formats=formats if formats else None,
            max_items=int(args.tnm_max_items),
            timeout_s=int(args.tnm_timeout),
        )
        log.info("[TNM] Total unique candidate products: %d", len(items))

        if items and not args.tnm_dry_run:
            downloads = tnm_download_products(items, dl_dir, timeout_s=int(args.tnm_timeout))
            log.info("[TNM] Downloaded %d files.", len(downloads))

            roots = extract_archives(downloads, wk_dir)
            src_path, src_layer = find_best_flowline_source(roots)

            if src_path:
                log.info("[TNM] Using flowlines source: %s (layer=%s)", src_path, src_layer or "<default>")
                gdf = ingest_flowlines(
                    flowlines_path=Path(src_path),
                    layer=src_layer,
                    aoi=aoi,
                    out_crs=args.out_crs,
                    id_field=args.id_field,
                    keep_fields=keep_fields,
                    simplify_m=float(args.simplify_m),
                )
                if gdf is not None and not gdf.empty:
                    gdf["source"] = "tnm_nhd"
                    log.info("[OK] Loaded %d reaches from TNM download.", len(gdf))
            else:
                log.warning("[TNM] Could not locate a flowline dataset/layer in downloaded products.")
        elif args.tnm_dry_run:
            log.info("[TNM] Dry-run enabled; skipping download.")
        else:
            log.warning("[TNM] No products found for AOI.")

    # 3) ArcGIS REST fallback (still NHD-derived): if TNM download yields nothing usable,
    # try querying NHDPlus_HR / NHD MapServers directly for flowlines.
    if (gdf is None or gdf.empty) and (not args.nhd_flowlines):
        gdf_arc = try_arcgis_nhd_flowlines(aoi=aoi, out_crs=args.out_crs, timeout_s=int(getattr(args, "arcgis_timeout", 120)))
        if gdf_arc is not None and not gdf_arc.empty:
            gdf = gdf_arc
            log.info("[OK] Loaded %d reaches via ArcGIS REST fallback (%s).", len(gdf), str(gdf["source"].iloc[0]) if "source" in gdf.columns else "arcgis")

# 3) HydroRIVERS fallback — local path OR auto-download
    if gdf is None or gdf.empty:
        hydrorivers_path = None

        if args.hydrorivers:
            hydrorivers_path = Path(args.hydrorivers)
            if not hydrorivers_path.exists():
                raise FileNotFoundError(hydrorivers_path)

        if hydrorivers_path is None and args.hydrorivers_auto_download:
            region = (args.hydrorivers_region or _guess_hydrorivers_region(aoi)).lower().strip()
            cache_root = Path(args.cache_dir) / f"hydrorivers_{region}_{_aoi_hash(aoi)}"
            dl_dir = cache_root / "downloads"
            wk_dir = cache_root / "work"

            zip_path = download_hydrorivers_zip(region, dl_dir, timeout_s=int(args.hydrorivers_timeout))
            extracted = extract_hydrorivers(zip_path, wk_dir)
            hydrorivers_path = find_hydrorivers_shp(extracted)
            log.info("[HYRIV] Using downloaded HydroRIVERS shapefile: %s", hydrorivers_path)

        if hydrorivers_path is None:
            raise RuntimeError(
                "No NHD flowlines found for AOI. Provide --nhd-flowlines OR enable TNM with --tnm-enable, "
                "OR enable HydroRIVERS fallback with --hydrorivers-auto-download (or provide --hydrorivers)."
            )

        log.warning("[FALLBACK] Using HydroRIVERS (%s).", hydrorivers_path)
        gdf = ingest_flowlines(
            flowlines_path=hydrorivers_path,
            layer=None,
            aoi=aoi,
            out_crs=args.out_crs,
            id_field="HYRIV_ID",
            keep_fields=None,
            simplify_m=float(args.simplify_m),
        )
        if gdf is None or gdf.empty:
            raise RuntimeError("HydroRIVERS fallback returned no reaches in AOI.")
        gdf["source"] = "hydrorivers"
        log.info("[OK] Loaded %d HydroRIVERS reaches in AOI.", len(gdf))

    # AOI polygon in projected CRS
    crs_out = CRS.from_user_input(gdf.crs)
    aoi_poly_proj = build_aoi_polygon_projected(aoi, crs_out)

    rivers_aoi = gdf.copy()
    rivers_clip = clip_lines_to_aoi(rivers_aoi, aoi_poly_proj)

    # Optional: NHDArea polygons (used downstream to constrain river bathymetry domain)
    nhdarea_aoi = gpd.GeoDataFrame(columns=["geometry"], crs=crs_out)
    nhdarea_clip = gpd.GeoDataFrame(columns=["geometry"], crs=crs_out)
    src_name = str(rivers_aoi["source"].iloc[0]) if "source" in rivers_aoi.columns and len(rivers_aoi) else ""
    if getattr(args, "nhdarea_enable", True) and (not src_name.startswith("hydrorivers")):
        try:
            nhdarea_aoi = try_arcgis_nhdarea_polygons(aoi=aoi, out_crs=str(crs_out), timeout_s=int(getattr(args, "arcgis_timeout", 120)))
            if nhdarea_aoi is not None and not nhdarea_aoi.empty:
                nhdarea_clip = clip_polys_to_aoi(nhdarea_aoi, aoi_poly_proj)
                log.info("[OK] Loaded %d NHDArea polygon(s) via ArcGIS REST (%s).", len(nhdarea_clip), str(nhdarea_aoi.get("source", ["arcgis"]).iloc[0]) if "source" in nhdarea_aoi.columns else "arcgis")
            else:
                log.info("[ARCGIS] No NHDArea polygons found in AOI (continuing).")
        except Exception as e:
            log.warning("[ARCGIS] NHDArea polygon fetch failed (continuing): %s", str(e))

    nodes_gdf, edges_gdf = build_reach_graph(rivers_clip, snap_m=float(args.snap_m))
    edges_gdf = station_edges_from_root(nodes_gdf, edges_gdf)

    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [s.strip() for s in args.write_layers.split(",") if s.strip()]

    log.info("[WRITE] %s", out_path)
    if "rivers_aoi" in layers:
        rivers_aoi.to_file(out_path, layer="rivers_aoi", driver="GPKG")
    if "rivers_clip" in layers:
        rivers_clip.to_file(out_path, layer="rivers_clip", driver="GPKG")
    if "graph_nodes" in layers:
        nodes_gdf.to_file(out_path, layer="graph_nodes", driver="GPKG")
    if "graph_edges" in layers:
        edges_gdf.to_file(out_path, layer="graph_edges", driver="GPKG")
    if "nhdarea_aoi" in layers and nhdarea_aoi is not None and not nhdarea_aoi.empty:
        nhdarea_aoi.to_file(out_path, layer="nhdarea_aoi", driver="GPKG")
    if "nhdarea_clip" in layers and nhdarea_clip is not None and not nhdarea_clip.empty:
        layer_name = getattr(args, "nhdarea_layer_name", "nhdarea_clip")
        nhdarea_clip.to_file(out_path, layer=layer_name, driver="GPKG")

    log.info(
        "[DONE] rivers_aoi=%d | rivers_clip=%d | nodes=%d | edges=%d | nhdarea=%d | CRS=%s | source=%s",
        len(rivers_aoi),
        len(rivers_clip),
        len(nodes_gdf),
        len(edges_gdf),
        len(nhdarea_clip) if 'nhdarea_clip' in locals() else 0,
        str(gdf.crs),
        str(rivers_aoi["source"].iloc[0]) if "source" in rivers_aoi.columns and len(rivers_aoi) else "unknown",
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()