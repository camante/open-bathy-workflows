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


import argparse
import hashlib
import warnings

# Silence GeoPandas GeoSeries.notna() behavior-change warning (we explicitly handle empties).
warnings.filterwarnings('ignore', message=r'GeoSeries.notna\(\) previously returned False.*', category=UserWarning)
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

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401
import geopandas as gpd
import requests
from shapely.geometry import box, Point
from pyproj import CRS
import fiona
from hydrologic_solve_domain import (
    build_hydrologic_solve_domain_artifacts,
    write_hydrologic_solve_domain_json,
)

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("river_network")


# --------------------------------------------------------------------------------------
# Provenance / determinism helpers
# --------------------------------------------------------------------------------------

def _sha256_bytes_iter(chunks_iter):
    h = hashlib.sha256()
    for ch in chunks_iter:
        if not ch:
            continue
        h.update(ch)
    return h.hexdigest()


def fingerprint_path(path: Path) -> Optional[Dict[str, Any]]:
    """Compute a deterministic fingerprint for a dataset path.

    - For regular files: SHA256 of bytes.
    - For directories (e.g., .gdb): SHA256 over a stable listing of relative paths + size + mtime_ns.

    This is meant to detect unexpected changes in cached datasets across runs/machines.
    """
    path = Path(path)
    if path.is_file():
        def _iter():
            with open(path, "rb") as f:
                while True:
                    b = f.read(1024 * 1024)
                    if not b:
                        break
                    yield b
        return {
            "path": str(path),
            "type": "file",
            "sha256": _sha256_bytes_iter(_iter()),
            "method": "sha256(file_bytes)",
        }
    if path.is_dir():
        # Stable listing hash with lightweight content sampling (avoid mtime-based drift).
        # More portable across machines than mtimes, while still detecting meaningful changes.
        def _sample_sha256(fp: Path, max_head: int = 65536, max_tail: int = 65536) -> str:
            try:
                size = fp.stat().st_size
                h = hashlib.sha256()
                with open(fp, "rb") as f:
                    head = f.read(min(max_head, size))
                    h.update(head)
                    if size > len(head):
                        tail_len = min(max_tail, max(0, size - len(head)))
                        if tail_len > 0:
                            try:
                                f.seek(max(0, size - tail_len))
                                h.update(f.read(tail_len))
                            except Exception:
                                log.debug("ignored", exc_info=True)
                h.update(str(size).encode("utf-8"))
                return h.hexdigest()
            except Exception:
                log.debug("_sample_sha256: suppressed exception", exc_info=True)
                return "stat_or_read_failed"

        entries = []
        n_files = 0
        for fp in sorted([pp for pp in path.rglob("*") if pp.is_file()], key=lambda x: str(x)):
            try:
                rel = str(fp.relative_to(path))
                size = fp.stat().st_size
                samp = _sample_sha256(fp)
                entries.append(f"{rel}	{size}	{samp}")
                n_files += 1
            except Exception:
                log.debug("_sample_sha256: suppressed exception", exc_info=True)
                try:
                    entries.append(str(fp.relative_to(path)))
                except Exception:
                    log.debug("_sample_sha256: suppressed exception", exc_info=True)
                    entries.append(str(fp))
        payload = ("\n".join(entries)).encode("utf-8", errors="replace")
        return {
            "path": str(path),
            "type": "dir",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "method": "sha256(dir_listing(relpath,size,sha256(head+tail+size)))",
            "n_files": n_files,
        }
    return None


def load_lock(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        log.warning("Failed to read provenance lock: %s", str(path), exc_info=True)
        return None


def write_lock(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def enforce_or_write_lock(lock_path: Optional[Path], provenance: Dict[str, Any]) -> None:
    """If lock exists, enforce exact match; else write it."""
    if not lock_path:
        return
    lock_path = Path(lock_path)
    existing = load_lock(lock_path)
    if existing is None:
        write_lock(lock_path, provenance)
        log.info("Wrote river network lock: %s", str(lock_path))
        return
    # Exact-match enforcement on key fields
    keys = ["source_path", "layer", "fingerprint"]
    mismatches = []
    for k in keys:
        if existing.get(k) != provenance.get(k):
            mismatches.append(k)
    if mismatches:
        msg = {
            "error": "River network provenance lock mismatch",
            "lock_path": str(lock_path),
            "mismatched_fields": mismatches,
            "expected": {k: existing.get(k) for k in keys},
            "got": {k: provenance.get(k) for k in keys},
        }
        raise RuntimeError(json.dumps(msg, indent=2))
    log.info("River network lock matched: %s", str(lock_path))




def _network_manifest_default_path(out_gpkg: Path) -> Path:
    out_gpkg = Path(out_gpkg)
    return out_gpkg.with_name(f"{out_gpkg.stem}_manifest.json")


def _parse_aoi_bounds(aoi_text: str | None) -> Dict[str, float] | None:
    if not aoi_text:
        return None
    try:
        aoi = [float(x) for x in str(aoi_text).split('/')[:4]]
        return {"west": aoi[0], "east": aoi[1], "south": aoi[2], "north": aoi[3]}
    except Exception:
        log.debug("_parse_aoi_bounds: suppressed exception", exc_info=True)
        return None


def _build_network_manifest(*, args: argparse.Namespace, out_gpkg: Path, hydrologic_domain_path: Path, hydrologic_manifest: Dict[str, Any], mainstem_solve_network, major_system_network, outlet_anchors, estuary_control_points, rivers_aoi, rivers_clip, nodes_gdf, edges_gdf, nhdarea_clip) -> Dict[str, Any]:
    solve_aoi = str(getattr(args, "solve_aoi", None) or args.aoi)
    export_aoi = str(getattr(args, "export_aoi", None) or args.aoi)
    scaffold_aoi = str(getattr(args, "scaffold_aoi", None) or solve_aoi)
    solve_domain_receipt = {
        "export_aoi": export_aoi,
        "export_aoi_bounds": _parse_aoi_bounds(export_aoi),
        "solve_aoi": solve_aoi,
        "solve_aoi_bounds": _parse_aoi_bounds(solve_aoi),
        "scaffold_aoi": scaffold_aoi,
        "scaffold_aoi_bounds": _parse_aoi_bounds(scaffold_aoi),
        "solve_role": str(getattr(args, "solve_domain_role", "broader river-network solve extent used to produce AOI-stable dominant trunk selection")),
        "export_role": str(getattr(args, "export_domain_role", "working river export clipped to the requested solve AOI grid/domain")),
        "scaffold_role": str(getattr(args, "scaffold_domain_role", "canonical scaffold domain used for stable river topology and guidance generation")),
        "solve_domain_rationale": str(getattr(args, "solve_domain_rationale", "Solve the dominant trunk on a broader halo-expanded network, then clip/rasterize to the export grid for AOI-stable results.")),
        "halo_km": float(getattr(args, "solve_halo_km", 0.0) or 0.0),
        "trusted_halo_m": float(getattr(args, "trusted_halo_m", 0.0) or 0.0),
    }
    topology_fields = [c for c in ["from_node", "to_node", "component_id", "root_node", "s_m_from", "s_m_to", "s_m_min", "s_m_max", "length_m"] if c in mainstem_solve_network.columns]
    return {
        "product_type": "river_network_manifest",
        "network_gpkg": str(Path(out_gpkg).resolve()),
        "solve_layer_name": "mainstem_solve_network",
        "major_system_layer_name": "major_system_network",
        "active_export_layer_name": "rivers_clip",
        "hydrologic_solve_domain_json": str(Path(hydrologic_domain_path).resolve()),
        "solve_domain_receipt": solve_domain_receipt,
        "hydrologic_solve_domain": hydrologic_manifest,
        "mainstem_solve_contract": {
            "method_intent": "solve dominant trunk on explicit broader network layer, then clip/rasterize to export grid",
            "layer_priority_if_auto": ["major_system_network", "mainstem_solve_network", "rivers_aoi", "rivers_clip"],
            "mainstem_solve_network_written": True,
            "major_system_network_written": True,
            "mainstem_solve_network_geometry_basis": "graph edges built on the broader solve domain so dominant trunk solving uses explicit directed topology when available",
            "mainstem_solve_network_topology_fields": topology_fields,
            "outlet_anchor_layer_name": "outlet_anchors",
            "estuary_control_layer_name": "estuary_control_points",
        },
        "layers": {
            "mainstem_solve_network": {"count": int(len(mainstem_solve_network)), "crs": str(mainstem_solve_network.crs)},
            "major_system_network": {"count": int(len(major_system_network)), "crs": str(major_system_network.crs)},
            "outlet_anchors": {"count": int(len(outlet_anchors)), "crs": str(outlet_anchors.crs)},
            "estuary_control_points": {"count": int(len(estuary_control_points)), "crs": str(estuary_control_points.crs)},
            "rivers_aoi": {"count": int(len(rivers_aoi)), "crs": str(rivers_aoi.crs)},
            "rivers_clip": {"count": int(len(rivers_clip)), "crs": str(rivers_clip.crs)},
            "graph_nodes": {"count": int(len(nodes_gdf)), "crs": str(nodes_gdf.crs)},
            "graph_edges": {"count": int(len(edges_gdf)), "crs": str(edges_gdf.crs)},
            "nhdarea_clip": {"count": int(len(nhdarea_clip) if nhdarea_clip is not None else 0), "crs": str(nhdarea_clip.crs) if nhdarea_clip is not None else None},
        },
        "hydrography_source": str(args.hydrography_source),
        "tnm_dataset": str(args.tnm_dataset),
        "tnm_enable": bool(args.tnm_enable),
        "snap_m": float(args.snap_m),
    }


def _write_network_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
        # Avoid pyogrio runtime warning on in-memory GeoJSON strings (driver/open option mismatch)
        try:
            gdf = gpd.GeoDataFrame.from_features(js.get("features", []), crs="EPSG:4326")
        except Exception:
            log.debug("river_network: suppressed exception", exc_info=True)
            gdf = gpd.read_file(json.dumps(js))
        if gdf is None or gdf.empty:
            break

        all_frames.append(gdf)
        if len(gdf) < max_records:
            break
        offset += max_records

        # Safety stop: avoid runaway downloads on huge AOIs
        if offset > 200000:
            log.warning("Stopping pagination after %d features (safety cap).", offset)
            break

    if not all_frames:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    out = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True), crs="EPSG:4326")
    return out


def try_arcgis_nhd_flowlines(
    aoi: Tuple[float, float, float, float],
    out_crs: Optional[str],
    timeout_s: int = 120,
    *,
    da_raster: Optional[str] = None,
    da_raster_band: int = 1,
    da_raster_units: str = "km2",
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
        log.info("Querying NHDPlus_HR NetworkNHDFlowline (layer 3) ...")
        # Do not rely on ArcGIS defaults for outFields. Request the specific
        # attributes we need for deterministic physics priors (e.g., drainage area).
        # Field names are case-insensitive on the server side, but GeoJSON property
        # keys may arrive with varying case across drivers. We normalize to lowercase.
        out_fields = ",".join([
            "COMID",
            "NHDPlusID",
            "FTYPE",
            "StreamOrde",
            "TotDASqKm",
            "TotDASqMI",
            # Slope fields — NHDPlus HR: Slope in cm/km (divide by 1e5 → m/m).
            # MinElevSmo / MaxElevSmo in cm; LengthKm for independent cross-check.
            "Slope",
            "LengthKm",
            "MinElevSmo",
            "MaxElevSmo",
        ])
        try:
            gdf_ll = arcgis_query_layer_geojson(NHDPLUS_HR_MAPSERVER, 3, aoi, timeout_s=timeout_s, out_fields=out_fields)
        except Exception as e_fields:
            # Some ArcGIS instances are strict about outFields; retry with '*' to avoid hard failure.
            log.warning("NHDPlus_HR query with explicit fields failed (%s); retrying with outFields='*'", str(e_fields))
            gdf_ll = arcgis_query_layer_geojson(NHDPLUS_HR_MAPSERVER, 3, aoi, timeout_s=timeout_s, out_fields="*")
        if gdf_ll is not None and not gdf_ll.empty:
            # Normalize columns to lowercase for deterministic downstream access.
            gdf_ll = gdf_ll.rename(columns={c: c.lower() for c in gdf_ll.columns if c != "geometry"})

            # Normalize id
            if "comid" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["comid"].astype(str)
            elif "nhdplusid" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["nhdplusid"].astype(str)
            else:
                gdf_ll["river_id"] = [f"reach_{i}" for i in range(len(gdf_ll))]

            # Deterministic DA availability report (km^2)
            da_field = "totdasqkm"
            if da_field in gdf_ll.columns:
                da = pd.to_numeric(gdf_ll[da_field], errors="coerce")
                n_valid = int(np.isfinite(da).sum())
                if n_valid > 0:
                    log.info("Drainage area available: field=%s valid=%d/%d", da_field, n_valid, int(len(da)))
                else:
                    log.warning("Drainage area field present but all null: field=%s", da_field)
            else:
                log.warning("Drainage area field not returned by service (expected %s)", da_field)
            # Standardized schema for downstream XS code
            # - drain_area_km2: used for DA/Q priors (energy solver)
            # - slope_mpm: placeholder; may be filled later from WSE fit or proxies
            if "drain_area_km2" not in gdf_ll.columns:
                if da_field in gdf_ll.columns:
                    gdf_ll["drain_area_km2"] = pd.to_numeric(gdf_ll[da_field], errors="coerce")
                else:
                    gdf_ll["drain_area_km2"] = np.nan
            # ── Slope normalisation ────────────────────────────────────────────
            # NHDPlus HR NetworkNHDFlowline carries Slope in cm/km (confirmed in
            # NHDPlus HR User Guide v2.1 Table 21).  Convert to m/m.
            # We also attempt an independent cross-check from smoothed elevation
            # endpoints (MaxElevSmo − MinElevSmo in cm, LengthKm in km).
            slope_mpm_ok = False
            if "slope" in gdf_ll.columns:
                sl = pd.to_numeric(gdf_ll["slope"], errors="coerce")
                sl_mpm = sl * 1e-5           # cm/km → m/m
                valid = np.isfinite(sl_mpm) & (sl_mpm > 0)
                if valid.any():
                    gdf_ll["slope_mpm"] = np.where(valid, sl_mpm, np.nan)
                    n_valid = int(valid.sum())
                    log.info(
                        "[ARCGIS][ATTR] Slope available from NHDPlus HR (Slope field): "
                        "n_valid=%d/%d  range=[%.2e, %.2e] m/m",
                        n_valid, len(gdf_ll),
                        float(sl_mpm[valid].min()), float(sl_mpm[valid].max()),
                    )
                    slope_mpm_ok = True
            if not slope_mpm_ok:
                # Derive from smoothed endpoint elevations when the Slope field is absent or all-null.
                if "minelevsmo" in gdf_ll.columns and "maxelevsmo" in gdf_ll.columns and "lengthkm" in gdf_ll.columns:
                    dz_m  = (pd.to_numeric(gdf_ll["maxelevsmo"], errors="coerce") -
                              pd.to_numeric(gdf_ll["minelevsmo"], errors="coerce")) * 0.01  # cm → m
                    len_m = pd.to_numeric(gdf_ll["lengthkm"], errors="coerce") * 1e3
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sl_derived = np.abs(dz_m / len_m)
                    valid = np.isfinite(sl_derived) & (sl_derived > 0)
                    if valid.any():
                        gdf_ll["slope_mpm"] = np.where(valid, sl_derived, np.nan)
                        log.info(
                            "[ARCGIS][ATTR] Slope derived from MinElevSmo/MaxElevSmo: "
                            "n_valid=%d/%d",
                            int(valid.sum()), len(gdf_ll),
                        )
                        slope_mpm_ok = True
            if not slope_mpm_ok:
                if "slope_mpm" not in gdf_ll.columns:
                    gdf_ll["slope_mpm"] = np.nan
                log.warning(
                    "[ARCGIS][ATTR] No slope information available from NHDPlus HR "
                    "(fields: %s). Will fall back to WSE-profile slope proxy.",
                    list(gdf_ll.columns),
                )

            # If drainage area is still missing, optionally sample a drainage-area raster (e.g., MERIT Hydro UPA).
            gdf_ll = _attach_drainage_area_from_raster(
                gdf_ll,
                da_raster=str(da_raster or ""),
                da_raster_band=int(da_raster_band or 1),
                da_raster_units=str(da_raster_units or "km2"),
                logger=log,
            )

            # Project
            lonc, latc = _aoi_center(aoi)
            crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))
            gdfp = gdf_ll.to_crs(crs_out)
            gdfp = _clean_lines(gdfp)
            gdfp["length_m"] = gdfp.geometry.length.astype("float64")
            gdfp["source"] = "arcgis_nhdplus_hr"
            return gdfp
    except Exception as e:
        log.warning("NHDPlus_HR query failed: %s", str(e))

    # 2) NHD Flowline Large Scale
    try:
        log.info("Querying NHD Flowline - Large Scale (layer 6) ...")
        gdf_ll = arcgis_query_layer_geojson(NHD_MAPSERVER, 6, aoi, timeout_s=timeout_s)
        if gdf_ll is not None and not gdf_ll.empty:
            if "Permanent_Identifier" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["Permanent_Identifier"].astype(str)
            elif "ReachCode" in gdf_ll.columns:
                gdf_ll["river_id"] = gdf_ll["ReachCode"].astype(str)
            else:
                gdf_ll["river_id"] = [f"reach_{i}" for i in range(len(gdf_ll))]
            # Standardized schema for downstream XS code
            if "drain_area_km2" not in gdf_ll.columns:
                gdf_ll["drain_area_km2"] = np.nan
            if "slope_mpm" not in gdf_ll.columns:
                gdf_ll["slope_mpm"] = np.nan

            # Optional drainage-area raster fallback for classic NHD flowlines.
            gdf_ll = _attach_drainage_area_from_raster(
                gdf_ll,
                da_raster=str(da_raster or ""),
                da_raster_band=int(da_raster_band or 1),
                da_raster_units=str(da_raster_units or "km2"),
                logger=log,
            )


            lonc, latc = _aoi_center(aoi)
            crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))
            gdfp = gdf_ll.to_crs(crs_out)
            gdfp = _clean_lines(gdfp)
            gdfp["length_m"] = gdfp.geometry.length.astype("float64")
            gdfp["source"] = "arcgis_nhd"
            return gdfp
    except Exception as e:
        log.warning("NHD query failed: %s", str(e))

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
        log.debug("arcgis_list_layers: suppressed exception", exc_info=True)
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
        log.warning("Fixing invalid polygon geometries with buffer(0); input data has geometry errors.")
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

def filter_nhdarea_by_ftype(gdf_in: gpd.GeoDataFrame, *, allow_ftypes: tuple = (460,)) -> gpd.GeoDataFrame:
    """Filter NHDArea polygons to specific FType values.

    Default keeps only StreamRiver (FType=460) for river domain construction.
    Sea/Ocean (FType=445 / FCode=44500) and other coastal types are excluded by default.
    """
    if gdf_in is None or gdf_in.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=gdf_in.crs if gdf_in is not None else None)

    gdf_in = gdf_in.copy()

    # Find a plausible feature-type field
    ftype_field = None
    for c in ["FType", "FTYPE", "ftype", "FTypeName", "FTYPENAME", "ftypename", "FeatureType", "FEATURETYPE"]:
        if c in gdf_in.columns:
            ftype_field = c
            break

    if ftype_field is None:
        # No reliable classification field; treat as unusable (forces corridor fallback)
        return gpd.GeoDataFrame(columns=["geometry"], crs=gdf_in.crs)

    vals = gdf_in[ftype_field]
    allow_set = set(int(x) for x in allow_ftypes)

    # Numeric coding (common: StreamRiver=460, Sea/Ocean=445/FCode 44500)
    keep = None
    try:
        vnum = vals.astype("float64")
        keep = vnum.isin(allow_set)
    except Exception:
        log.debug("filter_nhdarea_by_ftype: suppressed exception", exc_info=True)
        # String coding fallback
        vstr = vals.astype(str).str.lower()
        keep = vstr.apply(lambda _: False)
        if 460 in allow_set:
            keep = keep | vstr.str.contains("streamriver") | vstr.str.contains("stream river")
        if 445 in allow_set:
            keep = keep | vstr.str.contains("sea") | vstr.str.contains("ocean")

    gdf_out = gdf_in.loc[keep].copy()
    gdf_out = _clean_polys(gdf_out)
    return gdf_out


def try_arcgis_nhdarea_polygons(
    aoi: Tuple[float, float, float, float],
    out_crs: Optional[str],
    timeout_s: int = 120,
) -> gpd.GeoDataFrame:
    """Fetch ALL NHDArea polygons via ArcGIS REST (best-effort).

    Returns ALL water-type polygons (StreamRiver=460, Sea/Ocean=445/FCode 44500, etc.)
    from the NHD MapServer Area layers. Downstream consumers apply their own
    FType filter:

    - ``nhdarea_clip`` in the scaffold is filtered to StreamRiver (460) for
      river domain construction via :func:`filter_nhdarea_by_ftype`.
    - ``nhdarea_aoi`` preserves ALL types so that guidance_domains can find
      Sea/Ocean (FType=445 / FCode=44500) for the shared estuary/coastal handoff.

    Returns GeoDataFrame in projected CRS (out_crs or auto-UTM). Empty on failure.
    """
    lonc, latc = _aoi_center(aoi)
    crs_out = CRS.from_user_input(out_crs) if out_crs else CRS.from_epsg(_auto_utm_epsg_from_lonlat(lonc, latc))

    # Prefer the NHD MapServer "Area" layers (these are polygonal channel areas).
    # Layer IDs observed in the TNM NHD service:
    #   9 = Area - Large Scale
    #   7 = Area - Small Scale
    for layer_id, label in [(9, "area_large"), (7, "area_small")]:
        try:
            log.info("Querying NHD Area polygons (%s layer %d) ...", NHD_MAPSERVER, layer_id)
            gdf_ll = arcgis_query_layer_geojson(NHD_MAPSERVER, layer_id, aoi, timeout_s=timeout_s)
            if gdf_ll is None or gdf_ll.empty:
                continue
            gdfp = gdf_ll.to_crs(crs_out)
            gdfp = _clean_polys(gdfp)
            if gdfp is not None and not gdfp.empty:
                gdfp["source"] = f"arcgis_{label}"
                log.info("Fetched %d NHDArea polygon(s) from layer %d (all FTypes).", len(gdfp), layer_id)
                return gdfp
        except Exception as e:
            log.warning("Area polygon query failed (layer=%d): %s", layer_id, str(e))

    # If we couldn't get any NHDArea polygons, return empty to trigger downstream corridor fallback.
    return gpd.GeoDataFrame(columns=["geometry"], crs=crs_out)


def _parse_aoi(aoi_str: str) -> Tuple[float, float, float, float]:
    """Parse ``"W/E/S/N"`` → ``(W, E, S, N)``.  Delegates to :func:`pipeline.aoi.parse_aoi_wesn`."""
    from pipeline.aoi import parse_aoi_wesn
    return parse_aoi_wesn(aoi_str, strict=True)


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
        log.warning("Failed to list datasets (%s). Using common guesses.", str(e))
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

        log.info("Searching products datasets='%s' bbox=%s", ds, bbox)
        try:
            r = requests.get(TNM_PRODUCTS_URL, params=params, timeout=timeout_s)
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            log.warning("Search failed for dataset '%s': %s", ds, str(e))
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
            log.info("No items for '%s' (total=%s).", ds, str(tot))
            continue

        items.extend([x for x in ds_items if isinstance(x, dict)])
        log.info("Found %d items for '%s'.", len(ds_items), ds)

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
            log.info("Exists, skipping: %s", dst.name)
            downloaded.append(dst)
            continue

        log.info("Downloading %d/%d: %s", i, len(items), url)
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
            log.warning("Download failed: %s (%s)", url, str(e))
            try:
                part = dst.with_suffix(dst.suffix + ".part")
                if part.exists():
                    part.unlink()
            except Exception:
                log.debug("ignored", exc_info=True)

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
                log.warning("Failed to extract %s: %s", p.name, str(e))
        else:
            roots.append(p)

    return roots


def find_best_flowline_source(extract_roots: List[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """Return the single best flowline dataset from extracted roots.

    Determinism / no-guess policy
    -----------------------------
    This function is only used for *auto-downloaded* archives (TNM/HydroRIVERS). For strict
    reproducibility, users should prefer explicit inputs via --nhd-flowlines/--layer.

    We keep a deterministic heuristic for convenience, but we refuse to guess when the
    choice is ambiguous. Ambiguity is defined as a tie for best score among candidates.

    Returns (path, layer) where layer may be None for single-layer sources.
    """
    candidates: List[Path] = []
    for root in extract_roots:
        root = Path(root)
        if root.is_file():
            candidates.append(root)
            continue
        # rglob order is filesystem-dependent; we sort later for determinism.
        candidates.extend(root.rglob("*.gpkg"))
        candidates.extend([p for p in root.rglob("*.gdb") if p.is_dir()])
        candidates.extend(root.rglob("*.shp"))

    # De-dup + deterministic order
    candidates = sorted({Path(c) for c in candidates}, key=lambda p: str(p))

    if not candidates:
        return None, None

    layer_pref = ["NHDFlowline", "NHDFlowline_Network", "NetworkNHDFlowline", "Flowline", "Flowlines"]

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

    scored: List[Tuple[int, str, Path, Optional[str]]] = []

    for c in candidates:
        try:
            chosen: Optional[str] = None
            if c.is_dir() and c.suffix.lower() == ".gdb":
                layers = fiona.listlayers(str(c))
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
            else:
                layers: List[str] = []
                try:
                    layers = fiona.listlayers(str(c))
                except Exception:
                    log.debug("score: suppressed exception", exc_info=True)
                    layers = []
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
            scored.append((sc, str(c), c, chosen))
        except Exception:
            log.debug("Flowline candidate inspect failed: %s", str(c), exc_info=True)
            continue

    if not scored:
        return None, None

    scored.sort(key=lambda t: (t[0], t[1], t[3] or ""))  # deterministic

    best = scored[0]
    # Refuse ambiguous best-choice ties (no silent guessing)
    if len(scored) > 1 and scored[1][0] == best[0]:
        # Show a small list to help the user decide
        top = scored[: min(10, len(scored))]
        msg_lines = [
            "Ambiguous flowline source selection from auto-downloaded archives:",
            f"  best_score={best[0]} has multiple ties.",
            "  Provide explicit --nhd-flowlines/--layer to avoid guessing.",
            "  Top candidates:",
        ]
        for sc, _, pth, lyr in top:
            msg_lines.append(f"    score={sc:4d} path={pth} layer={lyr}")
        raise RuntimeError("\n".join(msg_lines))

    return best[2], best[3]
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
        log.info("Exists, skipping download: %s", dst.name)
        return dst

    log.info("Downloading HydroRIVERS region='%s' → %s", region, dst)
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
        log.error("Download failed (%s): %s", url, str(e))
        try:
            part = dst.with_suffix(dst.suffix + ".part")
            if part.exists():
                part.unlink()
        except Exception:
            log.debug("ignored", exc_info=True)
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
        log.warning("Fixing invalid geometries with buffer(0); input data has geometry errors.")
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

    log.info("%s (layer=%s)", flowlines_path, layer or "<default>")
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

    passthrough_cols = [
        c for c in [
            "streamorde", "streamorder", "streamord", "Strahler", "strahler",
            "totdasqkm", "divdasqkm", "areasqkm", "drainarea", "drainage_a", "upa_km2",
            "arbolatesu", "arbsumkm", "arbolate", "arbolate_km",
            "ftype", "FType", "gnis_name", "GNIS_NAME", "reachcode", "ReachCode", "source",
        ] if c in gdf_lines.columns
    ]

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
            edge_row = {
                "edge_id": f"e_{len(edges)+1:08d}",
                "river_id": row.get("river_id", str(idx)),
                "from_node": n0,
                "to_node": n1,
                "length_m": float(part.length),
                "geometry": part,
            }
            for col in passthrough_cols:
                edge_row[col] = row.get(col)
            edges.append(edge_row)

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
    p.add_argument("--export-aoi", default=None, help="Optional export AOI receipt for the clipped output domain.")
    p.add_argument("--solve-aoi", default=None, help="Optional broader solve AOI receipt for the dominant trunk domain.")
    p.add_argument("--scaffold-aoi", default=None, help="Optional scaffold AOI receipt for the canonical river-network domain.")
    p.add_argument("--solve-halo-km", type=float, default=0.0, help="Optional halo (km) used upstream to build the solve/scaffold domain.")
    p.add_argument("--trusted-halo-m", type=float, default=0.0, help="Optional trusted-export halo (m) used downstream to suppress edge-sensitive river guidance.")
    p.add_argument("--solve-domain-role", default="broader river-network solve extent used to produce AOI-stable dominant trunk selection")
    p.add_argument("--export-domain-role", default="working river export clipped to the requested solve AOI grid/domain")
    p.add_argument("--scaffold-domain-role", default="canonical scaffold domain used for stable river topology and guidance generation")
    p.add_argument("--solve-domain-rationale", default="Solve the dominant trunk on a broader halo-expanded network, then clip/rasterize to the export grid for AOI-stable results.")
    p.add_argument("--out-gpkg", required=True, help="Output GeoPackage path")
    p.add_argument("--provenance-lock", default=None, help="Optional JSON lock file path. If exists, enforce exact match of selected flowlines dataset + fingerprint; otherwise write it.")

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

    # Optional drainage-area raster fallback (e.g., MERIT Hydro upstream area grid).
    # This is only used when NHDPlus drainage-area attributes are unavailable.
    p.add_argument(
    "--da-raster",
    default=None,
    help=(
        "Optional drainage-area raster to sample when NHDPlus drainage area is unavailable. "
        "Example: MERIT Hydro upstream area (UPA)."
    ),
    )
    p.add_argument("--da-raster-band", type=int, default=1, help="Band index for --da-raster (1-based).")
    p.add_argument(
    "--da-raster-units",
    default="km2",
    choices=["km2", "m2"],
    help=(
        "Units of values stored in --da-raster. 'km2' is typical for MERIT UPA; "
        "set explicitly to avoid ambiguity."
    ),
    )
    
    p.add_argument("--hydrography-source", default="arcgis", choices=["arcgis","arcgis_tnm","tnm"],
                   help="Hydrography acquisition strategy: arcgis (default) queries ArcGIS REST first; arcgis_tnm uses TNM as a fallback; tnm prefers TNM.")

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
        default="mainstem_solve_network,major_system_network,outlet_anchors,estuary_control_points,rivers_aoi,rivers_clip,graph_nodes,graph_edges,nhdarea_aoi,nhdarea_clip",
        help="Comma-separated output layers to write.",
    )
    p.add_argument("--manifest-json", default=None, help="Optional river-network manifest JSON path. Defaults next to --out-gpkg.")
    return p.parse_args()


def _attach_drainage_area_from_raster(
    gdf_ll: "gpd.GeoDataFrame",
    *,
    da_raster: str,
    da_raster_band: int = 1,
    da_raster_units: str = "km2",
    logger: Optional[logging.Logger] = None,
) -> "gpd.GeoDataFrame":
    """Fill drain_area_km2 by sampling a drainage-area raster at reach midpoints.

    Intended as a deterministic fallback when NHDPlus_HR attributes are unavailable.
    - gdf_ll is expected in EPSG:4326.
    - Only fills rows where drain_area_km2 is NaN.
    """
    log_ = logger or logging.getLogger(__name__)
    if gdf_ll is None or gdf_ll.empty:
        return gdf_ll
    if not da_raster:
        return gdf_ll

    try:
        import rasterio
        from pyproj import Transformer
    except Exception as e:
        log.debug("_attach_drainage_area_from_raster: suppressed exception", exc_info=True)
        log_.warning("[DA_RASTER] raster sampling unavailable (missing deps): %s", e)
        return gdf_ll

    if "drain_area_km2" not in gdf_ll.columns:
        gdf_ll["drain_area_km2"] = np.nan

    need = pd.to_numeric(gdf_ll["drain_area_km2"], errors="coerce").isna()
    n_need = int(need.sum())
    if n_need == 0:
        return gdf_ll

    try:
        with rasterio.open(da_raster) as ds:
            if ds.crs is None:
                log_.warning("[DA_RASTER] Raster has no CRS; cannot sample: %s", da_raster)
                return gdf_ll
            band = int(da_raster_band)
            if band < 1 or band > ds.count:
                log_.warning("[DA_RASTER] Invalid band %s for raster with %d band(s): %s", band, ds.count, da_raster)
                return gdf_ll

            # Midpoints in EPSG:4326
            try:
                mids = gdf_ll.loc[need, "geometry"].apply(
                    lambda g: g.interpolate(0.5, normalized=True) if g is not None else None
                )
            except Exception:
                log.debug("river_network: suppressed exception", exc_info=True)
                mids = gdf_ll.loc[need, "geometry"].apply(
                    lambda g: g.representative_point() if g is not None else None
                )

            xs: List[float] = []
            ys: List[float] = []
            idxs: List[int] = []
            for idx, pt in mids.items():
                if pt is None or pt.is_empty:
                    continue
                idxs.append(int(idx))
                xs.append(float(pt.x))
                ys.append(float(pt.y))

            if not idxs:
                log_.warning("[DA_RASTER] No valid midpoint geometries to sample.")
                return gdf_ll

            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            rx, ry = tr.transform(xs, ys)
            coords = list(zip(rx, ry))
            vals = list(ds.sample(coords, indexes=band))
            sampled = np.array([v[0] if v is not None and len(v) else np.nan for v in vals], dtype=float)

            nodata = ds.nodata
            if nodata is not None and np.isfinite(nodata):
                sampled = np.where(sampled == float(nodata), np.nan, sampled)

            units = str(da_raster_units).strip().lower()
            if units == "km2":
                da_km2 = sampled
            elif units == "m2":
                da_km2 = sampled / 1e6
            else:
                log_.warning("[DA_RASTER] Unknown units '%s'; assuming km2.", units)
                da_km2 = sampled

            filled = 0
            for i, idx in enumerate(idxs):
                if np.isfinite(da_km2[i]) and da_km2[i] > 0:
                    gdf_ll.at[idx, "drain_area_km2"] = float(da_km2[i])
                    filled += 1

            log_.info(
                "[DA_RASTER] Filled drain_area_km2 from raster for %d/%d reach(es): %s (band=%d units=%s)",
                filled,
                n_need,
                da_raster,
                int(da_raster_band),
                units,
            )
    except Exception as e:
        log.debug("river_network: suppressed exception", exc_info=True)
        log_.warning("[DA_RASTER] Failed to sample drainage-area raster '%s': %s", da_raster, e)

    return gdf_ll


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
            log.info("Loaded %d reaches from --nhd-flowlines.", len(gdf))

            # Provenance lock (file-based)
            prov = {
                "hydrography_source": "nhd_local",
                "aoi": args.aoi,
                "source_path": str(Path(args.nhd_flowlines)),
                "layer": args.layer,
                "fingerprint": fingerprint_path(Path(args.nhd_flowlines)),
            }
            enforce_or_write_lock(Path(args.provenance_lock) if args.provenance_lock else None, prov)

    # 2) Hydrography acquisition (ArcGIS primary by default)
    hydro_src = getattr(args, "hydrography_source", "arcgis").lower().strip()

    # 2a) ArcGIS REST (primary/fast path)
    if (gdf is None or gdf.empty) and (not args.nhd_flowlines) and (hydro_src in ("arcgis", "arcgis_tnm")):
        gdf_arc = try_arcgis_nhd_flowlines(
            aoi=aoi,
            out_crs=args.out_crs,
            timeout_s=int(getattr(args, "arcgis_timeout", 120)),
            da_raster=getattr(args, "da_raster", None),
            da_raster_band=int(getattr(args, "da_raster_band", 1) or 1),
            da_raster_units=str(getattr(args, "da_raster_units", "km2") or "km2"),
        )
        if gdf_arc is not None and not gdf_arc.empty:
            gdf = gdf_arc
            log.info(
                "[OK] Loaded %d reaches via ArcGIS REST (%s).",
                len(gdf),
                str(gdf.get("source", ["arcgis"]).iloc[0]) if "source" in gdf.columns else "arcgis",
            )

    # 2b) TNM explicit fallback (ONLY when requested via --hydrography-source=arcgis_tnm or tnm)
    if (gdf is None or gdf.empty) and (not args.nhd_flowlines) and (hydro_src in ("arcgis_tnm", "tnm")) and args.tnm_enable:
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
        log.info("Total unique candidate products: %d", len(items))

        if items and not args.tnm_dry_run:
            downloads = tnm_download_products(items, dl_dir, timeout_s=int(args.tnm_timeout))
            log.info("Downloaded %d files.", len(downloads))

            roots = extract_archives(downloads, wk_dir)
            src_path, src_layer = find_best_flowline_source(roots)

            if src_path:
                log.info("Using flowlines source: %s (layer=%s)", src_path, src_layer or "<default>")

                # Provenance lock (file-based)
                prov = {
                    "hydrography_source": "tnm_nhd",
                    "aoi": args.aoi,
                    "source_path": str(Path(src_path)),
                    "layer": src_layer,
                    "fingerprint": fingerprint_path(Path(src_path)),
                }
                enforce_or_write_lock(Path(args.provenance_lock) if args.provenance_lock else None, prov)
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
                    log.info("Loaded %d reaches from TNM download.", len(gdf))
            else:
                log.warning("Could not locate a flowline dataset/layer in downloaded products.")
        elif args.tnm_dry_run:
            log.info("Dry-run enabled; skipping download.")
        else:
            log.warning("No products found for AOI.")

    # 2c) Guardrail: if user forced TNM but it failed, try ArcGIS so the run can proceed.
    if (gdf is None or gdf.empty) and (not args.nhd_flowlines) and hydro_src == "tnm":
        gdf_arc = try_arcgis_nhd_flowlines(
            aoi=aoi,
            out_crs=args.out_crs,
            timeout_s=int(getattr(args, "arcgis_timeout", 120)),
            da_raster=getattr(args, "da_raster", None),
            da_raster_band=int(getattr(args, "da_raster_band", 1) or 1),
            da_raster_units=str(getattr(args, "da_raster_units", "km2") or "km2"),
        )
        if gdf_arc is not None and not gdf_arc.empty:
            gdf = gdf_arc
            log.warning("No usable TNM flowlines; falling back to ArcGIS REST (%s).",
                        str(gdf.get("source", ["arcgis"]).iloc[0]) if "source" in gdf.columns else "arcgis")
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
            log.info("Using downloaded HydroRIVERS shapefile: %s", hydrorivers_path)

        if hydrorivers_path is None:
            raise RuntimeError(
                "No NHD flowlines found for AOI. Provide --nhd-flowlines OR enable TNM with --tnm-enable, "
                "OR enable HydroRIVERS fallback with --hydrorivers-auto-download (or provide --hydrorivers)."
            )

        log.warning("Using HydroRIVERS (%s).", hydrorivers_path)
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
        log.info("Loaded %d HydroRIVERS reaches in AOI.", len(gdf))

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
                # nhdarea_aoi stores ALL NHDArea types (StreamRiver=460, Sea/Ocean=445/FCode 44500, etc.)
                # nhdarea_clip is filtered to StreamRiver only for river domain construction.
                # Sea/Ocean polygons are preserved in nhdarea_aoi for the shared estuary/coastal
                # handoff in guidance_domains.py.
                all_clipped = clip_polys_to_aoi(nhdarea_aoi, aoi_poly_proj)
                nhdarea_clip = filter_nhdarea_by_ftype(all_clipped, allow_ftypes=(460,))
                n_sea_ocean = len(all_clipped) - len(nhdarea_clip) if all_clipped is not None else 0
                log.info("Loaded %d NHDArea polygon(s) via ArcGIS REST (%d StreamRiver, %d other including Sea/Ocean).",
                         len(all_clipped) if all_clipped is not None else 0,
                         len(nhdarea_clip),
                         n_sea_ocean)
                # Store the full unfiltered clipped set in nhdarea_aoi for downstream Sea/Ocean detection
                nhdarea_aoi = all_clipped
            else:
                log.info("No NHDArea polygons found in AOI (continuing).")
        except Exception as e:
            log.warning("NHDArea polygon fetch failed (continuing): %s", str(e))

    solve_nodes_gdf, solve_edges_gdf = build_reach_graph(rivers_aoi, snap_m=float(args.snap_m))
    solve_edges_gdf = station_edges_from_root(solve_nodes_gdf, solve_edges_gdf)
    mainstem_solve_network = solve_edges_gdf.copy()
    mainstem_solve_network["solve_domain_role"] = "dominant trunk solve layer"
    hydrologic_artifacts = build_hydrologic_solve_domain_artifacts(
        solve_network=mainstem_solve_network,
        nodes=solve_nodes_gdf,
        export_aoi=getattr(args, "export_aoi", None),
        solve_aoi=getattr(args, "solve_aoi", None),
        scaffold_aoi=getattr(args, "scaffold_aoi", None),
        solve_halo_km=float(getattr(args, "solve_halo_km", 0.0) or 0.0),
        trusted_halo_m=float(getattr(args, "trusted_halo_m", 0.0) or 0.0),
        solve_domain_role=str(getattr(args, "solve_domain_role", "broader river-network solve extent used to produce AOI-stable dominant trunk selection")),
        export_domain_role=str(getattr(args, "export_domain_role", "working river export clipped to the requested solve AOI grid/domain")),
        scaffold_domain_role=str(getattr(args, "scaffold_domain_role", "canonical scaffold domain used for stable river topology and guidance generation")),
        solve_domain_rationale=str(getattr(args, "solve_domain_rationale", "Solve the dominant trunk on a broader halo-expanded network, then clip/rasterize to the export grid for AOI-stable results.")),
    )
    major_system_network = hydrologic_artifacts.major_system_network.copy()
    major_system_network["solve_domain_role"] = "major hydrologic system used for dominant trunk selection"

    nodes_gdf, edges_gdf = build_reach_graph(rivers_clip, snap_m=float(args.snap_m))
    edges_gdf = station_edges_from_root(nodes_gdf, edges_gdf)

    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [s.strip() for s in args.write_layers.split(",") if s.strip()]

    log.info("%s", out_path)
    if "mainstem_solve_network" in layers:
        mainstem_solve_network.to_file(out_path, layer="mainstem_solve_network", driver="GPKG")
    if "major_system_network" in layers:
        major_system_network.to_file(out_path, layer="major_system_network", driver="GPKG")
    if "outlet_anchors" in layers and hydrologic_artifacts.outlet_anchors is not None and not hydrologic_artifacts.outlet_anchors.empty:
        hydrologic_artifacts.outlet_anchors.to_file(out_path, layer="outlet_anchors", driver="GPKG")
    if "estuary_control_points" in layers and hydrologic_artifacts.estuary_control_points is not None and not hydrologic_artifacts.estuary_control_points.empty:
        hydrologic_artifacts.estuary_control_points.to_file(out_path, layer="estuary_control_points", driver="GPKG")
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

    hydrologic_domain_path = out_path.with_name("hydrologic_solve_domain.json")
    write_hydrologic_solve_domain_json(hydrologic_domain_path, hydrologic_artifacts.hydrologic_manifest)

    manifest_path = Path(args.manifest_json) if getattr(args, "manifest_json", None) else _network_manifest_default_path(out_path)
    _write_network_manifest(
        manifest_path,
        _build_network_manifest(
            args=args,
            out_gpkg=out_path,
            hydrologic_domain_path=hydrologic_domain_path,
            hydrologic_manifest=hydrologic_artifacts.hydrologic_manifest,
            mainstem_solve_network=mainstem_solve_network,
            major_system_network=major_system_network,
            outlet_anchors=hydrologic_artifacts.outlet_anchors,
            estuary_control_points=hydrologic_artifacts.estuary_control_points,
            rivers_aoi=rivers_aoi,
            rivers_clip=rivers_clip,
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            nhdarea_clip=nhdarea_clip,
        ),
    )
    log.info("[DONE] river_network_manifest=%s", manifest_path)
    log.info("[DONE] hydrologic_solve_domain=%s", hydrologic_domain_path)

    log.info(
        "[DONE] mainstem_solve_network=%d | major_system_network=%d | outlet_anchors=%d | rivers_aoi=%d | rivers_clip=%d | nodes=%d | edges=%d | nhdarea=%d | CRS=%s | source=%s",
        len(mainstem_solve_network),
        len(major_system_network),
        len(hydrologic_artifacts.outlet_anchors),
        len(rivers_aoi),
        len(rivers_clip),
        len(nodes_gdf),
        len(edges_gdf),
        len(nhdarea_clip) if nhdarea_clip is not None else 0,
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
