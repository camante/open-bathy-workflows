#!/usr/bin/env python3
"""Utilities to auto-build and cache authoritative_base.tif from NOAA CUDEM products.

This module materializes an AOI-keyed authoritative base using:
- NOAA CUDEM tile index zip (for intersecting DEM tile names/URLs)
- NOAA spatial metadata zip (for polygons describing where each DEM tile is supported by inputs)

Core workflow
-------------
1. Select intersecting CUDEM DEM tiles for an AOI (W/E/S/N, EPSG:4269).
2. Download/reuse only those DEM tiles.
3. Match each tile to its spatial-metadata GeoPackage.
4. Rasterize the clipped support polygons to the DEM mosaic grid.
5. Mask the CUDEM DEM mosaic so only measurement-constrained cells remain.
6. Cache the result under: <cache_root>/authoritative_base/<cache_key>/

This gives bathy_main an AOI-reusable authoritative_base.tif that can be hard-locked
in the final support-aware terrain generation workflow.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urlparse

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import box

from cache_utils import artifact_cache_key, fingerprint_code, meta_payload, write_meta, cache_hit, canonical_json, sha1_hex

LOG = logging.getLogger("cudem_authoritative")

DEFAULT_TILE_INDEX_URL = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
    "NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip"
)
DEFAULT_SPATIAL_META_URL = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
    "NCEI_ninth_Topobathy_2014_8483/ninth_spatial_meta.zip"
)


@dataclass(frozen=True)
class TileRecord:
    tile_name: str
    tile_url: str
    tile_geom: object
    meta_path: Optional[Path] = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_aoi(aoi_str: str) -> tuple[float, float, float, float]:
    try:
        west, east, south, north = (float(v) for v in str(aoi_str).split("/"))
    except Exception as exc:
        LOG.debug("parse_aoi: suppressed exception", exc_info=True)
        raise ValueError(f"Could not parse AOI '{aoi_str}' as W/E/S/N") from exc
    if not (west < east and south < north):
        raise ValueError(f"Invalid AOI extents: {aoi_str}")
    return west, east, south, north


def download_file(url: str, dest: Path, *, overwrite: bool = False, timeout: int = 120, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or LOG
    ensure_dir(dest.parent)
    if dest.exists() and not overwrite:
        logger.info("[AUTHORITATIVE] Using existing file: %s", dest)
        return dest
    logger.info("[AUTHORITATIVE] Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def extract_zip(zip_path: Path, extract_dir: Path, *, overwrite: bool = False, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or LOG
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("[AUTHORITATIVE] Using existing extracted directory: %s", extract_dir)
        return extract_dir
    ensure_dir(extract_dir)
    logger.info("[AUTHORITATIVE] Extracting %s -> %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def discover_vector_file(root: Path) -> Path:
    candidates = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".shp", ".gpkg", ".geojson"}
    )
    if not candidates:
        raise FileNotFoundError(f"No vector file found under {root}")
    for suffix in (".shp", ".gpkg", ".geojson"):
        for p in candidates:
            if p.suffix.lower() == suffix:
                return p
    return candidates[0]


def detect_url_field(gdf: gpd.GeoDataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit in gdf.columns:
            return explicit
        raise KeyError(f"Requested tile URL field '{explicit}' not found in tile index columns")
    lower_map = {str(col).lower(): str(col) for col in gdf.columns}
    for key in ("url", "downloadurl", "download_url", "href", "link"):
        if key in lower_map:
            return lower_map[key]
    for col in gdf.columns:
        if "url" in str(col).lower():
            return str(col)
    raise KeyError("Could not find a URL-like field in the tile index")


def read_tile_index(tile_index_vector: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(tile_index_vector)
    if gdf.empty:
        raise ValueError(f"Tile index is empty: {tile_index_vector}")
    if gdf.crs is None:
        raise ValueError(f"Tile index has no CRS: {tile_index_vector}")
    return gdf


def select_tiles(tile_index_gdf: gpd.GeoDataFrame, aoi_bounds_4269: tuple[float, float, float, float], url_field: str) -> list[TileRecord]:
    west, east, south, north = aoi_bounds_4269
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4269")
    aoi_in_tile_crs = aoi_gdf.to_crs(tile_index_gdf.crs)
    hits = tile_index_gdf[tile_index_gdf.intersects(aoi_in_tile_crs.geometry.iloc[0])].copy()
    hits = hits.to_crs("EPSG:4269")
    if hits.empty:
        raise ValueError("No CUDEM tiles intersect the AOI")
    records: list[TileRecord] = []
    for _, row in hits.iterrows():
        tile_url = str(row[url_field])
        tile_name = Path(urlparse(tile_url).path).name
        records.append(TileRecord(tile_name=tile_name, tile_url=tile_url, tile_geom=row.geometry))
    records.sort(key=lambda r: r.tile_name)
    return records


def write_tile_manifest(records: list[TileRecord], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_name", "tile_url", "meta_path"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "tile_name": rec.tile_name,
                "tile_url": rec.tile_url,
                "meta_path": "" if rec.meta_path is None else str(rec.meta_path),
            })


def download_tiles(records: list[TileRecord], tile_dir: Path, *, overwrite: bool = False, logger: Optional[logging.Logger] = None) -> list[Path]:
    logger = logger or LOG
    ensure_dir(tile_dir)
    local_paths: list[Path] = []
    for rec in records:
        dest = tile_dir / rec.tile_name
        local_paths.append(download_file(rec.tile_url, dest, overwrite=overwrite, logger=logger))
    return local_paths


def tile_stem_tokens(tile_name: str) -> list[str]:
    stem = Path(tile_name).stem
    tokens = [stem]
    stem_no_version = re.sub(r"_\d{4}v\d+$", "", stem)
    if stem_no_version != stem:
        tokens.append(stem_no_version)
    nw_match = re.search(r"([ns]\d+x\d+_[ew]\d+x\d+)", stem, flags=re.IGNORECASE)
    if nw_match:
        tokens.append(nw_match.group(1).lower())
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key not in seen:
            out.append(token)
            seen.add(key)
    return out


def find_matching_metadata(tile_name: str, meta_root: Path) -> Optional[Path]:
    gpkg_files = sorted(p for p in meta_root.rglob("*.gpkg") if p.is_file())
    if not gpkg_files:
        return None
    tokens = tile_stem_tokens(tile_name)
    stem = Path(tile_name).stem.lower()
    for gpkg in gpkg_files:
        if gpkg.stem.lower() == stem:
            return gpkg
    for token in tokens[1:]:
        for gpkg in gpkg_files:
            if gpkg.stem.lower() == token.lower():
                return gpkg
    for token in tokens:
        token_l = token.lower()
        for gpkg in gpkg_files:
            gpkg_stem = gpkg.stem.lower()
            if token_l in gpkg_stem or gpkg_stem in token_l:
                return gpkg
    return None


def attach_metadata_paths(records: list[TileRecord], meta_root: Path, missing_meta_policy: str, logger: Optional[logging.Logger] = None) -> list[TileRecord]:
    logger = logger or LOG
    out: list[TileRecord] = []
    missing: list[str] = []
    for rec in records:
        meta_path = find_matching_metadata(rec.tile_name, meta_root)
        if meta_path is None:
            missing.append(rec.tile_name)
        out.append(TileRecord(rec.tile_name, rec.tile_url, rec.tile_geom, meta_path))
    if missing:
        message = "No matching spatial metadata found for tile(s): " + ", ".join(missing)
        if missing_meta_policy == "error":
            raise FileNotFoundError(message)
        logger.warning("[AUTHORITATIVE] %s", message)
    return out


def build_dem_mosaic(tile_paths: list[Path], aoi_bounds_4269: tuple[float, float, float, float]) -> tuple[np.ndarray, rasterio.Affine, dict]:
    if not tile_paths:
        raise ValueError("No DEM tiles provided to mosaic")
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        crs_set = {str(src.crs) for src in srcs}
        if len(crs_set) != 1:
            raise ValueError(f"Expected one DEM CRS, found: {sorted(crs_set)}")
        first_crs = srcs[0].crs
        west, east, south, north = aoi_bounds_4269
        if first_crs is not None and first_crs.is_geographic:
            bounds = (west, south, east, north)
        else:
            bounds = transform_bounds("EPSG:4269", first_crs, west, south, east, north, densify_pts=21)
        mosaic, transform = merge(srcs, bounds=bounds)
        if mosaic.shape[1] > 200000 or mosaic.shape[2] > 200000:
            raise RuntimeError(
                "Refusing to build an implausibly large DEM mosaic "
                f"({mosaic.shape[2]} x {mosaic.shape[1]} pixels). "
                "This usually indicates a CRS or bounds-order bug."
            )
        profile = srcs[0].profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": mosaic.shape[0],
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        })
        return mosaic, transform, profile
    finally:
        for src in srcs:
            src.close()


def polygon_layers(gpkg_path: Path) -> Iterator[str]:
    for layer in fiona.listlayers(gpkg_path):
        try:
            with fiona.open(gpkg_path, layer=layer) as src:
                geom_type = (src.schema or {}).get("geometry", "")
        except Exception:
            LOG.debug("polygon_layers: suppressed exception", exc_info=True)
            continue
        if "Polygon" in str(geom_type):
            yield layer


def collect_support_geometries(records: list[TileRecord], aoi_bounds_4269: tuple[float, float, float, float], out_crs: str, missing_meta_policy: str, logger: Optional[logging.Logger] = None) -> gpd.GeoDataFrame:
    logger = logger or LOG
    west, east, south, north = aoi_bounds_4269
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4269")
    pieces: list[gpd.GeoDataFrame] = []

    for rec in records:
        if rec.meta_path is None:
            if missing_meta_policy == "tile_extent":
                tile_gdf = gpd.GeoDataFrame(
                    {"tile_name": [rec.tile_name], "source": ["tile_extent_fallback"]},
                    geometry=[rec.tile_geom],
                    crs="EPSG:4269",
                )
                if tile_gdf.crs != aoi_gdf.crs:
                    tile_gdf = tile_gdf.to_crs(aoi_gdf.crs)
                tile_clip = gpd.clip(tile_gdf, aoi_gdf)
                if not tile_clip.empty:
                    pieces.append(tile_clip)
            continue

        for layer in polygon_layers(rec.meta_path):
            try:
                gdf = gpd.read_file(rec.meta_path, layer=layer)
            except Exception as exc:
                logger.warning("[AUTHORITATIVE] Skipping unreadable layer %s in %s: %s", layer, rec.meta_path, exc)
                continue
            if gdf.empty:
                continue
            if gdf.crs is None:
                logger.warning("[AUTHORITATIVE] Skipping layer %s in %s because CRS is missing", layer, rec.meta_path)
                continue
            gdf = gdf.to_crs(aoi_gdf.crs)
            try:
                clipped = gpd.clip(gdf, aoi_gdf)
            except Exception as exc:
                logger.warning("[AUTHORITATIVE] Clip failed for layer %s in %s: %s", layer, rec.meta_path, exc)
                continue
            if clipped.empty:
                continue
            clipped = clipped[[c for c in clipped.columns if c != "geometry"] + ["geometry"]].copy()
            clipped["tile_name"] = rec.tile_name
            clipped["meta_file"] = rec.meta_path.name
            clipped["meta_layer"] = layer
            pieces.append(clipped)

    if not pieces:
        raise ValueError("No authoritative support geometries were collected from spatial metadata")

    merged = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True, sort=False), geometry="geometry", crs=pieces[0].crs)
    merged = merged[(~merged.geometry.is_empty) & (merged.geometry.notnull())].copy()
    if out_crs and str(merged.crs) != str(out_crs):
        merged = merged.to_crs(out_crs)
    return merged


def rasterize_support_mask(support_gdf: gpd.GeoDataFrame, out_shape: tuple[int, int], transform: rasterio.Affine) -> np.ndarray:
    shapes = ((geom, 1) for geom in support_gdf.geometry if geom is not None and not geom.is_empty)
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )


def apply_mask_to_mosaic(mosaic: np.ndarray, mask: np.ndarray, nodata_value: float) -> np.ndarray:
    out = mosaic.copy()
    if out.shape[0] != 1:
        raise ValueError(f"Expected a single-band DEM mosaic, got shape {out.shape}")
    out[0, mask == 0] = nodata_value
    return out


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    ensure_dir(path.parent)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def materialize_authoritative_base_for_aoi(
    *,
    aoi: str,
    cache_root: Path,
    tile_index_url: str = DEFAULT_TILE_INDEX_URL,
    spatial_meta_url: str = DEFAULT_SPATIAL_META_URL,
    missing_meta_policy: str = "skip",
    tile_url_field: Optional[str] = None,
    force_rebuild: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Build or reuse authoritative_base.tif for an AOI under cache_root.

    Returns a dict with keys including:
      - cache_key
      - cache_hit
      - cache_dir
      - authoritative_base
      - authoritative_coverage_mask
      - authoritative_support_coverage
      - selected_cudem_tiles_csv
      - tile_count
    """
    logger = logger or LOG
    west, east, south, north = parse_aoi(aoi)
    cache_root = Path(cache_root).expanduser().resolve()
    stage_root = ensure_dir(cache_root / "authoritative_base")

    params = {
        "aoi": aoi,
        "aoi_bounds": [west, east, south, north],
        "tile_index_url": tile_index_url,
        "spatial_meta_url": spatial_meta_url,
        "missing_meta_policy": missing_meta_policy,
        "tile_url_field": tile_url_field,
        "stage_version": 2,
    }
    code_fp = fingerprint_code(Path(__file__))
    cache_key = artifact_cache_key(stage="authoritative_base_cudem", params=params, inputs={}, code_fp=code_fp)
    entry_dir = stage_root / cache_key
    meta_path = entry_dir / "cache_meta.json"
    auth_path = entry_dir / "authoritative_base.tif"
    baseline_path = entry_dir / "cudem_baseline_interpolation.tif"
    mask_path = entry_dir / "authoritative_coverage_mask.tif"
    gpkg_path = entry_dir / "authoritative_support_coverage.gpkg"
    manifest_csv = entry_dir / "selected_cudem_tiles.csv"

    if force_rebuild and entry_dir.exists():
        shutil.rmtree(entry_dir)
    ensure_dir(entry_dir)

    hit, _ = cache_hit(meta_path, expected_key=cache_key, expected_params=params)
    if hit and auth_path.exists() and baseline_path.exists() and mask_path.exists() and gpkg_path.exists():
        logger.info("[AUTHORITATIVE] Cache hit for AOI=%s -> %s", aoi, entry_dir)
        cached_extra: Dict[str, Any] = {}
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(cached_meta, dict):
                maybe_extra = cached_meta.get("extra", {})
                if isinstance(maybe_extra, dict):
                    cached_extra = maybe_extra
        except Exception:
            logger.debug("[AUTHORITATIVE] Failed to read cached authoritative-base metadata", exc_info=True)
        return {
            "mode": "auto_cudem",
            "cache_key": cache_key,
            "cache_hit": True,
            "cache_dir": str(entry_dir),
            "authoritative_base": str(auth_path),
            "baseline_cudem_interpolation": str(baseline_path),
            "authoritative_coverage_mask": str(mask_path),
            "authoritative_support_coverage": str(gpkg_path),
            "selected_cudem_tiles_csv": str(manifest_csv),
            "tile_count": cached_extra.get("tile_count"),
            "aoi": aoi,
            "source_urls": {
                "tile_index_url": tile_index_url,
                "spatial_meta_url": spatial_meta_url,
            },
            "missing_meta_policy": missing_meta_policy,
            **cached_extra,
        }

    shared_key = sha1_hex(canonical_json({
        "tile_index_url": tile_index_url,
        "spatial_meta_url": spatial_meta_url,
    }))[:12]
    shared_root = ensure_dir(stage_root / "_shared" / shared_key)
    tile_index_zip = shared_root / Path(urlparse(tile_index_url).path).name
    spatial_meta_zip = shared_root / Path(urlparse(spatial_meta_url).path).name
    tile_index_dir = shared_root / "tile_index"
    spatial_meta_dir = shared_root / "spatial_meta"
    # Share downloaded CUDEM DEM tiles across AOIs/settings as well, not just the
    # tile-index and spatial-metadata zips. The per-AOI cache entry still records
    # its own manifest and outputs, but overlapping AOIs no longer re-download the
    # same DEM tile TIFFs.
    tile_download_dir = ensure_dir(shared_root / "tiles")

    tile_index_reused_existing = tile_index_zip.exists()
    download_file(tile_index_url, tile_index_zip, overwrite=False, logger=logger)
    tile_index_extract_reused_existing = tile_index_dir.exists() and any(tile_index_dir.iterdir())
    extract_zip(tile_index_zip, tile_index_dir, overwrite=False, logger=logger)
    tile_index_vector = discover_vector_file(tile_index_dir)
    tile_index_gdf = read_tile_index(tile_index_vector)
    url_field = detect_url_field(tile_index_gdf, explicit=tile_url_field)
    logger.info("[AUTHORITATIVE] Using tile-index URL field: %s", url_field)

    records = select_tiles(tile_index_gdf, (west, east, south, north), url_field)
    logger.info("[AUTHORITATIVE] Selected %d intersecting CUDEM tile(s)", len(records))

    spatial_meta_reused_existing = spatial_meta_zip.exists()
    download_file(spatial_meta_url, spatial_meta_zip, overwrite=False, logger=logger)
    spatial_meta_extract_reused_existing = spatial_meta_dir.exists() and any(spatial_meta_dir.iterdir())
    extract_zip(spatial_meta_zip, spatial_meta_dir, overwrite=False, logger=logger)
    records = attach_metadata_paths(records, spatial_meta_dir, missing_meta_policy, logger=logger)
    write_tile_manifest(records, manifest_csv)

    preexisting_tiles = {rec.tile_name: (tile_download_dir / rec.tile_name).exists() for rec in records}
    tile_paths = download_tiles(records, tile_download_dir, overwrite=False, logger=logger)
    tile_download_stats = {
        "requested": int(len(records)),
        "reused_existing": int(sum(1 for rec in records if preexisting_tiles.get(rec.tile_name, False))),
        "downloaded_new": int(sum(1 for rec in records if not preexisting_tiles.get(rec.tile_name, False))),
    }
    mosaic, transform, dem_profile = build_dem_mosaic(tile_paths, (west, east, south, north))
    dem_crs = dem_profile.get("crs")

    support_gdf = collect_support_geometries(records, (west, east, south, north), str(dem_crs), missing_meta_policy, logger=logger)
    ensure_dir(gpkg_path.parent)
    support_gdf.to_file(gpkg_path, driver="GPKG")

    out_shape = (dem_profile["height"], dem_profile["width"])
    support_mask = rasterize_support_mask(support_gdf, out_shape, transform)

    nodata_value = dem_profile.get("nodata", -999999.0)
    if nodata_value is None:
        nodata_value = -999999.0
        dem_profile["nodata"] = nodata_value

    mask_profile = dem_profile.copy()
    mask_profile.update(dtype="uint8", nodata=0, count=1)
    write_raster(mask_path, support_mask[np.newaxis, :, :], mask_profile)

    authoritative_base = apply_mask_to_mosaic(mosaic, support_mask, float(nodata_value))
    base_profile = dem_profile.copy()
    base_profile.update(dtype=str(authoritative_base.dtype), count=1)
    write_raster(baseline_path, mosaic.astype(np.float32), base_profile)
    write_raster(auth_path, authoritative_base, base_profile)

    extra = {
        "cache_dir": str(entry_dir),
        "authoritative_base": str(auth_path),
        "baseline_cudem_interpolation": str(baseline_path),
        "authoritative_coverage_mask": str(mask_path),
        "authoritative_support_coverage": str(gpkg_path),
        "selected_cudem_tiles_csv": str(manifest_csv),
        "tile_count": len(records),
        "aoi": aoi,
        "source_urls": {
            "tile_index_url": tile_index_url,
            "spatial_meta_url": spatial_meta_url,
        },
        "missing_meta_policy": missing_meta_policy,
        "shared_root": str(shared_root),
        "shared_tile_cache": str(tile_download_dir),
        "shared_cache_reuse": {
            "tile_index_zip_reused_existing": bool(tile_index_reused_existing),
            "tile_index_extract_reused_existing": bool(tile_index_extract_reused_existing),
            "spatial_meta_zip_reused_existing": bool(spatial_meta_reused_existing),
            "spatial_meta_extract_reused_existing": bool(spatial_meta_extract_reused_existing),
            "tile_downloads": tile_download_stats,
        },
    }
    write_meta(meta_path, meta_payload(
        cache_key=cache_key,
        stage="authoritative_base_cudem",
        params=params,
        inputs={},
        code_fp=code_fp,
        extra=extra,
    ))

    return {
        "mode": "auto_cudem",
        "cache_key": cache_key,
        "cache_hit": False,
        **extra,
    }
