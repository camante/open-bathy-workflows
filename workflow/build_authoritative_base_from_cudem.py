#!/usr/bin/env python3
"""
Build an authoritative_base.tif from NOAA CUDEM tile-index and spatial-metadata products.

Workflow
--------
1. Download and extract the CUDEM tile index zip.
2. Select intersecting DEM tiles for a W/E/S/N AOI using the tile index.
3. Download only those DEM tiles.
4. Download and extract the spatial metadata zip.
5. Match each selected DEM tile to a corresponding spatial-metadata GeoPackage
   using the tile stem / NW-corner naming token (e.g. n36x50_w076x50).
6. Read polygon layers from the matched GeoPackages, clip to the AOI, and
   rasterize them onto the DEM grid.
7. Apply the rasterized coverage mask to the DEM mosaic to produce
   authoritative_base.tif.

Notes
-----
- Conservative default behavior for missing spatial metadata is "skip": cells
  from tiles without spatial metadata remain nodata in authoritative_base.tif.
- You can optionally fall back to the full tile footprint with
  --missing-meta-policy tile_extent if you want a less conservative mask.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional
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
from shapely.geometry import box, mapping

LOG = logging.getLogger("build_authoritative_base_from_cudem")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi",
        required=True,
        help="AOI as W/E/S/N in geographic coordinates, e.g. -71/-70.75/42.75/43",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Working directory for downloaded zips, extracted metadata, tiles, and outputs.",
    )
    parser.add_argument(
        "--tile-index-url",
        default=DEFAULT_TILE_INDEX_URL,
        help="NOAA CUDEM tile-index zip URL.",
    )
    parser.add_argument(
        "--spatial-meta-url",
        default=DEFAULT_SPATIAL_META_URL,
        help="NOAA CUDEM spatial-metadata zip URL.",
    )
    parser.add_argument(
        "--output-authoritative-base",
        default="authoritative_base.tif",
        help="Output filename for the authoritative-base raster (inside --work-dir if relative).",
    )
    parser.add_argument(
        "--output-coverage-mask",
        default="authoritative_coverage_mask.tif",
        help="Output filename for the rasterized authoritative coverage mask.",
    )
    parser.add_argument(
        "--output-clipped-meta",
        default="authoritative_support_coverage.gpkg",
        help="Output filename for clipped/merged authoritative support polygons.",
    )
    parser.add_argument(
        "--missing-meta-policy",
        choices=("skip", "tile_extent", "error"),
        default="skip",
        help=(
            "Behavior when an intersecting DEM tile has no matching spatial metadata. "
            "skip=leave those areas nodata; tile_extent=use the selected tile footprint as a fallback; "
            "error=abort."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloads and outputs.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip files after extraction.",
    )
    parser.add_argument(
        "--tile-url-field",
        default=None,
        help="Optional explicit field name in the tile index containing download URLs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_aoi(aoi_str: str) -> tuple[float, float, float, float]:
    try:
        west, east, south, north = (float(v) for v in aoi_str.split("/"))
    except Exception as exc:  # pragma: no cover - input parsing guard
        raise ValueError(f"Could not parse --aoi '{aoi_str}' as W/E/S/N") from exc
    if west >= east or south >= north:
        raise ValueError(f"Invalid AOI extents: {aoi_str}")
    return west, east, south, north


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_resolve_output(path_str: str, work_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else work_dir / path


def download_file(url: str, dest: Path, overwrite: bool = False, timeout: int = 120) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not overwrite:
        LOG.info("Using existing file: %s", dest)
        return dest
    LOG.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool = False) -> Path:
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        LOG.info("Using existing extracted directory: %s", extract_dir)
        return extract_dir
    ensure_dir(extract_dir)
    LOG.info("Extracting %s -> %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def discover_vector_file(root: Path) -> Path:
    candidates = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".shp", ".gpkg", ".geojson"}
        ]
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
        raise KeyError(f"Requested --tile-url-field '{explicit}' not found in tile index columns")
    lower_map = {str(col).lower(): str(col) for col in gdf.columns}
    preferred = ["url", "downloadurl", "download_url", "href", "link"]
    for key in preferred:
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


def select_tiles(
    tile_index_gdf: gpd.GeoDataFrame,
    aoi_bounds_4269: tuple[float, float, float, float],
    url_field: str,
) -> list[TileRecord]:
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
            writer.writerow(
                {
                    "tile_name": rec.tile_name,
                    "tile_url": rec.tile_url,
                    "meta_path": "" if rec.meta_path is None else str(rec.meta_path),
                }
            )


def download_tiles(records: list[TileRecord], tile_dir: Path, overwrite: bool = False) -> list[Path]:
    ensure_dir(tile_dir)
    local_paths: list[Path] = []
    for rec in records:
        dest = tile_dir / rec.tile_name
        local_paths.append(download_file(rec.tile_url, dest, overwrite=overwrite))
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
    deduped: list[str] = []
    for tok in tokens:
        if tok.lower() not in [x.lower() for x in deduped]:
            deduped.append(tok)
    return deduped


def find_matching_metadata(tile_name: str, meta_root: Path) -> Optional[Path]:
    gpkg_files = sorted(p for p in meta_root.rglob("*.gpkg") if p.is_file())
    if not gpkg_files:
        return None
    tokens = tile_stem_tokens(tile_name)

    # 1) exact stem match
    stem = Path(tile_name).stem.lower()
    for gpkg in gpkg_files:
        if gpkg.stem.lower() == stem:
            return gpkg

    # 2) exact no-version or NW-corner token match
    for token in tokens[1:]:
        for gpkg in gpkg_files:
            if gpkg.stem.lower() == token.lower():
                return gpkg

    # 3) containment match (robust to prefixes/suffixes in metadata names)
    for token in tokens:
        for gpkg in gpkg_files:
            gpkg_stem = gpkg.stem.lower()
            if token.lower() in gpkg_stem or gpkg_stem in token.lower():
                return gpkg

    return None


def attach_metadata_paths(
    records: list[TileRecord],
    meta_root: Path,
    missing_meta_policy: str,
) -> list[TileRecord]:
    out: list[TileRecord] = []
    missing: list[str] = []
    for rec in records:
        meta_path = find_matching_metadata(rec.tile_name, meta_root)
        if meta_path is None:
            missing.append(rec.tile_name)
        out.append(TileRecord(rec.tile_name, rec.tile_url, rec.tile_geom, meta_path))

    if missing:
        message = (
            "No matching spatial metadata found for the following tile(s): "
            + ", ".join(missing)
        )
        if missing_meta_policy == "error":
            raise FileNotFoundError(message)
        LOG.warning(message)
    return out


def build_dem_mosaic(
    tile_paths: list[Path],
    aoi_bounds_4269: tuple[float, float, float, float],
) -> tuple[np.ndarray, rasterio.Affine, dict]:
    if not tile_paths:
        raise ValueError("No DEM tiles provided to mosaic")
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        crs_set = {str(src.crs) for src in srcs}
        if len(crs_set) != 1:
            raise ValueError(f"Expected one DEM CRS, found: {sorted(crs_set)}")
        first_crs = srcs[0].crs
        west, east, south, north = aoi_bounds_4269
        # rasterio.merge() expects bounds as (left, bottom, right, top), not W/E/S/N.
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
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "compress": "deflate",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )
        return mosaic, transform, profile
    finally:
        for src in srcs:
            src.close()


def polygon_layers(gpkg_path: Path) -> Iterator[str]:
    for layer in fiona.listlayers(gpkg_path):
        try:
            with fiona.open(gpkg_path, layer=layer) as src:
                geom_type = (src.schema or {}).get("geometry", "")
        except Exception as exc:
            LOG.warning("Skipping unreadable layer %s in %s: %s", layer, gpkg_path, exc)
            continue
        if "Polygon" in str(geom_type):
            yield layer


def collect_support_geometries(
    records: list[TileRecord],
    aoi_bounds_4269: tuple[float, float, float, float],
    out_crs: str,
    missing_meta_policy: str,
) -> gpd.GeoDataFrame:
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
                LOG.warning("Skipping unreadable layer %s in %s: %s", layer, rec.meta_path, exc)
                continue
            if gdf.empty:
                continue
            if gdf.crs is None:
                LOG.warning("Skipping layer %s in %s because CRS is missing", layer, rec.meta_path)
                continue
            gdf = gdf.to_crs(aoi_gdf.crs)
            try:
                clipped = gpd.clip(gdf, aoi_gdf)
            except Exception as exc:
                LOG.warning("Clip failed for layer %s in %s: %s", layer, rec.meta_path, exc)
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

    merged = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True, sort=False),
        geometry="geometry",
        crs=pieces[0].crs,
    )
    merged = merged[(~merged.geometry.is_empty) & (merged.geometry.notnull())].copy()
    if out_crs and str(merged.crs) != str(out_crs):
        merged = merged.to_crs(out_crs)
    return merged


def rasterize_support_mask(
    support_gdf: gpd.GeoDataFrame,
    out_shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    shapes = ((geom, 1) for geom in support_gdf.geometry if geom is not None and not geom.is_empty)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )
    return mask


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


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    west, east, south, north = parse_aoi(args.aoi)
    work_dir = Path(args.work_dir).expanduser().resolve()
    ensure_dir(work_dir)

    out_authoritative = maybe_resolve_output(args.output_authoritative_base, work_dir)
    out_mask = maybe_resolve_output(args.output_coverage_mask, work_dir)
    out_meta = maybe_resolve_output(args.output_clipped_meta, work_dir)
    manifest_csv = work_dir / "selected_cudem_tiles.csv"

    tile_index_zip = work_dir / Path(urlparse(args.tile_index_url).path).name
    spatial_meta_zip = work_dir / Path(urlparse(args.spatial_meta_url).path).name
    tile_index_dir = work_dir / "tile_index"
    spatial_meta_dir = work_dir / "spatial_meta"
    tile_download_dir = work_dir / "tiles"

    download_file(args.tile_index_url, tile_index_zip, overwrite=args.overwrite)
    extract_zip(tile_index_zip, tile_index_dir, overwrite=args.overwrite)
    tile_index_vector = discover_vector_file(tile_index_dir)
    LOG.info("Tile index vector: %s", tile_index_vector)

    tile_index_gdf = read_tile_index(tile_index_vector)
    url_field = detect_url_field(tile_index_gdf, explicit=args.tile_url_field)
    LOG.info("Using tile-index URL field: %s", url_field)

    records = select_tiles(tile_index_gdf, (west, east, south, north), url_field)
    LOG.info("Selected %d intersecting CUDEM tile(s)", len(records))

    download_file(args.spatial_meta_url, spatial_meta_zip, overwrite=args.overwrite)
    extract_zip(spatial_meta_zip, spatial_meta_dir, overwrite=args.overwrite)
    records = attach_metadata_paths(records, spatial_meta_dir, args.missing_meta_policy)
    write_tile_manifest(records, manifest_csv)
    LOG.info("Wrote tile manifest: %s", manifest_csv)

    tile_paths = download_tiles(records, tile_download_dir, overwrite=args.overwrite)
    LOG.info("Downloaded/located %d tile raster(s)", len(tile_paths))

    mosaic, transform, dem_profile = build_dem_mosaic(tile_paths, (west, east, south, north))
    dem_crs = dem_profile["crs"]
    LOG.info(
        "Built DEM mosaic: width=%d height=%d crs=%s",
        dem_profile["width"],
        dem_profile["height"],
        dem_crs,
    )

    support_gdf = collect_support_geometries(
        records,
        (west, east, south, north),
        str(dem_crs),
        args.missing_meta_policy,
    )
    ensure_dir(out_meta.parent)
    support_gdf.to_file(out_meta, driver="GPKG")
    LOG.info("Wrote clipped authoritative support polygons: %s", out_meta)

    out_shape = (dem_profile["height"], dem_profile["width"])
    support_mask = rasterize_support_mask(support_gdf, out_shape, transform)

    nodata_value = dem_profile.get("nodata", -999999.0)
    if nodata_value is None:
        nodata_value = -999999.0
        dem_profile["nodata"] = nodata_value

    mask_profile = dem_profile.copy()
    mask_profile.update(dtype="uint8", nodata=0, count=1)
    write_raster(out_mask, support_mask[np.newaxis, :, :], mask_profile)
    LOG.info("Wrote authoritative coverage mask: %s", out_mask)

    authoritative_base = apply_mask_to_mosaic(mosaic, support_mask, float(nodata_value))
    base_profile = dem_profile.copy()
    base_profile.update(dtype=str(authoritative_base.dtype), count=1)
    write_raster(out_authoritative, authoritative_base, base_profile)
    LOG.info("Wrote authoritative base raster: %s", out_authoritative)

    if not args.keep_zips:
        for zip_path in (tile_index_zip, spatial_meta_zip):
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                LOG.warning("Could not delete zip: %s", zip_path)

    LOG.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
