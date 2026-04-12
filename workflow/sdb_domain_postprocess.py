from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import reproject

log = logging.getLogger(__name__)


def _load_manifest(sdb_dir: Path) -> tuple[Path, Dict[str, Any]]:
    manifest_path = sdb_dir / "artifacts_sdb.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"SDB manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("SDB manifest must be a JSON object")
    return manifest_path, data


def _resolve_manifest_path(base_dir: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    try:
        return p if p.exists() else None
    except OSError:
        return None


def _rel_to(base_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _align_binary_mask(mask_path: Path, ref_ds: rasterio.DatasetReader, *, water_is_zero: bool = False) -> np.ndarray:
    with rasterio.open(mask_path) as ds:
        src_arr = ds.read(1)
        src_nodata = ds.nodata
        if src_nodata in (0, 1):
            src_nodata = None
        dst_nodata = 255 if water_is_zero else 0
        dst = np.full((ref_ds.height, ref_ds.width), dst_nodata, dtype=np.uint8)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            src_nodata=src_nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=dst_nodata,
            resampling=Resampling.nearest,
        )
    if water_is_zero:
        return dst == 0
    return dst > 0


def _write_land_water_mask(path: Path, ref_ds: rasterio.DatasetReader, water_mask: np.ndarray) -> Path:
    profile = ref_ds.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=1, compress="deflate")
    arr = np.where(water_mask, 0, 1).astype(np.uint8)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _mask_raster_in_place(path: Path, keep_mask: np.ndarray) -> bool:
    if not path.exists():
        return False
    with rasterio.open(path, "r+") as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        kind = np.dtype(ds.dtypes[0]).kind
        if kind in ("f",):
            fill = np.array(nodata if nodata is not None else np.nan, dtype=arr.dtype)
        elif kind in ("i", "u"):
            fill = np.array(nodata if nodata is not None else 0, dtype=arr.dtype)
        else:
            fill = np.array(0, dtype=arr.dtype)
        out = np.where(keep_mask, arr, fill)
        ds.write(out, 1)
    return True


def _clip_guide_points_to_mask(guide_points_path: Path, depth_raster: Path, keep_mask: np.ndarray, *, logger: logging.Logger) -> Optional[Path]:
    try:
        import geopandas as gpd
    except Exception:
        log.debug("_clip_guide_points_to_mask: suppressed exception", exc_info=True)
        return None
    if not guide_points_path.exists():
        return None
    try:
        gdf = gpd.read_file(guide_points_path)
    except Exception:
        logger.debug("Failed reading SDB guide points for estuary-domain clipping.", exc_info=True)
        return None
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return guide_points_path
    with rasterio.open(depth_raster) as ds:
        if gdf.crs is not None and ds.crs is not None and str(gdf.crs) != str(ds.crs):
            try:
                gdf = gdf.to_crs(ds.crs)
            except Exception:
                logger.debug("Failed reprojecting SDB guide points to template CRS.", exc_info=True)
                return guide_points_path
        keep_rows = []
        for idx, geom in zip(gdf.index, gdf.geometry):
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                r, c = rowcol(ds.transform, geom.x, geom.y)
            except Exception:
                log.debug("_clip_guide_points_to_mask: suppressed exception", exc_info=True)
                continue
            rr = int(r)
            cc = int(c)
            if 0 <= rr < ds.height and 0 <= cc < ds.width and bool(keep_mask[rr, cc]):
                keep_rows.append(idx)
        gdf = gdf.loc[keep_rows].copy()
        gdf.to_file(guide_points_path, driver="GPKG")
    return guide_points_path


def load_sdb_artifacts_into_report(*, sdb_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    _, data = _load_manifest(sdb_dir)
    sdb = report.setdefault("sdb", {})
    sdb["artifacts"] = dict(data)
    active = _resolve_manifest_path(sdb_dir, data.get("sdb_guidance_active") or data.get("depth_raster"))
    raw = _resolve_manifest_path(sdb_dir, data.get("raw_prediction_raster"))
    locked = _resolve_manifest_path(sdb_dir, data.get("sdb_locked_guidance_raster"))
    if active is not None:
        sdb["guidance_active"] = str(active)
        report.setdefault("outputs", {})["sdb_guidance_active"] = str(active)
    if raw is not None:
        sdb["raw_prediction_raster"] = str(raw)
        report.setdefault("outputs", {})["sdb_raw_prediction_raster"] = str(raw)
    if locked is not None:
        sdb["locked_guidance_raster"] = str(locked)
        report.setdefault("outputs", {})["sdb_locked_guidance_raster"] = str(locked)
    if isinstance(data.get("guidance_mode"), str):
        sdb["guidance_mode"] = data.get("guidance_mode")
    if isinstance(data.get("depth_raster_role"), str):
        sdb["depth_raster_role"] = data.get("depth_raster_role")
    guidance_manifest = _resolve_manifest_path(sdb_dir, data.get("guidance_manifest"))
    if guidance_manifest is not None:
        sdb["guidance_manifest"] = str(guidance_manifest)
    return data


def apply_estuary_aware_sdb_domain(*, sdb_dir: Path, ocean_mask_path: Path, estuary_mask_path: Path, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    active_log = logger or log
    sdb_dir = Path(sdb_dir)
    if not sdb_dir.exists():
        return None
    manifest_path, manifest = _load_manifest(sdb_dir)
    depth_path = _resolve_manifest_path(sdb_dir, manifest.get("depth_raster"))
    if depth_path is None or not depth_path.exists():
        return None
    if not Path(ocean_mask_path).exists() or not Path(estuary_mask_path).exists():
        return None

    artifact_keys_to_mask = (
        "depth_raster",
        "confidence_raster",
        "provenance_raster",
        "guidance_weight_raster",
        "trusted_interior_raster",
        "admissibility_raster",
        "regime_class_raster",
        "lower_bound_raster",
        "upper_bound_raster",
    )

    with rasterio.open(depth_path) as ref_ds:
        ocean_water = _align_binary_mask(Path(ocean_mask_path), ref_ds, water_is_zero=True)
        estuary_water = _align_binary_mask(Path(estuary_mask_path), ref_ds, water_is_zero=False)
        sdb_water_domain = np.asarray(ocean_water | estuary_water, dtype=bool)
        domain_mask_path = depth_path.with_name(depth_path.stem + "_domain_mask.tif")
        _write_land_water_mask(domain_mask_path, ref_ds, sdb_water_domain)

    masked_paths: Dict[str, str] = {}
    for key in artifact_keys_to_mask:
        artifact_path = _resolve_manifest_path(sdb_dir, manifest.get(key))
        if artifact_path is None:
            continue
        if _mask_raster_in_place(artifact_path, sdb_water_domain):
            masked_paths[key] = _rel_to(sdb_dir, artifact_path)

    gp = _resolve_manifest_path(sdb_dir, manifest.get("guide_points"))
    if gp is not None:
        clipped_gp = _clip_guide_points_to_mask(gp, depth_path, sdb_water_domain, logger=active_log)
        if clipped_gp is not None:
            manifest["guide_points"] = _rel_to(sdb_dir, clipped_gp)

    manifest["land_mask"] = _rel_to(sdb_dir, domain_mask_path)
    manifest["sdb_domain_mask"] = _rel_to(sdb_dir, domain_mask_path)
    manifest.setdefault("domain_policy", {})
    manifest["domain_policy"].update({
        "type": "ocean_plus_estuary",
        "ocean_mask": str(Path(ocean_mask_path)),
        "estuary_clip_mask": str(Path(estuary_mask_path)),
        "water_semantics": "0=water,1=land",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    receipt = {
        "status": "applied",
        "depth_raster": str(depth_path),
        "ocean_mask": str(Path(ocean_mask_path)),
        "estuary_clip_mask": str(Path(estuary_mask_path)),
        "domain_mask": str(domain_mask_path),
        "domain_pixels": int(np.count_nonzero(sdb_water_domain)),
        "ocean_pixels": int(np.count_nonzero(ocean_water)),
        "estuary_pixels": int(np.count_nonzero(estuary_water)),
        "masked_artifacts": masked_paths,
    }
    receipt_path = sdb_dir / "sdb_estuary_domain_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    active_log.info("[SDB][DOMAIN] Applied ocean+estuary SDB domain mask: %s", domain_mask_path)
    return receipt
