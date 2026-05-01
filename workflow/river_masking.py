"""River/ocean mask helpers used by bathy_main and guidance domain construction.

This module restores the explicit API imported by the active workflow.  The
functions either perform a deterministic raster/mask operation or fail clearly;
they do not silently discover alternate science inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from core.exec import run_command
from core.paths import ensure_dir


def _stable_hash_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _as_path(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value)


def _water_bool(arr: np.ndarray, *, water_max: float = 0.5, nodata: float | None = None) -> np.ndarray:
    data = np.asarray(arr)
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= data != float(nodata)
    return valid & (data <= float(water_max))


def _read_mask_like(mask_path: Path, template_ds: rasterio.io.DatasetReader, *, water_max: float = 0.5) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        if (
            src.width == template_ds.width
            and src.height == template_ds.height
            and src.transform == template_ds.transform
            and src.crs == template_ds.crs
        ):
            arr = src.read(1)
            return _water_bool(arr, water_max=water_max, nodata=src.nodata)
        dst = np.zeros((template_ds.height, template_ds.width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=1.0,
        )
        return _water_bool(dst, water_max=water_max, nodata=None)


def _write_uint8_mask(path: Path, mask: np.ndarray, profile: dict[str, Any]) -> Path:
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="DEFLATE")
    out_profile.pop("blockxsize", None)
    out_profile.pop("blockysize", None)
    if int(out_profile.get("width", 0)) >= 16 and int(out_profile.get("height", 0)) >= 16:
        out_profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        out_profile.update(tiled=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(np.asarray(mask, dtype="uint8"), 1)
    return path


def find_latest_waffles_mask(cache_root: Path) -> Path | None:
    root = Path(cache_root)
    if not root.exists():
        return None
    candidates = [p for p in root.rglob("*.tif") if "waffles" in p.name.lower() and p.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def stage_cached_waffles_mask(src: Path, dst: Path, logger: logging.Logger | None = None) -> Path:
    log = logger or logging.getLogger(__name__)
    src = Path(src)
    dst = Path(dst)
    if not src.exists() or src.stat().st_size <= 0:
        raise FileNotFoundError(f"waffles_mask_source_missing:{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = Path(os.path.relpath(str(src.resolve()), str(dst.parent.resolve())))
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)
    log.info("[WAFFLES] Staged mask %s -> %s", src, dst)
    return dst


def ensure_waffles_coastline_mask(
    cache_masks: Path,
    aoi: str,
    inc_arcsec: float = 1.0,
    want_nhd: bool = True,
    want_lakes: bool = False,
    prefix: str = "waffles_coastline",
    out_tif: Path | None = None,
    force: bool = False,
    log: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """Build or reuse the exact WAFFLES coastline mask requested by callers."""
    cache_masks = ensure_dir(Path(cache_masks))
    active_log = logger or logging.getLogger(__name__)
    params = {
        "aoi": str(aoi),
        "inc_arcsec": float(inc_arcsec),
        "want_nhd": bool(want_nhd),
        "want_lakes": bool(want_lakes),
        "prefix": str(prefix),
    }
    chash = _stable_hash_str(json.dumps(params, sort_keys=True, default=str))
    if out_tif is not None:
        out_tif_path = Path(out_tif)
        out_prefix = out_tif_path.with_suffix("")
    else:
        out_prefix = cache_masks / f"{prefix}_{chash}"
        out_tif_path = out_prefix.with_suffix(".tif")

    if force:
        for fp in (out_tif_path, out_tif_path.with_suffix(out_tif_path.suffix + ".aux.xml")):
            try:
                if fp.exists() or fp.is_symlink():
                    fp.unlink()
            except OSError:
                active_log.debug("[WAFFLES] Failed removing stale mask %s", fp, exc_info=True)

    if not force and out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        if log:
            active_log.info("[WAFFLES] Cache hit: %s", out_tif_path)
        return out_tif_path

    module = f"coastline:want_nhd={'true' if want_nhd else 'false'}:want_lakes={'true' if want_lakes else 'false'}"
    cmd = ["waffles", "-R", str(aoi), "-E", f"{float(inc_arcsec)}s", "-M", module, "-O", str(out_prefix), "-F", "GTiff"]
    if log:
        active_log.info("[WAFFLES] Command: %s", " ".join(cmd))
    rc, out, err = run_command(cmd, stream_stdout=False, stream_stderr=False)
    if rc != 0:
        raise RuntimeError(
            f"waffles coastline failed (rc={rc}): stderr={(err or '').strip()[:500]} stdout={(out or '').strip()[:500]}"
        )
    if out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        return out_tif_path
    existing = sorted(p.name for p in cache_masks.glob("*.tif"))
    raise RuntimeError(f"WAFFLES did not produce expected mask: {out_tif_path}. Existing in {cache_masks}: {existing[:50]}")


def waffles_water_fraction(mask_tif: Path, max_stride: int = 8, logger: logging.Logger | None = None) -> float:
    with rasterio.open(mask_tif) as ds:
        stride = max(1, int(max_stride))
        arr = ds.read(1, out_shape=(max(1, ds.height // stride), max(1, ds.width // stride)))
        water = _water_bool(arr, nodata=ds.nodata)
        valid = np.isfinite(arr)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            valid &= arr != float(ds.nodata)
        denom = int(np.count_nonzero(valid))
        return 0.0 if denom == 0 else float(np.count_nonzero(water) / denom)


def count_mask_water_pixels(mask_tif: Path, aoi_bounds_wgs84: Tuple[float, float, float, float] | None = None, *, water_max: float = 0.5) -> int | None:
    path = Path(mask_tif)
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        if aoi_bounds_wgs84 is not None:
            try:
                from rasterio.warp import transform_bounds
                from rasterio.windows import from_bounds

                left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *aoi_bounds_wgs84, densify_pts=21)
                win = from_bounds(left, bottom, right, top, ds.transform).round_offsets().round_lengths()
                arr = ds.read(1, window=win, boundless=True, fill_value=1)
            except Exception:
                arr = ds.read(1)
        else:
            arr = ds.read(1)
        return int(np.count_nonzero(_water_bool(arr, water_max=water_max, nodata=ds.nodata)))


def choose_waffles_mask_for_river(cfg: Any, report: dict[str, Any] | None = None, logger: logging.Logger | None = None) -> Path | None:
    """Choose an existing explicit mask path recorded on cfg/report; do not invent one."""
    log = logger or logging.getLogger(__name__)
    candidates: list[Any] = []
    for attr in (
        "river_water_support_mask",
        "waffles_water_mask",
        "waffles_ocean_mask",
        "water_mask",
        "ocean_mask",
    ):
        candidates.append(getattr(cfg, attr, None))
    if isinstance(report, dict):
        candidates.extend(
            [
                report.get("river", {}).get("outputs", {}).get("waffles_water_mask") if isinstance(report.get("river"), dict) else None,
                report.get("river", {}).get("outputs", {}).get("waffles_ocean_mask") if isinstance(report.get("river"), dict) else None,
                report.get("guidance_domains", {}).get("outputs", {}).get("river_water_support_mask") if isinstance(report.get("guidance_domains"), dict) else None,
            ]
        )
    for cand in candidates:
        path = _as_path(cand)
        if path is not None and path.exists() and path.stat().st_size > 0:
            log.info("[WAFFLES] River mask selected: %s", path)
            return path
    cache_root = _as_path(getattr(cfg, "cache_root", None))
    if cache_root is not None:
        latest = find_latest_waffles_mask(cache_root / "masks")
        if latest is not None:
            log.info("[WAFFLES] River mask selected from cache: %s", latest)
            return latest
    return None


def determine_effective_methods_from_waffles(cfg: Any, report: dict[str, Any] | None = None, logger: logging.Logger | None = None) -> tuple[list[str], dict[str, Any]]:
    """Return configured methods plus auditable WAFFLES-water evidence."""
    log = logger or logging.getLogger(__name__)
    raw_methods = getattr(cfg, "methods", None) or getattr(cfg, "method", None) or []
    if isinstance(raw_methods, str):
        methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    else:
        methods = [str(m) for m in raw_methods]
    mask = choose_waffles_mask_for_river(cfg, report or {}, logger=log)
    frac = None
    if mask is not None:
        try:
            frac = waffles_water_fraction(mask)
        except Exception as exc:
            log.debug("[WAFFLES] Water fraction unavailable for %s: %s", mask, exc, exc_info=True)
    meta = {"waffles_mask": str(mask) if mask is not None else None, "water_fraction_sample": frac, "methods": methods}
    if isinstance(report, dict):
        report.setdefault("waffles", {}).update(meta)
    return methods, meta


def clip_channel_mask_for_estuary(
    channel_mask_tif: Path,
    cfg: Any,
    *,
    ocean_mask_path: Path | str | None = None,
    report: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[int, Path | None]:
    """Remove ocean-connected cells from a river channel mask and write an estuary mask.

    This is intentionally conservative: only cells that are already in the
    channel mask and also water in the supplied ocean mask are clipped.  If no
    ocean mask is supplied, the function writes an all-zero estuary mask and
    leaves the channel unchanged.
    """
    log = logger or logging.getLogger(__name__)
    channel_path = Path(channel_mask_tif)
    if not channel_path.exists():
        raise FileNotFoundError(f"channel_mask_missing:{channel_path}")
    out_path = channel_path.parent / "estuary_clip_mask.tif"
    transition_path = channel_path.parent / "estuary_transition_mask.tif"

    with rasterio.open(channel_path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        channel = np.asarray(arr) > 0
        ocean_path = _as_path(ocean_mask_path)
        if ocean_path is None or not ocean_path.exists():
            estuary = np.zeros(channel.shape, dtype=bool)
        else:
            ocean_water = _read_mask_like(ocean_path, ds)
            estuary = channel & ocean_water

    removed = int(np.count_nonzero(estuary))
    if removed > 0:
        arr_out = np.asarray(arr).copy()
        arr_out[estuary] = 0
        out_profile = profile.copy()
        out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="DEFLATE")
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)
        if int(out_profile.get("width", 0)) >= 16 and int(out_profile.get("height", 0)) >= 16:
            out_profile.update(tiled=True, blockxsize=256, blockysize=256)
        else:
            out_profile.update(tiled=False)
        with rasterio.open(channel_path, "w", **out_profile) as dst:
            dst.write((arr_out > 0).astype("uint8"), 1)

    _write_uint8_mask(out_path, estuary.astype("uint8"), profile)
    _write_uint8_mask(transition_path, estuary.astype("uint8"), profile)
    if isinstance(report, dict):
        report.setdefault("river", {}).setdefault("masking", {})["estuary_clip"] = {
            "channel_mask": str(channel_path),
            "ocean_mask": str(ocean_mask_path) if ocean_mask_path else None,
            "estuary_clip_mask": str(out_path),
            "removed_pixels": removed,
        }
    log.info("[ESTUARY] Clipped %d ocean-connected pixels from channel mask: %s", removed, channel_path)
    return removed, out_path

def build_hydraulic_estuary_hint_mask(*, cfg: Any, channel: Any, transform: Any, crs: Any, px_size_m: float, logger: logging.Logger | None = None) -> dict[str, Any]:
    arr = np.asarray(channel)
    return {
        "status": "not_applied",
        "reason": "no_explicit_hydraulic_estuary_inputs",
        "shape": [int(v) for v in arr.shape[:2]],
        "px_size_m": float(px_size_m),
    }


def apply_estuary_first_channel_domain(channel_mask: Any, estuary_transition: Any | None = None, *args: Any, **kwargs: Any) -> Any:
    channel = np.asarray(channel_mask)
    if estuary_transition is None:
        return channel
    estuary = np.asarray(estuary_transition).astype(bool)
    if estuary.shape != channel.shape:
        raise ValueError(f"estuary/channel shape mismatch: estuary={estuary.shape} channel={channel.shape}")
    out = channel.copy()
    out[estuary] = 0
    return out


__all__ = [
    "apply_estuary_first_channel_domain",
    "build_hydraulic_estuary_hint_mask",
    "choose_waffles_mask_for_river",
    "clip_channel_mask_for_estuary",
    "count_mask_water_pixels",
    "determine_effective_methods_from_waffles",
    "ensure_waffles_coastline_mask",
    "find_latest_waffles_mask",
    "stage_cached_waffles_mask",
    "waffles_water_fraction",
]
