
"""Helpers for WAFFLES-based river/coastal masking and estuary clipping."""
from __future__ import annotations
import numpy as np

import json
import pandas as pd
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.exec import run_command

log = logging.getLogger(__name__)

try:
    from core.json_io import write_json
except (ImportError, AttributeError):
    def write_json(path, obj):
        Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


try:
    from scipy.ndimage import distance_transform_edt, binary_dilation, binary_opening
except ImportError:  # pragma: no cover - runtime dependency expected in normal envs
    distance_transform_edt = binary_dilation = binary_opening = None


try:
    from core.hashing import stable_hash_json
except (ImportError, AttributeError):
    import hashlib, json as _json
    def stable_hash_json(obj):
        return hashlib.sha256(_json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

try:
    from core.fs import ensure_dir
except (ImportError, AttributeError):
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)






def _estimate_pixel_size_m(transform, crs, *, height: int | None = None) -> float:
    """Estimate pixel size in meters, including geographic grids."""
    try:
        px_x = abs(float(transform.a))
        px_y = abs(float(transform.e))
        if px_x <= 0.0 and px_y <= 0.0:
            return 1.0
        if bool(getattr(crs, "is_geographic", False)):
            row_count = int(height) if height is not None and int(height) > 0 else 1
            center_y = float(transform.f) - (0.5 * row_count * px_y)
            lat_rad = np.deg2rad(center_y)
            m_per_deg_lat = 111132.92 - (559.82 * np.cos(2.0 * lat_rad)) + (1.175 * np.cos(4.0 * lat_rad))
            m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
            sx = px_x * abs(m_per_deg_lon)
            sy = px_y * abs(m_per_deg_lat)
        else:
            sx = px_x
            sy = px_y
        vals = [v for v in (sx, sy) if np.isfinite(v) and v > 0.0]
        return float(min(vals)) if vals else 1.0
    except Exception:
        log.debug("_estimate_pixel_size_m: suppressed exception", exc_info=True)
        return 1.0




def _channel_distance_from_seed(channel_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Shortest 8-neighbor step distance inside a channel mask from seed pixels.

    Returns np.inf outside the channel or when no seed exists. Distances are in pixel
    steps so callers can scale by the target pixel size.
    """
    from collections import deque

    channel = np.asarray(channel_mask, dtype=bool)
    seeds = np.asarray(seed_mask, dtype=bool) & channel
    dist = np.full(channel.shape, np.inf, dtype=np.float32)
    if not np.any(channel) or not np.any(seeds):
        return dist
    q = deque()
    seed_rows, seed_cols = np.nonzero(seeds)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        dist[r, c] = 0.0
        q.append((r, c))
    nrows, ncols = channel.shape
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        r, c = q.popleft()
        base = float(dist[r, c])
        for dr, dc in nbrs:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols or (not channel[rr, cc]):
                continue
            step = 1.41421356 if (dr != 0 and dc != 0) else 1.0
            cand = base + step
            if cand + 1e-6 < float(dist[rr, cc]):
                dist[rr, cc] = cand
                q.append((rr, cc))
    return dist

def _mask_to_geometry(mask, transform):
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return None
    geoms = [shape(geom) for geom, value in shapes(mask_bool.astype("uint8"), mask=mask_bool, transform=transform) if int(value) == 1]
    if not geoms:
        return None
    return unary_union(geoms)


def waffles_preflight(*, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Return a lightweight WAFFLES runtime check before domain inference.

    This does not prove a given AOI will succeed, but it catches the common
    orchestration failures early and surfaces clearer diagnostics in logs.
    """
    log = logger or logging.getLogger(__name__)
    exe = shutil.which('waffles')
    info: Dict[str, Any] = {
        'available': bool(exe),
        'executable': str(exe) if exe else None,
        'run_command_bound': callable(run_command),
    }
    if not exe:
        info['error'] = 'waffles_not_found_on_path'
        return info
    try:
        rc, out, err = run_command([exe, '--help'], stream_stdout=False, stream_stderr=False)
        info['help_rc'] = int(rc)
        info['help_ok'] = bool(rc == 0)
        if rc != 0:
            info['error'] = 'waffles_help_failed'
            info['stderr_tail'] = (err or '').strip()[-500:]
            info['stdout_tail'] = (out or '').strip()[-500:]
    except Exception as exc:
        log.debug('WAFFLES preflight failed.', exc_info=True)
        info['error'] = f'waffles_preflight_exception: {exc}'
        info['help_ok'] = False
    return info

def choose_waffles_mask_for_river(cfg: Any, report: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    for attr in ("waffles_with_nhd_mask", "water_domain_mask_for_final"):
        p = getattr(cfg, attr, None)
        if not p:
            continue
        try:
            pp = Path(p)
        except (TypeError, ValueError, OSError):
            log.debug("Invalid WAFFLES mask path on cfg.%s: %r", attr, p, exc_info=True)
            continue
        try:
            if pp.exists() and pp.stat().st_size > 0:
                return pp
        except OSError:
            log.debug("Failed to stat WAFFLES mask candidate: %s", pp, exc_info=True)
    outputs = report.get("river", {}).get("outputs", {}) if isinstance(report, dict) else {}
    wm = outputs.get("waffles_water_mask")
    if wm:
        try:
            pp = Path(wm)
            if pp.exists() and pp.stat().st_size > 0:
                return pp
        except (TypeError, ValueError, OSError):
            log.debug("Invalid river report waffles_water_mask: %r", wm, exc_info=True)
    return None


def count_mask_water_pixels(mask_tif: Path, aoi_bounds_wgs84: Tuple[float, float, float, float], *, water_max: float = 0.5) -> Optional[int]:
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer

        xmin, xmax, ymin, ymax = aoi_bounds_wgs84
        with rasterio.open(mask_tif) as ds:
            if ds.crs and ds.crs.to_epsg() not in (None, 4326):
                tx = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
                xmin2, ymin2 = tx.transform(xmin, ymin)
                xmax2, ymax2 = tx.transform(xmax, ymax)
                xmin, xmax = min(xmin2, xmax2), max(xmin2, xmax2)
                ymin, ymax = min(ymin2, ymax2), max(ymin2, ymax2)

            bxmin, bymin, bxmax, bymax = ds.bounds
            ixmin, ixmax = max(xmin, bxmin), min(xmax, bxmax)
            iymin, iymax = max(ymin, bymin), min(ymax, bymax)
            if ixmin >= ixmax or iymin >= iymax:
                return 0

            win = from_bounds(ixmin, iymin, ixmax, iymax, transform=ds.transform)
            nd = ds.nodata
            nodata_is_zero = (nd is not None) and (abs(float(nd)) < 0.5)
            nodata_is_absent = (nd is None)

            if nodata_is_zero or nodata_is_absent:
                arr = ds.read(1, window=win, masked=False)
                if arr.size == 0:
                    return 0
                return int(np.count_nonzero(arr <= water_max))
            arr = ds.read(1, window=win, masked=True)
            if arr.size == 0:
                return 0
            water = (arr <= water_max)
            if hasattr(water, 'filled'):
                water = water.filled(False)
            return int(np.count_nonzero(water))
    except (OSError, ValueError, RuntimeError, ImportError):
        return None


def stage_cached_waffles_mask(src: Path, dst: Path, *, logger: Optional[logging.Logger] = None) -> Path:
    log = logger or logging.getLogger(__name__)
    ensure_dir(dst.parent)
    try:
        if dst.exists():
            return dst
        try:
            os.symlink(src, dst)
            return dst
        except OSError:
            shutil.copy2(src, dst)
            return dst
    except OSError as exc:
        log.warning("[WAFFLES] Failed to stage cached mask %s -> %s: %s", src, dst, exc, exc_info=True)
        return src


def find_latest_waffles_mask(cache_root: Path) -> Optional[Path]:
    masks_dir = cache_root / 'masks'
    if not masks_dir.is_dir():
        return None

    def _newest(candidates):
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            return None
        return max(existing, key=lambda p: p.stat().st_mtime)

    all_masks = list(masks_dir.glob('waffles_coastline*.tif'))
    if not all_masks:
        return None
    nhd = _newest([p for p in all_masks if 'with_nhd' in p.name])
    if nhd is not None:
        return nhd
    ocean = _newest([p for p in all_masks if 'ocean_only' in p.name])
    if ocean is not None:
        return ocean
    return _newest(all_masks)


def ensure_waffles_coastline_mask(cache_masks: Path, aoi: str, *, inc_arcsec: float = 1.0, want_nhd: bool = True, want_lakes: bool = False, prefix: str = 'waffles_coastline', out_tif: Optional[Path] = None, force: bool = False, log_enabled: bool = True, logger: Optional[logging.Logger] = None) -> Path:
    log = logger or logging.getLogger(__name__)
    ensure_dir(cache_masks)
    params = dict(aoi=str(aoi), inc_arcsec=float(inc_arcsec), want_nhd=bool(want_nhd), want_lakes=bool(want_lakes), prefix=str(prefix))
    chash = stable_hash_json(params)
    if out_tif is not None:
        out_tif_path = Path(out_tif)
        out_prefix = out_tif_path.with_suffix('')
    else:
        out_prefix = cache_masks / f'{prefix}_{chash}'
        out_tif_path = out_prefix.with_suffix('.tif')
    if force:
        for fp in [out_tif_path, out_tif_path.with_suffix(out_tif_path.suffix + '.aux.xml')]:
            try:
                if fp.exists():
                    fp.unlink()
            except OSError:
                log.debug('Failed to remove stale WAFFLES output: %s', fp, exc_info=True)
    if (not force) and out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        if log_enabled:
            log.info('[WAFFLES] Cache hit: %s', out_tif_path)
        return out_tif_path
    module = f"coastline:want_nhd={'true' if want_nhd else 'false'}:want_lakes={'true' if want_lakes else 'false'}"
    cmd = ['waffles', '-R', str(aoi), '-E', f'{inc_arcsec}s', '-M', module, '-O', str(out_prefix), '-F', 'GTiff']
    if log_enabled:
        log.info('[WAFFLES] Command: %s', ' '.join(cmd))
    rc, out, err = run_command(cmd, stream_stdout=False, stream_stderr=False)
    if rc != 0:
        raise RuntimeError(f"waffles coastline failed (rc={rc}): stderr={(err or '').strip()[:500]} stdout={(out or '').strip()[:500]}")
    if out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        return out_tif_path
    existing = sorted([p.name for p in cache_masks.glob('*.tif')])
    raise RuntimeError(f"WAFFLES did not produce expected mask: {out_tif_path}. Existing in {cache_masks}: {existing[:50]}")


def waffles_water_fraction(mask_tif: Path, *, max_stride: int = 8, logger: Optional[logging.Logger] = None) -> float:
    log = logger or logging.getLogger(__name__)
    try:
        import numpy as np
        import rasterio
        if mask_tif is None or (not Path(mask_tif).exists()):
            return 0.0
        with rasterio.open(str(mask_tif)) as ds:
            nd = ds.nodata
            data = ds.read(1, masked=False)
        h, w = data.shape
        stride = max(1, min(max_stride, h // 128, w // 128))
        data = data[::stride, ::stride]
        if nd is not None and np.isfinite(float(nd)) and abs(float(nd)) > 0.5:
            valid = (data != nd)
        else:
            valid = np.ones(data.shape, dtype=bool)
        total = int(np.count_nonzero(valid))
        if total == 0:
            return 0.0
        water = int(np.count_nonzero((data == 0) & valid))
        return float(water) / float(total)
    except (OSError, ValueError, RuntimeError, ImportError):
        log.debug('WAFFLES water fraction check failed.', exc_info=True)
        return 0.0


def determine_effective_methods_from_waffles(cfg: Any, report: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Tuple[List[str], Dict[str, Any]]:
    log = logger or logging.getLogger(__name__)
    requested = [m.strip().lower() for m in (cfg.methods or []) if m.strip()] or ['sdb', 'river', 'fuse']
    req_set = set(requested)
    want_sdb = 'sdb' in req_set
    want_river = 'river' in req_set
    want_fuse = 'fuse' in req_set or (want_sdb and want_river)
    if not (want_sdb or want_river or want_fuse):
        return requested, {'requested': requested, 'effective': requested, 'waffles': {}}
    cache_masks_shared = cfg.cache_root.resolve() / 'masks'
    ensure_dir(cache_masks_shared)
    force_masks = cfg.force_waffles_masks
    preflight = waffles_preflight(logger=log)
    report.setdefault('domain_inference', {}).setdefault('waffles_preflight', {}).update(preflight)
    if not preflight.get('available', False):
        raise RuntimeError('WAFFLES preflight failed: waffles executable not found on PATH')
    if preflight.get('help_ok') is False:
        raise RuntimeError(f"WAFFLES preflight failed: {preflight.get('error') or 'help_failed'}")
    ocean_mask_cache = ensure_waffles_coastline_mask(cache_masks_shared, cfg.aoi, inc_arcsec=float(cfg.waffles_inc_arcsec or 1.0), want_nhd=False, want_lakes=False, prefix='waffles_coastline_ocean_only', force=force_masks, logger=log)
    cache_masks_run = Path(cfg.derived_cache_root) / 'masks'
    ocean_mask = stage_cached_waffles_mask(ocean_mask_cache, cache_masks_run / 'waffles_coastline_ocean_only.tif', logger=log)
    nhd_mask_cache = None
    nhd_mask = None
    ocean_frac = waffles_water_fraction(ocean_mask, logger=log) if ocean_mask else 0.0
    nhd_frac = None
    if ocean_mask and ocean_frac == 0.0 and want_sdb:
        log.warning('[DOMAIN] WAFFLES ocean mask has 0%% water (all-land). If the AOI contains ocean/coast, the cached mask may be stale or waffles may have failed silently. Try --force-waffles-masks to regenerate, or --waffles-min-water-fraction=0 to bypass the gate and run SDB regardless. Mask: %s', ocean_mask)
    min_frac = float(cfg.waffles_min_water_fraction)
    run_sdb = bool(want_sdb and (ocean_frac >= min_frac))
    # River viability is determined later by extracted river-network polygons and the canonical
    # with-NHD mask built from them. Do not gate river execution here on a raw WAFFLES NHD mask.
    run_river = bool(want_river)
    effective: List[str] = []
    skipped: Dict[str, str] = {}
    if want_sdb:
        if run_sdb:
            effective.append('sdb')
        else:
            skipped['sdb'] = 'no_ocean_water_detected_by_waffles'
    if want_river:
        if run_river:
            effective.append('river')
        else:
            skipped['river'] = 'no_nhd_water_detected_by_waffles'
    if want_fuse:
        if effective:
            effective.append('fuse')
        else:
            skipped['fuse'] = 'no_upstream_sources'
    cfg.waffles_ocean_mask = Path(ocean_mask) if ocean_mask else None
    cfg.waffles_with_nhd_mask = None
    cfg.ocean_domain_mask_for_fusion = Path(ocean_mask) if ocean_mask else None
    cfg.water_domain_mask_for_final = None
    meta = {'requested': requested, 'effective': effective, 'skipped': skipped, 'waffles': {'ocean_only_mask': str(ocean_mask) if ocean_mask else None, 'with_nhd_mask': None, 'ocean_only_mask_cache': str(ocean_mask_cache) if ocean_mask_cache is not None else None, 'with_nhd_mask_cache': None, 'ocean_water_fraction': ocean_frac, 'with_nhd_water_fraction': nhd_frac, 'min_water_fraction': min_frac}}
    report.setdefault('domain_inference', {}).update(meta)
    return effective, meta


def build_hydraulic_estuary_hint_mask(*, cfg: Any, channel, transform, crs, px_size_m: float, logger: Optional[logging.Logger] = None):
    log = logger or logging.getLogger(__name__)
    import importlib
    import numpy as np
    try:
        gpd = importlib.import_module("geopandas")
    except ImportError:
        gpd = None
    try:
        rio_features = importlib.import_module("rasterio.features")
    except ImportError:
        rio_features = None

    meta = {
        "reason": "ok",
        "flagged_reaches": 0,
        "backwater_slope_reaches": 0,
        "near_mouth_reaches": 0,
        "dist_to_mouth_field": None,
        "slope_field": None,
    }
    zero = np.zeros_like(channel, dtype="uint8")
    try:
        river_gpkg = getattr(cfg, 'river_network_gpkg', None) or getattr(cfg, 'river_network', None)
        if not river_gpkg:
            root = getattr(cfg, 'derived_cache_root', None)
            if root:
                cand = Path(root) / 'river' / 'work' / 'river_network.gpkg'
                if cand.exists():
                    river_gpkg = cand
        if not river_gpkg:
            meta['reason'] = 'no_river_network'
            return zero, meta
        gpkg = Path(river_gpkg)
        if not gpkg.exists():
            meta['reason'] = 'river_network_missing'
            return zero, meta
        if gpd is None or rio_features is None:
            meta['reason'] = 'deps_unavailable'
            return zero, meta
        layer = None
        try:
            import fiona
            layers = list(fiona.listlayers(gpkg))
            for cand in ('rivers_clip', 'rivers_aoi', 'rivers', 'graph_edges'):
                if cand in layers:
                    layer = cand
                    break
        except Exception:
            log.debug("build_hydraulic_estuary_hint_mask: suppressed exception", exc_info=True)
            layer = None
        gdf = gpd.read_file(gpkg, layer=layer) if layer else gpd.read_file(gpkg)
        meta['river_network_layer'] = layer or 'default'
        if gdf.empty:
            meta['reason'] = 'empty_river_network'
            return zero, meta

        cols = list(gdf.columns)
        lower_map = {str(c).lower(): c for c in cols}
        slope_cands = [getattr(cfg, 'river_manning_slope_field', None), 'slope_mpm', 'slope', 'Slope', 'SLOPE']
        mouth_cands = [
            getattr(cfg, 'river_manning_dist_to_mouth_field', None),
            'dist_to_mouth_km', 'dist_to_mouth_m', 'distance_to_mouth_m', 'DistToMouth', 'MOUTH_DIST'
        ]
        def pick(cands):
            for c in cands:
                if not c:
                    continue
                if c in cols:
                    return c
                lc = lower_map.get(str(c).lower())
                if lc is not None:
                    return lc
            return None
        slope_field = pick(slope_cands)
        mouth_field = pick(mouth_cands)
        meta['slope_field'] = slope_field
        meta['dist_to_mouth_field'] = mouth_field

        selected_idx = np.zeros(len(gdf), dtype=bool)
        if mouth_field is not None:
            mouth_max_km = float(getattr(cfg, 'river_manning_dist_to_mouth_km_max', None) or ((float(getattr(cfg, 'estuary_transition_m', 500.0) or 500.0) * 4.0) / 1000.0))
            series = pd.to_numeric(gdf[mouth_field], errors='coerce')
            if 'km' not in str(mouth_field).lower():
                series = series / 1000.0
            near_mask = series.fillna(1e18).astype(float) <= mouth_max_km
            meta['near_mouth_reaches'] = int(np.count_nonzero(near_mask))
            selected_idx |= np.asarray(near_mask, dtype=bool)
        if slope_field is not None:
            slope_thresh = float(getattr(cfg, 'river_manning_backwater_slope_thresh', 1e-4) or 1e-4)
            slope_mask = pd.to_numeric(gdf[slope_field], errors='coerce').fillna(1e9).astype(float) <= slope_thresh
            meta['backwater_slope_reaches'] = int(np.count_nonzero(slope_mask))
            selected_idx |= np.asarray(slope_mask, dtype=bool)
        meta['flagged_reaches'] = int(np.count_nonzero(selected_idx))
        if not np.any(selected_idx):
            meta['reason'] = 'no_flagged_reaches'
            return zero, meta

        geoms = []
        for geom in gdf.loc[selected_idx, 'geometry']:
            if geom is None or getattr(geom, 'is_empty', False):
                continue
            geoms.append((geom, 1))
        if not geoms:
            meta['reason'] = 'no_valid_geometries'
            return zero, meta
        hint = rio_features.rasterize(shapes=geoms, out_shape=channel.shape, transform=transform, fill=0, dtype='uint8')
        hint = ((hint > 0) & np.asarray(channel, dtype=bool)).astype('uint8')
        if int(hint.sum()) <= 0:
            meta['reason'] = 'empty_rasterized_hint'
        return hint, meta
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError, TypeError, KeyError):
        log.debug('[ESTUARY-CLIP] Hydraulic hint mask failed', exc_info=True)
        meta['reason'] = 'error'
        return zero, meta




def apply_estuary_first_channel_domain(channel_mask, estuary_mask):
    """Return the structured river generation domain after estuary exclusion.

    This helper keeps river scaffold / bank / centerline generation bounded to the
    retained fluvial corridor instead of letting estuary pixels participate and be
    removed later. It is intentionally simple and deterministic so downstream code
    can reuse the same estuary-first mask contract.
    """
    import numpy as np

    channel = np.asarray(channel_mask) > 0
    estuary = np.asarray(estuary_mask) > 0
    return (channel & (~estuary)).astype(np.uint8)


def _write_debug_mask(path: Path, profile: Dict[str, Any], mask: np.ndarray) -> None:
    import rasterio

    out_prof = profile.copy()
    out_prof.pop('blockxsize', None)
    out_prof.pop('blockysize', None)
    out_prof.pop('tiled', None)
    out_prof.update(dtype='uint8', nodata=0, compress='deflate')
    with rasterio.open(path, 'w', **out_prof) as dst:
        dst.write((np.asarray(mask) > 0).astype('uint8'), 1)


def _mask_bbox_summary(mask: np.ndarray, transform) -> Dict[str, Any]:
    import numpy as np
    import rasterio.transform

    ys, xs = np.where(np.asarray(mask) > 0)
    if ys.size == 0:
        return {'pixels': 0, 'bbox': None}
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    x0, y0 = rasterio.transform.xy(transform, r0, c0, offset='center')
    x1, y1 = rasterio.transform.xy(transform, r1, c1, offset='center')
    return {
        'pixels': int(ys.size),
        'bbox': {
            'row_min': r0,
            'row_max': r1,
            'col_min': c0,
            'col_max': c1,
            'x_min_center': float(min(x0, x1)),
            'x_max_center': float(max(x0, x1)),
            'y_min_center': float(min(y0, y1)),
            'y_max_center': float(max(y0, y1)),
        },
    }


def clip_channel_mask_for_estuary(channel_mask_tif: Path, cfg: Any, *, ocean_mask_path: Optional[Path], report: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[int, Optional[Path]]:
    log = logger or logging.getLogger(__name__)
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    channel_mask_tif = Path(channel_mask_tif)
    if not channel_mask_tif.exists():
        return 0, None
    with rasterio.open(channel_mask_tif) as ds:
        channel = ds.read(1).astype('uint8')
        prof = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
        px_size_m = max(_estimate_pixel_size_m(transform, crs, height=ds.height), 1e-6)
    if not np.any(channel > 0):
        return 0, None

    channel_before = (channel > 0)
    estuary_mask = np.zeros_like(channel_before, dtype=bool)
    width_estuary = np.zeros_like(channel_before, dtype=bool)
    hydraulic_hint = np.zeros_like(channel_before, dtype=bool)
    slope_near_widening = np.zeros_like(channel_before, dtype=bool)
    ocean_water = np.zeros_like(channel_before, dtype=bool)
    ocean_touch = np.zeros_like(channel_before, dtype=bool)
    ocean_connected = np.zeros_like(channel_before, dtype=bool)
    near_mouth_corridor = np.zeros_like(channel_before, dtype=bool)
    width_support = np.zeros_like(channel_before, dtype=bool)
    constrained_corridor = np.zeros_like(channel_before, dtype=bool)
    local_width = np.zeros_like(channel, dtype='float32')
    valid_widths = np.array([], dtype='float32')
    transition_m = max(float(getattr(cfg, 'estuary_transition_m', 500.0) or 500.0), 3.0 * px_size_m)
    corridor_limit_m = transition_m
    meta = {'signals': {}}

    try:
        chan_bool = channel_before
        d_bank = distance_transform_edt(chan_bool, sampling=px_size_m).astype('float32')
        local_width = (2.0 * d_bank).astype('float32')
        valid_widths = local_width[chan_bool & (local_width > 0)]
        if valid_widths.size > 100:
            width_ratio_thresh = float(getattr(cfg, 'estuary_width_ratio_thresh', 3.0) or 3.0)
            median_width = float(np.median(valid_widths))
            p25 = float(np.percentile(valid_widths, 25))
            if median_width > 0 and p25 > 0:
                threshold_width = median_width * width_ratio_thresh
                width_estuary = chan_bool & (local_width > threshold_width)
                n_width = int(width_estuary.sum())
                if n_width > 0:
                    estuary_mask |= width_estuary
                    log.info('[ESTUARY-CLIP] Width-ratio: %d pixels exceed %.0f m (%.1fx median %.0f m, p25=%.0f m)', n_width, threshold_width, width_ratio_thresh, median_width, p25)
                meta['signals']['width_ratio'] = {
                    'median_width_m': round(float(median_width), 1),
                    'threshold_width_m': round(float(threshold_width), 1),
                    'ratio_thresh': float(width_ratio_thresh),
                    'pixels_flagged': n_width,
                }
    except (ValueError, RuntimeError, ImportError):
        log.debug('[ESTUARY-CLIP] Width-ratio signal failed', exc_info=True)

    # Intentionally keep estuary detection simple and deterministic for now:
    # width-ratio only, optionally filtered by ocean connectivity below.
    hydraulic_hint = np.zeros_like(channel_before, dtype=bool)
    slope_near_widening = np.zeros_like(channel_before, dtype=bool)

    if ocean_mask_path is not None and Path(ocean_mask_path).exists():
        try:
            from scipy.ndimage import label as _label

            ocean_raw = np.zeros_like(channel, dtype='uint8')
            with rasterio.open(ocean_mask_path) as om:
                reproject(
                    source=rasterio.band(om, 1),
                    destination=ocean_raw,
                    src_transform=om.transform,
                    src_crs=om.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                    src_nodata=om.nodata,
                    dst_nodata=255,
                )
            ocean_water = (ocean_raw == 0)
            if np.any(ocean_water):
                channel_domain = channel_before
                ocean_touch = channel_domain & binary_dilation(ocean_water, iterations=1)
                connect_domain = channel_domain | binary_dilation(ocean_water, iterations=1)
                labeled, n_components = _label(connect_domain)
                ocean_labels = set(np.unique(labeled[ocean_touch & (labeled > 0)]))
                if ocean_labels:
                    ocean_connected = np.isin(labeled, list(ocean_labels)) & channel_domain
                else:
                    ocean_connected = np.zeros_like(channel_domain, dtype=bool)
                n_before_connect = int(estuary_mask.sum())
                estuary_mask &= ocean_connected
                n_after_connect = int(estuary_mask.sum())
                if n_before_connect > n_after_connect:
                    log.info('[ESTUARY-CLIP] Ocean connectivity trim: removed %d disconnected inland pixels (%d channel/ocean components checked, %d ocean-connected)', n_before_connect - n_after_connect, n_components, len(ocean_labels))

                channel_dist_px = _channel_distance_from_seed(ocean_connected, ocean_touch)
                channel_dist_m = channel_dist_px * px_size_m
                finite_channel_dist = np.isfinite(channel_dist_m) & ocean_connected
                min_transition_m = max(float(getattr(cfg, 'estuary_connect_dist_m', 200.0) or 200.0), 3.0 * px_size_m)
                transition_m = max(float(getattr(cfg, 'estuary_transition_m', 500.0) or 500.0), min_transition_m)
                near_mouth_corridor = finite_channel_dist & (channel_dist_m <= transition_m)

                width_seed = np.asarray(estuary_mask, dtype=bool)
                if np.any(width_seed):
                    support_width_m = max(float(np.percentile(valid_widths, 60)) if valid_widths.size > 0 else 0.0, 1.5 * px_size_m)
                    width_support = ocean_connected & (local_width >= support_width_m)
                    farthest_seed_m = float(np.nanmax(channel_dist_m[width_seed])) if np.any(width_seed) else 0.0
                    corridor_limit_m = max(transition_m, farthest_seed_m + max(4.0 * px_size_m, 50.0))
                else:
                    corridor_limit_m = transition_m
                constrained_corridor = finite_channel_dist & (channel_dist_m <= corridor_limit_m)

                estuary_mask = (near_mouth_corridor | (width_support & constrained_corridor)) & ocean_connected

                if np.any(estuary_mask):
                    est_labeled, est_n = _label(estuary_mask)
                    if est_n > 1:
                        sizes = np.bincount(est_labeled.ravel())
                        keep = np.zeros(est_n + 1, dtype=bool)
                        min_keep_px = max(4, int(round((transition_m / max(px_size_m, 1e-6)) * 0.25)))
                        for idx in range(1, est_n + 1):
                            if sizes[idx] >= min_keep_px:
                                keep[idx] = True
                        if np.any(keep[1:]):
                            estuary_mask = keep[est_labeled]

                meta['signals']['ocean_connectivity'] = {
                    'method': 'ocean_connected_channel_corridor',
                    'n_components': int(n_components),
                    'ocean_connected_components': len(ocean_labels),
                    'trimmed_inland_pixels': int(n_before_connect - n_after_connect),
                    'transition_m': round(float(transition_m), 1),
                    'corridor_limit_m': round(float(corridor_limit_m), 1),
                }
        except (OSError, ValueError, RuntimeError, ImportError):
            log.debug('[ESTUARY-CLIP] Ocean connectivity filter failed', exc_info=True)

    n_removed = int((channel_before & estuary_mask).sum())

    estuary_clip_path = channel_mask_tif.parent / 'estuary_clip_mask.tif'
    estuary_transition_path = channel_mask_tif.parent / 'estuary_transition_mask.tif'
    estuary_debug_receipt_path = channel_mask_tif.parent / 'estuary_debug_receipt.json'
    estuary_debug_dir = channel_mask_tif.parent / 'estuary_debug'
    estuary_debug_dir.mkdir(parents=True, exist_ok=True)

    prof_u8 = prof.copy()
    prof_u8.pop('blockxsize', None)
    prof_u8.pop('blockysize', None)
    prof_u8.pop('tiled', None)
    prof_u8.update(dtype='uint8', nodata=0, compress='deflate')
    with rasterio.open(estuary_clip_path, 'w', **prof_u8) as dst:
        dst.write(estuary_mask.astype('uint8'), 1)

    if n_removed == 0:
        with rasterio.open(estuary_transition_path, 'w', **prof_u8) as dst:
            dst.write(np.zeros_like(channel, dtype='uint8'), 1)
        meta['signals']['result'] = {
            'estuary_pixels_removed': 0,
            'transition_pixels': 0,
            'border_eroded_pixels': 0,
        }
        estuary_debug_receipt_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
        log.info('[ESTUARY-CLIP] No estuary pixels detected in channel mask; wrote zero estuary and transition masks.')
        return 0, estuary_clip_path

    channel[estuary_mask] = 0
    n_border_eroded = 0
    try:
        estuary_border = binary_dilation(estuary_mask, iterations=2) & ~estuary_mask
        border_channel = (channel > 0) & estuary_border
        n_border_eroded = int(border_channel.sum())
        if n_border_eroded > 0:
            channel[border_channel] = 0
            log.info('[ESTUARY-CLIP] Border erosion: removed %d channel-edge pixels adjacent to estuary boundary', n_border_eroded)
    except (ValueError, RuntimeError, ImportError):
        estuary_border = np.zeros_like(estuary_mask, dtype=bool)
        border_channel = np.zeros_like(estuary_mask, dtype=bool)

    channel_after = (channel > 0)
    transition_seed = channel_after & binary_dilation(estuary_mask, iterations=1)
    if not np.any(transition_seed):
        transition_seed = channel_after & binary_dilation(border_channel, iterations=1)
    if not np.any(transition_seed):
        transition_seed = channel_after & binary_dilation(near_mouth_corridor, iterations=1)
    transition_channel_dist_px = _channel_distance_from_seed(channel_after, transition_seed)
    transition_channel_dist_m = transition_channel_dist_px * px_size_m
    estuary_transition = channel_after & np.isfinite(transition_channel_dist_m) & (transition_channel_dist_m <= transition_m)
    estuary_transition &= channel_after

    with rasterio.open(estuary_transition_path, 'w', **prof_u8) as dst:
        dst.write(estuary_transition.astype('uint8'), 1)

    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    prof.pop('tiled', None)
    prof.update(dtype='uint8', nodata=0)
    with rasterio.open(channel_mask_tif, 'w', **prof) as dst:
        dst.write(channel.astype('uint8'), 1)
    n_remaining = int(channel_after.sum())
    log.info('[ESTUARY-CLIP] Removed %d estuary pixels from channel mask (%d remaining). Estuary mask: %s', n_removed, n_remaining, estuary_clip_path)
    log.info('[ESTUARY-CLIP] Transition zone retained upstream of estuary clip: %d pixels (%.0f m along-channel)', int(estuary_transition.sum()), float(transition_m))

    debug_masks = {
        'channel_preclip': channel_before,
        'aligned_ocean_water': ocean_water,
        'width_ratio_seed': width_estuary,
        'hydraulic_hint': hydraulic_hint,
        'low_slope_extension': slope_near_widening,
        'ocean_touch': ocean_touch,
        'ocean_connected_channel': ocean_connected,
        'near_mouth_corridor': near_mouth_corridor,
        'width_support': width_support,
        'constrained_corridor': constrained_corridor,
        'estuary_clip': estuary_mask,
        'estuary_transition': estuary_transition,
        'channel_postclip': channel_after,
    }
    for name, mask in debug_masks.items():
        _write_debug_mask(estuary_debug_dir / f'{name}.tif', prof, mask)

    debug_receipt = {
        'method': 'width_ratio, ocean-connectivity filtered',
        'pixel_size_m': float(px_size_m),
        'transition_m': float(transition_m),
        'corridor_limit_m': float(corridor_limit_m),
        'signals': meta.get('signals', {}),
        'masks': {name: _mask_bbox_summary(mask, transform) for name, mask in debug_masks.items()},
        'paths': {
            'estuary_clip_mask': str(estuary_clip_path),
            'estuary_transition_mask': str(estuary_transition_path),
            'debug_dir': str(estuary_debug_dir),
        },
    }
    write_json(estuary_debug_receipt_path, debug_receipt)

    report.setdefault('river', {}).setdefault('estuary_clip', {}).update({
        'pixels_removed': n_removed,
        'pixels_remaining': n_remaining,
        'border_eroded_pixels': int(n_border_eroded),
        'estuary_clip_mask': str(estuary_clip_path),
        'estuary_transition_mask': str(estuary_transition_path),
        'estuary_debug_receipt': str(estuary_debug_receipt_path),
        'estuary_debug_dir': str(estuary_debug_dir),
        'method': 'width_ratio, ocean-connectivity filtered',
        'signals': meta.get('signals', {}),
    })
    report.setdefault('river', {}).setdefault('outputs', {}).update({
        'estuary_clip_mask': str(estuary_clip_path),
        'estuary_transition': str(estuary_transition_path),
        'estuary_debug_receipt': str(estuary_debug_receipt_path),
    })
    return n_removed, estuary_clip_path
