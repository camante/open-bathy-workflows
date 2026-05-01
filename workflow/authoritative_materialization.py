"""Coordinator for authoritative-base materialization and AOI-keyed cache reuse."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio




def _write_simple_hillshade(src: Path, dst: Path) -> Optional[Path]:
    try:
        with rasterio.open(src) as ds:
            arr = ds.read(1).astype(np.float32)
            profile = ds.profile.copy()
        nodata = profile.get("nodata")
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != float(nodata)
        if not np.any(valid):
            return None
        filled = arr.copy()
        fill_value = float(np.nanmedian(arr[valid]))
        filled[~valid] = fill_value
        x, y = np.gradient(filled)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        az = np.deg2rad(315.0)
        alt = np.deg2rad(45.0)
        shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        shaded = ((np.clip(shaded, -1.0, 1.0) + 1.0) * 127.5).astype(np.float32)
        shaded[~valid] = np.nan
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("BLOCKXSIZE", None)
        profile.pop("BLOCKYSIZE", None)
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, 'w', **profile) as out_ds:
            out_ds.write(shaded, 1)
            out_ds.update_tags(VALUE_TYPE='hillshade', ROLE='visual_comparison_hillshade')
        return dst if dst.exists() else None
    except Exception:
        return None


def _materialize_initial_final_baseline_products(cfg: Any, baseline_path: Path, *, logger: Optional[logging.Logger] = None) -> Dict[str, Optional[str]]:
    log = logger or logging.getLogger(__name__)
    outputs: Dict[str, Optional[str]] = {
        'final_folder_authoritative_base': None,
        'final_folder_authoritative_base_hillshade': None,
    }
    try:
        out_dir = Path(getattr(cfg, 'out_dir', '') or '')
        if not out_dir:
            return outputs
        final_dir = out_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        baseline_final = final_dir / 'authoritative_base_aligned.tif'
        hillshade_final = final_dir / 'authoritative_base_aligned_hillshade.tif'
        import shutil
        shutil.copy2(baseline_path, baseline_final)
        outputs['final_folder_authoritative_base'] = str(baseline_final)
        if hillshade_final.exists():
            hillshade_final.unlink()
        hillshade = _write_simple_hillshade(baseline_final, hillshade_final)
        if hillshade is not None:
            outputs['final_folder_authoritative_base_hillshade'] = str(hillshade)
        log.info('[AUTHORITATIVE] Wrote initial final-folder baseline products from pre-mask CUDEM mosaic: %s', baseline_final)
        return outputs
    except Exception:
        log.debug('[AUTHORITATIVE] Failed to materialize initial final-folder baseline products', exc_info=True)
        return outputs

def resolve_authoritative_base(cfg: Any, args: argparse.Namespace, *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Resolve or auto-build authoritative_base for the processing AOI.

    Behavior:
      - an explicit existing path wins
      - an explicit missing path is preserved when auto materialization is disabled
      - otherwise, auto-materialize from NOAA CUDEM tile index + spatial metadata
        into <cache_root>/authoritative_base/<cache_key>/ with reuse on repeat AOIs
    """
    log = logger or logging.getLogger(__name__)
    auth_path = getattr(cfg, "authoritative_base", None)
    auto = bool(getattr(cfg, "authoritative_base_auto", False))

    if auth_path is not None:
        auth_path = Path(auth_path)
        if auth_path.exists():
            cfg.authoritative_base = auth_path
            setattr(cfg, "export_authoritative_base", Path(auth_path))
            explicit_report = {
                "mode": "explicit_path",
                "authoritative_base": str(auth_path),
                "cache_hit": None,
                "source_role": "export_only",
                "routing_policy": "non_canonical_export_product",
            }
            try:
                sibling_baseline = auth_path.with_name('cudem_baseline_interpolation.tif')
                if sibling_baseline.exists():
                    explicit_report['baseline_cudem_interpolation'] = str(sibling_baseline)
            except Exception:
                log.debug('[AUTHORITATIVE] Failed resolving explicit authoritative path sibling baseline', exc_info=True)
            setattr(args, "_authoritative_base_auto_report", explicit_report)
            return auth_path
        if not auto:
            log.warning("[AUTHORITATIVE] Provided authoritative_base does not exist: %s", auth_path)
            cfg.authoritative_base = auth_path
            return auth_path
        log.info("[AUTHORITATIVE] Explicit authoritative_base path not found; falling back to auto-materialization for AOI.")

    if not auto:
        return auth_path

    try:
        from cudem_authoritative import materialize_authoritative_base_for_aoi
    except ImportError as exc:
        log.error("[AUTHORITATIVE] Failed to import cudem_authoritative auto-builder: %s", exc, exc_info=True)
        raise

    build_info = materialize_authoritative_base_for_aoi(
        aoi=str(getattr(cfg, "aoi", "") or ""),
        cache_root=Path(cfg.cache_root),
        tile_index_url=str(getattr(cfg, "authoritative_base_tile_index_url", "") or ""),
        spatial_meta_url=str(getattr(cfg, "authoritative_base_spatial_meta_url", "") or ""),
        missing_meta_policy=str(getattr(cfg, "authoritative_base_missing_meta_policy", "skip") or "skip"),
        tile_url_field=getattr(cfg, "authoritative_base_tile_url_field", None),
        force_rebuild=bool(getattr(cfg, "authoritative_base_force_rebuild", False)),
        logger=log,
    )
    build_info.setdefault('source_role', 'export_only')
    build_info.setdefault('routing_policy', 'non_canonical_export_product')
    cfg.authoritative_base = Path(build_info["authoritative_base"]).resolve()
    setattr(cfg, "export_authoritative_base", Path(cfg.authoritative_base))
    baseline_path = Path(build_info.get('baseline_cudem_interpolation', '') or '')
    if baseline_path.exists():
        build_info['baseline_cudem_interpolation'] = str(baseline_path)
    setattr(args, "_authoritative_base_auto_report", build_info)
    log.info(
        "[AUTHORITATIVE] %s authoritative_base: %s",
        "Reused cached" if bool(build_info.get("cache_hit")) else "Materialized",
        cfg.authoritative_base,
    )
    return cfg.authoritative_base
