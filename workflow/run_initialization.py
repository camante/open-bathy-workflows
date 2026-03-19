from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.paths import ensure_dir
from pipeline.aoi import parse_aoi_bbox as _parse_aoi_bbox, bbox_to_aoi_str as _bbox_to_aoi_str, expand_bbox_frac as _expand_bbox_frac


def initialize_run_args(args, *, logger: logging.Logger) -> Optional[str]:
    """Prepare AOI buffering, logging, auto coefficient lookup, SWOT auto-fetch, and CUDEM cache.

    Returns run_id when logging initialization succeeds, else None.
    """
    tile_bbox = _parse_aoi_bbox(getattr(args, "aoi", None))
    if not tile_bbox:
        raise SystemExit("Invalid --aoi. Expected bbox 'W/E/S/N' (lon_min/lon_max/lat_min/lat_max).")

    args.aoi_tile = getattr(args, "aoi", None)
    args.tile_bbox = tile_bbox

    try:
        legacy_buf_km = float(getattr(args, "tile_buffer_km", 0.0) or 0.0)
    except (TypeError, ValueError):
        legacy_buf_km = 0.0
    if legacy_buf_km not in (0.0, -0.0):
        logger.warning("[AOI] --tile-buffer-km is deprecated and ignored. Use --buff instead. (legacy=%s)", legacy_buf_km)

    try:
        buff_frac = float(getattr(args, "buff", 0.0) or 0.0)
    except (TypeError, ValueError):
        buff_frac = 0.0
    if buff_frac < 0:
        raise SystemExit("--buff must be >= 0")

    if buff_frac > 0:
        expanded = _expand_bbox_frac(tile_bbox, buff_frac)
        args.aoi = _bbox_to_aoi_str(expanded)
        logger.info("[AOI] Data buffer enabled: --buff=%s -> aoi_data=%s (aoi_tile=%s)", buff_frac, args.aoi, args.aoi_tile)
    else:
        logger.info("[AOI] No data buffer: --buff=0 -> aoi_data=aoi_tile=%s", args.aoi_tile)
    args.aoi_hydro = getattr(args, "aoi", None)

    run_id = None
    try:
        from logging_config import add_file_handler, install_screen_log, start_flight_recorder
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"bathy_{ts}_{os.getpid()}"
        out_dir = Path(getattr(args, 'out_dir', 'output'))
        (out_dir / "run_logs").mkdir(parents=True, exist_ok=True)
        screen_log_path = install_screen_log(out_dir / "run_logs" / f"screen_{run_id}.log")
        logger.info("[RUN] Screen log: %s", screen_log_path)
        add_file_handler(out_dir / "run_logs" / f"run_{run_id}.log", level=logging.INFO)
        fr_path = start_flight_recorder(out_dir, run_id=run_id)
        if fr_path is not None:
            logger.info("[RUN] Flight recorder: %s", fr_path)
        logger.info("[RUN] run_id=%s", run_id)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Unable to initialize run logs/flight recorder: %s", e)

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'sdb_config.json')
        cfg_json: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
        lon_c = lat_c = None
        try:
            if isinstance(getattr(args, 'aoi', None), str) and '/' in args.aoi:
                w,e,s,n = [float(x) for x in args.aoi.strip().strip('"').split('/')[:4]]
                lon_c = (w + e) / 2.0
                lat_c = (s + n) / 2.0
        except (TypeError, ValueError):
            lon_c = lat_c = None

        if lon_c is not None and lat_c is not None and cfg_json:
            from region_resolver import resolve_q2_regression, resolve_bankfull_curve
            if str(getattr(args, 'river_manning_region', 'default')).strip().lower() == 'auto':
                q2_curve, msg = resolve_q2_regression(cfg_json, lon_c, lat_c)
                if q2_curve is not None and q2_curve.name:
                    args.river_manning_region = q2_curve.name
                    logger.info("[RIVER][MANNING] Auto region: %s", msg)
                else:
                    from manning_inversion import infer_manning_region_from_aoi
                    region_key, st = infer_manning_region_from_aoi(getattr(args, 'aoi', None), state_region_map_json=getattr(args, 'river_manning_region_auto_map', None), default_region='default')
                    args.river_manning_region = region_key
                    logger.warning("[RIVER][MANNING] Published regression not found; using coarse region=%s (state=%s)", region_key, st)

            if getattr(args, 'river_regional_curve_enabled', False) and (getattr(args, 'river_regional_curve_c', None) is None or getattr(args, 'river_regional_curve_f', None) is None):
                bcurve, msg = resolve_bankfull_curve(cfg_json, lon_c, lat_c)
                if bcurve is not None:
                    args.river_regional_curve_region = bcurve.name
                    args.river_regional_curve_c = float(bcurve.c)
                    args.river_regional_curve_f = float(bcurve.f)
                    logger.info("[RIVER][REGIONAL] Auto coefficients: %s", msg)
                else:
                    logger.warning("[RIVER][REGIONAL] Auto coefficients unavailable: %s", msg)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as e:
        logger.debug("[RIVER] Auto coefficient resolution failed: %s", e)

    try:
        if (str(getattr(args, "river_skeleton_wse_mode", "bank")).strip().lower() == "bank_profile"
                and getattr(args, "river_swot_riversp", None) is None
                and bool(getattr(args, "river_swot_auto", True))):
            w = e = s = n = None
            try:
                if isinstance(getattr(args, "aoi", None), str) and "/" in args.aoi:
                    w, e, s, n = [float(x) for x in args.aoi.strip().strip('"').split("/")[:4]]
            except (TypeError, ValueError):
                w = e = s = n = None
            if None not in (w, e, s, n):
                swot_cache_root = getattr(args, "river_swot_cache_root", None) or os.path.join(str(getattr(args, "cache_root", "cache")), "swot")
                from swot_riversp_fetch import fetch_riversp
                res = fetch_riversp(
                    bbox_wesn=(float(w), float(e), float(s), float(n)),
                    start_date=str(getattr(args, "start_date", "")),
                    end_date=str(getattr(args, "end_date", "")),
                    cache_root=str(swot_cache_root),
                    product=str(getattr(args, "river_swot_product", "reach") or "reach"),
                    short_name=getattr(args, "river_swot_shortname", None),
                    logger=logger,
                )
                if res.files:
                    args.river_swot_riversp = res.files
                    logger.info("[SWOT] Auto-fetched RiverSP (%s file(s)) -> %s", len(res.files), res.cache_dir)
                else:
                    logger.info("[SWOT] RiverSP auto-fetch not used: %s", res.message)
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.debug("[SWOT] RiverSP auto-fetch failed: %s", e)

    if bool(getattr(args, "river_manning_enabled", False)) and str(getattr(args, "river_manning_mode", "off")) == "off":
        args.river_manning_mode = "q2_regional"
    if bool(getattr(args, "river_enable_1d_energy_solver", False)) and str(getattr(args, "river_manning_mode", "off")) == "off":
        args.river_manning_mode = "q2_regional"
        logger.info("[RIVER][MANNING] Auto-enabled manning_mode=q2_regional because --river-enable-1d-energy-solver is set")

    try:
        _cc = os.environ.get("CUDEM_CACHE", "").strip()
        if not _cc:
            cudem_cache_dir = Path(args.cache_root) / "cudem_cache"
            ensure_dir(cudem_cache_dir)
            try:
                ensure_dir(cudem_cache_dir / "tnm")
            except OSError:
                logger.debug("ignored", exc_info=True)
            os.environ["CUDEM_CACHE"] = str(cudem_cache_dir)
            logger.info("[CUDEM_CACHE] Using pipeline-scoped CUDEM cache: %s", cudem_cache_dir)
        else:
            ensure_dir(Path(_cc))
            try:
                ensure_dir(Path(_cc) / "tnm")
            except OSError:
                logger.debug("ignored", exc_info=True)
            logger.info("[CUDEM_CACHE] Using existing CUDEM_CACHE: %s", _cc)
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("[CUDEM_CACHE] Unable to prepare CUDEM cache directory: %s", e)

    return run_id


def resolve_authoritative_base_args(args) -> Dict[str, Any]:
    authoritative_base_arg = getattr(args, "authoritative_base", None)
    authoritative_base_auto = bool(getattr(args, "authoritative_base_auto", False))
    authoritative_base_path = None
    if authoritative_base_arg is not None:
        text = str(authoritative_base_arg).strip()
        low = text.lower() if text else ""
        if text and low in ("auto", "cudem", "auto_cudem"):
            authoritative_base_auto = True
            authoritative_base_path = None
        elif text and low in ("off", "none", "disable", "disabled"):
            authoritative_base_auto = False
            authoritative_base_path = None
        elif text:
            authoritative_base_path = Path(text)
    return {
        "authoritative_base_auto": authoritative_base_auto,
        "authoritative_base_path": authoritative_base_path,
    }
