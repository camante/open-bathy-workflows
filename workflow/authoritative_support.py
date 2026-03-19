from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional


LoggerLike = Optional[logging.Logger]


def _default_logger(logger: LoggerLike) -> logging.Logger:
    return logger or logging.getLogger("authoritative_support")


def _validate_written_points(path: Path, section: Dict[str, Any], *, cfg_attr: str, cfg: Any) -> Path:
    candidate = path
    reported_path = section.get("path")
    if reported_path:
        reported = Path(str(reported_path))
        if reported.exists() and reported.stat().st_size > 0:
            candidate = reported
    if not candidate.exists() or candidate.stat().st_size <= 0:
        raise RuntimeError(f"Expected authoritative support artifact was not written: {candidate}")
    setattr(cfg, cfg_attr, candidate)
    section["path"] = str(candidate)
    section.setdefault("cached", False)
    return candidate


def prepare_sdb_guidance_from_authoritative(
    *,
    cfg: Any,
    report: Dict[str, Any],
    ensure_dir_fn: Callable[[Path], Path],
    hash_key_fn: Callable[..., str],
    prepare_points_fn: Callable[..., Dict[str, Any]],
    logger: LoggerLike = None,
) -> Optional[Path]:
    """Prepare authoritative-base-derived SDB support points."""
    log = _default_logger(logger)
    auth = getattr(cfg, "authoritative_base", None)
    if not auth:
        return None
    auth_path = Path(auth)
    if not auth_path.exists():
        return None

    out_crs = str(getattr(cfg, "extra_xyz_crs", None) or getattr(cfg, "working_srs", None) or "EPSG:4326")
    cache_dir = ensure_dir_fn(Path(cfg.cache_root) / "authoritative_support")
    out_csv = cache_dir / (
        "authoritative_sdb_support_"
        f"{hash_key_fn(str(auth_path), cfg.aoi, out_crs, getattr(cfg, 'working_vcrs_epsg', 5703), getattr(cfg, 'sdb_source_vdatum', 'epsg:4269+5714'))}.csv"
    )
    section = report.setdefault("authoritative_base", {}).setdefault("sdb_guidance", {})
    if out_csv.exists() and out_csv.stat().st_size > 0:
        setattr(cfg, "sdb_authoritative_extra_xyz", out_csv)
        section.update({"cached": True, "path": str(out_csv), "source": "authoritative_base"})
        return out_csv

    try:
        info = prepare_points_fn(
            auth_path,
            out_csv,
            out_crs=out_crs,
            source_vdatum=f"epsg:4269+{int(getattr(cfg, 'working_vcrs_epsg', 5703) or 5703)}",
            target_vdatum=str(getattr(cfg, "sdb_source_vdatum", "epsg:4269+5714") or "epsg:4269+5714"),
            logger=log,
        )
        section.update(info)
        section.setdefault("source", "authoritative_base")
        return _validate_written_points(out_csv, section, cfg_attr="sdb_authoritative_extra_xyz", cfg=cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        section["error"] = str(exc)
        log.warning("[AUTHORITATIVE] Failed preparing SDB guidance points from authoritative_base: %s", exc, exc_info=True)
        return None


def prepare_river_support_from_authoritative(
    *,
    cfg: Any,
    report: Dict[str, Any],
    ensure_dir_fn: Callable[[Path], Path],
    hash_key_fn: Callable[..., str],
    prepare_points_fn: Callable[..., Dict[str, Any]],
    logger: LoggerLike = None,
) -> Optional[Path]:
    """Prepare authoritative-base-derived river sounding support points."""
    log = _default_logger(logger)
    auth = getattr(cfg, "authoritative_base", None)
    if not auth:
        return None
    auth_path = Path(auth)
    if not auth_path.exists():
        return None

    out_crs = str(getattr(cfg, "working_srs", None) or getattr(cfg, "extra_xyz_crs", None) or "EPSG:4326")
    cache_dir = ensure_dir_fn(Path(cfg.cache_root) / "authoritative_support")
    out_csv = cache_dir / (
        "authoritative_river_soundings_"
        f"{hash_key_fn(str(auth_path), cfg.aoi, out_crs, getattr(cfg, 'working_vcrs_epsg', 5703), 'river')}.csv"
    )
    section = report.setdefault("authoritative_base", {}).setdefault("river_guidance", {})
    if out_csv.exists() and out_csv.stat().st_size > 0:
        setattr(cfg, "river_authoritative_soundings", out_csv)
        section.update({"cached": True, "path": str(out_csv), "source": "authoritative_base"})
        return out_csv

    try:
        info = prepare_points_fn(
            auth_path,
            out_csv,
            out_crs=out_crs,
            logger=log,
        )
        section.update(info)
        section.setdefault("source", "authoritative_base")
        return _validate_written_points(out_csv, section, cfg_attr="river_authoritative_soundings", cfg=cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        section["error"] = str(exc)
        log.warning("[AUTHORITATIVE] Failed preparing river soundings support from authoritative_base: %s", exc, exc_info=True)
        return None


__all__ = [
    "prepare_sdb_guidance_from_authoritative",
    "prepare_river_support_from_authoritative",
]
