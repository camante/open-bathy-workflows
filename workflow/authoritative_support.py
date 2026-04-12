from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from river_withheld_support import attach_support_point_keys, load_withheld_support_keys, summarize_role_counts


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


def _apply_roleaware_support_outputs(cfg: Any, section: Dict[str, Any], *, info: Optional[Dict[str, Any]] = None, role_contract: Optional[Path] = None) -> None:
    info = info or {}
    contract_payload: Dict[str, Any] = {}
    if role_contract is not None and Path(role_contract).exists():
        try:
            import json as _json
            contract_payload = _json.loads(Path(role_contract).read_text(encoding="utf-8"))
        except Exception:
            contract_payload = {}
    artifacts = contract_payload.get("artifacts", {}) if isinstance(contract_payload, dict) else {}
    mapping = {
        "role_code_raster": "river_authoritative_role_code_raster",
        "role_confidence_raster": "river_authoritative_role_confidence_raster",
        "distance_to_bank_raster": "river_authoritative_distance_to_bank_raster",
        "normalized_channel_position_raster": "river_authoritative_normalized_channel_position_raster",
    }
    for key, attr in mapping.items():
        raw = info.get(key) or artifacts.get(key) or section.get(key)
        if raw:
            setattr(cfg, attr, Path(str(raw)))
            section[key] = str(raw)
    if role_contract is not None and Path(role_contract).exists():
        setattr(cfg, "river_authoritative_support_contract", Path(role_contract))
        section["role_contract"] = str(role_contract)




def _short_hash_for_path(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8", errors="ignore"))
    try:
        h.update(path.read_bytes())
    except Exception:
        pass
    return h.hexdigest()[:12]


def _maybe_apply_river_withheld_support_exclusion(*, cfg: Any, cache_dir: Path, section: Dict[str, Any], base_points_path: Path, logger: logging.Logger) -> Path:
    withheld_raw = getattr(cfg, "river_withheld_support_csv", None)
    if not withheld_raw:
        return base_points_path
    withheld_path = Path(str(withheld_raw)).resolve()
    if not withheld_path.exists():
        raise FileNotFoundError(f"River withheld-support CSV not found: {withheld_path}")

    suffix = _short_hash_for_path(withheld_path)
    filtered_path = cache_dir / f"{base_points_path.stem}_withheld_{suffix}.csv"
    receipt_path = cache_dir / f"{base_points_path.stem}_withheld_{suffix}.json"
    if filtered_path.exists() and filtered_path.stat().st_size > 0 and receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception:
            receipt = {"available": True, "status": "cached"}
        section["withheld_support"] = receipt
        section["path"] = str(filtered_path)
        setattr(cfg, "river_withheld_support_receipt", receipt_path)
        setattr(cfg, "river_authoritative_soundings", filtered_path)
        return filtered_path

    base_df = pd.read_csv(base_points_path)
    if base_df.empty:
        raise RuntimeError(f"River authoritative support CSV is empty before withheld-support filtering: {base_points_path}")
    base_df = attach_support_point_keys(base_df)
    withheld_keys = load_withheld_support_keys(withheld_path)
    if not withheld_keys:
        raise RuntimeError(f"River withheld-support CSV has no usable support_point_key rows: {withheld_path}")
    matched_mask = base_df["support_point_key"].astype(str).isin(withheld_keys)
    keep_mask = ~matched_mask
    removed = int((~keep_mask).sum())
    filtered_df = base_df.loc[keep_mask].copy()
    matched_df = base_df.loc[matched_mask].copy()
    if filtered_df.empty:
        raise RuntimeError(
            f"River withheld-support filtering removed all authoritative support points: base={base_points_path} withheld={withheld_path}"
        )
    filtered_df.to_csv(filtered_path, index=False)
    receipt = {
        "available": True,
        "base_points_path": str(base_points_path),
        "withheld_support_csv": str(withheld_path),
        "filtered_points_path": str(filtered_path),
        "base_point_count": int(len(base_df)),
        "requested_withheld_key_count": int(len(withheld_keys)),
        "matched_withheld_key_count": int(matched_mask.sum()),
        "unmatched_withheld_key_count": int(max(len(withheld_keys) - int(matched_mask.sum()), 0)),
        "excluded_count": removed,
        "removed_count": removed,
        "remaining_count": int(len(filtered_df)),
        "removed_fraction": float(removed / max(len(base_df), 1)),
        "base_role_counts": summarize_role_counts(base_df),
        "matched_role_counts": summarize_role_counts(matched_df),
        "excluded_role_counts": summarize_role_counts(matched_df),
        "remaining_role_counts": summarize_role_counts(filtered_df),
        "status": "applied" if removed > 0 else "no_matches_found",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    section["withheld_support"] = receipt
    section["path"] = str(filtered_path)
    setattr(cfg, "river_withheld_support_receipt", receipt_path)
    setattr(cfg, "river_authoritative_soundings", filtered_path)
    logger.info(
        "[AUTHORITATIVE] Applied river withheld-support exclusion: removed=%d remaining=%d withheld=%s -> %s",
        removed,
        int(len(filtered_df)),
        withheld_path,
        filtered_path,
    )
    return filtered_path


def prepare_sdb_guidance_from_authoritative(
    *,
    cfg: Any,
    report: Dict[str, Any],
    ensure_dir_fn: Callable[[Path], Path],
    hash_key_fn: Callable[..., str],
    prepare_points_fn: Callable[..., Dict[str, Any]],
    build_sdb_authoritative_support_products_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    logger: LoggerLike = None,
) -> Optional[Path]:
    """Prepare authoritative-base-derived SDB support points and explicit support artifacts."""
    log = _default_logger(logger)
    auth = getattr(cfg, "authoritative_base", None)
    if not auth:
        return None
    auth_path = Path(auth)
    if not auth_path.exists():
        return None

    out_crs = str(getattr(cfg, "extra_xyz_crs", None) or getattr(cfg, "working_srs", None) or "EPSG:4326")
    cache_dir = ensure_dir_fn(Path(cfg.cache_root) / "authoritative_support")
    section = report.setdefault("authoritative_base", {}).setdefault("sdb_guidance", {})
    contract_section = section.setdefault('explicit_support', {})

    domain_mask = getattr(cfg, 'sdb_candidate_domain_mask', None)
    if domain_mask and Path(domain_mask).exists() and build_sdb_authoritative_support_products_fn is not None:
        support_key = hash_key_fn(str(auth_path), str(domain_mask), out_crs, 'sdb_authoritative_support_v1')
        out_mask = cache_dir / f"authoritative_sdb_support_mask_{support_key}.tif"
        out_values = cache_dir / f"authoritative_sdb_support_values_{support_key}.tif"
        out_points = cache_dir / f"authoritative_sdb_support_points_{support_key}.csv"
        out_contract = cache_dir / f"authoritative_sdb_support_contract_{support_key}.json"
        try:
            if not (out_contract.exists() and out_points.exists() and out_mask.exists() and out_values.exists()):
                contract = build_sdb_authoritative_support_products_fn(
                    auth_path, domain_mask, out_mask, out_values, out_points, out_contract, out_crs=out_crs, logger=log
                )
            else:
                import json as _json
                contract = _json.loads(out_contract.read_text(encoding='utf-8'))
            contract_section.update({
                'cached': bool(out_contract.exists()),
                'support_mask': str(out_mask),
                'support_values': str(out_values),
                'support_points': str(out_points),
                'contract': str(out_contract),
                'support_pixels': int(contract.get('support_pixels', 0)),
                'candidate_pixels': int(contract.get('candidate_pixels', 0)),
            })
            setattr(cfg, 'sdb_authoritative_support_mask', out_mask)
            setattr(cfg, 'sdb_authoritative_support_values', out_values)
            setattr(cfg, 'sdb_authoritative_support_points', out_points)
            setattr(cfg, 'sdb_authoritative_support_contract', out_contract)
            setattr(cfg, 'sdb_authoritative_extra_xyz', out_points)
            section.update({'path': str(out_points), 'source': 'authoritative_sdb_support', 'cached': bool(out_contract.exists())})
            return out_points
        except Exception as exc:
            contract_section['error'] = str(exc)
            log.warning('[AUTHORITATIVE] Failed building explicit SDB authoritative-support products: %s', exc, exc_info=True)

    out_csv = cache_dir / (
        "authoritative_sdb_support_"
        f"{hash_key_fn(str(auth_path), cfg.aoi, out_crs, getattr(cfg, 'working_vcrs_epsg', 5703), getattr(cfg, 'sdb_source_vdatum', 'epsg:4269+5714'))}.csv"
    )
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
        f"{hash_key_fn(str(auth_path), cfg.aoi, out_crs, getattr(cfg, 'working_vcrs_epsg', 5703), 'river_roleaware_v3')}.csv"
    )
    section = report.setdefault("authoritative_base", {}).setdefault("river_guidance", {})
    role_contract = out_csv.with_suffix('.role_contract.json')
    if out_csv.exists() and out_csv.stat().st_size > 0:
        setattr(cfg, "river_authoritative_soundings", out_csv)
        section.update({"cached": True, "path": str(out_csv), "source": "authoritative_base", "role_contract": str(role_contract) if role_contract.exists() else None})
        _apply_roleaware_support_outputs(cfg, section, role_contract=role_contract if role_contract.exists() else None)
        return _maybe_apply_river_withheld_support_exclusion(cfg=cfg, cache_dir=cache_dir, section=section, base_points_path=out_csv, logger=log)

    try:
        mainstem_mask = None
        try:
            cand = Path(getattr(cfg, 'river_channel_mask', '') or '').parent / 'mainstem_mask.tif'
            if cand.exists():
                mainstem_mask = cand
        except Exception:
            mainstem_mask = None
        info = prepare_points_fn(
            auth_path,
            out_csv,
            out_crs=out_crs,
            negative_only=False,
            river_channel_mask=getattr(cfg, 'river_channel_mask', None),
            river_guidance_domain_mask=getattr(cfg, 'river_guidance_domain_mask', None),
            estuary_clip_mask=getattr(cfg, 'estuary_clip_mask', None),
            mainstem_mask=mainstem_mask,
            bank_margin_m=float(getattr(cfg, 'river_guidance_bank_margin_m', 3.0) or 3.0),
            logger=log,
        )
        section.update(info)
        section.setdefault("source", "authoritative_base")
        _apply_roleaware_support_outputs(
            cfg,
            section,
            info=info,
            role_contract=Path(str(info['role_contract'])) if info.get('role_contract') else (role_contract if role_contract.exists() else None),
        )
        validated = _validate_written_points(out_csv, section, cfg_attr="river_authoritative_soundings", cfg=cfg)
        return _maybe_apply_river_withheld_support_exclusion(cfg=cfg, cache_dir=cache_dir, section=section, base_points_path=validated, logger=log)
    except (OSError, RuntimeError, ValueError) as exc:
        section["error"] = str(exc)
        log.warning("[AUTHORITATIVE] Failed preparing river soundings support from authoritative_base: %s", exc, exc_info=True)
        return None


__all__ = [
    "prepare_sdb_guidance_from_authoritative",
    "prepare_river_support_from_authoritative",
]
