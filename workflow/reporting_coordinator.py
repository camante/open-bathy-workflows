"""Coordinator for final report/receipt writing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.json_io import write_json


def write_final_reporting_bundle(
    cfg: Any,
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
    write_authoritative_cache_receipt: Callable[..., Optional[Path]],
    write_guidance_manifest: Callable[..., Optional[Path]],
    write_explicit_final_outputs_manifest: Callable[..., Optional[Path]],
    write_support_provenance_summary: Callable[..., Optional[Path]],
    write_final_support_regime_audit: Callable[..., Optional[Path]],
    write_final_dem_selection_receipt: Callable[..., Optional[Path]],
    write_comparison_package: Callable[..., Optional[Path]],
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or logging.getLogger(__name__)
    report_path = Path(cfg.out_dir) / "bathy_report.json"
    write_json(report_path, report)

    writers = [
        ("authoritative_cache_receipt", lambda: write_authoritative_cache_receipt(cfg, report)),
        ("guidance_manifest", lambda: write_guidance_manifest(cfg, report)),
        ("explicit_final_outputs", lambda: write_explicit_final_outputs_manifest(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)),
        ("support_provenance_summary", lambda: write_support_provenance_summary(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)),
        ("final_support_regime_audit", lambda: write_final_support_regime_audit(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)),
        ("final_dem_selection_receipt", lambda: write_final_dem_selection_receipt(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)),
        ("comparison_package", lambda: write_comparison_package(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)),
    ]
    receipts = report.setdefault("final_reporting", {}).setdefault("receipts", {})
    for name, fn in writers:
        try:
            out = fn()
            receipts[name] = str(out) if out else None
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
            receipts[name] = None
            log.debug("Final reporting step failed [%s]: %s", name, exc, exc_info=True)
    try:
        write_json(report_path, report)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        log.debug("Final report refresh after reporting bundle failed: %s", exc, exc_info=True)
    return report_path
