from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

from pipeline.final_route.final_route_receipts import write_json_receipt
import logging
log = logging.getLogger(__name__)



def _as_path(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return None
    if not isinstance(value, (str, os.PathLike)):
        return None
    try:
        path_text = os.fspath(value).strip()
        if not path_text:
            return None
        return str(Path(path_text))
    except (TypeError, ValueError, OSError):
        log.debug("_as_path: suppressed exception", exc_info=True)
        return None


def build_legacy_cleanup_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    ab = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    ab_out = ab.get("outputs", {}) if isinstance(ab.get("outputs", {}), dict) else {}
    fusion = report.get("fusion", {}) if isinstance(report.get("fusion", {}), dict) else {}
    fusion_out = fusion.get("outputs", {}) if isinstance(fusion.get("outputs", {}), dict) else {}
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    sdb_art = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    river_out = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}

    deprecated_artifacts = {
        "legacy_fusion_depth": _as_path(fusion_out.get("depth")),
        "legacy_fusion_provenance": _as_path(fusion_out.get("provenance")),
        "diagnostic_guidance_surface": _as_path(ab_out.get("source_aware_candidate")),
        "diagnostic_guidance_surface_provenance": _as_path(ab_out.get("source_aware_candidate_provenance")),
        "dense_sdb_depth_raster": _as_path(sdb_art.get("depth_raster")),
        "dense_river_depth_raster": _as_path(river_out.get("depth_terrain")),
        "dense_river_bottom_elevation": _as_path(river_out.get("bottom_elevation")),
    }
    present = {k: v for k, v in deprecated_artifacts.items() if v}

    route_mode = final_route.get("route_mode")
    single_route = bool(final_route.get("single_authoritative_route_active"))
    legacy_retired = bool(final_route.get("legacy_parallel_route_retired"))
    structural = final_route.get("structural_inputs", {}) if isinstance(final_route.get("structural_inputs", {}), dict) else {}
    structural_values = {str(v) for v in structural.values() if v}
    legacy_values = {str(v) for v in present.values() if v}
    legacy_leak = sorted(structural_values & legacy_values)

    return {
        "phase": "K",
        "route_mode": route_mode,
        "single_authoritative_route_active": single_route,
        "legacy_parallel_route_retired": legacy_retired,
        "deprecated_artifacts_present": present,
        "deprecated_artifact_count": int(len(present)),
        "legacy_artifacts_leaking_into_structural_inputs": legacy_leak,
        "legacy_inputs_excluded_from_structural_route": not legacy_leak,
        "recommended_removals": [
            "inline legacy final-route orchestration",
            "legacy fusion-only output selection assumptions",
            "dense peer-surface interpretation of SDB/river rasters",
        ],
    }


def write_legacy_cleanup_receipt(*, report: Dict[str, Any], receipt_path: str | Path) -> Dict[str, Any]:
    payload = build_legacy_cleanup_summary(report)
    write_json_receipt(Path(receipt_path), payload)
    return payload


__all__ = ["build_legacy_cleanup_summary", "write_legacy_cleanup_receipt"]
