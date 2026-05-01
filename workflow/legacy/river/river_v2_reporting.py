from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

try:
    from core.constants import PIPELINE_VERSION
except Exception:
    PIPELINE_VERSION = "unknown"

from legacy.river.river_v2_contract import PASS1_STAGE_IDS, PASS2_STAGE_IDS, PASS3_STAGE_IDS, PASS4_STAGE_IDS


DEFAULT_STAGE_NUMBERS: Dict[str, int] = {
    "river_centerline": 1,
    "centerline_wse_support": 2,
    "centerline_wse_proxy": 3,
    "centerline_authoritative_support": 4,
    "centerline_authoritative_bed": 5,
    "centerline_observed_offset": 6,
    "component_offset_transfer_priors": 7,
    "centerline_offset_modeled": 8,
    "centerline_bed_backbone": 9,
    "centerline_bed_backbone_dense": 10,
    "river_primary_surface": 11,
    "river_primary_surface_authoritative_applied_solve_domain": 12,
    "river_export_subset": 13,
}


def jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_stage_products_overview(
    *,
    root: str | Path,
    plan: Sequence[Any],
    stage_status: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage_numbers: Mapping[str, int] | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    stage_numbers = dict(stage_numbers or DEFAULT_STAGE_NUMBERS)
    stages = []
    for step in plan:
        stage_id = getattr(step, "stage_id", None)
        if stage_id is None:
            continue
        item = stage_status.get(stage_id, {}) if isinstance(stage_status, Mapping) else {}
        stage_no = stage_numbers.get(stage_id, 0)
        entry = {
            "stage_number": stage_no,
            "stage_id": stage_id,
            "status": item.get("status", "not_run"),
            "implemented": bool(item.get("implemented", False)),
            "description": item.get("description"),
            "output_artifact": item.get("output_artifact"),
            "receipt_path": item.get("receipt_path"),
        }
        if stage_id in stage_results:
            entry["record_count"] = getattr(stage_results[stage_id], "record_count", None)
        stages.append(entry)
    overview = {
        "root": str(root),
        "pipeline_version": str(PIPELINE_VERSION),
        "river_method_selected": "v2",
        "river_path_used": "river_v2_only",
        "legacy_river_path_participated": False,
        "success": failed_stage is None,
        "failed_stage": failed_stage,
        "error": error,
        "stages": stages,
        "lineage": [
            {
                "stage_number": entry["stage_number"],
                "stage_id": entry["stage_id"],
                "output_artifact": entry.get("output_artifact"),
                "receipt_path": entry.get("receipt_path"),
            }
            for entry in stages
        ],
    }
    return jsonable(overview)



def write_stage_products_overview(
    path: str | Path,
    *,
    root: str | Path,
    plan: Sequence[Any],
    stage_status: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage_numbers: Mapping[str, int] | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
) -> Path:
    overview = build_stage_products_overview(
        root=root,
        plan=plan,
        stage_status=stage_status,
        stage_results=stage_results,
        stage_numbers=stage_numbers,
        failed_stage=failed_stage,
        error=error,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    return out_path



def write_summary(path: str | Path, payload: Dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out_path



def write_pipeline_summary(*, path: str | Path, result: Any) -> Path:
    return write_summary(path, result.to_dict())



def write_compatibility_pass_summary(*, pass1_path: str | Path, pass2_path: str | Path, pass3_path: str | Path, pass4_path: str | Path, result: Any) -> Path | None:
    stage_ids = tuple(getattr(result, "stage_results", {}).keys())
    summary_path: Path | None = None
    if stage_ids == PASS1_STAGE_IDS:
        summary_path = Path(pass1_path)
    elif stage_ids == PASS2_STAGE_IDS:
        summary_path = Path(pass2_path)
    elif stage_ids == PASS3_STAGE_IDS:
        summary_path = Path(pass3_path)
    elif stage_ids == PASS4_STAGE_IDS:
        summary_path = Path(pass4_path)
    if summary_path is None:
        return None
    return write_summary(summary_path, result.to_dict())



def _pass_manifest_entry(pass_report: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = dict(pass_report or {})
    return {
        "execution_mode": payload.get("execution_mode"),
        "success": payload.get("success"),
        "stage_status": payload.get("stage_status", {}),
        "stage_results": payload.get("stage_results", {}),
        "failed_stage": payload.get("failed_stage"),
        "error": payload.get("error"),
    }



def build_river_v2_manifest_sections(*, report: Dict[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
    river_report = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    v2_pass1 = river_report.get("v2_pass1") if isinstance(river_report.get("v2_pass1"), dict) else {}
    v2_pass2 = river_report.get("v2_pass2") if isinstance(river_report.get("v2_pass2"), dict) else {}
    v2_pass3 = river_report.get("v2_pass3") if isinstance(river_report.get("v2_pass3"), dict) else {}
    v2_pass4 = river_report.get("v2_pass4") if isinstance(river_report.get("v2_pass4"), dict) else {}
    active = bool(v2_pass4.get("success")) and bool(outputs.get("primary_river_guidance_surface"))
    return {
        "river_v2_pass1": _pass_manifest_entry(v2_pass1),
        "river_v2_pass2": _pass_manifest_entry(v2_pass2),
        "river_v2_pass3": _pass_manifest_entry(v2_pass3),
        "river_v2_pass4": _pass_manifest_entry(v2_pass4),
        "river_v2_final_route_participation": {
            "active": active,
            "active_raster": outputs.get("primary_river_guidance_surface"),
            "active_stage": "river_primary_surface_authoritative_applied" if active else None,
            "legacy_river_final_route_participation_blocked": bool(v2_pass4.get("success")),
            "river_method_selected": "v2",
            "river_path_used": "river_v2_only",
            "legacy_river_path_participated": False,
            "pipeline_version": str(PIPELINE_VERSION),
        },
    }


def _stage_domain_role(stage_id: str) -> str:
    if stage_id == "river_export_subset":
        return "public_export_subset"
    if stage_id in {
        "river_primary_surface",
        "river_primary_surface_authoritative_applied_solve_domain",
    }:
        return "solve_domain"
    return "solve_domain"


def _receipt_input_artifacts(receipt_path: str | Path | None) -> list[str]:
    if not receipt_path:
        return []
    try:
        payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("input_artifacts")
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if v]


def build_stage_trace(
    *,
    root: str | Path,
    plan: Sequence[Any],
    stage_status: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage_numbers: Mapping[str, int] | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    support_status: str | None = None,
    canonical_system_id: str | None = None,
    canonical_solve_aoi: str | None = None,
    export_aoi: str | None = None,
) -> Dict[str, Any]:
    stage_numbers = dict(stage_numbers or DEFAULT_STAGE_NUMBERS)
    stages = []
    for step in plan:
        stage_id = getattr(step, "stage_id", None)
        if stage_id is None:
            continue
        item = stage_status.get(stage_id, {}) if isinstance(stage_status, Mapping) else {}
        receipt_path = item.get("receipt_path")
        entry = {
            "stage_number": stage_numbers.get(stage_id, 0),
            "stage_id": stage_id,
            "status": item.get("status", "not_run"),
            "implemented": bool(item.get("implemented", False)),
            "domain_role": _stage_domain_role(stage_id),
            "main_input_artifacts": _receipt_input_artifacts(receipt_path),
            "main_output_artifact": item.get("output_artifact"),
            "receipt_path": receipt_path,
            "support_status": support_status,
        }
        if stage_id in stage_results:
            entry["record_count"] = getattr(stage_results[stage_id], "record_count", None)
        stages.append(entry)
    return jsonable({
        "root": str(root),
        "pipeline_version": str(PIPELINE_VERSION),
        "river_method_selected": "v2",
        "river_path_used": "river_v2_only",
        "legacy_river_path_participated": False,
        "canonical_system_id": canonical_system_id,
        "canonical_solve_aoi": canonical_solve_aoi,
        "export_aoi": export_aoi,
        "support_status": support_status,
        "success": failed_stage is None,
        "failed_stage": failed_stage,
        "error": error,
        "stages": stages,
    })


def write_stage_trace(
    path: str | Path,
    *,
    root: str | Path,
    plan: Sequence[Any],
    stage_status: Mapping[str, Any],
    stage_results: Mapping[str, Any],
    stage_numbers: Mapping[str, int] | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    support_status: str | None = None,
    canonical_system_id: str | None = None,
    canonical_solve_aoi: str | None = None,
    export_aoi: str | None = None,
) -> Path:
    payload = build_stage_trace(
        root=root,
        plan=plan,
        stage_status=stage_status,
        stage_results=stage_results,
        stage_numbers=stage_numbers,
        failed_stage=failed_stage,
        error=error,
        support_status=support_status,
        canonical_system_id=canonical_system_id,
        canonical_solve_aoi=canonical_solve_aoi,
        export_aoi=export_aoi,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
