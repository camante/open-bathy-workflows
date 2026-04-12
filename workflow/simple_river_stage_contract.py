from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION = "legacy_structured_transition"
ROUTE_MODE_SIMPLE_RIVER_PLAN_V1 = "simple_river_plan_v1"

STAGE_AUTHORITATIVE_BASE = "authoritative_base"
STAGE_RIVER_GUIDANCE_DOMAIN = "river_guidance_domain"
STAGE_RIVER_CENTERLINE = "river_centerline"
STAGE_CENTERLINE_WSE_PROXY = "centerline_wse_proxy"
STAGE_CENTERLINE_AUTHORITATIVE_BED = "centerline_authoritative_bed"
STAGE_CENTERLINE_OBSERVED_OFFSET = "centerline_observed_offset"
STAGE_CENTERLINE_OFFSET_MODELED = "centerline_offset_modeled"
STAGE_CENTERLINE_BED_BACKBONE = "river_centerline_bed_backbone"
STAGE_RIVER_PRIMARY_SURFACE = "river_primary_surface"
STAGE_RIVER_PRIMARY_SURFACE_LOCKED = "river_primary_surface_authoritative_locked"
STAGE_CONDITIONED_FINAL_INTERNAL = "conditioned_final_dem_internal"
STAGE_FINAL_DEM = "DEM_enhanced"


@dataclass(frozen=True)
class SimpleRiverStageSpec:
    stage_id: str
    canonical_filename: str
    artifact_kind: str
    required_in_target_route: bool
    description: str


_SIMPLE_RIVER_STAGE_SPECS: Dict[str, SimpleRiverStageSpec] = {
    STAGE_AUTHORITATIVE_BASE: SimpleRiverStageSpec(STAGE_AUTHORITATIVE_BASE, "authoritative_base.tif", "raster", True, "Best available authoritative terrain/depth base."),
    STAGE_RIVER_GUIDANCE_DOMAIN: SimpleRiverStageSpec(STAGE_RIVER_GUIDANCE_DOMAIN, "river_guidance_domain_mask.tif", "raster", True, "Explicit river guidance eligibility mask."),
    STAGE_RIVER_CENTERLINE: SimpleRiverStageSpec(STAGE_RIVER_CENTERLINE, "river_centerline_points.gpkg", "vector", True, "Canonical ordered river centerline scaffold."),
    STAGE_CENTERLINE_WSE_PROXY: SimpleRiverStageSpec(STAGE_CENTERLINE_WSE_PROXY, "centerline_wse_proxy_points.gpkg", "vector", True, "Centerline points carrying a WSE proxy."),
    STAGE_CENTERLINE_AUTHORITATIVE_BED: SimpleRiverStageSpec(STAGE_CENTERLINE_AUTHORITATIVE_BED, "centerline_authoritative_bed_points.gpkg", "vector", True, "Centerline points with authoritative bed support."),
    STAGE_CENTERLINE_OBSERVED_OFFSET: SimpleRiverStageSpec(STAGE_CENTERLINE_OBSERVED_OFFSET, "centerline_observed_offset_points.gpkg", "vector", True, "Observed WSE-to-bed offsets where both inputs exist."),
    STAGE_CENTERLINE_OFFSET_MODELED: SimpleRiverStageSpec(STAGE_CENTERLINE_OFFSET_MODELED, "centerline_offset_modeled_points.gpkg", "vector", True, "Modeled longitudinal centerline offset field."),
    STAGE_CENTERLINE_BED_BACKBONE: SimpleRiverStageSpec(STAGE_CENTERLINE_BED_BACKBONE, "river_centerline_bed_backbone_points.gpkg", "vector", True, "Centerline bed backbone reconstructed from WSE and modeled offset."),
    STAGE_RIVER_PRIMARY_SURFACE: SimpleRiverStageSpec(STAGE_RIVER_PRIMARY_SURFACE, "river_primary_surface.tif", "raster", True, "Channel-wide primary river surface from the centerline backbone."),
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED: SimpleRiverStageSpec(STAGE_RIVER_PRIMARY_SURFACE_LOCKED, "river_primary_surface_authoritative_locked.tif", "raster", True, "Primary river surface after authoritative lock."),
    STAGE_CONDITIONED_FINAL_INTERNAL: SimpleRiverStageSpec(STAGE_CONDITIONED_FINAL_INTERNAL, "combined/conditioned_final_dem_internal.tif", "raster", True, "Single internal conditioned final DEM."),
    STAGE_FINAL_DEM: SimpleRiverStageSpec(STAGE_FINAL_DEM, "combined/DEM_enhanced.tif", "raster", True, "Official final enhanced DEM written once."),
}


def simple_river_stage_sequence() -> List[str]:
    return list(_SIMPLE_RIVER_STAGE_SPECS.keys())


def simple_river_stage_spec(stage_id: str) -> SimpleRiverStageSpec:
    try:
        return _SIMPLE_RIVER_STAGE_SPECS[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown simple river stage_id: {stage_id}") from exc


def simple_river_stage_filename(stage_id: str) -> str:
    return simple_river_stage_spec(stage_id).canonical_filename


def simple_river_required_stage_ids() -> List[str]:
    return [stage_id for stage_id, spec in _SIMPLE_RIVER_STAGE_SPECS.items() if spec.required_in_target_route]


def simple_river_stage_status_placeholder() -> Dict[str, dict]:
    return {
        stage_id: {
            "canonical_filename": spec.canonical_filename,
            "artifact_kind": spec.artifact_kind,
            "implemented": False,
            "status": "not_yet_implemented",
            "legacy_equivalent": None,
            "required_in_target_route": spec.required_in_target_route,
            "description": spec.description,
        }
        for stage_id, spec in _SIMPLE_RIVER_STAGE_SPECS.items()
    }



def mark_stage_implemented(
    stage_status: Dict[str, dict],
    *,
    stage_id: str,
    output_artifact: str,
    record_count: int | None = None,
    receipt_path: str | None = None,
    warnings: List[str] | None = None,
) -> Dict[str, dict]:
    updated = {str(k): dict(v) for k, v in stage_status.items()}
    if stage_id not in updated:
        raise ValueError(f"Unknown simple river stage_id: {stage_id}")
    item = updated[stage_id]
    item["implemented"] = True
    item["status"] = "implemented"
    item["output_artifact"] = str(output_artifact)
    if record_count is not None:
        item["record_count"] = int(record_count)
    if receipt_path:
        item["receipt_path"] = str(receipt_path)
    if warnings is not None:
        item["warnings"] = [str(v) for v in warnings]
    return updated


def mark_stage_failed(
    stage_status: Dict[str, dict],
    *,
    stage_id: str,
    error: str,
    warnings: List[str] | None = None,
) -> Dict[str, dict]:
    updated = {str(k): dict(v) for k, v in stage_status.items()}
    if stage_id not in updated:
        raise ValueError(f"Unknown simple river stage_id: {stage_id}")
    item = updated[stage_id]
    item["implemented"] = False
    item["status"] = "failed"
    item["error"] = str(error)
    if warnings is not None:
        item["warnings"] = [str(v) for v in warnings]
    return updated

def simple_river_stage_contract_summary() -> dict:
    return {
        "route_modes": {
            "current_default": ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
            "target": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        },
        "stage_sequence": simple_river_stage_sequence(),
        "required_stage_ids": simple_river_required_stage_ids(),
        "stages": {
            stage_id: {
                "canonical_filename": spec.canonical_filename,
                "artifact_kind": spec.artifact_kind,
                "required_in_target_route": spec.required_in_target_route,
                "description": spec.description,
            }
            for stage_id, spec in _SIMPLE_RIVER_STAGE_SPECS.items()
        },
    }


__all__ = [
    "ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION",
    "ROUTE_MODE_SIMPLE_RIVER_PLAN_V1",
    "STAGE_AUTHORITATIVE_BASE",
    "STAGE_RIVER_GUIDANCE_DOMAIN",
    "STAGE_RIVER_CENTERLINE",
    "STAGE_CENTERLINE_WSE_PROXY",
    "STAGE_CENTERLINE_AUTHORITATIVE_BED",
    "STAGE_CENTERLINE_OBSERVED_OFFSET",
    "STAGE_CENTERLINE_OFFSET_MODELED",
    "STAGE_CENTERLINE_BED_BACKBONE",
    "STAGE_RIVER_PRIMARY_SURFACE",
    "STAGE_RIVER_PRIMARY_SURFACE_LOCKED",
    "STAGE_CONDITIONED_FINAL_INTERNAL",
    "STAGE_FINAL_DEM",
    "SimpleRiverStageSpec",
    "simple_river_stage_sequence",
    "simple_river_stage_spec",
    "simple_river_stage_filename",
    "simple_river_required_stage_ids",
    "simple_river_stage_status_placeholder",
    "simple_river_stage_contract_summary",
    "mark_stage_implemented",
    "mark_stage_failed",
]
