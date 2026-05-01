from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


RIVER_V2_METHODS = ("v2", "simple_v2")
RIVER_V2_ROUTE_MODE_PASS1 = "river_v2_pass1_evidence_only"
RIVER_V2_ROUTE_MODE_PASS2 = "river_v2_pass2_centerline_backbone_only"
RIVER_V2_ROUTE_MODE_PASS3 = "river_v2_pass3_primary_surface_only"
RIVER_V2_ROUTE_MODE_PASS4 = "river_v2_pass4_authoritative_lock_ready_for_final_dem"
RIVER_V2_TARGET_MODE = "river_v2_pipeline"

STAGE_RIVER_CENTERLINE = "river_centerline"
STAGE_CENTERLINE_WSE_SUPPORT = "centerline_wse_support"
STAGE_CENTERLINE_WSE_PROXY = "centerline_wse_proxy"
STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT = "centerline_authoritative_support"
STAGE_CENTERLINE_AUTHORITATIVE_BED = "centerline_authoritative_bed"
STAGE_CENTERLINE_OBSERVED_OFFSET = "centerline_observed_offset"
STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS = "component_offset_transfer_priors"
STAGE_CENTERLINE_OFFSET_MODELED = "centerline_offset_modeled"
STAGE_CENTERLINE_BED_BACKBONE = "river_centerline_bed_backbone"
STAGE_CENTERLINE_BED_BACKBONE_DENSE = "river_centerline_bed_backbone_dense"
STAGE_RIVER_PRIMARY_SURFACE_SUPPORT = "river_primary_surface_support"
STAGE_RIVER_PRIMARY_SURFACE = "river_primary_surface"
STAGE_RIVER_PRIMARY_SURFACE_LOCKED = "river_primary_surface_authoritative_applied_solve_domain"
STAGE_RIVER_EXPORT_SUBSET = "river_export_subset"

PASS1_STAGE_IDS = (
    STAGE_RIVER_CENTERLINE,
    STAGE_CENTERLINE_WSE_SUPPORT,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
)
PASS2_STAGE_IDS = PASS1_STAGE_IDS + (
    STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
)
PASS3_STAGE_IDS = PASS2_STAGE_IDS + (
    STAGE_RIVER_PRIMARY_SURFACE,
)
PASS4_STAGE_IDS = PASS3_STAGE_IDS + (
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
    STAGE_RIVER_EXPORT_SUBSET,
)


@dataclass(frozen=True)
class RiverV2StageSpec:
    stage_id: str
    canonical_filename: str
    description: str


_STAGE_SPECS: Dict[str, RiverV2StageSpec] = {
    STAGE_RIVER_CENTERLINE: RiverV2StageSpec(
        stage_id=STAGE_RIVER_CENTERLINE,
        canonical_filename="river_centerline_points.gpkg",
        description="Canonical ordered river centerline scaffold for River v2.",
    ),
    STAGE_CENTERLINE_WSE_SUPPORT: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_WSE_SUPPORT,
        canonical_filename="centerline_wse_support_points.gpkg",
        description="Explicit centerline WSE support points prepared for River v2 before WSE proxy estimation.",
    ),
    STAGE_CENTERLINE_WSE_PROXY: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_WSE_PROXY,
        canonical_filename="centerline_wse_proxy_points.gpkg",
        description="Centerline points carrying a WSE proxy for River v2.",
    ),
    STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
        canonical_filename="authoritative_sampling_support.tif",
        description="Explicit authoritative support raster selected for River v2 centerline bed sampling.",
    ),
    STAGE_CENTERLINE_AUTHORITATIVE_BED: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
        canonical_filename="centerline_authoritative_bed_points.gpkg",
        description="Centerline points with authoritative bed support for River v2.",
    ),
    STAGE_CENTERLINE_OBSERVED_OFFSET: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET,
        canonical_filename="centerline_observed_offset_points.gpkg",
        description="Observed WSE-to-bed offsets where both inputs exist for River v2.",
    ),
    STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS: RiverV2StageSpec(
        stage_id=STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
        canonical_filename="component_offset_transfer_priors.gpkg",
        description="Component-level transferred offset priors derived from connected observed offsets for River v2.",
    ),
    STAGE_CENTERLINE_OFFSET_MODELED: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_OFFSET_MODELED,
        canonical_filename="centerline_offset_modeled_points.gpkg",
        description="Modeled centerline bed depth below WSE for River v2, using local observed offsets first, transferred component priors second, and explicit placeholder depth only as a last resort.",
    ),
    STAGE_CENTERLINE_BED_BACKBONE: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE,
        canonical_filename="river_centerline_bed_backbone_points.gpkg",
        description="Centerline bed backbone reconstructed from WSE and modeled offset for River v2.",
    ),
    STAGE_CENTERLINE_BED_BACKBONE_DENSE: RiverV2StageSpec(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE_DENSE,
        canonical_filename="river_centerline_bed_backbone_dense_points.gpkg",
        description="Densified centerline bed backbone points interpolated between sparse backbone anchors for River v2.",
    ),
    STAGE_RIVER_PRIMARY_SURFACE_SUPPORT: RiverV2StageSpec(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_SUPPORT,
        canonical_filename="river_primary_surface_domain.tif",
        description="Explicit River v2 primary-surface fill domain derived from the river domain and centerline.",
    ),
    STAGE_RIVER_PRIMARY_SURFACE: RiverV2StageSpec(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE,
        canonical_filename="river_primary_surface_solve_domain.tif",
        description="Internal solve-domain River v2 primary surface raster derived from the centerline backbone.",
    ),
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED: RiverV2StageSpec(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
        canonical_filename="river_primary_surface_authoritative_applied_solve_domain.tif",
        description="Internal solve-domain River v2 primary surface after authoritative lock.",
    ),
    STAGE_RIVER_EXPORT_SUBSET: RiverV2StageSpec(
        stage_id=STAGE_RIVER_EXPORT_SUBSET,
        canonical_filename="river_primary_surface_authoritative_applied.tif",
        description="Public export subset of the canonical solve-domain River v2 rasters.",
    ),
}


@dataclass
class RiverV2StageResult:
    stage_id: str
    output_artifact: Path
    receipt_path: Optional[Path]
    record_count: int
    validation: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    aux_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "stage_id": str(self.stage_id),
            "output_artifact": str(self.output_artifact),
            "record_count": int(self.record_count),
            "validation": dict(self.validation or {}),
            "warnings": [str(v) for v in self.warnings],
            "aux_outputs": {str(k): str(v) for k, v in (self.aux_outputs or {}).items()},
        }
        if self.receipt_path is not None:
            payload["receipt_path"] = str(self.receipt_path)
        return payload


@dataclass
class RiverV2RunResult:
    success: bool
    execution_mode: str = RIVER_V2_TARGET_MODE
    stage_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage_results: Dict[str, RiverV2StageResult] = field(default_factory=dict)
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    aux_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "execution_mode": str(self.execution_mode),
            "stage_status": {str(k): dict(v) for k, v in (self.stage_status or {}).items()},
            "stage_results": {str(k): v.to_dict() for k, v in (self.stage_results or {}).items()},
            "failed_stage": None if self.failed_stage is None else str(self.failed_stage),
            "error": None if self.error is None else str(self.error),
            "aux_outputs": {str(k): str(v) for k, v in (self.aux_outputs or {}).items()},
        }


RiverV2Pass1Result = RiverV2RunResult
RiverV2Pass2Result = RiverV2RunResult
RiverV2Pass3Result = RiverV2RunResult
RiverV2Pass4Result = RiverV2RunResult


def river_v2_stage_sequence() -> List[str]:
    return [
        STAGE_RIVER_CENTERLINE,
        STAGE_CENTERLINE_WSE_SUPPORT,
        STAGE_CENTERLINE_WSE_PROXY,
        STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
        STAGE_CENTERLINE_AUTHORITATIVE_BED,
        STAGE_CENTERLINE_OBSERVED_OFFSET,
        STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
        STAGE_CENTERLINE_OFFSET_MODELED,
        STAGE_CENTERLINE_BED_BACKBONE,
        STAGE_CENTERLINE_BED_BACKBONE_DENSE,
        STAGE_RIVER_PRIMARY_SURFACE,
        STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
        STAGE_RIVER_EXPORT_SUBSET,
    ]


def river_v2_stage_spec(stage_id: str) -> RiverV2StageSpec:
    try:
        return _STAGE_SPECS[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unknown River v2 stage_id: {stage_id}") from exc


def river_v2_stage_status_placeholder(stage_ids: tuple[str, ...] | list[str] | None = None) -> Dict[str, Dict[str, Any]]:
    active_ids = list(stage_ids or river_v2_stage_sequence())
    return {
        stage_id: {
            "stage_id": stage_id,
            "canonical_filename": river_v2_stage_spec(stage_id).canonical_filename,
            "description": river_v2_stage_spec(stage_id).description,
            "implemented": False,
            "status": "not_run",
        }
        for stage_id in active_ids
    }


def mark_v2_stage_implemented(
    stage_status: Dict[str, Dict[str, Any]],
    *,
    stage_id: str,
    output_artifact: str | Path,
    record_count: int,
    receipt_path: str | Path | None = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    updated = {str(k): dict(v) for k, v in (stage_status or {}).items()}
    item = updated.setdefault(stage_id, {"stage_id": stage_id})
    item.update(
        {
            "implemented": True,
            "status": "implemented",
            "output_artifact": str(output_artifact),
            "record_count": int(record_count),
            "warnings": [str(v) for v in (warnings or [])],
        }
    )
    if receipt_path is not None:
        item["receipt_path"] = str(receipt_path)
    return updated


def mark_v2_stage_failed(
    stage_status: Dict[str, Dict[str, Any]],
    *,
    stage_id: str,
    error: str,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    updated = {str(k): dict(v) for k, v in (stage_status or {}).items()}
    item = updated.setdefault(stage_id, {"stage_id": stage_id})
    item.update(
        {
            "implemented": False,
            "status": "failed",
            "error": str(error),
            "warnings": [str(v) for v in (warnings or [])],
        }
    )
    return updated


def subset_stage_status(stage_status: Dict[str, Dict[str, Any]], stage_ids: tuple[str, ...] | list[str]) -> Dict[str, Dict[str, Any]]:
    return {str(stage_id): dict(stage_status[stage_id]) for stage_id in stage_ids if stage_id in stage_status}


def subset_stage_results(stage_results: Dict[str, RiverV2StageResult], stage_ids: tuple[str, ...] | list[str]) -> Dict[str, RiverV2StageResult]:
    return {str(stage_id): stage_results[stage_id] for stage_id in stage_ids if stage_id in stage_results}


__all__ = [
    "RIVER_V2_METHODS",
    "RIVER_V2_ROUTE_MODE_PASS1",
    "RIVER_V2_ROUTE_MODE_PASS2",
    "RIVER_V2_ROUTE_MODE_PASS3",
    "RIVER_V2_TARGET_MODE",
    "RIVER_V2_ROUTE_MODE_PASS4",
    "STAGE_RIVER_CENTERLINE",
    "STAGE_CENTERLINE_WSE_PROXY",
    "STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT",
    "STAGE_CENTERLINE_AUTHORITATIVE_BED",
    "STAGE_CENTERLINE_OBSERVED_OFFSET",
    "STAGE_CENTERLINE_OFFSET_MODELED",
    "STAGE_CENTERLINE_BED_BACKBONE",
    "STAGE_RIVER_PRIMARY_SURFACE_SUPPORT",
    "STAGE_RIVER_PRIMARY_SURFACE",
    "STAGE_RIVER_PRIMARY_SURFACE_LOCKED",
    "PASS1_STAGE_IDS",
    "PASS2_STAGE_IDS",
    "PASS3_STAGE_IDS",
    "PASS4_STAGE_IDS",
    "RiverV2StageSpec",
    "RiverV2StageResult",
    "RiverV2RunResult",
    "RiverV2Pass1Result",
    "RiverV2Pass2Result",
    "RiverV2Pass3Result",
    "RiverV2Pass4Result",
    "river_v2_stage_sequence",
    "river_v2_stage_spec",
    "river_v2_stage_status_placeholder",
    "mark_v2_stage_implemented",
    "mark_v2_stage_failed",
    "subset_stage_status",
    "subset_stage_results",
]
