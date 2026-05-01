from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pipeline.river_workflow.river_workflow_paths import RiverWorkflowPaths, build_river_workflow_paths
from repo_runtime_modes import ACTIVE_RIVER_WORKFLOW, normalize_active_river_workflow_name
from pipeline.river_workflow.river_workflow_execution_contract import CANONICAL_BUILD_ROLE, normalize_execution_role


@dataclass(frozen=True)
class RiverWorkflowSourceBundle:
    solve_source_aoi: str
    export_aoi: str
    projected_crs: str
    target_resolution_m: float
    solve_network_gpkg: Path
    export_network_gpkg: Path
    requested_solve_domain: str | None = None
    resolved_solve_domain: str | None = None
    solve_domain_source: str = "derived_from_aoi"
    solve_authoritative_source_path: Path | None = None
    export_authoritative_source_path: Path | None = None
    export_baseline_source_path: Path | None = None
    authoritative_source_contract_path: Path | None = None
    baseline_source_contract_path: Path | None = None
    canonical_solve_network_source_gpkg: Path | None = None
    canonical_solve_receipt_source_json: Path | None = None
    canonical_solve_contract_path: Path | None = None
    canonical_solve_grid_template_path: Path | None = None
    canonical_solve_authoritative_measured_only_path: Path | None = None
    canonical_solve_authoritative_support_mask_path: Path | None = None
    canonical_solve_baseline_background_path: Path | None = None
    canonical_solve_take_mask_path: Path | None = None
    canonical_solve_final_dem_path: Path | None = None
    canonical_solve_outputs_root: Path | None = None
    canonical_solve_cache_manifest_path: Path | None = None
    canonical_solve_cache_key: str | None = None
    canonical_source_context_path: Path | None = None
    canonical_solve_identity_path: Path | None = None
    canonical_identity_receipt_path: Path | None = None
    canonical_domain_bounds: str | None = None
    canonical_trace_distance_km: float | None = None
    river_network_fingerprint: str | None = None
    authoritative_source_fingerprint: str | None = None
    export_authoritative_source_role: str = "export_only"
    solve_authoritative_source_role: str = "shared_pending_canonical_prepare"
    authoritative_routing_policy: str = "canonical_only_after_prepare"
    canonical_solve_aoi: str | None = None
    canonical_system_id: str | None = None
    canonical_solve_stop_reason: str | None = None
    canonical_selected_reach_count: int | None = None
    canonical_selected_reach_length_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "solve_source_aoi": str(self.solve_source_aoi),
            "export_aoi": str(self.export_aoi),
            "requested_solve_domain": str(self.requested_solve_domain) if self.requested_solve_domain is not None else None,
            "resolved_solve_domain": str(self.resolved_solve_domain) if self.resolved_solve_domain is not None else None,
            "solve_domain_source": str(self.solve_domain_source),
            "projected_crs": str(self.projected_crs),
            "target_resolution_m": float(self.target_resolution_m),
            "solve_network_gpkg": str(self.solve_network_gpkg),
            "export_network_gpkg": str(self.export_network_gpkg),
            "solve_authoritative_source_path": str(self.solve_authoritative_source_path) if self.solve_authoritative_source_path is not None else None,
            "export_authoritative_source_path": str(self.export_authoritative_source_path) if self.export_authoritative_source_path is not None else None,
            "export_baseline_source_path": str(self.export_baseline_source_path) if self.export_baseline_source_path is not None else None,
            "authoritative_source_contract_path": str(self.authoritative_source_contract_path) if self.authoritative_source_contract_path is not None else None,
            "baseline_source_contract_path": str(self.baseline_source_contract_path) if self.baseline_source_contract_path is not None else None,
            "canonical_solve_network_source_gpkg": str(self.canonical_solve_network_source_gpkg) if self.canonical_solve_network_source_gpkg is not None else None,
            "canonical_solve_receipt_source_json": str(self.canonical_solve_receipt_source_json) if self.canonical_solve_receipt_source_json is not None else None,
            "canonical_solve_contract_path": str(self.canonical_solve_contract_path) if self.canonical_solve_contract_path is not None else None,
            "canonical_solve_grid_template_path": str(self.canonical_solve_grid_template_path) if self.canonical_solve_grid_template_path is not None else None,
            "canonical_solve_authoritative_measured_only_path": str(self.canonical_solve_authoritative_measured_only_path) if self.canonical_solve_authoritative_measured_only_path is not None else None,
            "canonical_solve_authoritative_support_mask_path": str(self.canonical_solve_authoritative_support_mask_path) if self.canonical_solve_authoritative_support_mask_path is not None else None,
            "canonical_solve_baseline_background_path": str(self.canonical_solve_baseline_background_path) if self.canonical_solve_baseline_background_path is not None else None,
            "canonical_solve_take_mask_path": str(self.canonical_solve_take_mask_path) if self.canonical_solve_take_mask_path is not None else None,
            "canonical_solve_final_dem_path": str(self.canonical_solve_final_dem_path) if self.canonical_solve_final_dem_path is not None else None,
            "canonical_solve_outputs_root": str(self.canonical_solve_outputs_root) if self.canonical_solve_outputs_root is not None else None,
            "canonical_solve_cache_manifest_path": str(self.canonical_solve_cache_manifest_path) if self.canonical_solve_cache_manifest_path is not None else None,
            "canonical_solve_cache_key": str(self.canonical_solve_cache_key) if self.canonical_solve_cache_key is not None else None,
            "canonical_cache_key": str(self.canonical_solve_cache_key) if self.canonical_solve_cache_key is not None else None,
            "canonical_source_context_path": str(self.canonical_source_context_path) if self.canonical_source_context_path is not None else None,
            "canonical_solve_identity_path": str(self.canonical_solve_identity_path) if self.canonical_solve_identity_path is not None else None,
            "canonical_identity_receipt_path": str(self.canonical_identity_receipt_path) if self.canonical_identity_receipt_path is not None else None,
            "canonical_domain_bounds": str(self.canonical_domain_bounds) if self.canonical_domain_bounds is not None else None,
            "canonical_trace_distance_km": float(self.canonical_trace_distance_km) if self.canonical_trace_distance_km is not None else None,
            "river_network_fingerprint": str(self.river_network_fingerprint) if self.river_network_fingerprint is not None else None,
            "authoritative_source_fingerprint": str(self.authoritative_source_fingerprint) if self.authoritative_source_fingerprint is not None else None,
            "export_authoritative_source_role": str(self.export_authoritative_source_role),
            "solve_authoritative_source_role": str(self.solve_authoritative_source_role),
            "authoritative_routing_policy": str(self.authoritative_routing_policy),
            "canonical_solve_aoi": str(self.canonical_solve_aoi) if self.canonical_solve_aoi is not None else None,
            "canonical_system_id": str(self.canonical_system_id) if self.canonical_system_id is not None else None,
            "canonical_solve_stop_reason": str(self.canonical_solve_stop_reason) if self.canonical_solve_stop_reason is not None else None,
            "canonical_selected_reach_count": int(self.canonical_selected_reach_count) if self.canonical_selected_reach_count is not None else None,
            "canonical_selected_reach_length_m": float(self.canonical_selected_reach_length_m) if self.canonical_selected_reach_length_m is not None else None,
        }


@dataclass(frozen=True)
class CanonicalRiverSourceContext:
    export_aoi: str
    shared_source_aoi: str
    canonical_solve_aoi: str
    projected_crs: str
    target_resolution_m: float
    canonical_system_id: str | None
    canonical_solve_stop_reason: str | None
    canonical_selected_reach_count: int | None
    canonical_selected_reach_length_m: float | None
    canonical_network_gpkg: Path
    requested_solve_domain: str | None = None
    resolved_solve_domain: str | None = None
    solve_domain_source: str = "derived_from_aoi"
    canonical_receipt_json: Path | None = None
    canonical_solve_contract_path: Path | None = None
    canonical_solve_grid_template_path: Path | None = None
    canonical_solve_authoritative_measured_only_path: Path | None = None
    canonical_solve_authoritative_support_mask_path: Path | None = None
    canonical_solve_baseline_background_path: Path | None = None
    canonical_support_coverage_path: Path | None = None
    canonical_selected_tiles_csv: Path | None = None
    canonical_materialization_meta_path: Path | None = None
    export_authoritative_source_path: Path | None = None
    export_baseline_source_path: Path | None = None
    authoritative_source_contract_path: Path | None = None
    baseline_source_contract_path: Path | None = None
    source_policy: str = "one_canonical_authoritative_source"
    canonical_solve_identity_path: Path | None = None
    canonical_identity_receipt_path: Path | None = None
    canonical_domain_bounds: str | None = None
    canonical_trace_distance_km: float | None = None
    river_network_fingerprint: str | None = None
    authoritative_source_fingerprint: str | None = None
    resolution_policy: str = "round_to_0.01m"
    raw_target_resolution_m: float | None = None
    canonical_solve_cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_aoi": str(self.export_aoi),
            "requested_solve_domain": str(self.requested_solve_domain) if self.requested_solve_domain is not None else None,
            "resolved_solve_domain": str(self.resolved_solve_domain) if self.resolved_solve_domain is not None else None,
            "solve_domain_source": str(self.solve_domain_source),
            "shared_source_aoi": str(self.shared_source_aoi),
            "canonical_solve_aoi": str(self.canonical_solve_aoi),
            "projected_crs": str(self.projected_crs),
            "target_resolution_m": float(self.target_resolution_m),
            "canonical_system_id": str(self.canonical_system_id) if self.canonical_system_id is not None else None,
            "canonical_solve_stop_reason": str(self.canonical_solve_stop_reason) if self.canonical_solve_stop_reason is not None else None,
            "canonical_selected_reach_count": int(self.canonical_selected_reach_count) if self.canonical_selected_reach_count is not None else None,
            "canonical_selected_reach_length_m": float(self.canonical_selected_reach_length_m) if self.canonical_selected_reach_length_m is not None else None,
            "canonical_network_gpkg": str(self.canonical_network_gpkg),
            "canonical_receipt_json": str(self.canonical_receipt_json) if self.canonical_receipt_json is not None else None,
            "canonical_solve_contract_path": str(self.canonical_solve_contract_path) if self.canonical_solve_contract_path is not None else None,
            "canonical_solve_grid_template_path": str(self.canonical_solve_grid_template_path) if self.canonical_solve_grid_template_path is not None else None,
            "canonical_solve_authoritative_measured_only_path": str(self.canonical_solve_authoritative_measured_only_path) if self.canonical_solve_authoritative_measured_only_path is not None else None,
            "canonical_solve_authoritative_support_mask_path": str(self.canonical_solve_authoritative_support_mask_path) if self.canonical_solve_authoritative_support_mask_path is not None else None,
            "canonical_solve_baseline_background_path": str(self.canonical_solve_baseline_background_path) if self.canonical_solve_baseline_background_path is not None else None,
            "canonical_support_coverage_path": str(self.canonical_support_coverage_path) if self.canonical_support_coverage_path is not None else None,
            "canonical_selected_tiles_csv": str(self.canonical_selected_tiles_csv) if self.canonical_selected_tiles_csv is not None else None,
            "canonical_materialization_meta_path": str(self.canonical_materialization_meta_path) if self.canonical_materialization_meta_path is not None else None,
            "export_authoritative_source_path": str(self.export_authoritative_source_path) if self.export_authoritative_source_path is not None else None,
            "export_baseline_source_path": str(self.export_baseline_source_path) if self.export_baseline_source_path is not None else None,
            "authoritative_source_contract_path": str(self.authoritative_source_contract_path) if self.authoritative_source_contract_path is not None else None,
            "baseline_source_contract_path": str(self.baseline_source_contract_path) if self.baseline_source_contract_path is not None else None,
            "source_policy": str(self.source_policy),
            "canonical_solve_identity_path": str(self.canonical_solve_identity_path) if self.canonical_solve_identity_path is not None else None,
            "canonical_identity_receipt_path": str(self.canonical_identity_receipt_path) if self.canonical_identity_receipt_path is not None else None,
            "canonical_domain_bounds": str(self.canonical_domain_bounds) if self.canonical_domain_bounds is not None else None,
            "canonical_trace_distance_km": float(self.canonical_trace_distance_km) if self.canonical_trace_distance_km is not None else None,
            "river_network_fingerprint": str(self.river_network_fingerprint) if self.river_network_fingerprint is not None else None,
            "authoritative_source_fingerprint": str(self.authoritative_source_fingerprint) if self.authoritative_source_fingerprint is not None else None,
            "resolution_policy": str(self.resolution_policy),
            "raw_target_resolution_m": float(self.raw_target_resolution_m) if self.raw_target_resolution_m is not None else None,
            "canonical_solve_cache_key": str(self.canonical_solve_cache_key) if self.canonical_solve_cache_key is not None else None,
        }


    @classmethod
    def from_json(cls, path: Path) -> "CanonicalRiverSourceContext":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            export_aoi=str(payload["export_aoi"]),
            shared_source_aoi=str(payload["shared_source_aoi"]),
            requested_solve_domain=(str(payload.get("requested_solve_domain")) if payload.get("requested_solve_domain") is not None else None),
            resolved_solve_domain=(str(payload.get("resolved_solve_domain")) if payload.get("resolved_solve_domain") is not None else None),
            solve_domain_source=str(payload.get("solve_domain_source", "derived_from_aoi")),
            canonical_solve_aoi=str(payload["canonical_solve_aoi"]),
            projected_crs=str(payload["projected_crs"]),
            target_resolution_m=float(payload["target_resolution_m"]),
            canonical_system_id=(str(payload.get("canonical_system_id")) if payload.get("canonical_system_id") is not None else None),
            canonical_solve_stop_reason=(str(payload.get("canonical_solve_stop_reason")) if payload.get("canonical_solve_stop_reason") is not None else None),
            canonical_selected_reach_count=(int(payload.get("canonical_selected_reach_count")) if payload.get("canonical_selected_reach_count") is not None else None),
            canonical_selected_reach_length_m=(float(payload.get("canonical_selected_reach_length_m")) if payload.get("canonical_selected_reach_length_m") is not None else None),
            canonical_network_gpkg=Path(payload["canonical_network_gpkg"]),
            canonical_receipt_json=(Path(payload["canonical_receipt_json"]) if payload.get("canonical_receipt_json") else None),
            canonical_solve_contract_path=(Path(payload["canonical_solve_contract_path"]) if payload.get("canonical_solve_contract_path") else None),
            canonical_solve_grid_template_path=(Path(payload["canonical_solve_grid_template_path"]) if payload.get("canonical_solve_grid_template_path") else None),
            canonical_solve_authoritative_measured_only_path=(Path(payload["canonical_solve_authoritative_measured_only_path"]) if payload.get("canonical_solve_authoritative_measured_only_path") else None),
            canonical_solve_authoritative_support_mask_path=(Path(payload["canonical_solve_authoritative_support_mask_path"]) if payload.get("canonical_solve_authoritative_support_mask_path") else None),
            canonical_solve_baseline_background_path=(Path(payload["canonical_solve_baseline_background_path"]) if payload.get("canonical_solve_baseline_background_path") else None),
            canonical_support_coverage_path=(Path(payload["canonical_support_coverage_path"]) if payload.get("canonical_support_coverage_path") else None),
            canonical_selected_tiles_csv=(Path(payload["canonical_selected_tiles_csv"]) if payload.get("canonical_selected_tiles_csv") else None),
            canonical_materialization_meta_path=(Path(payload["canonical_materialization_meta_path"]) if payload.get("canonical_materialization_meta_path") else None),
            export_authoritative_source_path=(Path(payload["export_authoritative_source_path"]) if payload.get("export_authoritative_source_path") else None),
            export_baseline_source_path=(Path(payload["export_baseline_source_path"]) if payload.get("export_baseline_source_path") else None),
            authoritative_source_contract_path=(Path(payload["authoritative_source_contract_path"]) if payload.get("authoritative_source_contract_path") else None),
            baseline_source_contract_path=(Path(payload["baseline_source_contract_path"]) if payload.get("baseline_source_contract_path") else None),
            source_policy=str(payload.get("source_policy", "one_canonical_authoritative_source")),
            canonical_solve_identity_path=(Path(payload["canonical_solve_identity_path"]) if payload.get("canonical_solve_identity_path") else None),
            resolution_policy=str(payload.get("resolution_policy", "round_to_0.01m")),
            raw_target_resolution_m=(float(payload.get("raw_target_resolution_m")) if payload.get("raw_target_resolution_m") is not None else None),
            canonical_solve_cache_key=(str(payload.get("canonical_solve_cache_key")) if payload.get("canonical_solve_cache_key") is not None else None),
        )

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class RiverWorkflowContext:
    cfg: Any
    out_dir: Path
    export_aoi: str
    projected_crs: str
    target_resolution_m: float
    run_id: str
    workflow_name: str = "river_workflow"
    canonical_max_trace_km: float = 200.0
    export_network_gpkg: Path | None = None
    network_gpkg: Path | None = None
    linear_inputs: RiverWorkflowSourceBundle | None = None
    requested_solve_domain: str | None = None
    resolved_solve_domain: str | None = None
    solve_domain_source: str = "derived_from_aoi"
    execution_role: str = CANONICAL_BUILD_ROLE
    logger_name: str = "river_workflow"
    adjacent_aoi_peer_run_dir: Path | None = None
    write_diagnostics: bool = True
    write_core_receipts: bool = True
    paths: RiverWorkflowPaths | None = None
    canonical_system_id: str | None = None
    canonical_parent_dem: Path | None = None
    aoi_export_dem: Path | None = None
    final_user_dem: Path | None = None
    aoi_export_receipt: Path | None = None
    final_materialization_receipt: Path | None = None
    canonical_identity_receipt_path: Path | None = None
    canonical_domain_bounds: str | None = None
    canonical_trace_distance_km: float | None = None
    river_network_fingerprint: str | None = None
    authoritative_source_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_role", normalize_execution_role(self.execution_role))
        selected_name = str(self.workflow_name or '').strip()
        object.__setattr__(self, "workflow_name", normalize_active_river_workflow_name(selected_name or ACTIVE_RIVER_WORKFLOW))
        if self.linear_inputs is None:
            object.__setattr__(
                self,
                "linear_inputs",
                RiverWorkflowSourceBundle(
                    solve_source_aoi=str(self.export_aoi),
                    export_aoi=str(self.export_aoi),
                    projected_crs=str(self.projected_crs),
                    target_resolution_m=float(self.target_resolution_m),
                    solve_network_gpkg=Path(self.network_gpkg),
                    export_network_gpkg=Path(self.export_network_gpkg),
                ),
            )
        if self.requested_solve_domain is None:
            object.__setattr__(
                self,
                "requested_solve_domain",
                (str(self.linear_inputs.requested_solve_domain) if self.linear_inputs is not None and self.linear_inputs.requested_solve_domain is not None else None),
            )
        if self.resolved_solve_domain is None:
            resolved = None
            if self.linear_inputs is not None:
                resolved = self.linear_inputs.resolved_solve_domain or self.linear_inputs.solve_source_aoi
            object.__setattr__(self, "resolved_solve_domain", (str(resolved) if resolved is not None else None))
        bundle_solve_domain_source = None
        if self.linear_inputs is not None:
            bundle_solve_domain_source = str(self.linear_inputs.solve_domain_source or "derived_from_aoi")
        if bundle_solve_domain_source is not None and (
            (not str(self.solve_domain_source or "").strip())
            or (str(self.solve_domain_source) == "derived_from_aoi" and bundle_solve_domain_source != "derived_from_aoi")
        ):
            object.__setattr__(self, "solve_domain_source", bundle_solve_domain_source)
        if self.paths is None:
            object.__setattr__(self, "paths", build_river_workflow_paths(self.out_dir, write_diagnostics=bool(self.write_diagnostics), write_core_receipts=bool(self.write_core_receipts)))
        if self.canonical_system_id is None and self.linear_inputs is not None:
            object.__setattr__(self, "canonical_system_id", self.linear_inputs.canonical_system_id)
        if self.canonical_parent_dem is None:
            object.__setattr__(self, "canonical_parent_dem", self.paths.canonical_parent_dem)
        if self.aoi_export_dem is None:
            object.__setattr__(self, "aoi_export_dem", self.paths.aoi_export_dem)
        if self.final_user_dem is None:
            object.__setattr__(self, "final_user_dem", self.paths.final_user_dem)
        if self.aoi_export_receipt is None:
            object.__setattr__(self, "aoi_export_receipt", self.paths.aoi_export_receipt)
        if self.final_materialization_receipt is None:
            object.__setattr__(self, "final_materialization_receipt", self.paths.final_materialization_receipt)
        if self.canonical_identity_receipt_path is None:
            object.__setattr__(self, "canonical_identity_receipt_path", self.paths.canonical_system_identity_receipt)
        if self.canonical_trace_distance_km is None:
            object.__setattr__(self, "canonical_trace_distance_km", float(self.canonical_max_trace_km))
        if self.canonical_domain_bounds is None and self.linear_inputs is not None and self.linear_inputs.canonical_solve_aoi is not None:
            object.__setattr__(self, "canonical_domain_bounds", str(self.linear_inputs.canonical_solve_aoi))
        if self.river_network_fingerprint is None and self.linear_inputs is not None:
            object.__setattr__(self, "river_network_fingerprint", None)
        if self.authoritative_source_fingerprint is None and self.linear_inputs is not None:
            object.__setattr__(self, "authoritative_source_fingerprint", None)

    @property
    def canonical_solve_aoi(self) -> str | None:
        return str(self.linear_inputs.solve_source_aoi) if self.linear_inputs is not None else None


__all__ = ["CanonicalRiverSourceContext", "RiverWorkflowContext", "RiverWorkflowSourceBundle"]
