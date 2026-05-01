from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from legacy.river.river_v2_paths import RiverV2Paths, build_river_v2_paths
from legacy.river.river_v2_validation import dedupe_input_artifacts

# RiverV2Context is orchestration/reporting state only. Production runtime stage
# inputs must come from the execution contract, not from context fallbacks.


@dataclass
class RiverV2Context:
    cfg: Any
    report: Dict[str, Any]
    out_dir: Path
    network_gpkg: Path
    export_aoi: Optional[str] = None
    export_network_gpkg: Optional[Path] = None
    canonical_system_id: Optional[str] = None
    canonical_solve_aoi: Optional[str] = None
    canonical_stop_reason: Optional[str] = None
    canonical_network_gpkg: Optional[Path] = None
    river_dem_path: Path = None
    channel_mask_path: Optional[Path] = None
    export_channel_mask_path: Optional[Path] = None
    authoritative_bed_path: Optional[Path] = None
    authoritative_base_path: Optional[Path] = None
    aligned_authoritative_base_path: Optional[Path] = None
    baseline_interpolated_path: Optional[Path] = None
    bank_wse_edge_guidance_path: Optional[Path] = None
    bank_materialization_diagnostics_path: Optional[Path] = None
    authoritative_sampling_source_raster_path: Optional[Path] = None
    authoritative_sampling_raster_path: Optional[Path] = None
    authoritative_support_coverage_path: Optional[Path] = None
    solve_aoi_authoritative_base_path: Optional[Path] = None
    solve_aoi_authoritative_support_coverage_path: Optional[Path] = None
    canonical_channel_mask_path: Optional[Path] = None
    canonical_support_coverage_path: Optional[Path] = None
    canonical_authoritative_sampling_source_raster_path: Optional[Path] = None
    canonical_authoritative_base_path: Optional[Path] = None
    authoritative_dem_mode: str = "mixed_requires_metadata"
    trusted_support_mode: str = "low_support_no_trusted_support"
    trusted_support_artifact_path: Optional[Path] = None
    authoritative_support_policy_warning: Optional[str] = None
    support_policy_source: Optional[str] = None
    vertical_reference: str = "unknown"
    centerline_spacing_m: Optional[float] = None
    centerline_points_gdf: Any | None = None  # test/dev compatibility; production route leaves this unset
    extra_input_artifacts: list[str] = field(default_factory=list)
    system_support_status: str = "unknown"
    authoritative_control_found: bool = False
    local_authoritative_control_found: bool = False
    downstream_authoritative_control_found: bool = False
    scaffold_inference_allowed: bool = False
    do_not_fill: bool = False
    support_decision_json_path: Optional[Path] = None
    downstream_support_search_json_path: Optional[Path] = None
    canonical_solve_domain_receipt: Optional[Path] = None
    canonical_inputs_receipt_path: Optional[Path] = None
    canonical_support_status_basis: Optional[str] = None
    river_canonical_max_trace_km: float = 200.0
    resolved_inputs_manifest_path: Optional[Path] = None
    preflight_receipt_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.network_gpkg = Path(self.network_gpkg)
        self.export_aoi = str(self.export_aoi).strip() if self.export_aoi not in (None, '', False) else None
        self.export_network_gpkg = Path(self.export_network_gpkg) if self.export_network_gpkg is not None else self.network_gpkg
        self.canonical_network_gpkg = Path(self.canonical_network_gpkg) if self.canonical_network_gpkg is not None else None
        self.river_dem_path = Path(self.river_dem_path)
        self.channel_mask_path = Path(self.channel_mask_path) if self.channel_mask_path is not None else None
        self.export_channel_mask_path = Path(self.export_channel_mask_path) if self.export_channel_mask_path is not None else None
        self.authoritative_bed_path = Path(self.authoritative_bed_path) if self.authoritative_bed_path is not None else None
        self.authoritative_base_path = Path(self.authoritative_base_path) if self.authoritative_base_path is not None else None
        self.aligned_authoritative_base_path = Path(self.aligned_authoritative_base_path) if self.aligned_authoritative_base_path is not None else None
        self.baseline_interpolated_path = Path(self.baseline_interpolated_path) if self.baseline_interpolated_path is not None else None
        self.bank_wse_edge_guidance_path = Path(self.bank_wse_edge_guidance_path) if self.bank_wse_edge_guidance_path is not None else None
        self.bank_materialization_diagnostics_path = Path(self.bank_materialization_diagnostics_path) if self.bank_materialization_diagnostics_path is not None else None
        self.authoritative_sampling_source_raster_path = Path(self.authoritative_sampling_source_raster_path) if self.authoritative_sampling_source_raster_path is not None else None
        self.authoritative_sampling_raster_path = Path(self.authoritative_sampling_raster_path) if self.authoritative_sampling_raster_path is not None else None
        self.authoritative_support_coverage_path = Path(self.authoritative_support_coverage_path) if self.authoritative_support_coverage_path is not None else None
        self.solve_aoi_authoritative_base_path = Path(self.solve_aoi_authoritative_base_path) if self.solve_aoi_authoritative_base_path is not None else None
        self.solve_aoi_authoritative_support_coverage_path = Path(self.solve_aoi_authoritative_support_coverage_path) if self.solve_aoi_authoritative_support_coverage_path is not None else None
        self.canonical_channel_mask_path = Path(self.canonical_channel_mask_path) if self.canonical_channel_mask_path is not None else None
        self.canonical_support_coverage_path = Path(self.canonical_support_coverage_path) if self.canonical_support_coverage_path is not None else None
        self.canonical_authoritative_sampling_source_raster_path = Path(self.canonical_authoritative_sampling_source_raster_path) if self.canonical_authoritative_sampling_source_raster_path is not None else None
        self.canonical_authoritative_base_path = Path(self.canonical_authoritative_base_path) if self.canonical_authoritative_base_path is not None else None
        self.trusted_support_artifact_path = Path(self.trusted_support_artifact_path) if self.trusted_support_artifact_path is not None else None
        self.support_decision_json_path = Path(self.support_decision_json_path) if self.support_decision_json_path is not None else None
        self.downstream_support_search_json_path = Path(self.downstream_support_search_json_path) if self.downstream_support_search_json_path is not None else None
        self.canonical_solve_domain_receipt = Path(self.canonical_solve_domain_receipt) if self.canonical_solve_domain_receipt is not None else None
        self.canonical_inputs_receipt_path = Path(self.canonical_inputs_receipt_path) if self.canonical_inputs_receipt_path is not None else None
        self.resolved_inputs_manifest_path = Path(self.resolved_inputs_manifest_path) if self.resolved_inputs_manifest_path is not None else None
        self.preflight_receipt_path = Path(self.preflight_receipt_path) if self.preflight_receipt_path is not None else None

    @property
    def paths(self) -> RiverV2Paths:
        return build_river_v2_paths(self.out_dir)

    def remember_preflight_artifacts(self, *, resolved_inputs_manifest_path: Path, preflight_receipt_path: Path) -> None:
        self.resolved_inputs_manifest_path = Path(resolved_inputs_manifest_path)
        self.preflight_receipt_path = Path(preflight_receipt_path)

    def stage_input_artifacts(self, *extra: Any) -> list[str]:
        return dedupe_input_artifacts([
            self.export_network_gpkg,
            self.river_dem_path,
            self.export_channel_mask_path,
            self.authoritative_bed_path,
            self.authoritative_base_path,
            self.aligned_authoritative_base_path,
            self.baseline_interpolated_path,
            self.bank_wse_edge_guidance_path,
            self.bank_materialization_diagnostics_path,
            self.authoritative_sampling_raster_path,
            self.solve_aoi_authoritative_base_path,
            self.solve_aoi_authoritative_support_coverage_path,
            self.trusted_support_artifact_path,
            self.support_decision_json_path,
            self.canonical_solve_domain_receipt,
            self.canonical_inputs_receipt_path,
            self.canonical_network_gpkg,
            self.canonical_channel_mask_path,
            self.canonical_support_coverage_path,
            self.canonical_authoritative_sampling_source_raster_path,
            self.canonical_authoritative_base_path,
            self.resolved_inputs_manifest_path,
            self.preflight_receipt_path,
            *self.extra_input_artifacts,
            *extra,
        ])

    def direct_stage_input_artifacts(self, *artifacts: Any) -> list[str]:
        return dedupe_input_artifacts([*artifacts])

    def centerline_stage_input_artifacts(self, *extra: Any) -> list[str]:
        return dedupe_input_artifacts([
            self.canonical_network_gpkg,
            self.export_network_gpkg,
            self.river_dem_path,
            self.canonical_channel_mask_path,
            self.export_channel_mask_path,
            self.authoritative_sampling_raster_path,
            *extra,
        ])


def _existing_path_str(value: Any) -> str | None:
    if value in (None, '', False):
        return None
    try:
        path = Path(value)
    except TypeError:
        return None
    return str(path) if path.exists() else None


def resolve_river_v2_bank_guidance_inputs(report: Dict[str, Any], river_v2_out_dir: str | Path) -> Dict[str, str]:
    """Resolve the production WSE support inputs.

    The active River v2 path only needs one canonical support raster. We still
    carry the materialization diagnostics path when present because it is useful
    for debugging, but the profile-summary side product is no longer part of the
    normal production contract.
    """
    river_block = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    river_outputs = river_block.get("outputs", {}) if isinstance(river_block.get("outputs", {}), dict) else {}
    simple_outputs = river_block.get("simple_river_stage_outputs", {}) if isinstance(river_block.get("simple_river_stage_outputs", {}), dict) else {}
    support_dir = build_river_v2_paths(Path(river_v2_out_dir)).support_dir

    edge_candidates = [
        simple_outputs.get("wse_proxy"),
        river_outputs.get("bank_wse_edge_guidance_path"),
        river_outputs.get("bank_elevation_path"),
        river_outputs.get("bank_elevation_xs"),
        support_dir / "river_bank_wse_edge_guidance.tif",
    ]
    edge = next((v for v in (_existing_path_str(c) for c in edge_candidates) if v), None)
    materialization_candidates = [
        river_outputs.get("bank_materialization_diagnostics_path"),
        support_dir / "river_bank_wse_materialization_diagnostics.json",
    ]
    materialization = next((v for v in (_existing_path_str(c) for c in materialization_candidates) if v), None)
    out: Dict[str, str] = {}
    if edge:
        out["bank_wse_edge_guidance_path"] = edge
    if materialization:
        out["bank_materialization_diagnostics_path"] = materialization
    return out
