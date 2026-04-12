from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from river_v2_paths import RiverV2Paths, build_river_v2_paths
from river_v2_validation import dedupe_input_artifacts


@dataclass
class RiverV2Context:
    cfg: Any
    report: Dict[str, Any]
    out_dir: Path
    network_gpkg: Path
    river_dem_path: Path
    channel_mask_path: Optional[Path] = None
    authoritative_bed_path: Optional[Path] = None
    authoritative_base_path: Optional[Path] = None
    aligned_authoritative_base_path: Optional[Path] = None
    baseline_interpolated_path: Optional[Path] = None
    bank_wse_edge_guidance_path: Optional[Path] = None
    bank_materialization_diagnostics_path: Optional[Path] = None
    authoritative_sampling_raster_path: Optional[Path] = None
    authoritative_support_coverage_path: Optional[Path] = None
    authoritative_dem_mode: str = "mixed_requires_metadata"
    trusted_support_mode: str = "low_support_no_trusted_support"
    trusted_support_artifact_path: Optional[Path] = None
    authoritative_support_policy_warning: Optional[str] = None
    support_policy_source: Optional[str] = None
    vertical_reference: str = "unknown"
    centerline_spacing_m: Optional[float] = None
    centerline_points_gdf: Any | None = None  # test/dev compatibility; production route leaves this unset
    extra_input_artifacts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.network_gpkg = Path(self.network_gpkg)
        self.river_dem_path = Path(self.river_dem_path)
        self.channel_mask_path = Path(self.channel_mask_path) if self.channel_mask_path is not None else None
        self.authoritative_bed_path = Path(self.authoritative_bed_path) if self.authoritative_bed_path is not None else None
        self.authoritative_base_path = Path(self.authoritative_base_path) if self.authoritative_base_path is not None else None
        self.aligned_authoritative_base_path = Path(self.aligned_authoritative_base_path) if self.aligned_authoritative_base_path is not None else None
        self.baseline_interpolated_path = Path(self.baseline_interpolated_path) if self.baseline_interpolated_path is not None else None
        self.bank_wse_edge_guidance_path = Path(self.bank_wse_edge_guidance_path) if self.bank_wse_edge_guidance_path is not None else None
        self.bank_materialization_diagnostics_path = Path(self.bank_materialization_diagnostics_path) if self.bank_materialization_diagnostics_path is not None else None
        self.authoritative_sampling_raster_path = Path(self.authoritative_sampling_raster_path) if self.authoritative_sampling_raster_path is not None else None
        self.authoritative_support_coverage_path = Path(self.authoritative_support_coverage_path) if self.authoritative_support_coverage_path is not None else None
        self.trusted_support_artifact_path = Path(self.trusted_support_artifact_path) if self.trusted_support_artifact_path is not None else None

        # Production runs should resolve authoritative support policy once in bathy_main.py.
        # Direct/test context construction still needs a small compatibility resolver so
        # stage and pipeline tests do not silently become low-support runs.
        mode = str(self.authoritative_dem_mode or "mixed_requires_metadata").strip().lower()
        trusted_mode = str(self.trusted_support_mode or "").strip().lower()
        if not self.support_policy_source:
            if mode == "measured_only" and self.authoritative_base_path is not None:
                self.trusted_support_mode = "all_finite_cells_trusted"
                self.support_policy_source = "direct_context_measured_only_default"
            elif self.trusted_support_artifact_path is not None or self.authoritative_support_coverage_path is not None:
                if self.trusted_support_artifact_path is None and self.authoritative_support_coverage_path is not None:
                    self.trusted_support_artifact_path = self.authoritative_support_coverage_path
                self.trusted_support_mode = "metadata_proven_only"
                self.support_policy_source = "direct_context_metadata_support_default"
            elif self.authoritative_base_path is not None:
                self.trusted_support_mode = "all_finite_cells_trusted"
                self.support_policy_source = "direct_context_compatibility_default"
            else:
                self.trusted_support_mode = trusted_mode or "low_support_no_trusted_support"
                self.support_policy_source = "direct_context_low_support_default"
        elif self.trusted_support_mode == "metadata_proven_only" and self.trusted_support_artifact_path is None and self.authoritative_support_coverage_path is not None:
            self.trusted_support_artifact_path = self.authoritative_support_coverage_path

    @property
    def paths(self) -> RiverV2Paths:
        return build_river_v2_paths(self.out_dir)

    def stage_input_artifacts(self, *extra: Any) -> list[str]:
        return dedupe_input_artifacts([
            self.network_gpkg,
            self.river_dem_path,
            self.channel_mask_path,
            self.authoritative_bed_path,
            self.authoritative_base_path,
            self.aligned_authoritative_base_path,
            self.baseline_interpolated_path,
            self.bank_wse_edge_guidance_path,
            self.bank_materialization_diagnostics_path,
            self.authoritative_sampling_raster_path,
            self.authoritative_support_coverage_path,
            self.trusted_support_artifact_path,
            *self.extra_input_artifacts,
            *extra,
        ])

    def direct_stage_input_artifacts(self, *artifacts: Any) -> list[str]:
        return dedupe_input_artifacts([*artifacts])

    def centerline_stage_input_artifacts(self, *extra: Any) -> list[str]:
        return dedupe_input_artifacts([
            self.network_gpkg,
            self.river_dem_path,
            self.channel_mask_path,
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
