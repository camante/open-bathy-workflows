from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

import geopandas as gpd
import numpy as np
import rasterio

SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND = "authoritative_control_found"
SUPPORT_STATUS_SCAFFOLD_INFERRED = "scaffold_inferred"
SUPPORT_STATUS_DO_NOT_FILL = "do_not_fill"
SUPPORT_STATUS_SEARCH_ERROR = "support_search_error"


@dataclass(frozen=True)
class RiverV2CanonicalSupportDecision:
    system_id: str
    support_status: str
    support_basis: str | None
    canonical_solve_aoi: str | None
    support_evidence_path: str | None
    scaffold_allowed: bool
    reason: str | None = None

    @property
    def final_support_status(self) -> str:
        return self.support_status

    @property
    def authoritative_control_found(self) -> bool:
        return self.support_status == SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND

    @property
    def do_not_fill(self) -> bool:
        return self.support_status == SUPPORT_STATUS_DO_NOT_FILL

    # Legacy compatibility properties for existing callers/reporting.
    @property
    def local_authoritative_control_found(self) -> bool:
        return self.authoritative_control_found

    @property
    def downstream_search_performed(self) -> bool:
        return False

    @property
    def downstream_search_stop_reason(self) -> str:
        return "canonical_domain_support_decision"

    @property
    def downstream_authoritative_control_found(self) -> bool:
        return False

    @property
    def scaffold_inference_allowed(self) -> bool:
        return self.scaffold_allowed

    @property
    def authoritative_control_basis(self) -> str | None:
        return self.support_basis

    @property
    def do_not_fill_reason(self) -> str | None:
        return self.reason if self.do_not_fill else None

    @property
    def support_search_error(self) -> None:
        return None

    @property
    def support_policy(self) -> str:
        return str(self.support_basis or "canonical_support_decision")

    @property
    def downstream_support_span_native_units(self) -> float:
        return 0.0

    @property
    def downstream_support_hit_reach_count(self) -> int:
        return 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "final_support_status": self.final_support_status,
            "authoritative_control_found": self.authoritative_control_found,
            "scaffold_inference_allowed": self.scaffold_inference_allowed,
            "do_not_fill": self.do_not_fill,
            "authoritative_control_basis": self.authoritative_control_basis,
            "do_not_fill_reason": self.do_not_fill_reason,
            "support_policy": self.support_policy,
            "decision_model": "canonical_only_active_path",
        })
        return payload


@dataclass(frozen=True)
class RiverV2SystemSupportDecision:
    system_id: str
    support_policy: str
    local_authoritative_control_found: bool
    downstream_search_performed: bool
    downstream_search_stop_reason: Optional[str]
    downstream_authoritative_control_found: bool
    scaffold_inference_allowed: bool
    final_support_status: str
    authoritative_control_basis: Optional[str] = None
    do_not_fill_reason: Optional[str] = None
    support_search_error: Optional[str] = None
    downstream_support_span_native_units: float = 0.0
    downstream_support_hit_reach_count: int = 0

    @property
    def authoritative_control_found(self) -> bool:
        return bool(self.final_support_status == SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND)

    @property
    def do_not_fill(self) -> bool:
        return bool(self.final_support_status == SUPPORT_STATUS_DO_NOT_FILL)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _path_exists(value: Any) -> bool:
    if value in (None, "", False):
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False


def _resolve_system_id(ctx: Any) -> str:
    for attr in ("canonical_system_id", "system_id", "major_system_id"):
        value = getattr(ctx, attr, None)
        if value not in (None, "", False):
            return str(value)
    report = getattr(ctx, "report", None)
    if isinstance(report, dict):
        river = report.get("river", {})
        if isinstance(river, dict):
            network = river.get("network", {})
            if isinstance(network, dict):
                for key in ("major_system_id", "system_id", "levelpathi", "levelpath_id"):
                    value = network.get(key)
                    if value not in (None, "", False):
                        return str(value)
    network_path = getattr(ctx, "canonical_network_gpkg", None) or getattr(ctx, "network_gpkg", None)
    if network_path not in (None, "", False):
        try:
            return Path(network_path).stem
        except TypeError:
            pass
    return "river_v2_system_unknown"


def _canonical_support_exists(path: Any) -> bool:
    if not _path_exists(path):
        return False
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".gpkg", ".geojson", ".shp"}:
            layers_df = gpd.list_layers(path)
            layer_names = [str(layer.name) for layer in layers_df.itertuples(index=False)] if layers_df is not None else []
            if not layer_names:
                return False
            for layer_name in layer_names:
                gdf = gpd.read_file(path, layer=layer_name)
                if not gdf.empty:
                    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
                    if not gdf.empty:
                        return True
            return False
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype("float32")
            nodata = ds.nodata
            if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                arr[np.isclose(arr, np.float32(nodata))] = np.nan
            return bool(np.count_nonzero(np.isfinite(arr)) > 0)
    except Exception:
        return False


def _canonical_authoritative_control(ctx: Any) -> tuple[bool, Optional[str], Optional[str]]:
    trusted_mode = str(getattr(ctx, "trusted_support_mode", "") or "").strip().lower()
    support_path = getattr(ctx, "canonical_support_coverage_path", None)
    if _canonical_support_exists(support_path):
        return True, "canonical_support_coverage", str(Path(support_path)) if support_path else None
    base_path = getattr(ctx, "canonical_authoritative_base_path", None)
    if trusted_mode == "all_finite_cells_trusted" and _path_exists(base_path):
        return True, "all_finite_canonical_authoritative_base", str(Path(base_path)) if base_path else None
    return False, None, None


def _canonical_scaffold_allowed(ctx: Any) -> tuple[bool, Optional[str]]:
    if not _path_exists(getattr(ctx, "canonical_network_gpkg", None)):
        return False, "missing_canonical_network_gpkg"
    if not _path_exists(getattr(ctx, "canonical_channel_mask_path", None)):
        return False, "missing_canonical_channel_mask"
    support_candidates = (
        getattr(ctx, "canonical_authoritative_sampling_source_raster_path", None),
        getattr(ctx, "canonical_authoritative_base_path", None),
        getattr(ctx, "river_dem_path", None),
        getattr(ctx, "bank_wse_edge_guidance_path", None),
        getattr(ctx, "authoritative_sampling_raster_path", None),
    )
    if not any(_path_exists(v) for v in support_candidates):
        return False, "missing_support_source_raster"
    return True, None


def evaluate_canonical_river_system_support_details(ctx: Any) -> RiverV2CanonicalSupportDecision:
    system_id = _resolve_system_id(ctx)
    solve_aoi = str(getattr(ctx, "canonical_solve_aoi", None) or "").strip() or None
    authoritative_control, support_basis, support_evidence_path = _canonical_authoritative_control(ctx)
    if authoritative_control:
        return RiverV2CanonicalSupportDecision(
            system_id=system_id,
            support_status=SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND,
            support_basis=support_basis,
            canonical_solve_aoi=solve_aoi,
            support_evidence_path=support_evidence_path,
            scaffold_allowed=True,
            reason=None,
        )
    scaffold_allowed, reason = _canonical_scaffold_allowed(ctx)
    if scaffold_allowed:
        return RiverV2CanonicalSupportDecision(
            system_id=system_id,
            support_status=SUPPORT_STATUS_SCAFFOLD_INFERRED,
            support_basis="canonical_scaffold_ready",
            canonical_solve_aoi=solve_aoi,
            support_evidence_path=None,
            scaffold_allowed=True,
            reason=None,
        )
    return RiverV2CanonicalSupportDecision(
        system_id=system_id,
        support_status=SUPPORT_STATUS_DO_NOT_FILL,
        support_basis="canonical_scaffold_unavailable",
        canonical_solve_aoi=solve_aoi,
        support_evidence_path=None,
        scaffold_allowed=False,
        reason=reason,
    )


# Legacy wrapper retained for older tests/call sites. It is intentionally kept
# separate from the active canonical-only production logic above.
def evaluate_river_system_support_details(ctx: Any) -> tuple[RiverV2SystemSupportDecision, Any, Any, Any, Any, Any]:
    from legacy.river.river_v2_downstream_support import RiverV2DownstreamSupportSearchResult, search_downstream_authoritative_support_details

    if getattr(ctx, "canonical_solve_aoi", None) or getattr(ctx, "canonical_network_gpkg", None) or getattr(ctx, "canonical_support_coverage_path", None):
        decision = evaluate_canonical_river_system_support_details(ctx)
        search_result = RiverV2DownstreamSupportSearchResult(
            search_performed=False,
            search_stop_reason="canonical_domain_support_decision",
            support_found=bool(decision.authoritative_control_found),
            searched_reach_count=0,
            downstream_reach_count=0,
            export_reach_count=0,
            support_hit_reach_count=0,
            searched_distance_native_units=0.0,
            support_span_native_units=0.0,
        )
        legacy_decision = RiverV2SystemSupportDecision(
            system_id=decision.system_id,
            support_policy=decision.support_policy,
            local_authoritative_control_found=decision.local_authoritative_control_found,
            downstream_search_performed=False,
            downstream_search_stop_reason=decision.downstream_search_stop_reason,
            downstream_authoritative_control_found=False,
            scaffold_inference_allowed=decision.scaffold_inference_allowed,
            final_support_status=decision.final_support_status,
            authoritative_control_basis=decision.authoritative_control_basis,
            do_not_fill_reason=decision.do_not_fill_reason,
        )
        return legacy_decision, search_result, None, None, None, None

    local_control = False
    control_basis = None
    trusted_mode = str(getattr(ctx, "trusted_support_mode", "") or "").strip().lower()
    if trusted_mode == "all_finite_cells_trusted":
        if _path_exists(getattr(ctx, "authoritative_base_path", None)):
            local_control, control_basis = True, "all_finite_authoritative_base"
        elif _path_exists(getattr(ctx, "trusted_support_artifact_path", None)):
            local_control, control_basis = True, "all_finite_trusted_support_artifact"
    elif trusted_mode == "metadata_proven_only":
        if _path_exists(getattr(ctx, "trusted_support_artifact_path", None)):
            local_control, control_basis = True, "metadata_proven_support_artifact"
        elif _path_exists(getattr(ctx, "authoritative_support_coverage_path", None)):
            local_control, control_basis = True, "metadata_proven_support_coverage"

    system_id = _resolve_system_id(ctx)
    support_policy = str(getattr(ctx, "trusted_support_mode", None) or "unknown_support_policy")
    if local_control:
        decision = RiverV2SystemSupportDecision(
            system_id=system_id,
            support_policy=support_policy,
            local_authoritative_control_found=True,
            downstream_search_performed=False,
            downstream_search_stop_reason="local_authoritative_control_already_sufficient",
            downstream_authoritative_control_found=False,
            scaffold_inference_allowed=True,
            final_support_status=SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND,
            authoritative_control_basis=control_basis,
        )
        search_result = RiverV2DownstreamSupportSearchResult(
            search_performed=False,
            search_stop_reason="local_authoritative_control_already_sufficient",
            support_found=False,
            searched_reach_count=0,
            downstream_reach_count=0,
            export_reach_count=0,
            support_hit_reach_count=0,
            searched_distance_native_units=0.0,
            support_span_native_units=0.0,
        )
        return decision, search_result, None, None, None, None

    search_result, export_reaches, downstream_reaches, support_hits, support_coverage = search_downstream_authoritative_support_details(ctx)
    if search_result.search_error:
        decision = RiverV2SystemSupportDecision(
            system_id=system_id,
            support_policy=support_policy,
            local_authoritative_control_found=False,
            downstream_search_performed=bool(search_result.search_performed),
            downstream_search_stop_reason=search_result.search_stop_reason,
            downstream_authoritative_control_found=False,
            scaffold_inference_allowed=False,
            final_support_status=SUPPORT_STATUS_SEARCH_ERROR,
            support_search_error=search_result.search_error,
            downstream_support_span_native_units=float(search_result.support_span_native_units),
            downstream_support_hit_reach_count=int(search_result.support_hit_reach_count),
        )
        return decision, search_result, export_reaches, downstream_reaches, support_hits, support_coverage
    if search_result.support_found:
        decision = RiverV2SystemSupportDecision(
            system_id=system_id,
            support_policy=support_policy,
            local_authoritative_control_found=False,
            downstream_search_performed=bool(search_result.search_performed),
            downstream_search_stop_reason=search_result.search_stop_reason,
            downstream_authoritative_control_found=True,
            scaffold_inference_allowed=True,
            final_support_status=SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND,
            authoritative_control_basis=search_result.support_basis,
            downstream_support_span_native_units=float(search_result.support_span_native_units),
            downstream_support_hit_reach_count=int(search_result.support_hit_reach_count),
        )
        return decision, search_result, export_reaches, downstream_reaches, support_hits, support_coverage

    scaffold_allowed = False
    do_not_fill_reason = None
    if _path_exists(getattr(ctx, "network_gpkg", None)) and _path_exists(getattr(ctx, "channel_mask_path", None)) and any(_path_exists(v) for v in (getattr(ctx, "river_dem_path", None), getattr(ctx, "bank_wse_edge_guidance_path", None), getattr(ctx, "authoritative_sampling_raster_path", None), getattr(ctx, "authoritative_base_path", None), getattr(ctx, "aligned_authoritative_base_path", None))):
        scaffold_allowed = True
    else:
        if not _path_exists(getattr(ctx, "network_gpkg", None)):
            do_not_fill_reason = "missing_network_gpkg"
        elif not _path_exists(getattr(ctx, "channel_mask_path", None)):
            do_not_fill_reason = "missing_channel_mask"
        else:
            do_not_fill_reason = "missing_support_source_raster"

    if scaffold_allowed:
        decision = RiverV2SystemSupportDecision(
            system_id=system_id,
            support_policy=support_policy,
            local_authoritative_control_found=False,
            downstream_search_performed=bool(search_result.search_performed),
            downstream_search_stop_reason=search_result.search_stop_reason,
            downstream_authoritative_control_found=False,
            scaffold_inference_allowed=True,
            final_support_status=SUPPORT_STATUS_SCAFFOLD_INFERRED,
            downstream_support_span_native_units=float(search_result.support_span_native_units),
            downstream_support_hit_reach_count=int(search_result.support_hit_reach_count),
        )
        return decision, search_result, export_reaches, downstream_reaches, support_hits, support_coverage

    decision = RiverV2SystemSupportDecision(
        system_id=system_id,
        support_policy=support_policy,
        local_authoritative_control_found=False,
        downstream_search_performed=bool(search_result.search_performed),
        downstream_search_stop_reason=search_result.search_stop_reason,
        downstream_authoritative_control_found=False,
        scaffold_inference_allowed=False,
        final_support_status=SUPPORT_STATUS_DO_NOT_FILL,
        do_not_fill_reason=do_not_fill_reason,
        downstream_support_span_native_units=float(search_result.support_span_native_units),
        downstream_support_hit_reach_count=int(search_result.support_hit_reach_count),
    )
    return decision, search_result, export_reaches, downstream_reaches, support_hits, support_coverage


def evaluate_river_system_support(ctx: Any) -> tuple[RiverV2SystemSupportDecision, Any]:
    decision, search, *_ = evaluate_river_system_support_details(ctx)
    return decision, search


def decide_river_system_support_status(ctx: Any) -> RiverV2SystemSupportDecision:
    decision, _ = evaluate_river_system_support(ctx)
    return decision


def write_river_v2_system_support_decision(decision: Any, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "SUPPORT_STATUS_AUTHORITATIVE_CONTROL_FOUND",
    "SUPPORT_STATUS_SCAFFOLD_INFERRED",
    "SUPPORT_STATUS_DO_NOT_FILL",
    "SUPPORT_STATUS_SEARCH_ERROR",
    "RiverV2CanonicalSupportDecision",
    "RiverV2SystemSupportDecision",
    "evaluate_canonical_river_system_support_details",
    "evaluate_river_system_support_details",
    "evaluate_river_system_support",
    "decide_river_system_support_status",
    "write_river_v2_system_support_decision",
]
