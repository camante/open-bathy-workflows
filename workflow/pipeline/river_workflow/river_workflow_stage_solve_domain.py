from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import geopandas as gpd

from core.json_io import write_json
from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_validation import (
    validate_linear_canonical_network_gpkg,
    validate_solve_aoi_json,
)
from pipeline.river_workflow.river_workflow_canonical_domain import (
    build_canonical_river_solve_domain,
    build_canonical_system_identity_payload,
    fingerprint_canonical_domain_inputs,
    resolve_canonical_system_id,
)
from pipeline.river_workflow.river_workflow_receipts import write_canonical_identity_receipt


_FLOW_LAYERS = ("mainstem_solve_network", "major_system_network", "major_system_network_clip", "rivers_clip")
_POLYGON_LAYERS = ("nhdarea_clip",)


def _normalize_aoi_string(aoi: str, export_aoi: str) -> str:
    west, east, south, north = [float(part) for part in str(aoi).split("/")]
    ew, ee, es, en = [float(part) for part in str(export_aoi).split("/")]
    if east <= west:
        pad = max(1e-6, (ee - ew) / 1000.0)
        west -= pad
        east += pad
    if north <= south:
        pad = max(1e-6, (en - es) / 1000.0)
        south -= pad
        north += pad
    return f"{west}/{east}/{south}/{north}"


@dataclass(frozen=True)
class SolveDomainStageResult:
    canonical_network_path: Path
    canonical_solve_aoi_path: Path
    canonical_solve_aoi: str
    selected_reach_count: int
    canonical_system_id: str | None
    stop_reason: str
    receipt_path: Path


def _read_layer_if_present(gpkg_path: Path, layer_name: str) -> gpd.GeoDataFrame | None:
    try:
        layers = gpd.list_layers(gpkg_path)
        names = set(layers["name"].tolist())
    except (OSError, RuntimeError, ValueError):
        names = set()
    if layer_name not in names:
        return None
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return None
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()


def _extract_linear_layers(source_gpkg: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    flows = None
    for layer_name in _FLOW_LAYERS:
        flows = _read_layer_if_present(source_gpkg, layer_name)
        if flows is not None and not flows.empty:
            break
    if flows is None or flows.empty:
        raise RuntimeError("river_workflow_missing_canonical_flow_layer")
    polygons = None
    for layer_name in _POLYGON_LAYERS:
        polygons = _read_layer_if_present(source_gpkg, layer_name)
        if polygons is not None and not polygons.empty:
            break
    return flows, polygons


def _write_linear_canonical_network(*, flows: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame | None, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    flows.to_file(dst_path, layer="linear_flows", driver="GPKG")
    if polygons is not None and not polygons.empty:
        polygons.to_file(dst_path, layer="linear_polygons", driver="GPKG")


def _prepared_solve_bundle(ctx: RiverWorkflowContext) -> dict[str, Any] | None:
    bundle = ctx.linear_inputs
    if bundle is None:
        return None
    source_path = getattr(bundle, 'canonical_solve_network_source_gpkg', None)
    solve_aoi = getattr(bundle, 'canonical_solve_aoi', None)
    if source_path is None or solve_aoi in (None, '', False):
        return None
    return {
        'canonical_network_gpkg': Path(source_path),
        'canonical_receipt_json': Path(bundle.canonical_solve_receipt_source_json) if getattr(bundle, 'canonical_solve_receipt_source_json', None) is not None else None,
        'canonical_solve_aoi': str(solve_aoi),
        'canonical_system_id': str(bundle.canonical_system_id) if getattr(bundle, 'canonical_system_id', None) not in (None, '') else None,
        'solve_stop_reason': str(getattr(bundle, 'canonical_solve_stop_reason', None) or 'prepared_bundle_registered'),
        'selected_reach_count': int(getattr(bundle, 'canonical_selected_reach_count', 0) or 0),
        'selected_reach_length_m': float(getattr(bundle, 'canonical_selected_reach_length_m', 0.0) or 0.0),
    }


def _prepared_summary_payload(prepared: dict[str, Any]) -> dict[str, Any]:
    receipt_path = prepared.get('canonical_receipt_json')
    if receipt_path is not None and Path(receipt_path).exists():
        return json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    return {
        'canonical_domain_summary': {},
        'canonical_solve_aoi': prepared['canonical_solve_aoi'],
        'canonical_system_id': prepared.get('canonical_system_id'),
        'solve_stop_reason': prepared.get('solve_stop_reason'),
        'selected_reach_count': int(prepared.get('selected_reach_count', 0) or 0),
        'selected_reach_length_m': float(prepared.get('selected_reach_length_m', 0.0) or 0.0),
    }



def run_solve_domain_stage(ctx: RiverWorkflowContext) -> SolveDomainStageResult:
    prepared = _prepared_solve_bundle(ctx)
    if prepared is not None:
        source_network = Path(prepared['canonical_network_gpkg'])
        payload = _prepared_summary_payload(prepared)
        canonical_solve_aoi_raw = str(prepared['canonical_solve_aoi'])
        canonical_system_id = prepared.get('canonical_system_id')
        solve_stop_reason = str(prepared['solve_stop_reason'])
        selected_reach_count = int(prepared['selected_reach_count'])
        selected_reach_length_m = float(prepared['selected_reach_length_m'])
        receipt_path = Path(prepared['canonical_receipt_json']) if prepared.get('canonical_receipt_json') is not None else ctx.paths.canonical_solve_domain_json
    else:
        result = build_canonical_river_solve_domain(ctx)
        payload = json.loads(Path(result.canonical_receipt_json).read_text(encoding="utf-8"))
        source_network = Path(result.canonical_network_gpkg)
        canonical_solve_aoi_raw = str(result.canonical_solve_aoi)
        canonical_system_id = result.canonical_system_id
        solve_stop_reason = str(result.solve_stop_reason)
        selected_reach_count = int(result.selected_reach_count)
        selected_reach_length_m = float(result.selected_reach_length_m)
        receipt_path = Path(result.canonical_receipt_json)
    flows, polygons = _extract_linear_layers(source_network)
    _write_linear_canonical_network(flows=flows, polygons=polygons, dst_path=ctx.paths.canonical_solve_network)
    validate_linear_canonical_network_gpkg(ctx.paths.canonical_solve_network)
    canonical_solve_aoi = _normalize_aoi_string(canonical_solve_aoi_raw, ctx.export_aoi)
    canonical_summary = payload.get("canonical_domain_summary", {}) if isinstance(payload, dict) else {}
    identity_fingerprints = canonical_summary.get("identity_fingerprints")
    source_system_id = canonical_summary.get("source_system_id")
    if not isinstance(identity_fingerprints, dict):
        identity_fingerprints = build_canonical_system_identity_payload(
            ctx,
            canonical_solve_aoi=canonical_solve_aoi,
            shared_system_id=source_system_id,
            canonical_trace_distance_km=getattr(ctx, "canonical_trace_distance_km", None) or getattr(ctx, "canonical_max_trace_km", None),
        )
    if canonical_system_id in (None, ""):
        canonical_system_id = resolve_canonical_system_id(
            ctx,
            canonical_solve_aoi=canonical_solve_aoi,
            shared_system_id=source_system_id,
            canonical_trace_distance_km=getattr(ctx, "canonical_trace_distance_km", None) or getattr(ctx, "canonical_max_trace_km", None),
        )
    identity_receipt_path = write_canonical_identity_receipt(
        ctx.paths.canonical_system_identity_receipt,
        canonical_system_id=str(canonical_system_id),
        user_aoi=ctx.export_aoi,
        canonical_domain_bounds=canonical_solve_aoi,
        fingerprints=identity_fingerprints,
        source_system_id=source_system_id,
    )
    object.__setattr__(ctx, "canonical_system_id", str(canonical_system_id))
    object.__setattr__(ctx, "canonical_domain_bounds", str(canonical_solve_aoi))
    object.__setattr__(ctx, "canonical_identity_receipt_path", identity_receipt_path)
    object.__setattr__(ctx, "canonical_trace_distance_km", identity_fingerprints.get("canonical_trace_distance_km"))
    object.__setattr__(ctx, "river_network_fingerprint", identity_fingerprints.get("river_network_fingerprint"))
    object.__setattr__(ctx, "authoritative_source_fingerprint", identity_fingerprints.get("authoritative_source_fingerprint"))

    solve_payload = {
        "export_aoi": ctx.export_aoi,
        "canonical_solve_aoi": canonical_solve_aoi,
        "canonical_system_id": canonical_system_id,
        "solve_stop_reason": solve_stop_reason,
        "selected_reach_count": int(selected_reach_count),
        "selected_reach_length_m": float(selected_reach_length_m),
        "canonical_network_gpkg": str(ctx.paths.canonical_solve_network),
        "canonical_receipt_json": str(receipt_path),
        "canonical_domain_summary": {**canonical_summary, "identity_receipt_path": str(identity_receipt_path)},
        "canonical_identity_receipt_path": str(identity_receipt_path),
    }
    write_json(ctx.paths.canonical_solve_aoi, solve_payload)
    validate_solve_aoi_json(ctx.paths.canonical_solve_aoi, export_aoi=ctx.export_aoi)
    return SolveDomainStageResult(
        canonical_network_path=ctx.paths.canonical_solve_network,
        canonical_solve_aoi_path=ctx.paths.canonical_solve_aoi,
        canonical_solve_aoi=canonical_solve_aoi,
        selected_reach_count=int(selected_reach_count),
        canonical_system_id=canonical_system_id,
        stop_reason=solve_stop_reason,
        receipt_path=receipt_path,
    )


__all__ = ["SolveDomainStageResult", "run_solve_domain_stage"]
