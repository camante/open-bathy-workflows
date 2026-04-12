from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from memory_diag import memory_checkpoint

from river_longitudinal_profile import build_and_write_longitudinal_profile
from river_reach_attributes import build_and_write_reach_attributes
from river_longitudinal_profile_contract import write_river_longitudinal_profile_contract
from river_channel_frame import build_channel_frame_products
from river_channel_scaffold import build_channel_scaffold_products
from river_channel_surface import build_channel_surface_products
from river_component_contract import resolve_centerline_component_expectation
from river_structured_scaffold import validate_centerline_station_component_contract
from river_workflow_diagnostics import build_river_runtime_diagnostics
from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    STAGE_AUTHORITATIVE_BASE,
    STAGE_RIVER_GUIDANCE_DOMAIN,
    STAGE_RIVER_CENTERLINE,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
    STAGE_CONDITIONED_FINAL_INTERNAL,
    STAGE_FINAL_DEM,
    simple_river_stage_status_placeholder,
)

log = logging.getLogger(__name__)


_RIVER_GUIDANCE_BASE_ALLOWED_STRUCTURAL_ARTIFACTS = [
    "admissibility", "authoritative_support", "authoritative_support_depth",
    "bank_confluence_damping", "bank_continuity_weight", "bank_distance",
    "bank_edge_mask", "bank_elevation_xs", "bank_graph_confidence",
    "bank_influence", "bank_pair_weight", "bank_points", "xs_bank_qc_points", "xs_bank_qc_summary",
    "bank_longitudinal_fit_points", "bank_longitudinal_fit_summary", "left_bank_fit_elevation", "right_bank_fit_elevation", "bank_pair_fit_elevation",
    "authoritative_bed_anchor_curve", "authoritative_bed_anchor_curve_summary",
    "bank_estuary_side_decay", "centerline_elevation", "centerline_influence",
    "centerline_points", "centerline_stationing", "corridor_mask",
    "guide_points", "guidance_weight", "regime_class", "retained_network",
    "scaffold_domains", "scaffold_manifest", "soft_guidance_domain",
    "trusted_interior", "longitudinal_profile_contract", "longitudinal_profile",
    "longitudinal_profile_points", "longitudinal_profile_summary",
    "longitudinal_profile_elevation", "longitudinal_profile_uncertainty",
    "longitudinal_profile_influence", "longitudinal_profile_local_authoritative_reconciliation",
    "longitudinal_profile_local_authoritative_reconciliation_influence", "active_core_support_elevation",
    "active_core_support_uncertainty", "active_core_support_influence", "hydraulic_backbone",
    "hydraulic_backbone_nodes", "hydraulic_backbone_edges",
    "channel_frame_points", "channel_frame_contract", "station_targets", "station_targets_summary",
    "anchor_table", "anchor_summary", "authoritative_centerline_anchors", "station_targets", "station_targets_summary", "channel_scaffold_nodes",
    "channel_scaffold_contract", "channel_surface",
    "channel_surface_confidence", "channel_surface_source_class",
    "channel_surface_support_count", "channel_surface_contract",
    "xs_participation_contract",
]

_RIVER_GUIDANCE_BASE_OPTIONAL_STRUCTURAL_ARTIFACTS = {
    "scaffold_manifest", "hydraulic_backbone_nodes", "hydraulic_backbone_edges",
    "longitudinal_profile_local_authoritative_reconciliation", "longitudinal_profile_local_authoritative_reconciliation_influence",
    "bank_longitudinal_fit_points", "bank_longitudinal_fit_summary", "left_bank_fit_elevation", "right_bank_fit_elevation", "bank_pair_fit_elevation",
    "authoritative_bed_anchor_curve", "authoritative_bed_anchor_curve_summary",
    "authoritative_centerline_anchors", "station_targets", "station_targets_summary", "anchor_table", "anchor_summary", "channel_scaffold_nodes",
    "channel_scaffold_contract", "channel_surface",
    "channel_surface_confidence", "channel_surface_source_class",
    "channel_surface_support_count", "channel_surface_contract",
}

_RIVER_GUIDANCE_XS_ARTIFACTS = {
    "xs_support_elevation", "xs_support_points", "xs_support_weight",
    "authoritative_xs_anchors",
}

_RIVER_GUIDANCE_FORBIDDEN_STRUCTURAL_INPUTS = [
    "legacy_fused_candidate_raster",
    "dense_river_depth_raster_as_peer_surface",
    "dense_sdb_depth_raster_as_peer_surface",
    "weighted_overlap_blended_bathymetry_as_structural_input",
]


_RIVER_GUIDANCE_ARTIFACT_ROLES = {
    "depth_terrain": "diagnostic_only",
    "bottom_elevation": "diagnostic_only",
    "guidance_weight": "soft_guidance_weight",
    "trusted_interior": "trusted_export_region",
    "soft_guidance_domain": "soft_guidance_domain",
    "admissibility": "admissible_guidance_domain",
    "regime_class": "shared_regime_contract",
    "guide_points": "structured_scaffold_points",
    "authoritative_support": "authoritative_anchor_support",
    "authoritative_support_depth": "authoritative_anchor_depth",
    "corridor_mask": "river_corridor",
    "bank_edge_mask": "corridor_bank_edge",
    "bank_distance": "corridor_bank_distance",
    "bank_influence": "corridor_bank_influence",
    "bank_elevation_xs": "xs_longitudinal_bank_elevation",
    "bank_pair_weight": "xs_bank_pair_strength",
    "bank_continuity_weight": "xs_bank_longitudinal_continuity",
    "bank_graph_confidence": "graph_informed_bank_confidence",
    "bank_confluence_damping": "confluence_damping",
    "bank_estuary_side_decay": "estuary_side_decay",
    "bank_points": "dense_polygon_bank_points",
    "bank_longitudinal_fit_points": "component_and_side_specific_monotone_bank_fit_points",
    "bank_longitudinal_fit_summary": "component_and_side_specific_monotone_bank_fit_summary",
    "left_bank_fit_elevation": "left_bank_longitudinal_fit_elevation",
    "right_bank_fit_elevation": "right_bank_longitudinal_fit_elevation",
    "bank_pair_fit_elevation": "bank_pair_longitudinal_fit_elevation",
    "authoritative_bed_anchor_curve": "stationized_authoritative_bed_anchor_curve",
    "authoritative_bed_anchor_curve_summary": "stationized_authoritative_bed_anchor_curve_summary",
    "xs_bank_qc_points": "xs_bank_qc_points",
    "xs_bank_qc_summary": "xs_bank_qc_summary",
    "centerline_points": "dense_centerline_points",
    "xs_support_points": "selected_xs_support_points",
    "centerline_elevation": "centerline_longitudinal_tendency",
    "centerline_influence": "centerline_longitudinal_influence",
    "centerline_stationing": "centerline_stationing_coordinate",
    "xs_support_elevation": "cross_stream_support_elevation",
    "xs_support_weight": "cross_stream_support_weight",
    "retained_network": "retained_mainstem_key_tributary_network",
    "scaffold_domains": "scaffold_domain_metadata",
    "scaffold_manifest": "scaffold_product_manifest",
    "longitudinal_profile_contract": "scaffold_vertical_backbone_contract",
    "longitudinal_profile": "station_indexed_longitudinal_profile_table",
    "longitudinal_profile_points": "station_indexed_longitudinal_profile_points",
    "longitudinal_profile_summary": "station_indexed_longitudinal_profile_summary",
    "authoritative_reconciliation_points": "authoritative_reconciliation_observation_table",
    "authoritative_reconciliation_field": "authoritative_reconciliation_field_table",
    "longitudinal_profile_elevation": "rasterized_longitudinal_profile_elevation",
    "longitudinal_profile_uncertainty": "rasterized_longitudinal_profile_uncertainty",
    "longitudinal_profile_influence": "rasterized_longitudinal_profile_influence",
    "longitudinal_profile_local_authoritative_reconciliation": "rasterized_longitudinal_profile_local_authoritative_reconciliation_delta",
    "longitudinal_profile_local_authoritative_reconciliation_influence": "rasterized_longitudinal_profile_local_authoritative_reconciliation_influence",
    "active_core_support_elevation": "explicit_fluvial_active_channel_core_support",
    "active_core_support_uncertainty": "explicit_fluvial_active_channel_core_support_uncertainty",
    "active_core_support_influence": "explicit_fluvial_active_channel_core_support_influence",
    "hydraulic_backbone": "network_aware_1d_hydraulic_backbone_table",
    "hydraulic_backbone_nodes": "network_aware_1d_hydraulic_backbone_nodes",
    "hydraulic_backbone_edges": "network_aware_1d_hydraulic_backbone_edges",
    "reach_attributes": "scientific_reach_attribute_table",
    "reach_components": "component_level_reach_attribute_table",
    "reach_attributes_summary": "scientific_reach_attribute_summary",
    "channel_frame_points": "authoritative_first_channel_frame_stations",
    "channel_frame_contract": "authoritative_first_channel_frame_contract",
    "station_targets": "canonical_station_target_table",
    "station_targets_summary": "canonical_station_target_summary",
    "anchor_table": "canonical_anchor_policy_table",
    "anchor_summary": "canonical_anchor_policy_summary",
    "authoritative_centerline_anchors": "authoritative_in_channel_centerline_anchors",
    "authoritative_xs_anchors": "authoritative_in_channel_xs_anchors",
    "channel_scaffold_nodes": "regularized_channel_scaffold_nodes",
    "channel_scaffold_contract": "regularized_channel_scaffold_contract",
    "channel_surface": "channel_fitted_river_surface",
    "channel_surface_confidence": "channel_fitted_river_surface_confidence",
    "channel_surface_source_class": "channel_fitted_river_surface_source_class",
    "channel_surface_support_count": "channel_fitted_river_surface_support_count",
    "channel_surface_contract": "channel_fitted_river_surface_contract",
    "longitudinal_tendency_profile": "scientific_longitudinal_tendency_profile",
    "longitudinal_tendency_summary": "scientific_longitudinal_tendency_summary",
    "xs_realism_profile": "scientific_xs_realism_profile",
    "xs_realism_summary": "scientific_xs_realism_summary",
    "prediction_confidence_profile": "scientific_prediction_confidence_profile",
    "prediction_confidence_summary": "scientific_prediction_confidence_summary",
    "channel_surface_prediction_support_confidence": "scientific_prediction_support_confidence",
    "channel_surface_measured_anchor_fraction": "scientific_measured_anchor_fraction",
    "channel_surface_structure_only_fraction": "scientific_structure_only_fraction",
    "channel_surface_low_support_caution": "scientific_low_support_caution",
    "channel_surface_prediction_admissibility": "scientific_prediction_admissibility",
    "xs_participation_contract": "xs_participation_contract",
}

_RIVER_GUIDANCE_STATIC_NOTES = {
    "depth_terrain": "Dense river depth surface is diagnostic and should not be treated as peer authoritative terrain.",
    "bottom_elevation": "Dense bed-elevation raster is an internal helper/diagnostic product.",
    "guide_points": "Structured scaffold union restricted to retained WAFFLES/NHD river polygons: dense bank boundary points, longitudinal centerline control points, and selected cross-stream support nodes.",
    "regime_class": "Shared regime contract classification packaged with the river guidance outputs.",
    "bank_influence": "Bank-boundary guidance derived from the WAFFLES/NHD corridor edge. This is a soft boundary tendency, not hard truth.",
    "bank_elevation_xs": "Longitudinal bank-elevation tendency derived from explicit left/right XS bank picks where available.",
    "bank_pair_weight": "Strength of paired left/right bank support for each corridor pixel.",
    "bank_continuity_weight": "Longitudinal continuity confidence for the persistent left/right bank network built from XS picks.",
    "bank_graph_confidence": "Graph-informed confidence from nearby persistent bank-network coherence.",
    "bank_confluence_damping": "Confidence damping near likely confluences where multiple bank-network components compete.",
    "bank_estuary_side_decay": "Side-confidence decay near estuary transition where left/right bank identity becomes less stable.",
    "bank_points": "Dense bank-boundary control points sampled along retained WAFFLES/NHD polygon banks from the authoritative baseline DEM.",
    "bank_longitudinal_fit_points": "Component- and side-specific bank points carrying raw, smoothed, and monotone downstream-fitted bank elevations.",
    "bank_longitudinal_fit_summary": "Summary receipt for the dedicated longitudinal bank-fitting stage, including raw-versus-fit ranges and monotonicity diagnostics by component and side.",
    "left_bank_fit_elevation": "Rasterized left-bank longitudinal fit tendency built from side-specific bank points and monotone downstream projection.",
    "right_bank_fit_elevation": "Rasterized right-bank longitudinal fit tendency built from side-specific bank points and monotone downstream projection.",
    "bank_pair_fit_elevation": "Rasterized bank-pair longitudinal fit tendency used as the active bank reference in unsupported fluvial reaches.",
    "xs_bank_qc_points": "Side-tagged XS bank points carrying raw bank candidates, multi-signal QC flags, QC actions, and longitudinal references used to suppress terrace, roadfill, and high-edge bank contamination before bank fitting.",
    "xs_bank_qc_summary": "Summary receipt for XS bank contamination filtering, including suspect counts plus clamp, envelope-replace, downgrade, and reject action totals.",
    "centerline_points": "Dense longitudinal control points sampled along retained mainstem/key-tributary flowlines inside the retained river polygon.",
    "xs_support_points": "Selected cross-stream support nodes inside the retained river polygon; where available they are rebuilt from XS bathymetry re-verticalized onto the solved longitudinal backbone so lateral XS shape is preserved while absolute bed placement follows the longitudinal profile.",
    "centerline_elevation": "Nearest centerline-derived longitudinal tendency raster sampled from the authoritative baseline DEM.",
    "centerline_influence": "Distance-tapered centerline influence used as longitudinal guidance inside the retained river polygon.",
    "xs_support_elevation": "Nearest selected-XS-node support elevation raster built from retained cross-stream guidance, preferring XS bathymetry re-verticalized onto the solved longitudinal backbone when available.",
    "xs_support_weight": "Distance-tapered XS support influence used as cross-stream guidance inside the retained river polygon.",
    "retained_network": "Retained river network package containing main stems, important tributaries, and retained WAFFLES/NHD polygons used for scaffold generation.",
    "final_route_contract": "Only the listed structured river guidance artifacts may structurally enter the final DEM route; dense river depth/elevation rasters remain diagnostic-only.",
    "xs_support_contract": "XS support artifacts are declared as structural only when this run actually produced usable sampled or re-verticalized XS support; otherwise the route contract remains valid without them.",
    "longitudinal_profile_contract": "Initial Phase C contract for a scaffold-owned longitudinal vertical backbone built from stationing, centerline tendency, and optional XS/anchor support.",
    "longitudinal_profile": "Station-indexed scaffold-owned longitudinal bed-profile object derived from centerline stationing plus sampled centerline, XS-support, and bank-elevation tendencies.",
    "longitudinal_profile_points": "Centerline points carrying interpolated longitudinal-profile elevation and uncertainty values for raster derivative generation.",
    "longitudinal_profile_summary": "Summary receipt for the scaffold-owned longitudinal profile object, including station counts and support evidence presence.",
    "authoritative_reconciliation_points": "Explicit table of authoritative interior-bed residual observations used to reconcile the generalized longitudinal bed locally.",
    "authoritative_reconciliation_field": "Explicit station-indexed authoritative reconciliation field consumed downstream by station targets, scaffold, surface, and final conditioning.",
    "longitudinal_profile_elevation": "Rasterized longitudinal-profile bed-elevation tendency generated from the station-indexed profile points.",
    "longitudinal_profile_uncertainty": "Rasterized uncertainty of the longitudinal-profile bed-elevation tendency.",
    "longitudinal_profile_influence": "Distance-tapered influence of the rasterized longitudinal-profile tendency inside the river corridor.",
    "longitudinal_profile_local_authoritative_reconciliation": "Local authoritative reconciliation delta applied to the generalized longitudinal profile near authoritative interior-bed support.",
    "longitudinal_profile_local_authoritative_reconciliation_influence": "Influence of the local authoritative reconciliation applied to the generalized longitudinal profile.",
    "active_core_support_elevation": "Explicit monotone fluvial channel-core support raster exported for downstream scaffold and surface construction.",
    "active_core_support_uncertainty": "Uncertainty raster paired with the explicit monotone fluvial channel-core support.",
    "active_core_support_influence": "Influence raster paired with the explicit monotone fluvial channel-core support.",
    "reach_attributes": "Station-binned scientific reach units derived from the longitudinal-profile object so later modeling can operate on stable support-aware river segments instead of a monolithic river summary.",
    "reach_components": "Component-level scientific reach summary derived from the longitudinal-profile object and graph topology receipts.",
    "reach_attributes_summary": "Summary receipt for scientific reach segmentation, including topology/junction truth cross-checks against hydraulic backbone edges.",
    "channel_frame_points": "Regularized channel-frame stations that combine centerline chainage, authoritative in-channel anchors, XS support, and low-bank stage controls for future channel-fitted river-surface solving.",
    "channel_frame_contract": "Contract/receipt for the authoritative-first channel-frame scaffold including support-class counts and per-component spacing diagnostics.",
    "station_targets": "Canonical per-station target table defining the generalized-thalweg section tendency, bank-envelope bounds, permissions, and target source class that downstream scaffold and surface stages are expected to consume.",
    "station_targets_summary": "Summary receipt for the canonical per-station target table, including target-source and permission counts.",
    "anchor_table": "Canonical per-station anchor-policy table defining exact/curve/bank-only/non-anchor semantics and downstream lock/rebuild permissions.",
    "anchor_summary": "Summary receipt for the canonical anchor-policy table, including anchor-class and permission counts.",
    "authoritative_centerline_anchors": "Subset of channel-frame stations with authoritative in-channel DEM support; these are intended as hard longitudinal/channel-core anchors.",
    "authoritative_xs_anchors": "Subset of XS support nodes derived from authoritative in-channel samples, exported separately so future channel-fitted solvers can preserve observed cross-stream shape.",
    "channel_scaffold_nodes": "Regularized channel-fitted scaffold nodes spanning left bank, inner channel, thalweg, and right bank roles at each station.",
    "channel_scaffold_contract": "Contract/receipt for the regularized channel scaffold used to construct the channel-fitted surface.",
    "channel_surface": "First structured river channel-surface raster built from regularized scaffold nodes in a channel-fitted frame; intended to carry the main channel shape into final conditioning more directly than corridor-spread hints.",
    "channel_surface_confidence": "Confidence of the channel-fitted river surface, reflecting source-class strength and position within the channel frame.",
    "channel_surface_source_class": "Integer-coded source class for the channel-fitted river surface distinguishing authoritative, XS-resampled, resolved-bed, and bank-prior driven cells.",
    "channel_surface_support_count": "Count of finite scaffold node roles contributing local cross-stream support at each corridor cell.",
    "channel_surface_xs_participation": "Binary raster marking corridor cells where xs_profile_resampled scaffold support participated in the final channel-surface construction.",
    "channel_surface_authoritative_participation": "Binary raster marking corridor cells where authoritative scaffold or in-channel support participated in the final channel-surface construction.",
    "channel_surface_influence_class": "Primary propagation audit class for each corridor cell: authoritative-only, XS-only, mixed authoritative+XS, or other scaffold-only.",
    "channel_surface_xs_admissibility_mask": "Support-aware corridor mask showing where XS influence is admissible in the final channel-surface stage because the pixel is not authoritative-locked.",
    "channel_surface_authoritative_lock_scope": "Support-aware corridor mask showing where exact authoritative support is allowed to hard-lock the final channel surface.",
    "channel_surface_authoritative_lock_applied": "Cells where the final channel surface was actually hard-locked to authoritative support after support-aware scope filtering.",
    "channel_surface_control_nodes": "Explicit channel-surface control-node export with admission reason, support class, and XS-expected flags for tracing scaffold-to-surface propagation.",
    "river_xs_propagation_audit": "Audit receipt tracing XS structure from scaffold nodes into admitted channel-surface controls and final populated corridor cells.",
    "channel_surface_contract": "Contract/receipt describing how the channel-fitted river surface was built from scaffold nodes.",
    "longitudinal_tendency_profile": "Station-indexed scientific receipt showing before/after longitudinal bed tendency, deltas, anchor distance, and reach context for the active channel-surface pass.",
    "longitudinal_tendency_summary": "Summary receipt for the active reach-aware longitudinal bed-tendency pass used to reduce paneling and abrupt station-to-station bed shifts between sparse anchors.",
    "xs_realism_profile": "Station-indexed scientific receipt showing before/after cross-section inner-node realism ratios, reach context, and per-station adjustment magnitudes for the active channel-surface pass.",
    "xs_realism_summary": "Summary receipt for the active support-aware cross-section realism pass used to improve inner-channel shape consistency without changing authoritative locks or bank boundary controls.",
    "prediction_confidence_profile": "Station-indexed scientific receipt showing support-aware prediction confidence, measured-anchor fraction, structure-only fraction, and admissibility along the active river surface backbone.",
    "prediction_confidence_summary": "Summary receipt for the active support-aware prediction confidence/admissibility pass distinguishing measured-anchor-supported shape from structure-only shape.",
    "channel_surface_prediction_support_confidence": "Scientific support-aware prediction confidence raster for the channel-fitted river surface, distinct from raw graph/source confidence.",
    "channel_surface_measured_anchor_fraction": "Scientific raster showing the station-level fraction of direct measured anchor support carried into the channel-surface prediction context.",
    "channel_surface_structure_only_fraction": "Scientific raster showing where channel shape is predominantly structure-only rather than directly measured-anchor-supported.",
    "channel_surface_low_support_caution": "Scientific caution mask highlighting weak-support river predictions that should be interpreted conservatively during evaluation.",
    "channel_surface_prediction_admissibility": "Scientific admissibility mask for river prediction evaluation, preserving authoritative cells while screening weak structure-only predictions.",
    "xs_participation_contract": "Contract showing whether XS is merely structural or active bed support in this run, based on frame/surface evidence rather than artifact presence alone.",
}


def _build_river_guidance_notes(guidance: Dict[str, Any]) -> Dict[str, Any]:
    notes = dict(_RIVER_GUIDANCE_STATIC_NOTES)
    notes.update({
        "trusted_interior": guidance.get("trusted_interior_definition"),
        "soft_guidance_domain": guidance.get("soft_guidance_definition"),
        "admissibility": guidance.get("admissibility_definition"),
    })
    return notes


def _simple_river_stage_status_for_legacy_route(xs_support_status: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stage_status = simple_river_stage_status_placeholder()
    legacy_map = {
        STAGE_AUTHORITATIVE_BASE: ("implemented_via_existing_pipeline", "authoritative_base/outputs/aligned_authoritative_base"),
        STAGE_RIVER_GUIDANCE_DOMAIN: ("implemented_via_existing_pipeline", "corridor_mask"),
        STAGE_RIVER_CENTERLINE: ("implemented_via_existing_pipeline", "centerline_points"),
        STAGE_CENTERLINE_WSE_PROXY: ("not_yet_implemented", "centerline_elevation"),
        STAGE_CENTERLINE_AUTHORITATIVE_BED: ("not_yet_implemented", "authoritative_bed_anchor_curve"),
        STAGE_CENTERLINE_OBSERVED_OFFSET: ("not_yet_implemented", None),
        STAGE_CENTERLINE_OFFSET_MODELED: ("not_yet_implemented", "longitudinal_profile"),
        STAGE_CENTERLINE_BED_BACKBONE: ("not_yet_implemented", "hydraulic_backbone"),
        STAGE_RIVER_PRIMARY_SURFACE: ("implemented_via_existing_pipeline", "channel_surface"),
        STAGE_RIVER_PRIMARY_SURFACE_LOCKED: ("not_yet_implemented", None),
        STAGE_CONDITIONED_FINAL_INTERNAL: ("implemented_via_existing_pipeline", "combined/conditioned_final_dem_internal.tif"),
        STAGE_FINAL_DEM: ("implemented_via_existing_pipeline", "combined/DEM_enhanced.tif"),
    }
    for stage_id, (status, legacy_equivalent) in legacy_map.items():
        if stage_id not in stage_status:
            continue
        stage_status[stage_id]["status"] = status
        stage_status[stage_id]["legacy_equivalent"] = legacy_equivalent
        stage_status[stage_id]["implemented"] = status.startswith("implemented")
    stage_status[STAGE_CENTERLINE_AUTHORITATIVE_BED]["notes"] = {
        "xs_bed_support_active": bool(xs_support_status.get("bed_support_active")),
        "xs_structural_artifacts_active": bool(xs_support_status.get("structural_artifacts_active")),
    }
    return stage_status


def _build_river_structural_artifact_contract(xs_support_status: Dict[str, Any]) -> Dict[str, Any]:
    allowed_structural_artifacts = list(_RIVER_GUIDANCE_BASE_ALLOWED_STRUCTURAL_ARTIFACTS)
    optional_structural_artifacts = set(_RIVER_GUIDANCE_BASE_OPTIONAL_STRUCTURAL_ARTIFACTS)
    if bool(xs_support_status.get("structural_artifacts_active")):
        allowed_structural_artifacts.extend(sorted(_RIVER_GUIDANCE_XS_ARTIFACTS))
        if not bool(xs_support_status.get("bed_support_active")):
            optional_structural_artifacts.update(_RIVER_GUIDANCE_XS_ARTIFACTS)
    else:
        optional_structural_artifacts.update(_RIVER_GUIDANCE_XS_ARTIFACTS)
    allowed_structural_artifacts = sorted(dict.fromkeys(allowed_structural_artifacts))
    required_structural_artifacts = sorted(
        name for name in allowed_structural_artifacts
        if name not in optional_structural_artifacts
    )
    optional_structural_artifacts_sorted = sorted(
        name for name in allowed_structural_artifacts if name not in set(required_structural_artifacts)
    )
    return {
        "route_mode": ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
        "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        "allowed_structural_artifacts": allowed_structural_artifacts,
        "required_structural_artifacts": required_structural_artifacts,
        "optional_structural_artifacts": optional_structural_artifacts_sorted,
        "simple_river_stage_status": _simple_river_stage_status_for_legacy_route(xs_support_status),
    }


def _collect_existing_output_artifacts(
    *,
    outputs: Dict[str, Any],
    out_root: Path,
    canonical_paths: Dict[str, Path],
) -> tuple[Dict[str, str], list[str]]:
    def _path_from_outputs(value: Any) -> Optional[Path]:
        if isinstance(value, Path):
            p = value
        elif isinstance(value, str) and value.strip():
            p = Path(value)
        else:
            return None
        if not p.is_absolute():
            p = (out_root / p).resolve()
        return p

    artifacts: Dict[str, str] = {}
    for key, raw in outputs.items():
        if key == "guidance_manifest":
            continue
        path_obj = _path_from_outputs(raw)
        if path_obj is None or not path_obj.exists():
            continue
        try:
            artifacts[key] = str(path_obj.relative_to(out_root))
        except ValueError:
            artifacts[key] = str(path_obj)

    missing = [
        key for key, canonical in canonical_paths.items()
        if key != "guidance_manifest" and canonical.exists() and key not in outputs
    ]
    return artifacts, missing


def guidance_artifact_paths(river_dir: str | Path) -> Dict[str, Path]:
    river_dir = Path(river_dir)
    return {
        "guidance_weight": river_dir / "river_guidance_weight.tif",
        "trusted_interior": river_dir / "river_trusted_interior.tif",
        "soft_guidance_domain": river_dir / "river_soft_guidance_domain.tif",
        "admissibility": river_dir / "river_admissibility.tif",
        "regime_class": river_dir / "river_regime_class.tif",
        "guide_points": river_dir / "river_guide_points.gpkg",
        "authoritative_support": river_dir / "river_authoritative_support.tif",
        "authoritative_support_depth": river_dir / "river_authoritative_support_depth.tif",
        "corridor_mask": river_dir / "river_corridor_mask.tif",
        "bank_edge_mask": river_dir / "river_bank_edge_mask.tif",
        "bank_distance": river_dir / "river_bank_distance_m.tif",
        "bank_influence": river_dir / "river_bank_influence.tif",
        "bank_elevation_xs": river_dir / "river_bank_elevation_xs.tif",
        "bank_pair_weight": river_dir / "river_bank_pair_weight.tif",
        "bank_continuity_weight": river_dir / "river_bank_continuity_weight.tif",
        "bank_graph_confidence": river_dir / "river_bank_graph_confidence.tif",
        "bank_confluence_damping": river_dir / "river_bank_confluence_damping.tif",
        "bank_estuary_side_decay": river_dir / "river_bank_estuary_side_decay.tif",
        "bank_points": river_dir / "river_bank_points.gpkg",
        "bank_longitudinal_fit_points": river_dir / "river_bank_longitudinal_fit_points.gpkg",
        "bank_longitudinal_fit_summary": river_dir / "river_bank_longitudinal_fit_summary.json",
        "left_bank_fit_elevation": river_dir / "river_left_bank_fit_elevation.tif",
        "right_bank_fit_elevation": river_dir / "river_right_bank_fit_elevation.tif",
        "bank_pair_fit_elevation": river_dir / "river_bank_pair_fit_elevation.tif",
        "authoritative_bed_anchor_curve": river_dir / "river_authoritative_bed_anchor_curve.csv",
        "authoritative_bed_anchor_curve_summary": river_dir / "river_authoritative_bed_anchor_curve_summary.json",
        "centerline_points": river_dir / "river_centerline_points.gpkg",
        "xs_support_points": river_dir / "river_xs_support_points.gpkg",
        "centerline_elevation": river_dir / "river_centerline_elevation.tif",
        "centerline_influence": river_dir / "river_centerline_influence.tif",
        "centerline_stationing": river_dir / "river_centerline_stationing_m.tif",
        "xs_support_elevation": river_dir / "river_xs_support_elevation.tif",
        "xs_support_weight": river_dir / "river_xs_support_weight.tif",
        "retained_network": river_dir / "river_retained_network.gpkg",
        "scaffold_domains": river_dir / "river_scaffold_domains.json",
        "scaffold_manifest": river_dir / "river_scaffold_manifest.json",
        "guidance_manifest": river_dir / "river_guidance_manifest.json",
        "depth_terrain": river_dir / "river_depth.tif",
        "bottom_elevation": river_dir / "river_bed_elev.tif",
        "longitudinal_profile": river_dir / "river_longitudinal_profile.csv",
        "longitudinal_profile_points": river_dir / "river_longitudinal_profile_points.gpkg",
        "longitudinal_profile_summary": river_dir / "river_longitudinal_profile_summary.json",
        "authoritative_reconciliation_points": river_dir / "river_authoritative_reconciliation_points.csv",
        "authoritative_reconciliation_field": river_dir / "river_authoritative_reconciliation_field.csv",
        "longitudinal_profile_elevation": river_dir / "river_longitudinal_profile_elevation.tif",
        "generalized_longitudinal_bed_base": river_dir / "river_generalized_longitudinal_bed_base_elevation.tif",
        "generalized_longitudinal_bed_reconciled": river_dir / "river_generalized_longitudinal_bed_reconciled_elevation.tif",
        "longitudinal_profile_uncertainty": river_dir / "river_longitudinal_profile_uncertainty.tif",
        "longitudinal_profile_influence": river_dir / "river_longitudinal_profile_influence.tif",
        "longitudinal_profile_local_authoritative_reconciliation": river_dir / "river_longitudinal_profile_local_authoritative_reconciliation.tif",
        "longitudinal_profile_local_authoritative_reconciliation_influence": river_dir / "river_longitudinal_profile_local_authoritative_reconciliation_influence.tif",
        "active_core_support_elevation": river_dir / "river_active_core_support_elevation.tif",
        "active_core_support_uncertainty": river_dir / "river_active_core_support_uncertainty.tif",
        "active_core_support_influence": river_dir / "river_active_core_support_influence.tif",
        "hydraulic_backbone": river_dir / "river_hydraulic_backbone.csv",
        "hydraulic_backbone_nodes": river_dir / "river_hydraulic_backbone_nodes.gpkg",
        "hydraulic_backbone_edges": river_dir / "river_hydraulic_backbone_edges.gpkg",
        "reach_attributes": river_dir / "river_reach_attributes.csv",
        "reach_components": river_dir / "river_reach_components.csv",
        "reach_attributes_summary": river_dir / "river_reach_attributes_summary.json",
        "channel_frame_points": river_dir / "river_channel_frame_points.gpkg",
        "channel_frame_contract": river_dir / "river_channel_frame_contract.json",
        "station_targets": river_dir / "river_station_targets.csv",
        "station_targets_summary": river_dir / "river_station_targets_summary.json",
        "anchor_table": river_dir / "river_anchor_table.csv",
        "anchor_summary": river_dir / "river_anchor_summary.json",
        "authoritative_centerline_anchors": river_dir / "river_authoritative_centerline_anchors.gpkg",
        "authoritative_xs_anchors": river_dir / "river_authoritative_xs_anchors.gpkg",
        "channel_scaffold_nodes": river_dir / "river_channel_scaffold_nodes.gpkg",
        "channel_scaffold_contract": river_dir / "river_channel_scaffold_contract.json",
        "channel_surface": river_dir / "river_channel_surface.tif",
        "channel_surface_confidence": river_dir / "river_channel_surface_confidence.tif",
        "channel_surface_source_class": river_dir / "river_channel_surface_source_class.tif",
        "channel_surface_support_count": river_dir / "river_channel_surface_support_count.tif",
        "channel_surface_xs_participation": river_dir / "river_channel_surface_xs_participation.tif",
        "channel_surface_authoritative_participation": river_dir / "river_channel_surface_authoritative_participation.tif",
        "channel_surface_influence_class": river_dir / "river_channel_surface_influence_class.tif",
        "channel_surface_xs_admissibility_mask": river_dir / "river_channel_surface_xs_admissibility_mask.tif",
        "channel_surface_authoritative_lock_scope": river_dir / "river_channel_surface_authoritative_lock_scope.tif",
        "channel_surface_authoritative_lock_applied": river_dir / "river_channel_surface_authoritative_lock_applied.tif",
        "channel_surface_control_nodes": river_dir / "river_channel_surface_control_nodes.gpkg",
        "river_xs_propagation_audit": river_dir / "river_xs_propagation_audit.json",
        "channel_surface_contract": river_dir / "river_channel_surface_contract.json",
        "longitudinal_tendency_profile": river_dir / "river_longitudinal_tendency_profile.csv",
        "longitudinal_tendency_summary": river_dir / "river_longitudinal_tendency_summary.json",
        "xs_realism_profile": river_dir / "river_xs_realism_profile.csv",
        "xs_realism_summary": river_dir / "river_xs_realism_summary.json",
        "prediction_confidence_profile": river_dir / "river_prediction_confidence_profile.csv",
        "prediction_confidence_summary": river_dir / "river_prediction_confidence_summary.json",
        "channel_surface_prediction_support_confidence": river_dir / "river_channel_surface_prediction_support_confidence.tif",
        "channel_surface_measured_anchor_fraction": river_dir / "river_channel_surface_measured_anchor_fraction.tif",
        "channel_surface_structure_only_fraction": river_dir / "river_channel_surface_structure_only_fraction.tif",
        "channel_surface_low_support_caution": river_dir / "river_channel_surface_low_support_caution.tif",
        "channel_surface_prediction_admissibility": river_dir / "river_channel_surface_prediction_admissibility.tif",
        "runtime_diagnostics": river_dir / "river_runtime_diagnostics.json",
    }


def _load_and_validate_centerline_component_contract(
    centerline_points_path: str | Path | None,
    *,
    retained_network_meta: Optional[Dict[str, Any]] = None,
    expected_component_count: int = 0,
    expected_component_source: str | None = None,
    logger: Optional[logging.Logger] = None,
):
    if centerline_points_path is None:
        raise RuntimeError('centerline_points_path_missing_before_longitudinal_profile')
    path = Path(centerline_points_path)
    if not path.exists():
        raise RuntimeError(f'centerline_points_path_missing_before_longitudinal_profile:{path}')
    upstream_meta = resolve_centerline_component_expectation(
        retained_network_meta,
        station_contract_path=(retained_network_meta or {}).get('centerline_station_contract_path') if isinstance(retained_network_meta, dict) else None,
    )
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(f'geopandas_required_for_centerline_contract_validation:{exc}') from exc
    centerline_points = gpd.read_file(path)
    contract = validate_centerline_station_component_contract(
        centerline_points,
        expected_component_count=int(expected_component_count or upstream_meta.get('expected_component_count', 0) or 0),
        expected_component_source=(
            expected_component_source
            or upstream_meta.get('component_id_source')
        ),
        logger=logger,
        context='centerline_points_file',
    )
    contract['upstream_component_count_before'] = int(upstream_meta.get('component_count_before', 0) or 0)
    contract['upstream_component_count_after'] = int(upstream_meta.get('component_count_after', 0) or 0)
    contract['upstream_station_contract_loaded'] = bool(upstream_meta.get('station_contract_loaded', False))
    contract['upstream_station_contract_path'] = upstream_meta.get('station_contract_path')
    return contract


def _resolve_retained_network_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    candidates = [
        river.get("retained_network"),
        (river.get("guidance", {}) or {}).get("retained_network_summary") if isinstance(river.get("guidance", {}), dict) else None,
        (river.get("structured_stage", {}) or {}).get("retained_network_summary") if isinstance(river.get("structured_stage", {}), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}

def _derive_xs_support_status(report: Dict[str, Any], river_outputs: Dict[str, Any]) -> Dict[str, Any]:
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    retained = _resolve_retained_network_summary(report)
    execution = river.get("execution_receipts", {}) if isinstance(river.get("execution_receipts", {}), dict) else {}
    contract = retained.get("xs_support_contract", {}) if isinstance(retained.get("xs_support_contract", {}), dict) else {}
    if bool(execution.get("xs_influence_disabled", False)):
        return {
            "status": "disabled_by_option",
            "reason": "river_disable_xs_influence",
            "retained_point_count": 0,
            "legacy_reverticalized_used": False,
            "existing_artifacts": [],
            "structural_artifacts_active": False,
            "bed_support_active": False,
            "frame_xs_candidate_count": 0,
            "frame_xs_supported_count": 0,
            "frame_authoritative_xs_anchor_count": 0,
            "scaffold_xs_profile_node_count": 0,
            "scaffold_xs_only_station_count": 0,
            "surface_xs_profile_cell_count": 0,
            "contract_authoritative_sample_count": 0,
            "contract_fallback_sample_count": 0,
            "contract_missing_sample_count": 0,
            "retained_sample_source_counts": {},
            "evidence": {"disabled_by_option": True},
        }

    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _load_json_metrics(path_value: Any) -> Dict[str, Any]:
        try:
            if not path_value:
                return {}
            path = Path(str(path_value))
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("metrics", {}) if isinstance(payload, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            log.debug("_derive_xs_support_status: failed to load metrics for %s", path_value, exc_info=True)
            return {}

    xs_output_artifacts = {
        key: bool(river_outputs.get(key))
        for key in (
            "xs_support_points",
            "xs_support_elevation",
            "xs_support_weight",
            "authoritative_xs_anchors",
        )
    }

    retained_count = max(
        _safe_int(contract.get("kept_final_count")),
        _safe_int(retained.get("xs_support_point_count", 0)),
        _safe_int(contract.get("sampled_point_count")),
    )
    contract_authoritative = _safe_int(contract.get("authoritative_sample_count"))
    contract_fallback = _safe_int(contract.get("fallback_sample_count"))
    contract_missing = _safe_int(contract.get("missing_sample_count"))
    contract_kept_pre_allowed = _safe_int(contract.get("kept_pre_allowed_count"))
    sample_source_counts = retained.get("xs_support_sample_source_counts", {}) if isinstance(retained.get("xs_support_sample_source_counts", {}), dict) else {}
    legacy_used = bool(execution.get("legacy_xs_inputs_used", False))

    frame_metrics = _load_json_metrics(river_outputs.get("channel_frame_contract"))
    scaffold_metrics = _load_json_metrics(river_outputs.get("channel_scaffold_contract"))
    surface_metrics = _load_json_metrics(river_outputs.get("channel_surface_contract"))

    frame_xs_supported = _safe_int(frame_metrics.get("support_class_counts", {}).get("xs_supported"))
    frame_auth_xs = _safe_int(frame_metrics.get("authoritative_xs_anchor_count"))
    frame_xs_candidate = _safe_int(frame_metrics.get("xs_candidate_station_count"))
    scaffold_xs_nodes = _safe_int(scaffold_metrics.get("xs_profile_resampled_node_count"))
    scaffold_xs_only_stations = _safe_int(scaffold_metrics.get("xs_only_station_count"))
    surface_xs_cells = _safe_int(surface_metrics.get("xs_profile_cells"))

    # Truthful bed-support status requires evidence that XS support survives into the
    # frame or final surface, not merely that scaffold-side structure exists.
    bed_support_active = bool(
        legacy_used
        or retained_count > 0
        or contract_authoritative > 0
        or frame_xs_supported > 0
        or frame_auth_xs > 0
        or surface_xs_cells > 0
    )
    structural_artifacts_present = bool(
        retained_count > 0
        or contract_kept_pre_allowed > 0
        or frame_xs_candidate > 0
        or scaffold_xs_nodes > 0
        or scaffold_xs_only_stations > 0
        or surface_xs_cells > 0
        or any(xs_output_artifacts.values())
    )

    contract_status = str(contract.get("status") or "").strip().lower()
    if contract_status in {"xs_source_missing_or_not_requested", "no_xs_source"} and not structural_artifacts_present and not bed_support_active:
        status = "no_xs_source"
        reason = contract_status or "xs_source_missing_or_not_requested"
    elif bed_support_active:
        status = "active_bed_support"
        if legacy_used:
            reason = "reverticalized_xs_support_active"
        elif surface_xs_cells > 0:
            reason = "xs_support_reaches_final_surface"
        elif frame_xs_supported > 0 or frame_auth_xs > 0:
            reason = "xs_support_active_in_channel_frame"
        elif contract_authoritative > 0 or retained_count > 0:
            reason = "sampled_xs_support_active"
        else:
            reason = "xs_bed_support_active"
    elif structural_artifacts_present:
        status = "structural_only"
        if scaffold_xs_nodes > 0 or scaffold_xs_only_stations > 0:
            reason = "xs_scaffold_structure_present_without_bed_support"
        elif frame_xs_candidate > 0:
            reason = "xs_candidates_present_without_bed_support"
        else:
            reason = "xs_structural_artifacts_present_without_bed_support"
    else:
        status = "inactive"
        reason = contract_status or "no_usable_xs_support"

    evidence = {
        "sampled_points": int(retained_count),
        "authoritative_samples": int(contract_authoritative),
        "fallback_samples": int(contract_fallback),
        "missing_samples": int(contract_missing),
        "frame_xs_candidates": int(frame_xs_candidate),
        "frame_xs_supported": int(frame_xs_supported),
        "frame_authoritative_xs_anchors": int(frame_auth_xs),
        "scaffold_xs_profile_nodes": int(scaffold_xs_nodes),
        "scaffold_xs_only_stations": int(scaffold_xs_only_stations),
        "surface_xs_profile_cells": int(surface_xs_cells),
    }
    return {
        "status": status,
        "reason": reason,
        "retained_point_count": int(retained_count),
        "legacy_reverticalized_used": legacy_used,
        "existing_artifacts": sorted(k for k, v in {
            "xs_support_points": xs_output_artifacts.get("xs_support_points", False) or retained_count > 0,
            "authoritative_xs_anchors": xs_output_artifacts.get("authoritative_xs_anchors", False) or frame_auth_xs > 0 or contract_authoritative > 0,
            "xs_support_elevation": xs_output_artifacts.get("xs_support_elevation", False) or frame_xs_candidate > 0 or retained_count > 0,
            "xs_support_weight": xs_output_artifacts.get("xs_support_weight", False) or frame_xs_candidate > 0 or retained_count > 0,
        }.items() if v),
        "structural_artifacts_active": bool(structural_artifacts_present),
        "bed_support_active": bool(bed_support_active),
        "frame_xs_candidate_count": int(frame_xs_candidate),
        "frame_xs_supported_count": int(frame_xs_supported),
        "frame_authoritative_xs_anchor_count": int(frame_auth_xs),
        "scaffold_xs_profile_node_count": int(scaffold_xs_nodes),
        "scaffold_xs_only_station_count": int(scaffold_xs_only_stations),
        "surface_xs_profile_cell_count": int(surface_xs_cells),
        "contract_authoritative_sample_count": int(contract_authoritative),
        "contract_fallback_sample_count": int(contract_fallback),
        "contract_missing_sample_count": int(contract_missing),
        "retained_sample_source_counts": {str(k): int(v) for k, v in sample_source_counts.items()},
        "evidence": evidence,
    }



def _write_xs_participation_contract(*, river_dir: Path, xs_status: Dict[str, Any]) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_family": "river_xs_participation",
        "notes": {
            "objective": "Summarize whether XS inputs are merely structural artifacts or active bed-support evidence in this run.",
            "truth_rule": "XS is active bed support only when sampled/authoritative XS evidence survives into the frame or final surface; scaffold-side XS structure alone is structural-only.",
        },
        "metrics": dict(xs_status),
    }
    path = river_dir / "river_xs_participation_contract.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def _coalesce_river_guide_values(gdf):
    """Return a per-row guide elevation vector from mixed structured guide-point layers."""
    import pandas as pd
    if gdf is None or len(gdf) == 0:
        return pd.Series(dtype='float64')
    preferred = [
        "xs_z_m",
        "centerline_z_m",
        "bank_z_m",
        "authoritative_z_m",
        "depth_m",
        "bottom_elevation",
        "bed_elev",
        "elevation_m",
        "elevation",
        "depth",
        "z",
        "value",
    ]
    cols = {str(c).lower(): c for c in gdf.columns}
    resolved = [cols[name] for name in preferred if name in cols]
    if not resolved:
        resolved = [orig for raw, orig in cols.items() if raw.endswith('_z_m') or raw.endswith('_elevation')]
    if not resolved:
        return pd.Series(np.nan, index=gdf.index, dtype='float64')
    out = pd.Series(np.nan, index=gdf.index, dtype='float64')
    for col in resolved:
        vals = pd.to_numeric(gdf[col], errors='coerce').astype('float64')
        take = (~np.isfinite(out.to_numpy(dtype='float64'))) & np.isfinite(vals.to_numpy(dtype='float64'))
        if np.any(take):
            out.loc[take] = vals.loc[take]
    return out.astype('float64')


def rasterize_river_guide_points_to_template(guide_points_path: str | Path, template_raster: str | Path, *, logger: Optional[logging.Logger] = None):
    import numpy as np

    import rasterio
    from rasterio.errors import RasterioError
    from rasterio.transform import rowcol
    try:
        import geopandas as gpd
    except ImportError:
        active_log = logger or logging.getLogger(__name__)
        active_log.warning("rasterize_river_guide_points_to_template: geopandas unavailable", exc_info=True)
        return None

    gp = Path(guide_points_path)
    tmpl = Path(template_raster)
    active_log = logger or logging.getLogger(__name__)
    if (not gp.exists()) or (not tmpl.exists()):
        return None
    mem_start = memory_checkpoint("river_guide_rasterize_start", guide_points=str(gp), template=str(tmpl))
    active_log.info("[MEMORY][RIVER] %s", mem_start)

    try:
        gdf = gpd.read_file(gp)
    except (OSError, ValueError, RuntimeError) as exc:
        active_log.warning("rasterize_river_guide_points_to_template: failed reading %s (%s)", gp, exc, exc_info=True)
        return None
    if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
        return None
    geom_valid = gdf.geometry.apply(lambda geom: geom is not None and (not getattr(geom, 'is_empty', True)))
    gdf = gdf.loc[geom_valid.to_numpy(dtype=bool)].copy()
    if gdf.empty:
        return None
    guide_vals = _coalesce_river_guide_values(gdf)
    if guide_vals.empty or not np.any(np.isfinite(np.asarray(guide_vals, dtype=float))):
        value_col = _pick_river_guide_value_column(list(gdf.columns))
        if value_col is None:
            return None
        guide_vals = pd.to_numeric(gdf[value_col], errors='coerce')
    gdf = gdf.copy()
    gdf['guide_z_m'] = np.asarray(guide_vals, dtype='float64')
    gdf = gdf[np.isfinite(gdf['guide_z_m'].to_numpy(dtype='float64'))].copy()
    if gdf.empty:
        return None

    try:
        with rasterio.open(tmpl) as ds:
            if gdf.crs is not None and ds.crs is not None:
                try:
                    gdf = _reproject_gdf_to_raster_crs(gdf, ds)
                except (TypeError, ValueError, RuntimeError) as exc:
                    active_log.warning("rasterize_river_guide_points_to_template: failed CRS conversion to %s (%s)", ds.crs, exc, exc_info=True)
                    return None
            geom_x = np.asarray(gdf.geometry.x, dtype=float)
            geom_y = np.asarray(gdf.geometry.y, dtype=float)
            finite_geom = np.isfinite(geom_x) & np.isfinite(geom_y)
            if not np.all(finite_geom):
                active_log.debug(
                    "rasterize_river_guide_points_to_template: dropping %d non-finite guide geometries before interpolation",
                    int(np.size(finite_geom) - np.count_nonzero(finite_geom)),
                )
                gdf = gdf.loc[finite_geom].copy()
            if gdf.empty:
                return None

            # Prefer a continuous guide surface from the mixed structured guide cloud.
            # This lets re-verticalized XS points actually affect the fallback river guidance
            # raster instead of collapsing the whole file to whichever single z-column won.
            arr = None
            try:
                from xs_infer_bathy_raster import _continuous_surface
                pixel_size_m = max(abs(float(ds.transform.a)), abs(float(ds.transform.e)), 1.0)
                arr_raw, arr_mask = _continuous_surface(
                    pts_gdf=gdf,
                    value_col='guide_z_m',
                    template_ds=ds,
                    method='walid_aniso',
                    buffer_m=max(pixel_size_m * 4.0, 12.0),
                    k=12,
                    idw_power=2.0,
                    aniso_along_scale_m=500.0,
                    aniso_cross_scale_m=max(pixel_size_m * 10.0, 30.0),
                    thalweg_weight=6.0,
                    nodata=-9999.0,
                    overlap_reducer='min',
                    max_query_dist_m=250.0,
                )
                arr = np.asarray(arr_raw, dtype=np.float32)
                arr[arr == np.float32(-9999.0)] = np.nan
                if arr_mask is not None:
                    arr[np.asarray(arr_mask) <= 0] = np.nan
            except ImportError:
                active_log.warning("rasterize_river_guide_points_to_template: continuous surface dependencies unavailable; falling back to point-cell averaging", exc_info=True)
                arr = None
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                active_log.warning("rasterize_river_guide_points_to_template: continuous surface build failed; falling back to point-cell averaging (%s)", exc, exc_info=True)
                arr = None

            if arr is None or not np.any(np.isfinite(arr)):
                arr = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
                sums = np.zeros((ds.height, ds.width), dtype=np.float64)
                counts = np.zeros((ds.height, ds.width), dtype=np.uint32)
                vals = np.asarray(gdf['guide_z_m'], dtype=float)
                for geom, val in zip(gdf.geometry, vals):
                    if geom is None or not np.isfinite(val):
                        continue
                    gx = float(getattr(geom, "x", np.nan))
                    gy = float(getattr(geom, "y", np.nan))
                    if (not np.isfinite(gx)) or (not np.isfinite(gy)):
                        active_log.debug("rasterize_river_guide_points_to_template: skipping non-finite point geometry")
                        continue
                    try:
                        r, c = rowcol(ds.transform, gx, gy)
                    except (TypeError, ValueError, OverflowError):
                        active_log.debug("rasterize_river_guide_points_to_template: skipping invalid point geometry", exc_info=True)
                        continue
                    if 0 <= int(r) < ds.height and 0 <= int(c) < ds.width:
                        sums[int(r), int(c)] += float(val)
                        counts[int(r), int(c)] += 1
                valid = counts > 0
                if not np.any(valid):
                    return None
                arr[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    except RasterioError as exc:
        active_log.warning("rasterize_river_guide_points_to_template: failed opening template %s (%s)", tmpl, exc, exc_info=True)
        return None
    populated = int(np.sum(np.isfinite(arr)))
    if populated == 0:
        active_log.warning("Rasterized river guide points produced zero populated cells: %s", gp)
        return None
    mem_end = memory_checkpoint(
        "river_guide_rasterize_end",
        guide_points=str(gp),
        template=str(tmpl),
        populated_cells=populated,
        point_rows=int(len(gdf)),
    )
    active_log.info('Rasterized river guide points onto template grid: %s -> %s populated cells', gp, populated)
    active_log.info('[MEMORY][RIVER] %s', mem_end)
    return arr


def _normalize_geodataframe_crs(gdf):
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    crs = getattr(gdf, "crs", None)
    if crs is None:
        return gdf
    normalized = CRS.from_user_input(crs)
    return gdf.set_crs(normalized, allow_override=True)


def _reproject_gdf_to_raster_crs(gdf, dataset):
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    if getattr(gdf, "crs", None) is None:
        raise ValueError("Guide points CRS missing.")
    if getattr(dataset, "crs", None) is None:
        raise ValueError("Template raster CRS missing.")
    source_crs = CRS.from_user_input(gdf.crs)
    target_crs = CRS.from_user_input(dataset.crs)
    normalized = gdf.copy()
    normalized = normalized.set_crs(source_crs, allow_override=True)
    same_crs = bool(source_crs == target_crs)
    try:
        same_crs = same_crs or bool(source_crs.equals(target_crs))
    except Exception:
        log.debug("_reproject_gdf_to_raster_crs: CRS equals() check failed", exc_info=True)
    if same_crs:
        return normalized
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    from shapely.ops import transform as _transform
    reprojected = normalized.copy()
    reprojected.geometry = reprojected.geometry.apply(
        lambda geom: geom if (geom is None or geom.is_empty) else _transform(lambda x, y, z=None: transformer.transform(x, y), geom)
    )
    return reprojected.set_crs(target_crs, allow_override=True)


def build_river_guidance_manifest(*, out_root: str | Path, river_dir: str | Path, report: Dict[str, Any]) -> Dict[str, Any]:
    out_root = Path(out_root)
    river_dir = Path(river_dir)
    river_report = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    outputs = river_report.get("outputs", {}) if isinstance(river_report.get("outputs", {}), dict) else {}
    guidance = river_report.get("guidance", {}) if isinstance(river_report.get("guidance", {}), dict) else {}

    canonical_paths = guidance_artifact_paths(river_dir)
    artifacts, output_contract_missing_from_outputs = _collect_existing_output_artifacts(
        outputs=outputs,
        out_root=out_root,
        canonical_paths=canonical_paths,
    )
    xs_support_status = _derive_xs_support_status(report, outputs)
    route_contract = _build_river_structural_artifact_contract(xs_support_status)
    report_stage_status = river_report.get("simple_stage_status") if isinstance(river_report.get("simple_stage_status"), dict) else None
    v2_pass1 = river_report.get("v2_pass1") if isinstance(river_report.get("v2_pass1"), dict) else {}
    v2_pass2 = river_report.get("v2_pass2") if isinstance(river_report.get("v2_pass2"), dict) else {}
    v2_pass3 = river_report.get("v2_pass3") if isinstance(river_report.get("v2_pass3"), dict) else {}
    v2_pass4 = river_report.get("v2_pass4") if isinstance(river_report.get("v2_pass4"), dict) else {}
    if report_stage_status:
        route_contract["simple_river_stage_status"] = report_stage_status

    manifest = {
        "schema_version": 2,
        "artifact_family": "river_guidance",
        "guidance_only": True,
        "current_route_mode": route_contract.get("route_mode"),
        "target_route_mode": route_contract.get("target_route_mode"),
        "artifacts": artifacts,
        "artifact_roles": dict(_RIVER_GUIDANCE_ARTIFACT_ROLES),
        "final_route_contract": {
            "route_role": "structured_subordinate_river_guidance_only",
            **route_contract,
            "diagnostic_only_artifacts": ["bottom_elevation", "depth_terrain"],
            "forbidden_structural_inputs": list(_RIVER_GUIDANCE_FORBIDDEN_STRUCTURAL_INPUTS),
        },
        "xs_support_status": xs_support_status,
        "simple_river_stage_status": route_contract.get("simple_river_stage_status", {}),
        "river_v2_pass1": {
            "execution_mode": v2_pass1.get("execution_mode"),
            "success": v2_pass1.get("success"),
            "stage_status": v2_pass1.get("stage_status", {}),
            "stage_results": v2_pass1.get("stage_results", {}),
            "failed_stage": v2_pass1.get("failed_stage"),
            "error": v2_pass1.get("error"),
        },
        "river_v2_pass2": {
            "execution_mode": v2_pass2.get("execution_mode"),
            "success": v2_pass2.get("success"),
            "stage_status": v2_pass2.get("stage_status", {}),
            "stage_results": v2_pass2.get("stage_results", {}),
            "failed_stage": v2_pass2.get("failed_stage"),
            "error": v2_pass2.get("error"),
        },
        "river_v2_pass3": {
            "execution_mode": v2_pass3.get("execution_mode"),
            "success": v2_pass3.get("success"),
            "stage_status": v2_pass3.get("stage_status", {}),
            "stage_results": v2_pass3.get("stage_results", {}),
            "failed_stage": v2_pass3.get("failed_stage"),
            "error": v2_pass3.get("error"),
        },
        "river_v2_pass4": {
            "execution_mode": v2_pass4.get("execution_mode"),
            "success": v2_pass4.get("success"),
            "stage_status": v2_pass4.get("stage_status", {}),
            "stage_results": v2_pass4.get("stage_results", {}),
            "failed_stage": v2_pass4.get("failed_stage"),
            "error": v2_pass4.get("error"),
        },
        "river_v2_final_route_participation": {
            "active": bool(v2_pass4.get("success")) and bool(outputs.get("primary_river_guidance_surface")),
            "active_raster": outputs.get("primary_river_guidance_surface"),
            "active_stage": "river_primary_surface_authoritative_applied" if (bool(v2_pass4.get("success")) and bool(outputs.get("primary_river_guidance_surface"))) else None,
            "legacy_river_final_route_participation_blocked": bool(v2_pass4.get("success")),
            "river_method_selected": "v2",
            "river_path_used": "river_v2_only",
            "legacy_river_path_participated": False,
            "pipeline_version": str(PIPELINE_VERSION),
        },
        "notes": _build_river_guidance_notes(guidance),
    }

    if output_contract_missing_from_outputs:
        manifest.setdefault("diagnostics", {})["output_contract_missing_from_outputs"] = sorted(output_contract_missing_from_outputs)
    return manifest


def write_river_guidance_manifest(*, out_root: str | Path, river_dir: str | Path, report: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Path:
    manifest = build_river_guidance_manifest(out_root=out_root, river_dir=river_dir, report=report)
    manifest_path = guidance_artifact_paths(river_dir)["guidance_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger or logging.getLogger(__name__)).info("Wrote river guidance manifest: %s", manifest_path)
    return manifest_path


def _write_channel_surface_primary_products(*, cfg, river_dir: Path, river_outputs: Dict[str, Any], logger: logging.Logger) -> Dict[str, str]:
    """Materialize primary river bed/depth products from channel_surface, keeping legacy hybrid bed as diagnostic only."""
    import shutil
    import rasterio
    import numpy as np


    channel_surface = Path(river_outputs.get("channel_surface")) if river_outputs.get("channel_surface") else None
    river_dem = Path(getattr(cfg, "river_dem", "") or "")
    if channel_surface is None or not channel_surface.exists() or not river_dem.exists():
        return {}
    primary_bed = river_dir / "river_bottom_navd88_patch_channel_surface.tif"
    primary_depth = river_dir / "river_depth_terrain_patch_channel_surface.tif"
    with rasterio.open(channel_surface) as bed_ds, rasterio.open(river_dem) as dem_ds:
        bed = bed_ds.read(1).astype(np.float32)
        bed_nodata = bed_ds.nodata if bed_ds.nodata is not None else float(getattr(cfg, "river_nodata", -9999.0) or -9999.0)
        dem = np.empty((bed_ds.height, bed_ds.width), dtype=np.float32)
        if (dem_ds.height, dem_ds.width) == (bed_ds.height, bed_ds.width) and dem_ds.transform == bed_ds.transform and dem_ds.crs == bed_ds.crs:
            dem = dem_ds.read(1).astype(np.float32)
            dem_nodata = dem_ds.nodata
        else:
            from rasterio.warp import reproject, Resampling
            reproject(
                source=rasterio.band(dem_ds, 1), destination=dem,
                src_transform=dem_ds.transform, src_crs=dem_ds.crs,
                dst_transform=bed_ds.transform, dst_crs=bed_ds.crs,
                resampling=Resampling.bilinear,
                src_nodata=dem_ds.nodata, dst_nodata=float(bed_nodata),
            )
            dem_nodata = float(bed_nodata)
        bed_valid = np.isfinite(bed) & (~np.isclose(bed, float(bed_nodata)))
        if dem_nodata is not None:
            dem = np.where(np.isclose(dem, float(dem_nodata)), np.nan, dem)
        depth = np.where(bed_valid & np.isfinite(dem), bed - dem, np.nan).astype(np.float32)
        bed_profile = bed_ds.profile.copy()
        bed_profile.update(dtype="float32", nodata=float(bed_nodata), compress="deflate")
        if primary_bed.exists():
            primary_bed.unlink()
        with rasterio.open(primary_bed, "w", **bed_profile) as dst:
            dst.write(np.where(np.isfinite(bed), bed, np.float32(bed_nodata)).astype(np.float32), 1)
        if primary_depth.exists():
            primary_depth.unlink()
        with rasterio.open(primary_depth, "w", **bed_profile) as dst:
            dst.write(np.where(np.isfinite(depth), depth, np.float32(bed_nodata)).astype(np.float32), 1)
    logger.info("[RIVER][FRAME] Primary river products now sourced from channel_surface: %s", primary_bed)
    return {"primary_bed": str(primary_bed), "primary_depth": str(primary_depth)}


def write_guidance_artifacts_with_reporting(
    *,
    writer: Callable[..., Dict[str, Optional[str]]],
    cfg,
    out_bed: Path,
    out_depth: Path,
    channel_mask_tif: Optional[Path],
    river_dir: Path,
    report: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Write river guidance artifacts and update the run report.

    Keeps bathy_main orchestration thinner while preserving the existing
    numerical guidance implementation.
    """
    try:
        guidance_outputs = writer(
            cfg,
            bed_tif=Path(out_bed),
            depth_tif=Path(out_depth),
            channel_mask_tif=(Path(channel_mask_tif) if channel_mask_tif is not None else None),
            river_dir=river_dir,
            report=report,
        )
        report.setdefault("river", {}).setdefault("outputs", {}).update(
            {k: v for k, v in guidance_outputs.items() if v}
        )
        river_outputs = report.setdefault("river", {}).setdefault("outputs", {})
        scaffold_products = report.setdefault("river", {}).get("scaffold_products", {}) if isinstance(report.setdefault("river", {}).get("scaffold_products", {}), dict) else {}
        network_edges_path = scaffold_products.get("graph_edges_gpkg") or scaffold_products.get("mainstem_edges_gpkg") or river_outputs.get("retained_network")
        def _declared_existing_output(*keys: str) -> Optional[str]:
            for key in keys:
                raw = river_outputs.get(key)
                if isinstance(raw, Path):
                    cand = raw
                elif isinstance(raw, str) and raw.strip():
                    cand = Path(raw)
                else:
                    cand = None
                if cand is not None and cand.exists():
                    return str(cand)
            return None

        wse_elevation_path = _declared_existing_output("wse", "wse_m", "water_surface", "water_surface_elevation")
        wse_uncertainty_path = _declared_existing_output("wse_uncertainty", "water_surface_uncertainty")
        wse_influence_path = _declared_existing_output("wse_influence", "water_surface_influence")
        retained_network_meta = _resolve_retained_network_summary(report)
        component_expectation = resolve_centerline_component_expectation(
            retained_network_meta,
            station_contract_path=retained_network_meta.get("centerline_station_contract_path") if isinstance(retained_network_meta, dict) else None,
        )
        centerline_component_contract = _load_and_validate_centerline_component_contract(
            river_outputs.get("centerline_points"),
            retained_network_meta=retained_network_meta,
            expected_component_count=int(component_expectation.get("expected_component_count", 0)),
            expected_component_source=str(component_expectation.get("component_id_source", "unknown")),
            logger=logger,
        )
        report.setdefault("river", {}).setdefault("execution_receipts", {})["centerline_component_contract"] = centerline_component_contract
        profile_outputs = build_and_write_longitudinal_profile(
            river_dir=river_dir,
            centerline_points_path=river_outputs.get("centerline_points"),
            centerline_elevation_path=river_outputs.get("centerline_elevation"),
            centerline_influence_path=river_outputs.get("centerline_influence"),
            centerline_stationing_path=river_outputs.get("centerline_stationing"),
            xs_support_elevation_path=river_outputs.get("xs_support_elevation"),
            xs_support_weight_path=river_outputs.get("xs_support_weight"),
            bank_elevation_path=river_outputs.get("bank_elevation_xs"),
            bank_points_path=river_outputs.get("bank_points"),
            bank_influence_path=river_outputs.get("bank_influence"),
            bank_graph_confidence_path=river_outputs.get("bank_graph_confidence"),
            bank_continuity_weight_path=river_outputs.get("bank_continuity_weight"),
            bank_confluence_damping_path=river_outputs.get("bank_confluence_damping"),
            bank_estuary_side_decay_path=river_outputs.get("bank_estuary_side_decay"),
            authoritative_support_depth_path=river_outputs.get("authoritative_support_depth"),
            authoritative_bed_elevation_path=(getattr(cfg, "river_dem", None) or getattr(cfg, "authoritative_base", None)),
            authoritative_support_mask_path=river_outputs.get("authoritative_support"),
            authoritative_support_points_path=getattr(cfg, "river_authoritative_soundings", None),
            authoritative_role_code_path=getattr(cfg, "river_authoritative_role_code_raster", None),
            authoritative_role_confidence_path=getattr(cfg, "river_authoritative_role_confidence_raster", None),
            authoritative_distance_to_bank_path=getattr(cfg, "river_authoritative_distance_to_bank_raster", None),
            authoritative_normalized_channel_position_path=getattr(cfg, "river_authoritative_normalized_channel_position_raster", None),
            wse_elevation_path=wse_elevation_path,
            wse_influence_path=wse_influence_path,
            wse_uncertainty_path=wse_uncertainty_path,
            corridor_mask_path=river_outputs.get("corridor_mask"),
            network_edges_path=network_edges_path,
        )
        river_outputs.update({k: v for k, v in profile_outputs.items() if v})
        if river_outputs.get("longitudinal_profile"):
            reach_outputs = build_and_write_reach_attributes(
                river_dir=river_dir,
                longitudinal_profile_path=river_outputs.get("longitudinal_profile"),
                longitudinal_profile_summary_path=river_outputs.get("longitudinal_profile_summary"),
                hydraulic_backbone_edges_path=river_outputs.get("hydraulic_backbone_edges"),
            )
            river_outputs.update({k: v for k, v in reach_outputs.items() if v})
        exec_receipts = report.setdefault("river", {}).setdefault("execution_receipts", {})
        exec_receipts["legacy_xs_inputs_permitted"] = False
        exec_receipts["legacy_xs_inputs_blocked"] = True
        exec_receipts["legacy_xs_profile_rebuild_removed"] = True
        exec_receipts["absolute_bed_fallback_removed"] = True
        exec_receipts["wse_on_disk_discovery_removed"] = True
        exec_receipts["legacy_xs_inputs_used"] = False
        exec_receipts["legacy_xs_inputs_detected"] = []
        long_profile_path = river_dir / "river_longitudinal_profile_contract.json"
        write_river_longitudinal_profile_contract(
            long_profile_path,
            outputs={
                "centerline_stationing": river_outputs.get("centerline_stationing"),
                "centerline_elevation": river_outputs.get("centerline_elevation"),
                "xs_support_elevation": river_outputs.get("xs_support_elevation"),
                "authoritative_support_depth": river_outputs.get("authoritative_support_depth"),
                "bank_elevation_xs": river_outputs.get("bank_elevation_xs"),
                "bank_longitudinal_fit_points": river_outputs.get("bank_longitudinal_fit_points"),
                "bank_longitudinal_fit_summary": river_outputs.get("bank_longitudinal_fit_summary"),
                "left_bank_fit_elevation": river_outputs.get("left_bank_fit_elevation"),
                "right_bank_fit_elevation": river_outputs.get("right_bank_fit_elevation"),
                "bank_pair_fit_elevation": river_outputs.get("bank_pair_fit_elevation"),
                "authoritative_bed_anchor_curve": river_outputs.get("authoritative_bed_anchor_curve"),
                "authoritative_bed_anchor_curve_summary": river_outputs.get("authoritative_bed_anchor_curve_summary"),
                "longitudinal_profile": river_outputs.get("longitudinal_profile"),
                "longitudinal_profile_coverage": river_outputs.get("longitudinal_profile_coverage"),
                "longitudinal_profile_elevation": river_outputs.get("longitudinal_profile_elevation"),
                "longitudinal_profile_uncertainty": river_outputs.get("longitudinal_profile_uncertainty"),
                "active_core_support_elevation": river_outputs.get("active_core_support_elevation"),
                "active_core_support_uncertainty": river_outputs.get("active_core_support_uncertainty"),
                "active_core_support_influence": river_outputs.get("active_core_support_influence"),
            },
        )
        river_outputs["longitudinal_profile_contract"] = str(long_profile_path)
        export_legacy_mixed_bed = False
        exec_receipts["legacy_mixed_bed_exported"] = False
        channel_frame_outputs = build_channel_frame_products(
            river_dir=river_dir,
            centerline_points_path=river_outputs.get("centerline_points"),
            xs_support_points_path=river_outputs.get("xs_support_points"),
            bank_elevation_path=river_outputs.get("bank_elevation_xs"),
            bank_influence_path=river_outputs.get("bank_influence"),
            left_bank_fit_elevation_path=river_outputs.get("left_bank_fit_elevation"),
            right_bank_fit_elevation_path=river_outputs.get("right_bank_fit_elevation"),
            bank_pair_fit_elevation_path=river_outputs.get("bank_pair_fit_elevation"),
            xs_support_elevation_path=river_outputs.get("xs_support_elevation"),
            xs_support_weight_path=river_outputs.get("xs_support_weight"),
            centerline_elevation_path=river_outputs.get("centerline_elevation"),
            centerline_influence_path=river_outputs.get("centerline_influence"),
            longitudinal_profile_elevation_path=river_outputs.get("longitudinal_profile_elevation"),
            active_core_support_elevation_path=river_outputs.get("active_core_support_elevation"),
            disable_xs_influence=bool(getattr(cfg, "river_disable_xs_influence", False)),
            authoritative_support_mask_path=river_outputs.get("authoritative_support"),
            authoritative_support_depth_path=river_outputs.get("authoritative_support_depth"),
            authoritative_bed_elevation_path=(getattr(cfg, "river_dem", None) or getattr(cfg, "authoritative_base", None)),
            authoritative_role_code_path=getattr(cfg, "river_authoritative_role_code_raster", None),
            authoritative_role_confidence_path=getattr(cfg, "river_authoritative_role_confidence_raster", None),
            authoritative_distance_to_bank_path=getattr(cfg, "river_authoritative_distance_to_bank_raster", None),
            authoritative_normalized_channel_position_path=getattr(cfg, "river_authoritative_normalized_channel_position_raster", None),
            export_legacy_mixed_bed=export_legacy_mixed_bed,
            logger=logger,
        )
        river_outputs.update({k: v for k, v in channel_frame_outputs.items() if v})
        scaffold_xs_path = None
        scaffold_outputs = build_channel_scaffold_products(
            river_dir=river_dir,
            channel_frame_points_path=river_outputs.get("channel_frame_points"),
            xs_bathy_gpkg_path=scaffold_xs_path,
            station_targets_path=river_outputs.get('station_targets'),
            authoritative_support_mask_path=river_outputs.get("authoritative_support"),
            authoritative_support_depth_path=river_outputs.get("authoritative_support_depth"),
            allow_absolute_bed_fallback=False,
            disable_xs_influence=bool(getattr(cfg, "river_disable_xs_influence", False)),
            logger=logger,
        )
        river_outputs.update({k: v for k, v in scaffold_outputs.items() if v})
        channel_surface_outputs = build_channel_surface_products(
            river_dir=river_dir,
            channel_scaffold_nodes_path=river_outputs.get("channel_scaffold_nodes"),
            corridor_mask_path=river_outputs.get("corridor_mask"),
            authoritative_support_mask_path=river_outputs.get("authoritative_support"),
            authoritative_support_depth_path=river_outputs.get("authoritative_support_depth"),
            authoritative_role_code_path=getattr(cfg, "river_authoritative_role_code_raster", None),
            longitudinal_profile_path=river_outputs.get("longitudinal_profile"),
            reach_attributes_path=river_outputs.get("reach_attributes"),
            disable_xs_influence=bool(getattr(cfg, "river_disable_xs_influence", False)),
            logger=logger,
        )
        river_outputs.update({k: v for k, v in channel_surface_outputs.items() if v})
        runtime_diag_outputs = build_river_runtime_diagnostics(
            river_dir=river_dir,
            river_outputs=river_outputs,
            report=report,
            cfg=cfg,
            logger=logger,
        )
        river_outputs.update({k: v for k, v in runtime_diag_outputs.items() if v})
        report.setdefault("river", {}).setdefault("diagnostics", {})["runtime"] = runtime_diag_outputs.get("runtime_diagnostics")
        exec_receipts["xs_influence_disabled"] = bool(getattr(cfg, "river_disable_xs_influence", False))
        report.setdefault("river", {}).setdefault("execution_receipts", {}).update(exec_receipts)
        exec_receipts["xs_support_status"] = _derive_xs_support_status(report, river_outputs)
        xs_participation_contract_path = _write_xs_participation_contract(
            river_dir=river_dir,
            xs_status=exec_receipts["xs_support_status"],
        )
        river_outputs["xs_participation_contract"] = str(xs_participation_contract_path)
        report.setdefault("river", {}).setdefault("outputs", {})["xs_participation_contract"] = str(xs_participation_contract_path)
        primary_products = _write_channel_surface_primary_products(
            cfg=cfg,
            river_dir=river_dir,
            river_outputs=river_outputs,
            logger=logger,
        )
        if primary_products:
            report.setdefault("river", {}).setdefault("outputs", {})["bottom_elevation_internal_helper"] = str(out_bed)
            report.setdefault("river", {}).setdefault("outputs", {})["depth_terrain_internal_helper"] = str(out_depth)
            report.setdefault("river", {}).setdefault("outputs", {})["bottom_elevation"] = primary_products["primary_bed"]
            report.setdefault("river", {}).setdefault("outputs", {})["depth_terrain"] = primary_products["primary_depth"]
            river_outputs["primary_bed_from_channel_surface"] = primary_products["primary_bed"]
            river_outputs["primary_depth_from_channel_surface"] = primary_products["primary_depth"]
            report.setdefault("river", {}).setdefault("guidance", {})["bed_surface_is_internal_helper"] = True
            report.setdefault("river", {}).setdefault("guidance", {})["channel_surface_is_primary_river_surface"] = True
        scaffold_manifest = report.setdefault("river", {}).get("scaffold_manifest")
        if scaffold_manifest and not river_outputs.get("scaffold_manifest"):
            river_outputs["scaffold_manifest"] = str(scaffold_manifest)
        manifest_path = write_river_guidance_manifest(
            out_root=Path(cfg.out_dir),
            river_dir=river_dir,
            report=report,
            logger=logger,
        )
        report.setdefault("river", {}).setdefault("outputs", {})["guidance_manifest"] = str(manifest_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        report.setdefault("river", {}).setdefault("guidance", {})["artifact_write_error"] = str(e)
        logger.error("Failed to write scientifically required river guidance artifacts: %s", e)
        raise


def apply_guidance_controls_with_reporting(
    *,
    apply_fn: Callable[..., None],
    out_depth: Path,
    river_fuse_path: Optional[Path],
    sdb_fuse_path: Optional[Path],
    out_prov: Optional[Path],
    report: Dict[str, Any],
    estuary_max_weight: float,
    logger: logging.Logger,
) -> None:
    """Apply river guidance controls to the fused raster and record failures explicitly."""
    try:
        river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
        apply_fn(
            Path(out_depth),
            river_path=Path(river_fuse_path) if river_fuse_path else None,
            sdb_path=Path(sdb_fuse_path) if sdb_fuse_path else None,
            provenance_path=Path(out_prov) if out_prov else None,
            river_outputs=river_outputs,
            report=report,
            estuary_max_weight=float(estuary_max_weight),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        report.setdefault("fusion", {}).setdefault("guidance_controls", {})["river_guidance_applied"] = False
        report["fusion"]["guidance_controls"]["error"] = str(e)
        logger.debug("river guidance fusion controls failed; keeping base fused output", exc_info=True)
