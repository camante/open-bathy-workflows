from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import logging
log = logging.getLogger(__name__)



DEFAULT_TIE_BREAK_POLICY: List[str] = [
    "outlet_connected wins",
    "larger drainage proxy wins",
    "larger upstream accumulated length wins",
    "higher stream order wins",
    "longer total system length wins",
    "stable component id ordering wins",
]


@dataclass(frozen=True)
class HydrologicSolveDomainArtifacts:
    hydrologic_manifest: Dict[str, Any]
    outlet_anchors: gpd.GeoDataFrame
    estuary_control_points: gpd.GeoDataFrame
    major_system_network: gpd.GeoDataFrame


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _safe_numeric(series: Optional[pd.Series], default: float = 0.0) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _component_rank_payload(edges: gpd.GeoDataFrame, component_id: str) -> Dict[str, float]:
    comp = edges.loc[edges["component_id"].astype(str) == str(component_id)].copy()
    if comp.empty:
        return {
            "component_id": str(component_id),
            "edge_count": 0.0,
            "total_length_m": 0.0,
            "max_drainage": 0.0,
            "max_arbolate": 0.0,
            "max_order": 0.0,
            "root_station_m": float("inf"),
            "score": float("-inf"),
        }
    length_col = _first_present(comp.columns, ["length_m", "lengthm", "lengthkm", "length_km"])
    drainage_col = _first_present(comp.columns, ["totdasqkm", "divdasqkm", "areasqkm", "drainarea", "drainage_a", "upa_km2", "drain_area_km2"])
    arbolate_col = _first_present(comp.columns, ["arbolatesu", "arbsumkm", "arbolate", "arbolate_km"])
    order_col = _first_present(comp.columns, ["streamorde", "streamorder", "streamord", "strahler"])
    station_col = _first_present(comp.columns, ["s_m_min", "s_m_from", "station_m"])

    length_vals = _safe_numeric(comp[length_col], default=0.0) if length_col else np.zeros(len(comp), dtype=float)
    if length_col and str(length_col).lower().endswith("km"):
        length_vals = length_vals * 1000.0
    drainage_vals = _safe_numeric(comp[drainage_col], default=0.0) if drainage_col else np.zeros(len(comp), dtype=float)
    arbolate_vals = _safe_numeric(comp[arbolate_col], default=0.0) if arbolate_col else np.zeros(len(comp), dtype=float)
    order_vals = _safe_numeric(comp[order_col], default=0.0) if order_col else np.zeros(len(comp), dtype=float)
    station_vals = _safe_numeric(comp[station_col], default=np.nan) if station_col else np.full(len(comp), np.nan)
    finite_station = station_vals[np.isfinite(station_vals)]
    root_station = float(np.nanmin(finite_station)) if finite_station.size else float("inf")

    total_length_m = float(np.nansum(np.where(np.isfinite(length_vals), length_vals, 0.0)))
    max_drainage = float(np.nanmax(drainage_vals)) if drainage_vals.size else 0.0
    max_arbolate = float(np.nanmax(arbolate_vals)) if arbolate_vals.size else 0.0
    max_order = float(np.nanmax(order_vals)) if order_vals.size else 0.0
    score = (
        1000.0 * max_drainage
        + 250.0 * max_arbolate
        + 50.0 * max_order
        + total_length_m
        - (0.001 * root_station if np.isfinite(root_station) else 0.0)
    )
    return {
        "component_id": str(component_id),
        "edge_count": float(len(comp)),
        "total_length_m": total_length_m,
        "max_drainage": max_drainage,
        "max_arbolate": max_arbolate,
        "max_order": max_order,
        "root_station_m": root_station,
        "score": score,
    }


def select_major_system_component(edges: gpd.GeoDataFrame) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if edges is None or edges.empty or "component_id" not in edges.columns:
        return None, []
    rows = [_component_rank_payload(edges, str(cid)) for cid in sorted(edges["component_id"].dropna().astype(str).unique())]
    rows = sorted(
        rows,
        key=lambda row: (
            row["score"],
            row["max_drainage"],
            row["max_arbolate"],
            row["max_order"],
            row["total_length_m"],
            -float(row["root_station_m"]) if np.isfinite(row["root_station_m"]) else float("-inf"),
            row["component_id"],
        ),
        reverse=True,
    )
    major_component = str(rows[0]["component_id"]) if rows else None
    return major_component, rows


def _node_geometries(nodes: Optional[gpd.GeoDataFrame]) -> Dict[int, Point]:
    if nodes is None or nodes.empty or "node_id" not in nodes.columns:
        return {}
    out: Dict[int, Point] = {}
    for _, row in nodes.iterrows():
        try:
            out[int(row["node_id"])] = row.geometry
        except Exception:
            log.debug("_node_geometries: suppressed exception", exc_info=True)
            continue
    return out


def build_outlet_anchors(
    *,
    nodes: Optional[gpd.GeoDataFrame],
    edges: gpd.GeoDataFrame,
    major_component_id: Optional[str],
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crs = edges.crs
    if edges is None or edges.empty:
        empty = gpd.GeoDataFrame(columns=["component_id", "anchor_role", "control_role", "geometry"], crs=crs)
        return empty.copy(), empty.copy()

    node_geom = _node_geometries(nodes)
    records: List[Dict[str, Any]] = []
    estuary_records: List[Dict[str, Any]] = []
    if "component_id" not in edges.columns:
        empty = gpd.GeoDataFrame(columns=["component_id", "anchor_role", "control_role", "geometry"], crs=crs)
        return empty.copy(), empty.copy()

    station_col = _first_present(edges.columns, ["s_m_min", "s_m_from", "station_m"])
    root_col = _first_present(edges.columns, ["root_node"])
    feature_id_col = _first_present(edges.columns, ["river_id", "edge_id", "reachcode", "comid", "nhdplusid"])
    for component_id in sorted(edges["component_id"].dropna().astype(str).unique()):
        comp = edges.loc[edges["component_id"].astype(str) == component_id].copy()
        if comp.empty:
            continue
        root_node = None
        if root_col and root_col in comp.columns and comp[root_col].notna().any():
            try:
                root_node = int(pd.to_numeric(comp[root_col], errors="coerce").dropna().iloc[0])
            except Exception:
                log.debug("build_outlet_anchors: suppressed exception", exc_info=True)
                root_node = None
        if root_node is None:
            station_vals = pd.to_numeric(comp[station_col], errors="coerce") if station_col else pd.Series([np.nan] * len(comp), index=comp.index)
            idx = station_vals.idxmin() if station_vals.notna().any() else comp.index[0]
            root_node = int(comp.loc[idx, "from_node"]) if "from_node" in comp.columns else None
        geom = node_geom.get(root_node) if root_node is not None else None
        if geom is None or geom.is_empty:
            geom = comp.geometry.iloc[0].boundary.geoms[0] if hasattr(comp.geometry.iloc[0].boundary, "geoms") else Point(comp.geometry.iloc[0].coords[0])
        representative_feature_id = str(comp[feature_id_col].iloc[0]) if feature_id_col and feature_id_col in comp.columns else None
        is_major = major_component_id is not None and str(component_id) == str(major_component_id)
        anchor_role = "major_system_outlet_anchor" if is_major else "component_outlet_anchor"
        records.append({
            "component_id": str(component_id),
            "is_major_system": bool(is_major),
            "anchor_role": anchor_role,
            "feature_id": representative_feature_id,
            "selection_method": "component_root_node",
            "geometry": geom,
        })
        estuary_records.append({
            "component_id": str(component_id),
            "is_major_system": bool(is_major),
            "control_role": "river_to_estuary_handoff_proxy" if is_major else "component_handoff_proxy",
            "feature_id": representative_feature_id,
            "selection_method": "component_root_node_proxy",
            "geometry": geom,
        })
    anchors = gpd.GeoDataFrame(records, crs=crs)
    estuary = gpd.GeoDataFrame(estuary_records, crs=crs)
    return anchors, estuary


def build_hydrologic_solve_domain_artifacts(
    *,
    solve_network: gpd.GeoDataFrame,
    nodes: Optional[gpd.GeoDataFrame],
    export_aoi: Optional[str],
    solve_aoi: Optional[str],
    scaffold_aoi: Optional[str],
    solve_halo_km: float,
    trusted_halo_m: float,
    solve_domain_role: str,
    export_domain_role: str,
    scaffold_domain_role: str,
    solve_domain_rationale: str,
    domain_type: str = "coastal_outlet",
) -> HydrologicSolveDomainArtifacts:
    major_component_id, component_ranks = select_major_system_component(solve_network)
    if major_component_id is not None and "component_id" in solve_network.columns:
        major_system_network = solve_network.loc[solve_network["component_id"].astype(str) == str(major_component_id)].copy()
    else:
        major_system_network = solve_network.copy()
    outlet_anchors, estuary_control_points = build_outlet_anchors(
        nodes=nodes,
        edges=solve_network,
        major_component_id=major_component_id,
    )
    hydrologic_manifest = {
        "product_type": "hydrologic_solve_domain",
        "domain_type": str(domain_type),
        "export_aoi": str(export_aoi) if export_aoi else None,
        "solve_aoi": str(solve_aoi) if solve_aoi else None,
        "scaffold_aoi": str(scaffold_aoi) if scaffold_aoi else None,
        "solve_halo_km": float(solve_halo_km),
        "trusted_halo_m": float(trusted_halo_m),
        "solve_domain_role": str(solve_domain_role),
        "export_domain_role": str(export_domain_role),
        "scaffold_domain_role": str(scaffold_domain_role),
        "solve_domain_rationale": str(solve_domain_rationale),
        "major_system_id": str(major_component_id) if major_component_id is not None else None,
        "major_system_selection_method": "component_ranking",
        "upstream_inclusion_rule": "retain the outlet-connected broader solve-domain network prepared upstream; downstream export is clipped later",
        "outlet_anchor_layer_name": "outlet_anchors",
        "estuary_control_layer_name": "estuary_control_points",
        "major_system_layer_name": "major_system_network",
        "tie_break_policy": list(DEFAULT_TIE_BREAK_POLICY),
        "component_rankings": component_ranks,
        "outlet_anchor_count": int(len(outlet_anchors)),
        "estuary_control_count": int(len(estuary_control_points)),
        "major_system_edge_count": int(len(major_system_network)),
        "major_system_total_length_m": float(pd.to_numeric(major_system_network.get("length_m"), errors="coerce").fillna(0.0).sum()) if len(major_system_network) else 0.0,
    }
    return HydrologicSolveDomainArtifacts(
        hydrologic_manifest=hydrologic_manifest,
        outlet_anchors=outlet_anchors,
        estuary_control_points=estuary_control_points,
        major_system_network=major_system_network,
    )


def write_hydrologic_solve_domain_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out
