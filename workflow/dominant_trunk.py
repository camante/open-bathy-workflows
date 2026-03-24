from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopologyColumns:
    upstream: Optional[str]
    downstream: Optional[str]
    length: Optional[str]
    order: Optional[str]
    drainage: Optional[str]
    arbolate: Optional[str]
    feature_id: Optional[str]
    method: str


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def infer_topology_columns(gdf: gpd.GeoDataFrame) -> TopologyColumns:
    cols = list(gdf.columns)
    feature_id = _find_col(cols, ["nhdplusid", "comid", "permanent_", "permanent_identifier", "reachcode", "river_id", "edge_id", "id"])
    length = _find_col(cols, ["lengthkm", "length_km", "len_km", "lengthm", "length_m"])
    order = _find_col(cols, ["streamorde", "streamorder", "streamord", "strahler"])
    drainage = _find_col(cols, ["totdasqkm", "divdasqkm", "areasqkm", "drainarea", "drainage_a", "upa_km2"])
    arbolate = _find_col(cols, ["arbolatesu", "arbsumkm", "arbolate", "arbolate_km"])

    fromnode = _find_col(cols, ["fromnode", "from_node"])
    tonode = _find_col(cols, ["tonode", "to_node"])
    if fromnode and tonode:
        return TopologyColumns(fromnode, tonode, length, order, drainage, arbolate, feature_id, "fromnode_tonode")

    # Fall back to directed endpoints from geometry.
    return TopologyColumns(None, None, length, order, drainage, arbolate, feature_id, "geometry_endpoints")


def _safe_numeric(series: Optional[pd.Series], default: float) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    vals = values.copy().astype(float)
    vals[~finite] = np.nan
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(values, dtype=float)
        out[finite] = 1.0
        return out
    out = (vals - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def _line_endpoints(geom: Any) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return ((np.nan, np.nan), (np.nan, np.nan))
    try:
        if geom.geom_type == "MultiLineString":
            line = max(list(geom.geoms), key=lambda g: g.length)
        else:
            line = geom
        coords = list(line.coords)
    except Exception:
        LOG.debug("_line_endpoints: suppressed exception", exc_info=True)
        return ((np.nan, np.nan), (np.nan, np.nan))
    if len(coords) < 2:
        return ((np.nan, np.nan), (np.nan, np.nan))
    return (tuple(coords[0]), tuple(coords[-1]))


def _endpoint_nodes(gdf: gpd.GeoDataFrame, tol: float) -> Tuple[List[str], List[str]]:
    upstream_nodes: List[str] = []
    downstream_nodes: List[str] = []
    scale = max(float(tol), 1e-9)
    for geom in gdf.geometry:
        start, end = _line_endpoints(geom)
        if np.isnan(start[0]) or np.isnan(end[0]):
            upstream_nodes.append("nan_u")
            downstream_nodes.append("nan_d")
            continue
        su = f"{round(start[0] / scale):.0f}_{round(start[1] / scale):.0f}"
        sd = f"{round(end[0] / scale):.0f}_{round(end[1] / scale):.0f}"
        upstream_nodes.append(su)
        downstream_nodes.append(sd)
    return upstream_nodes, downstream_nodes


def _build_edges(
    rivers: gpd.GeoDataFrame,
    topo: TopologyColumns,
    *,
    endpoint_tolerance: float,
) -> List[Dict[str, Any]]:
    gdf = rivers.reset_index(drop=False).rename(columns={"index": "_orig_index"}).copy()
    if topo.upstream and topo.downstream:
        up_nodes = gdf[topo.upstream].astype(str).tolist()
        down_nodes = gdf[topo.downstream].astype(str).tolist()
    else:
        up_nodes, down_nodes = _endpoint_nodes(gdf, tol=endpoint_tolerance)

    if topo.length:
        length_vals = _safe_numeric(gdf[topo.length], default=np.nan)
        if topo.length.lower().endswith("km"):
            length_m = np.where(np.isfinite(length_vals), length_vals * 1000.0, np.nan)
        else:
            length_m = length_vals
    else:
        length_m = np.array([float(geom.length) if geom is not None and not geom.is_empty else np.nan for geom in gdf.geometry], dtype=float)

    order_vals = _safe_numeric(gdf[topo.order], default=0.0) if topo.order else np.zeros(len(gdf), dtype=float)
    drainage_vals = _safe_numeric(gdf[topo.drainage], default=0.0) if topo.drainage else np.zeros(len(gdf), dtype=float)
    arbolate_vals = _safe_numeric(gdf[topo.arbolate], default=0.0) if topo.arbolate else np.zeros(len(gdf), dtype=float)

    length_norm = _norm(np.where(np.isfinite(length_m), length_m, 0.0))
    order_norm = _norm(order_vals)
    drainage_norm = _norm(drainage_vals)
    arbolate_norm = _norm(arbolate_vals)

    edges: List[Dict[str, Any]] = []
    for i, row in gdf.iterrows():
        fid = str(row[topo.feature_id]) if topo.feature_id and topo.feature_id in row and pd.notnull(row[topo.feature_id]) else str(int(row["_orig_index"]))
        score = (
            10.0 * float(length_norm[i])
            + 6.0 * float(order_norm[i])
            + 8.0 * float(drainage_norm[i])
            + 4.0 * float(arbolate_norm[i])
            + 0.5 * np.log1p(max(float(length_m[i]) if np.isfinite(length_m[i]) else 0.0, 0.0))
        )
        edges.append(
            {
                "edge_id": i,
                "feature_id": fid,
                "orig_index": int(row["_orig_index"]),
                "u": up_nodes[i],
                "v": down_nodes[i],
                "geom": row.geometry,
                "length_m": float(length_m[i]) if np.isfinite(length_m[i]) else 0.0,
                "order": float(order_vals[i]) if np.isfinite(order_vals[i]) else 0.0,
                "drainage": float(drainage_vals[i]) if np.isfinite(drainage_vals[i]) else 0.0,
                "arbolate": float(arbolate_vals[i]) if np.isfinite(arbolate_vals[i]) else 0.0,
                "score": float(score),
            }
        )
    return edges


def _geometry_downstream_touches_ocean(geom: Any, ocean_mask: np.ndarray, transform: Any) -> bool:
    if geom is None or geom.is_empty:
        return False
    try:
        _, end = _line_endpoints(geom)
        r, c = rowcol(transform, end[0], end[1])
        if 0 <= r < ocean_mask.shape[0] and 0 <= c < ocean_mask.shape[1] and bool(ocean_mask[r, c]):
            return True
    except Exception:
        LOG.debug("Ocean-touch check failed for geometry", exc_info=True)
    return False


def select_dominant_trunks(
    rivers: gpd.GeoDataFrame,
    *,
    ocean_mask: Optional[np.ndarray] = None,
    ocean_transform: Any = None,
    endpoint_tolerance: float = 5.0,
    logger: Optional[logging.Logger] = None,
    include_all_components: bool = False,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    log = logger or LOG
    gdf = rivers[rivers.geometry.notnull() & ~rivers.geometry.is_empty].copy()
    if gdf.empty:
        return gdf, {"method": "dominant_trunk", "status": "empty"}

    topo = infer_topology_columns(gdf)
    edges = _build_edges(gdf, topo, endpoint_tolerance=endpoint_tolerance)
    n_edges = len(edges)
    out_by_node: Dict[str, List[int]] = {}
    in_by_node: Dict[str, List[int]] = {}
    undirected: Dict[str, set] = {}
    for e in edges:
        out_by_node.setdefault(e["u"], []).append(e["edge_id"])
        in_by_node.setdefault(e["v"], []).append(e["edge_id"])
        undirected.setdefault(e["u"], set()).add(e["v"])
        undirected.setdefault(e["v"], set()).add(e["u"])

    edge_by_id = {e["edge_id"]: e for e in edges}
    outlet_edges: set = set()
    for e in edges:
        if ocean_mask is not None and ocean_transform is not None and _geometry_downstream_touches_ocean(e["geom"], ocean_mask, ocean_transform):
            outlet_edges.add(e["edge_id"])
        elif len(out_by_node.get(e["v"], [])) == 0:
            outlet_edges.add(e["edge_id"])

    component_by_node: Dict[str, int] = {}
    comp_id = 0
    for node in undirected:
        if node in component_by_node:
            continue
        stack = [node]
        component_by_node[node] = comp_id
        while stack:
            cur = stack.pop()
            for nbr in undirected.get(cur, ()):
                if nbr not in component_by_node:
                    component_by_node[nbr] = comp_id
                    stack.append(nbr)
        comp_id += 1

    edges_by_component: Dict[int, List[int]] = {}
    for e in edges:
        cid = component_by_node.get(e["u"], component_by_node.get(e["v"], -1))
        edges_by_component.setdefault(cid, []).append(e["edge_id"])

    best_cache: Dict[int, Tuple[float, List[int]]] = {}
    visiting: set = set()

    def best_path(edge_id: int) -> Tuple[float, List[int]]:
        if edge_id in best_cache:
            return best_cache[edge_id]
        if edge_id in visiting:
            # deterministic cycle break
            return edge_by_id[edge_id]["score"], [edge_id]
        visiting.add(edge_id)
        e = edge_by_id[edge_id]
        downstream_candidates = out_by_node.get(e["v"], [])
        best_score = e["score"]
        best_edges = [edge_id]
        if downstream_candidates:
            downstream_ranked = sorted(
                downstream_candidates,
                key=lambda cid: (
                    edge_by_id[cid]["drainage"],
                    edge_by_id[cid]["order"],
                    edge_by_id[cid]["length_m"],
                    edge_by_id[cid]["feature_id"],
                ),
                reverse=True,
            )
            child_best_score = None
            child_best_edges: List[int] = []
            for child_id in downstream_ranked:
                cscore, cedges = best_path(child_id)
                if child_best_score is None or cscore > child_best_score:
                    child_best_score = cscore
                    child_best_edges = cedges
            if child_best_score is not None:
                best_score = e["score"] + child_best_score
                best_edges = [edge_id] + child_best_edges
        visiting.remove(edge_id)
        best_cache[edge_id] = (best_score, best_edges)
        return best_cache[edge_id]

    selected_ids: List[int] = []
    component_summaries: List[Dict[str, Any]] = []
    component_paths: List[Dict[str, Any]] = []
    for cid, comp_edge_ids in sorted(edges_by_component.items()):
        outlet_candidates = sorted([eid for eid in comp_edge_ids if eid in outlet_edges])
        if not outlet_candidates:
            # choose any terminal edge in component
            outlet_candidates = sorted([eid for eid in comp_edge_ids if len(out_by_node.get(edge_by_id[eid]["v"], [])) == 0])
        if not outlet_candidates:
            outlet_candidates = [max(comp_edge_ids, key=lambda eid: (edge_by_id[eid]["drainage"], edge_by_id[eid]["length_m"], edge_by_id[eid]["feature_id"]))]
        upstream_candidates = sorted(set(comp_edge_ids))
        best_comp_score = None
        best_comp_edges: List[int] = []
        for eid in upstream_candidates:
            score, path_edges = best_path(eid)
            if path_edges and path_edges[-1] not in outlet_candidates:
                continue
            if best_comp_score is None or score > best_comp_score:
                best_comp_score = score
                best_comp_edges = path_edges
        if not best_comp_edges:
            # fall back to best path among outlet candidates walking upstream implicitly by choosing max path score
            eid = max(comp_edge_ids, key=lambda x: best_path(x)[0])
            best_comp_score, best_comp_edges = best_path(eid)
        selected_length = float(sum(edge_by_id[eid]["length_m"] for eid in best_comp_edges))
        comp_drainage = max((edge_by_id[eid]["drainage"] for eid in comp_edge_ids), default=0.0)
        comp_order = max((edge_by_id[eid]["order"] for eid in comp_edge_ids), default=0.0)
        summary = {
            "component_id": int(cid),
            "edge_count": int(len(comp_edge_ids)),
            "selected_edge_count": int(len(best_comp_edges)),
            "selected_length_m": selected_length,
            "outlet_edge_id": int(best_comp_edges[-1]) if best_comp_edges else None,
            "max_drainage": float(comp_drainage),
            "max_order": float(comp_order),
            "path_score": float(best_comp_score) if best_comp_score is not None else float("-inf"),
        }
        component_summaries.append(summary)
        component_paths.append({"summary": summary, "edge_ids": list(best_comp_edges)})

    if include_all_components:
        for payload in component_paths:
            selected_ids.extend(payload["edge_ids"])
        selected_component_ids = [int(payload["summary"]["component_id"]) for payload in component_paths]
    else:
        major_payload = max(
            component_paths,
            key=lambda payload: (
                payload["summary"]["max_drainage"],
                payload["summary"]["selected_length_m"],
                payload["summary"]["max_order"],
                payload["summary"]["path_score"],
                -int(payload["summary"]["component_id"]),
            ),
        ) if component_paths else None
        selected_ids = list(major_payload["edge_ids"]) if major_payload else []
        selected_component_ids = [int(major_payload["summary"]["component_id"])] if major_payload else []

    selected_ids = sorted(set(selected_ids))
    selected_indices = [edge_by_id[eid]["orig_index"] for eid in selected_ids]
    trunk = rivers.iloc[selected_indices].copy()
    diagnostics = {
        "method": "dominant_trunk",
        "topology_method": topo.method,
        "edge_count": int(n_edges),
        "component_count": int(len(edges_by_component)),
        "outlet_edge_count": int(len(outlet_edges)),
        "selected_edge_count": int(len(selected_ids)),
        "selected_feature_ids": [edge_by_id[eid]["feature_id"] for eid in selected_ids[:200]],
        "component_summaries": component_summaries[:50],
        "selected_component_ids": selected_component_ids[:20],
        "include_all_components": bool(include_all_components),
        "endpoint_tolerance": float(endpoint_tolerance),
    }
    log.info(
        "[MAINSTEM] Dominant trunk solve kept %d of %d reaches across %d component(s) using %s topology (selected components=%s).",
        len(selected_ids),
        n_edges,
        len(edges_by_component),
        topo.method,
        selected_component_ids[:10],
    )
    return trunk, diagnostics
