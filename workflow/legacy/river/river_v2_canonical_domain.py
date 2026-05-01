from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform

from legacy.river.river_v2_downstream_support import _read_network_lines


@dataclass(frozen=True)
class CanonicalRiverSolveDomainResult:
    canonical_system_id: Optional[str]
    export_aoi: str
    canonical_solve_aoi: str
    solve_stop_reason: str
    estuary_reached: bool
    max_trace_distance_reached: bool
    selected_reach_count: int
    selected_reach_length_m: float
    canonical_network_gpkg: Path
    canonical_receipt_json: Path
    canonical_domain_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["canonical_network_gpkg"] = str(self.canonical_network_gpkg)
        payload["canonical_receipt_json"] = str(self.canonical_receipt_json)
        return payload


def _parse_aoi(ctx: Any) -> str:
    explicit = getattr(ctx, "export_aoi", None)
    if explicit not in (None, "", False):
        return str(explicit)
    cfg = getattr(ctx, "cfg", None)
    raw = str(getattr(cfg, "aoi", "") or "").strip()
    if not raw:
        raise ValueError("missing_export_aoi")
    parts = [float(p) for p in raw.split("/")]
    if len(parts) != 4:
        raise ValueError("invalid_export_aoi")
    west, east, south, north = parts
    if not (west < east and south < north):
        raise ValueError("invalid_export_aoi")
    return f"{west}/{east}/{south}/{north}"


def _export_geom_in_network_crs(export_aoi: str, network_crs: Any):
    west, east, south, north = [float(p) for p in export_aoi.split("/")]
    geom = box(west, south, east, north)
    if network_crs is None:
        return geom
    target = CRS.from_user_input(str(network_crs))
    if target.to_epsg() == 4326:
        return geom
    transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    return transform(transformer.transform, geom)


def _line_length_native_units(gdf: gpd.GeoDataFrame) -> float:
    if gdf is None or gdf.empty:
        return 0.0
    if "length_m" in gdf.columns:
        values = pd.to_numeric(gdf["length_m"], errors="coerce")
        if values.notna().any():
            return float(values.fillna(0.0).sum())
    if "lengthkm" in gdf.columns:
        values = pd.to_numeric(gdf["lengthkm"], errors="coerce")
        if values.notna().any():
            return float(values.fillna(0.0).sum() * 1000.0)
    return float(gdf.geometry.apply(lambda geom: 0.0 if geom is None else geom.length).sum())


def _candidate_system_id(lines: gpd.GeoDataFrame) -> Optional[str]:
    for col in ("component_id", "levelpathi", "levelpath_id", "level_path_id", "major_system_id"):
        if col in lines.columns:
            vals = lines[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            if not vals.empty:
                uniq = vals.unique().tolist()
                if len(uniq) == 1:
                    return str(uniq[0])
    return None


def _select_shared_system(lines: gpd.GeoDataFrame, export_reaches: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame | None, Optional[str], Optional[str]]:
    for col in ("component_id", "levelpathi", "levelpath_id", "level_path_id", "major_system_id"):
        if col not in lines.columns or col not in export_reaches.columns:
            continue
        export_vals = export_reaches[col].dropna().astype(str).str.strip()
        export_vals = export_vals[export_vals != ""]
        if export_vals.empty:
            continue
        uniq = sorted(set(export_vals.tolist()))
        if len(uniq) != 1:
            continue
        system_id = uniq[0]
        all_vals = lines[col].astype(str).str.strip()
        selected = lines.loc[all_vals == system_id].copy()
        if selected.empty:
            continue
        return selected, col, system_id
    return None, None, None


def _seed_reaches(lines: gpd.GeoDataFrame, export_geom) -> gpd.GeoDataFrame:
    mask = lines.geometry.intersects(export_geom)
    return lines.loc[mask].copy()


def _primary_seed_ids(export_reaches: gpd.GeoDataFrame, export_geom) -> set[int]:
    work = export_reaches.copy()
    work = work[work.geometry.notnull() & ~work.geometry.is_empty].copy()
    if work.empty or "_rowid" not in work.columns:
        return set()
    work["_overlap_len"] = work.geometry.apply(lambda geom: 0.0 if geom is None or geom.is_empty else float(geom.intersection(export_geom).length))
    sort_cols = ["_overlap_len"]
    ascending = [False]
    for col in ("streamorde", "areasqkm", "totdasqkm", "length_m"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(False)
    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    first = work.iloc[0]
    return {int(first["_rowid"])}


def _read_estuary_points(network_gpkg: Path) -> gpd.GeoDataFrame | None:
    layers = [str(layer.name) for layer in gpd.list_layers(network_gpkg).itertuples(index=False)]
    if "estuary_control_points" not in layers:
        return None
    gdf = gpd.read_file(network_gpkg, layer="estuary_control_points")
    if gdf.empty or "geometry" not in gdf.columns:
        return None
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()



def _select_connected_component(lines: gpd.GeoDataFrame, seed_ids: set[int]) -> gpd.GeoDataFrame:
    work = lines.copy()
    if "_rowid" not in work.columns:
        work["_rowid"] = range(len(work))
    selected_ids: set[int] = set(int(v) for v in seed_ids)
    frontier_nodes: set[Any] = set()
    seed = work.loc[work["_rowid"].isin(selected_ids)]
    for col in ("from_node", "to_node"):
        if col in seed.columns:
            frontier_nodes.update({value for value in seed[col].tolist() if pd.notna(value)})
    while frontier_nodes:
        mask = (
            (work["from_node"].isin(frontier_nodes) | work["to_node"].isin(frontier_nodes))
            & ~work["_rowid"].isin(selected_ids)
        )
        next_reaches = work.loc[mask]
        if next_reaches.empty:
            break
        next_ids = set(int(v) for v in next_reaches["_rowid"].tolist())
        selected_ids.update(next_ids)
        frontier_nodes = set()
        for col in ("from_node", "to_node"):
            frontier_nodes.update({value for value in next_reaches[col].tolist() if pd.notna(value)})
    return work.loc[work["_rowid"].isin(selected_ids)].drop(columns=["_rowid"], errors="ignore").copy()

def _trace_direction(
    lines: gpd.GeoDataFrame,
    seed_ids: set[int],
    *,
    direction: str,
    max_trace_distance_m: float,
    estuary_union=None,
) -> tuple[set[int], bool, bool]:
    if direction not in {"upstream", "downstream"}:
        raise ValueError("invalid_trace_direction")
    work = lines.copy()
    if "_rowid" not in work.columns:
        work["_rowid"] = range(len(work))
    selected_ids: set[int] = set(seed_ids)
    traced_ids: set[int] = set()
    max_hit = False
    estuary_reached = False
    length_total = 0.0

    if direction == "downstream":
        frontier = {value for value in work.loc[work["_rowid"].isin(seed_ids), "to_node"].tolist() if pd.notna(value)}
        next_col = "from_node"
        frontier_col = "to_node"
    else:
        frontier = {value for value in work.loc[work["_rowid"].isin(seed_ids), "from_node"].tolist() if pd.notna(value)}
        next_col = "to_node"
        frontier_col = "from_node"

    while frontier:
        mask = work[next_col].isin(frontier) & ~work["_rowid"].isin(selected_ids)
        next_reaches = work.loc[mask].copy()
        if next_reaches.empty:
            break
        if direction == "downstream" and estuary_union is not None:
            hit_mask = next_reaches.geometry.intersects(estuary_union)
            if bool(hit_mask.any()):
                estuary_reached = True
                next_reaches = next_reaches.loc[hit_mask | ~hit_mask]  # keep the estuary-reaching segment(s)
        next_ids = set(int(v) for v in next_reaches["_rowid"].tolist())
        if not next_ids:
            break
        traced_ids.update(next_ids)
        selected_ids.update(next_ids)
        length_total += _line_length_native_units(next_reaches)
        if max_trace_distance_m > 0.0 and length_total >= max_trace_distance_m:
            max_hit = True
            break
        if estuary_reached:
            break
        frontier = {value for value in next_reaches[frontier_col].tolist() if pd.notna(value)}
    return traced_ids, estuary_reached, max_hit


def _to_epsg4269(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("network_missing_crs")
    src_crs = CRS.from_user_input(str(gdf.crs))
    if src_crs.to_epsg() == 4269:
        return gdf.copy()
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4269), always_xy=True)
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: transform(transformer.transform, geom) if geom is not None and not geom.is_empty else geom)
    out.set_crs("EPSG:4269", inplace=True, allow_override=True)
    return out


def _solve_aoi_from_reaches(selected: gpd.GeoDataFrame) -> str:
    gdf_4269 = _to_epsg4269(selected)
    west, south, east, north = gdf_4269.total_bounds.tolist()
    return f"{float(west)}/{float(east)}/{float(south)}/{float(north)}"


def _layer_selection_keys(selected: gpd.GeoDataFrame) -> dict[str, set[Any]]:
    keys: dict[str, set[Any]] = {}
    for col in ("nhdplusid", "comid", "reach_id", "from_node", "to_node"):
        if col in selected.columns:
            vals = {value for value in selected[col].tolist() if pd.notna(value)}
            if vals:
                keys[col] = vals
    return keys


def _subset_layer_to_selected(layer_gdf: gpd.GeoDataFrame, selected: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    if layer_gdf.empty:
        return layer_gdf.copy()
    keys = _layer_selection_keys(selected)
    if layer_name in {"major_system_network", "rivers_clip", "mainstem_solve_network", "major_system_network_clip"}:
        for col, vals in keys.items():
            if col in layer_gdf.columns:
                subset = layer_gdf.loc[layer_gdf[col].isin(vals)].copy()
                if not subset.empty:
                    return subset
    selected_union = selected.geometry.union_all()
    out = layer_gdf.copy()
    if getattr(out, "crs", None) is not None and getattr(selected, "crs", None) is not None and str(out.crs) != str(selected.crs):
        out = out.to_crs(selected.crs)
    out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
    return out.loc[out.geometry.intersects(selected_union)].copy()


def _write_canonical_network(ctx: Any, selected: gpd.GeoDataFrame) -> Path:
    out_path = Path(ctx.paths.canonical_solve_network)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    original_network = Path(getattr(ctx, "export_network_gpkg", None) or getattr(ctx, "network_gpkg", None))
    layers = [str(layer.name) for layer in gpd.list_layers(original_network).itertuples(index=False)]
    preferred_layers = [
        "major_system_network",
        "rivers_clip",
        "mainstem_solve_network",
        "major_system_network_clip",
        "nhdarea_clip",
        "nhdarea",
        "estuary_control_points",
        "outlet_anchors",
    ]
    wrote_any = False
    for layer_name in preferred_layers:
        if layer_name not in layers:
            continue
        layer_gdf = gpd.read_file(original_network, layer=layer_name)
        subset = _subset_layer_to_selected(layer_gdf, selected, layer_name)
        if subset.empty and layer_name in {"major_system_network", "rivers_clip"}:
            subset = selected.copy()
        if subset.empty:
            continue
        subset.to_file(out_path, layer=layer_name, driver="GPKG")
        wrote_any = True
    if not wrote_any:
        selected.to_file(out_path, layer="major_system_network", driver="GPKG")
        selected.to_file(out_path, layer="rivers_clip", driver="GPKG")
    return out_path


def _write_receipt(result: CanonicalRiverSolveDomainResult) -> Path:
    path = Path(result.canonical_receipt_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_canonical_river_solve_domain(ctx: Any) -> CanonicalRiverSolveDomainResult:
    export_aoi = _parse_aoi(ctx)
    network_path = Path(getattr(ctx, "export_network_gpkg", None) or getattr(ctx, "network_gpkg", None))
    lines, _network_layer = _read_network_lines(network_path)
    if not {"from_node", "to_node"}.issubset(lines.columns):
        raise ValueError("network_missing_required_columns")
    lines = lines.copy()
    lines["_rowid"] = range(len(lines))
    export_geom = _export_geom_in_network_crs(export_aoi, lines.crs)
    export_reaches = _seed_reaches(lines, export_geom)
    if export_reaches.empty:
        raise ValueError("canonical_solve_domain_no_export_intersecting_reaches")

    estuary_points = _read_estuary_points(network_path)
    estuary_union = None
    if estuary_points is not None and not estuary_points.empty:
        if getattr(estuary_points, "crs", None) is not None and str(estuary_points.crs) != str(lines.crs):
            estuary_points = estuary_points.to_crs(lines.crs)
        estuary_union = estuary_points.geometry.union_all()

    max_trace_distance_m = float(getattr(ctx, "river_canonical_max_trace_km", 200.0) or 0.0) * 1000.0
    selected_ids: set[int] = set()
    upstream_ids: set[int] = set()
    downstream_ids: set[int] = set()
    upstream_max = False
    downstream_max = False
    estuary_reached = False
    selected: gpd.GeoDataFrame
    all_seed_ids = {int(v) for v in export_reaches["_rowid"].tolist()}
    seed_ids = _primary_seed_ids(export_reaches, export_geom) or all_seed_ids
    connected_selected = _select_connected_component(lines, seed_ids)
    shared_selected, shared_col, shared_system_id = _select_shared_system(lines, export_reaches)
    if not connected_selected.empty:
        selected = connected_selected.drop(columns=["_rowid"], errors="ignore").copy()
        if estuary_union is not None and not selected.empty:
            estuary_reached = bool(selected.geometry.intersects(estuary_union).any())
        solve_stop_reason = "connected_component_selected"
        if estuary_reached:
            solve_stop_reason = "estuary_transition_reached"
    elif shared_selected is not None:
        selected = shared_selected.drop(columns=["_rowid"], errors="ignore").copy()
        if estuary_union is not None and not selected.empty:
            estuary_reached = bool(selected.geometry.intersects(estuary_union).any())
        solve_stop_reason = "shared_system_selected"
        if estuary_reached:
            solve_stop_reason = "estuary_transition_reached"
    else:
        upstream_ids, _upstream_estuary, upstream_max = _trace_direction(
            lines,
            seed_ids,
            direction="upstream",
            max_trace_distance_m=max_trace_distance_m,
        )
        downstream_ids, estuary_reached, downstream_max = _trace_direction(
            lines,
            seed_ids,
            direction="downstream",
            max_trace_distance_m=max_trace_distance_m,
            estuary_union=estuary_union,
        )
        selected_ids = seed_ids | upstream_ids | downstream_ids
        selected = lines.loc[lines["_rowid"].isin(selected_ids)].drop(columns=["_rowid"], errors="ignore").copy()
        selected = selected.drop_duplicates(subset=[c for c in ("nhdplusid", "from_node", "to_node") if c in selected.columns])
        solve_stop_reason = "connected_system_exhausted"
        if estuary_reached:
            solve_stop_reason = "estuary_transition_reached"
        elif upstream_max or downstream_max:
            solve_stop_reason = "max_trace_distance_reached"
    if selected.empty:
        raise ValueError("canonical_solve_domain_empty_after_trace")

    canonical_solve_aoi = _solve_aoi_from_reaches(selected)
    canonical_network_gpkg = _write_canonical_network(ctx, selected)
    canonical_system_id = shared_system_id or _candidate_system_id(selected) or _candidate_system_id(export_reaches) or Path(network_path).stem
    result = CanonicalRiverSolveDomainResult(
        canonical_system_id=str(canonical_system_id) if canonical_system_id is not None else None,
        export_aoi=export_aoi,
        canonical_solve_aoi=canonical_solve_aoi,
        solve_stop_reason=solve_stop_reason,
        estuary_reached=bool(estuary_reached),
        max_trace_distance_reached=bool(upstream_max or downstream_max),
        selected_reach_count=int(len(selected)),
        selected_reach_length_m=float(_line_length_native_units(selected)),
        canonical_network_gpkg=canonical_network_gpkg,
        canonical_receipt_json=Path(ctx.paths.canonical_solve_domain_json),
        canonical_domain_summary={
            "export_reach_count": int(len(export_reaches)),
            "upstream_added_reach_count": int(len(upstream_ids)),
            "downstream_added_reach_count": int(len(downstream_ids)),
            "trace_distance_limit_m": float(max_trace_distance_m),
            "shared_system_selection_column": str(shared_col) if 'shared_col' in locals() and shared_col else None,
        },
    )
    _write_receipt(result)
    return result


__all__ = [
    "CanonicalRiverSolveDomainResult",
    "build_canonical_river_solve_domain",
]
