from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform

from pipeline.river_linear.river_linear_stage_helpers import read_network_lines


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




def _stable_json_hash(payload: Mapping[str, Any] | Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_aoi_for_identity(aoi: str | None) -> str | None:
    if aoi in (None, ""):
        return None
    parts = [round(float(part), 12) for part in str(aoi).split("/")]
    if len(parts) != 4:
        return str(aoi)
    return "/".join(f"{part:.12f}" for part in parts)


def _snap_resolution_for_identity(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), 2)
    except Exception:
        return None


def _path_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    pth = Path(path)
    if not pth.exists():
        return f"missing:{pth.name}"
    h = hashlib.sha256()
    h.update(str(pth.name).encode("utf-8"))
    h.update(str(pth.stat().st_size).encode("utf-8"))
    # Hash a bounded prefix/suffix so the identity is stable without forcing a full
    # large-raster read during domain resolution. Bundle E can harden full cache keys.
    with pth.open("rb") as src:
        head = src.read(1024 * 1024)
        h.update(head)
        size = pth.stat().st_size
        if size > 1024 * 1024:
            src.seek(max(0, size - 1024 * 1024))
            h.update(src.read(1024 * 1024))
    return h.hexdigest()


def _selected_network_fingerprint(selected: gpd.GeoDataFrame | None, *, network_path: Path | None = None) -> str | None:
    if selected is None or selected.empty:
        return _path_fingerprint(network_path)
    work = selected.copy()
    if getattr(work, "crs", None) is not None:
        crs = str(work.crs)
    else:
        crs = None
    id_cols = [col for col in ("component_id", "levelpathi", "levelpath_id", "level_path_id", "major_system_id", "nhdplusid", "comid", "from_node", "to_node") if col in work.columns]
    rows: list[dict[str, Any]] = []
    sort_cols = id_cols or []
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    for _idx, row in work.iterrows():
        geom = row.geometry
        geom_hash = hashlib.sha256(geom.wkb).hexdigest() if geom is not None and not geom.is_empty else None
        rows.append({col: (None if pd.isna(row[col]) else str(row[col])) for col in id_cols} | {"geometry_sha256": geom_hash})
    bounds = [round(float(v), 6) for v in work.total_bounds.tolist()] if not work.empty else None
    return _stable_json_hash({"crs": crs, "bounds": bounds, "rows": rows})


def fingerprint_authoritative_source(ctx: Any) -> str | None:
    """Return only canonical/shared authoritative-source identity for system IDs.

    Canonical river system IDs must not depend on user-AOI local authoritative
    base rasters. Parent DEM cache validation still fingerprints the materialized
    canonical authoritative rasters separately in river_linear_cache.py.
    """
    bundle = getattr(ctx, "linear_inputs", None)
    candidates = [
        getattr(ctx, "authoritative_source_contract_path", None),
        getattr(bundle, "authoritative_source_contract_path", None) if bundle is not None else None,
        getattr(bundle, "canonical_solve_authoritative_measured_only_path", None) if bundle is not None else None,
    ]
    for value in candidates:
        fp = _path_fingerprint(Path(value)) if value not in (None, "") else None
        if fp is not None:
            return fp
    return None


def fingerprint_river_network(ctx: Any, selected: gpd.GeoDataFrame | None = None) -> str | None:
    network_path = Path(getattr(ctx, "export_network_gpkg", None) or getattr(ctx, "network_gpkg", "")) if (getattr(ctx, "export_network_gpkg", None) or getattr(ctx, "network_gpkg", None)) else None
    return _selected_network_fingerprint(selected, network_path=network_path)


def fingerprint_canonical_domain_inputs(
    ctx: Any,
    *,
    selected: gpd.GeoDataFrame | None = None,
    canonical_solve_aoi: str | None = None,
    canonical_trace_distance_km: float | None = None,
) -> dict[str, Any]:
    """Return path-free canonical river-system identity diagnostics.

    Do not include user-AOI local authoritative/baseline raster paths here.
    Those are parent-DEM cache validation inputs, not river-system identity.
    """
    cfg = getattr(ctx, "cfg", None)
    projected_crs = getattr(ctx, "projected_crs", None) or getattr(cfg, "dst_srs", None) or getattr(cfg, "srs", None)
    target_resolution_m = getattr(ctx, "target_resolution_m", None) or getattr(cfg, "xres", None) or getattr(cfg, "res", None)
    trace_km = canonical_trace_distance_km
    if trace_km is None:
        trace_km = getattr(ctx, "canonical_max_trace_km", None)
    if trace_km is None and cfg is not None:
        trace_km = getattr(cfg, "river_canonical_max_trace_km", None)
    return {
        "canonical_solve_aoi": _normalize_aoi_for_identity(canonical_solve_aoi),
        "canonical_trace_distance_km": float(trace_km) if trace_km is not None else None,
        "projected_crs": str(projected_crs) if projected_crs is not None else None,
        "target_resolution_m": _snap_resolution_for_identity(target_resolution_m),
        "river_network_fingerprint": fingerprint_river_network(ctx, selected),
        "authoritative_source_fingerprint": "excluded_from_system_identity",
        "workflow_identity_version": "river_canonical_identity_v2_path_free",
        "excluded_from_identity": [
            "user_aoi",
            "export_aoi",
            "out_dir",
            "timestamp",
            "temporary_directory",
            "run_id",
            "authoritative_base_path",
            "baseline_path",
            "canonical_authoritative_raster_path",
        ],
    }


def build_canonical_system_identity_payload(
    ctx: Any,
    *,
    selected: gpd.GeoDataFrame | None = None,
    canonical_solve_aoi: str | None = None,
    shared_system_id: str | None = None,
    canonical_trace_distance_km: float | None = None,
) -> dict[str, Any]:
    payload = fingerprint_canonical_domain_inputs(
        ctx,
        selected=selected,
        canonical_solve_aoi=canonical_solve_aoi,
        canonical_trace_distance_km=canonical_trace_distance_km,
    )
    if selected is not None and not selected.empty:
        payload["selected_reach_count"] = int(len(selected))
        payload["selected_reach_length_m"] = round(float(_line_length_native_units(selected)), 3)
    else:
        payload["selected_reach_count"] = None
        payload["selected_reach_length_m"] = None
    payload["source_system_id"] = str(shared_system_id) if shared_system_id not in (None, "") else None
    return payload


def resolve_canonical_system_id(
    ctx: Any,
    *,
    selected: gpd.GeoDataFrame | None = None,
    canonical_solve_aoi: str | None = None,
    shared_system_id: str | None = None,
    canonical_trace_distance_km: float | None = None,
) -> str:
    payload = build_canonical_system_identity_payload(
        ctx,
        selected=selected,
        canonical_solve_aoi=canonical_solve_aoi,
        shared_system_id=shared_system_id,
        canonical_trace_distance_km=canonical_trace_distance_km,
    )
    digest = _stable_json_hash(payload)[:20]
    return f"river_system_{digest}"


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
    lines, _network_layer = read_network_lines(network_path)
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

    cfg = getattr(ctx, "cfg", None)
    max_trace_km = getattr(ctx, "canonical_max_trace_km", None)
    if max_trace_km is None and cfg is not None:
        max_trace_km = getattr(cfg, "river_canonical_max_trace_km", None)
    if max_trace_km is None:
        max_trace_km = getattr(ctx, "river_canonical_max_trace_km", 200.0)
    max_trace_distance_m = float(max_trace_km or 0.0) * 1000.0
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
    source_system_id = shared_system_id or _candidate_system_id(selected) or _candidate_system_id(export_reaches)
    canonical_system_id = resolve_canonical_system_id(
        ctx,
        selected=selected,
        canonical_solve_aoi=canonical_solve_aoi,
        shared_system_id=source_system_id,
        canonical_trace_distance_km=float(max_trace_km or 0.0),
    )
    identity_fingerprints = build_canonical_system_identity_payload(
        ctx,
        selected=selected,
        canonical_solve_aoi=canonical_solve_aoi,
        shared_system_id=source_system_id,
        canonical_trace_distance_km=float(max_trace_km or 0.0),
    )
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
            "source_system_id": str(source_system_id) if source_system_id is not None else None,
            "identity_fingerprints": identity_fingerprints,
        },
    )
    _write_receipt(result)
    return result


__all__ = [
    "CanonicalRiverSolveDomainResult",
    "fingerprint_authoritative_source",
    "fingerprint_canonical_domain_inputs",
    "build_canonical_system_identity_payload",
    "fingerprint_river_network",
    "resolve_canonical_system_id",
    "build_canonical_river_solve_domain",
]
