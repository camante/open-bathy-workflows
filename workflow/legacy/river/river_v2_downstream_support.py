from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform

from cudem_authoritative import (
    DEFAULT_SPATIAL_META_URL,
    DEFAULT_TILE_INDEX_URL,
    attach_metadata_paths,
    collect_support_geometries,
    detect_url_field,
    discover_vector_file,
    download_file,
    extract_zip,
    read_tile_index,
    select_tiles,
)


@dataclass(frozen=True)
class RiverV2DownstreamSupportSearchResult:
    search_performed: bool
    search_stop_reason: str
    support_found: bool
    searched_reach_count: int
    downstream_reach_count: int
    export_reach_count: int
    support_hit_reach_count: int
    searched_distance_native_units: float
    support_span_native_units: float
    support_basis: Optional[str] = None
    search_error: Optional[str] = None
    network_layer_used: Optional[str] = None
    support_layer_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _path_exists(value: Any) -> bool:
    if value in (None, "", False):
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False


def _parse_aoi(cfg: Any):
    raw = str(getattr(cfg, "aoi", "") or "").strip()
    if not raw:
        raise ValueError("missing_aoi")
    parts = [float(p) for p in raw.split("/")]
    if len(parts) != 4:
        raise ValueError("invalid_aoi")
    west, east, south, north = parts
    if not (west < east and south < north):
        raise ValueError("invalid_aoi")
    return west, east, south, north


def _read_network_lines(network_gpkg: Path) -> tuple[gpd.GeoDataFrame, str]:
    preferred_layers = (
        "mainstem_solve_network",
        "major_system_network",
        "major_system_network_clip",
        "rivers_clip",
    )
    layers = [layer.name for layer in gpd.list_layers(network_gpkg).itertuples(index=False)]
    if not layers:
        raise ValueError("network_has_no_layers")

    def _read_valid_layer(layer_name: str) -> Optional[gpd.GeoDataFrame]:
        gdf = gpd.read_file(network_gpkg, layer=layer_name)
        if gdf.empty:
            return None
        if "geometry" not in gdf.columns:
            return None
        if not {"from_node", "to_node"}.issubset(gdf.columns):
            return None
        return gdf

    for chosen in preferred_layers:
        if chosen not in layers:
            continue
        gdf = _read_valid_layer(chosen)
        if gdf is not None:
            return gdf, chosen

    for chosen in layers:
        gdf = _read_valid_layer(chosen)
        if gdf is not None:
            return gdf, chosen

    raise ValueError("network_missing_required_columns")


def _read_support_gdf(path: Path) -> tuple[gpd.GeoDataFrame, Optional[str]]:
    layers_df = gpd.list_layers(path)
    layer_names = [layer.name for layer in layers_df.itertuples(index=False)] if layers_df is not None else []
    chosen = layer_names[0] if layer_names else None
    gdf = gpd.read_file(path, layer=chosen) if chosen else gpd.read_file(path)
    return gdf, chosen


def _downstream_reaches(lines: gpd.GeoDataFrame, export_mask) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    export_reaches = lines.loc[export_mask].copy()
    if export_reaches.empty:
        return export_reaches, lines.iloc[0:0].copy()
    if not {"from_node", "to_node"}.issubset(lines.columns):
        raise ValueError("network_missing_required_columns")

    lines = lines.copy()
    lines["_rowid"] = range(len(lines))
    export_with_ids = lines.loc[export_mask].copy()
    visited = set(export_with_ids["_rowid"].tolist())
    frontier = {value for value in export_with_ids["to_node"].tolist() if value is not None}
    downstream_ids: list[int] = []

    while frontier:
        mask = lines["from_node"].isin(frontier) & ~lines["_rowid"].isin(visited)
        next_reaches = lines.loc[mask].copy()
        if next_reaches.empty:
            break
        downstream_ids.extend(next_reaches["_rowid"].tolist())
        visited.update(next_reaches["_rowid"].tolist())
        frontier = {value for value in next_reaches["to_node"].tolist() if value is not None}

    downstream = lines.loc[lines["_rowid"].isin(downstream_ids)].drop(columns=["_rowid"]).copy()
    return export_with_ids.drop(columns=["_rowid"]), downstream


def _native_length_sum(gdf: Optional[gpd.GeoDataFrame]) -> float:
    if gdf is None or gdf.empty:
        return 0.0
    return float(gdf.geometry.apply(lambda g: 0.0 if g is None else g.length).sum())


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


def _downstream_bounds_4269(downstream_reaches: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    gdf_4269 = _to_epsg4269(downstream_reaches)
    west, south, east, north = gdf_4269.total_bounds.tolist()
    if not (west < east and south < north):
        raise ValueError("invalid_downstream_bounds")
    return float(west), float(east), float(south), float(north)


def _support_search_cache_root(ctx: Any) -> Path:
    cache_root = Path(getattr(getattr(ctx, "cfg", None), "cache_dir", None) or "cache")
    return cache_root / "authoritative_base" / "_support_search"


def _metadata_search_configured(ctx: Any) -> bool:
    cfg = getattr(ctx, "cfg", None)
    tile_index_url = getattr(cfg, "authoritative_base_tile_index_url", None)
    spatial_meta_url = getattr(cfg, "authoritative_base_spatial_meta_url", None)
    return bool(tile_index_url) and bool(spatial_meta_url)


def _collect_support_from_metadata_search(ctx: Any, downstream_reaches: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    bounds_4269 = _downstream_bounds_4269(downstream_reaches)
    cache_root = _support_search_cache_root(ctx)
    cache_root.mkdir(parents=True, exist_ok=True)

    cfg = getattr(ctx, "cfg", None)
    tile_index_url = str(getattr(cfg, "authoritative_base_tile_index_url", "") or DEFAULT_TILE_INDEX_URL)
    spatial_meta_url = str(getattr(cfg, "authoritative_base_spatial_meta_url", "") or DEFAULT_SPATIAL_META_URL)
    missing_meta_policy = str(getattr(cfg, "authoritative_base_missing_meta_policy", "skip") or "skip")
    explicit_url_field = getattr(cfg, "authoritative_base_tile_url_field", None)

    tile_index_zip = download_file(tile_index_url, cache_root / Path(tile_index_url).name)
    tile_index_root = extract_zip(tile_index_zip, cache_root / "tile_index")
    tile_index_vector = discover_vector_file(tile_index_root)
    tile_index_gdf = read_tile_index(tile_index_vector)
    url_field = detect_url_field(tile_index_gdf, explicit=explicit_url_field)
    records = select_tiles(tile_index_gdf, bounds_4269, url_field)

    spatial_meta_zip = download_file(spatial_meta_url, cache_root / Path(spatial_meta_url).name)
    spatial_meta_root = extract_zip(spatial_meta_zip, cache_root / "spatial_meta")
    records = attach_metadata_paths(records, spatial_meta_root, missing_meta_policy, logger=None)

    support_gdf = collect_support_geometries(
        records,
        bounds_4269,
        str(downstream_reaches.crs),
        missing_meta_policy,
        logger=None,
    )
    return support_gdf, "metadata_search_downstream_corridor"


def search_downstream_authoritative_support_details(
    ctx: Any,
) -> tuple[
    RiverV2DownstreamSupportSearchResult,
    Optional[gpd.GeoDataFrame],
    Optional[gpd.GeoDataFrame],
    Optional[gpd.GeoDataFrame],
    Optional[gpd.GeoDataFrame],
]:
    if not _path_exists(getattr(ctx, "network_gpkg", None)):
        return RiverV2DownstreamSupportSearchResult(
            search_performed=False,
            search_stop_reason="missing_network_gpkg",
            support_found=False,
            searched_reach_count=0,
            downstream_reach_count=0,
            export_reach_count=0,
            support_hit_reach_count=0,
            searched_distance_native_units=0.0,
            support_span_native_units=0.0,
        ), None, None, None, None

    support_path = getattr(ctx, "trusted_support_artifact_path", None) or getattr(ctx, "authoritative_support_coverage_path", None)
    if not _path_exists(support_path) and not _metadata_search_configured(ctx):
        return RiverV2DownstreamSupportSearchResult(
            search_performed=False,
            search_stop_reason="support_metadata_unavailable",
            support_found=False,
            searched_reach_count=0,
            downstream_reach_count=0,
            export_reach_count=0,
            support_hit_reach_count=0,
            searched_distance_native_units=0.0,
            support_span_native_units=0.0,
        ), None, None, None, None

    try:
        west, east, south, north = _parse_aoi(getattr(ctx, "cfg", None))
        export_geom = box(west, south, east, north)
        lines, network_layer = _read_network_lines(Path(ctx.network_gpkg))
        if lines.crs is None:
            raise ValueError("network_missing_crs")
        target_crs = str(lines.crs)
        if target_crs and target_crs.upper() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            export_box = transform(transformer.transform, export_geom)
        else:
            export_box = export_geom
        export_mask = lines.geometry.intersects(export_box)
        export_reaches, downstream_reaches = _downstream_reaches(lines, export_mask)
        if downstream_reaches.empty:
            return RiverV2DownstreamSupportSearchResult(
                search_performed=True,
                search_stop_reason="no_downstream_reaches_in_current_scaffold_domain",
                support_found=False,
                searched_reach_count=0,
                downstream_reach_count=0,
                export_reach_count=int(len(export_reaches)),
                support_hit_reach_count=0,
                searched_distance_native_units=0.0,
                support_span_native_units=0.0,
                network_layer_used=network_layer,
            ), export_reaches, downstream_reaches, downstream_reaches.iloc[0:0].copy(), None

        support_gdf = None
        support_layer = None
        support_basis = None
        support_stop_reason = None
        search_performed = True
        if _path_exists(support_path):
            support_gdf, support_layer = _read_support_gdf(Path(support_path))
            if support_gdf is None or support_gdf.empty:
                support_gdf = None
                support_layer = None
                support_stop_reason = "support_metadata_empty"
            else:
                support_stop_reason = "current_scaffold_support_artifact"
        elif not _metadata_search_configured(ctx):
            return RiverV2DownstreamSupportSearchResult(
                search_performed=False,
                search_stop_reason="support_metadata_unavailable",
                support_found=False,
                searched_reach_count=0,
                downstream_reach_count=int(len(downstream_reaches)),
                export_reach_count=int(len(export_reaches)),
                support_hit_reach_count=0,
                searched_distance_native_units=0.0,
                support_span_native_units=0.0,
                network_layer_used=network_layer,
                support_layer_used=None,
            ), export_reaches, downstream_reaches, downstream_reaches.iloc[0:0].copy(), None
        if support_gdf is None and _metadata_search_configured(ctx):
            try:
                support_gdf, support_basis = _collect_support_from_metadata_search(ctx, downstream_reaches)
                support_layer = support_basis
                support_stop_reason = "expanded_metadata_search"
            except Exception:
                support_gdf = None
                support_layer = "metadata_search_unavailable"
                search_performed = False
                support_stop_reason = "support_metadata_unavailable"

        if support_gdf is None or support_gdf.empty:
            return RiverV2DownstreamSupportSearchResult(
                search_performed=search_performed,
                search_stop_reason=(support_stop_reason or "no_support_found_in_expanded_downstream_domain"),
                support_found=False,
                searched_reach_count=int(len(downstream_reaches)) if search_performed else 0,
                downstream_reach_count=int(len(downstream_reaches)),
                export_reach_count=int(len(export_reaches)),
                support_hit_reach_count=0,
                searched_distance_native_units=_native_length_sum(downstream_reaches) if search_performed else 0.0,
                support_span_native_units=0.0,
                network_layer_used=network_layer,
                support_layer_used=support_layer,
            ), export_reaches, downstream_reaches, downstream_reaches.iloc[0:0].copy(), support_gdf

        if support_gdf.crs is None:
            raise ValueError("support_missing_crs")
        if str(support_gdf.crs) != str(downstream_reaches.crs):
            support_gdf = support_gdf.to_crs(str(downstream_reaches.crs))
        support_union = support_gdf.geometry.union_all()
        hit_mask = downstream_reaches.geometry.intersects(support_union)
        support_hits = downstream_reaches.loc[hit_mask].copy()
        support_span = _native_length_sum(support_hits)
        searched_distance = _native_length_sum(downstream_reaches)
        if support_basis is None:
            support_basis = "metadata_proven_downstream_support"
        if support_stop_reason == "current_scaffold_support_artifact":
            stop_reason = "support_found_in_current_scaffold_domain" if not support_hits.empty else "no_support_in_current_scaffold_domain"
        else:
            stop_reason = "support_found_in_expanded_downstream_domain" if not support_hits.empty else "no_support_found_in_expanded_downstream_domain"
        result = RiverV2DownstreamSupportSearchResult(
            search_performed=search_performed,
            search_stop_reason=stop_reason,
            support_found=bool(not support_hits.empty),
            searched_reach_count=int(len(downstream_reaches)),
            downstream_reach_count=int(len(downstream_reaches)),
            export_reach_count=int(len(export_reaches)),
            support_hit_reach_count=int(len(support_hits)),
            searched_distance_native_units=searched_distance,
            support_span_native_units=support_span,
            support_basis=support_basis,
            network_layer_used=network_layer,
            support_layer_used=support_layer,
        )
        return result, export_reaches, downstream_reaches, support_hits, support_gdf
    except Exception as exc:
        return RiverV2DownstreamSupportSearchResult(
            search_performed=True,
            search_stop_reason="support_search_error",
            support_found=False,
            searched_reach_count=0,
            downstream_reach_count=0,
            export_reach_count=0,
            support_hit_reach_count=0,
            searched_distance_native_units=0.0,
            support_span_native_units=0.0,
            search_error=f"{type(exc).__name__}: {exc}",
        ), None, None, None, None


def search_downstream_authoritative_support(ctx: Any) -> tuple[
    RiverV2DownstreamSupportSearchResult,
    Optional[gpd.GeoDataFrame],
    Optional[gpd.GeoDataFrame],
    Optional[gpd.GeoDataFrame],
]:
    result, export_reaches, downstream_reaches, support_hits, _support_coverage = search_downstream_authoritative_support_details(ctx)
    return result, export_reaches, downstream_reaches, support_hits


def _solve_domain_aoi_from_reaches(selected_reaches: gpd.GeoDataFrame) -> str:
    reaches_4269 = _to_epsg4269(selected_reaches)
    west, south, east, north = reaches_4269.total_bounds.tolist()
    return f"{west}/{east}/{south}/{north}"


def materialize_expanded_solve_support_artifacts(
    ctx: Any,
    *,
    export_reaches: Optional[gpd.GeoDataFrame],
    downstream_reaches: Optional[gpd.GeoDataFrame],
    support_hits: Optional[gpd.GeoDataFrame],
    support_coverage: Optional[gpd.GeoDataFrame],
    network_out_path: str | Path,
    support_out_path: str | Path,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    if export_reaches is None or export_reaches.empty:
        return None, None, None
    selected = export_reaches.copy()
    if downstream_reaches is not None and not downstream_reaches.empty and support_hits is not None and not support_hits.empty:
        downstream_sel = downstream_reaches.copy()
        if getattr(downstream_sel, "crs", None) is not None and str(downstream_sel.crs) != str(selected.crs):
            downstream_sel = downstream_sel.to_crs(selected.crs)
        # Avoid GeoPandas concat CRS edge cases from mixed GPKG-backed CRS wrappers.
        selected_cols = list(selected.columns)
        downstream_sel = downstream_sel.reindex(columns=selected_cols)
        merged_rows = selected.to_dict("records") + downstream_sel.to_dict("records")
        normalized_crs = str(selected.crs) if getattr(selected, "crs", None) is not None else None
        selected = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=normalized_crs)
        dedupe_cols = [c for c in ("from_node", "to_node", "nhdplusid") if c in selected.columns]
        if dedupe_cols:
            selected = selected.drop_duplicates(subset=dedupe_cols)
    if selected.empty:
        return None, None, None
    network_path = Path(network_out_path)
    network_path.parent.mkdir(parents=True, exist_ok=True)
    if network_path.exists():
        network_path.unlink()
    # Preserve the layer names expected by the active River v2 centerline/scaffold path.
    # The expanded solve-support network is conceptually a retained-flow substitute, so
    # write it under the same canonical flow layers rather than introducing a one-off
    # layer name that downstream code does not know how to open.
    selected.to_file(network_path, layer="rivers_clip", driver="GPKG")
    selected.to_file(network_path, layer="major_system_network", driver="GPKG")
    original_network = Path(getattr(ctx, 'network_gpkg', ''))
    if original_network.exists():
        original_layers = [str(layer.name) for layer in gpd.list_layers(original_network).itertuples(index=False)]
        polygon_layer = next((name for name in ('nhdarea_clip', 'nhdarea') if name in original_layers), None)
        if polygon_layer is not None:
            polygons = gpd.read_file(original_network, layer=polygon_layer)
            polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
            if not polygons.empty:
                if getattr(polygons, 'crs', None) is not None and str(polygons.crs) != str(selected.crs):
                    polygons = polygons.to_crs(selected.crs)
                selected_union = selected.geometry.union_all()
                polygons = polygons.loc[polygons.geometry.intersects(selected_union)].copy()
                if not polygons.empty:
                    polygons.to_file(network_path, layer='nhdarea_clip', driver='GPKG')
    solve_aoi = _solve_domain_aoi_from_reaches(selected)

    support_path = None
    if support_coverage is not None and not support_coverage.empty:
        support_path = Path(support_out_path)
        support_path.parent.mkdir(parents=True, exist_ok=True)
        if support_path.exists():
            support_path.unlink()
        support_coverage.to_file(support_path, layer="solve_support_coverage", driver="GPKG")
    return network_path, support_path, solve_aoi


def write_downstream_support_search_artifacts(
    result: RiverV2DownstreamSupportSearchResult,
    *,
    json_path: str | Path,
    export_reaches: Optional[gpd.GeoDataFrame] = None,
    downstream_reaches: Optional[gpd.GeoDataFrame] = None,
    support_hits: Optional[gpd.GeoDataFrame] = None,
    gpkg_path: str | Path | None = None,
) -> Path:
    json_out = Path(json_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if gpkg_path is not None:
        gpkg_out = Path(gpkg_path)
        gpkg_out.parent.mkdir(parents=True, exist_ok=True)
        if gpkg_out.exists():
            gpkg_out.unlink()
        layers = (
            ("export_reaches", export_reaches),
            ("downstream_reaches", downstream_reaches),
            ("support_hits", support_hits),
        )
        for layer_name, gdf in layers:
            if gdf is None:
                continue
            try:
                gdf.to_file(gpkg_out, layer=layer_name, driver="GPKG")
            except Exception:
                continue
    return json_out


__all__ = [
    "RiverV2DownstreamSupportSearchResult",
    "search_downstream_authoritative_support",
    "search_downstream_authoritative_support_details",
    "materialize_expanded_solve_support_artifacts",
    "write_downstream_support_search_artifacts",
]
