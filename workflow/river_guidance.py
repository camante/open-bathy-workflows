from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from river_longitudinal_profile import build_and_write_longitudinal_profile
from river_longitudinal_profile_contract import write_river_longitudinal_profile_contract

log = logging.getLogger(__name__)


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
        "longitudinal_profile_elevation": river_dir / "river_longitudinal_profile_elevation.tif",
        "longitudinal_profile_uncertainty": river_dir / "river_longitudinal_profile_uncertainty.tif",
        "longitudinal_profile_influence": river_dir / "river_longitudinal_profile_influence.tif",
        "hydraulic_backbone": river_dir / "river_hydraulic_backbone.csv",
        "hydraulic_backbone_nodes": river_dir / "river_hydraulic_backbone_nodes.gpkg",
        "hydraulic_backbone_edges": river_dir / "river_hydraulic_backbone_edges.gpkg",
    }




def _discover_optional_wse_path(
    river_dir: str | Path,
    river_outputs: Dict[str, Any],
    *,
    keys: tuple[str, ...],
    relpaths: tuple[str, ...],
) -> Optional[str]:
    river_dir = Path(river_dir)
    for key in keys:
        raw = river_outputs.get(key) if isinstance(river_outputs, dict) else None
        if isinstance(raw, Path):
            cand = raw
        elif isinstance(raw, str) and raw.strip():
            cand = Path(raw)
        else:
            cand = None
        if cand is not None and cand.exists():
            return str(cand)
    for rel in relpaths:
        cand = river_dir / rel
        if cand.exists():
            return str(cand)
    return None

def _pick_river_guide_value_column(columns: list[str]) -> Optional[str]:
    preferred = [
        "centerline_z_m",
        "xs_z_m",
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
    lowered = {str(c).lower(): c for c in columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    for raw, original in lowered.items():
        if raw.endswith('_z_m') or raw.endswith('_elevation'):
            return original
    return None


def rasterize_river_guide_points_to_template(guide_points_path: str | Path, template_raster: str | Path, *, logger: Optional[logging.Logger] = None):
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    try:
        import geopandas as gpd
    except Exception:
        log.debug("rasterize_river_guide_points_to_template: suppressed exception", exc_info=True)
        return None

    gp = Path(guide_points_path)
    tmpl = Path(template_raster)
    if (not gp.exists()) or (not tmpl.exists()):
        return None

    try:
        gdf = gpd.read_file(gp)
    except Exception:
        log.debug("rasterize_river_guide_points_to_template: suppressed exception", exc_info=True)
        return None
    if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
        return None
    value_col = _pick_river_guide_value_column(list(gdf.columns))
    if value_col is None:
        return None
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return None

    with rasterio.open(tmpl) as ds:
        if gdf.crs is not None and ds.crs is not None and str(gdf.crs) != str(ds.crs):
            try:
                gdf = gdf.to_crs(ds.crs)
            except Exception:
                log.debug("rasterize_river_guide_points_to_template: suppressed exception", exc_info=True)
                return None
        arr = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        sums = np.zeros((ds.height, ds.width), dtype=np.float64)
        counts = np.zeros((ds.height, ds.width), dtype=np.uint32)
        vals = np.asarray(gdf[value_col], dtype=float)
        for geom, val in zip(gdf.geometry, vals):
            if geom is None or not np.isfinite(val):
                continue
            try:
                r, c = rowcol(ds.transform, geom.x, geom.y)
            except Exception:
                log.debug("rasterize_river_guide_points_to_template: suppressed exception", exc_info=True)
                continue
            if 0 <= int(r) < ds.height and 0 <= int(c) < ds.width:
                sums[int(r), int(c)] += float(val)
                counts[int(r), int(c)] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        arr[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    (logger or logging.getLogger(__name__)).info('Rasterized river guide points onto template grid: %s -> %s populated cells', gp, int(np.sum(np.isfinite(arr))))
    return arr


def build_river_guidance_manifest(*, out_root: str | Path, river_dir: str | Path, report: Dict[str, Any]) -> Dict[str, Any]:
    out_root = Path(out_root)
    river_dir = Path(river_dir)
    outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
    guidance = report.get("river", {}).get("guidance", {}) if isinstance(report.get("river", {}), dict) else {}

    def _normalize_path(value: Any, fallback: Path) -> Optional[Path]:
        if isinstance(value, Path):
            return value
        if isinstance(value, str) and value.strip():
            return Path(value)
        return fallback if fallback.exists() else None

    paths = guidance_artifact_paths(river_dir)
    artifacts: Dict[str, Optional[str]] = {}
    for key, fallback in paths.items():
        if key == "guidance_manifest":
            continue
        raw = outputs.get(key)
        path_obj = _normalize_path(raw, fallback)
        if path_obj is None or not path_obj.exists():
            continue
        try:
            artifacts[key] = str(path_obj.relative_to(out_root))
        except ValueError:
            artifacts[key] = str(path_obj)

    manifest = {
        "schema_version": 2,
        "artifact_family": "river_guidance",
        "guidance_only": True,
        "artifacts": artifacts,
        "artifact_roles": {
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
            "longitudinal_profile_elevation": "rasterized_longitudinal_profile_elevation",
            "longitudinal_profile_uncertainty": "rasterized_longitudinal_profile_uncertainty",
            "longitudinal_profile_influence": "rasterized_longitudinal_profile_influence",
            "hydraulic_backbone": "network_aware_1d_hydraulic_backbone_table",
            "hydraulic_backbone_nodes": "network_aware_1d_hydraulic_backbone_nodes",
            "hydraulic_backbone_edges": "network_aware_1d_hydraulic_backbone_edges",
        },
        "final_route_contract": {
            "route_role": "structured_subordinate_river_guidance_only",
            "allowed_structural_artifacts": [
                "admissibility",
                "authoritative_support",
                "authoritative_support_depth",
                "bank_confluence_damping",
                "bank_continuity_weight",
                "bank_distance",
                "bank_edge_mask",
                "bank_elevation_xs",
                "bank_graph_confidence",
                "bank_influence",
                "bank_pair_weight",
                "bank_points",
                "bank_estuary_side_decay",
                "centerline_elevation",
                "centerline_influence",
                "centerline_points",
                "centerline_stationing",
                "corridor_mask",
                "guide_points",
                "guidance_weight",
                "regime_class",
                "retained_network",
                "scaffold_domains",
                "scaffold_manifest",
                "soft_guidance_domain",
                "trusted_interior",
                "xs_support_elevation",
                "xs_support_points",
                "xs_support_weight",
                "longitudinal_profile_contract",
                "longitudinal_profile",
                "longitudinal_profile_points",
                "longitudinal_profile_summary",
                "longitudinal_profile_elevation",
                "longitudinal_profile_uncertainty",
                "longitudinal_profile_influence",
                "hydraulic_backbone",
                "hydraulic_backbone_nodes",
                "hydraulic_backbone_edges",
            ],
            "diagnostic_only_artifacts": ["bottom_elevation", "depth_terrain"],
            "forbidden_structural_inputs": [
                "legacy_fused_candidate_raster",
                "dense_river_depth_raster_as_peer_surface",
                "dense_sdb_depth_raster_as_peer_surface",
                "weighted_overlap_blended_bathymetry_as_structural_input",
            ],
        },
        "notes": {
            "depth_terrain": "Dense river depth surface is diagnostic and should not be treated as peer authoritative terrain.",
            "bottom_elevation": "Dense bed-elevation raster is an internal helper/diagnostic product.",
            "guide_points": "Structured scaffold union restricted to retained WAFFLES/NHD river polygons: dense bank boundary points, longitudinal centerline control points, and selected cross-stream support nodes.",
            "trusted_interior": guidance.get("trusted_interior_definition"),
            "soft_guidance_domain": guidance.get("soft_guidance_definition"),
            "admissibility": guidance.get("admissibility_definition"),
            "regime_class": "Shared regime contract classification packaged with the river guidance outputs.",
            "bank_influence": "Bank-boundary guidance derived from the WAFFLES/NHD corridor edge. This is a soft boundary tendency, not hard truth.",
            "bank_elevation_xs": "Longitudinal bank-elevation tendency derived from explicit left/right XS bank picks where available.",
            "bank_pair_weight": "Strength of paired left/right bank support for each corridor pixel.",
            "bank_continuity_weight": "Longitudinal continuity confidence for the persistent left/right bank network built from XS picks.",
            "bank_graph_confidence": "Graph-informed confidence from nearby persistent bank-network coherence.",
            "bank_confluence_damping": "Confidence damping near likely confluences where multiple bank-network components compete.",
            "bank_estuary_side_decay": "Side-confidence decay near estuary transition where left/right bank identity becomes less stable.",
            "bank_points": "Dense bank-boundary control points sampled along retained WAFFLES/NHD polygon banks from the authoritative baseline DEM.",
            "centerline_points": "Dense longitudinal control points sampled along retained mainstem/key-tributary flowlines inside the retained river polygon.",
            "xs_support_points": "Selected cross-stream support nodes sampled from retained cross-sections inside the retained river polygon.",
            "centerline_elevation": "Nearest centerline-derived longitudinal tendency raster sampled from the authoritative baseline DEM.",
            "centerline_influence": "Distance-tapered centerline influence used as longitudinal guidance inside the retained river polygon.",
            "xs_support_elevation": "Nearest selected-XS-node support elevation raster sampled from the authoritative baseline DEM.",
            "xs_support_weight": "Distance-tapered XS support influence used as cross-stream guidance inside the retained river polygon.",
            "retained_network": "Retained river network package containing main stems, important tributaries, and retained WAFFLES/NHD polygons used for scaffold generation.",
            "final_route_contract": "Only the listed structured river guidance artifacts may structurally enter the final DEM route; dense river depth/elevation rasters remain diagnostic-only.",
            "longitudinal_profile_contract": "Initial Phase C contract for a scaffold-owned longitudinal vertical backbone built from stationing, centerline tendency, and optional XS/anchor support.",
            "longitudinal_profile": "Station-indexed scaffold-owned longitudinal bed-profile object derived from centerline stationing plus sampled centerline, XS-support, and bank-elevation tendencies.",
            "longitudinal_profile_points": "Centerline points carrying interpolated longitudinal-profile elevation and uncertainty values for raster derivative generation.",
            "longitudinal_profile_summary": "Summary receipt for the scaffold-owned longitudinal profile object, including station counts and support evidence presence.",
            "longitudinal_profile_elevation": "Rasterized longitudinal-profile bed-elevation tendency generated from the station-indexed profile points.",
            "longitudinal_profile_uncertainty": "Rasterized uncertainty of the longitudinal-profile bed-elevation tendency.",
            "longitudinal_profile_influence": "Distance-tapered influence of the rasterized longitudinal-profile tendency inside the river corridor.",
        },
    }
    return manifest


def write_river_guidance_manifest(*, out_root: str | Path, river_dir: str | Path, report: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Path:
    manifest = build_river_guidance_manifest(out_root=out_root, river_dir=river_dir, report=report)
    manifest_path = guidance_artifact_paths(river_dir)["guidance_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger or logging.getLogger(__name__)).info("Wrote river guidance manifest: %s", manifest_path)
    return manifest_path

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
        wse_elevation_path = _discover_optional_wse_path(
            river_dir,
            river_outputs,
            keys=("wse", "wse_m", "water_surface", "water_surface_elevation", "skeleton_wse"),
            relpaths=("skeleton_debug/wse_m.tif", "wse_m.tif"),
        )
        wse_uncertainty_path = _discover_optional_wse_path(
            river_dir,
            river_outputs,
            keys=("wse_uncertainty", "water_surface_uncertainty"),
            relpaths=("skeleton_debug/wse_uncertainty_m.tif",),
        )
        wse_influence_path = _discover_optional_wse_path(
            river_dir,
            river_outputs,
            keys=("wse_influence", "water_surface_influence"),
            relpaths=(),
        )
        profile_outputs = build_and_write_longitudinal_profile(
            river_dir=river_dir,
            centerline_points_path=river_outputs.get("centerline_points"),
            centerline_elevation_path=river_outputs.get("centerline_elevation"),
            centerline_influence_path=river_outputs.get("centerline_influence"),
            centerline_stationing_path=river_outputs.get("centerline_stationing"),
            xs_support_elevation_path=river_outputs.get("xs_support_elevation"),
            xs_support_weight_path=river_outputs.get("xs_support_weight"),
            bank_elevation_path=river_outputs.get("bank_elevation_xs"),
            bank_influence_path=river_outputs.get("bank_influence"),
            bank_graph_confidence_path=river_outputs.get("bank_graph_confidence"),
            bank_continuity_weight_path=river_outputs.get("bank_continuity_weight"),
            bank_confluence_damping_path=river_outputs.get("bank_confluence_damping"),
            bank_estuary_side_decay_path=river_outputs.get("bank_estuary_side_decay"),
            authoritative_support_depth_path=river_outputs.get("authoritative_support_depth"),
            authoritative_bed_elevation_path=(getattr(cfg, "river_dem", None) or getattr(cfg, "authoritative_base", None)),
            authoritative_support_mask_path=river_outputs.get("authoritative_support"),
            wse_elevation_path=wse_elevation_path,
            wse_influence_path=wse_influence_path,
            wse_uncertainty_path=wse_uncertainty_path,
            corridor_mask_path=river_outputs.get("corridor_mask"),
            network_edges_path=network_edges_path,
        )
        river_outputs.update({k: v for k, v in profile_outputs.items() if v})
        long_profile_path = river_dir / "river_longitudinal_profile_contract.json"
        write_river_longitudinal_profile_contract(
            long_profile_path,
            outputs={
                "centerline_stationing": river_outputs.get("centerline_stationing"),
                "centerline_elevation": river_outputs.get("centerline_elevation"),
                "xs_support_elevation": river_outputs.get("xs_support_elevation"),
                "authoritative_support_depth": river_outputs.get("authoritative_support_depth"),
                "bank_elevation_xs": river_outputs.get("bank_elevation_xs"),
                "longitudinal_profile": river_outputs.get("longitudinal_profile"),
                "longitudinal_profile_elevation": river_outputs.get("longitudinal_profile_elevation"),
                "longitudinal_profile_uncertainty": river_outputs.get("longitudinal_profile_uncertainty"),
            },
        )
        river_outputs["longitudinal_profile_contract"] = str(long_profile_path)
        manifest_path = write_river_guidance_manifest(
            out_root=Path(cfg.out_dir),
            river_dir=river_dir,
            report=report,
            logger=logger,
        )
        report.setdefault("river", {}).setdefault("outputs", {})["guidance_manifest"] = str(manifest_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        report.setdefault("river", {}).setdefault("guidance", {})["artifact_write_error"] = str(e)
        logger.debug("Failed to write river guidance artifacts; continuing.", exc_info=True)


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
