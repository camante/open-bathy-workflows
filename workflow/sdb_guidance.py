from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


def guidance_artifact_paths(depth_raster: str | Path) -> Dict[str, Path]:
    depth = Path(depth_raster)
    stem = depth.stem
    parent = depth.parent
    return {
        "depth_raster": depth,
        "confidence_raster": parent / f"{stem}_confidence.tif",
        "provenance_raster": parent / f"{stem}_provenance.tif",
        "uncertainty_raster": parent / f"{stem}_uncertainty.tif",
        "guidance_weight_raster": parent / f"{stem}_guidance_weight.tif",
        "trusted_interior_raster": parent / f"{stem}_trusted_interior.tif",
        "admissibility_raster": parent / f"{stem}_admissibility.tif",
        "regime_class_raster": parent / f"{stem}_regime_class.tif",
        "guide_points": parent / f"{stem}_guide_points.gpkg",
        "lower_bound_raster": parent / f"{stem}_lower_bound.tif",
        "upper_bound_raster": parent / f"{stem}_upper_bound.tif",
        "guidance_manifest": parent / f"{stem}_guidance_manifest.json",
    }


def write_sdb_guidance_bounds(depth_path: str | Path, uncertainty_path: str | Path, *, logger: Optional[logging.Logger] = None) -> Dict[str, Optional[str]]:
    import rasterio

    depth_path = Path(depth_path)
    uncertainty_path = Path(uncertainty_path)
    paths = guidance_artifact_paths(depth_path)
    lower_path = paths["lower_bound_raster"]
    upper_path = paths["upper_bound_raster"]
    active_log = logger or log

    if (not depth_path.exists()) or (not uncertainty_path.exists()):
        return {"lower_bound_raster": None, "upper_bound_raster": None}

    with rasterio.open(depth_path) as ds_depth, rasterio.open(uncertainty_path) as ds_unc:
        depth = ds_depth.read(1).astype("float32")
        unc = ds_unc.read(1).astype("float32")
        nodata = ds_depth.nodata if ds_depth.nodata is not None else -9999.0
        valid = np.isfinite(depth) & np.isfinite(unc) & (depth != nodata)
        prof = ds_depth.profile.copy()
        lower = np.full(depth.shape, nodata, dtype=np.float32)
        upper = np.full(depth.shape, nodata, dtype=np.float32)
        lower[valid] = depth[valid] - unc[valid]
        upper[valid] = depth[valid] + unc[valid]
        prof.update(dtype="float32", count=1, compress="deflate", nodata=nodata)
        with rasterio.open(lower_path, "w", **prof) as dst:
            dst.write(lower, 1)
        with rasterio.open(upper_path, "w", **prof) as dst:
            dst.write(upper, 1)
    active_log.info("SDB guidance bounds written: %s ; %s", lower_path, upper_path)
    return {"lower_bound_raster": str(lower_path), "upper_bound_raster": str(upper_path)}




def _pick_sdb_guide_value_column(columns: list[str]) -> Optional[str]:
    preferred = [
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
    return None


def rasterize_sdb_guide_points_to_template(guide_points_path: str | Path, template_raster: str | Path, *, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    import rasterio
    from rasterio.transform import rowcol
    try:
        import geopandas as gpd
    except Exception:
        log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
        return None

    gp = Path(guide_points_path)
    tmpl = Path(template_raster)
    if (not gp.exists()) or (not tmpl.exists()):
        return None

    try:
        gdf = gpd.read_file(gp)
    except Exception:
        log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
        return None
    if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
        return None
    value_col = _pick_sdb_guide_value_column(list(gdf.columns))
    if value_col is None:
        return None
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return None

    with rasterio.open(tmpl) as ds:
        if gdf.crs is not None and ds.crs is not None and str(gdf.crs) != str(ds.crs):
            try:
                gdf = gdf.to_crs(ds.crs)
            except Exception:
                log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
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
                log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
                continue
            if 0 <= int(r) < ds.height and 0 <= int(c) < ds.width:
                sums[int(r), int(c)] += float(val)
                counts[int(r), int(c)] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        arr[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    (logger or log).info('Rasterized SDB guide points onto template grid: %s -> %s populated cells', gp, int(np.sum(np.isfinite(arr))))
    return arr

def build_sdb_guidance_manifest(*, out_root: str | Path, depth_raster: str | Path, args: Any) -> Dict[str, Any]:
    out_root = Path(out_root)
    paths = guidance_artifact_paths(depth_raster)

    def _rel_or_abs(p: Path) -> Optional[str]:
        if not p.exists():
            return None
        try:
            return str(p.relative_to(out_root))
        except ValueError:
            return str(p)

    artifacts: Dict[str, Optional[str]] = {}
    for key, p in paths.items():
        if key == "guidance_manifest":
            continue
        rel = _rel_or_abs(p)
        if rel:
            artifacts[key] = rel

    artifacts["guidance_mode"] = "guidance_first"
    artifacts["depth_raster_role"] = "diagnostic_only"
    auth_base = getattr(args, "authoritative_base", None)
    if auth_base:
        artifacts["authoritative_base"] = str(auth_base)
    if hasattr(args, "_authoritative_base_auto_report"):
        artifacts["authoritative_base_auto"] = getattr(args, "_authoritative_base_auto_report")

    manifest = {
        "schema_version": 2,
        "artifact_family": "sdb_guidance",
        "guidance_only": True,
        "artifacts": artifacts,
        "artifact_roles": {
            "depth_raster": "diagnostic_only",
            "confidence_raster": "confidence",
            "provenance_raster": "provenance",
            "guidance_weight_raster": "soft_guidance_weight",
            "trusted_interior_raster": "trusted_export_region",
            "admissibility_raster": "soft_guidance_domain",
            "regime_class_raster": "shared_regime_contract",
            "guide_points": "sparse_guidance_points",
            "lower_bound_raster": "plausible_lower_bound",
            "upper_bound_raster": "plausible_upper_bound",
        },
        "final_route_contract": {
            "route_role": "subordinate_guidance_artifacts_only",
            "allowed_structural_artifacts": [
                "admissibility_raster",
                "confidence_raster",
                "guide_points",
                "guidance_weight_raster",
                "lower_bound_raster",
                "provenance_raster",
                "regime_class_raster",
                "trusted_interior_raster",
                "upper_bound_raster",
            ],
            "diagnostic_only_artifacts": ["depth_raster"],
            "forbidden_structural_inputs": [
                "legacy_fused_candidate_raster",
                "dense_river_depth_raster_as_peer_surface",
                "dense_sdb_depth_raster_as_peer_surface",
                "weighted_overlap_blended_bathymetry_as_structural_input",
            ],
        },
        "notes": {
            "depth_raster": "Dense SDB surface is diagnostic and should not be treated as peer authoritative terrain.",
            "guide_points": "Spatially thinned pseudo-soundings for confidence-weighted interpolation guidance.",
            "bounds": "Lower/upper bounds are uncertainty-derived plausible guidance envelopes, not hard truth.",
            "final_route_contract": "Only the listed subordinate guidance artifacts may structurally enter the final DEM route; dense SDB depth remains diagnostic-only.",
        },
    }
    return manifest


def write_sdb_guidance_manifest(*, out_root: str | Path, depth_raster: str | Path, args: Any, logger: Optional[logging.Logger] = None) -> Path:
    manifest = build_sdb_guidance_manifest(out_root=out_root, depth_raster=depth_raster, args=args)
    manifest_path = guidance_artifact_paths(depth_raster)["guidance_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger or log).info("Wrote SDB guidance manifest: %s", manifest_path)
    return manifest_path
