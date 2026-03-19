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
        "schema_version": 1,
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
        "notes": {
            "depth_raster": "Dense SDB surface is diagnostic and should not be treated as peer authoritative terrain.",
            "guide_points": "Spatially thinned pseudo-soundings for confidence-weighted interpolation guidance.",
            "bounds": "Lower/upper bounds are uncertainty-derived plausible guidance envelopes, not hard truth.",
        },
    }
    return manifest


def write_sdb_guidance_manifest(*, out_root: str | Path, depth_raster: str | Path, args: Any, logger: Optional[logging.Logger] = None) -> Path:
    manifest = build_sdb_guidance_manifest(out_root=out_root, depth_raster=depth_raster, args=args)
    manifest_path = guidance_artifact_paths(depth_raster)["guidance_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger or log).info("Wrote SDB guidance manifest: %s", manifest_path)
    return manifest_path
