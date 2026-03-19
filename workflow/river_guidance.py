from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional




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
        "scaffold_domains": river_dir / "river_scaffold_domains.json",
        "scaffold_manifest": river_dir / "river_scaffold_manifest.json",
        "guidance_manifest": river_dir / "river_guidance_manifest.json",
        "depth_terrain": river_dir / "river_depth.tif",
        "bottom_elevation": river_dir / "river_bed_elev.tif",
    }


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
        "schema_version": 1,
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
            "guide_points": "sparse_guidance_points",
            "authoritative_support": "authoritative_anchor_support",
            "authoritative_support_depth": "authoritative_anchor_depth",
            "corridor_mask": "river_corridor",
            "scaffold_domains": "scaffold_domain_metadata",
            "scaffold_manifest": "scaffold_product_manifest",
        },
        "notes": {
            "depth_terrain": "Dense river depth surface is diagnostic and should not be treated as peer authoritative terrain.",
            "bottom_elevation": "Dense bed-elevation raster is an internal helper/diagnostic product.",
            "guide_points": "Spatially thinned pseudo-soundings for confidence-weighted interpolation guidance.",
            "trusted_interior": guidance.get("trusted_interior_definition"),
            "soft_guidance_domain": guidance.get("soft_guidance_definition"),
            "admissibility": guidance.get("admissibility_definition"),
            "regime_class": "Shared regime contract classification packaged with the river guidance outputs.",
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
