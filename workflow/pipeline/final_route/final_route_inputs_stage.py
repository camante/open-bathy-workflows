from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pipeline.final_route.final_route_receipts import write_json_receipt


@dataclass
class FinalRoutePaths:
    template_path: Path
    auth_src: Path
    baseline_cudem_src: Path | None
    combined_dir: Path
    aligned_auth_path: Path
    gap_mask_path: Path
    eligible_mask_path: Path
    support_class_path: Path
    regime_class_path: Path
    support_distance_path: Path
    support_density_path: Path
    anchor_uncertainty_path: Path
    guidance_uncertainty_path: Path
    conditioned_uncertainty_path: Path
    guidance_influence_path: Path
    source_candidate_path: Path
    source_candidate_prov_path: Path
    coastal_sdb_confidence_path: Path
    river_anchor_distance_path: Path
    river_anchor_density_path: Path
    river_scaffold_confidence_path: Path
    river_bank_distance_path: Path
    river_bank_influence_runtime_path: Path
    river_bank_elevation_path: Path
    river_primary_surface_path: Path
    river_primary_surface_confidence_path: Path
    river_primary_surface_source_class_path: Path
    river_primary_surface_support_count_path: Path
    river_primary_surface_domain_path: Path
    river_primary_surface_channel_core_preserve_path: Path
    river_channel_core_preservation_zone_path: Path
    river_channel_core_prepost_delta_path: Path
    river_channel_core_bank_pull_risk_path: Path
    river_channel_core_preservation_receipt_path: Path
    river_primary_surface_contract_path: Path
    river_primary_guidance_summary_path: Path
    conditioned_path: Path
    conditioned_prov_path: Path
    precedence_audit_path: Path
    inputs_receipt_path: Path
    guidance_receipt_path: Path
    terrain_receipt_path: Path
    outputs_receipt_path: Path
    final_route_receipt_path: Path
    guidance_uncertainty_contract_path: Path
    conditioning_audit_path: Path
    final_output_layer_contract_path: Path
    final_vertical_semantics_contract_path: Path
    support_note_route: str = "direct_guidance_artifacts_only"


def ensure_dir(path: Path) -> Path:
    """Delegate to canonical implementation in core.paths."""
    from core.paths import ensure_dir as _canonical
    return _canonical(path)


def _find_sdb_depth_raster_local(sdb_dir: Path) -> Optional[Path]:
    sdb_dir = Path(sdb_dir)
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    rel = data.get("sdb_guidance_active") or data.get("depth_raster")
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = (sdb_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    return p if p.exists() else None


def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    """Use the canonical resolver when available, otherwise fall back to the local manifest-only contract."""
    try:
        from core import process_utils
        canonical = getattr(process_utils, "find_sdb_depth_raster", None)
        if callable(canonical):
            return canonical(sdb_dir)
    except ImportError:
        pass
    return _find_sdb_depth_raster_local(sdb_dir)


def find_sdb_guidance_artifact(sdb_dir: Path, key: str, suffix: str) -> Optional[Path]:
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rel = data.get(key)
    if isinstance(rel, str) and rel.strip():
        p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
        if p.exists():
            return p
    # Fallback: construct candidate path from depth_raster stem + suffix.
    # Try both the depth raster's extension (.tif) and .gpkg since some
    # guidance artifacts (e.g. guide_points) are vector, not raster.
    depth_rel = data.get("depth_raster")
    if isinstance(depth_rel, str) and depth_rel.strip():
        dp = (sdb_dir / depth_rel).resolve() if not os.path.isabs(depth_rel) else Path(depth_rel).resolve()
        for ext in (dp.suffix, ".gpkg"):
            cand = dp.with_name(dp.stem + suffix + ext)
            if cand.exists():
                return cand
    return None


def resolve_existing_output_path(mapping: Any, key: str, *, base_dir: Optional[Path] = None) -> Optional[Path]:
    if not isinstance(mapping, dict):
        return None
    cand = mapping.get(key)
    if not cand:
        return None
    try:
        p = Path(str(cand))
    except (TypeError, ValueError):
        return None
    if not p.is_absolute() and base_dir is not None:
        p = (Path(base_dir) / p).resolve()
    try:
        return p if p.exists() else None
    except OSError:
        return None


def resolve_baseline_cudem_interpolation(*, cfg, report: dict, auth_src: Path | None = None) -> Optional[Path]:
    auth_auto = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}
    cand = auth_auto.get("baseline_cudem_interpolation")
    if cand:
        try:
            p = Path(str(cand))
        except (TypeError, ValueError, OSError):
            p = None
        if p is not None and p.exists():
            return p.resolve()
    auth_candidate = auth_src if auth_src is not None else getattr(cfg, "authoritative_base", None)
    if auth_candidate:
        try:
            sibling = Path(str(auth_candidate)).with_name("cudem_baseline_interpolation.tif")
        except (TypeError, ValueError, OSError):
            sibling = None
        if sibling is not None and sibling.exists():
            return sibling.resolve()
    return None


def _select_final_route_template(*, auth_src: Path, baseline_cudem_src: Path | None, sdb_depth_path: Path | None, river_depth_path: Path | None, candidate_path: Path | None) -> tuple[Path | None, str | None, list[str]]:
    """Select the canonical final-route template grid.

    Phase 2 rule: the authoritative base defines the canonical final conditioning
    and validation grid. Other rasters remain usable as background/guidance sources
    but no longer define the template grid.
    """
    selection_order = [
        "authoritative_base_canonical_grid",
        "baseline_cudem_interpolation_background_only",
        "dense_sdb_depth_raster_guidance_only",
        "dense_river_depth_raster_guidance_only",
        "legacy_candidate_diagnostic_only",
    ]
    p = Path(auth_src).resolve()
    if p.exists():
        return p, "authoritative_base_canonical_grid", selection_order
    return None, None, selection_order


def collect_final_route_inputs(*, cfg, candidate_path: Optional[Path], report: dict) -> Optional[FinalRoutePaths]:
    out_dir = Path(getattr(cfg, "out_dir", Path.cwd()))
    auth_src = getattr(cfg, "authoritative_base", None)
    if auth_src is None:
        return None
    auth_src = Path(auth_src).resolve()
    if not auth_src.exists():
        return None

    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
    outputs_base_dir = out_dir
    sdb_dir = outputs_base_dir / "sdb"
    sdb_depth_path = find_sdb_depth_raster(sdb_dir)
    river_depth_path = resolve_existing_output_path(river_outputs, "depth_terrain", base_dir=outputs_base_dir)
    baseline_cudem_src = resolve_baseline_cudem_interpolation(cfg=cfg, report=report, auth_src=auth_src)

    template_path, template_role, template_selection_order = _select_final_route_template(
        auth_src=auth_src,
        baseline_cudem_src=baseline_cudem_src,
        sdb_depth_path=sdb_depth_path,
        river_depth_path=river_depth_path,
        candidate_path=candidate_path,
    )
    if template_path is None:
        return None

    combined_dir = ensure_dir(out_dir / "combined")
    paths = FinalRoutePaths(
        template_path=template_path,
        auth_src=auth_src,
        baseline_cudem_src=baseline_cudem_src,
        combined_dir=combined_dir,
        aligned_auth_path=combined_dir / "authoritative_base_aligned.tif",
        gap_mask_path=combined_dir / "authoritative_gap_mask.tif",
        eligible_mask_path=combined_dir / "authoritative_eligible_fill_mask.tif",
        support_class_path=combined_dir / "support_class.tif",
        regime_class_path=combined_dir / "regime_class.tif",
        support_distance_path=combined_dir / "support_distance.tif",
        support_density_path=combined_dir / "support_density.tif",
        anchor_uncertainty_path=combined_dir / "anchor_uncertainty.tif",
        guidance_uncertainty_path=combined_dir / "guidance_uncertainty.tif",
        conditioned_uncertainty_path=combined_dir / "conditioned_uncertainty.tif",
        guidance_influence_path=combined_dir / "guidance_influence.tif",
        source_candidate_path=combined_dir / "bathy_combined_depth_source_aware_candidate.tif",
        source_candidate_prov_path=combined_dir / "bathy_combined_depth_source_aware_candidate_provenance.tif",
        coastal_sdb_confidence_path=combined_dir / "coastal_sdb_confidence.tif",
        river_anchor_distance_path=combined_dir / "river_anchor_distance.tif",
        river_anchor_density_path=combined_dir / "river_anchor_density.tif",
        river_scaffold_confidence_path=combined_dir / "river_scaffold_confidence.tif",
        river_bank_distance_path=combined_dir / "river_bank_distance.tif",
        river_bank_influence_runtime_path=combined_dir / "river_bank_influence_runtime.tif",
        river_bank_elevation_path=combined_dir / "river_bank_elevation.tif",
        river_primary_surface_path=combined_dir / "river_primary_surface.tif",
        river_primary_surface_confidence_path=combined_dir / "river_primary_surface_confidence.tif",
        river_primary_surface_source_class_path=combined_dir / "river_primary_surface_source_class.tif",
        river_primary_surface_support_count_path=combined_dir / "river_primary_surface_support_count.tif",
        river_primary_surface_domain_path=combined_dir / "river_primary_surface_domain.tif",
        river_primary_surface_channel_core_preserve_path=combined_dir / "river_primary_surface_channel_core_preserve.tif",
        river_channel_core_preservation_zone_path=combined_dir / "river_channel_core_preservation_zone.tif",
        river_channel_core_prepost_delta_path=combined_dir / "river_channel_core_prepost_delta.tif",
        river_channel_core_bank_pull_risk_path=combined_dir / "river_channel_core_bank_pull_risk.tif",
        river_channel_core_preservation_receipt_path=combined_dir / "river_channel_core_preservation_receipt.json",
        river_primary_surface_contract_path=combined_dir / "river_primary_surface_contract.json",
        river_primary_guidance_summary_path=combined_dir / "river_primary_guidance_summary.json",
        conditioned_path=combined_dir / "DEM_enhanced.tif",
        conditioned_prov_path=combined_dir / "DEM_enhanced_provenance.tif",
        precedence_audit_path=combined_dir / "authoritative_precedence_audit.json",
        inputs_receipt_path=combined_dir / "final_route_inputs_receipt.json",
        guidance_receipt_path=combined_dir / "final_route_guidance_receipt.json",
        terrain_receipt_path=combined_dir / "final_route_terrain_receipt.json",
        outputs_receipt_path=combined_dir / "final_route_outputs_receipt.json",
        final_route_receipt_path=combined_dir / "final_route_receipt.json",
        guidance_uncertainty_contract_path=combined_dir / "guidance_uncertainty_contract.json",
        conditioning_audit_path=combined_dir / "conditioning_audit.json",
        final_output_layer_contract_path=combined_dir / "final_output_layer_contract.json",
        final_vertical_semantics_contract_path=combined_dir / "final_vertical_semantics_contract.json",
    )
    write_json_receipt(paths.inputs_receipt_path, {
        "stage": "final_route_inputs",
        "route_mode": "staged_final_route_single_source_of_truth",
        "structural_inputs": {
            "authoritative_base": str(auth_src),
            "baseline_cudem_interpolation": str(baseline_cudem_src) if baseline_cudem_src is not None else None,
            "template_path": str(template_path),
            "canonical_final_grid_source": str(auth_src),
        },
        "diagnostic_only_inputs": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
        },
        "artifact_roles": {
            "authoritative_base": "structural_canonical_grid",
            "baseline_cudem_interpolation": "structural_background" if baseline_cudem_src is not None else None,
            "template_path": "structural_canonical_grid",
            "sdb_depth_raster": "diagnostic_only",
            "river_depth_raster": "diagnostic_only",
            "legacy_candidate": "diagnostic_only",
        },
        "template_role": template_role,
        "template_selection_order": template_selection_order,
    })
    return paths
