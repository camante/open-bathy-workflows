from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from final_route_receipts import write_json_receipt


@dataclass
class FinalRoutePaths:
    template_path: Path
    auth_src: Path
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
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rel = data.get("depth_raster")
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(rel, str) and rel.strip():
        p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
        if p.exists():
            return p
    return None


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
    depth_rel = data.get("depth_raster")
    if isinstance(depth_rel, str) and depth_rel.strip():
        dp = (sdb_dir / depth_rel).resolve() if not os.path.isabs(depth_rel) else Path(depth_rel).resolve()
        cand = dp.with_name(dp.stem + suffix + dp.suffix)
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

    template_path = None
    for cand in (sdb_depth_path, river_depth_path, auth_src, candidate_path):
        if cand is None:
            continue
        p = Path(cand).resolve()
        if p.exists():
            template_path = p
            break
    if template_path is None:
        return None

    combined_dir = ensure_dir(out_dir / "combined")
    paths = FinalRoutePaths(
        template_path=template_path,
        auth_src=auth_src,
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
        conditioned_path=combined_dir / "bathy_combined_depth_conditioned.tif",
        conditioned_prov_path=combined_dir / "bathy_combined_depth_conditioned_provenance.tif",
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
            "template_path": str(template_path),
        },
        "diagnostic_only_inputs": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
        },
        "artifact_roles": {
            "authoritative_base": "structural",
            "template_path": "structural",
            "sdb_depth_raster": "diagnostic_only",
            "river_depth_raster": "diagnostic_only",
            "legacy_candidate": "diagnostic_only",
        },
        "template_selection_order": [
            "dense_sdb_depth_raster",
            "dense_river_depth_raster",
            "authoritative_base",
            "legacy_candidate_last_resort_only",
        ],
    })
    return paths
