from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from precedence_audit import summarize_precedence_audit, write_precedence_audit
from conditioning_audit import write_conditioning_audit
from final_guidance_uncertainty_contract import write_guidance_uncertainty_contract
from final_output_layer_contract import write_final_output_layer_contract
from final_vertical_semantics_contract import write_final_vertical_semantics_contract
from final_dem_policy import build_final_dem_policy_dict
from final_route_contract import validate_final_route_contract
from final_route_receipts import write_json_receipt
from legacy_cleanup_stage import write_legacy_cleanup_receipt


def write_final_route_outputs(*, cfg, paths, guidance, terrain, candidate_path: Path | None, provenance_path: Path | None, report: dict) -> tuple[Path, Path, Path, Path, Path, Path]:
    result = terrain.result
    auth = guidance.auth
    profile = guidance.profile
    nodata = guidance.nodata
    source_candidate = terrain.source_candidate
    candidate_prov = terrain.candidate_prov

    locked = result["locked"]
    gap = result["gap"]
    eligible = result["eligible"]
    support = result["support"]
    regime = result["regime"]
    support_distance_m = result["support_distance_m"]
    support_density = result["support_density"]
    anchor_uncertainty = result["anchor_uncertainty"]
    guidance_uncertainty = result["guidance_uncertainty"]
    conditioned_uncertainty = result["conditioned_uncertainty"]
    guidance_influence = result["guidance_influence"]
    coastal_sdb_confidence = result["coastal_sdb_confidence"]
    river_anchor_distance_m = result["river_anchor_distance_m"]
    river_anchor_density = result["river_anchor_density"]
    river_scaffold_confidence = result["river_scaffold_confidence"]
    river_bank_distance_m = result["river_bank_distance_m"]
    river_bank_influence_runtime = result["river_bank_influence"]
    river_bank_elevation = result["river_bank_elevation"]
    river_bank_continuity_weight_runtime = result.get("river_bank_continuity_weight")
    river_bank_graph_confidence_runtime = result.get("river_bank_graph_confidence")
    river_bank_confluence_damping_runtime = result.get("river_bank_confluence_damping")
    river_bank_estuary_side_decay_runtime = result.get("river_bank_estuary_side_decay")
    conditioned = result["conditioned"]
    prov_out = result["provenance"]
    support_note = result["support_note"]

    prof_f32 = profile.copy(); prof_f32.update(dtype="float32", nodata=float(nodata), count=1, compress="deflate")
    prof_u8 = profile.copy(); prof_u8.update(dtype="uint8", nodata=0, count=1, compress="deflate")

    vertical_datum = str(getattr(cfg, "output_vdatum", None) or getattr(cfg, "target_vcrs", None) or getattr(cfg, "working_vcrs", None) or "NAVD88")
    vertical_datum_epsg = str(getattr(cfg, "output_vdatum_epsg", None) or getattr(cfg, "target_vdatum_epsg", None) or getattr(cfg, "working_vdatum_epsg", None) or "5703")

    def _write(path: Path, arr: np.ndarray, prof: dict, *, tags: dict | None = None):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr, 1)
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items() if v is not None})

    unresolved_conditioned = int(np.count_nonzero(~np.isfinite(conditioned)))
    if unresolved_conditioned:
        raise RuntimeError(f"authoritative conditioning produced non-finite final DEM cells: {unresolved_conditioned}")

    auth_out = auth.copy(); auth_out[~np.isfinite(auth_out)] = np.float32(nodata)
    cond_out = conditioned.copy(); cond_out[~np.isfinite(cond_out)] = np.float32(nodata)

    _write(paths.aligned_auth_path, auth_out.astype("float32"), prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "aligned_authoritative_base"})
    _write(paths.gap_mask_path, gap.astype("uint8"), prof_u8)
    _write(paths.eligible_mask_path, eligible.astype("uint8"), prof_u8)
    _write(paths.support_class_path, support.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "support_class"})
    _write(paths.regime_class_path, regime.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "regime_class"})
    guidance_surface = result.get("guidance_surface")
    if guidance_surface is None:
        guidance_surface = np.full(auth.shape, np.nan, dtype="float32")
    _write(paths.source_candidate_path, np.where(np.isfinite(guidance_surface), guidance_surface, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "guidance_surface_diagnostic"})
    _write(paths.source_candidate_prov_path, candidate_prov.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "guidance_surface_provenance"})
    _write(paths.support_distance_path, np.where(np.isfinite(support_distance_m), support_distance_m, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "support_distance"})
    _write(paths.support_density_path, np.clip(support_density, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "support_density"})
    _write(paths.anchor_uncertainty_path, np.where(np.isfinite(anchor_uncertainty), anchor_uncertainty, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "anchor_uncertainty"})
    _write(paths.guidance_uncertainty_path, np.where(np.isfinite(guidance_uncertainty), guidance_uncertainty, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "guidance_uncertainty"})
    _write(paths.conditioned_uncertainty_path, np.where(np.isfinite(conditioned_uncertainty), conditioned_uncertainty, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "conditioned_uncertainty"})
    _write(paths.guidance_influence_path, np.clip(guidance_influence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "guidance_influence"})
    _write(paths.coastal_sdb_confidence_path, np.clip(coastal_sdb_confidence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "coastal_sdb_confidence"})
    _write(paths.river_anchor_distance_path, np.where(np.isfinite(river_anchor_distance_m), river_anchor_distance_m, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "river_anchor_distance"})
    _write(paths.river_anchor_density_path, np.clip(river_anchor_density, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_anchor_density"})
    _write(paths.river_scaffold_confidence_path, np.clip(river_scaffold_confidence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_scaffold_confidence"})
    _write(paths.river_bank_distance_path, np.where(np.isfinite(river_bank_distance_m), river_bank_distance_m, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "river_bank_distance"})
    _write(paths.river_bank_influence_runtime_path, np.clip(river_bank_influence_runtime, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_influence"})
    _write(paths.river_bank_elevation_path, np.where(np.isfinite(river_bank_elevation), river_bank_elevation, np.float32(nodata)).astype("float32"), prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "river_bank_elevation_guidance"})
    river_bank_continuity_runtime_path = paths.combined_dir / "river_bank_continuity_weight.tif"
    river_bank_graph_confidence_runtime_path = paths.combined_dir / "river_bank_graph_confidence.tif"
    river_bank_confluence_damping_runtime_path = paths.combined_dir / "river_bank_confluence_damping.tif"
    river_bank_estuary_side_decay_runtime_path = paths.combined_dir / "river_bank_estuary_side_decay.tif"
    _write(river_bank_continuity_runtime_path, np.clip(np.nan_to_num(river_bank_continuity_weight_runtime, nan=0.0), 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_continuity_weight"})
    _write(river_bank_graph_confidence_runtime_path, np.clip(np.nan_to_num(river_bank_graph_confidence_runtime, nan=0.0), 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_graph_confidence"})
    _write(river_bank_confluence_damping_runtime_path, np.clip(np.nan_to_num(river_bank_confluence_damping_runtime, nan=1.0), 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_confluence_damping"})
    _write(river_bank_estuary_side_decay_runtime_path, np.clip(np.nan_to_num(river_bank_estuary_side_decay_runtime, nan=1.0), 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_estuary_side_decay"})
    _write(paths.conditioned_path, cond_out.astype("float32"), prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "conditioned_final_depth"})
    _write(paths.conditioned_prov_path, prov_out.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "conditioned_provenance"})

    audit = summarize_precedence_audit(auth=auth, conditioned=conditioned, support=support, provenance=prov_out, guidance_influence=guidance_influence)
    write_precedence_audit(paths.precedence_audit_path, audit, extra={
        "source_authoritative_base": str(paths.auth_src),
        "candidate_input": str(candidate_path) if candidate_path else None,
        "source_aware_candidate": str(paths.source_candidate_path),
        "conditioned_output": str(paths.conditioned_path),
    })
    write_conditioning_audit(
        paths.conditioning_audit_path,
        auth=auth,
        conditioned=conditioned,
        support=support,
        provenance=prov_out,
        guidance_influence=guidance_influence,
        conditioned_uncertainty=conditioned_uncertainty,
        extra={
            "source_authoritative_base": str(paths.auth_src),
            "conditioned_output": str(paths.conditioned_path),
            "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
            "guidance_influence_raster": str(paths.guidance_influence_path),
        },
    )
    contract_outputs = {
        "support_class": str(paths.support_class_path),
        "regime_class": str(paths.regime_class_path),
        "support_distance": str(paths.support_distance_path),
        "support_density": str(paths.support_density_path),
        "anchor_uncertainty": str(paths.anchor_uncertainty_path),
        "guidance_uncertainty": str(paths.guidance_uncertainty_path),
        "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
        "guidance_influence": str(paths.guidance_influence_path),
        "conditioned_depth": str(paths.conditioned_path),
        "conditioned_provenance": str(paths.conditioned_prov_path),
        "coastal_sdb_confidence": str(paths.coastal_sdb_confidence_path),
        "river_anchor_distance": str(paths.river_anchor_distance_path),
        "river_anchor_density": str(paths.river_anchor_density_path),
        "river_scaffold_confidence": str(paths.river_scaffold_confidence_path),
        "river_bank_distance": str(paths.river_bank_distance_path),
        "river_bank_influence": str(paths.river_bank_influence_runtime_path),
        "river_bank_elevation": str(paths.river_bank_elevation_path),
    }
    write_guidance_uncertainty_contract(
        paths.guidance_uncertainty_contract_path,
        outputs=contract_outputs,
        support_note=support_note,
    )
    write_final_output_layer_contract(
        paths.final_output_layer_contract_path,
        outputs=contract_outputs,
    )
    write_final_vertical_semantics_contract(
        paths.final_vertical_semantics_contract_path,
        outputs={**contract_outputs,
            "source_candidate": str(paths.source_candidate_path),
            "river_bank_continuity_weight": str(river_bank_continuity_runtime_path),
            "river_bank_graph_confidence": str(river_bank_graph_confidence_runtime_path),
            "river_bank_confluence_damping": str(river_bank_confluence_damping_runtime_path),
            "river_bank_estuary_side_decay": str(river_bank_estuary_side_decay_runtime_path),
        },
    )

    write_json_receipt(paths.outputs_receipt_path, {
        "stage": "final_route_outputs",
        "route_mode": "staged_final_route_single_source_of_truth",
        "written_outputs": {
            "aligned_authoritative_base": str(paths.aligned_auth_path),
            "conditioned_depth": str(paths.conditioned_path),
            "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
            "conditioned_provenance": str(paths.conditioned_prov_path),
            "precedence_audit": str(paths.precedence_audit_path),
            "conditioning_audit": str(paths.conditioning_audit_path),
            "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
            "final_output_layer_contract": str(paths.final_output_layer_contract_path),
            "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
        },
        "diagnostic_only_outputs": {
            "guidance_surface": str(paths.source_candidate_path),
            "guidance_surface_provenance": str(paths.source_candidate_prov_path),
        },
        "invariants": {
            "continuous_output": True,
            "authoritative_hard_lock": True,
            "diagnostic_dense_surfaces_only": True,
        },
    })

    legacy_cleanup_path = paths.combined_dir / "legacy_cleanup_receipt.json"
    legacy_cleanup = write_legacy_cleanup_receipt(report=report, receipt_path=legacy_cleanup_path)

    final_route_receipt = {
        "route_mode": "staged_final_route_single_source_of_truth",
        "single_authoritative_route_active": True,
        "legacy_parallel_route_retired": True,
        "stage_receipts": {
            "inputs": str(paths.inputs_receipt_path),
            "guidance": str(getattr(guidance, "receipt_path", paths.guidance_receipt_path)),
            "terrain": str(getattr(terrain, "receipt_path", paths.terrain_receipt_path)),
            "outputs": str(paths.outputs_receipt_path),
        },
        "structural_inputs": {
            "authoritative_base": str(paths.auth_src),
            "sdb_guide_points": str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
            "river_guide_points": str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
        },
        "diagnostic_only_inputs": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
            "dense_sdb_depth_raster": str(guidance.sdb_depth_path) if guidance.sdb_depth_path else None,
        },
        "written_outputs": {
            "final_depth": str(paths.conditioned_path),
            "final_uncertainty": str(paths.conditioned_uncertainty_path),
            "final_provenance": str(paths.conditioned_prov_path),
            "precedence_audit": str(paths.precedence_audit_path),
            "conditioning_audit": str(paths.conditioning_audit_path),
            "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
            "final_output_layer_contract": str(paths.final_output_layer_contract_path),
            "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
        },
        "artifacts": {
            "diagnostic_guidance_surface": str(paths.source_candidate_path),
            "diagnostic_guidance_surface_provenance": str(paths.source_candidate_prov_path),
        },
        "manifest_contract": validate_final_route_contract(report),
        "legacy_cleanup_receipt": str(legacy_cleanup_path),
        "legacy_cleanup": legacy_cleanup,
    }
    write_json_receipt(paths.final_route_receipt_path, final_route_receipt)

    report.setdefault("authoritative_base", {})["status"] = "applied"
    report["authoritative_base"]["inputs"] = {"source": str(paths.auth_src)}
    baseline_cudem_interpolation = None
    auth_auto = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}
    cand_baseline = auth_auto.get("baseline_cudem_interpolation")
    if cand_baseline:
        try:
            baseline_path = Path(str(cand_baseline))
        except (TypeError, ValueError, OSError):
            baseline_path = None
        if baseline_path and baseline_path.exists():
            baseline_cudem_interpolation = str(baseline_path)
    if baseline_cudem_interpolation is None:
        try:
            sibling = Path(str(paths.auth_src)).with_name("cudem_baseline_interpolation.tif")
        except (TypeError, ValueError, OSError):
            sibling = None
        if sibling and sibling.exists():
            baseline_cudem_interpolation = str(sibling)

    report["authoritative_base"]["outputs"] = {
        "aligned_authoritative_base": str(paths.aligned_auth_path),
        "baseline_cudem_interpolation": baseline_cudem_interpolation,
        "gap_mask": str(paths.gap_mask_path),
        "eligible_fill_mask": str(paths.eligible_mask_path),
        "support_class": str(paths.support_class_path),
        "regime_class": str(paths.regime_class_path),
        "source_aware_candidate": str(paths.source_candidate_path),
        "source_aware_candidate_provenance": str(paths.source_candidate_prov_path),
        "support_distance": str(paths.support_distance_path),
        "support_density": str(paths.support_density_path),
        "anchor_uncertainty": str(paths.anchor_uncertainty_path),
        "guidance_uncertainty": str(paths.guidance_uncertainty_path),
        "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
        "guidance_influence": str(paths.guidance_influence_path),
        "coastal_sdb_confidence": str(paths.coastal_sdb_confidence_path),
        "river_anchor_distance": str(paths.river_anchor_distance_path),
        "river_anchor_density": str(paths.river_anchor_density_path),
        "river_scaffold_confidence": str(paths.river_scaffold_confidence_path),
        "river_bank_distance": str(paths.river_bank_distance_path),
        "river_bank_influence": str(paths.river_bank_influence_runtime_path),
        "river_bank_elevation": str(paths.river_bank_elevation_path),
        "river_bank_continuity_weight": str(river_bank_continuity_runtime_path),
        "conditioned_depth": str(paths.conditioned_path),
        "conditioned_provenance": str(paths.conditioned_prov_path),
        "source_provenance_input": str(provenance_path) if provenance_path else None,
        "precedence_audit": str(paths.precedence_audit_path),
        "conditioning_audit": str(paths.conditioning_audit_path),
        "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
        "final_output_layer_contract": str(paths.final_output_layer_contract_path),
        "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
        "final_route_receipt": str(paths.final_route_receipt_path),
        "legacy_cleanup_receipt": str(legacy_cleanup_path),
    }
    report["authoritative_base"]["candidate_generation"] = {
        "mode": "staged_final_route_single_source_of_truth",
        "template_path": str(paths.template_path),
        "sdb_depth": str(guidance.sdb_depth_path) if guidance.sdb_depth_path else None,
        "river_depth": None,
        "legacy_candidate": None,
        "diagnostic_only_artifacts": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
            "guidance_surface": str(paths.source_candidate_path),
        },
        "stats": source_candidate.get("stats", {}),
        "backstop_policy": source_candidate.get("backstop_policy", {}),
        "provenance_codes": source_candidate.get("provenance_codes", {}),
    }
    report["final_dem_route"] = final_route_receipt
    report["authoritative_base"]["policy"] = build_final_dem_policy_dict(
        cfg,
        support_note=f"{support_note}; direct_guidance_route=authoritative_base_plus_guidance_artifacts",
        guidance_masks={
            "sdb_admissibility": None,
            "sdb_guidance_weight": None,
            "sdb_trusted_interior": None,
            "river_admissibility": None,
            "river_guidance_weight": None,
            "river_trusted_interior": None,
            "river_authoritative_support": None,
            "river_authoritative_support_depth": None,
        },
    )
    try:
        report["authoritative_base"]["precedence_audit"] = json.loads(paths.precedence_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return paths.conditioned_path, paths.conditioned_prov_path, paths.aligned_auth_path, paths.gap_mask_path, paths.eligible_mask_path, paths.support_class_path
