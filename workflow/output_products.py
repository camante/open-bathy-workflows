from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def _runtime_state(report: Dict[str, Any]) -> Dict[str, Any]:
    state = report.get("final_dem_runtime", {})
    return state if isinstance(state, dict) else {}


def _existing_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    return str(p) if p.exists() else None


def _guidance_manifests(report: Dict[str, Any]) -> Dict[str, Optional[str]]:
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    sdb_artifacts = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    return {
        "sdb_guidance_manifest": _existing_path(sdb_artifacts.get("guidance_manifest")),
        "river_guidance_manifest": _existing_path(river_outputs.get("guidance_manifest")),
        "river_scaffold_manifest": _existing_path(river_outputs.get("scaffold_manifest")),
    }


def build_final_output_contract(
    cfg: Any,
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Dict[str, Any]:
    native_path = Path(final_native) if final_native else None
    user_path = Path(final_for_user) if final_for_user else None
    prov_path = Path(final_provenance) if final_provenance else None

    ab = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    ab_out = ab.get("outputs", {}) if isinstance(ab.get("outputs", {}), dict) else {}
    gapfill = report.get("gapfill", {}) if isinstance(report.get("gapfill", {}), dict) else {}
    gapfill_out = gapfill.get("outputs", {}) if isinstance(gapfill.get("outputs", {}), dict) else {}
    fusion = report.get("fusion", {}) if isinstance(report.get("fusion", {}), dict) else {}
    fusion_out = fusion.get("outputs", {}) if isinstance(fusion.get("outputs", {}), dict) else {}

    conditioned_depth = _existing_path(ab_out.get("conditioned_depth"))
    gapfill_depth = _existing_path(gapfill_out.get("depth"))
    fusion_depth = _existing_path(fusion_out.get("depth"))
    final_native_existing = _existing_path(native_path)
    final_user_existing = _existing_path(user_path)
    final_prov_existing = _existing_path(prov_path)

    selected_native = final_native_existing
    selected_user = final_user_existing
    selected_prov = final_prov_existing

    runtime_state = _runtime_state(report)
    explicit_runtime_engine = runtime_state.get("runtime_engine") if isinstance(runtime_state.get("runtime_engine"), dict) else {}
    explicit_engine_active = explicit_runtime_engine.get("active") if isinstance(explicit_runtime_engine.get("active"), bool) else None
    explicit_engine_module = explicit_runtime_engine.get("module") if isinstance(explicit_runtime_engine.get("module"), str) and explicit_runtime_engine.get("module") else None
    explicit_selected_uses_engine = explicit_runtime_engine.get("selected_output_uses_engine") if isinstance(explicit_runtime_engine.get("selected_output_uses_engine"), bool) else None
    explicit_authoritative_conditioning_top = runtime_state.get("authoritative_conditioning_applied") if isinstance(runtime_state.get("authoritative_conditioning_applied"), bool) else None
    explicit_authoritative_conditioning_nested = explicit_runtime_engine.get("authoritative_conditioning_applied") if isinstance(explicit_runtime_engine.get("authoritative_conditioning_applied"), bool) else None
    explicit_gapfill_applied = runtime_state.get("gapfill_applied") if isinstance(runtime_state.get("gapfill_applied"), bool) else None

    stage = "none"
    route = "none"
    if selected_native and gapfill_depth and Path(selected_native) == Path(gapfill_depth):
        stage = "gapfill"
        route = "support_aware_terrain_interpolator_plus_gapfill"
    elif selected_native and conditioned_depth and Path(selected_native) == Path(conditioned_depth):
        stage = "authoritative_conditioned"
        route = "support_aware_terrain_interpolator"
    elif selected_native and fusion_depth and Path(selected_native) == Path(fusion_depth):
        stage = "fusion"
        route = "legacy_fusion_only"
    elif selected_native:
        stage = "final_native"
        route = "explicit_final_native"

    if selected_user:
        delivery_stage = "user_delivery"
    else:
        delivery_stage = stage

    explicit_route = runtime_state.get("final_generation_route") if isinstance(runtime_state.get("final_generation_route"), str) and runtime_state.get("final_generation_route") else None
    if explicit_route:
        route = explicit_route
    elif explicit_gapfill_applied is True:
        route = "support_aware_terrain_interpolator_plus_gapfill"
    elif explicit_authoritative_conditioning_top is True or explicit_authoritative_conditioning_nested is True:
        route = "support_aware_terrain_interpolator"

    inferred_engine_active = route.startswith("support_aware_terrain_interpolator") or bool(conditioned_depth)
    engine_active = explicit_engine_active if explicit_engine_active is not None else inferred_engine_active
    if explicit_selected_uses_engine is not None:
        selected_output_uses_engine = explicit_selected_uses_engine
    elif selected_native and conditioned_depth and Path(selected_native) == Path(conditioned_depth):
        selected_output_uses_engine = True
    elif selected_native and gapfill_depth and Path(selected_native) == Path(gapfill_depth):
        selected_output_uses_engine = True
    else:
        selected_output_uses_engine = route.startswith("support_aware_terrain_interpolator")
    authoritative_conditioning_applied = (
        explicit_authoritative_conditioning_top
        if explicit_authoritative_conditioning_top is not None
        else explicit_authoritative_conditioning_nested
        if explicit_authoritative_conditioning_nested is not None
        else bool(ab.get("status") == "applied" and conditioned_depth)
    )
    runtime_engine_module = explicit_engine_module if explicit_engine_module is not None else ("terrain_interpolator" if engine_active else None)

    payload: Dict[str, Any] = {
        "selected_final_depth": selected_user or selected_native,
        "selected_final_native": selected_native,
        "selected_final_user": selected_user,
        "selected_final_provenance": selected_prov,
        "selected_final_stage": stage,
        "delivery_stage": delivery_stage,
        "final_generation_route": route,
        "runtime_engine": {
            "module": runtime_engine_module,
            "active": engine_active,
            "selected_output_uses_engine": selected_output_uses_engine,
            "authoritative_conditioning_applied": authoritative_conditioning_applied,
            "gapfill_applied": explicit_gapfill_applied if explicit_gapfill_applied is not None else bool(gapfill.get("status") == "applied" and gapfill_depth),
        },
        "guidance_manifests": _guidance_manifests(report),
        "support_artifacts": {
            "support_class": _existing_path(ab_out.get("support_class")),
            "regime_class": _existing_path(ab_out.get("regime_class")),
            "support_distance": _existing_path(ab_out.get("support_distance")),
            "support_density": _existing_path(ab_out.get("support_density")),
            "guidance_influence": _existing_path(ab_out.get("guidance_influence")),
            "coastal_sdb_confidence": _existing_path(ab_out.get("coastal_sdb_confidence")),
            "river_anchor_distance": _existing_path(ab_out.get("river_anchor_distance")),
            "river_anchor_density": _existing_path(ab_out.get("river_anchor_density")),
            "river_scaffold_confidence": _existing_path(ab_out.get("river_scaffold_confidence")),
            "sdb_regime_class": _existing_path((report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}).get("artifacts", {}).get("regime_class_raster")),
            "river_regime_class": _existing_path((report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}).get("outputs", {}).get("regime_class")),
        },
        "regime_artifacts": {
            "final": _existing_path(ab_out.get("regime_class")),
            "sdb": _existing_path((report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}).get("artifacts", {}).get("regime_class_raster")),
            "river": _existing_path((report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}).get("outputs", {}).get("regime_class")),
        },
        "candidates": {
            "fusion_depth": fusion_depth,
            "authoritative_conditioned_depth": conditioned_depth,
            "gapfill_depth": gapfill_depth,
            "final_depth_native": selected_native,
            "final_depth_user": selected_user,
            "final_provenance_native": selected_prov,
        },
        "authoritative_policy": {
            "hard_lock_finite_authoritative_cells": True,
            "continuous_output": True,
        },
    }
    return payload


__all__ = ["build_final_output_contract"]
