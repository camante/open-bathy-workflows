from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _postrun_outputs_requested(args, cfg) -> bool:
    seam_ios = list(getattr(args, "seam_compare_with_io", []) or [])
    seam_list = getattr(args, "seam_compare_with_io_list", None)
    nested_ios = list(getattr(args, "nested_aoi_compare_with_final_outputs", []) or [])
    nested_list = getattr(args, "nested_aoi_compare_with_final_outputs_list", None)
    return bool(
        getattr(cfg, "save_intermediates", False)
        or seam_ios
        or seam_list
        or nested_ios
        or nested_list
    )

from final_postrun_contract import build_final_postrun_context
from postrun_output_stage import run_postrun_output_checks, write_final_output_bundle


def execute_final_run_stage(*, cfg, args, report: Dict[str, Any], log, fatal_errors: List[str],
                            sdb_raster, river_raster, river_for_fuse, river_excluded,
                            fuse_fn, condition_fn, reproject_fn, write_bundle_fn,
                            finalize_run_fn, write_io_manifest_fn, emit_artifacts_fn,
                            run_seam_comparisons_fn, conditioned_gapfill_fn):
    final = None
    conditioned_prov = None
    final_provenance = None
    authoritative_aligned = None
    authoritative_gap_mask = None
    authoritative_eligible_fill_mask = None
    support_class = None

    if sdb_raster is None and river_for_fuse is None:
        report["fusion"] = {
            "status": "skipped",
            "reason": "no_fusable_sources",
            "note": "Both sources missing or river excluded by constraint guardrail.",
            "river_excluded": river_excluded,
        }
    else:
        final = fuse_fn(cfg, sdb_raster, river_for_fuse, report)
        if river_excluded is not None:
            report.setdefault("fusion", {}).setdefault("notes", [])
            report["fusion"]["notes"].append(river_excluded)

    if final:
        final, conditioned_prov, authoritative_aligned, authoritative_gap_mask, authoritative_eligible_fill_mask, support_class = condition_fn(
            cfg, Path(final), Path(report.get("fusion", {}).get("outputs", {}).get("provenance")) if report.get("fusion", {}).get("outputs", {}).get("provenance") else None, report
        )
        final_provenance = conditioned_prov

    if cfg.gapfill_enabled and final:
        final, final_provenance = conditioned_gapfill_fn(
            cfg=cfg, args=args, report=report, log=log, final=final, final_provenance=final_provenance,
            river_raster=river_raster, authoritative_aligned=authoritative_aligned,
            authoritative_eligible_fill_mask=authoritative_eligible_fill_mask,
        )
        if final:
            final, conditioned_prov, authoritative_aligned, authoritative_gap_mask, authoritative_eligible_fill_mask, support_class = condition_fn(
                cfg,
                Path(final),
                Path(final_provenance) if final_provenance else None,
                report,
            )
            final_provenance = conditioned_prov or final_provenance
            report.setdefault("final_dem_runtime", {})["post_gapfill_reconditioned"] = True

    if final_provenance is None:
        final_provenance = conditioned_prov or report.get("fusion", {}).get("outputs", {}).get("provenance")

    final_native_path = Path(final) if final else None
    authoritative_conditioning_applied = bool(report.get("authoritative_base", {}).get("status") == "applied")
    gapfill_applied = bool(cfg.gapfill_enabled and report.get("gapfill", {}).get("status") == "applied")
    final_generation_route = "none"
    if final_native_path is not None:
        if gapfill_applied:
            final_generation_route = "support_aware_terrain_interpolator_plus_gapfill"
        elif authoritative_conditioning_applied:
            final_generation_route = "support_aware_terrain_interpolator"
        else:
            final_generation_route = "legacy_fusion_only"
    runtime_engine = {
        "module": "terrain_interpolator" if authoritative_conditioning_applied else None,
        "active": bool(authoritative_conditioning_applied),
        "selected_output_uses_engine": bool(authoritative_conditioning_applied),
        "authoritative_conditioning_applied": authoritative_conditioning_applied,
        "gapfill_applied": gapfill_applied,
    }
    report.setdefault("final_dem_runtime", {}).update({
        "final_native_candidate": str(final_native_path) if final_native_path else None,
        "final_provenance_candidate": str(final_provenance) if final_provenance else None,
        "authoritative_conditioning_applied": authoritative_conditioning_applied,
        "gapfill_applied": gapfill_applied,
        "final_generation_route": final_generation_route,
        "runtime_engine": runtime_engine,
    })
    final_for_user = reproject_fn(cfg, final, report, sdb_raster, river_raster, fatal_errors)
    postrun_context = build_final_postrun_context(
        cfg=cfg,
        args=args,
        report=report,
        run_id=report.get("run", {}).get("run_id") or "unknown",
        final_native=final_native_path,
        final_for_user=final_for_user,
        final_provenance=final_provenance,
        fatal_errors=fatal_errors,
    )
    if _postrun_outputs_requested(args, cfg):
        report_path = write_final_output_bundle(
            context=postrun_context,
            logger=log,
            write_bundle_fn=write_bundle_fn,
            write_io_manifest_fn=write_io_manifest_fn,
            emit_artifacts_fn=emit_artifacts_fn,
        )

        run_postrun_output_checks(
            context=postrun_context,
            run_seam_comparisons_fn=run_seam_comparisons_fn,
        )
    else:
        report_path = write_final_output_bundle(
            context=postrun_context,
            logger=log,
            write_bundle_fn=write_bundle_fn,
            write_io_manifest_fn=write_io_manifest_fn,
            emit_artifacts_fn=lambda *_args, **_kwargs: None,
        )
    return finalize_run_fn(cfg, report, args, final, final_for_user, final_provenance, fatal_errors)
