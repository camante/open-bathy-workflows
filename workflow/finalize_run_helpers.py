
from __future__ import annotations

import json
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any


def write_review_only_run_summary_files(*, cfg: Any, report: dict, run_id: str, logger) -> None:
    from reporting.run_summary import write_run_summary_files
    from core.flight_recorder import current_flight_path

    frp = current_flight_path()
    write_run_summary_files(
        cfg.out_dir,
        run_id=run_id,
        stats=report,
        fr_path=(frp or None),
        write_machine_json=bool(getattr(cfg, "write_legacy_run_summary_json", False)),
        write_legacy_text_summaries=bool(getattr(cfg, "write_legacy_text_summaries", False)),
    )
    logger.info("[DOMAIN] Review-only run summaries written under: %s", Path(cfg.out_dir) / "run_logs")


def build_and_log_human_summary(*, cfg: Any, report: dict, logger) -> None:
    from reporting.run_summary import print_human_run_summary

    summary_stats = {
        "command": " ".join(sys.argv),
        "aoi": cfg.aoi,
        "time_window": {"start": cfg.start_date, "end": cfg.end_date},
        "methods": list(cfg.methods),
        "priority": cfg.priority,
        "outputs": report.get("outputs", {}),
        # Keep the human summary aligned with the route that actually ran.
        # These are reporting-only fields; they do not alter workflow routing.
        "sdb": report.get("sdb", {}),
        "river": report.get("river", {}),
        "fusion": report.get("fusion", {}),
        "river_workflow": report.get("river_workflow", {}),
        "active_river": report.get("active_river", {}),
        "seamless_dem": report.get("seamless_dem", {}),
        "route": report.get("route", {}),
        "guidance_domains": report.get("guidance_domains", {}),
    }
    print_human_run_summary(summary_stats, log_fn=logger.info)
    report["human_summary"] = summary_stats


def apply_energy_solver_top_level_summary(*, cfg: Any, report: dict, logger) -> None:
    from core.flight_recorder import current_run_id

    rid = current_run_id()
    meta_p = Path(cfg.out_dir) / "derived_cache" / str(rid) / "river" / "work" / "xs_mainstem_constraints_meta.json"
    if not meta_p.exists():
        cand = sorted((Path(cfg.out_dir) / "derived_cache").glob("*/river/work/xs_mainstem_constraints_meta.json"))
        meta_p = cand[-1] if cand else meta_p
    if not meta_p.exists():
        return

    m = json.loads(meta_p.read_text(encoding="utf-8"))
    es = m.get("energy_solver", {}) if isinstance(m, dict) else {}
    if not es and isinstance(m, dict):
        es = {
            "enabled": m.get("energy_solver_enabled"),
            "reason": m.get("energy_solver_reason"),
            "n_total": m.get("energy_solver_n_total"),
            "n_applied": m.get("energy_solver_n_applied"),
            "blend_n": m.get("energy_solver_blend_n"),
            "changed_n": m.get("energy_solver_changed_n"),
            "max_run": m.get("energy_solver_changed_max_run"),
            "wse_source": m.get("energy_solver_wse_source"),
            "wse_anchor": m.get("wse_anchor_source"),
            "delta_median": m.get("energy_solver_delta_dmax_m_median"),
            "delta_p95": m.get("energy_solver_delta_dmax_m_p95"),
        }

    logger.info(
        "[ENERGY][SUMMARY] enabled=%s reason=%s n_total=%s n_applied=%s blend_n=%s changed_n=%s max_run=%s wse_source=%s wse_anchor=%s |delta_dmax| median=%s p95=%s (meta=%s)",
        es.get("enabled"), es.get("reason"), es.get("n_total"), es.get("n_applied"),
        es.get("blend_n"), es.get("changed_n"), es.get("max_run"),
        es.get("wse_source"), es.get("wse_anchor"),
        es.get("delta_dmax_m_median", es.get("delta_median")),
        es.get("delta_dmax_m_p95", es.get("delta_p95")),
        str(meta_p),
    )
    report.setdefault("river", {}).setdefault("energy_solver", {}).update(es)


def run_runtime_sign_semantics(*, cfg: Any, report: dict, logger) -> None:
    from contracts_sign_semantics_runtime import (
        run_sign_semantics_runtime_contracts,
        run_sign_semantics_stage_contracts,
    )

    semantic_suite = run_sign_semantics_runtime_contracts(cfg, report, contracts_dir=Path(cfg.out_dir) / "contracts")
    logger.info(
        "[CONTRACT][SIGN_SEMANTICS] pass=%s warn=%s fail=%s",
        semantic_suite.get("pass"), semantic_suite.get("warn"), semantic_suite.get("fail"),
    )
    stage_suite = run_sign_semantics_stage_contracts(cfg, report, contracts_dir=Path(cfg.out_dir) / "contracts")
    logger.info(
        "[CONTRACT][SIGN_SEMANTICS_STAGE] pass=%s warn=%s fail=%s",
        stage_suite.get("pass"), stage_suite.get("warn"), stage_suite.get("fail"),
    )
    if cfg.strict and (int(semantic_suite.get("fail", 0)) > 0 or int(stage_suite.get("fail", 0)) > 0):
        raise RuntimeError("Runtime sign semantics contracts failed")


def write_detailed_run_summaries(*, cfg: Any, report: dict, logger) -> None:
    from reporting.run_summary import write_run_summary_files
    from core.flight_recorder import current_run_id, current_flight_path

    rid = current_run_id()
    frp = current_flight_path()
    write_run_summary_files(
        cfg.out_dir,
        run_id=rid,
        stats=report,
        fr_path=(frp or None),
        write_machine_json=bool(getattr(cfg, "write_legacy_run_summary_json", False)),
        write_legacy_text_summaries=bool(getattr(cfg, "write_legacy_text_summaries", False)),
    )
    logger.info("Run summaries written under: %s", Path(cfg.out_dir) / "run_logs")


def apply_output_retention_policy_best_effort(*, cfg: Any, logger, report: dict, final_path, final_for_user_path) -> None:
    from bathy_main import _apply_output_retention_policy  # late import to avoid circular startup impact

    _apply_output_retention_policy(cfg, logger, report, final_path=final_path, final_for_user_path=final_for_user_path)


NONFATAL_FINALIZE_EXCEPTIONS = (ImportError, ModuleNotFoundError, OSError, ValueError, TypeError, RuntimeError, JSONDecodeError)


def write_workflow_actual_trace_file(*, out_dir: str | Path, report: dict, logger) -> str:
    from tools.debug.workflow_actual_trace import write_workflow_actual_trace

    path = write_workflow_actual_trace(out_dir=out_dir, report=report)
    logger.info("Workflow input/output trace written: %s", path)
    return str(path)


def write_connected_diagnostics_files(*, out_dir: str | Path, report: dict, logger) -> dict[str, str]:
    from tools.debug.workflow_connected_diagnostics import write_connected_diagnostics

    outputs = write_connected_diagnostics(out_dir=out_dir, report=report)
    logger.info("Connected workflow diagnostics written: %s", outputs)
    return outputs


def write_run_diagnosis_files(*, out_dir: str | Path, report: dict, logger) -> dict[str, str]:
    from tools.debug.workflow_run_diagnosis import write_run_diagnosis

    outputs = write_run_diagnosis(out_dir=out_dir, report=report)
    logger.info("Workflow run diagnosis written: %s", outputs)
    return outputs




def write_curated_reports_suite(*, out_dir: str | Path, report: dict, logger) -> dict[str, str]:
    outputs: dict[str, str] = {}
    trace_path = write_workflow_actual_trace_file(out_dir=out_dir, report=report, logger=logger)
    outputs["workflow_input_output_trace"] = str(trace_path)
    outputs.update(write_connected_diagnostics_files(out_dir=out_dir, report=report, logger=logger))
    outputs.update(write_run_diagnosis_files(out_dir=out_dir, report=report, logger=logger))
    outputs.update(write_reports_hub_files(out_dir=out_dir, report=report, logger=logger))
    report.setdefault("outputs", {}).update(outputs)
    report.setdefault("final_reporting", {}).setdefault("receipts", {}).update(outputs)
    return outputs


def refresh_bathy_report_output(*, report: dict, logger) -> str | None:
    from core.json_io import write_json

    outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    report_path = outputs.get("bathy_report")
    if not isinstance(report_path, str) or not report_path.strip():
        return None
    try:
        write_json(Path(report_path), report)
        logger.info("Final report refreshed: %s", report_path)
        return report_path
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Final report refresh skipped: %s", exc, exc_info=True)
        return None

def write_reports_hub_files(*, out_dir: str | Path, report: dict, logger) -> dict[str, str]:
    from tools.debug.workflow_reports_hub import write_reports_hub

    outputs = write_reports_hub(out_dir=out_dir, report=report)
    logger.info("Reports hub written: %s", outputs)
    return outputs
