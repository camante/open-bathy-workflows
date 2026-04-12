from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.json_io import write_json
from workflow_execution_state import WorkflowExecutionState, build_workflow_execution_state


@dataclass
class FinalRunState:
    cfg: Any
    report: Dict[str, Any]
    execution_state: WorkflowExecutionState
    final_native: Optional[Path]
    final_for_user: Optional[Path | str]
    final_provenance: Optional[Path | str]


def build_final_run_state(
    *,
    cfg: Any,
    report: Dict[str, Any],
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> FinalRunState:
    execution_state = build_workflow_execution_state(
        cfg=cfg,
        report=report,
        final_native=final_native,
        final_for_user=final_for_user,
        final_provenance=final_provenance,
    )
    return FinalRunState(
        cfg=cfg,
        report=report,
        execution_state=execution_state,
        final_native=final_native,
        final_for_user=final_for_user,
        final_provenance=final_provenance,
    )


def build_final_run_report(state: FinalRunState) -> Dict[str, Any]:
    report_payload = state.report
    report_payload["workflow_execution_state"] = state.execution_state.to_report_dict()
    return report_payload


def write_final_run_reporting_bundle(
    state: FinalRunState,
    *,
    write_authoritative_cache_receipt: Callable[..., Optional[Path]],
    write_guidance_manifest: Callable[..., Optional[Path]],
    write_explicit_final_outputs_manifest: Callable[..., Optional[Path]],
    write_support_provenance_summary: Callable[..., Optional[Path]],
    write_final_support_regime_audit: Callable[..., Optional[Path]],
    write_final_dem_selection_receipt: Callable[..., Optional[Path]],
    write_comparison_package: Callable[..., Optional[Path]],
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or logging.getLogger(__name__)
    report = build_final_run_report(state)
    minimal_mode = hasattr(state.cfg, "save_intermediates") and (not bool(getattr(state.cfg, "save_intermediates", False)))
    derived_cache_root = getattr(state.cfg, "derived_cache_root", None)
    minimal_report_root = Path(derived_cache_root) if derived_cache_root else (Path(state.cfg.out_dir) / "derived_cache")
    report_path = minimal_report_root / "final_reporting" / "bathy_report.json" if minimal_mode else (Path(state.cfg.out_dir) / "bathy_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)

    receipts = report.setdefault("final_reporting", {}).setdefault("receipts", {})
    receipts["mode"] = "minimal" if minimal_mode else "full"
    if minimal_mode:
        for name in (
            "authoritative_cache_receipt",
            "guidance_manifest",
            "explicit_final_outputs",
            "support_provenance_summary",
            "final_support_regime_audit",
            "final_dem_selection_receipt",
            "comparison_package",
        ):
            receipts[name] = None
        try:
            write_json(report_path, report)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            log.debug("Final report refresh after minimal reporting bundle failed: %s", exc, exc_info=True)
        return report_path

    writer_specs = [
        ("authoritative_cache_receipt", lambda: write_authoritative_cache_receipt(state.cfg, report)),
        ("guidance_manifest", lambda: write_guidance_manifest(state.cfg, report)),
        ("explicit_final_outputs", lambda: write_explicit_final_outputs_manifest(state.cfg, report, final_native=state.final_native, final_for_user=state.final_for_user, final_provenance=state.final_provenance)),
        ("support_provenance_summary", lambda: write_support_provenance_summary(state.cfg, report, final_native=state.final_native, final_for_user=state.final_for_user, final_provenance=state.final_provenance)),
        ("final_support_regime_audit", lambda: write_final_support_regime_audit(state.cfg, report, final_native=state.final_native, final_for_user=state.final_for_user, final_provenance=state.final_provenance)),
        ("final_dem_selection_receipt", lambda: write_final_dem_selection_receipt(state.cfg, report, final_native=state.final_native, final_for_user=state.final_for_user, final_provenance=state.final_provenance)),
        ("comparison_package", lambda: write_comparison_package(state.cfg, report, final_native=state.final_native, final_for_user=state.final_for_user, final_provenance=state.final_provenance)),
    ]
    for name, fn in writer_specs:
        try:
            out = fn()
            receipts[name] = str(out) if out else None
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
            receipts[name] = None
            log.debug("Final reporting step failed [%s]: %s", name, exc, exc_info=True)
    try:
        write_json(report_path, report)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        log.debug("Final report refresh after reporting bundle failed: %s", exc, exc_info=True)
    return report_path
