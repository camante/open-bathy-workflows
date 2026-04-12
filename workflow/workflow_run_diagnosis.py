from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from workflow_connected_diagnostics import build_connected_diagnostics_payload

REPORTS_DIRNAME = "reports"
FIRST_BAD_JSON = "workflow_first_bad_artifact_summary.json"
ACCURACY_JSON = "workflow_accuracy_anomalies.json"
RUN_DIAGNOSIS_JSON = "workflow_run_diagnosis_summary.json"
RUN_DIAGNOSIS_TXT = "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"


STAGE_ORDER = {
    "workflow_execution_state": 0,
    "authoritative_base": 10,
    "authoritative_base_auto": 11,
    "shared_domain_stage": 20,
    "guidance_domains": 21,
    "river": 30,
    "sdb": 31,
    "fusion": 40,
    "final_dem_route": 50,
    "final_dem_runtime": 60,
    "benchmark": 70,
    "final_reporting": 80,
    "outputs": 90,
}
SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "critical": 3}
ACTIVE_STAGE_STATUSES = {"ran", "partial", "success", "applied"}
TERMINAL_ORPHAN_STAGE_NAMES = {"workflow_execution_state", "final_reporting", "outputs"}


def _stage_is_active(status: Any) -> bool:
    return str(status or "").strip().lower() in ACTIVE_STAGE_STATUSES


def _is_deprioritized_orphan_anomaly(anomaly: dict[str, Any]) -> bool:
    anomaly_id = str(anomaly.get("anomaly_id") or "")
    stage_name = str(anomaly.get("stage_name") or "")
    if not anomaly_id.startswith("stage_orphan_outputs::"):
        return False
    return stage_name in TERMINAL_ORPHAN_STAGE_NAMES


def _first_bad_rank(anomaly: dict[str, Any], stage_order: dict[str, int]) -> tuple[int, int, int, str]:
    deprioritized = 1 if _is_deprioritized_orphan_anomaly(anomaly) else 0
    return (
        deprioritized,
        stage_order.get(str(anomaly.get("stage_name")), STAGE_ORDER.get(str(anomaly.get("stage_name")), 9999)),
        -SEVERITY_RANK.get(str(anomaly.get("severity")), -1),
        str(anomaly.get("anomaly_id")),
    )


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _receipt_path(report: dict[str, Any], *keys: str) -> str | None:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, str) else None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _add_anomaly(anomalies: list[dict[str, Any]], *, anomaly_id: str, severity: str, stage_name: str,
                 metric_name: str, observed_value: Any, threshold: Any, why_it_matters: str,
                 likely_stage: str, recommended_fix_target: str, artifact_path: str | None = None,
                 reason: str | None = None) -> None:
    anomalies.append({
        "anomaly_id": anomaly_id,
        "severity": severity,
        "stage_name": stage_name,
        "metric_name": metric_name,
        "observed_value": observed_value,
        "threshold": threshold,
        "why_it_matters": why_it_matters,
        "likely_stage": likely_stage,
        "recommended_fix_target": recommended_fix_target,
        "artifact_path": artifact_path,
        "reason": reason,
    })


def _collect_accuracy_anomalies(*, report: dict[str, Any], out_dir: Path, stage_trace: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    top_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river"), dict) else {}
    benchmark = report.get("benchmark", {}) if isinstance(report.get("benchmark"), dict) else {}

    for stage in stage_trace or []:
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage_name") or "unknown")
        status = str(stage.get("status") or "").strip()
        reason = str(stage.get("reason") or "").strip() or None
        output_files = stage.get("output_files") if isinstance(stage.get("output_files"), list) else []
        missing_outputs = stage.get("missing_output_files") if isinstance(stage.get("missing_output_files"), list) else []
        orphan_outputs = stage.get("orphan_output_files") if isinstance(stage.get("orphan_output_files"), list) else []
        stage_artifact_path = None
        if missing_outputs:
            stage_artifact_path = (missing_outputs[0] or {}).get("path")
        elif output_files:
            stage_artifact_path = (output_files[0] or {}).get("path")
        if status == "failed":
            _add_anomaly(
                anomalies,
                anomaly_id=f"stage_failed::{stage_name}",
                severity="critical",
                stage_name=stage_name,
                metric_name="stage_status",
                observed_value=status,
                threshold="ran",
                why_it_matters="A stage marked as failed in the connected stage trace produced an incomplete or invalid workflow branch before downstream interpretation.",
                likely_stage=stage_name,
                recommended_fix_target=str(stage.get("module") or stage_name),
                artifact_path=stage_artifact_path,
                reason=reason,
            )
        elif status == "partial":
            _add_anomaly(
                anomalies,
                anomaly_id=f"stage_partial::{stage_name}",
                severity="warning",
                stage_name=stage_name,
                metric_name="stage_status",
                observed_value=status,
                threshold="ran",
                why_it_matters="A stage only partially engaged, so downstream artifacts may exist without the intended effect or full coverage.",
                likely_stage=stage_name,
                recommended_fix_target=str(stage.get("module") or stage_name),
                artifact_path=stage_artifact_path,
                reason=reason,
            )
        missing_count = int(stage.get("missing_output_count", 0) or 0)
        if missing_count > 0:
            _add_anomaly(
                anomalies,
                anomaly_id=f"stage_missing_outputs::{stage_name}",
                severity="critical" if status == "ran" else "error",
                stage_name=stage_name,
                metric_name="missing_output_count",
                observed_value=missing_count,
                threshold=0,
                why_it_matters="The stage reported output paths that do not exist on disk, which usually means the producer write step failed or the report references outputs that were never created.",
                likely_stage=stage_name,
                recommended_fix_target=str(stage.get("module") or stage_name),
                artifact_path=(missing_outputs[0] or {}).get("path") if missing_outputs else stage_artifact_path,
                reason=reason or "stage trace reported missing output files",
            )
        orphan_count = int(stage.get("orphan_output_count", 0) or 0)
        if orphan_count > 0:
            _add_anomaly(
                anomalies,
                anomaly_id=f"stage_orphan_outputs::{stage_name}",
                severity="info",
                stage_name=stage_name,
                metric_name="orphan_output_count",
                observed_value=orphan_count,
                threshold=0,
                why_it_matters="The stage wrote outputs that no later workflow stage consumed. These may be valid terminal products, but they can also indicate disconnected diagnostics or intermediates that never influence the final result.",
                likely_stage=stage_name,
                recommended_fix_target=str(stage.get("module") or stage_name),
                artifact_path=(orphan_outputs[0] or {}).get("path") if orphan_outputs else stage_artifact_path,
                reason=reason or "stage trace reported orphan outputs with no downstream consumers",
            )

    if benchmark.get("status") == "skipped":
        _add_anomaly(
            anomalies,
            anomaly_id="benchmark_skipped",
            severity="info",
            stage_name="benchmark",
            metric_name="benchmark_status",
            observed_value="skipped",
            threshold="requested",
            why_it_matters="Benchmark outputs are unavailable when holdout benchmarking was not requested, so downstream evaluation files may be absent by design.",
            likely_stage="benchmark",
            recommended_fix_target="postrun_benchmark_stage.py",
            artifact_path=None,
            reason=str(benchmark.get("reason") or "no_holdout_flag"),
        )

    effect_path = _receipt_path(report, "river", "outputs", "channel_surface_effect_summary")
    effect = _load_json(effect_path)
    if effect:
        thalweg_components = int(effect.get("selected_for_thalweg_render_component_count", 0) or 0)
        candidate_pixels = int(effect.get("candidate_pixel_count", 0) or 0)
        final_changed = int(effect.get("final_changed_pixel_count", 0) or 0)
        if thalweg_components > 0 and candidate_pixels > 0 and final_changed == 0:
            _add_anomaly(
                anomalies,
                anomaly_id="thalweg_render_inert",
                severity="error",
                stage_name="river",
                metric_name="thalweg_render_final_changed_pixel_count",
                observed_value=final_changed,
                threshold=f">0 when selected_for_thalweg_render_component_count={thalweg_components} and candidate_pixel_count={candidate_pixels}",
                why_it_matters="Weak-support components entered thalweg-dominant rendering, but the final raster did not change relative to the fallback comparison surface. This means the thalweg-led render was selected but effectively inert.",
                likely_stage="river_channel_surface",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=effect_path,
                reason="thalweg render had finite candidate pixels but no final changed pixels",
            )
        section_geom = int(effect.get("section_target_geometry_count", 0) or 0)
        section_applied = int(effect.get("section_target_applied_pixel_count", 0) or 0)
        if section_geom > 0 and section_applied == 0:
            _add_anomaly(
                anomalies,
                anomaly_id="section_target_inert",
                severity="warning",
                stage_name="river",
                metric_name="section_target_applied_pixel_count",
                observed_value=section_applied,
                threshold=f">0 when section_target_geometry_count={section_geom}",
                why_it_matters="The renderer had usable section-target geometry, but section-target influence never applied to the final channel surface. This can leave the weak-support river shape underconstrained laterally.",
                likely_stage="river_channel_surface",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=effect_path,
                reason="section-target geometry was present but no section-target pixels were applied",
            )

    transition_path = _receipt_path(report, "river", "outputs", "channel_surface_authoritative_transition_summary")
    transition = _load_json(transition_path)
    if transition:
        candidate = int(transition.get("candidate_cell_count", 0) or 0)
        nonzero = int(transition.get("nonzero_cell_count", 0) or 0)
        if candidate > 0 and nonzero == 0:
            _add_anomaly(
                anomalies,
                anomaly_id="authoritative_transition_inert",
                severity="critical",
                stage_name="river",
                metric_name="transition_nonzero_cell_count",
                observed_value=nonzero,
                threshold=f">0 when candidate_cell_count={candidate}",
                why_it_matters="The workflow found transition candidates near authoritative support, but zero final transition weights means the taper never affected the rendered river surface.",
                likely_stage="river_channel_surface",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=transition_path,
                reason="candidate cells exist but final transition weights remained zero",
            )

    backbone_path = _receipt_path(report, "river", "outputs", "backbone_smoothing_summary")
    backbone = _load_json(backbone_path)
    if backbone:
        adjusted = int(backbone.get("adjusted_station_count", 0) or 0)
        candidate = int(backbone.get("candidate_station_count", 0) or 0)
        weight_mean = _coerce_float(((backbone.get("weight_summary") or {}) if isinstance(backbone.get("weight_summary"), dict) else {}).get("mean"))
        if adjusted == 0 and ((candidate > 0) or ((weight_mean or 0.0) > 0.0)):
            _add_anomaly(
                anomalies,
                anomaly_id="backbone_smoothing_inert",
                severity="error",
                stage_name="river",
                metric_name="adjusted_station_count",
                observed_value=adjusted,
                threshold=">0 when candidate stations or smoothing weights are present",
                why_it_matters="Longitudinal scaffold smoothing is configured but not moving any stations, so centerline coherence improvements are being deferred to later raster-space smoothing instead of the upstream backbone.",
                likely_stage="river_primary_surface_rebuild",
                recommended_fix_target="river_primary_surface_rebuild.py",
                artifact_path=backbone_path,
                reason="candidate/weighted smoothing present but no stations were adjusted",
            )

    long_path = _receipt_path(report, "river", "outputs", "channel_surface_longitudinal_smoothing_summary")
    longitudinal = _load_json(long_path)
    if longitudinal:
        eligible = int(longitudinal.get("eligible_count", 0) or 0)
        changed = int(longitudinal.get("changed_count", 0) or 0)
        if eligible > 0 and changed == 0:
            _add_anomaly(
                anomalies,
                anomaly_id="rendered_longitudinal_smoothing_inert",
                severity="warning",
                stage_name="river",
                metric_name="longitudinal_smoothing_changed_count",
                observed_value=changed,
                threshold=f">0 when eligible_count={eligible}",
                why_it_matters="Rendered longitudinal smoothing found eligible weak-support pixels but did not change the channel surface, suggesting the pass is being clamped away or neutralized downstream.",
                likely_stage="river_channel_surface",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=long_path,
                reason=str(longitudinal.get("reason") or "eligible smoothing produced no final changes"),
            )

    triage_path = top_outputs.get("benchmark_river_receipt_triage_summary_json") if isinstance(top_outputs, dict) else None
    triage = _load_json(triage_path)
    if triage:
        centerline = (triage.get("receipts") or {}).get("centerline_agreement") if isinstance(triage.get("receipts"), dict) else None
        if isinstance(centerline, dict):
            roughness = _coerce_float(centerline.get("roughness_ratio_p95_abs"))
            if roughness is not None and roughness > 3.0:
                _add_anomaly(
                    anomalies,
                    anomaly_id="centerline_roughness_high",
                    severity="error",
                    stage_name="benchmark",
                    metric_name="centerline_roughness_ratio_p95_abs",
                    observed_value=roughness,
                    threshold="<=3.0",
                    why_it_matters="The final river surface is still much rougher than the reconciled centerline/backbone target, indicating that longitudinal coherence remains poor in unsupported reaches.",
                    likely_stage="benchmark/river",
                    recommended_fix_target="river_primary_surface_rebuild.py or river_channel_surface.py",
                    artifact_path=triage_path,
                    reason="centerline agreement receipt reports elevated roughness",
                )
        role = (triage.get("receipts") or {}).get("role_agreement") if isinstance(triage.get("receipts"), dict) else None
        if isinstance(role, dict) and role.get("available") is False:
            _add_anomaly(
                anomalies,
                anomaly_id="role_agreement_unavailable",
                severity="warning",
                stage_name="river",
                metric_name="role_agreement_available",
                observed_value=False,
                threshold=True,
                why_it_matters="Without role agreement diagnostics the workflow cannot distinguish thalweg, inner-shape, and bank-edge mismatch, making river errors harder to localize.",
                likely_stage="river_channel_surface diagnostics",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=_receipt_path(report, "river", "outputs", "channel_surface_role_agreement_summary"),
                reason=str(role.get("reason") or "role agreement receipt unavailable"),
            )
        sec = (triage.get("receipts") or {}).get("section_target_agreement") if isinstance(triage.get("receipts"), dict) else None
        if isinstance(sec, dict) and sec.get("available") is False:
            _add_anomaly(
                anomalies,
                anomaly_id="section_target_agreement_unavailable",
                severity="warning",
                stage_name="river",
                metric_name="section_target_agreement_available",
                observed_value=False,
                threshold=True,
                why_it_matters="Without section-target agreement diagnostics the workflow cannot confirm whether the final surface matches the target section geometry it used during rendering.",
                likely_stage="river_channel_surface diagnostics",
                recommended_fix_target="river_channel_surface.py",
                artifact_path=_receipt_path(report, "river", "outputs", "channel_surface_section_target_agreement_summary"),
                reason=str(sec.get("reason") or "section-target agreement receipt unavailable"),
            )

    focus_path = top_outputs.get("benchmark_river_primary_focus_summary_json") if isinstance(top_outputs, dict) else None
    focus = _load_json(focus_path)
    if focus and focus.get("hard_problem_holdout_blind") is True:
        _add_anomaly(
            anomalies,
            anomaly_id="unsupported_river_holdout_blind",
            severity="warning",
            stage_name="benchmark",
            metric_name="hard_problem_holdout_blind",
            observed_value=True,
            threshold=False,
            why_it_matters="Classic holdout metrics are not measuring unsupported-river quality in this run, so support-aware river diagnostics should take precedence over holdout accuracy tables.",
            likely_stage="benchmark",
            recommended_fix_target="benchmark_workflow_stage.py",
            artifact_path=focus_path,
            reason=str(focus.get("hard_problem_blind_reason") or "holdout does not cover unsupported river"),
        )

    return anomalies


def _choose_first_bad_artifact(*, anomalies: list[dict[str, Any]], stage_trace: list[dict[str, Any]]) -> dict[str, Any]:
    stage_order = {row.get("stage_name"): int(row.get("order", STAGE_ORDER.get(str(row.get("stage_name")), 9999))) for row in stage_trace if isinstance(row, dict)}
    if not anomalies:
        return {
            "available": False,
            "final_problem_detected": False,
            "first_bad_stage": None,
            "first_bad_artifact": None,
            "failure_mode": None,
            "why_flagged": "No workflow anomalies were detected by the connected diagnosis layer.",
            "recommended_fix_module": None,
        }

    ranked = sorted(anomalies, key=lambda a: _first_bad_rank(a, stage_order))
    chosen = ranked[0]
    return {
        "available": True,
        "final_problem_detected": True,
        "first_bad_stage": chosen.get("stage_name"),
        "first_bad_artifact": chosen.get("artifact_path"),
        "failure_mode": chosen.get("anomaly_id"),
        "why_flagged": chosen.get("why_it_matters"),
        "recommended_fix_module": chosen.get("recommended_fix_target"),
        "reason": chosen.get("reason"),
    }


def _summarize_run(*, report: dict[str, Any], stage_trace: list[dict[str, Any]], anomalies: list[dict[str, Any]], first_bad: dict[str, Any]) -> dict[str, Any]:
    active = [row.get("stage_name") for row in stage_trace if _stage_is_active(row.get("status"))]
    inert = [a.get("anomaly_id") for a in anomalies if a.get("severity") in {"error", "critical"}]
    benchmark = report.get("benchmark", {}) if isinstance(report.get("benchmark"), dict) else {}
    top3 = [
        {
            "anomaly_id": a.get("anomaly_id"),
            "stage_name": a.get("stage_name"),
            "severity": a.get("severity"),
            "artifact_path": a.get("artifact_path"),
            "recommended_fix_target": a.get("recommended_fix_target"),
        }
        for a in sorted(anomalies, key=lambda a: -SEVERITY_RANK.get(str(a.get("severity")), -1))[:3]
    ]
    return {
        "available": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_requested": bool(benchmark.get("requested", False)),
        "benchmark_status": benchmark.get("status") if isinstance(benchmark, dict) else None,
        "primary_active_branch": "river" if "river" in active else ("sdb" if "sdb" in active else None),
        "stages_ran": active,
        "stages_with_major_anomalies": inert,
        "first_bad_artifact": first_bad,
        "top_suspected_causes": top3,
        "recommended_next_modules": sorted({str(a.get("recommended_fix_target")) for a in anomalies if a.get("recommended_fix_target")}),
        "supporting_receipts": sorted({str(a.get("artifact_path")) for a in anomalies if a.get("artifact_path")}),
    }


def _render_run_summary_text(*, summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> str:
    lines = [
        "WORKFLOW RUN DIAGNOSIS SUMMARY",
        f"created_utc: {summary.get('created_utc')}",
        f"benchmark_requested: {summary.get('benchmark_requested')}",
        f"benchmark_status: {summary.get('benchmark_status')}",
        f"primary_active_branch: {summary.get('primary_active_branch')}",
        "",
        "FIRST BAD ARTIFACT",
    ]
    first_bad = summary.get("first_bad_artifact") or {}
    lines.extend([
        f"- stage: {first_bad.get('first_bad_stage')}",
        f"- artifact: {first_bad.get('first_bad_artifact')}",
        f"- failure_mode: {first_bad.get('failure_mode')}",
        f"- why_flagged: {first_bad.get('why_flagged')}",
        f"- recommended_fix_module: {first_bad.get('recommended_fix_module')}",
        "",
        "TOP SUSPECTED CAUSES",
    ])
    for item in summary.get("top_suspected_causes", []):
        lines.append(
            f"- {item.get('anomaly_id')} [{item.get('severity')}] stage={item.get('stage_name')} artifact={item.get('artifact_path')} fix={item.get('recommended_fix_target')}"
        )
    lines.append("")
    lines.append("ACCURACY / WORKFLOW ANOMALIES")
    if not anomalies:
        lines.append("- none detected")
    for anomaly in anomalies:
        lines.append(f"- {anomaly.get('anomaly_id')} [{anomaly.get('severity')}] stage={anomaly.get('stage_name')}")
        lines.append(f"  metric: {anomaly.get('metric_name')} observed={anomaly.get('observed_value')} threshold={anomaly.get('threshold')}")
        lines.append(f"  why: {anomaly.get('why_it_matters')}")
        if anomaly.get("reason"):
            lines.append(f"  reason: {anomaly.get('reason')}")
        if anomaly.get("artifact_path"):
            lines.append(f"  artifact: {anomaly.get('artifact_path')}")
        lines.append(f"  fix_target: {anomaly.get('recommended_fix_target')}")
    lines.append("")
    lines.append("FIRST FILES TO OPEN")
    for path in summary.get("supporting_receipts", [])[:8]:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"


def write_run_diagnosis(*, out_dir: str | Path, report: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    reports_dir = out_dir / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    stage_payload = build_connected_diagnostics_payload(out_dir=out_dir, report=report)
    stage_trace = stage_payload.get("stage_trace", []) if isinstance(stage_payload, dict) else []
    anomalies = _collect_accuracy_anomalies(report=report, out_dir=out_dir, stage_trace=stage_trace)
    first_bad = _choose_first_bad_artifact(anomalies=anomalies, stage_trace=stage_trace)
    summary = _summarize_run(report=report, stage_trace=stage_trace, anomalies=anomalies, first_bad=first_bad)

    accuracy_path = reports_dir / ACCURACY_JSON
    accuracy_path.write_text(json.dumps(anomalies, indent=2, sort_keys=True), encoding="utf-8")

    first_bad_path = reports_dir / FIRST_BAD_JSON
    first_bad_path.write_text(json.dumps(first_bad, indent=2, sort_keys=True), encoding="utf-8")

    summary_path = reports_dir / RUN_DIAGNOSIS_JSON
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    summary_txt_path = reports_dir / RUN_DIAGNOSIS_TXT
    summary_txt_path.write_text(_render_run_summary_text(summary=summary, anomalies=anomalies), encoding="utf-8")

    outputs = {
        "workflow_accuracy_anomalies_json": str(accuracy_path),
        "workflow_first_bad_artifact_summary_json": str(first_bad_path),
        "workflow_run_diagnosis_summary_json": str(summary_path),
        "workflow_run_diagnosis_summary_txt": str(summary_txt_path),
    }
    report.setdefault("outputs", {}).update(outputs)
    report.setdefault("final_reporting", {}).setdefault("receipts", {}).update(outputs)
    return outputs
