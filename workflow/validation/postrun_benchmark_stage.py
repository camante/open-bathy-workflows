from __future__ import annotations

from typing import Any

from validation.river_validation_policy import benchmark_requested, suppressed_optional_validation_payload


def run_postrun_benchmark_stage(*, context: Any, logger: Any | None = None, run_workflow_benchmark_fn: Any | None = None) -> dict[str, Any]:
    report = getattr(context, "report", None)
    args = getattr(context, "args", None)
    requested = benchmark_requested(args)
    if not requested:
        payload = suppressed_optional_validation_payload("benchmark", reason="no_benchmark_holdout_configured")
        payload["requested"] = False
        payload["metrics"] = {}
        if isinstance(report, dict):
            report.setdefault("optional_validation", {})["benchmark"] = payload
        if logger is not None:
            logger.debug("[BENCHMARK] not run; no holdout benchmark was configured")
        return payload

    if run_workflow_benchmark_fn is not None:
        payload = run_workflow_benchmark_fn(context=context)
    else:
        payload = {
            "status": "skipped",
            "reason": "benchmark_runner_not_available",
            "requested": True,
            "metrics": {},
        }
    if isinstance(report, dict):
        report["benchmark"] = payload
    if logger is not None:
        logger.info("[BENCHMARK] %s", payload.get("status"))
    return payload
