from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from validation.river_validation_policy import regression_case_requested, suppressed_optional_validation_payload


def run_postrun_regression_stage(*, args: Any, cfg: Any, report: dict[str, Any], run_id: str, final: Any, final_for_user: Any, report_path: Any, logger: Any | None = None) -> dict[str, Any]:
    if not regression_case_requested(args):
        payload = suppressed_optional_validation_payload("postrun_regression", reason="no_adjacent_or_nested_regression_case_configured")
        payload.update({
            "run_id": str(run_id),
            "final": str(final) if final not in (None, "") else None,
            "final_for_user": str(final_for_user) if final_for_user not in (None, "") else None,
        })
        report.setdefault("optional_validation", {})["postrun_regression"] = payload
        if logger is not None:
            logger.debug("[REGRESSION] not run; no adjacent/nested regression case was configured")
        return payload

    payload = {
        "status": "skipped",
        "reason": "regression_case_evaluation_not_implemented_in_packaged_helper",
        "run_id": str(run_id),
        "final": str(final) if final not in (None, "") else None,
        "final_for_user": str(final_for_user) if final_for_user not in (None, "") else None,
    }
    out_dir = Path(getattr(cfg, "out_dir", ".")) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "postrun_regression_stage.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.setdefault("outputs", {})["postrun_regression_stage"] = str(out_path)
    report.setdefault("postrun_regression", {}).update(payload)
    if logger is not None:
        logger.info("[REGRESSION] %s", payload.get("status"))
    return payload
