from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from validation.river_validation_policy import suppressed_optional_validation_payload, validation_case_requested


def write_scientific_validation_summary(*, out_path: Any, final_outputs_manifest: Any, overlap_identity_evaluation: Any = None, validation_truth: Any = None, case_specs: list[Any] | None = None, case_manifest: Any = None, guidance_baseline_case: str = "baseline_cudem_interpolation", guidance_target_case: str = "selected_final", guidance_rmse_tolerance: float = 0.0) -> dict[str, Any]:
    requested = validation_case_requested(validation_truth=validation_truth, case_specs=case_specs, case_manifest=case_manifest)
    if not requested:
        payload = suppressed_optional_validation_payload("scientific_validation", reason="no_validation_truth_or_case_specs_configured")
        payload.update({
            "final_outputs_manifest": str(final_outputs_manifest) if final_outputs_manifest not in (None, "") else None,
            "overlap_identity_evaluation_present": overlap_identity_evaluation is not None,
            "validation_truth": None,
            "case_manifest": None,
            "case_count": 0,
            "metrics": {},
        })
        return payload

    payload = {
        "status": "not_evaluated",
        "reason": "scientific_validation_case_evaluation_not_implemented_in_packaged_helper",
        "final_outputs_manifest": str(final_outputs_manifest) if final_outputs_manifest not in (None, "") else None,
        "overlap_identity_evaluation_present": overlap_identity_evaluation is not None,
        "validation_truth": str(validation_truth) if validation_truth else None,
        "case_manifest": str(case_manifest) if case_manifest else None,
        "case_count": len(case_specs or []),
        "guidance_baseline_case": guidance_baseline_case,
        "guidance_target_case": guidance_target_case,
        "guidance_rmse_tolerance": float(guidance_rmse_tolerance),
        "metrics": {},
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
