from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.river_validation_policy import suppressed_optional_validation_payload, validation_case_requested


def run_validation_invariance_framework(*, final_outputs_manifest: Any, overlap_identity_evaluation: Any = None, validation_truth: Any = None, case_specs: list[Any] | None = None, case_manifest: Any = None, guidance_baseline_case: str = "baseline_cudem_interpolation", guidance_target_case: str = "selected_final", require_guidance_non_degradation: bool = False, guidance_rmse_tolerance: float = 0.0) -> dict[str, Any]:
    hard_failures: list[str] = []
    manifest_exists = Path(final_outputs_manifest).is_file() if final_outputs_manifest not in (None, "") else False
    if not manifest_exists:
        hard_failures.append("final_outputs_manifest_missing")
    truth_configured = validation_case_requested(validation_truth=validation_truth, case_specs=case_specs, case_manifest=case_manifest)
    if not truth_configured and overlap_identity_evaluation is None:
        payload = suppressed_optional_validation_payload("validation_invariance", reason="no_validation_truth_case_specs_or_overlap_identity_configured")
        payload.update({
            "final_outputs_manifest": str(final_outputs_manifest) if final_outputs_manifest not in (None, "") else None,
            "final_outputs_manifest_exists": manifest_exists,
            "hard_failures": hard_failures,
            "all_hard_invariants_ok": not hard_failures,
        })
        return payload

    return {
        "status": "skipped" if not truth_configured else "not_evaluated",
        "reason": "overlap_identity_only_no_validation_truth" if not truth_configured else "validation_case_evaluation_not_implemented_in_packaged_helper",
        "final_outputs_manifest": str(final_outputs_manifest) if final_outputs_manifest not in (None, "") else None,
        "final_outputs_manifest_exists": manifest_exists,
        "overlap_identity_evaluation_present": overlap_identity_evaluation is not None,
        "validation_truth": str(validation_truth) if validation_truth else None,
        "case_manifest": str(case_manifest) if case_manifest else None,
        "case_count": len(case_specs or []),
        "guidance_baseline_case": guidance_baseline_case,
        "guidance_target_case": guidance_target_case,
        "require_guidance_non_degradation": bool(require_guidance_non_degradation),
        "guidance_rmse_tolerance": float(guidance_rmse_tolerance),
        "hard_failures": hard_failures,
        "all_hard_invariants_ok": not hard_failures,
    }
