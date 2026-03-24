from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from contract_enforcement import ContractResult, ContractSuiteResult
from contract_tests import ContractTestSuite


def run_sdb_runtime_contracts(*, out_root: Path, fused_df: Any, training_df: Any, run_report: Dict[str, Any] | None, model_meta: Dict[str, Any] | None, args: Any) -> ContractSuiteResult:
    suite = ContractSuiteResult(stage="sdb")
    context = {
        "fused_df": fused_df,
        "training_df": training_df if training_df is not None else fused_df,
        "run_report": run_report or {},
        "xyz_provided": bool(getattr(args, "extra_xyz", None)),
        "adaptive_sampling_enabled": bool(getattr(args, "enable_adaptive_sampling", False)),
        "chunked_mode": getattr(args, "chunked_prediction", "off"),
    }
    legacy = ContractTestSuite()
    legacy.add_standard_tests()
    legacy.run_all(context)
    for result in legacy.results.values():
        suite.add(ContractResult(
            name=result["name"],
            stage=suite.stage,
            severity="error" if bool(result.get("critical", True)) else "warning",
            passed=bool(result.get("passed", False)),
            message=str(result.get("message", "")),
            metrics={"value": result.get("value")},
        ))

    guidance = (model_meta or {}).get("physics_guidance", {}) if isinstance(model_meta, dict) else {}
    anchor_support_good = bool(guidance.get("anchor_support_good", False))
    if anchor_support_good:
        actual_training_depth_max = guidance.get("actual_training_depth_max")
        optical_max_depth = guidance.get("optical_max_depth")
        ok = True
        msg = "Anchor-support optical cap contract satisfied."
        if actual_training_depth_max is not None and optical_max_depth is not None:
            try:
                atd = float(actual_training_depth_max)
                omd = float(optical_max_depth)
                ok = omd >= atd * 1.05 - 1e-6
                if not ok:
                    msg = f"optical_max_depth={omd:.3f} is below anchor-aware floor {atd*1.05:.3f}."
            except (TypeError, ValueError):
                ok = False
                msg = "Anchor-support depth metadata is not numeric."
        suite.add(ContractResult(
            name="anchor_support_optical_cap",
            stage=suite.stage,
            severity="error",
            passed=ok,
            message=msg,
            metrics={"actual_training_depth_max": actual_training_depth_max, "optical_max_depth": optical_max_depth},
        ))
    return suite
