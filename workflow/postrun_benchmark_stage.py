from __future__ import annotations

from typing import Any, Callable, Optional

from final_postrun_contract import FinalPostRunContext


RunWorkflowBenchmarkFn = Callable[..., Optional[dict[str, Any]]]



def benchmark_requested(args: Any) -> bool:
    return bool(getattr(args, "benchmark_holdout", None)) or bool(getattr(args, "benchmark_auto_holdout", False))



def run_postrun_benchmark_stage(
    *,
    context: FinalPostRunContext,
    logger,
    run_workflow_benchmark_fn: RunWorkflowBenchmarkFn,
) -> Optional[dict[str, Any]]:
    if not benchmark_requested(context.args):
        context.report.setdefault("benchmark", {})["status"] = "skipped"
        context.report["benchmark"].update({
            "requested": False,
            "reason": "no_holdout_flag",
        })
        logger.info(
            "[BENCHMARK] Skipped postrun benchmark: neither --benchmark-holdout nor --benchmark-auto-holdout was provided."
        )
        return None
    final_path = context.artifacts.final_for_user or context.artifacts.final_native
    context.report.setdefault("benchmark", {})["requested"] = True
    return run_workflow_benchmark_fn(
        cfg=context.cfg,
        args=context.args,
        report=context.report,
        logger=logger,
        final_path=final_path,
    )
