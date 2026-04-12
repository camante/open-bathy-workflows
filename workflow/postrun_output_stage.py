from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from final_postrun_contract import FinalPostRunContext


WriteBundleFn = Callable[..., Path]
WriteIoManifestFn = Callable[..., tuple[Optional[Path], Optional[Path]]]
EmitArtifactsFn = Callable[..., None]
RunSeamComparisonsFn = Callable[..., None]



def write_final_output_bundle(
    *,
    context: FinalPostRunContext,
    logger,
    write_bundle_fn: WriteBundleFn,
    write_io_manifest_fn: WriteIoManifestFn,
    emit_artifacts_fn: EmitArtifactsFn,
) -> Path:
    report_path = write_bundle_fn(
        context.cfg,
        context.report,
        final_native=context.artifacts.final_native,
        final_for_user=context.artifacts.final_for_user,
        final_provenance=context.artifacts.final_provenance,
    )
    try:
        io_json, io_md = write_io_manifest_fn(context.cfg.out_dir, context.report)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.debug("Optional IO manifest write failed: %s", exc)
        io_json, io_md = None, None
    emit_artifacts_fn(report_path, io_json, io_md, context.report, logger)
    logger.info("Report written: %s", report_path)
    context.report_path = Path(report_path)
    return context.report_path



def run_postrun_output_checks(
    *,
    context: FinalPostRunContext,
    run_seam_comparisons_fn: RunSeamComparisonsFn,
) -> None:
    if context.report_path is None:
        raise RuntimeError("postrun output checks require a written report_path")
    run_seam_comparisons_fn(
        context.args,
        context.cfg,
        context.report,
        context.artifacts.final_native,
        context.artifacts.final_for_user,
        context.report_path,
    )
