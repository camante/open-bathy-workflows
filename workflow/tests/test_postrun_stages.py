from pathlib import Path
from types import SimpleNamespace

from final_postrun_contract import build_final_postrun_context
from postrun_benchmark_stage import benchmark_requested, run_postrun_benchmark_stage
from postrun_output_stage import write_final_output_bundle, run_postrun_output_checks


class DummyLogger:
    def __init__(self):
        self.messages = []

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args))

    def info(self, *args, **kwargs):
        self.messages.append(("info", args))



def _context(tmp_path):
    cfg = SimpleNamespace(out_dir=tmp_path)
    args = SimpleNamespace(benchmark_holdout=None, benchmark_auto_holdout=False)
    return build_final_postrun_context(
        cfg=cfg,
        args=args,
        report={},
        run_id="run123",
        final_native=tmp_path / "final_native.tif",
        final_for_user=str(tmp_path / "final_for_user.tif"),
        final_provenance=tmp_path / "prov.tif",
        fatal_errors=["x"],
    )



def test_build_final_postrun_context_normalizes_paths(tmp_path):
    context = _context(tmp_path)
    assert context.run_id == "run123"
    assert context.artifacts.final_native == tmp_path / "final_native.tif"
    assert context.artifacts.final_for_user == tmp_path / "final_for_user.tif"
    assert context.artifacts.final_provenance == tmp_path / "prov.tif"
    assert context.fatal_errors == ("x",)



def test_write_final_output_bundle_and_postrun_checks(tmp_path):
    context = _context(tmp_path)
    logger = DummyLogger()
    calls = {}

    def _write_bundle(cfg, report, *, final_native, final_for_user, final_provenance):
        calls["bundle"] = (cfg, report, final_native, final_for_user, final_provenance)
        return tmp_path / "bathy_report.json"

    def _write_io_manifest(out_dir, report):
        assert out_dir == tmp_path
        calls["io"] = True
        return tmp_path / "io_manifest.json", tmp_path / "io_manifest.md"

    def _emit(report_path, io_json, io_md, report, logger_obj):
        calls["emit"] = (report_path, io_json, io_md, report, logger_obj)

    def _run_seams(args, cfg, report, final_native, final_for_user, report_path):
        calls["seams"] = (args, cfg, report, final_native, final_for_user, report_path)

    report_path = write_final_output_bundle(
        context=context,
        logger=logger,
        write_bundle_fn=_write_bundle,
        write_io_manifest_fn=_write_io_manifest,
        emit_artifacts_fn=_emit,
    )
    assert report_path == tmp_path / "bathy_report.json"
    assert context.report_path == report_path

    run_postrun_output_checks(context=context, run_seam_comparisons_fn=_run_seams)
    assert calls["bundle"][2:] == (
        tmp_path / "final_native.tif",
        tmp_path / "final_for_user.tif",
        tmp_path / "prov.tif",
    )
    assert calls["seams"][3:] == (
        tmp_path / "final_native.tif",
        tmp_path / "final_for_user.tif",
        tmp_path / "bathy_report.json",
    )



def test_benchmark_stage_respects_request_flag(tmp_path):
    context = _context(tmp_path)
    logger = DummyLogger()
    observed = {}

    assert benchmark_requested(context.args) is False
    assert run_postrun_benchmark_stage(
        context=context,
        logger=logger,
        run_workflow_benchmark_fn=lambda **kwargs: observed.update(kwargs) or {"ok": True},
    ) is None
    assert observed == {}

    context.args = SimpleNamespace(benchmark_holdout="holdout.gpkg", benchmark_auto_holdout=False)
    assert benchmark_requested(context.args) is True
    out = run_postrun_benchmark_stage(
        context=context,
        logger=logger,
        run_workflow_benchmark_fn=lambda **kwargs: observed.update(kwargs) or {"ok": True},
    )
    assert out == {"ok": True}
    assert observed["final_path"] == tmp_path / "final_for_user.tif"
