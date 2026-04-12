import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from final_run_reporting import build_final_run_state, build_final_run_report, write_final_run_reporting_bundle


def test_build_final_run_report_includes_execution_state():
    cfg = SimpleNamespace(out_dir="/tmp/out")
    report = {
        "method_activation_truth": {
            "river": {
                "method_name": "river",
                "requested": True,
                "requested_reason": "requested",
                "domain_summary_should_run": True,
                "validated_mask_should_run": True,
                "validated_mask_pixels": 12,
                "effective_should_run": True,
                "effective_reason": "validated_mask_positive",
                "candidate_domain_mask": "/tmp/candidate.tif",
                "active_domain_mask": "/tmp/active.tif",
                "semantic_valid": True,
                "semantic_reason": "ok",
                "active_guidance_product": "/tmp/river.tif",
            }
        },
        "guidance_domains": {"activation": {"requested": ["river", "sdb"], "effective": ["river"]}},
    }
    state = build_final_run_state(cfg=cfg, report=report, final_native="a.tif", final_for_user="b.tif", final_provenance="c.tif")
    payload = build_final_run_report(state)
    assert payload["workflow_execution_state"]["effective_methods"] == ["river"]
    assert payload["workflow_execution_state"]["final_outputs"]["final_native"] == "a.tif"


def test_write_final_run_reporting_bundle_updates_receipts_and_report():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        cfg = SimpleNamespace(out_dir=out_dir)
        report = {"guidance_domains": {"activation": {"requested": [], "effective": []}}}
        state = build_final_run_state(cfg=cfg, report=report, final_native=None, final_for_user=None, final_provenance=None)

        def _writer(*args, **kwargs):
            p = out_dir / "x.json"
            p.write_text("{}")
            return p

        out = write_final_run_reporting_bundle(
            state,
            write_authoritative_cache_receipt=_writer,
            write_guidance_manifest=_writer,
            write_explicit_final_outputs_manifest=_writer,
            write_support_provenance_summary=_writer,
            write_final_support_regime_audit=_writer,
            write_final_dem_selection_receipt=_writer,
            write_comparison_package=_writer,
        )
        payload = json.loads(Path(out).read_text())
        assert "workflow_execution_state" in payload
        assert payload["final_reporting"]["receipts"]["guidance_manifest"].endswith("x.json")
