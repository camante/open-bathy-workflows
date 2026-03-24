from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from output_products import build_final_output_contract
from final_dem_contract import build_final_dem_contract_summary


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_phase_i_final_route_contract_reports_unified_route(tmp_path: Path):
    final_depth = _touch(tmp_path / "combined" / "bathy_combined_depth_conditioned.tif")
    final_prov = _touch(tmp_path / "combined" / "bathy_combined_depth_conditioned_provenance.tif")
    report = {
        "final_dem_route": {
            "route_mode": "staged_final_route_single_source_of_truth",
            "single_authoritative_route_active": True,
            "legacy_parallel_route_retired": True,
            "written_outputs": {"final_depth": str(final_depth), "final_provenance": str(final_prov)},
        },
        "authoritative_base": {
            "status": "applied",
            "outputs": {"conditioned_depth": str(final_depth), "aligned_authoritative_base": str(final_depth), "support_class": str(final_prov)},
            "candidate_generation": {"mode": "staged_final_route_single_source_of_truth", "backstop_policy": {"legacy_candidate_enabled": False}},
        },
    }
    summary = build_final_dem_contract_summary(report)
    assert summary["invariants"]["single_authoritative_route_active"] is True
    assert summary["invariants"]["legacy_parallel_route_retired"] is True
    assert summary["route_cleanup"]["route_mode"] == "staged_final_route_single_source_of_truth"

    contract = build_final_output_contract(SimpleNamespace(out_dir=tmp_path, authoritative_base=""), report, final_native=final_depth, final_for_user=None, final_provenance=final_prov)
    assert contract["final_dem_route_receipt"]["single_authoritative_route_active"] is True
    assert contract["final_dem_contract"]["invariants"]["single_authoritative_route_active"] is True
    assert contract["final_dem_contract"]["route_cleanup"]["route_mode"] == "staged_final_route_single_source_of_truth"
