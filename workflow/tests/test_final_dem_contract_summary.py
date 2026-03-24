from pathlib import Path
from types import SimpleNamespace

from output_products import build_final_output_contract


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_final_output_contract_reports_gap_only_legacy_backstop(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    report = {
        "authoritative_base": {
            "status": "applied",
            "outputs": {"conditioned_depth": str(final_native)},
            "candidate_generation": {
                "mode": "support_aware_guidance_only_no_legacy_backstop",
                "stats": {
                    "legacy_fallback_pixels": 0,
                    "legacy_blocked_in_river_corridor_pixels": 8,
                },
                "backstop_policy": {
                    "legacy_candidate_enabled": False,
                    "legacy_candidate_role": "disabled",
                    "disallow_legacy_in_river_corridor_outside_estuary": True,
                },
            },
        },
        "sdb": {"artifacts": {"depth_raster_role": "diagnostic_only"}},
        "river": {"outputs": {}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    summary = contract["final_dem_contract"]
    assert summary["invariants"]["no_legacy_candidate_backstop_in_final_route"] is True
    assert summary["route_cleanup"]["legacy_backstop_used"] is False
    assert summary["route_cleanup"]["legacy_gap_only_backstop_pixels"] == 0
    assert summary["route_cleanup"]["legacy_blocked_in_river_corridor_pixels"] == 8
    assert summary["guidance_roles"]["dense_sdb_depth"] == "diagnostic_only"
