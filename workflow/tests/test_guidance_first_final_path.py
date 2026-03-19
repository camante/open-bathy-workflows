from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from output_products import build_final_output_contract


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_final_output_contract_exposes_guidance_first_inputs(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    guide_points = _touch(tmp_path / "sdb" / "pred_depth_guide_points.gpkg")
    gw = _touch(tmp_path / "sdb" / "pred_depth_guidance_weight.tif")
    adm = _touch(tmp_path / "sdb" / "pred_depth_admissibility.tif")
    river_gp = _touch(tmp_path / "river" / "river_guide_points.gpkg")
    corridor = _touch(tmp_path / "river" / "river_corridor_mask.tif")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"conditioned_depth": str(final_native)}},
        "sdb": {"artifacts": {"guide_points": str(guide_points), "guidance_weight_raster": str(gw), "admissibility_raster": str(adm)}},
        "river": {"outputs": {"guide_points": str(river_gp), "corridor_mask": str(corridor)}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    assert contract["guidance_contract"]["mode"] == "guidance_first"
    assert contract["guidance_contract"]["dense_sdb_depth_role"] == "diagnostic_only"
    assert contract["guidance_artifacts"]["sdb_guide_points"] == str(guide_points)
    assert contract["guidance_artifacts"]["river_corridor_mask"] == str(corridor)
    assert contract["guidance_contract"]["artifacts_present"]["sdb_guide_points"] is True
    assert contract["guidance_contract"]["final_dem_inputs"][0] == "authoritative_hard_locks"



def test_final_output_contract_reports_guidance_readiness(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    guide_points = _touch(tmp_path / "sdb" / "pred_depth_guide_points.gpkg")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"conditioned_depth": str(final_native)}},
        "sdb": {"artifacts": {"guide_points": str(guide_points)}},
        "river": {"outputs": {}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    readiness = contract["guidance_contract"]["guidance_readiness"]
    assert readiness["sdb_guidance_ready"] is False
    assert "sdb_guidance_weight" in readiness["missing_core_artifacts"]
    assert readiness["guidance_ready"] is False
