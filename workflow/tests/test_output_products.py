from pathlib import Path
from types import SimpleNamespace

from core.json_io import write_json
from output_products import build_final_output_contract
from final_reporting import write_explicit_final_outputs_manifest, write_final_dem_selection_receipt


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_build_final_output_contract_prefers_engine_gapfill(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "gapfill_depth.tif")
    final_user = _touch(tmp_path / "deliver" / "final_user.tif")
    final_prov = _touch(tmp_path / "combined" / "final_prov.tif")
    conditioned = _touch(tmp_path / "combined" / "conditioned.tif")
    support = _touch(tmp_path / "combined" / "support_class.tif")
    guidance = _touch(tmp_path / "sdb" / "sdb_guidance_manifest.json")
    river_manifest = _touch(tmp_path / "river" / "river_guidance_manifest.json")
    scaffold_manifest = _touch(tmp_path / "river" / "river_scaffold_manifest.json")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"conditioned_depth": str(conditioned), "support_class": str(support)}},
        "gapfill": {"outputs": {"depth": str(final_native)}},
        "sdb": {"artifacts": {"guidance_manifest": str(guidance)}},
        "river": {"outputs": {"guidance_manifest": str(river_manifest), "scaffold_manifest": str(scaffold_manifest)}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base=str(tmp_path / "auth.tif"))
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=final_user, final_provenance=final_prov)
    assert contract["final_generation_route"] == "support_aware_terrain_interpolator_plus_gapfill"
    assert contract["runtime_engine"]["active"] is True
    assert contract["runtime_engine"]["selected_output_uses_engine"] is True
    assert contract["guidance_manifests"]["sdb_guidance_manifest"] == str(guidance)
    assert contract["regime_artifacts"] == {"final": None, "sdb": None, "river": None}


def test_final_output_manifests_include_runtime_engine(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    support = _touch(tmp_path / "combined" / "support_class.tif")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"conditioned_depth": str(final_native), "support_class": str(support)}},
        "sdb": {"artifacts": {}},
        "river": {"outputs": {}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    outputs_path = write_explicit_final_outputs_manifest(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    selection_path = write_final_dem_selection_receipt(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    outputs = __import__("json").loads(Path(outputs_path).read_text(encoding="utf-8"))
    selection = __import__("json").loads(Path(selection_path).read_text(encoding="utf-8"))
    assert outputs["runtime_engine"]["module"] == "terrain_interpolator"
    assert outputs["final_generation_route"] == "support_aware_terrain_interpolator"
    assert selection["runtime_engine"]["selected_output_uses_engine"] is True
    assert report["final_dem_runtime"]["runtime_engine"]["module"] == "terrain_interpolator"


def test_build_final_output_contract_prefers_explicit_runtime_engine_state(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "final_native.tif")
    final_prov = _touch(tmp_path / "combined" / "final_prov.tif")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {}},
        "final_dem_runtime": {
            "final_generation_route": "support_aware_terrain_interpolator_plus_gapfill",
            "runtime_engine": {
                "module": "terrain_interpolator",
                "active": True,
                "selected_output_uses_engine": True,
                "authoritative_conditioning_applied": True,
            },
        },
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    assert contract["final_generation_route"] == "support_aware_terrain_interpolator_plus_gapfill"
    assert contract["runtime_engine"]["active"] is True
    assert contract["runtime_engine"]["module"] == "terrain_interpolator"
    assert contract["runtime_engine"]["selected_output_uses_engine"] is True
    assert contract["runtime_engine"]["authoritative_conditioning_applied"] is True


def test_build_final_output_contract_exposes_regime_artifacts(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    final_regime = _touch(tmp_path / "combined" / "regime_class.tif")
    sdb_regime = _touch(tmp_path / "sdb" / "sdb_regime.tif")
    river_regime = _touch(tmp_path / "river" / "river_regime.tif")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"regime_class": str(final_regime)}},
        "sdb": {"artifacts": {"regime_class_raster": str(sdb_regime)}},
        "river": {"outputs": {"regime_class": str(river_regime)}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    assert contract["regime_artifacts"] == {"final": str(final_regime), "sdb": str(sdb_regime), "river": str(river_regime)}
    assert contract["support_artifacts"]["regime_class"] == str(final_regime)
    assert contract["support_artifacts"]["sdb_regime_class"] == str(sdb_regime)
    assert contract["support_artifacts"]["river_regime_class"] == str(river_regime)
