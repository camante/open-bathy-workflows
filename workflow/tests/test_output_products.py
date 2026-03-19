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
    assert contract["final_dem_validation"]["skip_reason"] is not None
    assert contract["written_precedence_audit"]["skip_reason"] is not None
    assert contract["written_precedence_audit"]["skip_reason"] is not None
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
    assert contract["final_dem_validation"]["skip_reason"] is not None
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
    assert contract["final_dem_validation"]["skip_reason"] is not None
    assert contract["support_artifacts"]["regime_class"] == str(final_regime)
    assert contract["support_artifacts"]["sdb_regime_class"] == str(sdb_regime)
    assert contract["support_artifacts"]["river_regime_class"] == str(river_regime)


def test_build_final_output_contract_includes_written_precedence_audit_for_real_rasters(tmp_path: Path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    def _write_tif(path: Path, arr: np.ndarray, *, nodata=-9999.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = "float32" if np.issubdtype(arr.dtype, np.floating) else "uint8"
        profile = {
            "driver": "GTiff",
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
            "count": 1,
            "dtype": dtype,
            "crs": "EPSG:4326",
            "transform": from_origin(-71.0, 43.0, 1.0, 1.0),
            "nodata": nodata if np.issubdtype(arr.dtype, np.floating) else 0,
        }
        out = arr.copy()
        if np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float32)
            out[~np.isfinite(out)] = np.float32(nodata)
        else:
            out = out.astype(np.uint8)
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(out, 1)
        return path

    final_native = _write_tif(tmp_path / "combined" / "conditioned_depth.tif", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    final_prov = _write_tif(tmp_path / "combined" / "conditioned_prov.tif", np.array([[10, 30], [60, 10]], dtype=np.uint8))
    auth = _write_tif(tmp_path / "combined" / "aligned_authoritative_base.tif", np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32))
    support = _write_tif(tmp_path / "combined" / "support_class.tif", np.array([[1, 3], [6, 1]], dtype=np.uint8))
    guidance = _write_tif(tmp_path / "combined" / "guidance_influence.tif", np.array([[0.0, 0.4], [0.2, 0.0]], dtype=np.float32))
    report = {
        "authoritative_base": {
            "status": "applied",
            "outputs": {
                "conditioned_depth": str(final_native),
                "aligned_authoritative_base": str(auth),
                "support_class": str(support),
                "guidance_influence": str(guidance),
            },
        },
        "sdb": {"artifacts": {}},
        "river": {"outputs": {}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    audit = contract["written_precedence_audit"]
    assert audit["validated"] is True
    assert audit["all_ok"] is True
    assert audit["authoritative_lock"]["locked_cell_count"] == 2
    assert audit["gap_fill"]["continuous_fill_achieved"] is True
