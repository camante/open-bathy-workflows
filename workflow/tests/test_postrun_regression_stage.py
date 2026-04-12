from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

import postrun_regression_stage as prs


def _write_tif(path: Path, arr: np.ndarray, *, transform=None, crs: str = "EPSG:4326") -> Path:
    transform = transform or from_origin(0.0, float(arr.shape[0]), 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(arr.astype("float32"), 1)
    return path


def test_postrun_regression_stage_writes_nested_outputs(tmp_path: Path, monkeypatch) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    final_raster = _write_tif(tmp_path / "final.tif", arr)
    support = _write_tif(tmp_path / "support.tif", arr + 10.0)
    trusted = _write_tif(tmp_path / "trusted.tif", np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    neighbor_final = _write_tif(tmp_path / "neighbor_final.tif", arr)
    neighbor_support = _write_tif(tmp_path / "neighbor_support.tif", arr + 10.0)
    neighbor_trusted = _write_tif(tmp_path / "neighbor_trusted.tif", np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    final_outputs = tmp_path / "final_outputs.json"
    final_outputs.write_text(json.dumps({
        "aoi": "current",
        "selected_final_depth": str(final_raster),
        "support_class": str(support),
        "final_provenance_native": str(support),
        "river_trusted_interior": str(trusted),
    }), encoding="utf-8")
    neighbor_outputs = tmp_path / "neighbor_final_outputs.json"
    neighbor_outputs.write_text(json.dumps({
        "aoi": "neighbor",
        "selected_final_depth": str(neighbor_final),
        "support_class": str(neighbor_support),
        "final_provenance_native": str(neighbor_support),
        "river_trusted_interior": str(neighbor_trusted),
    }), encoding="utf-8")

    cfg = SimpleNamespace(
        out_dir=tmp_path,
        methods=["river"],
        priority="river",
        derived_cache_root=tmp_path / "cache",
        river_channel_mask=None,
    )
    cfg.derived_cache_root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        seam_compare_with_io=[],
        seam_compare_with_io_list=None,
        seam_strip_px=3,
        nested_aoi_compare_with_final_outputs=[str(neighbor_outputs)],
        nested_aoi_compare_with_final_outputs_list=None,
        nested_aoi_overlap_tolerance=1.0e-6,
        nested_aoi_trusted_tolerance=1.0e-6,
    )
    report = {"run": {"run_id": "test-run"}, "outputs": {"selected_final_provenance": None}, "river": {"outputs": {}}}

    monkeypatch.setattr(prs, "write_validation_invariance_summary", lambda *a, **k: tmp_path / "validation.json")
    monkeypatch.setattr(prs, "write_river_stability_summary", lambda *a, **k: tmp_path / "stability.json")

    prs.run_postrun_regression_stage(
        args=args,
        cfg=cfg,
        report=report,
        run_id="test-run",
        final=str(final_raster),
        final_for_user=str(final_raster),
        report_path=tmp_path / "report.json",
        logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
    )

    assert (tmp_path / "nested_aoi_regression.json").exists()
    assert (tmp_path / "nested_aoi_contract_evaluation.json").exists()
    assert (tmp_path / "nested_aoi_regression_metrics.csv").exists()
    assert (tmp_path / "postrun_regression_summary.json").exists()
    payload = json.loads((tmp_path / "nested_aoi_regression.json").read_text(encoding="utf-8"))
    contracts = json.loads((tmp_path / "nested_aoi_contract_evaluation.json").read_text(encoding="utf-8"))
    assert payload["all_overlap_identity_ok"] is True
    assert payload["all_trusted_interior_identity_ok"] is True
    assert contracts["all_overlap_contracts_ok"] is True
    assert contracts["all_trusted_contracts_ok"] is True
