from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from final_vertical_semantics_contract import write_final_vertical_semantics_contract


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0, tags: dict | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:4326",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
        if tags:
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
    return str(path)


def _base_outputs(tmp_path: Path) -> dict:
    vertical = {"UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": "NAVD88", "VERTICAL_DATUM_EPSG": "5703"}
    return {
        "conditioned_depth": _write_raster(tmp_path / "conditioned_depth.tif", np.ones((2, 2), dtype=np.float32), tags={**vertical, "VALUE_TYPE": "elevation", "ROLE": "conditioned_final_depth"}),
        "conditioned_provenance": _write_raster(tmp_path / "conditioned_provenance.tif", np.ones((2, 2), dtype=np.uint8), nodata=0, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "conditioned_provenance"}),
        "support_class": _write_raster(tmp_path / "support_class.tif", np.ones((2, 2), dtype=np.uint8), nodata=0, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "support_class"}),
        "support_distance": _write_raster(tmp_path / "support_distance.tif", np.ones((2, 2), dtype=np.float32), tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "support_distance"}),
        "guidance_influence": _write_raster(tmp_path / "guidance_influence.tif", np.full((2, 2), 0.5, dtype=np.float32), tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "guidance_influence"}),
        "anchor_uncertainty": _write_raster(tmp_path / "anchor_uncertainty.tif", np.ones((2, 2), dtype=np.float32), tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "anchor_uncertainty"}),
        "guidance_uncertainty": _write_raster(tmp_path / "guidance_uncertainty.tif", np.ones((2, 2), dtype=np.float32), tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "guidance_uncertainty"}),
        "conditioned_uncertainty": _write_raster(tmp_path / "conditioned_uncertainty.tif", np.ones((2, 2), dtype=np.float32), tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "conditioned_uncertainty"}),
        "source_candidate": _write_raster(tmp_path / "source_candidate.tif", np.ones((2, 2), dtype=np.float32), tags={**vertical, "VALUE_TYPE": "elevation", "ROLE": "guidance_surface_diagnostic"}),
    }


def test_final_vertical_semantics_contract_writes_when_tags_align(tmp_path: Path):
    out = write_final_vertical_semantics_contract(tmp_path / "final_vertical_semantics_contract.json", outputs=_base_outputs(tmp_path))
    payload = json.loads(out.read_text())
    assert payload["ok"] is True
    assert payload["vertical_reference"]["vertical_datum"] == "NAVD88"


def test_final_vertical_semantics_contract_fails_on_mismatched_vertical_datum(tmp_path: Path):
    outputs = _base_outputs(tmp_path)
    _write_raster(tmp_path / "river_bank_elevation.tif", np.ones((2, 2), dtype=np.float32), tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": "MSL", "VERTICAL_DATUM_EPSG": "5714", "ROLE": "river_bank_elevation_guidance"})
    outputs["river_bank_elevation"] = str(tmp_path / "river_bank_elevation.tif")
    with pytest.raises(ValueError, match="vertical_datum_mismatch"):
        write_final_vertical_semantics_contract(tmp_path / "final_vertical_semantics_contract.json", outputs=outputs)
