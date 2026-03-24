from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from contracts_sign_semantics_runtime import run_sdb_sign_semantics_runtime_contracts
from final_dem_contract_validator import validate_written_final_dem_contract


def _write_raster(path: Path, arr: np.ndarray, *, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4269",
        "transform": from_origin(0, 0, 1, 1),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        if tags:
            dst.update_tags(**tags)


class _Args:
    def __init__(self, out_dir: Path):
        self.out_dir = str(out_dir)


def test_standalone_sdb_semantics_contracts_pass(tmp_path: Path):
    out_root = tmp_path / "sdb"
    rast = out_root / "rasters"
    _write_raster(rast / "run_sdb_depth.tif", np.ones((5, 5), dtype=np.float32), tags={"VALUE_TYPE": "elevation", "SIGN_CONVENTION": "relative_to_datum"})
    suite = run_sdb_sign_semantics_runtime_contracts(_Args(out_root), contracts_dir=out_root / "contracts", report={"outputs": {}})
    assert suite["fail"] == 0
    assert suite["pass"] >= 1


def test_final_dem_validator_includes_semantic_contract(tmp_path: Path):
    final_p = tmp_path / "final.tif"
    auth_p = tmp_path / "auth.tif"
    support_p = tmp_path / "support.tif"
    _write_raster(final_p, np.ones((4, 4), dtype=np.float32), tags={"VALUE_TYPE": "elevation", "SIGN_CONVENTION": "relative_to_datum"})
    _write_raster(auth_p, np.ones((4, 4), dtype=np.float32), tags={"VALUE_TYPE": "elevation", "SIGN_CONVENTION": "relative_to_datum"})
    _write_raster(support_p, np.zeros((4, 4), dtype=np.float32))
    payload = validate_written_final_dem_contract(final_depth=final_p, aligned_authoritative_base=auth_p, support_class=support_p)
    assert payload["semantic_contract"]["ok"] is True


def test_final_dem_validator_flags_depth_semantics_for_final(tmp_path: Path):
    final_p = tmp_path / "final_bad.tif"
    _write_raster(final_p, -np.ones((4, 4), dtype=np.float32), tags={"VALUE_TYPE": "depth", "SIGN_CONVENTION": "negative_down"})
    payload = validate_written_final_dem_contract(final_depth=final_p)
    assert payload["semantic_contract"]["ok"] is False
