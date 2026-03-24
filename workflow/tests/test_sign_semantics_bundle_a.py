from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from contracts_sign_semantics_runtime import run_sign_semantics_runtime_contracts


class DummyCfg:
    def __init__(self, out_dir: Path, authoritative_base: Path | None = None, strict: bool = False):
        self.out_dir = out_dir
        self.authoritative_base = authoritative_base
        self.strict = strict


def _write_tif(path: Path, arr: np.ndarray, **tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=from_origin(0, 1, 1, 1),
        nodata=np.nan,
    ) as ds:
        ds.write(arr.astype('float32'), 1)
        ds.update_tags(**tags)


def test_bundle_a_reports_expected_semantics(tmp_path: Path):
    final = tmp_path / 'final.tif'
    auth = tmp_path / 'auth.tif'
    river = tmp_path / 'river_warped.tif'
    _write_tif(final, np.array([[1.0, 2.0], [3.0, 4.0]]), VALUE_TYPE='elevation', SIGN_CONVENTION='relative_to_datum')
    _write_tif(auth, np.array([[5.0, 6.0], [7.0, 8.0]]), VALUE_TYPE='elevation', SIGN_CONVENTION='relative_to_datum')
    _write_tif(river, np.array([[-1.0, -2.0], [-3.0, -4.0]]), VALUE_TYPE='depth', SIGN_CONVENTION='negative_down')
    cfg = DummyCfg(tmp_path, authoritative_base=auth)
    report = {'outputs': {'final_depth_native': str(final), 'river_warped': str(river)}}
    suite = run_sign_semantics_runtime_contracts(cfg, report)
    assert suite['fail'] == 0
    assert (tmp_path / 'contracts' / 'contracts_sign_semantics.json').exists()
    assert report['contracts']['sign_semantics']['all_ok'] is True


def test_bundle_a_flags_semantic_mismatch(tmp_path: Path):
    final = tmp_path / 'final_bad.tif'
    _write_tif(final, np.array([[-1.0, -2.0], [-3.0, -4.0]]), VALUE_TYPE='depth', SIGN_CONVENTION='negative_down')
    cfg = DummyCfg(tmp_path)
    report = {'outputs': {'final_depth_native': str(final)}}
    suite = run_sign_semantics_runtime_contracts(cfg, report)
    assert suite['fail'] >= 1
