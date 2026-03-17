from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sys
import types
import rasterio

from bathy_main import _apply_river_guidance_to_fused_output


def _write_tif(path: Path, arr: np.ndarray, nodata: float = -9999.0, dtype: str = 'float32') -> None:
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs='EPSG:4326',
        transform=(1, 0, 0, 0, -1, 3),
        nodata=nodata,
    ) as ds:
        ds.write(arr.astype(dtype), 1)


def test_river_guidance_uses_authoritative_depth_for_exact_overwrite() -> None:
    with TemporaryDirectory() as td:
        td = Path(td)
        combined = td / 'combined.tif'
        river = td / 'river.tif'
        sdb = td / 'sdb.tif'
        prov = td / 'prov.tif'
        gw = td / 'gw.tif'
        ti = td / 'ti.tif'
        adm = td / 'adm.tif'
        sup = td / 'sup.tif'
        sup_depth = td / 'sup_depth.tif'

        shape = (64, 64)
        center = (32, 32)
        _write_tif(combined, np.zeros(shape, dtype=np.float32))
        _write_tif(river, np.full(shape, 5.0, dtype=np.float32))
        _write_tif(sdb, np.full(shape, 1.0, dtype=np.float32))
        _write_tif(prov, np.zeros(shape, dtype=np.uint8), nodata=0, dtype='uint8')
        _write_tif(gw, np.ones(shape, dtype=np.float32), nodata=0.0)
        ti_arr = np.zeros(shape, dtype=np.uint8)
        ti_arr[center] = 1
        _write_tif(ti, ti_arr, nodata=0, dtype='uint8')
        _write_tif(adm, np.ones(shape, dtype=np.uint8), nodata=0, dtype='uint8')
        _write_tif(sup, ti_arr, nodata=0, dtype='uint8')
        supd = np.full(shape, -9999.0, dtype=np.float32)
        supd[center] = 7.0
        _write_tif(sup_depth, supd, nodata=-9999.0)

        rasterio.band = lambda src, idx: src
        fake_warp = types.ModuleType('rasterio.warp')
        class _Resampling:
            nearest = 0
        def _reproject(source, destination, **kwargs):
            if hasattr(source, 'read'):
                destination[:] = source.read(1)
            else:
                destination[:] = source
        fake_warp.Resampling = _Resampling
        fake_warp.reproject = _reproject
        sys.modules['rasterio.warp'] = fake_warp
        report = {}
        _apply_river_guidance_to_fused_output(
            combined,
            river_path=river,
            sdb_path=sdb,
            provenance_path=prov,
            river_outputs={
                'guidance_weight': str(gw),
                'trusted_interior': str(ti),
                'admissibility': str(adm),
                'authoritative_support': str(sup),
                'authoritative_support_depth': str(sup_depth),
            },
            report=report,
        )

        with rasterio.open(combined) as ds:
            out = ds.read(1)
        assert out[center] == 7.0
        assert out[0, 0] == 5.0

        with rasterio.open(prov) as ds:
            prov_out = ds.read(1)
        assert prov_out[center] == 3
        assert report['fusion']['guidance_controls']['authoritative_exact_overwrite_pixels'] == 1


def test_estuary_transition_does_not_inject_pure_river_when_no_base_value() -> None:
    with TemporaryDirectory() as td:
        td = Path(td)
        combined = td / "combined.tif"
        river = td / "river.tif"
        prov = td / "prov.tif"
        gw = td / "gw.tif"
        adm = td / "adm.tif"
        et = td / "et.tif"

        nodata = -9999.0
        shape = (64, 64)
        target = (3, 4)
        base = np.full(shape, nodata, dtype=np.float32)
        base[0, 0] = 1.0  # keep at least one valid pixel in the array
        _write_tif(combined, base, nodata=nodata)
        _write_tif(river, np.full(shape, 5.0, dtype=np.float32), nodata=nodata)
        _write_tif(prov, np.zeros(shape, dtype=np.uint8), nodata=0, dtype='uint8')
        _write_tif(gw, np.ones(shape, dtype=np.float32), nodata=0.0)
        _write_tif(adm, np.ones(shape, dtype=np.uint8), nodata=0, dtype='uint8')
        et_arr = np.zeros(shape, dtype=np.uint8)
        et_arr[target] = 1
        _write_tif(et, et_arr, nodata=0, dtype='uint8')

        rasterio.band = lambda src, idx: src
        fake_warp = types.ModuleType('rasterio.warp')
        class _Resampling:
            nearest = 0
        def _reproject(source, destination, **kwargs):
            if hasattr(source, 'read'):
                destination[:] = source.read(1)
            else:
                destination[:] = source
        fake_warp.Resampling = _Resampling
        fake_warp.reproject = _reproject
        sys.modules['rasterio.warp'] = fake_warp
        report = {}
        _apply_river_guidance_to_fused_output(
            combined,
            river_path=river,
            sdb_path=None,
            provenance_path=prov,
            river_outputs={
                'guidance_weight': str(gw),
                'admissibility': str(adm),
                'estuary_transition': str(et),
            },
            report=report,
            estuary_max_weight=0.25,
        )

        with rasterio.open(combined) as ds:
            out = ds.read(1)
        assert out[target] == nodata
        assert report['fusion']['guidance_controls']['estuary_transition_no_base_fill_skipped_pixels'] == 1
