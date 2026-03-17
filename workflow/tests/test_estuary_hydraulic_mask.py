from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pandas as pd

import bathy_main
from bathy_main import _build_hydraulic_estuary_hint_mask


class _FakeGeom:
    is_empty = False


class _FakeGeoSeries:
    def __init__(self, items):
        self.items = list(items)

    def notna(self):
        return np.array([item is not None for item in self.items], dtype=bool)

    @property
    def is_empty(self):
        return np.array([bool(getattr(item, 'is_empty', False)) for item in self.items], dtype=bool)

    @property
    def empty(self):
        return len(self.items) == 0

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray, pd.Series)):
            mask = np.asarray(key, dtype=bool)
            return _FakeGeoSeries([item for item, keep in zip(self.items, mask) if keep])
        return self.items[key]

    def __iter__(self):
        return iter(self.items)


class _FakeLoc:
    def __init__(self, gdf):
        self._gdf = gdf

    def __getitem__(self, key):
        rows, col = key
        subset = self._gdf._df.loc[rows, col]
        if col == 'geometry':
            if isinstance(subset, pd.Series):
                return _FakeGeoSeries(list(subset))
            return _FakeGeoSeries([subset])
        return subset


class _FakeGeoDataFrame:
    def __init__(self, records, crs='EPSG:3857'):
        self._df = pd.DataFrame(records)
        self.crs = crs
        self.loc = _FakeLoc(self)

    @property
    def columns(self):
        return self._df.columns

    @property
    def empty(self):
        return self._df.empty

    def to_crs(self, crs):
        self.crs = crs
        return self

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key):
        return self._df[key]


def test_hydraulic_estuary_hint_uses_near_mouth_and_low_slope_fields(monkeypatch) -> None:
    fake_gpd = ModuleType('geopandas')
    fake_features = ModuleType('rasterio.features')

    def _read_file(path, layer=None):
        return _FakeGeoDataFrame(
            [
                {'river_id': 'r1', 'dist_to_mouth_km': 4.0, 'slope_mpm': 5e-5, 'geometry': _FakeGeom()},
                {'river_id': 'r2', 'dist_to_mouth_km': 30.0, 'slope_mpm': 5e-3, 'geometry': _FakeGeom()},
            ]
        )

    def _rasterize(shapes, out_shape, transform, fill=0, dtype='uint8'):
        arr = np.zeros(out_shape, dtype=np.uint8)
        if shapes:
            arr[5, 2:8] = 1
        return arr

    fake_gpd.read_file = _read_file
    fake_features.rasterize = _rasterize
    monkeypatch.setitem(sys.modules, 'geopandas', fake_gpd)
    monkeypatch.setitem(sys.modules, 'rasterio.features', fake_features)

    with TemporaryDirectory() as td:
        derived = Path(td) / 'derived'
        network_gpkg = derived / 'river' / 'work' / 'river_network.gpkg'
        network_gpkg.parent.mkdir(parents=True)
        network_gpkg.write_text('placeholder')

        cfg = SimpleNamespace(
            derived_cache_root=derived,
            estuary_transition_m=500.0,
            river_channel_buffer_m=400.0,
            river_max_channel_width_m=600.0,
            river_manning_dist_to_mouth_field='dist_to_mouth_km',
            river_manning_dist_to_mouth_km_max=10.0,
            river_manning_backwater_slope_thresh=1e-4,
        )
        channel = np.ones((12, 12), dtype=np.uint8)
        transform = SimpleNamespace(a=10.0, e=-10.0)

        mask, meta = _build_hydraulic_estuary_hint_mask(
            cfg=cfg,
            channel=channel,
            transform=transform,
            crs='EPSG:3857',
            px_size_m=10.0,
        )

        assert int(mask.sum()) > 0
        assert meta['reason'] == 'ok'
        assert meta['flagged_reaches'] == 1
        assert meta['near_mouth_reaches'] == 1
        assert meta['backwater_slope_reaches'] == 1
        assert meta['dist_to_mouth_field'] == 'dist_to_mouth_km'
        assert meta['slope_field'] == 'slope_mpm'
