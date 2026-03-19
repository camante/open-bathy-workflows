import numpy as np

import authoritative_guidance as ag


class _FakeTransformer:
    def transform(self, xs, ys, zs=None):
        if zs is None:
            return xs, ys
        return xs, ys, np.asarray(zs, dtype=float) + 1.25


class _FakeDataset:
    def __init__(self):
        self._arr = np.array([[-2.0, 1.0], [-3.5, -0.5]], dtype='float32')
        self.nodata = -9999.0
        self.transform = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        self.crs = type('CRS', (), {'to_epsg': lambda self: 4269, '__str__': lambda self: 'EPSG:4269'})()
    def read(self, idx):
        return self._arr
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_prepare_authoritative_sdb_training_points_uses_point_transform_not_dlim(tmp_path, monkeypatch):
    monkeypatch.setattr(ag.rasterio, 'open', lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(ag.Transformer, 'from_crs', staticmethod(lambda *args, **kwargs: _FakeTransformer()))

    out_csv = tmp_path / 'support.csv'
    info = ag.prepare_authoritative_sdb_training_points(
        tmp_path / 'auth.tif',
        out_csv,
        out_crs='EPSG:4269',
        source_vdatum='epsg:4269+5703',
        target_vdatum='epsg:4269+5714',
        max_points=100,
        negative_only=True,
    )

    text = out_csv.read_text(encoding='utf-8').strip().splitlines()
    assert text[0] == 'x,y,depth_m,source'
    assert len(text) == 3
    assert info['converted_raster'] is None
    assert info['count'] == 2
    assert any('-0.750000' in line for line in text)
