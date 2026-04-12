import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bathy_main


def test_prepare_authoritative_river_soundings_populates_cfg(tmp_path: Path):
    auth = tmp_path / 'auth.tif'
    auth.write_bytes(b'x')
    cfg = SimpleNamespace(
        authoritative_base=auth,
        cache_root=tmp_path,
        aoi='-71/-70.75/42.75/43',
        working_srs='EPSG:26919',
        extra_xyz_crs='EPSG:4326',
        working_vcrs_epsg=5703,
        river_authoritative_soundings=None,
    )
    report = {}
    out_csv = tmp_path / 'authoritative_support' / 'pts.csv'
    def _fake_prepare(*args, **kwargs):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text('x,y,depth_m,source\n1,2,-3,authoritative_base\n', encoding='utf-8')
        return {'path': str(out_csv), 'count': 1}
    with mock.patch.object(bathy_main, '_prepare_authoritative_river_soundings_points', side_effect=_fake_prepare):
        bathy_main._prepare_authoritative_river_soundings(cfg, report)
    assert cfg.river_authoritative_soundings is not None
    assert 'river_guidance' in report['authoritative_base']



def test_normalize_multi_path_value_handles_list_repr_string():
    vals = bathy_main._normalize_multi_path_value("['/tmp/a.csv','/tmp/b.csv']")
    assert vals == ['/tmp/a.csv', '/tmp/b.csv']


def test_append_river_soundings_args_normalizes_list_repr_string():
    cfg = SimpleNamespace(
        river_soundings="['/tmp/a.csv','/tmp/b.csv']",
        river_soundings_crs='EPSG:26919',
        river_soundings_calib_max_dist_m=0.0,
        river_soundings_calib_stat='',
        river_soundings_mode='auto',
        river_soundings_cell_percentile=25.0,
        river_soundings_max_dist_m=1500.0,
        river_soundings_min_r=0.25,
        river_no_soundings_enforce=False,
        river_dem=None,
    )
    cmd = ['python', 'xs_infer_bathy_raster.py']
    bathy_main._append_river_soundings_args(cmd, cfg, include_calib_args=True, include_mode_args=False)
    assert '--soundings=/tmp/a.csv' in cmd
    assert '--soundings=/tmp/b.csv' in cmd
    assert all("['" not in part for part in cmd)


def test_load_river_authoritative_support_points_prefers_river_soundings_crs():
    cfg = SimpleNamespace(
        river_soundings='/tmp/river.csv',
        extra_xyz_files=None,
        river_soundings_crs='EPSG:26919',
        working_srs='EPSG:32619',
        extra_xyz_crs='EPSG:4326',
        aoi='-71/-70/42/43',
    )
    fake = pd.DataFrame({
        'longitude': [-70.9],
        'latitude': [42.8],
        'depth_m': [-3.0],
        'source': ['authoritative_base'],
    })
    with mock.patch('support_points.load_extra_xyz_points', return_value=fake) as m:
        out = bathy_main._load_river_authoritative_support_points(cfg)
    assert m.call_args.kwargs['crs'] == 'EPSG:26919'
    assert list(out.columns) == ['lon', 'lat', 'depth_m', 'source']
    assert out.iloc[0]['lon'] == -70.9
    assert out.iloc[0]['lat'] == 42.8


def test_load_river_authoritative_support_points_direct_preserves_role_columns(tmp_path: Path):
    csv_path = tmp_path / "river_roleaware.csv"
    csv_path.write_text(
        "x,y,depth_m,source,authoritative_role,role_confidence,distance_to_bank_m,normalized_channel_position,inside_channel_mask\n"
        "1,2,3,embedded_source_name,authoritative_bed_core,0.9,4.0,0.8,1\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        river_soundings=None,
        river_authoritative_soundings=str(csv_path),
        extra_xyz_files=None,
        river_soundings_crs='EPSG:32619',
        working_srs='EPSG:32619',
        extra_xyz_crs='EPSG:4326',
        aoi='-71/-70/42/43',
    )
    out = bathy_main._load_river_authoritative_support_points(cfg)
    assert out is not None
    assert 'authoritative_role' in out.columns
    assert 'role_confidence' in out.columns
    assert out.iloc[0]['authoritative_role'] == 'authoritative_bed_core'
    assert float(out.iloc[0]['role_confidence']) == 0.9
    assert out.iloc[0]['source'] == 'embedded_source_name'
