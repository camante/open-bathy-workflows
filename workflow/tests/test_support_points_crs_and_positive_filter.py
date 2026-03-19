from pathlib import Path

import pandas as pd

from support_points import load_extra_xyz_points


def test_load_extra_xyz_points_keeps_lonlat_when_declared_crs_is_projected(tmp_path: Path) -> None:
    p = tmp_path / 'authoritative_support.csv'
    pd.DataFrame(
        {
            'longitude': [-70.95, -70.90],
            'latitude': [42.80, 42.85],
            'depth_m': [-5.0, -6.0],
        }
    ).to_csv(p, index=False)

    out = load_extra_xyz_points([str(p)], crs='EPSG:32619', aoi_str='-71/-70.75/42.75/43')
    assert len(out) == 2
    assert out['longitude'].between(-71, -70.75).all()
    assert out['latitude'].between(42.75, 43.0).all()


def test_load_extra_xyz_points_drops_positive_rows_for_mixed_sign_support(tmp_path: Path) -> None:
    p = tmp_path / 'mixed.xyz'
    pd.DataFrame(
        {
            'x': [-70.95, -70.94, -70.93],
            'y': [42.80, 42.81, 42.82],
            'z': [-4.0, 1.5, -3.0],
        }
    ).to_csv(p, index=False)

    out = load_extra_xyz_points([str(p)], crs='EPSG:4326', aoi_str='-71/-70.75/42.75/43')
    assert len(out) == 2
    assert (out['depth_m'] <= 0).all()
