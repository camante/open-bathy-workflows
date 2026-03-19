from pathlib import Path

from support_points import load_extra_xyz_points


def test_load_extra_xyz_points_drops_sentinel_depth_rows(tmp_path: Path):
    csv_path = tmp_path / "sentinel_support.csv"
    csv_path.write_text(
        "longitude,latitude,depth_m,source\n"
        "-70.90,42.90,-999999,authoritative\n"
        "-70.91,42.91,-12.5,authoritative\n"
        "-70.92,42.92,-3.0,authoritative\n"
    )

    df = load_extra_xyz_points([str(csv_path)], crs="EPSG:4326", aoi_str="-71/-70.75/42.75/43")
    assert len(df) == 2
    assert df["depth_m"].min() > -1000.0
    assert not (df["depth_m"] == -999999.0).any()
