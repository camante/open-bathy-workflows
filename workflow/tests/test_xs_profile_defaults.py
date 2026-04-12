import sys

import bathy_main
import xs_infer_bathy_raster as xir


def test_bathy_main_default_river_xs_profile_shape_is_parabolic(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bathy_main.py",
            "--aoi=-71/-70.75/42.75/43",
            "--start=2025-01-01",
            "--end=2026-01-01",
            "--out-dir=output/test",
        ],
    )
    args = bathy_main.parse_args()
    assert args.river_xs_profile_shape == "parabolic"


def test_xs_infer_default_profile_shape_is_parabolic(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["xs_infer_bathy_raster.py", "--xs-gpkg=dummy.gpkg", "--out-gpkg=out.gpkg"],
    )
    args = xir._parse_args()
    assert args.xs_profile_shape == "parabolic"
