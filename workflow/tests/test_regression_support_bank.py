from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_bank import RES_NAME, update_bank
from train import _compute_physics_guidance_settings


def test_model_bank_backfills_lonlat_and_source_norm(tmp_path: Path):
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir(parents=True)

    old_res = pd.DataFrame(
        {
            "lon": [-70.90, -70.89],
            "lat": [42.80, 42.81],
            "source": ["extra_xyz:hydronos", "extra_xyz:ehydro"],
            "depth_m": [-2.0, -3.0],
            "B02": [0.01, 0.02],
        }
    )
    old_res.to_pickle(bank_dir / RES_NAME, compression="gzip")

    # Use a fresh row so update_bank exercises the alias-migration path instead of
    # only reading the old reservoir back untouched.
    new_df = pd.DataFrame(
        {
            "longitude": [-70.88],
            "latitude": [42.82],
            "source": ["extra_xyz:new"],
            "source_norm": ["extra_xyz:new"],
            "depth_m": [-4.0],
            "B02": [0.03],
        }
    )

    res, _ = update_bank(
        bank_dir,
        new_df,
        target_col="depth_m",
        max_samples=10,
        seed=42,
        schema_cols=["B02", "longitude", "latitude", "source", "source_norm", "depth_m"],
    )

    old_rows = res[res["source"].isin(["extra_xyz:hydronos", "extra_xyz:ehydro"])].copy()
    assert len(old_rows) == 2
    assert old_rows["longitude"].notna().all()
    assert old_rows["latitude"].notna().all()
    assert old_rows["source_norm"].astype(str).str.startswith("extra_xyz").all()


def test_physics_guidance_support_points_accept_lon_lat_aliases():
    df = pd.DataFrame(
        {
            "lon": [-70.90, -70.89, -70.88],
            "lat": [42.80, 42.81, 42.82],
            "depth_m": [-1.0, -2.0, -3.0],
            "stumpf_depth": [1.1, 1.9, 3.1],
            "stumpf_idx": [0.10, 0.20, 0.30],
            "source": ["extra_xyz:hydronos", "extra_xyz:hydronos", "extra_xyz:ehydro"],
        }
    )
    g = _compute_physics_guidance_settings(df)
    assert len(g["support_points_lonlat"]) == 3
    assert g["support_points_lonlat"][0] == [-70.9, 42.8]
