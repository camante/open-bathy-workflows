from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import atl


def test_collect_training_points_from_atl03_filters_shallow_cluster_segments(monkeypatch):
    def fake_read(h5_file, laser_num='1'):
        if str(laser_num) != '1':
            raise RuntimeError('skip non-primary beams')
        lat = np.array([42.0, 42.00008, 42.00016, 42.00120, 42.00128, 42.00136], dtype=float)
        lon = np.array([-70.0, -70.0, -70.0, -70.0, -70.0, -70.0], dtype=float)
        h = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        conf = np.array([4, 4, 4, 4, 4, 4], dtype=float)
        return lat, lon, h, conf, np.array([], dtype=float), None, None, None, None

    def fake_bin(df, *args, **kwargs):
        return df

    def fake_surface(*args, **kwargs):
        return pd.DataFrame({"latitude": [42.0], "longitude": [-70.0], "surface_h": [0.0]})

    def fake_bottom(*args, **kwargs):
        return pd.DataFrame(
            {
                "longitude": [-70.0] * 6,
                "latitude": [42.0, 42.00008, 42.00016, 42.00120, 42.00128, 42.00136],
                "photon_height": [-0.5, -0.5, -0.5, -0.8, -1.2, -1.5],
                "depth_app": [0.5, 0.5, 0.5, 0.8, 1.2, 1.5],
                "ws_h": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "n_bottom": [3, 3, 3, 3, 3, 3],
                "n_subsurface": [10, 10, 10, 10, 10, 10],
                "frac_bottom": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            }
        )

    monkeypatch.setattr(atl, "read_atl03_basic", fake_read)
    monkeypatch.setattr(atl, "bin_data_safe", fake_bin)
    monkeypatch.setattr(atl, "infer_surface_from_atl03_binned", fake_surface)
    monkeypatch.setattr(atl, "infer_bottom_from_atl03_binned", fake_bottom)

    df, audit = atl.collect_training_points_from_atl03(
        [str(Path("fake_atl03.h5"))],
        lat_res=0.00005,
        height_res=0.25,
        aoi_str="-70.1/-69.9/41.9/42.1",
        atl03_conf_min=1,
        atl03_bottom_percentile=90.0,
        use_refraction=False,
        default_temp_c=20.0,
        default_wavelength_nm=532.0,
        min_bottom_photons=2,
        min_bottom_frac=0.05,
        min_depth_m=0.5,
        max_depth_m=5.0,
        debug_atl03_qc=False,
        return_audit=True,
    )

    assert len(df) == 3
    assert set(df["atl03_segment_admissible"].tolist()) == {True}
    assert audit["admissibility"]["candidate_segments"] == 2
    assert audit["admissibility"]["admissible_segments"] == 1
    assert audit["admissibility"]["rejected_segments"] == 1
    assert audit["admissibility"]["rejection_reason_counts"]["shallow_floor_cluster"] >= 1
    stage_names = {row["stage"] for row in audit["stage_counts"]}
    assert "candidate_segments" in stage_names
    assert "admissible_segments" in stage_names


def test_segment_atl03_candidates_preserves_east_west_track_span():
    bath_df = pd.DataFrame(
        {
            "longitude": [-70.0, -69.9998, -69.9996, -69.9994],
            "latitude": [42.0, 42.0, 42.0, 42.0],
            "photon_height": [-1.0, -1.1, -1.2, -1.3],
            "depth_m": [-1.0, -1.1, -1.2, -1.3],
            "ws_h": [0.0, 0.0, 0.0, 0.0],
            "n_bottom": [3, 3, 3, 3],
            "frac_bottom": [0.3, 0.3, 0.3, 0.3],
            "granule": ["g"] * 4,
            "beam": ["gt1"] * 4,
        }
    )
    seg_df = atl._segment_atl03_candidates(bath_df, max_gap_m=100.0)
    summary = atl._score_atl03_segment_admissibility(seg_df, min_track_span_m=10.0, shallow_floor_m=0.5)
    assert len(summary) == 1
    assert summary.loc[0, "segment_track_span_m"] > 40.0
    assert summary.loc[0, "primary_rejection_reason"] == "accepted"
