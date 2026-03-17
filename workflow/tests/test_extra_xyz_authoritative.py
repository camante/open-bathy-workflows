from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from atl import load_extra_xyz_points
from spatial_sampling import adaptive_spatial_sample, SamplingConfig
from train import apply_stumpf_residual_filter


def test_load_extra_xyz_points_supports_parquet_subset() -> None:
    fake_df = pd.DataFrame({
        "x": [-70.90, -70.89],
        "y": [42.80, 42.81],
        "z": [-4.0, -6.0],
        "_src_file": ["hydronos_subset", "ehydro_subset"],
    })
    with TemporaryDirectory() as td:
        p = Path(td) / "river_soundings_subset.parquet"
        p.write_bytes(b"PAR1")
        with mock.patch("pandas.read_parquet", return_value=fake_df):
            out = load_extra_xyz_points([str(p)], crs="EPSG:4326", aoi_str="-71/-70/42/43")
    assert len(out) == 2
    assert set(out.columns) == {"longitude", "latitude", "depth_m", "source"}
    assert set(out["source"].astype(str)) == {"hydronos_subset", "ehydro_subset"}
    assert np.all(out["depth_m"].to_numpy() < 0)


def test_stumpf_residual_filter_preserves_authoritative_extra_xyz() -> None:
    n_auth = 240
    n_atl = 240
    auth_depth = np.linspace(-2.0, -18.0, n_auth)
    atl_depth = np.linspace(-2.0, -18.0, n_atl)
    atl_stumpf = np.linspace(0.8, 1.6, n_atl)
    # Make the authoritative soundings look optically inconsistent on purpose;
    # they should still be preserved because they are authoritative anchors.
    auth_stumpf = np.full(n_auth, 0.9)
    df = pd.DataFrame({
        "stumpf_idx": np.concatenate([auth_stumpf, atl_stumpf]),
        "depth_m": np.concatenate([auth_depth, atl_depth]),
        "source": ["extra_xyz:hydronos"] * n_auth + ["atl03"] * n_atl,
        "source_norm": ["extra_xyz:hydronos"] * n_auth + ["atl03"] * n_atl,
    })
    out = apply_stumpf_residual_filter(df, min_points=50, min_inliers=20, random_state=42)
    auth_out = out[out["source_norm"].str.startswith("extra_xyz")]
    assert len(auth_out) == n_auth


def test_adaptive_sampling_global_cap_preserves_non_xyz_sources() -> None:
    rng = np.random.default_rng(42)

    def _block(n: int, source: str, lon0: float, lat0: float):
        return pd.DataFrame({
            "longitude": lon0 + 0.01 * rng.random(n),
            "latitude": lat0 + 0.01 * rng.random(n),
            "depth_m": -(1.0 + 20.0 * rng.random(n)),
            "source": [source] * n,
            "sample_weight": np.ones(n, dtype=float),
        })

    df = pd.concat([
        _block(4000, "extra_xyz:hydronos", -70.90, 42.80),
        _block(4000, "extra_xyz:ehydro", -70.88, 42.81),
        _block(500, "atl24", -70.86, 42.82),
        _block(500, "atl03", -70.84, 42.83),
    ], ignore_index=True)

    sampled, _stats = adaptive_spatial_sample(
        df,
        sampling_config=SamplingConfig(target_total_points=500, min_points_per_source=40, depth_bins=6),
    )
    counts = sampled["source"].astype(str).value_counts().to_dict()
    assert counts.get("atl24", 0) >= 40
    assert counts.get("atl03", 0) >= 40
