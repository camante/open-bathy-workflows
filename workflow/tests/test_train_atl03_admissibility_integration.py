from __future__ import annotations

import tempfile
from pathlib import Path

from conftest import _install_mocks, build_training_df

_install_mocks()


def test_train_sdb_model_records_atl03_admissibility_summary():
    import train

    df = build_training_df().copy()
    with tempfile.TemporaryDirectory() as d:
        rf, _, _, _, meta = train.train_sdb_model(
            train_df=df,
            max_depth_sdb=20.0,
            seed=42,
            plots_dir=Path(d) / "plots",
            water_class="ocean",
            use_stumpf_depth=True,
            min_training_points_for_sdb=50,
            atl03_admissibility_summary={
                "candidate_segments": 12,
                "admissible_segments": 9,
                "rejected_segments": 3,
                "retained_points": 180,
                "rejection_reason_counts": {"shallow_floor_cluster": 2},
            },
        )
    assert hasattr(rf, "predict")
    assert meta["atl03_admissibility"]["candidate_segments"] == 12
    assert meta["training_qc"]["final_fit_rows"] > 0
    assert meta["training_qc"]["final_fit_features"] > 0
