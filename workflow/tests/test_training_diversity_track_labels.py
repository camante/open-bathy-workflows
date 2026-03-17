import pandas as pd

from training_diversity import analyze_track_diversity, generate_recommendations


def test_analyze_track_diversity_avoids_nan_nan_labels():
    df = pd.DataFrame({
        "source": ["extra_xyz:ehydro", "extra_xyz:ehydro", None],
        "granule": [None, None, None],
        "beam": [None, None, None],
    })
    n_unique, counts = analyze_track_diversity(df, track_col="track_id", date_col="datetime", source_col="source")
    assert n_unique == 2
    assert "nan::nan" not in counts
    assert counts["extra_xyz:ehydro"] == 2
    assert counts["unknown"] == 1


def test_generate_recommendations_uses_support_group_wording():
    recs = generate_recommendations(0.2, [], 0.8, [], 0.8, 1, {"unknown": 10})
    assert any("support groups/passes" in r for r in recs)
    assert not any("ICESat-2 passes" in r for r in recs)
