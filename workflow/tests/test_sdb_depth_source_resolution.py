import json
from pathlib import Path

import predict


def test_predict_prefers_physics_when_selected():
    meta = {
        "max_depth_sdb_auto": 1.5,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_options": {"selected_source": "physics"},
    }
    val, src = predict._get_max_depth_from_meta(meta)
    assert val == 25.4
    assert src == "max_depth_sdb_auto_physics"


def test_predict_prefers_combined_when_selected():
    meta = {
        "max_depth_sdb_auto": 1.5,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_sdb_combined": 18.0,
        "max_depth_options": {"selected_source": "combined"},
    }
    val, src = predict._get_max_depth_from_meta(meta)
    assert val == 18.0
    assert src == "max_depth_sdb_combined"
