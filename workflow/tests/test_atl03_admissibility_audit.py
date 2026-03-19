from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import atl


def test_write_atl03_admissibility_artifacts_outputs_csv_and_json(tmp_path: Path):
    audit = {
        "admissibility": {
            "candidate_segments": 2,
            "admissible_segments": 1,
            "rejected_segments": 1,
            "candidate_points": 6,
            "retained_points": 3,
            "rejected_points": 3,
            "rejection_reason_counts": {"shallow_floor_cluster": 1},
        },
        "admissibility_summary_rows": [
            {
                "atl03_segment_id": "g::b::seg0",
                "segment_points": 3,
                "segment_track_span_m": 20.0,
                "ws_height_std_m": 0.0,
                "near_floor_fraction": 1.0,
                "median_frac_bottom": 0.3,
                "median_n_bottom": 3.0,
                "depth_range_m": 0.0,
                "admissible": False,
                "rejection_reason": "shallow_floor_cluster",
                "longitude": -70.0,
                "latitude": 42.0,
            },
            {
                "atl03_segment_id": "g::b::seg1",
                "segment_points": 3,
                "segment_track_span_m": 25.0,
                "ws_height_std_m": 0.0,
                "near_floor_fraction": 0.0,
                "median_frac_bottom": 0.3,
                "median_n_bottom": 3.0,
                "depth_range_m": 0.7,
                "admissible": True,
                "rejection_reason": "accepted",
                "longitude": -70.0,
                "latitude": 42.001,
            },
        ],
    }
    outputs = atl.write_atl03_admissibility_artifacts(audit=audit, out_dir=tmp_path)
    assert outputs["csv"] is not None and Path(outputs["csv"]).exists()
    assert outputs["json"] is not None and Path(outputs["json"]).exists()
    df = pd.read_csv(outputs["csv"])
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert len(df) == 2
    assert payload["rejected_segments"] == 1



def test_build_atl03_admissibility_audit_reports_primary_reason_counts():
    summary = pd.DataFrame(
        [
            {"segment_points": 4, "admissible": False, "rejection_reason": "short_track_span;weak_bottom_support", "primary_rejection_reason": "short_track_span", "segment_track_span_m": 9.0, "depth_range_m": 0.3},
            {"segment_points": 5, "admissible": True, "rejection_reason": "accepted", "primary_rejection_reason": "accepted", "segment_track_span_m": 21.0, "depth_range_m": 1.1},
        ]
    )
    audit = atl._build_atl03_admissibility_audit(summary)
    assert audit["primary_rejection_reason_counts"]["short_track_span"] == 1
    assert audit["retained_fraction"] == 5 / 9
    assert audit["median_track_span_m"] == 15.0
