# -*- coding: utf-8 -*-
"""Regression tests for ATL audit collector interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atl


def test_collect_training_points_from_atl03_supports_return_audit_empty():
    df, audit = atl.collect_training_points_from_atl03(
        [],
        lat_res=0.00005,
        height_res=0.25,
        aoi_str="-71/-70/41/42",
        atl03_conf_min=1,
        atl03_bottom_percentile=90.0,
        use_refraction=False,
        default_temp_c=20.0,
        default_wavelength_nm=532.0,
        min_bottom_photons=2,
        min_bottom_frac=0.1,
        min_depth_m=0.5,
        max_depth_m=30.0,
        debug_atl03_qc=False,
        return_audit=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(audit, dict)
    assert audit.get("source") == "atl03"
    assert any(row.get("stage") == "input_files" for row in audit.get("stage_counts", []))
    assert int(audit.get("retained_points", 0)) == 0


def test_collect_training_points_from_atl24_supports_return_audit_empty():
    df, audit = atl.collect_training_points_from_atl24(
        [],
        aoi_str="-71/-70/41/42",
        max_depth_m=30.0,
        train_relax_buffer=0.01,
        limit_train_samples=None,
        seed=42,
        return_audit=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(audit, dict)
    assert audit.get("source") == "atl24"
    assert any(row.get("stage") == "input_files" for row in audit.get("stage_counts", []))
    assert int(audit.get("retained_points", 0)) == 0


def test_summarize_atl_raw_to_retained_audit_handles_missing_audits():
    df03 = pd.DataFrame({"longitude": [-70.0, -70.0], "latitude": [41.0, 41.0], "depth_m": [-1.0, -1.5], "granule": ["g1", "g1"], "beam": ["gt1", "gt1"]})
    df24 = pd.DataFrame({"longitude": [-70.0], "latitude": [41.0], "depth_m": [-2.0], "granule": ["g2"], "beam": ["gt2"]})
    summary, rows = atl.summarize_atl_raw_to_retained_audit(None, None, df_atl03=df03, df_atl24=df24)
    assert summary["combined"]["retained_points"] == 3
    assert summary["cudem_framework_assessment"]["recommended_use"] == "multi_source_atl"
    assert isinstance(rows, pd.DataFrame)
    assert set(rows["source"]) == {"atl03", "atl24"}


def test_cmr_latest_concept_id_survives_bad_query_and_selects_latest_version(monkeypatch):
    class _Resp:
        def __init__(self, payload=None, error=False):
            self._payload = payload or {}
            self._error = error
        def raise_for_status(self):
            if self._error:
                raise RuntimeError("400 bad request")
        def json(self):
            return self._payload

    class _Requests:
        def __init__(self):
            self.calls = 0
        def get(self, url, params=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                return _Resp(error=True)
            return _Resp({
                "feed": {
                    "entry": [
                        {"id": "C1-OLD", "version_id": "005", "revision_id": 3},
                        {"id": "C1-NEW", "version_id": "006", "revision_id": 1},
                    ]
                }
            })

    req = _Requests()
    monkeypatch.setattr(atl, "_lazy_requests", lambda: req)
    cid = atl._cmr_latest_concept_id("ATL03")
    assert cid == "C1-NEW"
    assert req.calls == 2
