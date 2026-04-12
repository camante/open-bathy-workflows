import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import river_bank_longitudinal_fit as rblf


def test_fit_bank_longitudinal_points_respects_qc_actions_and_weights():
    bank_points = gpd.GeoDataFrame(
        {
            "component_id": ["7", "7", "7", "7"],
            "river_id": ["r", "r", "r", "r"],
            "side": ["left", "left", "left", "left"],
            "s_center_m": [0.0, 100.0, 200.0, 300.0],
            "bank_z_raw_m": [2.0, 7.5, 2.2, 2.3],
            "bank_z_m": [2.0, 7.5, 2.2, 2.3],
            "bank_z_final_m": [2.0, np.nan, 2.2, 2.3],
            "bank_longitudinal_ref_m": [2.0, 2.1, 2.2, 2.3],
            "continuity_weight": [1.0, 1.0, 1.0, 1.0],
            "bank_high_contamination_suspect": [False, True, False, False],
            "bank_strong_contamination": [False, True, False, False],
            "bank_qc_action": ["keep", "reject", "keep", "keep"],
            "bank_qc_weight": [1.0, 0.0, 1.0, 1.0],
        },
        geometry=[Point(0,0), Point(1,0), Point(2,0), Point(3,0)],
        crs="EPSG:32619",
    )
    endpoint_meta = {"7": {"station_direction": "increasing_station_downstream"}}
    fit, diag = rblf.fit_bank_longitudinal_points(bank_points, endpoint_meta=endpoint_meta)
    rejected = fit.loc[fit["s_center_m"] == 100.0].iloc[0]
    assert np.isnan(float(rejected["bank_fit_smoothed_m"]))
    assert np.isnan(float(rejected["bank_fit_monotone_m"]))
    stats = diag["7:left"]
    assert int(stats["rejected_rows"]) == 1


def test_load_bank_points_defaults_missing_qc_columns(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "component_id": [7],
            "river_id": [1],
            "side": ["left"],
            "s_center_m": [0.0],
            "bank_z_m": [2.0],
        },
        geometry=[Point(0,0)],
        crs="EPSG:32619",
    )
    path = tmp_path / "banks.gpkg"
    gdf.to_file(path, driver="GPKG")
    loaded = rblf.load_bank_points(path)
    row = loaded.iloc[0]
    assert str(row["bank_qc_action"]) == "keep"
    assert float(row["bank_qc_weight"]) == 1.0
