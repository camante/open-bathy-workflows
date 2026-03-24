from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString

from dominant_trunk import select_dominant_trunks


def test_dominant_trunk_selects_single_major_component_by_default():
    rivers = gpd.GeoDataFrame(
        {
            "from_node": [1, 2, 10],
            "to_node": [2, 3, 11],
            "length_m": [100.0, 100.0, 500.0],
            "streamorde": [6, 6, 2],
            "totdasqkm": [30.0, 30.0, 1.0],
            "river_id": ["a", "b", "c"],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)]), LineString([(10, 0), (20, 0)])],
        },
        crs="EPSG:32619",
    )
    trunk, diag = select_dominant_trunks(rivers)
    assert len(trunk) == 2
    assert diag["selected_component_ids"] == [0]
    assert diag["include_all_components"] is False
