import json
import types
from pathlib import Path

import pandas as pd

from canonical_river_scaffold import (
    persist_scaffold_products,
    scaffold_product_paths,
    scaffold_products_complete,
    get_river_aoi_domains,
)


class FakeGDF(pd.DataFrame):
    @property
    def _constructor(self):
        return FakeGDF

    def to_file(self, path, layer=None, driver=None):
        Path(path).write_text(json.dumps({"layer": layer, "driver": driver, "rows": len(self)}), encoding="utf-8")


def test_persist_scaffold_products_exports_mainstem_and_stationing(tmp_path: Path, monkeypatch):
    edges = FakeGDF(
        {
            "from_node": [1, 2, 3],
            "to_node": [2, 3, 4],
            "component_id": [10, 10, 99],
            "length_m": [100.0, 120.0, 50.0],
            "s_m_from": [0.0, 100.0, 0.0],
            "s_m_to": [100.0, 220.0, 50.0],
        }
    )
    nodes = FakeGDF({"node_id": [1, 2, 3, 4]})

    fake_gpd = types.SimpleNamespace(
        read_file=lambda path, layer=None: edges if layer == "graph_edges" else nodes,
    )
    monkeypatch.setitem(__import__("sys").modules, "geopandas", fake_gpd)

    network = tmp_path / "river_network.gpkg"
    network.write_text("placeholder", encoding="utf-8")
    domains = get_river_aoi_domains("-71/-70.75/42.75/43", halo_km=2.0, trusted_halo_m=60.0)
    product_paths = scaffold_product_paths(
        cache_root=tmp_path,
        domains=domains,
        hydrography_source="tnm",
        tnm_dataset="NHD",
        tnm_enable=True,
        snap_m=25.0,
    )
    meta = persist_scaffold_products(network_gpkg=network, product_paths=product_paths)
    assert meta["mainstem_component_id"] == "10"
    assert Path(meta["mainstem_edges_gpkg"]).exists()
    assert Path(meta["stationing_json"]).exists()
    assert Path(meta["summary_json"]).exists()
    stationing = json.loads(Path(meta["stationing_json"]).read_text(encoding="utf-8"))
    assert stationing["stationing_origin_m"] == 0.0
    assert stationing["stationing_terminus_m"] == 220.0
    summary = json.loads(Path(meta["summary_json"]).read_text(encoding="utf-8"))
    assert summary["mainstem_edges_count"] == 2
    assert summary["topology_signature"]
    assert scaffold_products_complete(product_paths)
