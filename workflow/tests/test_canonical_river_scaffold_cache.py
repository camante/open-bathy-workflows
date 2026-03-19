import json
from pathlib import Path

from canonical_river_scaffold import (
    cached_scaffold_paths,
    get_river_aoi_domains,
    scaffold_cache_hit,
    scaffold_identity_hash,
    write_scaffold_manifest,
)


def test_scaffold_identity_hash_changes_with_halo():
    d1 = get_river_aoi_domains('-71/-70.75/42.75/43', 2.0, 60.0)
    d2 = get_river_aoi_domains('-71/-70.75/42.75/43', 3.0, 60.0)
    h1 = scaffold_identity_hash(domains=d1, hydrography_source='tnm', tnm_dataset='NHD', tnm_enable=True, snap_m=25.0)
    h2 = scaffold_identity_hash(domains=d2, hydrography_source='tnm', tnm_dataset='NHD', tnm_enable=True, snap_m=25.0)
    assert h1 != h2


def test_cached_scaffold_paths_and_cache_hit(tmp_path: Path):
    domains = get_river_aoi_domains('-71/-70.75/42.75/43', 2.0, 60.0)
    paths = cached_scaffold_paths(
        cache_root=tmp_path,
        domains=domains,
        hydrography_source='tnm',
        tnm_dataset='NHD',
        tnm_enable=True,
        snap_m=25.0,
    )
    network = Path(paths['network_gpkg'])
    manifest = Path(paths['manifest'])
    network.parent.mkdir(parents=True, exist_ok=True)
    network.write_text('placeholder', encoding='utf-8')
    write_scaffold_manifest(
        manifest,
        domains=domains,
        network_gpkg=str(network),
        provenance_lock='lock.json',
        extra={'network_status': 'success', 'cache_hit': False},
    )
    assert scaffold_cache_hit(manifest_path=manifest, expected_network_gpkg=network)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert data['network_status'] == 'success'



def test_scaffold_product_paths_expose_derived_products(tmp_path):
    from canonical_river_scaffold import get_river_aoi_domains, scaffold_product_paths
    domains = get_river_aoi_domains("0/1/2/3", halo_km=2.0, trusted_halo_m=60.0)
    paths = scaffold_product_paths(cache_root=tmp_path, domains=domains, hydrography_source="tnm", tnm_dataset="nhd", tnm_enable=True, snap_m=5.0)
    assert paths["graph_edges_gpkg"].endswith("scaffold_graph_edges.gpkg")
    assert paths["stationing_json"].endswith("scaffold_stationing_basis.json")
