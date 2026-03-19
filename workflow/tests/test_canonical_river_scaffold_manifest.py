import json
from pathlib import Path

from canonical_river_scaffold import get_river_aoi_domains, write_scaffold_manifest


def test_write_scaffold_manifest_records_artifact_lineage(tmp_path: Path):
    out = tmp_path / 'river_scaffold_manifest.json'
    domains = get_river_aoi_domains('-71/-70.75/42.75/43', 2.0, 60.0)
    write_scaffold_manifest(
        out,
        domains=domains,
        network_gpkg='network.gpkg',
        provenance_lock='lock.json',
        extra={'network_status': 'success'},
    )
    data = json.loads(out.read_text(encoding='utf-8'))
    assert data['scaffold_product_type'] == 'canonical_river_scaffold_domain_manifest'
    assert data['network_gpkg'] == 'network.gpkg'
    assert data['provenance_lock'] == 'lock.json'
    assert data['network_status'] == 'success'
    assert 'trusted export interior' in data['trusted_export_role']



def test_build_scaffold_manifest_accepts_scaffold_products_metadata():
    from canonical_river_scaffold import build_scaffold_manifest, get_river_aoi_domains
    domains = get_river_aoi_domains("0/1/2/3", halo_km=1.0, trusted_halo_m=50.0)
    payload = build_scaffold_manifest(domains=domains, network_gpkg="net.gpkg", provenance_lock="lock")
    assert payload["scaffold_product_type"] == "canonical_river_scaffold_domain_manifest"
