from canonical_river_scaffold import get_river_aoi_domains, scaffold_domain_metadata


def test_get_river_aoi_domains_returns_dataclass_and_halo_metadata():
    domains = get_river_aoi_domains("-71/-70.75/42.75/43", 2.0, 60.0)
    as_dict = domains.as_dict()
    assert as_dict["export_aoi"] == "-71/-70.75/42.75/43"
    assert as_dict["solve_aoi"] == as_dict["scaffold_aoi"]
    assert as_dict["halo_km"] == 2.0
    assert as_dict["trusted_halo_m"] == 60.0


def test_scaffold_domain_metadata_explains_roles():
    meta = scaffold_domain_metadata("-71/-70.75/42.75/43", 2.0, 60.0)
    assert "halo-expanded" in meta["solve_role"]
    assert "trusted export interior" in meta["trusted_export_role"]
