from provenance_schema import (
    ProvenanceClass,
    provenance_class_code_from_name,
    provenance_class_is_authoritative_locked,
    provenance_schema_summary,
)
from river_support_roles import canonical_river_support_class, river_support_roles_summary
from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    STAGE_FINAL_DEM,
    simple_river_stage_contract_summary,
    simple_river_stage_filename,
    simple_river_stage_sequence,
    simple_river_stage_status_placeholder,
)
from support_classes import (
    SupportClass,
    support_class_code_from_name,
    support_class_is_final_authoritative,
    support_schema_summary,
)


def test_support_schema_bidirectional():
    assert support_class_code_from_name("authoritative_locked") == int(SupportClass.AUTHORITATIVE_LOCKED)
    summary = support_schema_summary()
    assert summary["codes"]["1"] == "authoritative_locked"


def test_provenance_schema_bidirectional():
    assert provenance_class_code_from_name("authoritative_locked") == int(ProvenanceClass.AUTHORITATIVE_LOCKED)
    assert provenance_class_is_authoritative_locked(int(ProvenanceClass.AUTHORITATIVE_LOCKED))
    summary = provenance_schema_summary()
    assert summary["support_family_hints"]["10"] == "authoritative_locked"


def test_river_support_roles_summary_and_classification():
    summary = river_support_roles_summary()
    assert "authoritative_interior" in summary["canonical_classes"]
    assert canonical_river_support_class(station_authoritative_bed_support_present=True) == "authoritative_interior"


def test_simple_river_stage_contract_basics():
    assert simple_river_stage_sequence()[-1] == STAGE_FINAL_DEM
    assert simple_river_stage_filename(STAGE_FINAL_DEM) == "combined/DEM_enhanced.tif"
    placeholder = simple_river_stage_status_placeholder()
    assert placeholder[STAGE_FINAL_DEM]["implemented"] is False
    summary = simple_river_stage_contract_summary()
    assert summary["route_modes"]["current_default"] == ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION
    assert summary["route_modes"]["target"] == ROUTE_MODE_SIMPLE_RIVER_PLAN_V1


def test_support_authoritative_predicate():
    assert support_class_is_final_authoritative(int(SupportClass.AUTHORITATIVE_LOCKED))


from final_dem_policy import default_final_dem_policy
from final_dem_contract import build_final_dem_contract_summary
from final_route_contract import allowed_river_structural_artifacts


def test_final_dem_policy_defaults_bundle_a_patch2():
    policy = default_final_dem_policy()
    assert policy.final_dem_filename == "DEM_enhanced.tif"
    assert policy.internal_final_dem_filename == "conditioned_final_dem_internal.tif"
    assert policy.write_final_dem_once is True
    assert policy.verify_only_postwrite is True


def test_final_dem_contract_summary_exposes_route_modes():
    summary = build_final_dem_contract_summary({})
    assert summary["current_route_mode"] == "legacy_structured_transition"
    assert summary["target_route_mode"] == "simple_river_plan_v1"
    assert summary["final_dem_filename"] == "DEM_enhanced.tif"
    assert "simple_river_stage_status" in summary


def test_allowed_river_structural_artifacts_switches_by_route_mode():
    legacy = allowed_river_structural_artifacts("legacy_structured_transition")
    target = allowed_river_structural_artifacts("simple_river_plan_v1")
    assert "guide_points" in legacy
    assert "river_centerline_points.gpkg" in target
    assert "guide_points" not in target
