from contract_tests import ContractTestSuite
from final_dem_contract import build_final_dem_contract_summary
from final_dem_policy import default_final_dem_policy
from simple_river_stage_contract import simple_river_stage_status_placeholder


def test_contract_test_suite_registers_bundle_a_wiring_test():
    suite = ContractTestSuite()
    suite.add_standard_tests()
    names = [t.name for t in suite.tests]
    assert "bundle_a_contract_wiring" in names


def test_bundle_a_contract_summary_matches_policy_defaults():
    policy = default_final_dem_policy()
    stage_status = simple_river_stage_status_placeholder()
    summary = build_final_dem_contract_summary({
        "simple_river_stage_status": stage_status,
        "legacy_transitional_artifacts_present": [],
    }, policy=policy)
    assert summary["current_route_mode"] == policy.current_route_mode
    assert summary["target_route_mode"] == policy.target_route_mode
    assert summary["final_dem_filename"] == policy.final_dem_filename
    assert summary["internal_final_dem_filename"] == policy.internal_final_dem_filename
    assert set(summary["simple_river_stage_status"]) == set(stage_status)
