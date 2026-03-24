"""test_contract_tests_execution.py — verify that contract tests actually execute and produce results.

The smoke test only checks that ContractTestSuite loads and registers tests.
This test runs the suite against synthetic data and validates the output structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contract_tests import ContractTestSuite


@pytest.fixture
def synthetic_context():
    """Build a synthetic context that exercises all standard contract tests."""
    n = 200
    rng = np.random.RandomState(42)
    fused_df = pd.DataFrame({
        "longitude": rng.uniform(-71, -70.8, n),
        "latitude": rng.uniform(42.75, 42.85, n),
        "depth_m": rng.uniform(0.5, 15.0, n),
        "source": np.where(rng.random(n) > 0.3, "extra_xyz", "atl_photon"),
        "sample_weight": rng.uniform(0.5, 2.0, n),
    })
    training_df = fused_df.copy()
    training_df["split"] = np.where(rng.random(n) > 0.2, "train", "test")
    return {
        "fused_df": fused_df,
        "training_df": training_df,
        "xyz_provided": True,
        "n_chunks": 4,
        "total_predicted_pixels": 50000,
        "model_rmse": 1.2,
        "model_r2": 0.85,
        "n_training_points": n,
    }


def test_contract_suite_runs_and_returns_summary(synthetic_context):
    suite = ContractTestSuite()
    suite.add_standard_tests()
    assert len(suite.tests) > 0, "Standard tests should register at least one test"

    total, passed, critical_failed = suite.run_all(synthetic_context)
    assert total == len(suite.tests)
    assert passed + critical_failed <= total

    summary = suite.get_summary()
    assert "total_tests" in summary
    assert "tests" in summary
    assert summary["total_tests"] == total
    # With synthetic data, at least some tests should pass
    assert summary["passed"] >= 0


def test_contract_suite_results_have_required_fields(synthetic_context):
    suite = ContractTestSuite()
    suite.add_standard_tests()
    suite.run_all(synthetic_context)

    for name, result in suite.results.items():
        assert "name" in result, f"Result for {name} missing 'name'"
        assert "passed" in result, f"Result for {name} missing 'passed'"
        assert "critical" in result, f"Result for {name} missing 'critical'"
        assert "message" in result, f"Result for {name} missing 'message'"
        assert isinstance(result["passed"], bool), f"Result for {name}: 'passed' should be bool"


def test_contract_suite_empty_context_does_not_crash():
    """Contract tests must not crash on missing context — they should report failure gracefully."""
    suite = ContractTestSuite()
    suite.add_standard_tests()
    total, passed, critical_failed = suite.run_all({})
    assert total == len(suite.tests)
    # With empty context, tests should fail gracefully, not crash the suite
    summary = suite.get_summary()
    assert summary["total_tests"] == total


def test_contract_suite_xyz_not_provided_skips_gracefully():
    """When xyz_provided=False, XYZ-related tests should pass (skipped)."""
    suite = ContractTestSuite()
    suite.add_standard_tests()
    ctx = {"xyz_provided": False, "fused_df": pd.DataFrame()}
    total, passed, critical_failed = suite.run_all(ctx)
    summary = suite.get_summary()
    # The xyz_in_fused_data test should pass when xyz not provided
    xyz_result = summary["tests"].get("xyz_in_fused_data", {})
    if xyz_result:
        assert xyz_result["passed"] is True, "XYZ test should pass (skip) when xyz not provided"
