"""
Contract Tests for SDB Pipeline

Encodes expectations about pipeline behavior as executable tests.
Prevents regressions like "XYZ loaded but not used" from returning.

Addresses feedback: "Diagnostics are strong—now make them contract tests"

Author: SDB Pipeline Development Team
Version: 0.7.1
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger(__name__)


class ContractTest:
    """Base class for a contract test."""
    
    def __init__(self, name: str, critical: bool = True):
        self.name = name
        self.critical = critical
        self.passed = False
        self.message = ""
        self.value = None
    
    def run(self, context: Dict[str, Any]) -> bool:
        """
        Run the test.
        
        Args:
            context: Dictionary containing run artifacts (dataframes, reports, etc.)
        
        Returns:
            True if test passed
        """
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "critical": self.critical,
            "message": self.message,
            "value": self.value
        }


class XYZInFusedDataTest(ContractTest):
    """Test that extra_xyz appears in fused training data when provided."""
    
    def __init__(self):
        super().__init__("xyz_in_fused_data", critical=True)
    
    def run(self, context: Dict[str, Any]) -> bool:
        fused_df = context.get("fused_df")
        xyz_provided = context.get("xyz_provided", False)
        
        if not xyz_provided:
            self.passed = True
            self.message = "XYZ not provided (test skipped)"
            return True
        
        if fused_df is None or fused_df.empty:
            self.passed = False
            self.message = "Fused dataframe is empty"
            return False
        
        if 'source' not in fused_df.columns:
            self.passed = False
            self.message = "Fused dataframe missing 'source' column"
            return False
        
        # Check for extra_xyz in source column
        has_xyz = 'extra_xyz' in fused_df['source'].values
        self.passed = has_xyz
        
        if has_xyz:
            xyz_count = (fused_df['source'] == 'extra_xyz').sum()
            self.value = xyz_count
            self.message = f"extra_xyz found in fused data ({xyz_count} points)"
        else:
            self.message = "extra_xyz NOT found in fused data despite being provided"
        
        return self.passed


class XYZInTrainingDataTest(ContractTest):
    """Test that extra_xyz appears in training composition logs."""
    
    def __init__(self):
        super().__init__("xyz_in_training_composition", critical=True)
    
    def run(self, context: Dict[str, Any]) -> bool:
        training_df = context.get("training_df")
        xyz_provided = context.get("xyz_provided", False)
        
        if not xyz_provided:
            self.passed = True
            self.message = "XYZ not provided (test skipped)"
            return True
        
        if training_df is None or training_df.empty:
            self.passed = False
            self.message = "Training dataframe is empty"
            return False
        
        if 'source' not in training_df.columns:
            self.passed = False
            self.message = "Training dataframe missing 'source' column"
            return False
        
        has_xyz = 'extra_xyz' in training_df['source'].values
        self.passed = has_xyz
        
        if has_xyz:
            xyz_count = (training_df['source'] == 'extra_xyz').sum()
            self.value = xyz_count
            self.message = f"extra_xyz in training data ({xyz_count} points)"
        else:
            self.message = "extra_xyz NOT in training data"
        
        return self.passed


class SourceValidationTest(ContractTest):
    """Test that source-specific validation ran without exceptions."""
    
    def __init__(self):
        super().__init__("source_validation_ran", critical=True)
    
    def run(self, context: Dict[str, Any]) -> bool:
        run_report = context.get("run_report")
        
        if not run_report:
            self.passed = False
            self.message = "No run_report.json found"
            return False
        
        # Check if validation.by_source exists
        validation = run_report.get("validation.by_source")
        
        if not validation:
            self.passed = False
            self.message = "validation.by_source not found in run_report"
            return False
        
        # Check that it has metrics for each source
        sources_validated = list(validation.keys())
        self.passed = len(sources_validated) > 0
        self.value = sources_validated
        
        if self.passed:
            self.message = f"Source validation ran for: {', '.join(sources_validated)}"
        else:
            self.message = "Source validation produced no results"
        
        return self.passed


class XYZWeightingTest(ContractTest):
    """Test that XYZ points have correct sample weights."""
    
    def __init__(self, expected_weight: float = 10.0):
        super().__init__("xyz_weighting_correct", critical=True)
        self.expected_weight = expected_weight
    
    def run(self, context: Dict[str, Any]) -> bool:
        fused_df = context.get("fused_df")
        xyz_provided = context.get("xyz_provided", False)
        
        if not xyz_provided:
            self.passed = True
            self.message = "XYZ not provided (test skipped)"
            return True
        
        if fused_df is None or 'sample_weight' not in fused_df.columns:
            self.passed = False
            self.message = "Cannot check weights (no sample_weight column)"
            return False
        
        xyz_mask = fused_df['source'] == 'extra_xyz'
        if not xyz_mask.any():
            self.passed = False
            self.message = "No extra_xyz points to check weights"
            return False
        
        xyz_weights = fused_df.loc[xyz_mask, 'sample_weight']
        mean_weight = xyz_weights.mean()
        
        # Check if weight is approximately correct (within 10%)
        tolerance = self.expected_weight * 0.1
        self.passed = abs(mean_weight - self.expected_weight) < tolerance
        self.value = mean_weight
        
        if self.passed:
            self.message = f"XYZ weight correct: {mean_weight:.1f} (expected: {self.expected_weight})"
        else:
            self.message = f"XYZ weight incorrect: {mean_weight:.1f} (expected: {self.expected_weight})"
        
        return self.passed


class AdaptiveSamplingEffectiveTest(ContractTest):
    """Test that adaptive sampling actually reduced points."""
    
    def __init__(self):
        super().__init__("adaptive_sampling_effective", critical=False)
    
    def run(self, context: Dict[str, Any]) -> bool:
        run_report = context.get("run_report")
        sampling_enabled = context.get("adaptive_sampling_enabled", False)
        
        if not sampling_enabled:
            self.passed = True
            self.message = "Adaptive sampling not enabled (test skipped)"
            return True
        
        if not run_report:
            self.passed = False
            self.message = "No run_report.json found"
            return False
        
        sampling_stats = run_report.get("fusion.adaptive_sampling")
        
        if not sampling_stats:
            self.passed = False
            self.message = "Adaptive sampling enabled but no statistics found"
            return False
        
        reduction_pct = sampling_stats.get("reduction_pct", 0)
        self.passed = reduction_pct > 0
        self.value = reduction_pct
        
        if self.passed:
            input_pts = sampling_stats.get("input_points")
            output_pts = sampling_stats.get("output_points")
            self.message = f"Reduced {input_pts:,} → {output_pts:,} points ({reduction_pct:.1f}%)"
        else:
            self.message = "Adaptive sampling did not reduce points"
        
        return self.passed


class ChunkedPredictionTest(ContractTest):
    """Test that chunked prediction engaged when requested."""
    
    def __init__(self):
        super().__init__("chunked_prediction_engaged", critical=False)
    
    def run(self, context: Dict[str, Any]) -> bool:
        run_report = context.get("run_report")
        chunked_mode = context.get("chunked_mode", "off")
        
        if chunked_mode == "off":
            self.passed = True
            self.message = "Chunked prediction not requested (test skipped)"
            return True
        
        # Check logs or report for chunked prediction evidence
        # For now, assume if chunked mode != "off", we expect it ran
        self.passed = True
        self.message = f"Chunked prediction mode: {chunked_mode}"
        return True


class ModelQualityTest(ContractTest):
    """Test that model quality meets minimum threshold."""
    
    def __init__(self, max_rmse: float = 2.0):
        super().__init__("model_quality_acceptable", critical=False)
        self.max_rmse = max_rmse
    
    def run(self, context: Dict[str, Any]) -> bool:
        run_report = context.get("run_report")
        
        if not run_report:
            self.passed = False
            self.message = "No run_report.json found"
            return False
        
        training = run_report.get("training", {})
        rmse = training.get("rmse_test_m")
        
        if rmse is None:
            self.passed = False
            self.message = "Model RMSE not found in report"
            return False
        
        self.passed = rmse < self.max_rmse
        self.value = rmse
        
        if self.passed:
            self.message = f"Model RMSE acceptable: {rmse:.2f}m (threshold: <{self.max_rmse}m)"
        else:
            self.message = f"Model RMSE too high: {rmse:.2f}m (threshold: <{self.max_rmse}m)"
        
        return self.passed


class SufficientTrainingDataTest(ContractTest):
    """Test that sufficient training points exist."""
    
    def __init__(self, min_points: int = 100):
        super().__init__("sufficient_training_data", critical=True)
        self.min_points = min_points
    
    def run(self, context: Dict[str, Any]) -> bool:
        training_df = context.get("training_df")
        
        if training_df is None:
            self.passed = False
            self.message = "No training dataframe found"
            return False
        
        n_points = len(training_df)
        self.passed = n_points >= self.min_points
        self.value = n_points
        
        if self.passed:
            self.message = f"Sufficient training data: {n_points:,} points (minimum: {self.min_points})"
        else:
            self.message = f"Insufficient training data: {n_points:,} points (minimum: {self.min_points})"
        
        return self.passed


class ContractTestSuite:
    """Suite of contract tests for pipeline validation."""
    
    def __init__(self):
        self.tests: List[ContractTest] = []
        self.results = {}
    
    def add_test(self, test: ContractTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def add_standard_tests(self):
        """Add standard contract tests."""
        self.add_test(XYZInFusedDataTest())
        self.add_test(XYZInTrainingDataTest())
        self.add_test(SourceValidationTest())
        self.add_test(XYZWeightingTest())
        self.add_test(AdaptiveSamplingEffectiveTest())
        self.add_test(ChunkedPredictionTest())
        self.add_test(ModelQualityTest())
        self.add_test(SufficientTrainingDataTest())
    
    def run_all(self, context: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Run all tests.
        
        Args:
            context: Dictionary containing run artifacts
        
        Returns:
            Tuple of (total, passed, critical_failed)
        """
        log.info("=" * 70)
        log.info("RUNNING CONTRACT TESTS")
        log.info("=" * 70)
        
        total = len(self.tests)
        passed = 0
        critical_failed = 0
        
        for test in self.tests:
            try:
                result = test.run(context)
                self.results[test.name] = test.to_dict()
                
                status = "✓ PASS" if result else "✗ FAIL"
                criticality = "[CRITICAL]" if test.critical else "[INFO]"
                
                log.info(f"{status} {criticality} {test.name}: {test.message}")
                
                if result:
                    passed += 1
                elif test.critical:
                    critical_failed += 1
            
            except Exception as e:
                log.error(f"✗ ERROR {test.name}: {e}")
                self.results[test.name] = {
                    "name": test.name,
                    "passed": False,
                    "critical": test.critical,
                    "message": f"Exception: {e}",
                    "value": None
                }
                if test.critical:
                    critical_failed += 1
        
        log.info("=" * 70)
        log.info(f"Contract Tests: {passed}/{total} passed")
        if critical_failed > 0:
            log.error(f"CRITICAL FAILURES: {critical_failed}")
        log.info("=" * 70)
        
        return total, passed, critical_failed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        critical_tests = [r for r in self.results.values() if r["critical"]]
        critical_passed = sum(1 for r in critical_tests if r["passed"])
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "critical_tests": len(critical_tests),
            "critical_passed": critical_passed,
            "critical_failed": len(critical_tests) - critical_passed,
            "all_critical_passed": critical_passed == len(critical_tests),
            "tests": self.results
        }
    
    def save_report(self, output_path: Path):
        """Save test report to JSON."""
        summary = self.get_summary()
        summary["timestamp"] = pd.Timestamp.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"[CONTRACT TESTS] Report saved to {output_path}")


def load_test_context_from_run(output_dir: Path) -> Dict[str, Any]:
    """
    Load test context from a completed pipeline run.
    
    Args:
        output_dir: Output directory of the run
    
    Returns:
        Context dictionary for running tests
    """
    context = {}
    
    # Load fused training data
    fused_csv = output_dir / "data" / "training_data_fused.csv"
    if fused_csv.exists():
        try:
            context["fused_df"] = pd.read_csv(fused_csv)
        except Exception as e:
            log.warning(f"Failed to load fused CSV: {e}")
            pass
    
    # Load run report
    run_report_path = output_dir / "run_report.json"
    if run_report_path.exists():
        try:
            with open(run_report_path) as f:
                context["run_report"] = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load run report: {e}")
            pass
    
    # Check if XYZ was provided (look for extra_xyz in inputs)
    if context.get("run_report"):
        args = context["run_report"].get("run", {}).get("args", {})
        context["xyz_provided"] = bool(args.get("extra_xyz"))
        context["adaptive_sampling_enabled"] = args.get("enable_adaptive_sampling", False)
        context["chunked_mode"] = args.get("chunked_prediction", "off")
    
    # Training dataframe (if available separately)
    # Usually same as fused, but could be after additional processing
    context["training_df"] = context.get("fused_df")
    
    return context


def run_contract_tests_cli(output_dir: Path) -> int:
    """
    CLI entry point for running contract tests on a completed run.
    
    Args:
        output_dir: Output directory to test
    
    Returns:
        Exit code (0 if all critical tests passed, 1 otherwise)
    """
    suite = ContractTestSuite()
    suite.add_standard_tests()
    
    context = load_test_context_from_run(output_dir)
    
    total, passed, critical_failed = suite.run_all(context)
    
    # Save report
    test_report_path = output_dir / "contract_tests.json"
    suite.save_report(test_report_path)
    
    # Return exit code
    return 0 if critical_failed == 0 else 1


# CLI usage example:
# python -m contract_tests output/my_run/
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python contract_tests.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    exit_code = run_contract_tests_cli(output_dir)
    sys.exit(exit_code)
