"""
Unified Bathymetry Report

Combines SDB and River pipeline reports into a single unified view.
Provides contract tests and verification across both methods.

Author: SDB Pipeline Development Team
Version: 0.7.1
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)


class UnifiedBathyReport:
    """
    Unified report for combined SDB + River bathymetry runs.
    """
    
    def __init__(self, output_dir: Path, version: str = "0.7.1"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            "unified_report": True,
            "version": version,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "duration_seconds": None,
            "methods_requested": [],
            "priority_method": None,
            "methods": {},
            "fusion": {},
            "contract_tests": {},
            "errors": []
        }
    
    def set_methods(self, methods: List[str], priority: str):
        """Set which methods were requested and priority."""
        self.data["methods_requested"] = methods
        self.data["priority_method"] = priority
    
    def add_method_result(
        self,
        method: str,
        status: str,
        report_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        **kwargs
    ):
        """
        Add results from a method (sdb or river).
        
        Args:
            method: 'sdb' or 'river'
            status: 'success', 'failed', 'skipped'
            report_path: Path to method's report JSON
            output_path: Path to method's output raster
            **kwargs: Additional method-specific data
        """
        self.data["methods"][method] = {
            "status": status,
            "report_path": str(report_path) if report_path else None,
            "output_path": str(output_path) if output_path else None,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        # Load and attach method report if available
        if report_path and report_path.exists():
            try:
                with open(report_path) as f:
                    method_report = json.load(f)
                self.data["methods"][method]["report_summary"] = self._summarize_method_report(
                    method, method_report
                )
            except Exception as e:
                log.warning(f"[UNIFIED REPORT] Could not load {method} report: {e}")
    
    def _summarize_method_report(self, method: str, report: Dict) -> Dict:
        """Extract key summary from method report."""
        summary = {
            "status": report.get("status", "unknown"),
            "duration_seconds": report.get("duration_seconds")
        }
        
        if method == "sdb":
            # Extract SDB-specific summary
            if "fusion" in report:
                summary["training_points"] = report.get("fusion", {}).get("schema_weighted.final", {}).get("n")
            
            if "training" in report:
                summary["rmse_m"] = report.get("training", {}).get("rmse_test_m")
                summary["r2"] = report.get("training", {}).get("r2_test")
            
            # Adaptive sampling if used
            if "fusion.adaptive_sampling" in report:
                sampling = report["fusion.adaptive_sampling"]
                summary["adaptive_sampling"] = {
                    "enabled": True,
                    "reduction_pct": sampling.get("reduction_pct"),
                    "output_points": sampling.get("output_points")
                }
        
        elif method == "river":
            # Extract river-specific summary
            soundings = report.get("diagnostics", {}).get("soundings_usage", {})
            if soundings:
                summary["soundings"] = {
                    "loaded": soundings.get("soundings_loaded"),
                    "segments_calibrated": soundings.get("calibration", {}).get("segments_calibrated")
                }
            
            # Network stats
            for step in report.get("steps", []):
                if step.get("name") == "network_generation":
                    summary["network_segments"] = step.get("statistics", {}).get("segments_count")
                elif step.get("name") == "cross_section_generation":
                    summary["cross_sections"] = step.get("statistics", {}).get("cross_sections_count")
        
        return summary
    
    def set_fusion_result(
        self,
        method: str,
        output_path: Path,
        status: str = "success",
        **kwargs
    ):
        """
        Set fusion/blending result.
        
        Args:
            method: Fusion method used (e.g., 'priority_blend', 'seamless_blend')
            output_path: Path to final fused raster
            status: Fusion status
            **kwargs: Additional fusion parameters
        """
        self.data["fusion"] = {
            "method": method,
            "output_path": str(output_path),
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
    
    def add_error(self, error_msg: str, method: Optional[str] = None):
        """Add error message."""
        self.data["errors"].append({
            "message": error_msg,
            "method": method,
            "timestamp": datetime.now().isoformat()
        })
    
    def run_contract_tests(self) -> Dict[str, Any]:
        """
        Run contract tests to verify pipeline did what it claimed.
        
        Returns:
            Dictionary of test results
        """
        tests = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Test 1: Requested methods ran
        for method in self.data["methods_requested"]:
            method_data = self.data["methods"].get(method, {})
            tests["tests"].append({
                "name": f"{method}_completed",
                "passed": method_data.get("status") == "success",
                "message": f"{method.upper()} pipeline completed successfully",
                "critical": True
            })
        
        # Test 2: Output files exist
        for method, method_data in self.data["methods"].items():
            output_path = method_data.get("output_path")
            if output_path:
                path = Path(output_path)
                tests["tests"].append({
                    "name": f"{method}_output_exists",
                    "passed": path.exists() and path.stat().st_size > 0,
                    "message": f"{method.upper()} output raster exists: {output_path}",
                    "critical": True
                })
        
        # Test 3: SDB-specific tests
        if "sdb" in self.data["methods"]:
            sdb_summary = self.data["methods"]["sdb"].get("report_summary", {})
            
            # Training points test
            training_points = sdb_summary.get("training_points")
            if training_points:
                tests["tests"].append({
                    "name": "sdb_sufficient_training_points",
                    "passed": training_points >= 100,
                    "message": f"SDB has {training_points} training points (minimum: 100)",
                    "value": training_points,
                    "critical": True
                })
            
            # Model quality test
            rmse = sdb_summary.get("rmse_m")
            if rmse:
                tests["tests"].append({
                    "name": "sdb_model_quality",
                    "passed": rmse < 2.0,  # Reasonable threshold
                    "message": f"SDB model RMSE: {rmse:.2f}m (threshold: <2.0m)",
                    "value": rmse,
                    "critical": False
                })
            
            # Adaptive sampling if used
            if "adaptive_sampling" in sdb_summary:
                sampling = sdb_summary["adaptive_sampling"]
                tests["tests"].append({
                    "name": "sdb_adaptive_sampling_effective",
                    "passed": sampling.get("reduction_pct", 0) > 0,
                    "message": f"Adaptive sampling reduced points by {sampling.get('reduction_pct', 0):.1f}%",
                    "value": sampling.get("reduction_pct"),
                    "critical": False
                })
        
        # Test 4: River-specific tests
        if "river" in self.data["methods"]:
            river_summary = self.data["methods"]["river"].get("report_summary", {})
            
            # Network generation test
            segments = river_summary.get("network_segments")
            if segments:
                tests["tests"].append({
                    "name": "river_network_generated",
                    "passed": segments > 0,
                    "message": f"River network has {segments} segments",
                    "value": segments,
                    "critical": True
                })
            
            # Soundings usage test
            soundings_data = river_summary.get("soundings", {})
            if soundings_data.get("loaded", 0) > 0:
                tests["tests"].append({
                    "name": "river_soundings_used",
                    "passed": soundings_data.get("segments_calibrated", 0) > 0,
                    "message": f"Soundings used to calibrate {soundings_data.get('segments_calibrated', 0)} segments",
                    "value": soundings_data.get("segments_calibrated"),
                    "critical": False
                })
        
        # Test 5: Fusion test (if both methods ran)
        if len(self.data["methods"]) > 1 and self.data["fusion"]:
            fusion_output = self.data["fusion"].get("output_path")
            if fusion_output:
                path = Path(fusion_output)
                tests["tests"].append({
                    "name": "fusion_output_exists",
                    "passed": path.exists() and path.stat().st_size > 0,
                    "message": f"Fused output created: {fusion_output}",
                    "critical": True
                })
        
        # Calculate summary
        passed = sum(1 for t in tests["tests"] if t.get("passed", False))
        total = len(tests["tests"])
        critical_tests = [t for t in tests["tests"] if t.get("critical", False)]
        critical_passed = sum(1 for t in critical_tests if t.get("passed", False))
        
        tests["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "critical_tests": len(critical_tests),
            "critical_passed": critical_passed,
            "all_critical_passed": critical_passed == len(critical_tests)
        }
        
        self.data["contract_tests"] = tests
        return tests
    
    def finalize(self, status: str = "success"):
        """
        Finalize the report and write to disk.
        
        Args:
            status: Overall status ('success', 'failed', 'partial')
        """
        self.data["timestamp_end"] = datetime.now().isoformat()
        
        # Calculate duration
        if self.data["timestamp_start"]:
            try:
                start = datetime.fromisoformat(self.data["timestamp_start"])
                end = datetime.fromisoformat(self.data["timestamp_end"])
                self.data["duration_seconds"] = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass
        
        # Run contract tests
        self.run_contract_tests()
        
        # Set final status
        self.data["status"] = status
        
        # Write to file
        report_path = self.output_dir / "unified_bathy_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        log.info(f"[UNIFIED REPORT] Written to {report_path}")
        
        # Print summary
        self._print_summary()
        
        return report_path
    
    def _print_summary(self):
        """Print human-readable summary."""
        log.info("=" * 70)
        log.info("UNIFIED BATHYMETRY PIPELINE SUMMARY")
        log.info("=" * 70)
        
        # Methods
        log.info(f"Methods requested: {', '.join(self.data['methods_requested'])}")
        log.info(f"Priority method: {self.data['priority_method']}")
        
        # Results
        for method, method_data in self.data["methods"].items():
            status = method_data.get("status", "unknown")
            log.info(f"  {method.upper()}: {status}")
            if method_data.get("output_path"):
                log.info(f"    Output: {method_data['output_path']}")
        
        # Fusion
        if self.data["fusion"]:
            fusion_status = self.data["fusion"].get("status", "unknown")
            log.info(f"Fusion: {fusion_status}")
            if self.data["fusion"].get("output_path"):
                log.info(f"  Final output: {self.data['fusion']['output_path']}")
        
        # Contract tests
        if self.data["contract_tests"]:
            test_summary = self.data["contract_tests"]["summary"]
            log.info(f"\nContract Tests: {test_summary['passed']}/{test_summary['total_tests']} passed")
            
            # Show failed critical tests
            failed_critical = [
                t for t in self.data["contract_tests"]["tests"]
                if t.get("critical") and not t.get("passed")
            ]
            if failed_critical:
                log.warning("  FAILED CRITICAL TESTS:")
                for test in failed_critical:
                    log.warning(f"    ✗ {test['name']}: {test.get('message', 'No message')}")
        
        log.info("=" * 70)


def load_and_verify_report(report_path: Path) -> Dict[str, Any]:
    """
    Load a report and verify its contract tests.
    
    Args:
        report_path: Path to unified_bathy_report.json
    
    Returns:
        Report data with verification results
    """
    with open(report_path) as f:
        report = json.load(f)
    
    # Check contract tests
    if "contract_tests" in report:
        test_summary = report["contract_tests"]["summary"]
        
        if not test_summary.get("all_critical_passed", False):
            log.error(f"[VERIFY] Critical tests failed in {report_path}")
            log.error(f"[VERIFY] {test_summary['critical_passed']}/{test_summary['critical_tests']} critical tests passed")
            return None
        
        log.info(f"[VERIFY] All critical tests passed ({test_summary['passed']}/{test_summary['total_tests']} total)")
    
    return report
