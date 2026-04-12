from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from river_guidance import build_river_guidance_manifest


@dataclass
class RiverGuidanceContract:
    river_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    final_route_contract: Dict[str, Any]
    required_structural_artifacts: list[str]
    missing_required_structural_artifacts: list[str]
    output_contract_missing_from_outputs: list[str]

    def reporting_payload(self) -> Dict[str, Any]:
        validation_errors = validate_river_guidance_contract(self)
        return {
            "manifest_path": str(self.manifest_path),
            "required_structural_artifacts": list(self.required_structural_artifacts),
            "missing_required_structural_artifacts": list(self.missing_required_structural_artifacts),
            "output_contract_missing_from_outputs": list(self.output_contract_missing_from_outputs),
            "valid": len(validation_errors) == 0,
            "validation_errors": validation_errors,
        }


def _required_structural_artifacts(final_route_contract: Dict[str, Any]) -> list[str]:
    required = final_route_contract.get("required_structural_artifacts")
    if isinstance(required, list):
        return sorted(str(x) for x in required if isinstance(x, str) and x)
    return []


def build_river_guidance_contract(*, out_root: str | Path, river_dir: str | Path, report: Dict[str, Any]) -> RiverGuidanceContract:
    river_dir = Path(river_dir)
    manifest = build_river_guidance_manifest(out_root=out_root, river_dir=river_dir, report=report)
    final_route_contract = manifest.get("final_route_contract", {}) if isinstance(manifest.get("final_route_contract"), dict) else {}
    required = _required_structural_artifacts(final_route_contract)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
    missing_required = [name for name in required if not artifacts.get(name)]
    diagnostics = manifest.get("diagnostics", {}) if isinstance(manifest.get("diagnostics"), dict) else {}
    output_contract_missing = diagnostics.get("output_contract_missing_from_outputs")
    if not isinstance(output_contract_missing, list):
        output_contract_missing = []
    return RiverGuidanceContract(
        river_dir=river_dir,
        manifest_path=river_dir / "river_guidance_manifest.json",
        manifest=manifest,
        final_route_contract=final_route_contract,
        required_structural_artifacts=required,
        missing_required_structural_artifacts=missing_required,
        output_contract_missing_from_outputs=sorted(str(x) for x in output_contract_missing if isinstance(x, str) and x),
    )


def validate_river_guidance_contract(contract: RiverGuidanceContract) -> list[str]:
    errors: list[str] = []
    if contract.missing_required_structural_artifacts:
        missing = ", ".join(contract.missing_required_structural_artifacts)
        errors.append(f"missing_required_river_guidance_artifacts: {missing}")
    return errors


def write_river_guidance_contract(contract: RiverGuidanceContract, *, logger: Optional[logging.Logger] = None) -> Path:
    contract.manifest_path.write_text(json.dumps(contract.manifest, indent=2), encoding="utf-8")
    (logger or logging.getLogger(__name__)).info("Wrote river guidance manifest: %s", contract.manifest_path)
    return contract.manifest_path


def apply_river_guidance_contract_reporting(*, contract: RiverGuidanceContract, report: Dict[str, Any]) -> None:
    river_report = report.setdefault("river", {})
    river_report["guidance_contract"] = contract.reporting_payload()
    river_report.setdefault("outputs", {})["guidance_manifest"] = str(contract.manifest_path)
