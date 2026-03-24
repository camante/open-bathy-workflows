from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from contract_enforcement import (
    ContractSuiteResult,
    EnforcementPolicy,
    enforce_contract_suite,
    log_contract_suite,
    merge_contract_suites,
    update_contract_summary,
    write_contract_suite_receipt,
    write_contract_summary_bundle,
)
from contracts_guidance_domains import run_guidance_domain_contracts
from contracts_river_masks import run_river_mask_contracts
from contracts_final_dem_runtime import run_final_dem_runtime_contracts
from contracts_stability_runtime import run_stability_runtime_contracts


def _policy_from_obj(obj: Any) -> EnforcementPolicy:
    mode = str(getattr(obj, "contract_mode", "warn") or "warn").lower()
    fail_on_warning = bool(getattr(obj, "fail_on_contract_warning", False))
    return EnforcementPolicy(mode=mode, fail_on_warning=fail_on_warning)


def _contracts_root(base_dir: Path) -> Path:
    return base_dir / "contracts"


def run_guidance_domain_stage_contracts(*, cfg: Any, outputs: Dict[str, str], report: Optional[Dict[str, Any]], base_dir: Path, logger: Any) -> ContractSuiteResult:
    suite = run_guidance_domain_contracts(outputs, debug_dir=_contracts_root(base_dir) / "debug")
    contracts_dir = _contracts_root(base_dir)
    receipt = write_contract_suite_receipt(contracts_dir, suite)
    update_contract_summary(contracts_dir / "contracts_summary.json", suite)
    log_contract_suite(logger, suite)
    if report is not None:
        report.setdefault("contracts", {})[suite.stage] = suite.to_dict()
        report.setdefault("outputs", {})[f"contracts_{suite.stage}_json"] = str(receipt)
    enforce_contract_suite(suite, _policy_from_obj(cfg))
    return suite


def run_river_mask_stage_contracts(*, cfg: Any, channel_mask_tif: Path, open_water_mask_tif: Path, mainstem_mask_tif: Path, estuary_clip_mask_tif: Path, report: Optional[Dict[str, Any]], base_dir: Path, logger: Any) -> ContractSuiteResult:
    suite = run_river_mask_contracts(
        channel_mask_tif=Path(channel_mask_tif),
        open_water_mask_tif=Path(open_water_mask_tif),
        mainstem_mask_tif=Path(mainstem_mask_tif),
        estuary_clip_mask_tif=Path(estuary_clip_mask_tif),
        debug_dir=_contracts_root(base_dir) / "debug",
    )
    contracts_dir = _contracts_root(base_dir)
    receipt = write_contract_suite_receipt(contracts_dir, suite)
    update_contract_summary(contracts_dir / "contracts_summary.json", suite)
    log_contract_suite(logger, suite)
    if report is not None:
        report.setdefault("contracts", {})[suite.stage] = suite.to_dict()
        report.setdefault("outputs", {})[f"contracts_{suite.stage}_json"] = str(receipt)
    enforce_contract_suite(suite, _policy_from_obj(cfg))
    return suite


def run_final_dem_stage_contracts(*, cfg: Any, report: Dict[str, Any], final_depth: Path | None, base_dir: Path, logger: Any) -> ContractSuiteResult:
    suite = run_final_dem_runtime_contracts(report=report, final_depth=final_depth, debug_dir=_contracts_root(base_dir) / "debug")
    contracts_dir = _contracts_root(base_dir)
    receipt = write_contract_suite_receipt(contracts_dir, suite)
    update_contract_summary(contracts_dir / "contracts_summary.json", suite)
    log_contract_suite(logger, suite)
    report.setdefault("contracts", {})[suite.stage] = suite.to_dict()
    report.setdefault("outputs", {})[f"contracts_{suite.stage}_json"] = str(receipt)
    enforce_contract_suite(suite, _policy_from_obj(cfg))
    return suite




def run_stability_stage_contracts(*, cfg: Any, report: Dict[str, Any], base_dir: Path, logger: Any) -> ContractSuiteResult:
    suite = run_stability_runtime_contracts(cfg=cfg, report=report)
    contracts_dir = _contracts_root(base_dir)
    receipt = write_contract_suite_receipt(contracts_dir, suite)
    update_contract_summary(contracts_dir / "contracts_summary.json", suite)
    log_contract_suite(logger, suite)
    report.setdefault("contracts", {})[suite.stage] = suite.to_dict()
    report.setdefault("outputs", {})[f"contracts_{suite.stage}_json"] = str(receipt)
    enforce_contract_suite(suite, _policy_from_obj(cfg))
    return suite

def _iter_contract_receipts(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    found = []
    for path in root.rglob('contracts_*.json'):
        if path.name in {'contracts_summary.json', 'contracts_run_summary.json'}:
            continue
        found.append(path)
    return sorted(found)


def aggregate_run_contracts(*, out_dir: Path, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    suites = []
    seen = set()
    if isinstance(report, dict):
        for stage_name, suite in sorted((report.get('contracts') or {}).items()):
            if isinstance(suite, dict) and str(suite.get('stage') or stage_name):
                suites.append(suite)
                seen.add(str(suite.get('stage') or stage_name))
    for receipt in _iter_contract_receipts(Path(out_dir)):
        try:
            suite = json.loads(receipt.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            continue
        stage = str((suite or {}).get('stage') or '').strip()
        if not stage or stage in seen:
            continue
        suites.append(suite)
        seen.add(stage)
    payload = merge_contract_suites(None, suites)
    summary_json, summary_md = write_contract_summary_bundle(Path(out_dir) / 'contracts', payload, basename='contracts_run_summary')
    payload['summary_json'] = str(summary_json)
    payload['summary_md'] = str(summary_md)
    if isinstance(report, dict):
        report['contracts_overall'] = payload
        report.setdefault('outputs', {})['contracts_run_summary_json'] = str(summary_json)
        report.setdefault('outputs', {})['contracts_run_summary_md'] = str(summary_md)
    return payload
