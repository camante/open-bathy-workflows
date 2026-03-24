from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ContractResult:
    name: str
    stage: str
    severity: str
    passed: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContractSuiteResult:
    stage: str
    results: List[ContractResult] = field(default_factory=list)
    receipt_json: Optional[str] = None

    def add(self, result: ContractResult) -> None:
        self.results.append(result)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.results if (not r.passed) and r.severity == "warning")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if (not r.passed) and r.severity == "error")

    @property
    def hard_fail_triggered(self) -> bool:
        return self.n_fail > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "n_pass": self.n_pass,
            "n_warn": self.n_warn,
            "n_fail": self.n_fail,
            "hard_fail_triggered": self.hard_fail_triggered,
            "receipt_json": self.receipt_json,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class EnforcementPolicy:
    mode: str = "warn"  # off|report|warn|strict
    fail_on_warning: bool = False
    enabled_stages: Optional[Iterable[str]] = None

    def stage_enabled(self, stage: str) -> bool:
        if self.mode == "off":
            return False
        if self.enabled_stages is None:
            return True
        return stage in set(self.enabled_stages)


def write_contract_suite_receipt(out_dir: Path, suite: ContractSuiteResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / f"contracts_{suite.stage}.json"
    suite.receipt_json = str(receipt)
    receipt.write_text(json.dumps(suite.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return receipt




def merge_contract_suites(existing: Dict[str, Any] | None, suites: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"stages": {}, "overall": {"n_pass": 0, "n_warn": 0, "n_fail": 0}}
    if isinstance(existing, dict):
        payload = json.loads(json.dumps(existing))
    payload.setdefault("stages", {})
    for suite in suites:
        if not isinstance(suite, dict):
            continue
        stage = str(suite.get("stage") or "").strip()
        if not stage:
            continue
        payload["stages"][stage] = suite
    stages = list((payload.get("stages") or {}).values())
    payload["overall"] = {
        "n_pass": sum(int((s or {}).get("n_pass", 0)) for s in stages),
        "n_warn": sum(int((s or {}).get("n_warn", 0)) for s in stages),
        "n_fail": sum(int((s or {}).get("n_fail", 0)) for s in stages),
    }
    payload["overall"]["hard_fail_triggered"] = int(payload["overall"]["n_fail"]) > 0
    return payload


def write_contract_summary_bundle(contracts_dir: Path, payload: Dict[str, Any], *, basename: str = "contracts_summary") -> tuple[Path, Path]:
    contracts_dir.mkdir(parents=True, exist_ok=True)
    summary_json = contracts_dir / f"{basename}.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_contract_summary_markdown(summary_json)
    return summary_json, summary_json.with_suffix('.md')

def write_contract_summary_markdown(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    lines = ["# Contract Summary", ""]
    overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
    lines.append(
        f"Overall: pass={int(overall.get('n_pass', 0))} warn={int(overall.get('n_warn', 0))} fail={int(overall.get('n_fail', 0))}"
    )
    lines.append("")
    stages = payload.get("stages", {}) if isinstance(payload, dict) else {}
    for stage_name in sorted(stages):
        stage = stages.get(stage_name, {}) or {}
        lines.append(f"## {stage_name}")
        lines.append(f"pass={int(stage.get('n_pass', 0))} warn={int(stage.get('n_warn', 0))} fail={int(stage.get('n_fail', 0))}")
        receipt_json = stage.get("receipt_json")
        if receipt_json:
            lines.append(f"receipt: {receipt_json}")
        lines.append("")
        for result in stage.get("results", []) or []:
            status = "PASS" if result.get("passed") else (result.get("severity") or "FAIL").upper()
            lines.append(f"- {status} {result.get('name')}: {result.get('message')}")
            artifact_paths = result.get("artifact_paths") or {}
            for key, value in sorted(artifact_paths.items()):
                lines.append(f"  - {key}: {value}")
        lines.append("")
    summary_md = summary_path.with_suffix(".md")
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def update_contract_summary(summary_path: Path, suite: ContractSuiteResult) -> None:
    payload: Dict[str, Any] = {"stages": {}, "overall": {"n_pass": 0, "n_warn": 0, "n_fail": 0}}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {"stages": {}, "overall": {"n_pass": 0, "n_warn": 0, "n_fail": 0}}
    payload.setdefault("stages", {})[suite.stage] = suite.to_dict()
    stages = payload["stages"].values()
    payload["overall"] = {
        "n_pass": sum(int(s.get("n_pass", 0)) for s in stages),
        "n_warn": sum(int(s.get("n_warn", 0)) for s in stages),
        "n_fail": sum(int(s.get("n_fail", 0)) for s in stages),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_contract_summary_markdown(summary_path)


def enforce_contract_suite(suite: ContractSuiteResult, policy: EnforcementPolicy) -> None:
    if policy.mode in {"off", "report"}:
        return
    if suite.n_fail > 0:
        failures = "; ".join(f"{r.name}: {r.message}" for r in suite.results if (not r.passed) and r.severity == "error")
        raise RuntimeError(f"Contract enforcement failed for stage '{suite.stage}': {failures}")
    if (policy.mode == "strict" or policy.fail_on_warning) and suite.n_warn > 0:
        warnings = "; ".join(f"{r.name}: {r.message}" for r in suite.results if (not r.passed) and r.severity == "warning")
        raise RuntimeError(f"Contract warnings promoted to failure for stage '{suite.stage}': {warnings}")


def log_contract_suite(logger: logging.Logger, suite: ContractSuiteResult) -> None:
    logger.info("[CONTRACT][%s] pass=%d warn=%d fail=%d", suite.stage.upper(), suite.n_pass, suite.n_warn, suite.n_fail)
    for result in suite.results:
        if result.passed:
            logger.debug("[CONTRACT][%s] PASS %s: %s", suite.stage.upper(), result.name, result.message)
        else:
            level = logging.ERROR if result.severity == "error" else logging.WARNING
            logger.log(level, "[CONTRACT][%s] %s %s: %s", suite.stage.upper(), result.severity.upper(), result.name, result.message)
