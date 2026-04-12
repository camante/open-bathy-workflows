from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from method_activation import MethodActivationTruth


@dataclass
class WorkflowExecutionState:
    out_dir: Optional[str]
    requested_methods: List[str]
    effective_methods: List[str]
    method_activation: Dict[str, MethodActivationTruth] = field(default_factory=dict)
    final_native: Optional[str] = None
    final_for_user: Optional[str] = None
    final_provenance: Optional[str] = None

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "requested_methods": list(self.requested_methods),
            "effective_methods": list(self.effective_methods),
            "method_activation": {k: asdict(v) for k, v in self.method_activation.items()},
            "final_outputs": {
                "final_native": self.final_native,
                "final_for_user": self.final_for_user,
                "final_provenance": self.final_provenance,
            },
        }


def build_workflow_execution_state(*, cfg: Any, report: Dict[str, Any], final_native: Any, final_for_user: Any, final_provenance: Any) -> WorkflowExecutionState:
    activation_payload = report.get("method_activation_truth") or {}
    method_activation: Dict[str, MethodActivationTruth] = {}
    for name, payload in activation_payload.items():
        if isinstance(payload, dict):
            try:
                method_activation[name] = MethodActivationTruth(**payload)
            except TypeError:
                continue
    activation_meta = ((report.get("guidance_domains") or {}).get("activation") or {})
    return WorkflowExecutionState(
        out_dir=str(getattr(cfg, "out_dir", None)) if getattr(cfg, "out_dir", None) is not None else None,
        requested_methods=list(activation_meta.get("requested") or []),
        effective_methods=list(activation_meta.get("effective") or []),
        method_activation=method_activation,
        final_native=str(final_native) if final_native is not None else None,
        final_for_user=str(final_for_user) if final_for_user is not None else None,
        final_provenance=str(final_provenance) if final_provenance is not None else None,
    )
