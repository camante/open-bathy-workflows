from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ActiveWorkflowContext:
    cfg: Any
    args: Any
    report: Dict[str, Any]
    log: Any
    fatal_errors: List[str]
    run_id: str
    callbacks: Dict[str, Callable[..., Any]] = field(default_factory=dict)




@dataclass
class ActiveStageResult:
    stage_name: str
    stage_class: str
    status: str
    primary_input: Optional[Path]
    primary_output: Optional[Path]
    receipt_path: Optional[Path] = None
    detail: Optional[str] = None
    primary_artifact_role: Optional[str] = None

@dataclass
class ActiveRiverFinalizeTargets:
    final_native: Optional[Path]
    final_for_user: Optional[Path]


@dataclass
class ActiveRiverWorkflowResult:
    river_raster: Optional[Path]
    river_deliverable: Optional[Path]
    finalize_targets: ActiveRiverFinalizeTargets
    stage_results: List[ActiveStageResult] = field(default_factory=list)
    stage_chain_summary_path: Optional[Path] = None

    @property
    def linear_deliverable(self) -> Optional[Path]:
        """Deprecated compatibility alias for older callers."""
        return self.river_deliverable


@dataclass
class ActiveWorkflowFinalizeResult:
    workflow_result: ActiveRiverWorkflowResult
    exit_code: int


# Bundle C containment aliases: ActiveRiver* are the public active names.
# ActiveLinear* remain only for older imports and should not be used by active code.
ActiveLinearFinalizeTargets = ActiveRiverFinalizeTargets
ActiveLinearWorkflowResult = ActiveRiverWorkflowResult
