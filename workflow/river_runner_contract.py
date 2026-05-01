"""Active river runner contract types.

This module is intentionally small: it defines the public data shapes used by
active imports while the implementation delegates to the shared-solve /
river-workflow pipeline result.  It does not create an alternate river workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RiverRunnerPrimaryArtifacts:
    """Primary artifacts returned by the active river runner."""

    canonical_parent_dem: Path | None = None
    aoi_export_dem: Path | None = None
    final_user_dem: Path | None = None
    run_contract: Path | None = None
    stage_chain_summary: Path | None = None


@dataclass(frozen=True)
class RiverRunnerRegistrationPayload:
    """Report updates that should be registered after the runner completes."""

    outputs: Mapping[str, Any] = field(default_factory=dict)
    river_workflow: Mapping[str, Any] = field(default_factory=dict)
    active_river: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiverRunnerResult:
    """Compatibility wrapper around the underlying river pipeline result."""

    primary_artifacts: RiverRunnerPrimaryArtifacts
    registration_payload: RiverRunnerRegistrationPayload
    pipeline_result: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline_result, name)


__all__ = [
    "RiverRunnerPrimaryArtifacts",
    "RiverRunnerRegistrationPayload",
    "RiverRunnerResult",
]
