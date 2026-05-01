from __future__ import annotations


class RiverWorkflowStageError(RuntimeError):
    """Base error for active river workflow stage failures."""


class RiverWorkflowInputError(RiverWorkflowStageError):
    """Raised when a required input artifact or value is missing/invalid."""


class RiverWorkflowDiagnosticError(RiverWorkflowStageError):
    """Raised only inside optional diagnostic helpers; construction must not catch this silently."""
