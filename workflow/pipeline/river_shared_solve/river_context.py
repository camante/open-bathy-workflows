"""Active shared-solve context surface.

Compatibility wrapper around the current implementation in
``pipeline.river_workflow.river_workflow_context``.
"""

from pipeline.river_workflow.river_workflow_context import (
    CanonicalRiverSourceContext,
    RiverWorkflowContext,
    RiverWorkflowSourceBundle,
)

CanonicalSourceContext = CanonicalRiverSourceContext
RiverContext = RiverWorkflowContext
RiverSourceBundle = RiverWorkflowSourceBundle
SharedSolveContext = RiverWorkflowContext
SharedSolveSourceBundle = RiverWorkflowSourceBundle

__all__ = [
    'CanonicalRiverSourceContext',
    'RiverWorkflowContext',
    'RiverWorkflowSourceBundle',
    'CanonicalSourceContext',
    'RiverContext',
    'RiverSourceBundle',
    'SharedSolveContext',
    'SharedSolveSourceBundle',
]
