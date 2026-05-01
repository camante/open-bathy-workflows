"""Active shared-solve river workflow namespace.

This package is the user-facing / active-package surface for the built-in river
workflow. During the transition, it re-exports the mature implementation that
still lives under ``pipeline.river_workflow``.
"""

from .river_canonical_domain import (
    CanonicalRiverSolveDomainResult,
    CanonicalSolveDomainResult,
    build_canonical_river_solve_domain,
    build_river_shared_solve_domain,
    build_shared_solve_domain,
)
from .river_context import (
    CanonicalRiverSourceContext,
    CanonicalSourceContext,
    RiverContext,
    RiverWorkflowContext,
    RiverWorkflowSourceBundle,
    RiverSourceBundle,
    SharedSolveContext,
    SharedSolveSourceBundle,
)
from .river_contract import WORKFLOW_STAGE_ORDER, artifact_name, validate_run_contract
from .river_paths import (
    RiverWorkflowPaths,
    RiverPaths,
    build_river_workflow_paths,
    build_river_paths,
    build_shared_solve_paths,
)
from .river_pipeline import (
    RiverWorkflowPipelineResult,
    RiverPipelineResult,
    run_river_workflow_pipeline,
    run_river_pipeline,
    run_shared_solve_pipeline,
)

__all__ = [
    'CanonicalRiverSolveDomainResult',
    'CanonicalSolveDomainResult',
    'build_canonical_river_solve_domain',
    'build_river_shared_solve_domain',
    'build_shared_solve_domain',
    'CanonicalRiverSourceContext',
    'CanonicalSourceContext',
    'RiverContext',
    'RiverWorkflowContext',
    'RiverWorkflowSourceBundle',
    'RiverSourceBundle',
    'SharedSolveContext',
    'SharedSolveSourceBundle',
    'WORKFLOW_STAGE_ORDER',
    'artifact_name',
    'validate_run_contract',
    'RiverWorkflowPaths',
    'RiverPaths',
    'build_river_workflow_paths',
    'build_river_paths',
    'build_shared_solve_paths',
    'RiverWorkflowPipelineResult',
    'RiverPipelineResult',
    'run_river_workflow_pipeline',
    'run_river_pipeline',
    'run_shared_solve_pipeline',
]
