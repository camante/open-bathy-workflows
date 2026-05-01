"""Active shared-solve path helpers.

Compatibility wrapper around the current implementation in
``pipeline.river_workflow.river_workflow_paths``.
"""

from pipeline.river_workflow.river_workflow_paths import (
    RiverWorkflowPaths,
    build_river_workflow_paths,
)

RiverPaths = RiverWorkflowPaths
build_river_paths = build_river_workflow_paths
build_shared_solve_paths = build_river_workflow_paths

__all__ = [
    'RiverWorkflowPaths',
    'build_river_workflow_paths',
    'RiverPaths',
    'build_river_paths',
    'build_shared_solve_paths',
]
