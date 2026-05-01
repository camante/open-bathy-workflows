"""Active shared-solve pipeline entrypoint.

Compatibility wrapper around the current implementation in
``pipeline.river_workflow.river_workflow_pipeline``.
"""

from pipeline.river_workflow.river_workflow_pipeline import (
    RiverWorkflowPipelineResult,
    run_river_workflow_pipeline,
)

RiverPipelineResult = RiverWorkflowPipelineResult
run_river_pipeline = run_river_workflow_pipeline
run_shared_solve_pipeline = run_river_workflow_pipeline

__all__ = [
    'RiverWorkflowPipelineResult',
    'run_river_workflow_pipeline',
    'RiverPipelineResult',
    'run_river_pipeline',
    'run_shared_solve_pipeline',
]
