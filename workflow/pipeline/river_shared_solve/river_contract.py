"""Active shared-solve contract surface.

Compatibility wrapper around the current implementation in
``pipeline.river_workflow.river_workflow_contract``.
"""

from pipeline.river_workflow.river_workflow_contract import (
    WORKFLOW_STAGE_ORDER,
    artifact_name,
    validate_run_contract,
)

__all__ = ['WORKFLOW_STAGE_ORDER', 'artifact_name', 'validate_run_contract']
