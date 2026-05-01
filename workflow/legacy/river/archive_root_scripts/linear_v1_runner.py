"""Legacy compatibility wrapper for archived linear_v1 imports.

The active built-in river workflow lives in ``river_runner.py``. Deprecated
linear_v1 names are defined only in this archived wrapper so the active runner
module no longer exports old method names.
"""

from river_runner import (
    register_river_runner_result,
    run_river_workflow_direct,
)

register_linear_v1_runner_result = register_river_runner_result
run_linear_v1_direct = run_river_workflow_direct

__all__ = [
    "register_linear_v1_runner_result",
    "register_river_runner_result",
    "run_linear_v1_direct",
    "run_river_workflow_direct",
]
