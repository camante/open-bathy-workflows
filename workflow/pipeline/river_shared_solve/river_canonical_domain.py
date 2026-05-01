"""Active shared-solve canonical-domain builder.

Compatibility wrapper around the current implementation in
``pipeline.river_workflow.river_workflow_canonical_domain``.
"""

from pipeline.river_workflow.river_workflow_canonical_domain import (
    CanonicalRiverSolveDomainResult,
    build_canonical_river_solve_domain,
)

CanonicalSolveDomainResult = CanonicalRiverSolveDomainResult
build_river_shared_solve_domain = build_canonical_river_solve_domain
build_shared_solve_domain = build_canonical_river_solve_domain

__all__ = [
    'CanonicalRiverSolveDomainResult',
    'build_canonical_river_solve_domain',
    'CanonicalSolveDomainResult',
    'build_river_shared_solve_domain',
    'build_shared_solve_domain',
]
