from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CANONICAL_BUILD_ROLE = "canonical_build"
AOI_EXPORT_ONLY_ROLE = "aoi_export_only"
VALID_EXECUTION_ROLES = frozenset({CANONICAL_BUILD_ROLE, AOI_EXPORT_ONLY_ROLE})

CANONICAL_CONSTRUCTION_STAGES = frozenset({
    "solve_domain",
    "grids",
    "authoritative_inputs",
    "centerline_points",
    "centerline_wse_proxy",
    "centerline_authoritative_bed",
    "centerline_observed_offset",
    "centerline_modeled_offset",
    "centerline_bed_backbone",
    "river_corridor_solve",
    "river_primary_surface_solve",
    "river_primary_surface_solve_locked",
    "canonical_parent_dem_build",
})

AOI_EXPORT_ONLY_STAGES = frozenset({
    "read_canonical_manifest",
    "validate_canonical_parent",
    "build_export_grid",
    "subset_parent_to_aoi",
    "river_export_handoff",
    "materialize_combined_dem",
    "verify_export_identity",
    "write_run_summary",
})


class RiverExecutionContractError(RuntimeError):
    """Raised when a river execution role attempts a forbidden stage."""


@dataclass(frozen=True)
class RiverExecutionContract:
    execution_role: str
    allowed_stages: tuple[str, ...]
    forbidden_stages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_role": self.execution_role,
            "allowed_stages": list(self.allowed_stages),
            "forbidden_stages": list(self.forbidden_stages),
        }


def normalize_execution_role(execution_role: str | None) -> str:
    role = str(execution_role or CANONICAL_BUILD_ROLE).strip()
    if role not in VALID_EXECUTION_ROLES:
        raise RiverExecutionContractError(
            f"invalid river execution_role={role!r}; expected one of {sorted(VALID_EXECUTION_ROLES)}"
        )
    return role


def build_execution_contract(execution_role: str | None) -> RiverExecutionContract:
    role = normalize_execution_role(execution_role)
    if role == AOI_EXPORT_ONLY_ROLE:
        return RiverExecutionContract(
            execution_role=role,
            allowed_stages=tuple(sorted(AOI_EXPORT_ONLY_STAGES)),
            forbidden_stages=tuple(sorted(CANONICAL_CONSTRUCTION_STAGES)),
        )
    return RiverExecutionContract(
        execution_role=role,
        allowed_stages=tuple(sorted(CANONICAL_CONSTRUCTION_STAGES | AOI_EXPORT_ONLY_STAGES)),
        forbidden_stages=(),
    )


def ensure_stage_allowed(execution_role: str | None, stage_name: str) -> None:
    role = normalize_execution_role(execution_role)
    stage = str(stage_name).strip()
    if role == AOI_EXPORT_ONLY_ROLE and stage in CANONICAL_CONSTRUCTION_STAGES:
        raise RiverExecutionContractError(
            "AOI export-only river runs may not execute canonical construction stage "
            f"{stage!r}. Build the canonical river solution once, then export/window the AOI "
            "from that existing canonical parent manifest."
        )


def ensure_stages_allowed(execution_role: str | None, stage_names: Iterable[str]) -> None:
    for stage_name in stage_names:
        ensure_stage_allowed(execution_role, stage_name)


__all__ = [
    "AOI_EXPORT_ONLY_ROLE",
    "CANONICAL_BUILD_ROLE",
    "CANONICAL_CONSTRUCTION_STAGES",
    "AOI_EXPORT_ONLY_STAGES",
    "RiverExecutionContract",
    "RiverExecutionContractError",
    "build_execution_contract",
    "ensure_stage_allowed",
    "ensure_stages_allowed",
    "normalize_execution_role",
]
