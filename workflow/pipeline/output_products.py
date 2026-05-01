from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class OutputProductRole:
    """Named output-product role used by the active AOI-independent route."""

    role: str
    description: str
    user_facing: bool
    may_write_combined_dem_enhanced: bool = False


CANONICAL_PARENT_DEM: Final[OutputProductRole] = OutputProductRole(
    role="canonical_parent_dem",
    description="Scientific DEM on the canonical solve grid; never cropped to a user AOI.",
    user_facing=False,
)
AOI_EXPORT_DEM: Final[OutputProductRole] = OutputProductRole(
    role="aoi_export_dem",
    description="Exact parent-grid subset exported for the user AOI; no local solve, blend, or re-lock.",
    user_facing=False,
)
FINAL_USER_DEM: Final[OutputProductRole] = OutputProductRole(
    role="final_user_dem",
    description="Stable user-facing materialized AOI export at combined/DEM_enhanced.tif.",
    user_facing=True,
    may_write_combined_dem_enhanced=True,
)
DISPLAY_DEM: Final[OutputProductRole] = OutputProductRole(
    role="display_dem",
    description="Optional display/comparison raster; may be cropped or reprojected but is not the scientific final DEM.",
    user_facing=True,
)
DIAGNOSTIC_DEM: Final[OutputProductRole] = OutputProductRole(
    role="diagnostic_dem",
    description="Diagnostic raster for troubleshooting only.",
    user_facing=False,
)
INTERNAL_CONDITIONING_DEM: Final[OutputProductRole] = OutputProductRole(
    role="internal_conditioning_dem",
    description="Internal terrain-generation/conditioning raster that must not be exposed as the final user DEM.",
    user_facing=False,
)

PRODUCT_ROLES: Final[dict[str, OutputProductRole]] = {
    item.role: item
    for item in (
        CANONICAL_PARENT_DEM,
        AOI_EXPORT_DEM,
        FINAL_USER_DEM,
        DISPLAY_DEM,
        DIAGNOSTIC_DEM,
        INTERNAL_CONDITIONING_DEM,
    )
}


def role_for_final_path(path: Path) -> str | None:
    """Classify the stable final path without authorizing other writers."""

    p = Path(path)
    if p.name == "DEM_enhanced.tif" and p.parent.name == "combined":
        return FINAL_USER_DEM.role
    return None


def assert_only_materializer_writes_final_user_dem(*, role: str, writer_role: str) -> None:
    """Guard the active final DEM against accidental writer proliferation."""

    if role != FINAL_USER_DEM.role:
        return
    if writer_role != "final_dem_materializer":
        raise ValueError(f"final_user_dem_invalid_writer:{writer_role}")


__all__ = [
    "AOI_EXPORT_DEM",
    "CANONICAL_PARENT_DEM",
    "DIAGNOSTIC_DEM",
    "DISPLAY_DEM",
    "FINAL_USER_DEM",
    "INTERNAL_CONDITIONING_DEM",
    "OutputProductRole",
    "PRODUCT_ROLES",
    "assert_only_materializer_writes_final_user_dem",
    "role_for_final_path",
]
