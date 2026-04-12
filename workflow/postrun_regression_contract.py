from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _normalize_paths(values: Iterable[str | Path] | None) -> tuple[Path, ...]:
    out = []
    for value in values or ():
        if value is None:
            continue
        out.append(Path(str(value)))
    return tuple(out)


@dataclass(frozen=True)
class PostrunRegressionContext:
    current_final_outputs_manifest: Optional[Path]
    current_io_manifest: Optional[Path]
    seam_neighbor_io_manifests: tuple[Path, ...]
    nested_aoi_neighbor_final_outputs: tuple[Path, ...]
    seam_strip_px: int = 3
    overlap_tolerance: float = 1.0e-6
    trusted_tolerance: float = 1.0e-6



def build_postrun_regression_context(
    *,
    current_final_outputs_manifest: str | Path | None,
    current_io_manifest: str | Path | None,
    seam_neighbor_io_manifests: Iterable[str | Path] | None = None,
    nested_aoi_neighbor_final_outputs: Iterable[str | Path] | None = None,
    seam_strip_px: int = 3,
    overlap_tolerance: float = 1.0e-6,
    trusted_tolerance: float = 1.0e-6,
) -> PostrunRegressionContext:
    return PostrunRegressionContext(
        current_final_outputs_manifest=Path(str(current_final_outputs_manifest)) if current_final_outputs_manifest else None,
        current_io_manifest=Path(str(current_io_manifest)) if current_io_manifest else None,
        seam_neighbor_io_manifests=_normalize_paths(seam_neighbor_io_manifests),
        nested_aoi_neighbor_final_outputs=_normalize_paths(nested_aoi_neighbor_final_outputs),
        seam_strip_px=int(seam_strip_px),
        overlap_tolerance=float(overlap_tolerance),
        trusted_tolerance=float(trusted_tolerance),
    )
