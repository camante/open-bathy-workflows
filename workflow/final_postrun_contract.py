from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence



def _optional_path(value: Optional[Path | str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    return path


@dataclass(frozen=True)
class FinalArtifacts:
    final_native: Optional[Path]
    final_for_user: Optional[Path]
    final_provenance: Optional[Path]


@dataclass
class FinalPostRunContext:
    cfg: Any
    args: Any
    report: Dict[str, Any]
    run_id: str
    artifacts: FinalArtifacts
    fatal_errors: tuple[str, ...] = ()
    report_path: Optional[Path] = None



def build_final_artifacts(
    *,
    final_native: Optional[Path | str],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> FinalArtifacts:
    return FinalArtifacts(
        final_native=_optional_path(final_native),
        final_for_user=_optional_path(final_for_user),
        final_provenance=_optional_path(final_provenance),
    )



def build_final_postrun_context(
    *,
    cfg: Any,
    args: Any,
    report: Dict[str, Any],
    run_id: str,
    final_native: Optional[Path | str],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
    fatal_errors: Sequence[str] | None = None,
    report_path: Optional[Path | str] = None,
) -> FinalPostRunContext:
    return FinalPostRunContext(
        cfg=cfg,
        args=args,
        report=report,
        run_id=str(run_id),
        artifacts=build_final_artifacts(
            final_native=final_native,
            final_for_user=final_for_user,
            final_provenance=final_provenance,
        ),
        fatal_errors=tuple(str(v) for v in (fatal_errors or ())),
        report_path=_optional_path(report_path),
    )
