"""Helpers for passing authoritative support inputs into child entry points.

Extracted from bathy_main.py to keep orchestration code smaller and easier to debug.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_authoritative_passthrough_args(cfg: Any, *, for_river: bool = False, logger: Optional[logging.Logger] = None) -> List[str]:
    """Build deterministic child-process args for authoritative support inputs."""
    log = logger or logging.getLogger(__name__)
    args: List[str] = []
    if for_river:
        bed = getattr(cfg, "river_authoritative_bed", None) or getattr(cfg, "authoritative_base", None)
        if bed:
            try:
                bed_path = Path(bed)
            except (TypeError, ValueError, OSError):
                log.debug("[AUTHORITATIVE] Invalid river authoritative-bed value for passthrough", exc_info=True)
            else:
                if bed_path.exists():
                    args.append(f"--authoritative-bed-raster={bed_path}")
                    args.append("--no-authoritative-bed-auto")
        return args

    auth = getattr(cfg, "authoritative_base", None)
    if auth:
        try:
            auth_path = Path(auth)
        except (TypeError, ValueError, OSError):
            log.debug("[AUTHORITATIVE] Invalid authoritative-base value for passthrough", exc_info=True)
        else:
            if auth_path.exists():
                args.append(f"--authoritative-base={auth_path}")
                args.append("--no-authoritative-base-auto")
    return args


def record_authoritative_child_passthrough(report: Dict[str, Any], stage: str, cmd: List[str]) -> None:
    """Record explicit authoritative-base passthrough status for child entry points."""
    info: Dict[str, Any] = {
        "command_stage": str(stage),
        "explicit_authoritative_path": None,
        "explicit_river_authoritative_bed": None,
        "auto_materialization_disabled": False,
    }
    for token in cmd:
        text = token if isinstance(token, str) else str(token)
        if text.startswith("--authoritative-base="):
            info["explicit_authoritative_path"] = text.split("=", 1)[1]
        elif text.startswith("--authoritative-bed-raster="):
            info["explicit_river_authoritative_bed"] = text.split("=", 1)[1]
        elif text in ("--no-authoritative-base-auto", "--no-authoritative-bed-auto"):
            info["auto_materialization_disabled"] = True
    report.setdefault("authoritative_base_auto", {}).setdefault("downstream_child_passthrough", {})[str(stage)] = info
