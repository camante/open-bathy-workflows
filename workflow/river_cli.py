"""Command builder for the legacy river skeleton subprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_river_skeleton_command(cfg: Any, **kwargs: Any) -> list[str]:
    script = Path(__file__).parent / "river_skeleton.py"
    cmd = ["python", str(script)]
    for key, value in kwargs.items():
        if value in (None, False):
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            cmd.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([flag, str(item)])
        else:
            cmd.extend([flag, str(value)])
    return cmd


__all__ = ["build_river_skeleton_command"]
