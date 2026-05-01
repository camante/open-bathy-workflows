"""SDB subprocess command builders."""

from __future__ import annotations

from typing import Any


def build_sdb_command(cfg: Any, *, out_dir: str, authoritative_passthrough_args: list[str] | None = None) -> list[str]:
    cmd = ["python", "sdb_main.py", "--out-dir", str(out_dir)]
    if getattr(cfg, "aoi", None):
        cmd.extend(["--aoi", str(cfg.aoi)])
    if getattr(cfg, "start", None):
        cmd.extend(["--start", str(cfg.start)])
    if getattr(cfg, "end", None):
        cmd.extend(["--end", str(cfg.end)])
    cmd.extend(list(authoritative_passthrough_args or []))
    return cmd


def augment_sdb_command(cfg: Any, cmd: list[str], *, sdb_main_path: str, logger=None) -> list[str]:
    out = list(cmd)
    if len(out) >= 2 and out[1] == "sdb_main.py":
        out[1] = str(sdb_main_path)
    return out


__all__ = ["augment_sdb_command", "build_sdb_command"]
