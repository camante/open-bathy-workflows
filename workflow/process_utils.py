"""Small subprocess helpers.

Goal: consistent, non-shell command execution with stdout/stderr capture,
optional cwd/timeout, and convenient tail fields for logs/reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Mapping, Any
import subprocess
import json
from pathlib import Path

def _tail(s: str, n: int = 4000) -> str:
    if not s:
        return ""
    return s[-n:]

@dataclass(frozen=True)
class CmdResult:
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def cmd_str(self) -> str:
        return " ".join(self.cmd)

    @property
    def stdout_tail(self) -> str:
        return _tail(self.stdout)

    @property
    def stderr_tail(self) -> str:
        return _tail(self.stderr)

def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = False,
) -> CmdResult:
    """Run a command safely (shell=False) and capture output."""
    res = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=None if env is None else dict(env),
        shell=False,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    out = CmdResult(cmd=list(cmd), returncode=res.returncode, stdout=res.stdout or "", stderr=res.stderr or "")
    if check and out.returncode != 0:
        raise RuntimeError(f"Command failed ({out.returncode}): {out.cmd_str}\nSTDERR:\n{out.stderr_tail}")
    return out


def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    """Resolve the final SDB depth raster from the explicit SDB manifest only.

    No directory scanning or filename guessing is allowed here. This helper exists so
    downstream modules can share the same no-guess manifest contract without importing
    bathy_main and creating circular dependencies.
    """
    sdb_dir = Path(sdb_dir)
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    rel = data.get("depth_raster")
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = (sdb_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    return p if p.exists() else None
