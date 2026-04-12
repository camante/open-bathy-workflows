from __future__ import annotations

import os
import resource
from typing import Any, Dict


def current_memory_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = float(usage.ru_maxrss)
        if os.uname().sysname == "Darwin":
            rss_mb = rss_kb / (1024.0 * 1024.0)
        else:
            rss_mb = rss_kb / 1024.0
        snapshot["maxrss_mb"] = round(rss_mb, 3)
    except Exception:
        pass
    status = "/proc/self/status"
    try:
        with open(status, "r", encoding="utf-8") as src:
            for line in src:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        snapshot["vmrss_mb"] = round(float(parts[1]) / 1024.0, 3)
                elif line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        snapshot["vmhwm_mb"] = round(float(parts[1]) / 1024.0, 3)
    except Exception:
        pass
    return snapshot


def memory_checkpoint(label: str, **extra: Any) -> Dict[str, Any]:
    snap = current_memory_snapshot()
    snap["label"] = label
    snap.update(extra)
    return snap
