from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json_receipt(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return path
