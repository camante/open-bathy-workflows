from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class LinearAuthoritativeSourceContract:
    solve_authoritative_source_path: Path
    export_authoritative_source_path: Path
    resolution_source_path: Path
    source_kind: str
    solve_source_role: str = "solve_routing"
    export_source_role: str = "export_only"
    routing_policy: str = "canonical_only_after_prepare"

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {k: str(v) if isinstance(v, Path) else str(v) for k, v in payload.items()}


@dataclass(frozen=True)
class LinearBaselineSourceContract:
    export_baseline_source_path: Path
    source_kind: str
    source_role: str = "export_background_only"

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {k: str(v) if isinstance(v, Path) else str(v) for k, v in payload.items()}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_linear_authoritative_source_contract(
    path: Path,
    *,
    solve_authoritative_source_path: Path,
    export_authoritative_source_path: Path,
    resolution_source_path: Path,
    source_kind: str,
    solve_source_role: str = "solve_routing",
    export_source_role: str = "export_only",
    routing_policy: str = "canonical_only_after_prepare",
) -> Path:
    contract = LinearAuthoritativeSourceContract(
        solve_authoritative_source_path=Path(solve_authoritative_source_path),
        export_authoritative_source_path=Path(export_authoritative_source_path),
        resolution_source_path=Path(resolution_source_path),
        source_kind=str(source_kind),
        solve_source_role=str(solve_source_role),
        export_source_role=str(export_source_role),
        routing_policy=str(routing_policy),
    )
    return _write_json(path, contract.to_dict())


def write_linear_baseline_source_contract(
    path: Path,
    *,
    export_baseline_source_path: Path,
    source_kind: str,
    source_role: str = "export_background_only",
) -> Path:
    contract = LinearBaselineSourceContract(
        export_baseline_source_path=Path(export_baseline_source_path),
        source_kind=str(source_kind),
        source_role=str(source_role),
    )
    return _write_json(path, contract.to_dict())


def _load_json(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"linear_source_contract_invalid_json:{path}")
    return data


def load_linear_authoritative_source_contract(path: Path) -> LinearAuthoritativeSourceContract:
    payload = _load_json(path)
    return LinearAuthoritativeSourceContract(
        solve_authoritative_source_path=Path(payload["solve_authoritative_source_path"]),
        export_authoritative_source_path=Path(payload["export_authoritative_source_path"]),
        resolution_source_path=Path(payload["resolution_source_path"]),
        source_kind=str(payload["source_kind"]),
        solve_source_role=str(payload.get("solve_source_role", "solve_routing")),
        export_source_role=str(payload.get("export_source_role", "export_only")),
        routing_policy=str(payload.get("routing_policy", "canonical_only_after_prepare")),
    )


def load_linear_baseline_source_contract(path: Path) -> LinearBaselineSourceContract:
    payload = _load_json(path)
    return LinearBaselineSourceContract(
        export_baseline_source_path=Path(payload["export_baseline_source_path"]),
        source_kind=str(payload["source_kind"]),
        source_role=str(payload.get("source_role", "export_background_only")),
    )


__all__ = [
    "LinearAuthoritativeSourceContract",
    "LinearBaselineSourceContract",
    "load_linear_authoritative_source_contract",
    "load_linear_baseline_source_contract",
    "write_linear_authoritative_source_contract",
    "write_linear_baseline_source_contract",
]
