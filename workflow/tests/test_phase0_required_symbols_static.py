from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_SYMBOLS = {
    "support_classes.py": ["build_regime_masks", "regime_array_from_masks"],
    "river_masking.py": [
        "ensure_waffles_coastline_mask",
        "clip_channel_mask_for_estuary",
        "apply_estuary_first_channel_domain",
        "choose_waffles_mask_for_river",
    ],
    "sign_semantics.py": [
        "semantics_from_depth_positive_down_flag",
        "semantics_from_soundings_mode",
        "summarize_numeric_sign",
    ],
    "tools/debug/workflow_actual_trace.py": ["write_workflow_actual_trace"],
    "tools/debug/workflow_connected_diagnostics.py": ["write_connected_diagnostics"],
    "tools/debug/workflow_run_diagnosis.py": ["write_run_diagnosis"],
    "tools/debug/workflow_reports_hub.py": ["write_reports_hub"],
    "validation/validation_invariance_framework.py": ["run_validation_invariance_framework"],
    "validation/scientific_validation_stage.py": ["write_scientific_validation_summary"],
}


def _top_level_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_required_active_import_symbols_exist() -> None:
    missing: list[str] = []
    for rel, required in REQUIRED_SYMBOLS.items():
        path = ROOT / rel
        assert path.exists(), f"required module missing: {rel}"
        symbols = _top_level_symbols(path)
        for name in required:
            if name not in symbols:
                missing.append(f"{rel}:{name}")
    assert not missing, "missing required symbols: " + ", ".join(missing)
