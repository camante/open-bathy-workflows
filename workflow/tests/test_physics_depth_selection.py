import json
from pathlib import Path

import numpy as np

from predict import _get_max_depth_from_meta


def _extract_loader_from_sdb_main():
    source = Path(__file__).resolve().parents[1] / "sdb_main.py"
    text = source.read_text(encoding="utf-8")
    start = text.index("            def _load_train_report_auto_depth() -> tuple:")
    end = text.index("            if auto_max_depth:", start)
    block = text[start:end]
    dedented = "\n".join(line[12:] if line.startswith("            ") else line for line in block.splitlines())
    ns = {"json": json, "np": np}
    exec(dedented, ns)
    return ns["_load_train_report_auto_depth"]


def _bind_loader(loader, **kwargs):
    class _Log:
        def debug(self, *args, **kwargs):
            pass

    state = loader.__globals__.copy()
    loader.__globals__.update({"log": _Log(), **kwargs})
    return state


def _restore_loader(loader, state):
    loader.__globals__.clear()
    loader.__globals__.update(state)


def test_auto_depth_prefers_final_physics_from_model_meta(tmp_path):
    loader = _extract_loader_from_sdb_main()
    model_meta = {
        "max_depth_sdb_auto": 1.49,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_sdb_final": 25.4,
        "max_depth_options": {"selected_source": "physics"},
    }
    out_root = tmp_path
    dir_logs = tmp_path / "logs"
    dir_logs.mkdir()
    dir_model = tmp_path / "model"
    dir_model.mkdir()
    state = _bind_loader(
        loader,
        model_meta=model_meta,
        out_root=out_root,
        dir_logs=dir_logs,
        dir_model=dir_model,
    )
    try:
        value, src = loader()
    finally:
        _restore_loader(loader, state)
    assert value == 25.4
    assert src == "model_meta.max_depth_sdb_final"


def test_auto_depth_prefers_report_final_physics_when_model_meta_missing(tmp_path):
    loader = _extract_loader_from_sdb_main()
    model_meta = {"max_depth_options": {"selected_source": "physics"}}
    out_root = tmp_path
    dir_logs = tmp_path / "logs"
    dir_logs.mkdir()
    dir_model = tmp_path / "model"
    dir_model.mkdir()
    report = {
        "train": {
            "max_depth_sdb_auto_m": 1.49,
            "max_depth_sdb_auto_physics": 25.4,
            "max_depth_sdb_final": 25.4,
        }
    }
    (out_root / "train_report.json").write_text(json.dumps(report), encoding="utf-8")
    state = _bind_loader(
        loader,
        model_meta=model_meta,
        out_root=out_root,
        dir_logs=dir_logs,
        dir_model=dir_model,
    )
    try:
        value, src = loader()
    finally:
        _restore_loader(loader, state)
    assert value == 25.4
    assert src.endswith(":train.max_depth_sdb_final")


def test_auto_depth_uses_final_source_family_when_selected_source_missing(tmp_path):
    loader = _extract_loader_from_sdb_main()
    model_meta = {
        "max_depth_sdb_auto": 1.49,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_sdb_final_source": "physics_kd",
    }
    out_root = tmp_path
    dir_logs = tmp_path / "logs"
    dir_logs.mkdir()
    dir_model = tmp_path / "model"
    dir_model.mkdir()
    state = _bind_loader(
        loader,
        model_meta=model_meta,
        out_root=out_root,
        dir_logs=dir_logs,
        dir_model=dir_model,
    )
    try:
        value, src = loader()
    finally:
        _restore_loader(loader, state)
    assert value == 25.4
    assert src == "model_meta.max_depth_sdb_auto_physics"


def test_auto_depth_uses_report_final_source_family_when_model_meta_is_ambiguous(tmp_path):
    loader = _extract_loader_from_sdb_main()
    model_meta = {}
    out_root = tmp_path
    dir_logs = tmp_path / "logs"
    dir_logs.mkdir()
    dir_model = tmp_path / "model"
    dir_model.mkdir()
    report = {
        "train": {
            "max_depth_sdb_final_source": "physics_kd",
            "max_depth_sdb_auto_m": 1.49,
            "max_depth_sdb_auto_physics": 25.4,
        }
    }
    (dir_logs / "train_report.json").write_text(json.dumps(report), encoding="utf-8")
    state = _bind_loader(
        loader,
        model_meta=model_meta,
        out_root=out_root,
        dir_logs=dir_logs,
        dir_model=dir_model,
    )
    try:
        value, src = loader()
    finally:
        _restore_loader(loader, state)
    assert value == 25.4
    assert src.endswith(":train.max_depth_sdb_auto_physics")


def test_predict_depth_prefers_physics_key_when_final_missing():
    meta = {
        "max_depth_sdb_auto": 1.49,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_sdb_final_source": "physics_kd",
    }
    value, src = _get_max_depth_from_meta(meta)
    assert value == 25.4
    assert src == "max_depth_sdb_auto_physics"


def test_predict_depth_prefers_combined_key_when_final_missing():
    meta = {
        "max_depth_sdb_auto": 4.0,
        "max_depth_sdb_auto_physics": 25.4,
        "max_depth_sdb_combined": 12.0,
        "max_depth_sdb_final_source": "combined_computed",
    }
    value, src = _get_max_depth_from_meta(meta)
    assert value == 12.0
    assert src == "max_depth_sdb_combined"
