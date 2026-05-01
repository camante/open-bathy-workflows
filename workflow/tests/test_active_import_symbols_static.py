from pathlib import Path
import importlib.util
import sys


def _load_checker():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "check_fresh_package_integrity.py"
    spec = importlib.util.spec_from_file_location("check_fresh_package_integrity", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_active_imported_local_symbols_exist():
    checker = _load_checker()
    result = checker.run_checks(write_report=False)
    assert result["missing_local_modules"] == []
    assert result["missing_local_symbols"] == []
