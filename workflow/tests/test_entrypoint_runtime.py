import os

from entrypoint_runtime import configure_cli_runtime, execute_cli


def test_configure_cli_runtime_sets_bytecode_guard(monkeypatch):
    monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
    configure_cli_runtime()
    assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'


def test_execute_cli_returns_rc():
    assert execute_cli(lambda: 7) == 7
