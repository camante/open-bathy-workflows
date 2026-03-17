from pathlib import Path
import json

from flight_recorder import FlightRecorder, emit_event


def test_emit_event_expands_payload_fields(tmp_path: Path):
    path = tmp_path / 'flight.jsonl'
    rec = FlightRecorder.start_global(path, 'run_test_emit')
    try:
        emit_event('artifact_written', artifact_path='x.tif', artifact_kind='raster')
    finally:
        FlightRecorder.stop_global()
    lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    matches = [obj for obj in lines if obj.get('event') == 'artifact_written']
    assert matches, 'expected artifact_written event'
    event = matches[-1]
    assert event.get('artifact_path') == 'x.tif'
    assert event.get('artifact_kind') == 'raster'
    assert 'payload' not in event
