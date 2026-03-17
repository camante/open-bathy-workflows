from pathlib import Path

from run_summary import build_scientific_summary, write_run_summary_files


def test_build_scientific_summary_accepts_flight_artifacts():
    stats = {'aoi': '0/1/2/3', 'fusion': {'mode': 'weighted_overlap'}}
    fr_summary = {'artifacts': [{'path': '/tmp/final_depth.tif', 'role': 'final_depth', 'kind': 'raster'}]}
    text = build_scientific_summary(stats, fr_summary)
    assert 'Artifacts recorded' in text
    assert '/tmp/final_depth.tif' in text


def test_write_run_summary_files_passes_flight_summary(tmp_path: Path):
    out_dir = tmp_path / 'out'
    stats = {'aoi': '0/1/2/3'}
    paths = write_run_summary_files(out_dir, 'abc123', stats=stats)
    sci = paths['scientific_md'].read_text(encoding='utf-8')
    assert 'Run scientific summary' in sci
