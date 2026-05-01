from pathlib import Path


def test_tnm_warp_forces_workflow_nodata_contract():
    src = Path(__file__).resolve().parents[1] / "bathy_main.py"
    text = src.read_text()
    assert "sanitized_v5_workflow_nodata_9999" in text
    assert '"-dstnodata"' in text
    assert '"-9999"' in text
    assert '"-srcnodata"' in text
    assert "cached_raster_semantics_valid(" in text
    assert "Ignoring invalid cached TNM DEM" in text
