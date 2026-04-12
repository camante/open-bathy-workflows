from pathlib import Path

import pandas as pd
import rasterio
import numpy as np
from rasterio.transform import from_origin

from authoritative_guidance import prepare_authoritative_river_soundings_points


def test_prepare_authoritative_river_soundings_points_writes_role_columns(tmp_path: Path):
    src = tmp_path / "auth.tif"
    arr = np.arange(25, dtype="float32").reshape(5, 5)
    channel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype="uint8")
    guidance = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype="uint8")
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:32619",
        transform=from_origin(0, 5, 1, 1),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)
        ds.update_tags(VALUE_TYPE="elevation")
    channel_mask = tmp_path / "channel.tif"
    guidance_mask = tmp_path / "guidance.tif"
    for path, data in ((channel_mask, channel), (guidance_mask, guidance)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=5,
            width=5,
            count=1,
            dtype="uint8",
            crs="EPSG:32619",
            transform=from_origin(0, 5, 1, 1),
            nodata=0,
        ) as ds:
            ds.write(data, 1)
    out_csv = tmp_path / "river_pts.csv"
    info = prepare_authoritative_river_soundings_points(
        src,
        out_csv,
        out_crs="EPSG:32619",
        river_channel_mask=channel_mask,
        river_guidance_domain_mask=guidance_mask,
        bank_margin_m=0.5,
    )
    df = pd.read_csv(out_csv)
    assert info["target_role"] == "river_soundings"
    assert info["negative_only"] is False
    assert "authoritative_role" in df.columns
    assert "distance_to_bank_m" in df.columns
    assert set(df["authoritative_role"]) >= {"authoritative_bank_margin", "authoritative_bed_core", "authoritative_overbank_or_ambiguous"}
    assert (df.loc[df["inside_channel_mask"] == 0, "authoritative_role"] == "authoritative_overbank_or_ambiguous").all()
    assert out_csv.with_suffix(".role_contract.json").exists()
    assert Path(info["role_code_raster"]).exists()
    assert Path(info["role_confidence_raster"]).exists()
    assert Path(info["distance_to_bank_raster"]).exists()
