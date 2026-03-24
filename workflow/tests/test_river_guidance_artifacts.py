import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace

from river_guidance import (
    build_river_guidance_manifest,
    guidance_artifact_paths,
    rasterize_river_guide_points_to_template,
    write_river_guidance_manifest,
)


def test_build_and_write_river_guidance_manifest(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in (
        "guidance_weight",
        "trusted_interior",
        "soft_guidance_domain",
        "admissibility",
        "corridor_mask",
        "authoritative_support",
        "authoritative_support_depth",
        "bank_edge_mask",
        "bank_distance",
        "bank_influence",
        "bank_continuity_weight",
        "bank_graph_confidence",
        "bank_confluence_damping",
        "bank_estuary_side_decay",
        "depth_terrain",
        "bottom_elevation",
        "scaffold_domains",
        "scaffold_manifest",
        "guide_points",
    ):
        p = paths[key]
        if p.suffix in {".json", ".gpkg"}:
            p.write_text("{}", encoding="utf-8")
        else:
            p.write_bytes(b"x")

    report = {
        "river": {
            "outputs": {
                "guidance_weight": str(paths["guidance_weight"]),
                "trusted_interior": str(paths["trusted_interior"]),
                "soft_guidance_domain": str(paths["soft_guidance_domain"]),
                "admissibility": str(paths["admissibility"]),
                "guide_points": str(paths["guide_points"]),
                "depth_terrain": str(paths["depth_terrain"]),
                "bottom_elevation": str(paths["bottom_elevation"]),
            },
            "guidance": {
                "trusted_interior_definition": "trusted export interior",
                "soft_guidance_definition": "soft guidance domain",
                "admissibility_definition": "anchor-excluded admissible guidance",
            },
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["guidance_only"] is True
    assert manifest["artifact_roles"]["depth_terrain"] == "diagnostic_only"
    assert "depth_terrain" in manifest["final_route_contract"]["diagnostic_only_artifacts"]
    assert "guide_points" in manifest["final_route_contract"]["allowed_structural_artifacts"]
    assert manifest["artifact_roles"]["guide_points"] == "structured_scaffold_points"
    assert manifest["artifact_roles"]["bank_influence"] == "corridor_bank_influence"
    assert manifest["notes"]["trusted_interior"] == "trusted export interior"
    assert manifest["artifact_roles"]["bank_continuity_weight"] == "xs_bank_longitudinal_continuity"
    assert manifest["artifact_roles"]["bank_graph_confidence"] == "graph_informed_bank_confidence"

    manifest_path = write_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["artifacts"]["guide_points"].endswith("river_guide_points.gpkg")



def test_rasterize_river_guide_points_to_template(tmp_path: Path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    gpkg = paths["guide_points"]
    gdf = gpd.GeoDataFrame({"centerline_z_m": [5.0, 7.0]}, geometry=[Point(0.5, 3.5), Point(1.5, 2.5)], crs="EPSG:4326")
    gdf.to_file(gpkg, driver="GPKG")

    tmpl = tmp_path / "template.tif"
    with rasterio.open(tmpl, 'w', driver='GTiff', height=4, width=4, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(0, 4, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.full((4,4), -9999.0, dtype='float32'), 1)

    arr = rasterize_river_guide_points_to_template(gpkg, tmpl)
    assert arr is not None
    assert float(arr[0, 0]) == 5.0
    assert float(arr[1, 1]) == 7.0
