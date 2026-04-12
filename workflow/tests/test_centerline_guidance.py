import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from rasterio.transform import from_origin

from river_structured_scaffold import (
    rasterize_point_seed_surface,
    build_centerline_core_influence,
)
from bathy_main import _apply_authoritative_bed_to_centerline_points


def _centerline_points():
    pts = [Point(2.5, 4.5), Point(2.5, 3.5), Point(2.5, 2.5), Point(2.5, 1.5), Point(2.5, 0.5)]
    return gpd.GeoDataFrame(
        {
            "centerline_z_m": np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32),
            "station_m": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "geometry": pts,
        },
        geometry="geometry",
        crs="EPSG:32619",
    )


def test_rasterize_point_seed_surface_keeps_centerline_values_on_seed_cells_only():
    shape = (5, 5)
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    domain = np.ones(shape, dtype=bool)
    seeds = rasterize_point_seed_surface(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=_centerline_points(),
        value_field="centerline_z_m",
    )
    assert np.count_nonzero(np.isfinite(seeds)) == 5
    assert np.all(np.isfinite(seeds[:, 2]))
    assert np.count_nonzero(np.isfinite(seeds[:, :2])) == 0
    assert np.count_nonzero(np.isfinite(seeds[:, 3:])) == 0


def test_build_centerline_core_influence_decays_to_banks_and_scale_changes_core_width():
    shape = (5, 5)
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    domain = np.ones(shape, dtype=bool)
    bank_distance = np.tile(np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32), (5, 1))
    points = _centerline_points()

    narrow = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=3.0,
        influence_scale=0.5,
    )
    base = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=3.0,
        influence_scale=1.0,
    )
    broad = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=3.0,
        influence_scale=2.0,
    )

    assert np.allclose(base[:, 0], 0.0)
    assert np.allclose(base[:, 4], 0.0)
    assert np.all(base[:, 2] > base[:, 1])
    assert np.all(base[:, 2] > base[:, 3])
    assert np.all(broad[:, 1] > base[:, 1])
    assert np.all(base[:, 1] > narrow[:, 1])


def test_build_centerline_core_influence_scale_changes_usable_footprint_not_just_values():
    shape = (5, 9)
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    domain = np.ones(shape, dtype=bool)
    # Approximate a channel where bank distance increases toward the middle.
    bank_distance = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32), (5, 1))
    points = gpd.GeoDataFrame(
        {
            "centerline_z_m": np.linspace(10.0, 14.0, 5, dtype=np.float32),
            "station_m": np.arange(5, dtype=np.float32),
            "geometry": [Point(4.5, 4.5), Point(4.5, 3.5), Point(4.5, 2.5), Point(4.5, 1.5), Point(4.5, 0.5)],
        },
        geometry="geometry",
        crs="EPSG:32619",
    )

    narrow = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=50.0,
        influence_scale=0.5,
    )
    base = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=50.0,
        influence_scale=1.0,
    )
    broad = build_centerline_core_influence(
        shape=shape,
        transform=transform,
        domain_mask=domain,
        points_gdf=points,
        bank_distance_m=bank_distance,
        max_distance_m=50.0,
        influence_scale=10.0,
    )

    narrow_usable = np.count_nonzero(narrow >= 0.05)
    base_usable = np.count_nonzero(base >= 0.05)
    broad_usable = np.count_nonzero(broad >= 0.05)
    assert narrow_usable < base_usable < broad_usable
    assert np.allclose(broad[:, 0], 0.0)
    assert np.allclose(broad[:, -1], 0.0)


def test_primary_river_surface_centerline_respects_centerline_influence_footprint():
    from terrain_interpolator import _build_primary_river_surface

    domain = np.ones((5, 5), dtype=bool)
    centerline_stationing = np.full((5, 5), np.nan, dtype=np.float32)
    centerline_stationing[:, 2] = np.arange(5, dtype=np.float32)
    centerline_elevation = np.full((5, 5), np.nan, dtype=np.float32)
    centerline_elevation[:, 2] = np.linspace(-2.0, -6.0, 5, dtype=np.float32)

    narrow_influence = np.zeros((5, 5), dtype=np.float32)
    narrow_influence[:, 2] = 1.0
    broad_influence = np.zeros((5, 5), dtype=np.float32)
    broad_influence[:, 1:4] = np.array([0.2, 1.0, 0.2], dtype=np.float32)

    zeros = np.zeros((5, 5), dtype=np.float32)
    nans = np.full((5, 5), np.nan, dtype=np.float32)
    centerline_mask = np.isfinite(centerline_stationing)

    zeros_u8 = np.zeros((5, 5), dtype=np.uint8)
    narrow_surface, _, _, _, _, _, _, _, _ = _build_primary_river_surface(
        river_primary_guidance_domain=domain,
        river_transport_centerline=centerline_mask,
        river_centerline_stationing=centerline_stationing,
        river_channel_surface=nans,
        river_channel_surface_confidence=zeros,
        river_channel_surface_source_class=zeros_u8,
        river_channel_surface_support_count=zeros_u8,
        river_channel_surface_prediction_support_confidence=zeros,
        river_channel_surface_measured_anchor_fraction=zeros,
        river_channel_surface_structure_only_fraction=zeros,
        river_channel_surface_low_support_caution=zeros_u8,
        river_channel_surface_prediction_admissibility=zeros,
        river_longitudinal_profile_elevation=nans,
        river_longitudinal_profile_confidence=zeros,
        river_longitudinal_profile_influence=zeros,
        river_centerline_elevation=centerline_elevation,
        river_centerline_confidence=narrow_influence,
        river_centerline_influence=narrow_influence,
        river_xs_support_elevation=nans,
        river_xs_confidence=zeros,
        river_xs_support_weight=zeros,
        river_bank_elevation=nans,
        river_bank_influence=zeros,
        pixel_size_m=1.0,
        along_scale_m=10.0,
        cross_scale_m=2.0,
    )
    broad_surface, _, _, _, _, _, _, _, _ = _build_primary_river_surface(
        river_primary_guidance_domain=domain,
        river_transport_centerline=centerline_mask,
        river_centerline_stationing=centerline_stationing,
        river_channel_surface=nans,
        river_channel_surface_confidence=zeros,
        river_channel_surface_source_class=zeros_u8,
        river_channel_surface_support_count=zeros_u8,
        river_channel_surface_prediction_support_confidence=zeros,
        river_channel_surface_measured_anchor_fraction=zeros,
        river_channel_surface_structure_only_fraction=zeros,
        river_channel_surface_low_support_caution=zeros_u8,
        river_channel_surface_prediction_admissibility=zeros,
        river_longitudinal_profile_elevation=nans,
        river_longitudinal_profile_confidence=zeros,
        river_longitudinal_profile_influence=zeros,
        river_centerline_elevation=centerline_elevation,
        river_centerline_confidence=broad_influence,
        river_centerline_influence=broad_influence,
        river_xs_support_elevation=nans,
        river_xs_confidence=zeros,
        river_xs_support_weight=zeros,
        river_bank_elevation=nans,
        river_bank_influence=zeros,
        pixel_size_m=1.0,
        along_scale_m=10.0,
        cross_scale_m=2.0,
    )

    assert np.count_nonzero(np.isfinite(narrow_surface)) == 5
    assert np.count_nonzero(np.isfinite(broad_surface)) == 5
    assert np.count_nonzero(np.isfinite(narrow_surface[:, :2])) == 0
    assert np.count_nonzero(np.isfinite(narrow_surface[:, 3:])) == 0


def test_centerline_authoritative_support_upgrade_prefers_nearby_absolute_bed_support():
    points = _centerline_points()
    tx = Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True)
    lon, lat = tx.transform(2.5, 4.5)
    support = __import__("pandas").DataFrame({
        "lon": [lon],
        "lat": [lat],
        "depth_m": [7.5],
        "value_semantics": ["absolute_elevation"],
        "authoritative_role": ["authoritative_bed_core"],
        "inside_channel_mask": [1],
        "inside_river_guidance_domain": [1],
        "inside_estuary_clip": [0],
        "role_confidence": [0.95],
    })
    upgraded, receipt = _apply_authoritative_bed_to_centerline_points(
        points, support, target_crs="EPSG:32619", max_distance_m=0.6
    )
    assert int(receipt["matched_centerline_points"]) == 1
    assert bool(upgraded.loc[0, "centerline_authoritative_support_applied"])
    assert float(upgraded.loc[0, "centerline_z_m"]) == 7.5
    assert str(upgraded.loc[0, "centerline_sample_source"]) == "authoritative_support_point"
    assert float(upgraded.loc[1, "centerline_z_m"]) == 11.0
