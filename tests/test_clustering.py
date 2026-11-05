"""Unit tests for spatial DBSCAN clustering module."""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry
from shapely.geometry import Point

from waste_hotspot.clustering import cluster_hotspots, reproject_to_metric_crs


def test_reproject_to_metric_crs_degrees_to_meters():
    gdf = gpd.GeoDataFrame([
        {"geometry": Point(9.1824, 45.4641)},
        {"geometry": Point(9.1830, 45.4645)},
    ], crs="EPSG:4326")

    metric_gdf = reproject_to_metric_crs(gdf)
    assert metric_gdf.crs.is_projected
    assert abs(metric_gdf.geometry.iloc[0].x) > 1000.0


def test_cluster_hotspots_grouping():
    # 3 points close together (~10m), 1 far point (~10km)
    gdf = gpd.GeoDataFrame([
        {"prob_waste": 0.90, "geometry": Point(9.18000, 45.47000)},
        {"prob_waste": 0.85, "geometry": Point(9.18005, 45.47005)},
        {"prob_waste": 0.88, "geometry": Point(9.18010, 45.47000)},
        {"prob_waste": 0.75, "geometry": Point(9.30000, 45.55000)},
    ], crs="EPSG:4326")

    clustered, summary = cluster_hotspots(gdf, eps_meters=150.0, min_samples=2)
    assert "cluster_id" in clustered.columns
    assert (clustered["cluster_id"] == 0).sum() == 3
    assert (clustered["cluster_id"] == -1).sum() == 1  # Noise
    assert len(summary) == 1
    assert summary["detection_count"].iloc[0] == 3
    assert isinstance(summary["geometry"].iloc[0], shapely.geometry.Polygon)
    assert summary["geometry"].iloc[0].area > 0


def test_cluster_hotspots_empty():
    gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    clustered, summary = cluster_hotspots(gdf, eps_meters=150.0, min_samples=2)
    assert len(clustered) == 0
    assert len(summary) == 0
    assert "cluster_id" in clustered.columns

