"""Unit tests for geospatial mapping and GeoJSON export module."""

from pathlib import Path
import json
import numpy as np
import pytest
import shapely.geometry
import geopandas as gpd

from waste_hotspot.geospatial import pixel_to_geospatial, export_geojson
from tests.e2e.conftest import create_synthetic_geotiff, create_synthetic_standard_image


def test_pixel_to_geospatial_valid_geotiff(tmp_path: Path):
    tif_path = tmp_path / "valid.tif"
    create_synthetic_geotiff(tif_path, width=256, height=256, crs="EPSG:4326", origin_x=10.0, origin_y=50.0)

    res = pixel_to_geospatial((0, 0, 100, 100), tif_path)
    assert res["has_georeference"] is True
    assert "4326" in str(res["crs"])
    assert isinstance(res["geometry"], shapely.geometry.Polygon)


def test_pixel_to_geospatial_non_georeferenced_png(tmp_path: Path):
    png_path = tmp_path / "photo.png"
    create_synthetic_standard_image(png_path, width=200, height=200)

    res = pixel_to_geospatial((0, 0, 100, 100), png_path)
    assert res["has_georeference"] is False
    assert res["crs"] is None
    assert res["geometry"] is None


def test_export_geojson_feature_collection(tmp_path: Path):
    p = shapely.geometry.Point(10.5, 45.5)
    gdf = gpd.GeoDataFrame([{"id": 1, "status": "candidate", "geometry": p}], crs="EPSG:4326")
    out_file = tmp_path / "out.geojson"
    export_geojson(gdf, out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1


def test_export_geojson_empty(tmp_path: Path):
    gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    out_file = tmp_path / "empty.geojson"
    export_geojson(gdf, out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 0

