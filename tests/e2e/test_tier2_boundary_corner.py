"""Tier 2: Boundary & Corner Cases (Boundary Value Analysis).

Validates extreme input boundaries, edge cases, and graceful degradation:
1. Empty image directory passed to inference CLI
2. 1-pixel image (1x1)
3. Corrupt/truncated image files in batch inference
4. Zero candidate detections above threshold (empty detection set)
5. Single isolated candidate detection (min_samples >= 2 -> noise)
6. Collinear candidate points (convex hull collapse -> buffered polygon)
7. Multiple identical coordinate detections (overlapping tiles distance 0.0)
8. Missing or invalid CRS in GeoTIFF
9. Single-class evaluation split (ROC-AUC handling)
10. Extreme decision thresholds (T=0.0 and T=1.0)
11. Tiny image smaller than tile size (e.g. 128x128 with tile=256x256)
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import geopandas as gpd
from PIL import Image


class TestBoundaryAndCornerCases:
    """Verifies pipeline robustness under boundary values and failure conditions."""

    def test_boundary_empty_image_directory(self, tmp_path: Path):
        """Test inference on an empty directory logs warning, writes empty CSV with headers, exits cleanly."""
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not yet implemented")
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tmp_path / "empty_out.csv"

        # Mock model checkpoint
        from conftest import LightweightDummyClassifier
        import torch
        ckpt = tmp_path / "dummy.pth"
        torch.save({"model_state_dict": LightweightDummyClassifier().state_dict(), "threshold": 0.50}, ckpt)

        inf_mod.predict_folder(
            folder_path=str(empty_dir),
            checkpoint_path=str(ckpt),
            output_csv=str(out_csv),
            threshold=0.50,
        )
        assert out_csv.exists(), "CSV with headers should be created even if 0 images"
        df = pd.read_csv(out_csv)
        assert len(df) == 0, "Empty directory must produce 0 prediction rows"

    def test_boundary_single_pixel_image(self, tmp_path: Path):
        """Test 1x1 image is handled without IndexOutOfBounds error."""
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not yet implemented")
        from conftest import create_synthetic_standard_image
        p1 = create_synthetic_standard_image(tmp_path / "p1.png", width=1, height=1)

        tiles = list(tile_mod.generate_tiles(str(p1), tile_size=256, stride=256))
        assert len(tiles) >= 1
        tile_arr, meta = tiles[0]
        # Must be padded or resized to tile_size
        assert tile_arr.shape[:2] == (256, 256) or tile_arr.shape[:2] == (1, 1)

    def test_boundary_corrupt_truncated_file(self, synthetic_mixed_images_dir: Path, tmp_path: Path):
        """Test batch inference catches corrupt files, logs warning, and completes valid files."""
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not yet implemented")
        from conftest import LightweightDummyClassifier
        import torch
        ckpt = tmp_path / "dummy.pth"
        torch.save({"model_state_dict": LightweightDummyClassifier().state_dict(), "threshold": 0.50}, ckpt)

        out_csv = tmp_path / "mixed_out.csv"
        # synthetic_mixed_images_dir contains 2 valid images, 2 tiny images, and 3 corrupt/non-image files
        inf_mod.predict_folder(
            folder_path=str(synthetic_mixed_images_dir),
            checkpoint_path=str(ckpt),
            output_csv=str(out_csv),
            threshold=0.50,
        )
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        # Should process valid images and skip corrupt files without raising unhandled exception
        assert len(df) >= 2, "Valid images in mixed directory must be successfully processed"

    def test_boundary_zero_candidate_detections(self, empty_detections_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """Test DBSCAN and GeoJSON export on 0 candidate detections returns empty structures gracefully."""
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not yet implemented")
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not yet implemented")

        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=empty_detections_gdf,
            eps_meters=100.0,
            min_samples=2,
        )
        assert len(clustered_gdf) == 0
        assert len(summary_gdf) == 0

        out_geojson = tmp_path / "empty_hotspots.geojson"
        geo_mod.export_geojson(summary_gdf, str(out_geojson))
        assert out_geojson.exists()
        with open(out_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 0

    def test_boundary_single_candidate_detection(self, single_detection_gdf: gpd.GeoDataFrame):
        """Test DBSCAN with single point marks it as noise (-1) and reports 0 clusters without crashing."""
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not yet implemented")
        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=single_detection_gdf,
            eps_meters=100.0,
            min_samples=2,
        )
        assert len(clustered_gdf) == 1
        assert clustered_gdf["cluster_id"].iloc[0] == -1, "Single point must be labeled as noise (-1)"
        assert len(summary_gdf) == 0, "No valid clusters should be formed from a single point when min_samples=2"

    def test_boundary_collinear_candidate_points(self, collinear_detections_gdf: gpd.GeoDataFrame):
        """Test collinear points where convex hull collapses to a line/point are buffered to valid Polygons."""
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not yet implemented")
        # eps=2000m to cluster all collinear points
        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=collinear_detections_gdf,
            eps_meters=2000.0,
            min_samples=2,
        )
        assert len(summary_gdf) == 1
        hull = summary_gdf["geometry"].iloc[0]
        # Must be a Polygon (either directly or via buffering)
        assert isinstance(hull, shapely.geometry.Polygon), f"Expected Polygon for GeoJSON export, got {type(hull)}"
        assert hull.area > 0, "Convex hull polygon must have non-zero area"

    def test_boundary_identical_coordinate_detections(self):
        """Test multiple detections at exact same coordinate (distance 0.0) cluster cleanly."""
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not yet implemented")
        same_pt = shapely.geometry.Point(9.1824, 45.4641)
        gdf = gpd.GeoDataFrame({
            "image_id": ["tile_a", "tile_b", "tile_c"],
            "prob_waste": [0.85, 0.92, 0.88],
            "geometry": [same_pt, same_pt, same_pt],
        }, crs="EPSG:4326")

        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=gdf,
            eps_meters=100.0,
            min_samples=2,
        )
        assert len(summary_gdf) == 1
        assert summary_gdf["detection_count"].iloc[0] == 3
        assert np.isclose(summary_gdf["max_confidence"].iloc[0], 0.92)

    def test_boundary_missing_or_invalid_crs(self, tmp_path: Path):
        """Test image with missing CRS falls back to non-georeferenced handling without crashing."""
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not yet implemented")
        # Create a TIFF without CRS
        import rasterio
        from rasterio.transform import Affine
        t_path = tmp_path / "nocrs.tif"
        data = np.zeros((3, 100, 100), dtype=np.uint8)
        with rasterio.open(t_path, "w", driver="GTiff", height=100, width=100, count=3, dtype=np.uint8) as dst:
            dst.write(data)

        geo_info = geo_mod.pixel_to_geospatial((0, 0, 50, 50), str(t_path))
        # Finding no CRS, should set has_georeference=False
        assert geo_info["has_georeference"] is False

    def test_boundary_single_class_evaluation_split(self):
        """Test evaluation handles single-class test set by setting roc_auc=None without crashing."""
        eval_mod = pytest.importorskip("waste_hotspot.evaluate", reason="evaluate not yet implemented")
        # All samples belong to class 0 (non-waste)
        y_true = np.zeros(20, dtype=int)
        y_probs = np.random.uniform(0.1, 0.4, size=(20, 2))
        y_probs[:, 0] = 1.0 - y_probs[:, 1]

        metrics = eval_mod.compute_evaluation_metrics(y_true, y_probs, threshold=0.50)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        # ROC-AUC cannot be calculated for single class
        assert metrics.get("roc_auc") is None or np.isnan(metrics.get("roc_auc", 0))

    def test_boundary_extreme_thresholds(self):
        """Test metrics computation at threshold 0.0 (all positive) and 1.0 (all negative)."""
        eval_mod = pytest.importorskip("waste_hotspot.evaluate", reason="evaluate not yet implemented")
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9],
        ])

        # T = 0.0 -> all positive
        m_zero = eval_mod.compute_evaluation_metrics(y_true, y_probs, threshold=0.0)
        assert m_zero["recall"] == 1.0

        # T = 1.0 -> all negative (recall = 0.0)
        m_one = eval_mod.compute_evaluation_metrics(y_true, y_probs, threshold=1.0)
        assert m_one["recall"] == 0.0

    def test_boundary_tiny_image_smaller_than_tile(self, tmp_path: Path):
        """Test 128x128 image with tile_size=256 produces padded or clamped tile without crashing."""
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not yet implemented")
        from conftest import create_synthetic_standard_image
        small_img = create_synthetic_standard_image(tmp_path / "small.png", 128, 128)

        tiles = list(tile_mod.generate_tiles(str(small_img), tile_size=256, stride=256))
        assert len(tiles) >= 1
        tile_arr, _ = tiles[0]
        assert tile_arr.shape[:2] == (256, 256) or tile_arr.shape[:2] == (128, 128)
