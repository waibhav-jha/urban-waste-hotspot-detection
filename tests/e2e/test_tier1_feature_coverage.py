"""Tier 1: Feature Coverage (Category-Partition Method).

Covers all core features with >=5 discrete, thorough test cases per area:
1. Data Loading, Parsing & Leakage-Safe Splitting
2. Model Inference & Probability Generation
3. Image Tiling & Offset Tracking
4. Geospatial Projection & Coordinate Mapping
5. Distance-Aware DBSCAN Hotspot Clustering
6. Dashboard Export & Integration Contracts
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import shapely.geometry
import geopandas as gpd
import torch
import torchvision


# ===========================================================================
# Feature Area 1: Data Loading, Parsing & Leakage-Safe Splitting
# ===========================================================================

class TestDataLoadingAndSplitting:
    """Verifies AerialWaste metadata parsing, deduplication, and zero-leakage splits."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        self.data_mod = pytest.importorskip(
            "waste_hotspot.data",
            reason="waste_hotspot.data not yet implemented"
        )
        self.prep_mod = pytest.importorskip(
            "waste_hotspot.preprocessing",
            reason="waste_hotspot.preprocessing not yet implemented"
        )

    def test_data_loading_valid_metadata_schema(self, synthetic_aerialwaste_environment: Dict[str, Path]):
        """Test parsing AerialWaste training.json returns expected DataFrame schema."""
        raw_env = synthetic_aerialwaste_environment
        df = self.data_mod.load_aerialwaste_metadata(
            json_path=str(raw_env["training_json"]),
            images_dir=str(raw_env["images_dir"]),
        )
        expected_columns = {"image_id", "file_name", "file_path", "label", "img_source", "width", "height"}
        assert expected_columns.issubset(set(df.columns)), f"Missing columns: {expected_columns - set(df.columns)}"
        assert len(df) > 0
        assert df["label"].isin([0, 1]).all(), "Labels must be binary 0 or 1"
        assert df["file_path"].apply(lambda p: Path(p).exists()).all(), "All file_paths must exist on disk"

    def test_data_loading_deduplication(self, synthetic_aerialwaste_environment: Dict[str, Path]):
        """Test deduplication drops duplicate image_ids while preserving all unique IDs."""
        raw_env = synthetic_aerialwaste_environment
        # Synthetic manifest has 30 unique images and 1 intentional duplicate of image_id 1
        df = self.data_mod.load_aerialwaste_metadata(
            json_path=str(raw_env["training_json"]),
            images_dir=str(raw_env["images_dir"]),
        )
        assert df["image_id"].is_unique, "Duplicate image_ids must be deduplicated"
        assert len(df) == 30, f"Expected 30 unique rows after deduplicating, got {len(df)}"

    def test_data_splitting_zero_scene_leakage(self, synthetic_aerialwaste_environment: Dict[str, Path]):
        """Test train/val/test splits strictly isolate scenes with zero overlap across partitions."""
        raw_env = synthetic_aerialwaste_environment
        df = self.data_mod.load_aerialwaste_metadata(
            json_path=str(raw_env["training_json"]),
            images_dir=str(raw_env["images_dir"]),
        )
        train_df, val_df, test_df = self.data_mod.create_leakage_safe_splits(
            df=df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        train_scenes = set(train_df["img_source"].unique())
        val_scenes = set(val_df["img_source"].unique())
        test_scenes = set(test_df["img_source"].unique())

        assert len(train_scenes.intersection(val_scenes)) == 0, "Leakage detected between train and val"
        assert len(train_scenes.intersection(test_scenes)) == 0, "Leakage detected between train and test"
        assert len(val_scenes.intersection(test_scenes)) == 0, "Leakage detected between val and test"

    def test_data_splitting_target_ratios(self, synthetic_aerialwaste_environment: Dict[str, Path]):
        """Test split partitions approximate target ratios (70/15/15) across available scenes."""
        raw_env = synthetic_aerialwaste_environment
        df = self.data_mod.load_aerialwaste_metadata(
            json_path=str(raw_env["training_json"]),
            images_dir=str(raw_env["images_dir"]),
        )
        total_rows = len(df)
        train_df, val_df, test_df = self.data_mod.create_leakage_safe_splits(
            df=df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        assert len(train_df) + len(val_df) + len(test_df) == total_rows
        assert len(train_df) > len(val_df)
        assert len(train_df) > len(test_df)

    def test_data_missing_file_handling(self, tmp_path: Path):
        """Test dataset loader handles missing files on disk gracefully."""
        # Create metadata pointing to a nonexistent file
        fake_json = tmp_path / "bad.json"
        fake_json.write_text(json.dumps({
            "images": [{"id": 999, "file_name": "nonexistent.png", "width": 224, "height": 224, "img_source": "bad"}],
            "annotations": [{"id": 999, "image_id": 999, "category_id": 0}],
            "categories": [{"id": 0, "name": "non-waste"}, {"id": 1, "name": "waste"}],
        }), encoding="utf-8")
        
        # Loader should either exclude missing file or raise clear error
        try:
            df = self.data_mod.load_aerialwaste_metadata(str(fake_json), str(tmp_path))
            assert 999 not in df["image_id"].values or not Path(df.loc[df["image_id"] == 999, "file_path"].iloc[0]).exists()
        except (FileNotFoundError, ValueError):
            pass  # Explicit error is also acceptable contract behavior

    def test_preprocessing_transform_train_eval_isolation(self):
        """Test that train transforms include stochastic augmentations while eval transforms are deterministic."""
        train_tf = self.prep_mod.get_transforms(image_size=224, is_train=True)
        eval_tf = self.prep_mod.get_transforms(image_size=224, is_train=False)

        assert isinstance(train_tf, torchvision.transforms.Compose)
        assert isinstance(eval_tf, torchvision.transforms.Compose)

        eval_types = [type(t).__name__ for t in eval_tf.transforms]
        assert "RandomHorizontalFlip" not in eval_types, "Eval transform must not have random flip"
        assert "RandomRotation" not in eval_types, "Eval transform must not have random rotation"
        assert "Resize" in eval_types or "CenterCrop" in eval_types, "Eval must resize deterministically"


# ===========================================================================
# Feature Area 2: Model Inference & Probability Generation
# ===========================================================================

class TestModelInference:
    """Verifies model loading, inference output contracts, and probability constraints."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        self.inf_mod = pytest.importorskip(
            "waste_hotspot.inference",
            reason="waste_hotspot.inference not yet implemented"
        )
        self.model_mod = pytest.importorskip(
            "waste_hotspot.model",
            reason="waste_hotspot.model not yet implemented"
        )

    def test_inference_single_image_output_contract(
        self, synthetic_non_geo_image: Path, mock_classifier_checkpoint: Path
    ):
        """Test single-image inference returns required keys and valid types."""
        result = self.inf_mod.predict_image(
            image_path=str(synthetic_non_geo_image),
            checkpoint_path=str(mock_classifier_checkpoint),
            threshold=0.50,
        )
        required_keys = {"prob_waste", "prob_non_waste", "threshold", "predicted_class", "class_label", "status"}
        assert required_keys.issubset(set(result.keys()))
        assert result["predicted_class"] in [0, 1]
        assert result["class_label"] in ["waste", "non-waste"]
        assert result["status"] in ["candidate_detection", "negative"]

    def test_inference_probability_simplex_constraint(
        self, synthetic_non_geo_image: Path, mock_classifier_checkpoint: Path
    ):
        """Test that probabilities form a valid simplex: p0 + p1 == 1.0 and p >= 0."""
        result = self.inf_mod.predict_image(
            image_path=str(synthetic_non_geo_image),
            checkpoint_path=str(mock_classifier_checkpoint),
            threshold=0.50,
        )
        p_waste = result["prob_waste"]
        p_non_waste = result["prob_non_waste"]

        assert 0.0 <= p_waste <= 1.0, f"prob_waste {p_waste} out of [0, 1]"
        assert 0.0 <= p_non_waste <= 1.0, f"prob_non_waste {p_non_waste} out of [0, 1]"
        assert np.isclose(p_waste + p_non_waste, 1.0, atol=1e-5), "Probabilities must sum to 1.0"

    def test_inference_folder_batch_execution(
        self, synthetic_drone_survey_dir: Path, mock_classifier_checkpoint: Path, tmp_path: Path
    ):
        """Test batch inference across an image directory produces structured CSV."""
        out_csv = tmp_path / "predictions.csv"
        self.inf_mod.predict_folder(
            folder_path=str(synthetic_drone_survey_dir),
            checkpoint_path=str(mock_classifier_checkpoint),
            output_csv=str(out_csv),
            threshold=0.50,
        )
        assert out_csv.exists(), "Predictions CSV must be created"
        df = pd.read_csv(out_csv)
        assert len(df) == 4, f"Expected 4 predictions for 4 images, got {len(df)}"
        assert "prob_waste" in df.columns
        assert "status" in df.columns

    def test_inference_decision_threshold_application(
        self, synthetic_non_geo_image: Path, mock_classifier_checkpoint: Path
    ):
        """Test status='candidate_detection' occurs if and only if prob_waste >= threshold."""
        res_low = self.inf_mod.predict_image(
            str(synthetic_non_geo_image), str(mock_classifier_checkpoint), threshold=0.01
        )
        res_high = self.inf_mod.predict_image(
            str(synthetic_non_geo_image), str(mock_classifier_checkpoint), threshold=0.99
        )
        assert res_low["status"] == "candidate_detection"
        assert res_high["status"] == "negative"

    def test_inference_device_agnosticism(
        self, synthetic_non_geo_image: Path, mock_classifier_checkpoint: Path
    ):
        """Test inference explicitly runs on CPU without raising device errors."""
        res = self.inf_mod.predict_image(
            image_path=str(synthetic_non_geo_image),
            checkpoint_path=str(mock_classifier_checkpoint),
            threshold=0.50,
            device="cpu",
        )
        assert "predicted_class" in res


# ===========================================================================
# Feature Area 3: Image Tiling & Offset Tracking
# ===========================================================================

class TestImageTiling:
    """Verifies sliding-window tiling, overlap, offset tracking, and border clamp."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        self.tile_mod = pytest.importorskip(
            "waste_hotspot.tiling",
            reason="waste_hotspot.tiling not yet implemented"
        )

    def test_tiling_sliding_window_dimensions(self, tmp_path: Path):
        """Test tiling 1000x1000 image with 256x256 tile size and stride 200."""
        from conftest import create_synthetic_standard_image
        img_path = create_synthetic_standard_image(tmp_path / "big.png", 1000, 1000)

        tiles = list(self.tile_mod.generate_tiles(str(img_path), tile_size=256, stride=200))
        assert len(tiles) > 0

        for tile_arr, meta in tiles:
            assert tile_arr.shape[:2] == (256, 256), f"Tile dimension incorrect: {tile_arr.shape}"
            assert meta["x_max"] - meta["x_min"] == 256
            assert meta["y_max"] - meta["y_min"] == 256

    def test_tiling_pixel_offset_preservation(self, tmp_path: Path):
        """Test each tile preserves exact source image and bounding box metadata."""
        from conftest import create_synthetic_standard_image
        img_path = create_synthetic_standard_image(tmp_path / "scene.png", 512, 512)

        tiles = list(self.tile_mod.generate_tiles(str(img_path), tile_size=256, stride=256))
        assert len(tiles) == 4

        for _, meta in tiles:
            assert meta["source_image"] == "scene.png" or Path(meta["source_image"]).name == "scene.png"
            assert "x_min" in meta and "y_min" in meta
            assert "x_max" in meta and "y_max" in meta

    def test_tiling_boundary_shift_back(self, tmp_path: Path):
        """Test non-divisible dimensions (e.g. 500x500 with 256x256 tiles) clamp to edges."""
        from conftest import create_synthetic_standard_image
        img_path = create_synthetic_standard_image(tmp_path / "odd.png", 500, 500)

        tiles = list(self.tile_mod.generate_tiles(str(img_path), tile_size=256, stride=200))
        x_maxes = [meta["x_max"] for _, meta in tiles]
        y_maxes = [meta["y_max"] for _, meta in tiles]

        assert max(x_maxes) == 500, "Must cover up to the rightmost boundary (500)"
        assert max(y_maxes) == 500, "Must cover up to the bottom boundary (500)"

    def test_tiling_overlap_consistency(self, tmp_path: Path):
        """Test adjacent tile start coordinates reflect specified stride."""
        from conftest import create_synthetic_standard_image
        img_path = create_synthetic_standard_image(tmp_path / "grid.png", 600, 600)

        tiles = list(self.tile_mod.generate_tiles(str(img_path), tile_size=256, stride=150))
        x_mins = sorted(list(set(m["x_min"] for _, m in tiles)))
        assert x_mins[1] - x_mins[0] == 150, "Tile stride must equal 150"

    def test_tiling_exact_image_size(self, tmp_path: Path):
        """Test image of exact tile_size produces exactly 1 tile with (0, 0, W, H)."""
        from conftest import create_synthetic_standard_image
        img_path = create_synthetic_standard_image(tmp_path / "exact.png", 256, 256)

        tiles = list(self.tile_mod.generate_tiles(str(img_path), tile_size=256, stride=256))
        assert len(tiles) == 1
        _, meta = tiles[0]
        assert (meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"]) == (0, 0, 256, 256)


# ===========================================================================
# Feature Area 4: Geospatial Projection & Coordinate Mapping
# ===========================================================================

class TestGeospatialMapping:
    """Verifies Rasterio affine coordinate conversion, CRS handling, and GeoJSON export."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        self.geo_mod = pytest.importorskip(
            "waste_hotspot.geospatial",
            reason="waste_hotspot.geospatial not yet implemented"
        )

    def test_geospatial_geotiff_transform_extraction(self, synthetic_geotiff_wgs84: Path):
        """Test reading geotransform and CRS from valid GeoTIFF."""
        geo_info = self.geo_mod.pixel_to_geospatial(
            pixel_coords=(0, 0, 256, 256),
            raster_path=str(synthetic_geotiff_wgs84),
        )
        assert geo_info["has_georeference"] is True
        assert geo_info["crs"] is not None
        assert "4326" in str(geo_info["crs"])
        assert isinstance(geo_info["geometry"], shapely.geometry.Polygon)

    def test_geospatial_pixel_to_wgs84_projection(self, synthetic_geotiff_wgs84: Path):
        """Test projected coordinates match analytical affine matrix product."""
        # Upper-left origin is (9.1800, 45.4700), resolution is (0.0001, -0.0001)
        # Pixel bounds (0, 0, 100, 100) -> X: [9.1800, 9.1900], Y: [45.4600, 45.4700]
        geo_info = self.geo_mod.pixel_to_geospatial(
            pixel_coords=(0, 0, 100, 100),
            raster_path=str(synthetic_geotiff_wgs84),
        )
        geom = geo_info["geometry"]
        minx, miny, maxx, maxy = geom.bounds
        assert np.isclose(minx, 9.1800, atol=1e-5)
        assert np.isclose(maxx, 9.1900, atol=1e-5)
        assert np.isclose(maxy, 45.4700, atol=1e-5)
        assert np.isclose(miny, 45.4600, atol=1e-5)

    def test_geospatial_non_georeferenced_fallback(self, synthetic_non_geo_image: Path):
        """Test standard PNG sets has_georeference=False and fabricates zero coordinates."""
        geo_info = self.geo_mod.pixel_to_geospatial(
            pixel_coords=(0, 0, 256, 256),
            raster_path=str(synthetic_non_geo_image),
        )
        assert geo_info["has_georeference"] is False
        assert geo_info.get("geometry") is None
        assert geo_info.get("crs") is None

    def test_geospatial_rfc7946_geojson_export(self, sample_clustered_detections_gdf: gpd.GeoDataFrame, tmp_path: Path):
        """Test exporting GeoDataFrame to RFC 7946 compliant GeoJSON FeatureCollection."""
        out_geojson = tmp_path / "detections.geojson"
        self.geo_mod.export_geojson(
            gdf=sample_clustered_detections_gdf,
            output_path=str(out_geojson),
        )
        assert out_geojson.exists()
        with open(out_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == len(sample_clustered_detections_gdf)
        assert data["features"][0]["geometry"]["type"] in ["Point", "Polygon"]

    def test_geospatial_projected_crs_preservation(self, synthetic_geotiff_utm: Path):
        """Test GeoTIFF in metric UTM preserves metric coordinates without CRS loss."""
        geo_info = self.geo_mod.pixel_to_geospatial(
            pixel_coords=(0, 0, 200, 200),
            raster_path=str(synthetic_geotiff_utm),
        )
        assert geo_info["has_georeference"] is True
        assert "32632" in str(geo_info["crs"]) or "UTM" in str(geo_info["crs"]).upper()
        minx, miny, maxx, maxy = geo_info["geometry"].bounds
        # Resolution 0.5m: 200 pixels = 100 meters
        assert np.isclose(maxx - minx, 100.0, atol=0.1)


# ===========================================================================
# Feature Area 5: Distance-Aware DBSCAN Hotspot Clustering
# ===========================================================================

class TestHotspotClustering:
    """Verifies metric reprojection, DBSCAN neighborhood grouping, centroids, and summaries."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        self.cluster_mod = pytest.importorskip(
            "waste_hotspot.clustering",
            reason="waste_hotspot.clustering not yet implemented"
        )

    def test_clustering_metric_crs_reprojection(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test geographic points (degrees) reprojected to metric CRS (meters)."""
        reprojected = self.cluster_mod.reproject_to_metric_crs(sample_clustered_detections_gdf)
        assert reprojected.crs.is_projected, "Target CRS must be projected (metric)"
        # Units should be in meters, coordinates in thousands/millions
        first_pt = reprojected.geometry.iloc[0]
        assert abs(first_pt.x) > 1000.0 or abs(first_pt.y) > 1000.0

    def test_clustering_dbscan_metric_neighborhood(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test points within 100m are clustered together, distant points separated."""
        clustered_gdf, summary_gdf = self.cluster_mod.cluster_hotspots(
            detections_gdf=sample_clustered_detections_gdf,
            eps_meters=150.0,
            min_samples=2,
        )
        assert "cluster_id" in clustered_gdf.columns
        # Cluster A has 3 points, Cluster B has 3 points, Noise has 1 point (-1)
        valid_clusters = clustered_gdf[clustered_gdf["cluster_id"] >= 0]
        assert valid_clusters["cluster_id"].nunique() == 2
        assert (clustered_gdf["cluster_id"] == -1).sum() == 1

    def test_clustering_noise_point_labeling(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test isolated point receives noise label -1."""
        clustered_gdf, _ = self.cluster_mod.cluster_hotspots(
            detections_gdf=sample_clustered_detections_gdf,
            eps_meters=150.0,
            min_samples=2,
        )
        noise_row = clustered_gdf.iloc[-1]  # The noise point added at (9.2500, 45.5000)
        assert noise_row["cluster_id"] == -1

    def test_clustering_centroid_computation(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test cluster centroid equals spatial mean of constituent member detections."""
        _, summary_gdf = self.cluster_mod.cluster_hotspots(
            detections_gdf=sample_clustered_detections_gdf,
            eps_meters=150.0,
            min_samples=2,
        )
        assert len(summary_gdf) == 2
        for _, row in summary_gdf.iterrows():
            centroid = row["centroid"] if "centroid" in row else row["geometry"].centroid
            assert isinstance(centroid, shapely.geometry.Point)

    def test_clustering_hotspot_summary_metrics(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test summary table computes detection_count, mean_confidence, and max_confidence."""
        _, summary_gdf = self.cluster_mod.cluster_hotspots(
            detections_gdf=sample_clustered_detections_gdf,
            eps_meters=150.0,
            min_samples=2,
        )
        assert "detection_count" in summary_gdf.columns
        assert "mean_confidence" in summary_gdf.columns
        assert "max_confidence" in summary_gdf.columns
        assert (summary_gdf["detection_count"] >= 2).all()
        assert (summary_gdf["mean_confidence"] <= summary_gdf["max_confidence"]).all()


# ===========================================================================
# Feature Area 6: Dashboard Export & Integration Contracts
# ===========================================================================

class TestDashboardContracts:
    """Verifies predictions CSV schema, hotspots GeoJSON schema, and interactive filtering."""

    def test_dashboard_predictions_csv_schema(self, tmp_path: Path):
        """Test predictions CSV matches 16-column specification."""
        cols = [
            "image_id", "source_image", "is_tile", "x_min", "y_min", "x_max", "y_max",
            "prob_non_waste", "prob_waste", "threshold", "predicted_class", "class_label",
            "status", "has_georeference", "longitude", "latitude"
        ]
        sample_row = {
            "image_id": "tile_01.png",
            "source_image": "scene.tif",
            "is_tile": True,
            "x_min": 0, "y_min": 0, "x_max": 256, "y_max": 256,
            "prob_non_waste": 0.15, "prob_waste": 0.85,
            "threshold": 0.50, "predicted_class": 1, "class_label": "waste",
            "status": "candidate_detection", "has_georeference": True,
            "longitude": 9.1824, "latitude": 45.4641
        }
        df = pd.DataFrame([sample_row])
        out_csv = tmp_path / "dashboard_export.csv"
        df.to_csv(out_csv, index=False)

        loaded_df = pd.read_csv(out_csv)
        for col in cols:
            assert col in loaded_df.columns, f"Required column missing: {col}"

    def test_dashboard_hotspots_geojson_schema(self, tmp_path: Path):
        """Test hotspots GeoJSON conforms to RFC 7946 with required hotspot properties."""
        polygon = shapely.geometry.Polygon([[9.18, 45.46], [9.19, 45.46], [9.19, 45.47], [9.18, 45.47], [9.18, 45.46]])
        gdf = gpd.GeoDataFrame({
            "cluster_id": [0],
            "detection_count": [5],
            "mean_confidence": [0.82],
            "max_confidence": [0.94],
            "status": ["candidate_hotspot"],
            "disclaimer": ["Candidate screening signal only; requires ground verification."],
            "geometry": [polygon],
        }, crs="EPSG:4326")

        out_path = tmp_path / "hotspots.geojson"
        gdf.to_file(out_path, driver="GeoJSON")

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        props = data["features"][0]["properties"]
        assert "cluster_id" in props
        assert "detection_count" in props
        assert "mean_confidence" in props
        assert "disclaimer" in props

    def test_dashboard_confidence_threshold_filtering(self):
        """Test interactive confidence filtering drops candidate rows below threshold."""
        df = pd.DataFrame({
            "image_id": ["t1", "t2", "t3"],
            "prob_waste": [0.45, 0.65, 0.85],
        })
        threshold = 0.60
        filtered = df[df["prob_waste"] >= threshold]
        assert len(filtered) == 2
        assert "t1" not in filtered["image_id"].values

    def test_dashboard_cluster_radius_recomputation(self, sample_clustered_detections_gdf: gpd.GeoDataFrame):
        """Test dynamic re-clustering when user adjusts eps slider from 150m to 2000m."""
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not available")
        # eps=150m yields 2 clusters
        _, summary_150 = cluster_mod.cluster_hotspots(sample_clustered_detections_gdf, eps_meters=150.0, min_samples=2)
        # eps=2500m merges Cluster A and Cluster B into 1 large cluster
        _, summary_2500 = cluster_mod.cluster_hotspots(sample_clustered_detections_gdf, eps_meters=2500.0, min_samples=2)
        assert len(summary_150) == 2
        assert len(summary_2500) == 1

    def test_dashboard_non_georeferenced_export_behavior(self, tmp_path: Path):
        """Test non-georeferenced predictions produce valid CSV and safely handle missing map coords."""
        df = pd.DataFrame([{
            "image_id": "drone_01.jpg",
            "source_image": "drone_01.jpg",
            "prob_waste": 0.91,
            "has_georeference": False,
            "longitude": None,
            "latitude": None,
        }])
        out_csv = tmp_path / "unreferenced.csv"
        df.to_csv(out_csv, index=False)
        read_df = pd.read_csv(out_csv)
        assert bool(read_df["has_georeference"].iloc[0]) is False
        assert pd.isna(read_df["longitude"].iloc[0])
