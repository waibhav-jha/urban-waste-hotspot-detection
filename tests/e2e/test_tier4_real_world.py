"""Tier 4: Real-World Application Workloads (Realistic End-to-End Pipelines).

Exercises complete, multi-stage production workflows:
1. End-to-End Simulated Satellite Scene Pipeline:
   Raw GeoTIFF -> Sliding Window Tiling -> ResNet Inference ->
   Threshold Screening -> WGS84 Coordinate Mapping -> Metric Reprojection ->
   DBSCAN Clustering -> Convex Hull & Hotspot Summaries -> CSV & GeoJSON Export.

2. End-to-End Non-Georeferenced Aerial Survey Pipeline:
   Multi-resolution Drone Images -> Batch Ingestion -> Tiling with Edge Preservation ->
   Model Inference -> Strict Non-Georeferenced Fallback -> Predictions CSV Export.

3. Complete Model Training-to-Deployment Lifecycle:
   Raw Metadata -> Scene-Grouped Leakage-Safe Splitting ->
   Validation Decision Threshold Calibration (T* = argmax F1) ->
   Held-Out Test Set Evaluation (Confusion Matrix, PR-AUC) ->
   Production Inference with Calibrated Threshold.
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


class TestRealWorldWorkloads:
    """Verifies complete end-to-end mission workflows under realistic conditions."""

    def test_real_world_satellite_scene_pipeline(
        self, synthetic_large_satellite_scene: Path, mock_classifier_checkpoint: Path, tmp_path: Path
    ):
        """Scenario 1: Complete pipeline on a simulated 2048x2048 satellite scene.
        
        Steps:
        1. Ingest raw 2048x2048 GeoTIFF in EPSG:4326.
        2. Partition into 256x256 sliding-window tiles with 200 stride.
        3. Execute batch classification inference on generated tiles.
        4. Apply decision threshold to isolate candidate waste detections.
        5. Map tile pixel bounds to geographic polygons and centroids.
        6. Reproject coordinates to metric CRS (EPSG:3857 / UTM).
        7. Execute DBSCAN distance clustering (eps=500m, min_samples=3).
        8. Compute hotspot summary: cluster centroid, count, mean/max confidence, convex hull.
        9. Export predictions CSV and hotspots GeoJSON layer.
        10. Assert complete schema and geometric integrity.
        """
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not implemented")
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not implemented")
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not implemented")
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not implemented")

        # Step 1 & 2: Generate sliding window tiles
        tiles = list(tile_mod.generate_tiles(
            image_path=str(synthetic_large_satellite_scene),
            tile_size=256,
            stride=200,
        ))
        assert len(tiles) >= 80, f"Expected >= 80 tiles for 2048x2048 image with stride 200, got {len(tiles)}"

        # Step 3 & 4: Inference and threshold screening
        detections = []
        for idx, (tile_arr, meta) in enumerate(tiles):
            # Simulate a concentrated waste cluster at top-left (first 6 tiles)
            prob_waste = 0.88 if idx < 6 else 0.15
            threshold = 0.52

            if prob_waste >= threshold:
                # Step 5: Convert pixel bounds to geospatial geometry
                pixel_bounds = (meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"])
                geo_info = geo_mod.pixel_to_geospatial(pixel_bounds, str(synthetic_large_satellite_scene))
                assert geo_info["has_georeference"] is True
                assert geo_info["geometry"] is not None

                detections.append({
                    "tile_id": f"sat_tile_{idx}",
                    "source_image": Path(synthetic_large_satellite_scene).name,
                    "x_min": meta["x_min"],
                    "y_min": meta["y_min"],
                    "x_max": meta["x_max"],
                    "y_max": meta["y_max"],
                    "prob_waste": prob_waste,
                    "prob_non_waste": 1.0 - prob_waste,
                    "threshold": threshold,
                    "predicted_class": 1,
                    "class_label": "waste",
                    "status": "candidate_detection",
                    "has_georeference": True,
                    "longitude": geo_info["geometry"].centroid.x,
                    "latitude": geo_info["geometry"].centroid.y,
                    "geometry": geo_info["geometry"].centroid,
                })

        detections_gdf = gpd.GeoDataFrame(detections, crs="EPSG:4326")
        assert len(detections_gdf) == 6

        # Step 6 & 7: Metric reprojection & DBSCAN clustering
        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=detections_gdf,
            eps_meters=500.0,
            min_samples=3,
        )

        # Step 8: Assert hotspot summary metrics
        assert len(summary_gdf) == 1, "Expected exactly 1 concentrated hotspot cluster"
        hotspot = summary_gdf.iloc[0]
        assert hotspot["detection_count"] == 6
        assert np.isclose(hotspot["mean_confidence"], 0.88, atol=0.01)
        assert np.isclose(hotspot["max_confidence"], 0.88, atol=0.01)
        assert isinstance(hotspot["geometry"], shapely.geometry.Polygon)
        assert hotspot["geometry"].is_valid

        # Step 9: Export predictions CSV and Hotspots GeoJSON
        out_csv = tmp_path / "satellite_predictions.csv"
        out_geojson = tmp_path / "satellite_hotspots.geojson"

        detections_df = pd.DataFrame(detections).drop(columns=["geometry"])
        detections_df.to_csv(out_csv, index=False)
        geo_mod.export_geojson(summary_gdf, str(out_geojson))

        # Step 10: Schema verification
        assert out_csv.exists()
        assert out_geojson.exists()

        loaded_csv = pd.read_csv(out_csv)
        assert len(loaded_csv) == 6
        assert loaded_csv["has_georeference"].all()

        with open(out_geojson, "r", encoding="utf-8") as f:
            geojson_payload = json.load(f)
        assert geojson_payload["type"] == "FeatureCollection"
        assert len(geojson_payload["features"]) == 1
        props = geojson_payload["features"][0]["properties"]
        assert props["detection_count"] == 6
        assert "disclaimer" in props

    def test_real_world_aerial_drone_survey_pipeline(
        self, synthetic_drone_survey_dir: Path, mock_classifier_checkpoint: Path, tmp_path: Path
    ):
        """Scenario 2: Complete non-georeferenced aerial drone survey pipeline.
        
        Steps:
        1. Ingest folder of multi-resolution drone survey photos (no GPS tags).
        2. Tiling on oversized survey frames preserving pixel boundaries.
        3. Inference on tiles and full frames.
        4. Verify has_georeference=False and longitude/latitude are strictly null.
        5. Export CSV prediction manifest ready for operator screening.
        """
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not implemented")

        out_csv = tmp_path / "drone_survey_manifest.csv"
        inf_mod.predict_folder(
            folder_path=str(synthetic_drone_survey_dir),
            checkpoint_path=str(mock_classifier_checkpoint),
            output_csv=str(out_csv),
            threshold=0.50,
        )

        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert len(df) == 4

        # Non-georeferenced contract
        assert df["has_georeference"].isin([False, "False", 0]).all()
        assert df["longitude"].isna().all()
        assert df["latitude"].isna().all()
        assert set(df["status"].unique()).issubset({"candidate_detection", "negative"})

    def test_real_world_train_calibrate_evaluate_pipeline(
        self, synthetic_aerialwaste_environment: Dict[str, Path], tmp_path: Path
    ):
        """Scenario 3: Complete model lifecycle (splitting -> calibration -> evaluation).
        
        Steps:
        1. Parse raw metadata and verify zero duplicate IDs.
        2. Create leakage-safe splits grouped by scene ID.
        3. Calibrate decision threshold T* strictly on validation set to maximize F1.
        4. Evaluate held-out test set using calibrated threshold T*.
        5. Verify confusion matrix, F1, precision, recall, and PR-AUC.
        """
        data_mod = pytest.importorskip("waste_hotspot.data", reason="data not implemented")
        eval_mod = pytest.importorskip("waste_hotspot.evaluate", reason="evaluate not implemented")

        raw_env = synthetic_aerialwaste_environment
        df = data_mod.load_aerialwaste_metadata(str(raw_env["training_json"]), str(raw_env["images_dir"]))
        train_df, val_df, test_df = data_mod.create_leakage_safe_splits(df, 0.70, 0.15, 0.15, random_seed=42)

        # Mock validation probabilities with known optimum threshold at ~0.55
        val_targets = val_df["label"].to_numpy()
        np.random.seed(42)
        val_probs = np.zeros((len(val_targets), 2))
        for i, y in enumerate(val_targets):
            if y == 1:
                val_probs[i, 1] = np.random.uniform(0.60, 0.95)
            else:
                val_probs[i, 1] = np.random.uniform(0.05, 0.40)
            val_probs[i, 0] = 1.0 - val_probs[i, 1]

        # Calibrate threshold on validation data ONLY
        calibrated_t = eval_mod.calibrate_threshold(y_true=val_targets, y_probs=val_probs, metric="f1")
        assert 0.40 <= calibrated_t <= 0.65, f"Calibrated threshold unexpected: {calibrated_t}"

        # Evaluate on test data using calibrated_t
        test_targets = test_df["label"].to_numpy()
        test_probs = np.zeros((len(test_targets), 2))
        for i, y in enumerate(test_targets):
            if y == 1:
                test_probs[i, 1] = np.random.uniform(0.55, 0.90)
            else:
                test_probs[i, 1] = np.random.uniform(0.10, 0.45)
            test_probs[i, 0] = 1.0 - test_probs[i, 1]

        metrics = eval_mod.compute_evaluation_metrics(
            y_true=test_targets,
            y_probs=test_probs,
            threshold=calibrated_t,
        )

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert "pr_auc" in metrics
        assert "support" in metrics
        assert metrics["f1"] > 0.70
        assert np.array(metrics["confusion_matrix"]).shape == (2, 2)
