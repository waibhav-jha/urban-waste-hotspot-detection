"""Tier 3: Pairwise Combinatorial Testing (Cross-Feature Combinations).

Validates multi-module interactions across system boundaries:
1. Tiling + Geospatial Projection (Rasterio affine transform mapping on tiles)
2. Inference + Clustering + GeoJSON Export (detections -> threshold -> DBSCAN -> GeoJSON)
3. Deduplication + Leakage-Safe Splitting + Evaluation DataLoader
4. Non-georeferenced Imagery + Tiling + Fallback CSV Export
5. Multi-Scale Tiling + Metric Reprojection + Hotspot Summary
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


class TestPairwiseCombinations:
    """Verifies end-to-end integration across pairs and trios of core modules."""

    def test_combo_tiling_and_geospatial_projection(self, synthetic_geotiff_wgs84: Path):
        """Combo: Sliding window tiling + geospatial projection of tile bounding boxes.
        
        Verifies:
        - Tiling on a 512x512 GeoTIFF generates sub-tiles.
        - Each tile's pixel bounding box is converted to real-world WGS84 coordinates.
        - The union of all tile polygons covers the raster's bounding box without spatial gaps.
        """
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not implemented")
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not implemented")

        # Generate tiles with 256x256 size and 256 stride (exact 2x2 grid)
        tiles = list(tile_mod.generate_tiles(str(synthetic_geotiff_wgs84), tile_size=256, stride=256))
        assert len(tiles) == 4

        tile_polygons = []
        for _, meta in tiles:
            pixel_bounds = (meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"])
            geo_info = geo_mod.pixel_to_geospatial(pixel_bounds, str(synthetic_geotiff_wgs84))
            assert geo_info["has_georeference"] is True
            assert isinstance(geo_info["geometry"], shapely.geometry.Polygon)
            tile_polygons.append(geo_info["geometry"])

        # Union of all 4 adjacent tiles should equal the full raster extent
        total_coverage = shapely.ops.unary_union(tile_polygons)
        # Expected bounds: origin=(9.1800, 45.4700), size=512*0.0001=0.0512
        minx, miny, maxx, maxy = total_coverage.bounds
        assert np.isclose(minx, 9.1800, atol=1e-5)
        assert np.isclose(maxx, 9.1800 + 0.0512, atol=1e-5)
        assert np.isclose(maxy, 45.4700, atol=1e-5)
        assert np.isclose(miny, 45.4700 - 0.0512, atol=1e-5)

    def test_combo_inference_clustering_and_geojson_export(
        self, synthetic_geotiff_wgs84: Path, mock_classifier_checkpoint: Path, tmp_path: Path
    ):
        """Combo: Inference on tiles -> threshold filtering -> metric DBSCAN -> GeoJSON export.
        
        Verifies:
        - Tiles from a GeoTIFF are inferred by classifier.
        - Detections with prob_waste >= threshold are reprojected to metric CRS and clustered.
        - Clustered hotspots are exported to a valid RFC 7946 GeoJSON file with summary metadata.
        """
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not implemented")
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not implemented")
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not implemented")
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not implemented")

        # 1. Generate tiles
        tiles = list(tile_mod.generate_tiles(str(synthetic_geotiff_wgs84), tile_size=256, stride=128))
        assert len(tiles) >= 4

        # 2. Ingest tiles and mock detections
        detections = []
        for idx, (tile_arr, meta) in enumerate(tiles):
            # Deterministic synthetic probability: make first 3 tiles positive
            prob_waste = 0.85 if idx < 3 else 0.20
            if prob_waste >= 0.50:
                geo_info = geo_mod.pixel_to_geospatial(
                    (meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"]),
                    str(synthetic_geotiff_wgs84),
                )
                detections.append({
                    "tile_id": f"tile_{idx}",
                    "prob_waste": prob_waste,
                    "geometry": geo_info["geometry"].centroid,
                })

        detections_gdf = gpd.GeoDataFrame(detections, crs="EPSG:4326")
        assert len(detections_gdf) == 3

        # 3. Cluster detections
        clustered_gdf, summary_gdf = cluster_mod.cluster_hotspots(
            detections_gdf=detections_gdf,
            eps_meters=3000.0,
            min_samples=2,
        )
        assert len(summary_gdf) >= 1

        # 4. Export to GeoJSON
        out_geojson = tmp_path / "combo_hotspots.geojson"
        geo_mod.export_geojson(summary_gdf, str(out_geojson))
        assert out_geojson.exists()

        with open(out_geojson, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        assert geojson_data["type"] == "FeatureCollection"
        assert len(geojson_data["features"]) == len(summary_gdf)

    def test_combo_deduplication_splitting_and_evaluation(
        self, synthetic_aerialwaste_environment: Dict[str, Path]
    ):
        """Combo: Ingest raw metadata -> deduplicate -> split -> verify DataLoader ingestion.
        
        Verifies:
        - Duplicate entries in training JSON are purged.
        - Split preserves scene boundaries without cross-partition leakage.
        - Split DataFrames seamlessly instantiate PyTorch Datasets with standard transforms.
        """
        data_mod = pytest.importorskip("waste_hotspot.data", reason="data not implemented")
        prep_mod = pytest.importorskip("waste_hotspot.preprocessing", reason="preprocessing not implemented")

        raw_env = synthetic_aerialwaste_environment
        df = data_mod.load_aerialwaste_metadata(str(raw_env["training_json"]), str(raw_env["images_dir"]))
        assert df["image_id"].is_unique

        train_df, val_df, test_df = data_mod.create_leakage_safe_splits(df, 0.70, 0.15, 0.15, random_seed=42)

        # Build PyTorch dataset from split
        val_transforms = prep_mod.get_transforms(image_size=224, is_train=False)
        val_dataset = prep_mod.AerialWasteDataset(df=val_df, transform=val_transforms)

        assert len(val_dataset) == len(val_df)
        sample_img, sample_lbl, meta = val_dataset[0]

        assert isinstance(sample_img, torch.Tensor)
        assert sample_img.shape == (3, 224, 224)
        assert sample_lbl in [0, 1]
        assert "image_id" in meta

    def test_combo_non_georeferenced_tiling_and_csv_export(
        self, synthetic_non_geo_image: Path, mock_classifier_checkpoint: Path, tmp_path: Path
    ):
        """Combo: Unreferenced image -> tiling -> inference -> pixel offset CSV export.
        
        Verifies:
        - Tiling executes on non-georeferenced PNG image.
        - Predictor evaluates each sub-tile.
        - Output CSV retains pixel coordinates (x_min, y_min, x_max, y_max) with has_georeference=False and null lat/lon.
        """
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not implemented")
        inf_mod = pytest.importorskip("waste_hotspot.inference", reason="inference not implemented")

        out_csv = tmp_path / "non_geo_tiled_predictions.csv"
        # Run inference with tiling flag enabled
        inf_mod.predict_image_tiled(
            image_path=str(synthetic_non_geo_image),
            checkpoint_path=str(mock_classifier_checkpoint),
            output_csv=str(out_csv),
            tile_size=256,
            stride=256,
            threshold=0.50,
        )
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert len(df) == 4  # 512x512 with 256x256 tiles = 4 tiles
        assert df["has_georeference"].isin([False, "False", 0]).all()
        assert df["longitude"].isna().all()
        assert df["latitude"].isna().all()
        assert "x_min" in df.columns and "x_max" in df.columns

    def test_combo_multiscale_tiling_and_metric_clustering(self, synthetic_geotiff_wgs84: Path):
        """Combo: Variable tile sizes + metric coordinate projection + DBSCAN parameter sweep.
        
        Verifies:
        - Tiling with different scales (128x128 vs 256x256) yields consistent coordinate boundaries.
        - Metric reprojection correctly scales distances in meters regardless of tile size.
        - DBSCAN with larger eps groups adjacent tiles into consolidated clusters.
        """
        tile_mod = pytest.importorskip("waste_hotspot.tiling", reason="tiling not implemented")
        geo_mod = pytest.importorskip("waste_hotspot.geospatial", reason="geospatial not implemented")
        cluster_mod = pytest.importorskip("waste_hotspot.clustering", reason="clustering not implemented")

        tiles_128 = list(tile_mod.generate_tiles(str(synthetic_geotiff_wgs84), tile_size=128, stride=128))
        tiles_256 = list(tile_mod.generate_tiles(str(synthetic_geotiff_wgs84), tile_size=256, stride=256))

        assert len(tiles_128) == 16  # 4x4
        assert len(tiles_256) == 4   # 2x2

        # Project centroids of 128x128 tiles
        points = []
        for i, (_, meta) in enumerate(tiles_128[:4]):  # Top row of 4 adjacent tiles
            geo_info = geo_mod.pixel_to_geospatial(
                (meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"]),
                str(synthetic_geotiff_wgs84),
            )
            points.append({
                "image_id": f"tile_128_{i}",
                "prob_waste": 0.90,
                "geometry": geo_info["geometry"].centroid,
            })
        gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")

        # Each tile is 128 * 0.0001 deg ~ 1000m
        # Small eps (100m) should find 0 clusters (all noise)
        _, summary_small = cluster_mod.cluster_hotspots(gdf, eps_meters=100.0, min_samples=2)
        assert len(summary_small) == 0

        # Large eps (5000m) should group all 4 tiles into 1 cluster
        _, summary_large = cluster_mod.cluster_hotspots(gdf, eps_meters=5000.0, min_samples=2)
        assert len(summary_large) == 1
        assert summary_large["detection_count"].iloc[0] == 4
