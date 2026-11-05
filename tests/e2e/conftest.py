"""Synthetic fixtures and helper generators for E2E tests.

Provides deterministic, self-contained fixtures for:
- Synthetic GeoTIFFs (WGS84, UTM, arbitrary affine transforms)
- Synthetic non-georeferenced images (PNG, JPEG)
- Corrupt/truncated image files
- Synthetic AerialWaste dataset manifests (with known duplicates and scene clusters)
- Mock ResNet-50 weights and state dicts
- Detections GeoDataFrames (normal, collinear, empty, single point)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Ensure urban-waste-hotspot-detection/src is on sys.path for test discovery
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import rasterio
from rasterio.transform import from_origin
import shapely.geometry
import geopandas as gpd
import torch
import torch.nn as nn


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return repository root path."""
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Synthetic Raster and Image Generators
# ---------------------------------------------------------------------------

def create_synthetic_geotiff(
    file_path: Path,
    width: int = 512,
    height: int = 512,
    bands: int = 3,
    crs: str = "EPSG:4326",
    origin_x: float = 9.1800,
    origin_y: float = 45.4700,
    pixel_size_x: float = 0.0001,
    pixel_size_y: float = 0.0001,
    pattern: str = "random",
) -> Path:
    """Generate a valid GeoTIFF with known affine transform and CRS.
    
    Upper-left corner: (origin_x, origin_y)
    Resolution: (pixel_size_x, -pixel_size_y)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(origin_x, origin_y, pixel_size_x, pixel_size_y)

    if pattern == "uniform":
        data = np.full((bands, height, width), 128, dtype=np.uint8)
    elif pattern == "gradient":
        row = np.linspace(0, 255, width, dtype=np.uint8)
        data = np.tile(row, (bands, height, 1))
    else:
        rng = np.random.RandomState(42)
        data = rng.randint(0, 256, size=(bands, height, width), dtype=np.uint8)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


def create_synthetic_standard_image(
    file_path: Path,
    width: int = 256,
    height: int = 256,
    format: str = "PNG",
    color: Tuple[int, int, int] = (100, 150, 200),
) -> Path:
    """Create a standard non-georeferenced image file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(file_path, format=format)
    return file_path


def create_corrupt_file(file_path: Path, mode: str = "empty") -> Path:
    """Create a corrupt or 0-byte file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "empty":
        file_path.write_bytes(b"")
    elif mode == "corrupt_header":
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR_CORRUPTED_HEADER_DATA_00000000")
    else:
        file_path.write_text("This is not an image file.", encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Fixtures for Image Datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_geotiff_wgs84(tmp_path: Path) -> Path:
    """GeoTIFF in EPSG:4326 (WGS84), 512x512 pixels, Milan coordinates."""
    path = tmp_path / "imagery" / "milan_scene.tif"
    return create_synthetic_geotiff(
        file_path=path,
        width=512,
        height=512,
        crs="EPSG:4326",
        origin_x=9.1800,
        origin_y=45.4700,
        pixel_size_x=0.0001,
        pixel_size_y=0.0001,
    )


@pytest.fixture
def synthetic_geotiff_utm(tmp_path: Path) -> Path:
    """GeoTIFF in EPSG:32632 (UTM Zone 32N, metric), 1024x1024 pixels."""
    path = tmp_path / "imagery" / "utm_scene.tif"
    return create_synthetic_geotiff(
        file_path=path,
        width=1024,
        height=1024,
        crs="EPSG:32632",
        origin_x=500000.0,
        origin_y=5035000.0,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
    )


@pytest.fixture
def synthetic_large_satellite_scene(tmp_path: Path) -> Path:
    """Large 2048x2048 GeoTIFF representing a full satellite pass."""
    path = tmp_path / "imagery" / "large_satellite_scene.tif"
    return create_synthetic_geotiff(
        file_path=path,
        width=2048,
        height=2048,
        crs="EPSG:4326",
        origin_x=12.4800,
        origin_y=41.8900,
        pixel_size_x=0.00002,
        pixel_size_y=0.00002,
    )


@pytest.fixture
def synthetic_non_geo_image(tmp_path: Path) -> Path:
    """Standard RGB PNG image without georeferencing."""
    path = tmp_path / "imagery" / "drone_photo.png"
    return create_synthetic_standard_image(path, width=512, height=512, format="PNG")


@pytest.fixture
def synthetic_drone_survey_dir(tmp_path: Path) -> Path:
    """Directory with multiple aerial survey images and varying resolutions."""
    survey_dir = tmp_path / "drone_survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    sizes = [(640, 480), (800, 600), (1024, 768), (512, 512)]
    for idx, (w, h) in enumerate(sizes):
        create_synthetic_standard_image(
            survey_dir / f"survey_frame_{idx:03d}.jpg",
            width=w,
            height=h,
            format="JPEG",
            color=(idx * 40, 100, 200 - idx * 30),
        )
    return survey_dir


@pytest.fixture
def synthetic_mixed_images_dir(tmp_path: Path) -> Path:
    """Directory containing valid images, tiny images, and corrupt files."""
    mix_dir = tmp_path / "mixed_inputs"
    mix_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_standard_image(mix_dir / "valid_01.png", 256, 256)
    create_synthetic_standard_image(mix_dir / "valid_02.jpg", 300, 300, format="JPEG")
    create_synthetic_standard_image(mix_dir / "tiny_1x1.png", 1, 1)
    create_synthetic_standard_image(mix_dir / "small_64x64.png", 64, 64)
    create_corrupt_file(mix_dir / "empty_file.png", mode="empty")
    create_corrupt_file(mix_dir / "corrupted_header.jpg", mode="corrupt_header")
    create_corrupt_file(mix_dir / "readme.txt", mode="text")

    return mix_dir


# ---------------------------------------------------------------------------
# Synthetic AerialWaste Metadata & Dataset Manifests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_aerialwaste_environment(tmp_path: Path) -> Dict[str, Path]:
    """Generates a complete mock AerialWaste raw dataset structure.
    
    Includes:
    - training.json with known scene groups and duplicate image IDs
    - testing.json
    - Corresponding dummy image files
    """
    raw_dir = tmp_path / "data" / "raw"
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 10 unique scenes, each containing 3 tiles (total 30 unique physical images)
    # plus 2 intentional duplicate entries in JSON
    images_list = []
    annotations_list = []
    
    img_counter = 1
    for scene_idx in range(1, 11):
        scene_name = f"scene_{scene_idx:02d}"
        for tile_idx in range(1, 4):
            img_id = img_counter
            file_name = f"{scene_name}_tile_{tile_idx}.png"
            file_path = images_dir / file_name
            create_synthetic_standard_image(file_path, width=224, height=224)

            # Assign alternating waste label across scenes to guarantee both classes in all splits
            is_waste = 1 if scene_idx % 2 == 1 else 0
            
            images_list.append({
                "id": img_id,
                "file_name": file_name,
                "width": 224,
                "height": 224,
                "img_source": scene_name,
            })
            annotations_list.append({
                "id": img_id,
                "image_id": img_id,
                "category_id": is_waste,
            })
            img_counter += 1

    # Insert an intentional duplicate ID entry in training.json to test deduplication
    images_list.append({
        "id": 1,  # duplicate of first image
        "file_name": "scene_01_tile_1.png",
        "width": 224,
        "height": 224,
        "img_source": "scene_01",
    })

    categories = [
        {"id": 0, "name": "non-waste"},
        {"id": 1, "name": "waste"},
    ]

    training_data = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }

    # Split: create testing.json with a held-out scene
    test_images = [
        {
            "id": 101,
            "file_name": "test_scene_01.png",
            "width": 224,
            "height": 224,
            "img_source": "test_scene_01",
        },
        {
            "id": 102,
            "file_name": "test_scene_02.png",
            "width": 224,
            "height": 224,
            "img_source": "test_scene_02",
        },
    ]
    for img in test_images:
        create_synthetic_standard_image(images_dir / img["file_name"], 224, 224)

    test_annotations = [
        {"id": 101, "image_id": 101, "category_id": 1},
        {"id": 102, "image_id": 102, "category_id": 0},
    ]

    testing_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories,
    }

    training_json = raw_dir / "training.json"
    testing_json = raw_dir / "testing.json"

    with open(training_json, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    with open(testing_json, "w", encoding="utf-8") as f:
        json.dump(testing_data, f, indent=2)

    return {
        "raw_dir": raw_dir,
        "images_dir": images_dir,
        "training_json": training_json,
        "testing_json": testing_json,
    }


# ---------------------------------------------------------------------------
# Synthetic Checkpoints & Model Fixtures
# ---------------------------------------------------------------------------

class LightweightDummyClassifier(nn.Module):
    """Minimal 2-class classifier matching ResNet-50 output interface for fast unit tests."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.conv(x))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


@pytest.fixture
def mock_classifier_checkpoint(tmp_path: Path) -> Path:
    """Save a synthetic model checkpoint for evaluation and inference tests."""
    ckpt_path = tmp_path / "artifacts" / "best_model.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = LightweightDummyClassifier(num_classes=2)
    checkpoint = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_val_loss": 0.35,
        "best_val_f1": 0.88,
        "threshold": 0.52,
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Synthetic Candidate Detections GeoDataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clustered_detections_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with 2 distinct spatial clusters and 1 noise point in WGS84."""
    data = {
        "image_id": [f"tile_{i}" for i in range(7)],
        "prob_waste": [0.85, 0.90, 0.88, 0.92, 0.79, 0.82, 0.75],
        "threshold": [0.52] * 7,
        "status": ["candidate_detection"] * 7,
        "geometry": [
            # Cluster A (Milan center, ~50m apart)
            shapely.geometry.Point(9.1824, 45.4641),
            shapely.geometry.Point(9.1828, 45.4643),
            shapely.geometry.Point(9.1826, 45.4639),
            # Cluster B (~1.5km away, 3 points)
            shapely.geometry.Point(9.2000, 45.4700),
            shapely.geometry.Point(9.2004, 45.4702),
            shapely.geometry.Point(9.2002, 45.4698),
            # Noise point (far away)
            shapely.geometry.Point(9.2500, 45.5000),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def collinear_detections_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with 3 strictly collinear points (tests convex hull degeneracy)."""
    data = {
        "image_id": ["collinear_0", "collinear_1", "collinear_2"],
        "prob_waste": [0.85, 0.90, 0.88],
        "status": ["candidate_detection"] * 3,
        "geometry": [
            shapely.geometry.Point(9.1800, 45.4600),
            shapely.geometry.Point(9.1810, 45.4610),
            shapely.geometry.Point(9.1820, 45.4620),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def single_detection_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame with exactly one detection point."""
    data = {
        "image_id": ["single_0"],
        "prob_waste": [0.89],
        "status": ["candidate_detection"],
        "geometry": [shapely.geometry.Point(9.1824, 45.4641)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def empty_detections_gdf() -> gpd.GeoDataFrame:
    """Empty GeoDataFrame matching the detections schema."""
    schema = {
        "image_id": pd.Series(dtype="str"),
        "prob_waste": pd.Series(dtype="float32"),
        "threshold": pd.Series(dtype="float32"),
        "status": pd.Series(dtype="str"),
        "geometry": pd.Series(dtype="geometry"),
    }
    return gpd.GeoDataFrame(schema, crs="EPSG:4326")
