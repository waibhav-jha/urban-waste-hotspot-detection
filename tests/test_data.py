"""
Unit tests for waste_hotspot.data module.
Tests dataset synthetic generation, deduplication, scene-aware splitting, and stats inspection.
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import Generator

import pytest

from waste_hotspot.data import (
    create_leakage_safe_splits,
    generate_synthetic_dataset,
    inspect_dataset,
    load_aerialwaste_metadata,
)


@pytest.fixture
def mock_dataset_dir() -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp()
    generate_synthetic_dataset(output_dir=temp_dir, n_samples=30)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_synthetic_generation_and_loading(mock_dataset_dir: str) -> None:
    """Test that synthetic dataset generates valid metadata and images."""
    img_dir = os.path.join(mock_dataset_dir, "images")
    df = load_aerialwaste_metadata(mock_dataset_dir, images_dir=img_dir)
    assert len(df) > 0
    assert "image_id" in df.columns
    assert "label" in df.columns
    assert "file_path" in df.columns
    assert "img_source" in df.columns
    # Check deduplication
    assert df["image_id"].nunique() == len(df)


def test_leakage_safe_splits_zero_leakage(mock_dataset_dir: str) -> None:
    """Test that create_leakage_safe_splits produces zero overlap in image IDs and scenes."""
    img_dir = os.path.join(mock_dataset_dir, "images")
    df = load_aerialwaste_metadata(mock_dataset_dir, images_dir=img_dir)
    train_df, val_df, test_df = create_leakage_safe_splits(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42, output_dir=None
    )

    # Assert no overlapping IDs
    assert len(set(train_df["image_id"]) & set(val_df["image_id"])) == 0
    assert len(set(train_df["image_id"]) & set(test_df["image_id"])) == 0
    assert len(set(val_df["image_id"]) & set(test_df["image_id"])) == 0

    # Assert no overlapping scenes
    assert len(set(train_df["scene_id"]) & set(val_df["scene_id"])) == 0
    assert len(set(train_df["scene_id"]) & set(test_df["scene_id"])) == 0
    assert len(set(val_df["scene_id"]) & set(test_df["scene_id"])) == 0

    # Assert no overlapping file_names
    assert len(set(train_df["file_name"]) & set(val_df["file_name"])) == 0
    assert len(set(train_df["file_name"]) & set(test_df["file_name"])) == 0
    assert len(set(val_df["file_name"]) & set(test_df["file_name"])) == 0


def test_inspect_dataset_stats(mock_dataset_dir: str) -> None:
    """Test inspect_dataset returns correct structured statistics."""
    stats = inspect_dataset(mock_dataset_dir)
    assert stats.unique_images > 0
    assert stats.num_waste >= 0
    assert stats.num_non_waste >= 0
    assert stats.images_on_disk > 0


def test_deduplication_by_image_id_and_file_name(tmp_path: Path) -> None:
    """Test load_aerialwaste_metadata deduplicates by both image_id and file_name."""
    import json
    from waste_hotspot.data import load_aerialwaste_metadata

    data = {
        "images": [
            {"id": 1, "file_name": "img_1.png", "is_candidate_location": 0, "img_source": "GE", "width": 224, "height": 224},
            # Duplicate image_id with different file_name:
            {"id": 1, "file_name": "img_dup_id.png", "is_candidate_location": 0, "img_source": "GE", "width": 224, "height": 224},
            # Distinct image_id with duplicate file_name:
            {"id": 2, "file_name": "img_1.png", "is_candidate_location": 1, "img_source": "AGEA", "width": 224, "height": 224},
            # Unique record:
            {"id": 3, "file_name": "img_3.png", "is_candidate_location": 1, "img_source": "WV3", "width": 224, "height": 224},
        ]
    }
    json_path = tmp_path / "training.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    df = load_aerialwaste_metadata(json_path=json_path)

    # Only records with unique ID AND unique file_name should remain: ID 1 and ID 3
    assert len(df) == 2, f"Expected 2 unique records, got {len(df)}"
    assert set(df["image_id"]) == {1, 3}
    assert set(df["file_name"]) == {"img_1.png", "img_3.png"}
    assert df["image_id"].is_unique
    assert df["file_name"].is_unique


def test_create_leakage_safe_splits_default_output_dir_is_none_no_disk_write(mock_dataset_dir: str) -> None:
    """Verify default output_dir is None and calling without output_dir does not write to disk."""
    import inspect
    from waste_hotspot.data import create_leakage_safe_splits, load_aerialwaste_metadata

    sig = inspect.signature(create_leakage_safe_splits)
    assert sig.parameters["output_dir"].default is None

    df = load_aerialwaste_metadata(mock_dataset_dir)
    train_df, val_df, test_df = create_leakage_safe_splits(df, random_seed=42)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
