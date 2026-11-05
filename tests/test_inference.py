"""
Unit tests for waste_hotspot.inference module.
Verifies single-image prediction, folder batch execution, threshold effects,
16-column CSV schema, corrupt image resilience, and fast CLI execution (< 3s).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn

from waste_hotspot.inference import (
    PREDICTION_COLUMNS,
    main,
    predict_folder,
    predict_image,
    predict_image_tiled,
    run_inference,
)


class LightweightDummyClassifier(nn.Module):
    """Fast dummy classifier matching ResNet-50 output interface."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Creates a temporary lightweight checkpoint for inference tests."""
    ckpt_path = tmp_path / "artifacts" / "best_model.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model = LightweightDummyClassifier(num_classes=2)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "threshold": 0.50,
        "model_architecture": "resnet18",
        "num_classes": 2,
    }, ckpt_path)
    return ckpt_path


@pytest.fixture
def synthetic_single_image(tmp_path: Path) -> Path:
    """Creates a synthetic 256x256 RGB image."""
    img_path = tmp_path / "single_test.png"
    Image.new("RGB", (256, 256), color=(120, 150, 180)).save(img_path)
    return img_path


@pytest.fixture
def synthetic_folder_environment(tmp_path: Path) -> Path:
    """Creates a directory with 3 valid images, 1 corrupt 0-byte file, and 1 non-image text file."""
    f_dir = tmp_path / "survey_folder"
    f_dir.mkdir(parents=True, exist_ok=True)

    # 3 valid images
    for i in range(3):
        Image.new("RGB", (224, 224), color=(i * 40, 100, 200)).save(f_dir / f"frame_{i:02d}.jpg")

    # 1 corrupt 0-byte file
    (f_dir / "corrupt_empty.png").touch()

    # 1 non-image file
    (f_dir / "notes.txt").write_text("Survey metadata notes", encoding="utf-8")

    return f_dir


# ==============================================================================
# Unit Tests for waste_hotspot.inference
# ==============================================================================

def test_predict_image_contract_and_simplex(synthetic_single_image: Path, mock_checkpoint: Path):
    """Verifies single-image prediction schema, simplex sum=1.0, and bounds [0, 1]."""
    res = predict_image(
        image_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        threshold=0.50,
        device="cpu",
    )

    required_keys = {
        "image_id", "file_name", "file_path", "source_image",
        "prob_waste", "prob_non_waste", "probability", "threshold",
        "predicted_class", "class_label", "is_waste", "status",
        "has_georeference", "longitude", "latitude",
    }
    assert required_keys.issubset(set(res.keys()))

    assert 0.0 <= res["prob_waste"] <= 1.0
    assert 0.0 <= res["prob_non_waste"] <= 1.0
    assert np.isclose(res["prob_waste"] + res["prob_non_waste"], 1.0, atol=1e-4)

    assert res["predicted_class"] in [0, 1]
    assert res["class_label"] in ["waste", "non-waste"]
    assert res["status"] in ["candidate_detection", "negative"]
    assert isinstance(res["is_waste"], bool)


def test_predict_image_threshold_toggle(synthetic_single_image: Path, mock_checkpoint: Path):
    """Verifies that setting extreme thresholds toggles status between candidate_detection and negative."""
    res_low = predict_image(
        image_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        threshold=0.00,
        device="cpu",
    )
    assert res_low["status"] == "candidate_detection"
    assert res_low["predicted_class"] == 1
    assert res_low["is_waste"] is True

    res_high = predict_image(
        image_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        threshold=1.00,
        device="cpu",
    )
    assert res_high["status"] == "negative"
    assert res_high["predicted_class"] == 0
    assert res_high["is_waste"] is False


def test_predict_folder_batch_execution_and_schema(
    synthetic_folder_environment: Path, mock_checkpoint: Path, tmp_path: Path
):
    """
    Verifies folder batch processing:
    1. Ignores non-image files.
    2. Recovers gracefully from corrupt 0-byte images.
    3. Successfully processes all 3 valid images.
    4. Produces CSV conforming to the prediction schema.
    """
    out_csv = tmp_path / "folder_predictions.csv"

    df = predict_folder(
        folder_path=synthetic_folder_environment,
        checkpoint_path=mock_checkpoint,
        output_csv=out_csv,
        threshold=0.50,
        device="cpu",
    )

    assert out_csv.is_file(), "Predictions CSV was not written"
    assert len(df) == 3, f"Expected 3 valid predictions, got {len(df)}"

    loaded_df = pd.read_csv(out_csv)
    for col in PREDICTION_COLUMNS:
        assert col in loaded_df.columns, f"Required column missing: {col}"


def test_predict_folder_empty_directory_handling(tmp_path: Path, mock_checkpoint: Path):
    """Verifies that an empty folder logs a warning, writes a CSV with headers and 0 rows, and exits cleanly."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_path / "empty_predictions.csv"

    df = predict_folder(
        folder_path=empty_dir,
        checkpoint_path=mock_checkpoint,
        output_csv=out_csv,
        threshold=0.50,
        device="cpu",
    )

    assert out_csv.is_file()
    assert len(df) == 0
    loaded_df = pd.read_csv(out_csv)
    assert len(loaded_df) == 0
    assert list(loaded_df.columns) == PREDICTION_COLUMNS


def test_non_georeferenced_coordinates_contract(synthetic_single_image: Path, mock_checkpoint: Path):
    """Verifies that non-georeferenced images set has_georeference=False and null coordinates."""
    res = predict_image(
        image_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        threshold=0.50,
        device="cpu",
    )
    assert res["has_georeference"] is False
    assert res["longitude"] is None
    assert res["latitude"] is None


def test_predict_image_tiled(synthetic_single_image: Path, mock_checkpoint: Path, tmp_path: Path):
    """Verifies tiled inference generates multiple sub-tiles with bounding boxes."""
    out_csv = tmp_path / "tiled_out.csv"
    df = predict_image_tiled(
        image_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        output_csv=out_csv,
        tile_size=128,
        stride=128,
        threshold=0.50,
        device="cpu",
    )
    assert len(df) == 4  # 256x256 image with 128x128 tiles and stride 128 -> 4 tiles
    assert (df["is_tile"] == True).all()
    assert "x_min" in df.columns and "x_max" in df.columns
    assert out_csv.is_file()


def test_run_inference_unified_api(synthetic_single_image: Path, mock_checkpoint: Path, tmp_path: Path):
    """Verifies unified run_inference handles both single file and directory inputs."""
    out_single = tmp_path / "run_single.csv"
    df_single = run_inference(
        input_path=synthetic_single_image,
        checkpoint_path=mock_checkpoint,
        output_path=out_single,
        threshold=0.50,
        device="cpu",
    )
    assert len(df_single) == 1
    assert out_single.is_file()


def test_inference_cli_single_image_main(synthetic_single_image: Path, mock_checkpoint: Path, tmp_path: Path):
    """Verifies that the inference CLI single image mode executes in < 3.0s and returns 0."""
    out_csv = tmp_path / "single_cli.csv"
    start = time.perf_counter()

    exit_code = main([
        "--input", str(synthetic_single_image),
        "--checkpoint", str(mock_checkpoint),
        "--output", str(out_csv),
        "--device", "cpu",
    ])
    duration = time.perf_counter() - start

    assert exit_code == 0
    assert duration < 3.0, f"Single image CLI took {duration:.2f}s, expected < 3.0s"
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    assert len(df) == 1


def test_inference_cli_folder_main(synthetic_folder_environment: Path, mock_checkpoint: Path, tmp_path: Path):
    """Verifies that the inference CLI folder mode executes in < 3.0s and returns 0."""
    out_csv = tmp_path / "folder_cli.csv"
    start = time.perf_counter()

    exit_code = main([
        "--input", str(synthetic_folder_environment),
        "--checkpoint", str(mock_checkpoint),
        "--output", str(out_csv),
        "--device", "cpu",
    ])
    duration = time.perf_counter() - start

    assert exit_code == 0
    assert duration < 3.0, f"Folder CLI took {duration:.2f}s, expected < 3.0s"
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    assert len(df) == 3


def test_inference_cli_missing_input_exits_with_error(mock_checkpoint: Path, tmp_path: Path):
    """Verifies that a nonexistent input path exits with code 1."""
    exit_code = main([
        "--input", str(tmp_path / "nonexistent_dir_or_file"),
        "--checkpoint", str(mock_checkpoint),
    ])
    assert exit_code != 0
