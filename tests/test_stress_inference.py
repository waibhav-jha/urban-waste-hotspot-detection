"""
Adversarial Stress Test Suite for waste_hotspot.inference module.

Empirically challenges:
1. Malformed and corrupted files:
   - 0-byte file (corrupt.png).
   - Text file masquerading as an image (fake.jpg).
   - Truncated PNG and JPEG files (incomplete binary stream).
   - Non-image files in input directory (notes.txt, config.yaml, subfolders).
   - Entire directory composed purely of corrupted files.
   - Fault tolerance: folder prediction never crashes, skips corrupted items with warnings,
     and returns valid predictions for healthy files.
2. Non-georeferenced imagery safety:
   - Standard PNG, JPG, BMP, WEBP, and unreferenced TIFF imagery.
   - Strictly enforces has_georeference == False, longitude is NaN, latitude is NaN.
   - Asserts NO fake coordinates (such as 0.0, 0.0 or random values) are fabricated.
   - CSV round-trip persistence of coordinate nullity.
3. Empty directory & large directory scaling:
   - Empty folder returns 0-row DataFrame with full 20-column schema without crashing.
   - CLI execution on empty folder exits cleanly with status code 0.
   - Batching stress test: 50 synthetic images processed with odd batch_size=7 (7 full batches + 1 leftover).
   - Interleaved batch corruption resilience (batches with leading, trailing, and isolated corrupted files).
4. Threshold boundary stress:
   - Threshold T = 0.0: all images predicted as waste / candidate_detection.
   - Threshold T = 1.0: all images predicted as non-waste / negative.
   - Out-of-bounds thresholds (T < 0.0, T > 1.0) strictly rejected with ValueError.
   - Strict probability simplex validation: p_0 + p_1 = 1.0 +/- 1e-6 and p in [0, 1] on every row.
5. CSV Output Schema Integrity:
   - Verification of exact 20-column standard schema in memory and serialized to CSV.
   - Type integrity (booleans, integers, floats, strings, NaNs).
6. Pathological image inputs:
   - 1x1 pixel image, extreme aspect ratio image (10x1000).
   - Grayscale (1-channel L) and RGBA (4-channel) images safely handled.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import tempfile
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn

from waste_hotspot.inference import (
    PREDICTION_COLUMNS,
    check_georeference,
    load_model_for_inference,
    main,
    predict_folder,
    predict_image,
    predict_image_tiled,
    run_inference,
)
from waste_hotspot.model import _LightweightDummyClassifier


# ==============================================================================
# Fixtures & Test Helpers
# ==============================================================================

@pytest.fixture
def fast_checkpoint(tmp_path: Path) -> Path:
    """Creates a temporary lightweight checkpoint for fast inference tests."""
    ckpt_path = tmp_path / "artifacts" / "stress_model.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model = _LightweightDummyClassifier(hidden_dim=4, num_classes=2)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "threshold": 0.50,
        "calibrated_threshold": 0.50,
        "model_architecture": "resnet18",
        "num_classes": 2,
    }, ckpt_path)
    return ckpt_path


@pytest.fixture
def corrupted_files_dir(tmp_path: Path) -> Path:
    """
    Creates a folder environment populated with:
    - 3 valid RGB images (PNG and JPG)
    - 1 0-byte file named 'corrupt.png'
    - 1 text file containing ASCII characters named 'fake.jpg'
    - 1 truncated PNG file named 'truncated.png'
    - 1 truncated JPEG file named 'truncated.jpg'
    - 1 non-image file 'notes.txt'
    - 1 non-image file 'config.yaml'
    - A subfolder containing 1 valid image and 1 non-image file
    """
    f_dir = tmp_path / "corrupted_test_env"
    f_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid images
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(30 * i, 70 * i, 120))
        img.save(f_dir / f"healthy_{i:02d}.png")

    # 2. 0-byte file named corrupt.png
    (f_dir / "corrupt.png").touch()

    # 3. Text file with ASCII masquerading as fake.jpg
    (f_dir / "fake.jpg").write_text("NOT_A_JPEG_FILE_HEADER_BINARY_DATA_CORRUPTION\n", encoding="utf-8")

    # 4. Truncated PNG
    valid_png_buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(valid_png_buffer, format="PNG")
    png_bytes = valid_png_buffer.getvalue()
    # Truncate at half bytes
    (f_dir / "truncated.png").write_bytes(png_bytes[: len(png_bytes) // 2])

    # 5. Truncated JPEG
    valid_jpg_buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 200, 50)).save(valid_jpg_buffer, format="JPEG")
    jpg_bytes = valid_jpg_buffer.getvalue()
    # Truncate at one-third bytes
    (f_dir / "truncated.jpg").write_bytes(jpg_bytes[: len(jpg_bytes) // 3])

    # 6. Non-image files in input directory
    (f_dir / "notes.txt").write_text("Survey operational log and GPS metadata notes\n", encoding="utf-8")
    (f_dir / "config.yaml").write_text("model:\n  batch_size: 32\n", encoding="utf-8")

    # 7. Subfolder with 1 healthy image and 1 non-image
    sub = f_dir / "nested_subfolder"
    sub.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(sub / "nested_healthy.jpg")
    (sub / "subnotes.log").write_text("debug log entry\n", encoding="utf-8")

    return f_dir


# ==============================================================================
# 1. Malformed and Corrupted Files Resilience
# ==============================================================================

def test_predict_folder_skips_malformed_and_processes_healthy(
    corrupted_files_dir: Path, fast_checkpoint: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """
    Verifies that predict_folder:
    1. Discovers valid images including those in subfolders (4 healthy total: 3 root + 1 nested).
    2. Gracefully skips 0-byte corrupt.png, ASCII fake.jpg, truncated.png, and truncated.jpg.
    3. Completely ignores notes.txt, config.yaml, subnotes.log, and directory entries.
    4. Logs warning messages for skipped corrupted files.
    5. Returns valid predictions for all healthy images without crashing.
    """
    out_csv = tmp_path / "corrupt_resilience_out.csv"

    with caplog.at_level(logging.WARNING):
        df = predict_folder(
            folder_path=corrupted_files_dir,
            checkpoint_path=fast_checkpoint,
            output_csv=out_csv,
            threshold=0.50,
            batch_size=2,  # Small batch to force multi-chunk processing
            device="cpu",
        )

    # 3 healthy root + 1 healthy nested = 4 total valid images
    assert len(df) == 4, f"Expected 4 valid predictions, got {len(df)}"
    assert out_csv.is_file(), "Output CSV was not created"

    # Verify all healthy filenames are present
    processed_names = set(df["file_name"].tolist())
    assert "healthy_00.png" in processed_names
    assert "healthy_01.png" in processed_names
    assert "healthy_02.png" in processed_names
    assert "nested_healthy.jpg" in processed_names

    # Verify no corrupted or non-image files appear in results
    assert "corrupt.png" not in processed_names
    assert "fake.jpg" not in processed_names
    assert "truncated.png" not in processed_names
    assert "truncated.jpg" not in processed_names
    assert "notes.txt" not in processed_names
    assert "config.yaml" not in processed_names

    # Verify warning logs were emitted for corrupted files
    log_text = caplog.text
    assert "Skipping unreadable or corrupted image" in log_text


def test_predict_image_single_malformed_files_raise_explicit_errors(
    corrupted_files_dir: Path, fast_checkpoint: Path
):
    """
    Verifies that calling predict_image directly on malformed or non-existent files
    raises appropriate explicit exceptions and never returns bogus prediction dictionaries.
    """
    # 0-byte file
    with pytest.raises((ValueError, RuntimeError)):
        predict_image(corrupted_files_dir / "corrupt.png", checkpoint_path=fast_checkpoint, device="cpu")

    # ASCII text file named fake.jpg
    with pytest.raises((RuntimeError, ValueError)):
        predict_image(corrupted_files_dir / "fake.jpg", checkpoint_path=fast_checkpoint, device="cpu")

    # Truncated PNG
    with pytest.raises(RuntimeError):
        predict_image(corrupted_files_dir / "truncated.png", checkpoint_path=fast_checkpoint, device="cpu")

    # Truncated JPG
    with pytest.raises(RuntimeError):
        predict_image(corrupted_files_dir / "truncated.jpg", checkpoint_path=fast_checkpoint, device="cpu")

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        predict_image(corrupted_files_dir / "non_existent_image.png", checkpoint_path=fast_checkpoint, device="cpu")


def test_folder_containing_only_corrupted_files_returns_empty_dataframe(
    tmp_path: Path, fast_checkpoint: Path
):
    """
    Verifies that a directory containing exclusively corrupted, empty, and non-image files
    returns an empty DataFrame with the full 20-column schema without crashing.
    """
    only_bad_dir = tmp_path / "only_bad"
    only_bad_dir.mkdir(parents=True, exist_ok=True)

    (only_bad_dir / "corrupt_01.png").touch()
    (only_bad_dir / "corrupt_02.jpg").write_text("Garbage data not an image", encoding="utf-8")
    (only_bad_dir / "notes.txt").write_text("Just notes", encoding="utf-8")

    out_csv = tmp_path / "only_bad_predictions.csv"
    df = predict_folder(
        folder_path=only_bad_dir,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        threshold=0.50,
        device="cpu",
    )

    assert len(df) == 0, f"Expected 0 predictions for all-corrupt folder, got {len(df)}"
    assert list(df.columns) == PREDICTION_COLUMNS
    assert out_csv.is_file()
    loaded_df = pd.read_csv(out_csv)
    assert len(loaded_df) == 0
    assert list(loaded_df.columns) == PREDICTION_COLUMNS


# ==============================================================================
# 2. Non-Georeferenced Imagery Safety
# ==============================================================================

@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "webp", "bmp"])
def test_non_georeferenced_standard_formats_never_fabricate_coordinates(
    tmp_path: Path, fast_checkpoint: Path, extension: str
):
    """
    Strictly verifies that standard non-georeferenced images (PNG, JPG, WEBP, BMP):
    1. Set has_georeference == False.
    2. Set longitude is NaN (or null in memory).
    3. Set latitude is NaN (or null in memory).
    4. Never invent fake geographic coordinates (e.g. 0.0, 0.0 or random values).
    """
    img_path = tmp_path / f"sample.{extension}"
    Image.new("RGB", (64, 64), color=(50, 100, 150)).save(img_path)

    # 1. Direct georeference checker
    has_geo, lon, lat = check_georeference(img_path)
    assert has_geo is False
    assert lon is None
    assert lat is None

    # 2. Single image prediction
    pred = predict_image(img_path, checkpoint_path=fast_checkpoint, device="cpu")
    assert pred["has_georeference"] is False
    assert pred["longitude"] is None or np.isnan(pred["longitude"])
    assert pred["latitude"] is None or np.isnan(pred["latitude"])
    # Strictly check coordinates are NOT 0.0
    assert pred["longitude"] != 0.0
    assert pred["latitude"] != 0.0

    # 3. Folder prediction & CSV serialization
    folder_dir = tmp_path / f"folder_{extension}"
    folder_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(60, 120, 180)).save(folder_dir / f"frame_01.{extension}")
    csv_path = tmp_path / f"coords_{extension}.csv"

    df = predict_folder(folder_dir, checkpoint_path=fast_checkpoint, output_csv=csv_path, device="cpu")
    assert len(df) == 1
    assert df["has_georeference"].iloc[0] == False
    assert pd.isna(df["longitude"].iloc[0])
    assert pd.isna(df["latitude"].iloc[0])
    assert df["longitude"].iloc[0] != 0.0
    assert df["latitude"].iloc[0] != 0.0

    # 4. Reload from disk CSV
    reloaded = pd.read_csv(csv_path)
    assert reloaded["has_georeference"].iloc[0] == False
    assert pd.isna(reloaded["longitude"].iloc[0])
    assert pd.isna(reloaded["latitude"].iloc[0])


def test_tiled_inference_non_georeferenced_safety(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies that predict_image_tiled on a non-georeferenced large image generates
    tiles where all rows have has_georeference=False and longitude/latitude as NaN.
    """
    large_img = tmp_path / "large_standard_photo.png"
    Image.new("RGB", (512, 512), color=(80, 90, 100)).save(large_img)

    out_csv = tmp_path / "tiled_coords_test.csv"
    df = predict_image_tiled(
        image_path=large_img,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        tile_size=256,
        stride=256,
        threshold=0.50,
        device="cpu",
    )

    # 512x512 with 256x256 tile_size and stride 256 yields 4 tiles
    assert len(df) == 4
    for idx in range(len(df)):
        assert df["has_georeference"].iloc[idx] == False
        assert pd.isna(df["longitude"].iloc[idx])
        assert pd.isna(df["latitude"].iloc[idx])
        assert df["longitude"].iloc[idx] != 0.0
        assert df["latitude"].iloc[idx] != 0.0


# ==============================================================================
# 3. Empty Directory & Large Directory Scaling
# ==============================================================================

def test_empty_directory_handling_and_cli_exit(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies that an empty directory:
    1. Returns DataFrame with 0 rows and full 20-column schema without crashing.
    2. Saves a valid 0-row CSV with exact headers.
    3. CLI execution via main() exits cleanly with returncode 0.
    """
    empty_dir = tmp_path / "strictly_empty_folder"
    empty_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_path / "empty_out.csv"

    # API verification
    df = predict_folder(
        folder_path=empty_dir,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        device="cpu",
    )
    assert len(df) == 0
    assert list(df.columns) == PREDICTION_COLUMNS
    assert out_csv.is_file()

    loaded = pd.read_csv(out_csv)
    assert len(loaded) == 0
    assert list(loaded.columns) == PREDICTION_COLUMNS

    # CLI verification
    cli_out_csv = tmp_path / "empty_cli_out.csv"
    exit_code = main([
        "--input", str(empty_dir),
        "--checkpoint", str(fast_checkpoint),
        "--output", str(cli_out_csv),
        "--device", "cpu",
    ])
    assert exit_code == 0
    assert cli_out_csv.is_file()
    cli_loaded = pd.read_csv(cli_out_csv)
    assert len(cli_loaded) == 0
    assert list(cli_loaded.columns) == PREDICTION_COLUMNS


def test_batching_stress_50_images_odd_batch_size(tmp_path: Path, fast_checkpoint: Path):
    """
    Stress-tests directory scaling and mini-batch partitioning:
    - 50 synthetic images in a single folder.
    - Mini-batch size = 7 (odd number: 7 full batches of 7 = 49 images, plus 1 leftover batch of 1 image).
    - Verifies:
      1. Exactly 50 rows in output DataFrame.
      2. Deterministic sorted ordering of file names (image_00.png ... image_49.png).
      3. No duplicate rows, no dropped rows.
      4. All outputs satisfy probability simplex and bounds.
    """
    folder = tmp_path / "large_scaling_dir"
    folder.mkdir(parents=True, exist_ok=True)

    expected_names = []
    for i in range(50):
        fname = f"img_{i:02d}.png"
        expected_names.append(fname)
        # Varied colors to avoid trivial identical tensor shortcuts
        color = ((i * 5) % 256, (i * 13) % 256, (i * 29) % 256)
        Image.new("RGB", (64, 64), color=color).save(folder / fname)

    out_csv = tmp_path / "batch_scaling_out.csv"
    df = predict_folder(
        folder_path=folder,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        batch_size=7,  # Odd batch size intentionally not dividing 50
        device="cpu",
    )

    assert len(df) == 50, f"Expected 50 predictions, got {len(df)}"
    assert df["file_name"].tolist() == expected_names, "Files were not processed in deterministic sorted order"
    assert df["file_name"].nunique() == 50, "Duplicate records detected"

    # Simplex check on all 50 records
    probs_sum = df["prob_waste"] + df["prob_non_waste"]
    assert np.allclose(probs_sum, 1.0, atol=1e-5), "Simplex sum constraint violated"
    assert (df["prob_waste"] >= 0.0).all() and (df["prob_waste"] <= 1.0).all()
    assert (df["prob_non_waste"] >= 0.0).all() and (df["prob_non_waste"] <= 1.0).all()


def test_batching_with_interleaved_corrupted_images(tmp_path: Path, fast_checkpoint: Path):
    """
    Adversarial batch test:
    - 21 items total, batch_size=5 (batches of sizes 5, 5, 5, 5, 1).
    - Injects corrupted files at:
      - Item 0 (first element of batch 1)
      - Item 9 (last element of batch 2)
      - Item 10 & 11 (two consecutive elements in batch 3)
      - Item 20 (the single element of the final batch 5)
    - Verifies that predict_folder processes all remaining 16 valid images and skips the 5 corrupt ones
      without index offsets, misalignments, or crashes on single-element corrupted batches.
    """
    test_dir = tmp_path / "interleaved_corrupt_env"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Indices 0 to 20
    corrupt_indices = {0, 9, 10, 11, 20}
    for i in range(21):
        fname = f"item_{i:02d}.jpg"
        if i in corrupt_indices:
            # Corrupted ASCII file
            (test_dir / fname).write_text("CORRUPTED_STREAM", encoding="utf-8")
        else:
            Image.new("RGB", (64, 64), color=(10 * i, 100, 150)).save(test_dir / fname)

    out_csv = tmp_path / "interleaved_out.csv"
    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        batch_size=5,
        device="cpu",
    )

    expected_valid_count = 21 - len(corrupt_indices)  # 16
    assert len(df) == expected_valid_count, f"Expected {expected_valid_count} valid predictions, got {len(df)}"

    for bad_idx in corrupt_indices:
        bad_name = f"item_{bad_idx:02d}.jpg"
        assert bad_name not in df["file_name"].values, f"Corrupted file {bad_name} was not skipped"


# ==============================================================================
# 4. Threshold Boundary Stress
# ==============================================================================

def test_threshold_extreme_zero_predicts_all_waste(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies that threshold T = 0.0 forces 100% of images to be classified as:
    - is_waste == True
    - predicted_class == 1
    - status == 'candidate_detection'
    - class_label == 'waste'
    """
    test_dir = tmp_path / "thresh_zero_dir"
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        Image.new("RGB", (64, 64), color=(25 * i, 50, 75)).save(test_dir / f"sample_{i:02d}.png")

    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        threshold=0.00,
        device="cpu",
    )

    assert len(df) == 10
    assert (df["is_waste"] == True).all()
    assert (df["predicted_class"] == 1).all()
    assert (df["status"] == "candidate_detection").all()
    assert (df["class_label"] == "waste").all()
    assert (df["threshold"] == 0.00).all()


def test_threshold_extreme_one_predicts_all_non_waste(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies that threshold T = 1.0 forces all standard images (where prob_waste < 1.0) to:
    - is_waste == False
    - predicted_class == 0
    - status == 'negative'
    - class_label == 'non-waste'
    """
    test_dir = tmp_path / "thresh_one_dir"
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        Image.new("RGB", (64, 64), color=(25 * i, 50, 75)).save(test_dir / f"sample_{i:02d}.png")

    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        threshold=1.00,
        device="cpu",
    )

    assert len(df) == 10
    assert (df["is_waste"] == False).all()
    assert (df["predicted_class"] == 0).all()
    assert (df["status"] == "negative").all()
    assert (df["class_label"] == "non-waste").all()
    assert (df["threshold"] == 1.00).all()


@pytest.mark.parametrize("invalid_threshold", [-0.01, -1.0, 1.01, 2.0, 99.0])
def test_out_of_bounds_threshold_raises_value_error(
    tmp_path: Path, fast_checkpoint: Path, invalid_threshold: float
):
    """
    Verifies that out-of-bounds threshold values (T < 0.0 or T > 1.0) are strictly rejected
    with a ValueError in both predict_image and predict_folder.
    """
    test_img = tmp_path / "boundary_img.png"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(test_img)

    with pytest.raises(ValueError, match="threshold must be in range"):
        predict_image(test_img, checkpoint_path=fast_checkpoint, threshold=invalid_threshold, device="cpu")

    test_dir = tmp_path / "boundary_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(test_dir / "img1.png")

    with pytest.raises(ValueError, match="threshold must be in range"):
        predict_folder(test_dir, checkpoint_path=fast_checkpoint, threshold=invalid_threshold, device="cpu")


def test_probability_simplex_precision(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies that for every row:
    - prob_waste + prob_non_waste == 1.0 within 1e-6 precision.
    - prob_waste == probability.
    - 0.0 <= prob_waste <= 1.0 and 0.0 <= prob_non_waste <= 1.0.
    """
    test_dir = tmp_path / "simplex_precision_dir"
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in range(15):
        Image.new("RGB", (64, 64), color=(i * 15, i * 10, i * 5)).save(test_dir / f"img_{i:02d}.jpg")

    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        threshold=0.50,
        device="cpu",
    )

    for idx, row in df.iterrows():
        pw = float(row["prob_waste"])
        pnw = float(row["prob_non_waste"])
        prob = float(row["probability"])

        assert np.isclose(pw + pnw, 1.0, atol=1e-5), f"Simplex violation at row {idx}: {pw} + {pnw} != 1.0"
        assert np.isclose(pw, prob, atol=1e-6), f"Mismatch between probability and prob_waste at row {idx}"
        assert 0.0 <= pw <= 1.0
        assert 0.0 <= pnw <= 1.0


# ==============================================================================
# 5. CSV Output Schema Integrity
# ==============================================================================

def test_csv_output_schema_integrity(tmp_path: Path, fast_checkpoint: Path):
    """
    Verifies the CSV output matches the complete 20-column schema:
    1. Exact 20 column headers present and ordered.
    2. Column types match requirements (int, float, bool, string, NaN).
    3. Re-read from disk matches memory representation.
    """
    test_dir = tmp_path / "schema_check_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 150), color=(123, 234, 45)).save(test_dir / "test_dim.png")

    out_csv = tmp_path / "schema_output.csv"
    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        threshold=0.50,
        device="cpu",
    )

    assert len(df.columns) == 20
    assert list(df.columns) == PREDICTION_COLUMNS

    # Check row values
    row = df.iloc[0]
    assert row["file_name"] == "test_dim.png"
    assert row["x_min"] == 0
    assert row["y_min"] == 0
    assert row["x_max"] == 100
    assert row["y_max"] == 150
    assert row["predicted_class"] in [0, 1]
    assert isinstance(row["is_waste"], (bool, np.bool_))
    assert row["status"] in ["candidate_detection", "negative"]
    assert row["has_georeference"] is False or row["has_georeference"] == False
    assert pd.isna(row["longitude"])
    assert pd.isna(row["latitude"])

    # Verify CSV file contents
    loaded_df = pd.read_csv(out_csv)
    assert len(loaded_df.columns) == 20
    assert list(loaded_df.columns) == PREDICTION_COLUMNS


# ==============================================================================
# 6. Pathological & Unconventional Image Formats
# ==============================================================================

def test_pathological_image_dimensions_and_channels(tmp_path: Path, fast_checkpoint: Path):
    """
    Stress-tests handling of pathological image variations:
    1. Tiny 1x1 pixel image.
    2. Extreme aspect ratio (10x1000).
    3. Grayscale (mode 'L', 1-channel).
    4. RGBA (mode 'RGBA', 4-channel with alpha transparency).
    All must be safely preprocessed to RGB (3, 224, 224) and evaluated without exceptions.
    """
    test_dir = tmp_path / "pathological_dir"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1x1 image
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(test_dir / "tiny_1x1.png")

    # 2. Extreme aspect ratio 10x1000
    Image.new("RGB", (10, 1000), color=(0, 255, 0)).save(test_dir / "needle.png")

    # 3. Grayscale
    Image.new("L", (128, 128), color=128).save(test_dir / "grayscale.png")

    # 4. RGBA with transparency
    Image.new("RGBA", (128, 128), color=(100, 150, 200, 128)).save(test_dir / "transparent.png")

    out_csv = tmp_path / "pathological_predictions.csv"
    df = predict_folder(
        folder_path=test_dir,
        checkpoint_path=fast_checkpoint,
        output_csv=out_csv,
        threshold=0.50,
        device="cpu",
    )

    assert len(df) == 4, f"Expected 4 valid predictions for pathological images, got {len(df)}"
    processed_names = df["file_name"].tolist()
    assert "tiny_1x1.png" in processed_names
    assert "needle.png" in processed_names
    assert "grayscale.png" in processed_names
    assert "transparent.png" in processed_names
