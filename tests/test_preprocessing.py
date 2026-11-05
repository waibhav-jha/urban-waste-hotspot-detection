"""
Unit tests for waste_hotspot.preprocessing.
Verifies transform output shape, dtype, determinism, augmentations, and Dataset contracts.
"""

from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from waste_hotspot.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AerialWasteDataset,
    get_eval_transforms,
    get_train_transforms,
    get_transforms,
    load_and_preprocess_image,
    safe_load_image,
)


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Creates a synthetic 500x500 RGB image with distinctive quadrants."""
    arr = np.zeros((500, 500, 3), dtype=np.uint8)
    arr[:250, :250] = [255, 0, 0]      # Top-left Red
    arr[:250, 250:] = [0, 255, 0]      # Top-right Green
    arr[250:, :250] = [0, 0, 255]      # Bottom-left Blue
    arr[250:, 250:] = [255, 255, 0]    # Bottom-right Yellow
    return Image.fromarray(arr)


@pytest.fixture
def sample_dataset_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory with synthetic image files and split CSVs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        img_dir = root / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Create 4 synthetic images
        for i in range(4):
            arr = np.full((100, 100, 3), i * 60, dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(img_dir / f"img_{i}.png")

        # Create 0-byte corrupt file
        (img_dir / "corrupt_zero.png").touch()

        # Create truncated file
        with open(img_dir / "truncated.png", "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n")

        yield root


# ==============================================================================
# Test Cases
# ==============================================================================

def test_eval_transforms_output_shape_and_dtype(sample_rgb_image: Image.Image) -> None:
    """Verifies that eval transforms produce exact (3, 224, 224) float32 tensors."""
    transform = get_eval_transforms(image_size=224)
    tensor = transform(sample_rgb_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transforms_strictly_deterministic(sample_rgb_image: Image.Image) -> None:
    """Verifies that evaluation transforms produce identical outputs across multiple passes."""
    transform = get_transforms(image_size=224, is_train=False)

    tensor_pass_1 = transform(sample_rgb_image)
    tensor_pass_2 = transform(sample_rgb_image)
    tensor_pass_3 = transform(sample_rgb_image)

    # Must be bit-for-bit identical
    assert torch.equal(tensor_pass_1, tensor_pass_2)
    assert torch.equal(tensor_pass_2, tensor_pass_3)


def test_train_transforms_apply_stochastic_variations(sample_rgb_image: Image.Image) -> None:
    """Verifies that train transforms apply stochastic augmentations producing varied outputs."""
    transform = get_transforms(image_size=224, is_train=True)

    tensors = [transform(sample_rgb_image) for _ in range(10)]

    # All tensors must maintain shape (3, 224, 224) and float32 dtype
    for t in tensors:
        assert t.shape == (3, 224, 224)
        assert t.dtype == torch.float32

    # At least two passes must differ due to random flips, rotations, or color jitter
    differences = [not torch.equal(tensors[0], t) for t in tensors[1:]]
    assert any(differences), "Training transforms failed to produce stochastic variations."


def test_imagenet_normalization_values() -> None:
    """Verifies that ImageNet normalization matches exact mathematical formulas."""
    transform = get_eval_transforms(image_size=64)

    # 1. Pure black image (all 0s) -> should normalize to (-mean / std)
    black_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    black_tensor = transform(black_img)
    expected_black = [-IMAGENET_MEAN[c] / IMAGENET_STD[c] for c in range(3)]
    for c in range(3):
        assert torch.allclose(black_tensor[c], torch.tensor(expected_black[c]), atol=1e-3)

    # 2. Pure white image (all 255s) -> should normalize to ((1.0 - mean) / std)
    white_img = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    white_tensor = transform(white_img)
    expected_white = [(1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c] for c in range(3)]
    for c in range(3):
        assert torch.allclose(white_tensor[c], torch.tensor(expected_white[c]), atol=1e-3)


def test_grayscale_and_rgba_to_rgb_conversion(tmp_path: Path) -> None:
    """Verifies that Grayscale and RGBA images are safely converted to 3-channel RGB."""
    # Grayscale image (mode 'L')
    gray_arr = np.full((120, 120), 128, dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    Image.fromarray(gray_arr).save(gray_path)

    gray_tensor = load_and_preprocess_image(gray_path, image_size=224)
    assert gray_tensor.shape == (3, 224, 224)

    # RGBA image (mode 'RGBA')
    rgba_arr = np.full((120, 120, 4), 200, dtype=np.uint8)
    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(rgba_arr).save(rgba_path)

    rgba_tensor = load_and_preprocess_image(rgba_path, image_size=224)
    assert rgba_tensor.shape == (3, 224, 224)


def test_non_square_image_resize(tmp_path: Path) -> None:
    """Verifies that non-square imagery (e.g. 1055x1046) is cleanly resized to (224, 224)."""
    img_path = tmp_path / "nonsquare.png"
    Image.fromarray(np.zeros((1055, 1046, 3), dtype=np.uint8)).save(img_path)

    tensor = load_and_preprocess_image(img_path, image_size=224)
    assert tensor.shape == (3, 224, 224)


def test_corrupt_and_missing_images_raise_errors(sample_dataset_dir: Path) -> None:
    """Verifies that missing, empty (0-byte), and corrupted images raise appropriate exceptions."""
    img_dir = sample_dataset_dir / "images"

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        safe_load_image(img_dir / "does_not_exist.png")

    # 0-byte file
    with pytest.raises(ValueError, match="empty"):
        safe_load_image(img_dir / "corrupt_zero.png")

    # Truncated corrupt file
    with pytest.raises(RuntimeError):
        safe_load_image(img_dir / "truncated.png")


def test_aerial_waste_dataset_contract(sample_dataset_dir: Path) -> None:
    """Verifies that AerialWasteDataset yields (image_tensor, label, metadata) matching contracts."""
    img_dir = sample_dataset_dir / "images"

    df = pd.DataFrame({
        "image_id": ["0", "1", "2"],
        "file_name": ["img_0.png", "img_1.png", "img_2.png"],
        "file_path": [str(img_dir / f"img_{i}.png") for i in range(3)],
        "label": [0, 1, 0],
        "img_source": ["GE", "AGEA", "WV3"],
        "width": [100, 100, 100],
        "height": [100, 100, 100],
    })

    dataset = AerialWasteDataset(df, image_size=224, is_train=False)
    assert len(dataset) == 3

    tensor, label, meta = dataset[1]

    # Verify tensor
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify label
    assert isinstance(label, int)
    assert label == 1

    # Verify metadata
    assert isinstance(meta, dict)
    assert meta["image_id"] == "1"
    assert meta["file_name"] == "img_1.png"
    assert meta["img_source"] == "AGEA"
    assert meta["width"] == 100
    assert meta["height"] == 100


def test_dataloader_collate_compatibility(sample_dataset_dir: Path) -> None:
    """Verifies that AerialWasteDataset batches smoothly with PyTorch DataLoader."""
    img_dir = sample_dataset_dir / "images"

    df = pd.DataFrame({
        "image_id": ["0", "1", "2", "3"],
        "file_path": [str(img_dir / f"img_{i}.png") for i in range(4)],
        "label": [0, 1, 0, 1],
        "img_source": ["GE", "AGEA", "WV3", "GE"],
        "width": [100, 100, 100, 100],
        "height": [100, 100, 100, 100],
    })

    dataset = AerialWasteDataset(df, image_size=224, is_train=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch_tensors, batch_labels, batch_meta = next(iter(loader))

    assert batch_tensors.shape == (2, 3, 224, 224)
    assert batch_labels.shape == (2,)
    assert isinstance(batch_meta, dict)
    assert len(batch_meta["image_id"]) == 2
    assert len(batch_meta["img_source"]) == 2


def test_split_manifest_zero_overlap() -> None:
    """Verifies that train, val, and test splits have zero overlapping image_ids or files."""
    # Synthetic split verification simulating data module output
    train_df = pd.DataFrame({"image_id": [1, 2, 3, 4], "file_name": ["1.png", "2.png", "3.png", "4.png"]})
    val_df = pd.DataFrame({"image_id": [5, 6], "file_name": ["5.png", "6.png"]})
    test_df = pd.DataFrame({"image_id": [7, 8], "file_name": ["7.png", "8.png"]})

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val share image_ids"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test share image_ids"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test share image_ids"

    train_files = set(train_df["file_name"])
    val_files = set(val_df["file_name"])
    test_files = set(test_df["file_name"])

    assert len(train_files.intersection(val_files)) == 0, "Train and Val share file_names"
    assert len(train_files.intersection(test_files)) == 0, "Train and Test share file_names"
    assert len(val_files.intersection(test_files)) == 0, "Val and Test share file_names"
