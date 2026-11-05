"""
Adversarial stress-testing suite for waste_hotspot.preprocessing.
Tests extreme image dimensions, unusual color formats, 1,000-pass determinism,
training stochasticity variance, and DataLoader collation under edge-case metadata.
"""

from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader

from waste_hotspot.preprocessing import (
    AerialWasteDataset,
    get_eval_transforms,
    get_train_transforms,
    get_transforms,
    load_and_preprocess_image,
    safe_load_image,
)


@pytest.fixture
def stress_temp_dir() -> Generator[Path, None, None]:
    """Provides an isolated temporary directory for stress artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def texture_image() -> Image.Image:
    """Creates a high-entropy 256x256 RGB image with spatial gradient and noise."""
    np.random.seed(42)
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.int32)
    x = np.arange(256).reshape(1, 256, 1)
    y = np.arange(256).reshape(256, 1, 1)
    combined = (noise + x + y) % 256
    return Image.fromarray(combined.astype(np.uint8))


# ==============================================================================
# 1. Extreme Image Dimensions Stress Tests
# ==============================================================================

@pytest.mark.parametrize("width,height", [
    (1, 1),
    (10000, 1),
    (1, 10000),
    (4096, 4096),
])
def test_extreme_image_dimensions(stress_temp_dir: Path, width: int, height: int) -> None:
    """
    Stress-tests preprocessing against pathological dimensions:
    - 1x1: Minimal single-pixel edge case
    - 10000x1: Extreme horizontal sliver
    - 1x10000: Extreme vertical sliver
    - 4096x4096: Large high-resolution aerial tile
    """
    img_path = stress_temp_dir / f"extreme_{width}x{height}.png"
    
    # Generate image with non-trivial values
    if width * height <= 10000:
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        # Large image: use solid color without allocating full 4096x4096x3 in memory at once
        img = Image.new("RGB", (width, height), (135, 150, 165))
        
    img.save(img_path)

    # 1. Test safe_load_image
    loaded_img = safe_load_image(img_path)
    assert loaded_img.size == (width, height)
    assert loaded_img.mode == "RGB"

    # 2. Test eval transforms (Resize -> ToTensor -> Normalize)
    eval_transform = get_eval_transforms(image_size=224)
    tensor_eval = eval_transform(loaded_img)
    assert isinstance(tensor_eval, torch.Tensor)
    assert tensor_eval.shape == (3, 224, 224)
    assert tensor_eval.dtype == torch.float32
    assert not torch.isnan(tensor_eval).any(), "NaN found in eval tensor output"
    assert not torch.isinf(tensor_eval).any(), "Inf found in eval tensor output"

    # 3. Test train transforms
    train_transform = get_train_transforms(image_size=224)
    tensor_train = train_transform(loaded_img)
    assert isinstance(tensor_train, torch.Tensor)
    assert tensor_train.shape == (3, 224, 224)
    assert tensor_train.dtype == torch.float32
    assert not torch.isnan(tensor_train).any(), "NaN found in train tensor output"
    assert not torch.isinf(tensor_train).any(), "Inf found in train tensor output"


# ==============================================================================
# 2. Unusual Color Formats Stress Tests
# ==============================================================================

def test_color_format_1bit_monochrome(stress_temp_dir: Path) -> None:
    """Tests 1-bit binary monochrome image (PIL mode '1')."""
    path = stress_temp_dir / "mono_1bit.png"
    arr = (np.random.rand(128, 128) > 0.5).astype(bool)
    Image.fromarray(arr).save(path)

    tensor = load_and_preprocess_image(path, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()


def test_color_format_16bit_tiff_single_and_multiband(stress_temp_dir: Path) -> None:
    """Tests 16-bit TIFF imagery (mode 'I;16' and 3-band uint16 TIFF)."""
    # 1. Single channel uint16 TIFF (common in elevation/SAR imagery)
    path_single = stress_temp_dir / "single_16bit.tif"
    arr_single = (np.random.rand(100, 100) * 65535).astype(np.uint16)
    Image.fromarray(arr_single).save(path_single)

    tensor_single = load_and_preprocess_image(path_single, image_size=224)
    assert tensor_single.shape == (3, 224, 224)
    assert tensor_single.dtype == torch.float32
    assert not torch.isnan(tensor_single).any()

    # 2. 3-band multi-channel uint16 TIFF (common in high-bit depth satellite sensors)
    path_multi = stress_temp_dir / "multi_16bit.tif"
    arr_multi = (np.random.rand(100, 100, 3) * 65535).astype(np.uint16)
    tifffile.imwrite(path_multi, arr_multi)

    tensor_multi = load_and_preprocess_image(path_multi, image_size=224)
    assert tensor_multi.shape == (3, 224, 224)
    assert tensor_multi.dtype == torch.float32
    assert not torch.isnan(tensor_multi).any()


def test_color_format_cmyk(stress_temp_dir: Path) -> None:
    """Tests 4-channel CMYK print-ready imagery (PIL mode 'CMYK')."""
    path = stress_temp_dir / "cmyk.jpg"
    img = Image.new("CMYK", (150, 150), (20, 80, 100, 10))
    img.save(path)

    tensor = load_and_preprocess_image(path, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()


def test_color_format_palette_p_mode(stress_temp_dir: Path) -> None:
    """Tests 8-bit color mapped Palette image (PIL mode 'P')."""
    path = stress_temp_dir / "palette.png"
    img = Image.new("P", (120, 120), 0)
    # Define custom palette
    palette = [255, 0, 0, 0, 255, 0, 0, 0, 255] + [128, 128, 128] * 253
    img.putpalette(palette)
    img.save(path)

    tensor = load_and_preprocess_image(path, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()


def test_color_format_rgba_with_alpha_masking(stress_temp_dir: Path) -> None:
    """Tests RGBA imagery with varying alpha channels (transparent margins, semi-transparency)."""
    path = stress_temp_dir / "alpha_masked.png"
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[:30, :, :] = [255, 0, 0, 255]       # Opaque red
    arr[30:70, :, :] = [0, 255, 0, 128]     # Semi-transparent green
    arr[70:, :, :] = [0, 0, 255, 0]         # Fully transparent blue margin
    Image.fromarray(arr).save(path)

    tensor = load_and_preprocess_image(path, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()


# ==============================================================================
# 3. Transform Determinism Stress Test (1,000 passes)
# ==============================================================================

def test_transform_determinism_1000_passes(texture_image: Image.Image) -> None:
    """
    Stress-tests that 1,000 passes of get_transforms(is_train=False) on the same
    image yield bit-for-bit identical PyTorch tensors across all 1,000 passes.
    """
    eval_transform = get_transforms(image_size=224, is_train=False)

    baseline_tensor = eval_transform(texture_image)
    assert isinstance(baseline_tensor, torch.Tensor)

    # Execute 1,000 passes
    for i in range(1000):
        current_tensor = eval_transform(texture_image)
        if not torch.equal(baseline_tensor, current_tensor):
            diff = torch.abs(baseline_tensor - current_tensor).max().item()
            pytest.fail(f"Pass {i + 1}/1000 violated bit-for-bit determinism! Max diff: {diff}")


# ==============================================================================
# 4. Transform Stochasticity Variance Stress Test
# ==============================================================================

def test_transform_stochasticity_variance(texture_image: Image.Image) -> None:
    """
    Verifies that get_transforms(is_train=True) generates substantial stochastic
    variance over 100 repeated runs on the same input image.
    """
    train_transform = get_transforms(image_size=224, is_train=True)

    passes = 100
    tensors = [train_transform(texture_image) for _ in range(passes)]

    # Stack along batch dimension: shape (100, 3, 224, 224)
    stacked = torch.stack(tensors, dim=0)

    # 1. Compute pixel-wise variance across passes
    pixel_variance = torch.var(stacked, dim=0)
    mean_variance = torch.mean(pixel_variance).item()

    # 2. Check that mean variance is significantly positive (> 0.10)
    assert mean_variance > 0.10, (
        f"Mean pixel-wise variance {mean_variance:.4f} is too low; "
        "transforms lack sufficient augmentation entropy."
    )

    # 3. Verify distinct tensor outputs
    unique_hashes = set(t.numpy().tobytes() for t in tensors)
    assert len(unique_hashes) == passes, (
        f"Expected {passes} distinct tensor outputs, but observed only {len(unique_hashes)}"
    )


# ==============================================================================
# 5. DataLoader Batching with Edge-Case Metadata Stress Test
# ==============================================================================

def test_dataloader_batch_64_with_workers_and_edge_case_metadata(stress_temp_dir: Path) -> None:
    """
    Stress-tests DataLoader with batch_size=64 and num_workers=0 and num_workers=2.
    Asserts zero TypeError: default_collate when metadata contains:
    - None / np.nan values
    - Mixed numeric and string identifiers
    - Float / missing widths and heights
    - Missing labels
    - Unrelated non-collatable columns in source DataFrame
    """
    img_dir = stress_temp_dir / "batch_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    records = []

    for i in range(batch_size):
        w = 80 + (i % 40)
        h = 70 + (i % 50)
        img_file = img_dir / f"batch_img_{i}.png"
        
        # Save varied synthetic image
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_file)

        # Inject adversarial metadata edge cases
        record = {
            # Mixed types: int, string, None, NaN
            "image_id": i if i % 4 == 0 else (f"id_{i}" if i % 4 == 1 else (None if i % 4 == 2 else np.nan)),
            # Missing or None file names
            "file_name": img_file.name if i % 2 == 0 else None,
            "file_path": str(img_file),
            # Missing labels, None, NaN, integer
            "label": (i % 2) if i % 5 != 0 else (None if i % 5 == 1 else np.nan),
            # None or NaN image sources
            "img_source": "GE" if i % 3 == 0 else (None if i % 3 == 1 else np.nan),
            # Floats, None, and NaNs for width/height
            "width": float(w) if i % 3 == 0 else (None if i % 3 == 1 else np.nan),
            "height": int(h) if i % 2 == 0 else np.nan,
            # Extra arbitrary un-collatable DataFrame columns
            "uncollatable_none": None,
            "uncollatable_dict": {"idx": i, "data": [1, 2, 3]},
            "uncollatable_obj": object(),
        }
        records.append(record)

    df = pd.DataFrame(records)
    dataset = AerialWasteDataset(df, image_size=224, is_train=False)
    assert len(dataset) == batch_size

    # Test both single-process and multi-worker modes
    for num_workers in [0, 2]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        batches = list(loader)
        assert len(batches) == 1, f"Expected 1 batch of size {batch_size}, got {len(batches)}"

        imgs, labels, metadata = batches[0]

        # Verify image batch
        assert isinstance(imgs, torch.Tensor)
        assert imgs.shape == (batch_size, 3, 224, 224)
        assert imgs.dtype == torch.float32

        # Verify label batch
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (batch_size,)
        assert labels.dtype == torch.int64

        # Verify sanitized metadata batch
        assert isinstance(metadata, dict)
        expected_keys = {"image_id", "file_name", "file_path", "img_source", "width", "height"}
        assert set(metadata.keys()) == expected_keys

        # Check metadata lengths and collation integrity
        assert len(metadata["image_id"]) == batch_size
        assert len(metadata["file_name"]) == batch_size
        assert len(metadata["file_path"]) == batch_size
        assert len(metadata["img_source"]) == batch_size
        assert isinstance(metadata["width"], torch.Tensor)
        assert metadata["width"].shape == (batch_size,)
        assert isinstance(metadata["height"], torch.Tensor)
        assert metadata["height"].shape == (batch_size,)
