"""
Preprocessing, image transformations, and PyTorch Dataset class for AerialWaste imagery.
"""

from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Ensure src/ is on sys.path if invoked directly
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

# Catch truncated or corrupted images cleanly
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Standard ImageNet normalization statistics
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def safe_load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Safely opens an image file and converts it to 3-channel RGB.

    Handles:
        - Grayscale ('L'): Replicated to 3 identical RGB channels.
        - RGBA ('RGBA'): Strips alpha and converts to RGB.
        - Palette ('P') and CMYK: Converts to standard RGB.
        - Empty files (0 bytes): Raises ValueError.
        - Missing files: Raises FileNotFoundError.
        - Corrupt files: Raises RuntimeError.

    Args:
        image_path: Filesystem path to image.

    Returns:
        PIL.Image.Image in 'RGB' mode.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {path}")

    try:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            # Force decoding to catch truncated or corrupted payload
            rgb_img.load()
            return rgb_img
    except Exception as e:
        raise RuntimeError(f"Failed to decode image from {path}: {e}") from e


def get_transforms(
    image_size: int = 224,
    is_train: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    rotation_degrees: int = 15,
    brightness_jitter: float = 0.1,
    contrast_jitter: float = 0.1,
) -> transforms.Compose:
    """
    Constructs torchvision transform pipeline for aerial imagery.

    Args:
        image_size: Target square dimensions (width and height). Default 224.
        is_train: If True, applies random horizontal/vertical flips, rotation,
                  and color jitter. If False, strictly deterministic resize + norm.
        mean: Normalization means (3-channel). Defaults to ImageNet values.
        std: Normalization standard deviations (3-channel). Defaults to ImageNet values.
        rotation_degrees: Maximum rotation range for training (+/- degrees).
        brightness_jitter: Brightness jitter factor for training.
        contrast_jitter: Contrast jitter factor for training.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    norm_mean = list(mean) if mean is not None else IMAGENET_MEAN
    norm_std = list(std) if std is not None else IMAGENET_STD

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])


def get_train_transforms(image_size: int = 224, **kwargs: Any) -> transforms.Compose:
    """Convenience alias for training transform pipeline."""
    return get_transforms(image_size=image_size, is_train=True, **kwargs)


def get_eval_transforms(image_size: int = 224, **kwargs: Any) -> transforms.Compose:
    """Convenience alias for validation, test, and inference pipeline."""
    return get_transforms(image_size=image_size, is_train=False, **kwargs)


def load_and_preprocess_image(
    image_path: Union[str, Path],
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Loads an image from disk, converts to RGB, and applies transform pipeline.

    Args:
        image_path: Filesystem path to image.
        transform: Optional torchvision transform. If None, uses eval transform.
        image_size: Square target size if default eval transform is used.

    Returns:
        torch.Tensor of shape (3, image_size, image_size) and dtype float32.
    """
    img = safe_load_image(image_path)
    if transform is None:
        transform = get_eval_transforms(image_size=image_size)
    return transform(img)


class AerialWasteDataset(Dataset):
    """
    PyTorch Dataset for AerialWaste binary image classification.

    Loads aerial imagery listed in a pandas DataFrame or CSV file, converts to RGB,
    applies torchvision transforms, and yields (image_tensor, label, metadata).
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str, Path],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        image_size: int = 224,
        is_train: bool = False,
    ) -> None:
        """
        Initializes the AerialWasteDataset.

        Args:
            df: Either a pandas DataFrame or a path to a CSV manifest file.
            transform: Optional torchvision transform. If None, automatically creates
                       pipeline via get_transforms(image_size=image_size, is_train=is_train).
            data_dir: Optional base directory to prepend to relative image paths.
            image_size: Target square image dimensions (default 224).
            is_train: Used when transform is None to configure train vs eval transforms.
        """
        if isinstance(df, (str, Path)):
            csv_path = Path(df)
            if not csv_path.is_file():
                raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
        elif isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            raise TypeError(f"Expected df to be DataFrame, str, or Path; received {type(df)}")

        self.df = self.df.reset_index(drop=True)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.image_size = image_size
        self.is_train = is_train

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series, idx: int) -> Path:
        """Resolves existing image file path from candidate dataframe columns."""
        # Check 'file_path'
        if "file_path" in row and pd.notna(row["file_path"]):
            cand = Path(str(row["file_path"]))
            if cand.is_file():
                return cand
            if self.data_dir and (self.data_dir / cand).is_file():
                return self.data_dir / cand
            if self.data_dir and "file_name" in row and pd.notna(row["file_name"]):
                alt = self.data_dir / str(row["file_name"])
                if alt.is_file():
                    return alt
            return cand

        # Check 'file_name'
        if "file_name" in row and pd.notna(row["file_name"]):
            cand = Path(str(row["file_name"]))
            if self.data_dir:
                return self.data_dir / cand
            return cand

        # Check 'image_id'
        if "image_id" in row and pd.notna(row["image_id"]):
            cand = Path(f"{row['image_id']}.png")
            if self.data_dir:
                return self.data_dir / cand
            return cand

        raise KeyError(f"Row {idx} missing 'file_path', 'file_name', and 'image_id'")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Retrieves sample at index.

        Returns:
            Tuple of:
                - image_tensor: torch.Tensor of shape (3, H, W) and dtype float32.
                - label: int (0: non-waste, 1: waste, or -1 if unlabeled).
                - metadata: dict of sanitized metadata (image_id, file_name, file_path,
                            img_source, width, height).
        """
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row, idx)

        # Load RGB image
        img = safe_load_image(img_path)

        # Apply transform
        tensor = self.transform(img)

        # Extract label
        if "label" in row and pd.notna(row["label"]):
            label = int(row["label"])
        elif "is_candidate_location" in row and pd.notna(row["is_candidate_location"]):
            label = int(row["is_candidate_location"])
        else:
            label = -1

        # Sanitize metadata to guarantee compatibility with PyTorch default_collate
        metadata: Dict[str, Any] = {
            "image_id": str(row["image_id"]) if "image_id" in row and pd.notna(row["image_id"]) else str(idx),
            "file_name": str(row["file_name"]) if "file_name" in row and pd.notna(row["file_name"]) else img_path.name,
            "file_path": str(img_path),
            "img_source": str(row["img_source"]) if "img_source" in row and pd.notna(row["img_source"]) else "unknown",
            "width": int(row["width"]) if "width" in row and pd.notna(row["width"]) else int(img.width),
            "height": int(row["height"]) if "height" in row and pd.notna(row["height"]) else int(img.height),
        }

        return tensor, label, metadata
