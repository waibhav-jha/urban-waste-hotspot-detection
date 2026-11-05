"""
Inference module for satellite and aerial urban waste hotspot detection.

Provides single-image and batch folder inference, decision thresholding,
probability simplex generation, non-georeferenced coordinate safety,
and structured CSV prediction exports.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

# Ensure src/ is on sys.path
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from waste_hotspot.model import (
    _LightweightDummyClassifier,
    build_classifier,
    load_checkpoint,
)
from waste_hotspot.preprocessing import (
    get_eval_transforms,
    load_and_preprocess_image,
    safe_load_image,
)

logger = logging.getLogger("waste_hotspot.inference")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Supported image file extensions (case-insensitive)
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
IMAGE_EXTENSIONS = VALID_IMAGE_EXTENSIONS

# Standard output schema matching both dispatch and E2E test contracts
PREDICTION_COLUMNS = [
    "image_id",
    "file_name",
    "file_path",
    "source_image",
    "is_tile",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "predicted_class",
    "class_label",
    "probability",
    "prob_waste",
    "prob_non_waste",
    "is_waste",
    "status",
    "threshold",
    "has_georeference",
    "longitude",
    "latitude",
]

STANDARD_COLUMNS = PREDICTION_COLUMNS


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolves execution device with safe GPU/CPU fallback.

    Args:
        device: Device request string ('cpu', 'cuda', 'cuda:0') or torch.device.

    Returns:
        torch.device instance.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device

    dev_str = str(device).strip().lower()
    if dev_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(dev_str)
        logger.warning("CUDA requested ('%s') but not available. Falling back to CPU.", device)
        return torch.device("cpu")
    return torch.device("cpu")


def check_georeference(image_path: Union[str, Path]) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Inspects an image for valid geospatial metadata (CRS and affine transform).

    Strict Contract:
        Non-georeferenced images return (False, None, None).
        Never fabricates arbitrary geographic coordinates.

    Args:
        image_path: Filesystem path to image.

    Returns:
        Tuple of (has_georeference, longitude, latitude).
    """
    path = Path(image_path)
    if not path.is_file():
        return False, None, None

    # Fast rejection for non-TIFF formats
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return False, None, None

    try:
        import rasterio
        from rasterio.warp import transform as warp_transform

        with rasterio.open(path) as src:
            if src.crs is not None and src.transform is not None and not src.transform.is_identity:
                cx, cy = src.xy(src.height // 2, src.width // 2)
                if src.crs.is_geographic:
                    return True, float(cx), float(cy)
                elif src.crs.is_projected:
                    xs, ys = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
                    return True, float(xs[0]), float(ys[0])
                else:
                    return True, float(cx), float(cy)
    except Exception as exc:
        logger.debug("Failed rasterio georeference check on '%s': %s", path, exc)

    return False, None, None


check_raster_georeference = check_georeference


def load_model_for_inference(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, float, Dict[str, Any]]:
    """
    Loads model weights and default decision threshold from a checkpoint.

    Handles:
        1. Fully populated checkpoints from train.py (with model_architecture).
        2. Synthetic unit test checkpoints (e.g., LightweightDummyClassifier).
        3. Raw state dict dictionaries.
        4. User-provided pre-instantiated model instances.

    Args:
        checkpoint_path: Path to .pth checkpoint. Defaults to 'artifacts/best_model.pth'.
        device: Target execution device.
        model: Optional pre-instantiated nn.Module.

    Returns:
        Tuple of (model, default_threshold, checkpoint_dict).
    """
    target_device = resolve_device(device)

    # If caller provided model instance directly
    if model is not None:
        default_threshold = 0.50
        checkpoint_dict: Dict[str, Any] = {}
        if checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.is_file():
                try:
                    ckpt = torch.load(ckpt_p, map_location=target_device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(ckpt_p, map_location=target_device)
                if isinstance(ckpt, dict):
                    checkpoint_dict = ckpt
                    default_threshold = float(ckpt.get("calibrated_threshold", ckpt.get("threshold", 0.50)))
                    sd = ckpt.get("model_state_dict", ckpt)
                    model.load_state_dict(sd, strict=False)
        model.to(target_device)
        model.eval()
        return model, default_threshold, checkpoint_dict

    ckpt_file: Optional[Path] = None
    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
    else:
        # Check standard default locations
        candidates = [
            Path("artifacts/best_model.pth"),
            Path("artifacts/model.pth"),
            Path("artifacts/checkpoint.pth"),
        ]
        ckpt_file = next((c for c in candidates if c.is_file()), candidates[0])

    if not ckpt_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_file}. Train a model first via 'python -m waste_hotspot.train'."
        )

    try:
        checkpoint = torch.load(ckpt_file, map_location=target_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_file, map_location=target_device)

    # Extract default decision threshold from checkpoint
    default_threshold = 0.50
    if isinstance(checkpoint, dict):
        if "calibrated_threshold" in checkpoint and checkpoint["calibrated_threshold"] is not None:
            default_threshold = float(checkpoint["calibrated_threshold"])
        elif "threshold" in checkpoint and checkpoint["threshold"] is not None:
            default_threshold = float(checkpoint["threshold"])

    # Extract state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Check for unit test synthetic dummy classifier
    if isinstance(state_dict, dict) and "conv.weight" in state_dict and "fc.weight" in state_dict and "layer1.0.conv1.weight" not in state_dict:
        hidden_dim = state_dict["conv.weight"].shape[0]
        num_classes = state_dict["fc.weight"].shape[0]
        net = _LightweightDummyClassifier(hidden_dim=hidden_dim, num_classes=num_classes)
        net.load_state_dict(state_dict)
        net.to(target_device)
        net.eval()
        return net, default_threshold, checkpoint if isinstance(checkpoint, dict) else {}

    # Standard model loading via model.py
    arch = checkpoint.get("model_architecture", checkpoint.get("model_name", "resnet50")) if isinstance(checkpoint, dict) else "resnet50"
    num_classes = checkpoint.get("num_classes", 2) if isinstance(checkpoint, dict) else 2

    net = build_classifier(model_name=arch, pretrained=False, num_classes=num_classes)
    net.load_state_dict(state_dict, strict=False)
    net.to(target_device)
    net.eval()
    return net, default_threshold, checkpoint if isinstance(checkpoint, dict) else {}


def predict_image(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    threshold: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Executes binary classification inference on a single image.

    Args:
        image_path: Filesystem path to image.
        checkpoint_path: Path to checkpoint .pth file.
        model: Optional pre-loaded nn.Module instance.
        threshold: Decision threshold override [0.0, 1.0]. If None, uses checkpoint threshold.
        device: Target hardware device ('cpu', 'cuda').

    Returns:
        Dict matching E2E contracts and dispatch requirements:
            - 'image_id': Base name or identifier.
            - 'file_name': File name with extension.
            - 'file_path': Absolute path.
            - 'source_image': Root source file name.
            - 'is_tile': False for full frame image.
            - 'x_min', 'y_min', 'x_max', 'y_max': Pixel bounding box.
            - 'prob_waste': Class 1 probability [0.0, 1.0].
            - 'prob_non_waste': Class 0 probability [0.0, 1.0].
            - 'probability': Positive class probability.
            - 'threshold': Applied decision threshold.
            - 'predicted_class': 1 if prob_waste >= threshold else 0.
            - 'class_label': 'waste' or 'non-waste'.
            - 'status': 'candidate_detection' if prob_waste >= threshold else 'negative'.
            - 'is_waste': Boolean indicator.
            - 'has_georeference': Boolean.
            - 'longitude': Float or None.
            - 'latitude': Float or None.

    Raises:
        FileNotFoundError: If image_path does not exist.
        RuntimeError: If image decoding fails.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    target_device = resolve_device(device)

    # Load model and threshold
    if model is None:
        model, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device)
    else:
        default_thresh = 0.50
        if checkpoint_path is not None:
            _, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device, model=model)

    eff_threshold = float(threshold) if threshold is not None else default_thresh
    if not (0.0 <= eff_threshold <= 1.0):
        raise ValueError(f"threshold must be in range [0.0, 1.0], got {eff_threshold}")

    # Load and preprocess image
    transform = get_eval_transforms(image_size=224)
    pil_img = safe_load_image(path)
    width, height = pil_img.size
    tensor = transform(pil_img).unsqueeze(0).to(target_device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    p_non_waste = float(probs[0])
    p_waste = float(probs[1]) if len(probs) > 1 else 1.0 - p_non_waste

    # Clamp simplex constraints: p >= 0 and p0 + p1 == 1.0
    p_waste = max(0.0, min(1.0, p_waste))
    p_non_waste = float(round(1.0 - p_waste, 6))
    p_waste = float(round(p_waste, 6))

    is_waste = bool(p_waste >= eff_threshold)
    pred_class = 1 if is_waste else 0
    class_label = "waste" if is_waste else "non-waste"
    status = "candidate_detection" if is_waste else "negative"

    has_geo, lon, lat = check_georeference(path)

    return {
        "image_id": path.name,
        "file_name": path.name,
        "file_path": str(path),
        "source_image": path.name,
        "is_tile": False,
        "x_min": 0,
        "y_min": 0,
        "x_max": int(width),
        "y_max": int(height),
        "predicted_class": int(pred_class),
        "class_label": class_label,
        "probability": float(p_waste),
        "prob_waste": float(p_waste),
        "prob_non_waste": float(p_non_waste),
        "is_waste": is_waste,
        "status": status,
        "threshold": float(eff_threshold),
        "has_georeference": bool(has_geo),
        "longitude": lon,
        "latitude": lat,
    }


def predict_folder(
    folder_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    output_csv: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> pd.DataFrame:
    """
    Executes batch inference across all valid images within a directory.

    Fault Tolerance:
        - Catches corrupted or truncated images, logs warnings, and continues processing.
        - Empty directories log a warning, generate an empty CSV with valid headers, and exit cleanly.

    Args:
        folder_path: Directory path containing imagery.
        checkpoint_path: Path to model checkpoint .pth file.
        model: Optional pre-loaded nn.Module.
        output_csv: Optional path to export CSV predictions.
        threshold: Decision threshold override [0.0, 1.0].
        batch_size: Number of images per inference batch.
        device: Target execution device.

    Returns:
        pd.DataFrame with standard schema.
    """
    f_path = Path(folder_path).resolve()
    if not f_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {f_path}")

    target_device = resolve_device(device)

    # Discover candidate image files (deterministic sorting)
    all_files = sorted(
        [p for p in f_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS],
        key=lambda x: str(x),
    )

    # Handle boundary case: empty directory
    if len(all_files) == 0:
        logger.warning("No valid images found in directory: %s", f_path)
        empty_df = pd.DataFrame(columns=STANDARD_COLUMNS)
        if output_csv is not None:
            out_p = Path(output_csv).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(out_p, index=False)
        return empty_df

    # Load model and threshold once
    if model is None:
        loaded_model, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device)
    else:
        loaded_model = model
        default_thresh = 0.50
        if checkpoint_path is not None:
            _, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device, model=loaded_model)

    eff_threshold = float(threshold) if threshold is not None else default_thresh
    if not (0.0 <= eff_threshold <= 1.0):
        raise ValueError(f"threshold must be in range [0.0, 1.0], got {eff_threshold}")

    transform = get_eval_transforms(image_size=224)
    records: List[Dict[str, Any]] = []

    # Process files in mini-batches
    total_files = len(all_files)
    for chunk_start in range(0, total_files, batch_size):
        chunk_paths = all_files[chunk_start : chunk_start + batch_size]
        batch_tensors: List[torch.Tensor] = []
        valid_paths: List[Path] = []
        dimensions: List[Tuple[int, int]] = []

        for p in chunk_paths:
            try:
                img = safe_load_image(p)
                w, h = img.size
                tensor = transform(img)
                batch_tensors.append(tensor)
                valid_paths.append(p)
                dimensions.append((w, h))
            except Exception as exc:
                logger.warning("Skipping unreadable or corrupted image '%s': %s", p, exc)

        if len(batch_tensors) == 0:
            continue

        stacked = torch.stack(batch_tensors).to(target_device)

        loaded_model.eval()
        with torch.no_grad():
            logits = loaded_model(stacked)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for path, (w, h), prob_row in zip(valid_paths, dimensions, probs):
            p_non_waste = float(prob_row[0])
            p_waste = float(prob_row[1]) if len(prob_row) > 1 else 1.0 - p_non_waste

            p_waste = max(0.0, min(1.0, p_waste))
            p_non_waste = float(round(1.0 - p_waste, 6))
            p_waste = float(round(p_waste, 6))

            is_waste = bool(p_waste >= eff_threshold)
            pred_class = 1 if is_waste else 0
            class_label = "waste" if is_waste else "non-waste"
            status = "candidate_detection" if is_waste else "negative"

            has_geo, lon, lat = check_georeference(path)

            records.append({
                "image_id": path.name,
                "file_name": path.name,
                "file_path": str(path),
                "source_image": path.name,
                "is_tile": False,
                "x_min": 0,
                "y_min": 0,
                "x_max": int(w),
                "y_max": int(h),
                "predicted_class": int(pred_class),
                "class_label": class_label,
                "probability": float(p_waste),
                "prob_waste": float(p_waste),
                "prob_non_waste": float(p_non_waste),
                "is_waste": is_waste,
                "status": status,
                "threshold": float(eff_threshold),
                "has_georeference": bool(has_geo),
                "longitude": lon if lon is not None else np.nan,
                "latitude": lat if lat is not None else np.nan,
            })

    df = pd.DataFrame(records, columns=STANDARD_COLUMNS)

    if output_csv is not None:
        out_p = Path(output_csv).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved %d predictions to %s", len(df), out_p)

    return df


def _sliding_window_tiles(
    img: Image.Image,
    tile_size: int = 256,
    stride: int = 200,
) -> Iterator[Tuple[Image.Image, int, int, int, int]]:
    """
    Generates sliding window crops with border shift-back to prevent fractional border clipping.
    """
    w, h = img.size

    # Handle smaller than tile_size
    if w <= tile_size and h <= tile_size:
        yield img, 0, 0, w, h
        return

    y = 0
    while y < h:
        y_min = y
        y_max = min(y + tile_size, h)
        if y_max - y_min < tile_size and h >= tile_size:
            y_min = max(0, h - tile_size)
            y_max = h

        x = 0
        while x < w:
            x_min = x
            x_max = min(x + tile_size, w)
            if x_max - x_min < tile_size and w >= tile_size:
                x_min = max(0, w - tile_size)
                x_max = w

            crop_box = (x_min, y_min, x_max, y_max)
            tile_crop = img.crop(crop_box)
            yield tile_crop, x_min, y_min, x_max, y_max

            if x + stride >= w:
                break
            x += stride

        if y + stride >= h:
            break
        y += stride


def predict_image_tiled(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    output_csv: Optional[Union[str, Path]] = None,
    tile_size: int = 256,
    stride: int = 200,
    threshold: Optional[float] = None,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> pd.DataFrame:
    """
    Partitions an oversized image into sliding window tiles, runs inference,
    preserves pixel coordinates, and exports prediction records.

    Args:
        image_path: Path to large image file.
        checkpoint_path: Path to checkpoint .pth file.
        model: Optional pre-loaded nn.Module.
        output_csv: Path to destination CSV.
        tile_size: Square tile dimension.
        stride: Step stride between consecutive tiles.
        threshold: Decision threshold override [0.0, 1.0].
        batch_size: Batch size for tile evaluation.
        device: Target execution device.

    Returns:
        pd.DataFrame containing tiled prediction records.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    target_device = resolve_device(device)
    if model is None:
        loaded_model, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device)
    else:
        loaded_model = model
        default_thresh = 0.50
        if checkpoint_path is not None:
            _, default_thresh, _ = load_model_for_inference(checkpoint_path, device=target_device, model=loaded_model)

    eff_threshold = float(threshold) if threshold is not None else default_thresh

    img = safe_load_image(path)
    transform = get_eval_transforms(image_size=224)

    # Check for georeferencing
    has_geo, lon_c, lat_c = check_georeference(path)

    # Collect tiles
    tile_crops = list(_sliding_window_tiles(img, tile_size=tile_size, stride=stride))
    records: List[Dict[str, Any]] = []

    for chunk_start in range(0, len(tile_crops), batch_size):
        chunk = tile_crops[chunk_start : chunk_start + batch_size]
        tensors = [transform(t[0]) for t in chunk]
        batch_tensor = torch.stack(tensors).to(target_device)

        loaded_model.eval()
        with torch.no_grad():
            logits = loaded_model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for idx_in_chunk, (_, x_min, y_min, x_max, y_max) in enumerate(chunk):
            global_idx = chunk_start + idx_in_chunk
            prob_row = probs[idx_in_chunk]

            p_non_waste = float(prob_row[0])
            p_waste = float(prob_row[1]) if len(prob_row) > 1 else 1.0 - p_non_waste

            p_waste = max(0.0, min(1.0, p_waste))
            p_non_waste = float(round(1.0 - p_waste, 6))
            p_waste = float(round(p_waste, 6))

            is_waste = bool(p_waste >= eff_threshold)
            pred_class = 1 if is_waste else 0
            class_label = "waste" if is_waste else "non-waste"
            status = "candidate_detection" if is_waste else "negative"

            tile_id = f"{path.stem}_tile_{global_idx:04d}"

            records.append({
                "image_id": tile_id,
                "file_name": path.name,
                "file_path": str(path),
                "source_image": path.name,
                "is_tile": True,
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
                "predicted_class": int(pred_class),
                "class_label": class_label,
                "probability": float(p_waste),
                "prob_waste": float(p_waste),
                "prob_non_waste": float(p_non_waste),
                "is_waste": is_waste,
                "status": status,
                "threshold": float(eff_threshold),
                "has_georeference": bool(has_geo),
                "longitude": lon_c if (has_geo and lon_c is not None) else np.nan,
                "latitude": lat_c if (has_geo and lat_c is not None) else np.nan,
            })

    df = pd.DataFrame(records, columns=STANDARD_COLUMNS)

    if output_csv is not None:
        out_p = Path(output_csv).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved %d tiled predictions to %s", len(df), out_p)

    return df


def run_inference(
    input_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    tile_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> pd.DataFrame:
    """
    Unified programmatic entrypoint for running inference on an image file or directory.

    Args:
        input_path: Path to image file or directory.
        checkpoint_path: Path to checkpoint .pth file.
        output_path: Destination path for output CSV. Defaults to 'outputs/predictions.csv'.
        threshold: Decision threshold override [0.0, 1.0].
        batch_size: Mini-batch size.
        device: Hardware device string or torch.device.
        tile_size: Optional tile size for large images.
        stride: Optional stride for large images.

    Returns:
        pd.DataFrame containing inference predictions.
    """
    target_path = Path(input_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {target_path}")

    dest_csv = output_path if output_path is not None else "outputs/predictions.csv"

    if target_path.is_file():
        if tile_size is not None and tile_size > 0:
            st = stride if stride is not None else tile_size
            return predict_image_tiled(
                image_path=target_path,
                checkpoint_path=checkpoint_path,
                output_csv=dest_csv,
                tile_size=tile_size,
                stride=st,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
            )
        else:
            single_res = predict_image(
                image_path=target_path,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
                device=device,
            )
            # Format as 1-row DataFrame with standard columns
            row = {col: single_res.get(col, np.nan) for col in STANDARD_COLUMNS}
            df = pd.DataFrame([row], columns=STANDARD_COLUMNS)
            if dest_csv is not None:
                out_p = Path(dest_csv).resolve()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_p, index=False)
            return df

    elif target_path.is_dir():
        return predict_folder(
            folder_path=target_path,
            checkpoint_path=checkpoint_path,
            output_csv=dest_csv,
            threshold=threshold,
            batch_size=batch_size,
            device=device,
        )

    else:
        raise ValueError(f"Input path is neither a file nor a directory: {target_path}")


def build_parser() -> argparse.ArgumentParser:
    """Constructs command-line interface arguments parser."""
    parser = argparse.ArgumentParser(
        prog="python -m waste_hotspot.inference",
        description="Run urban waste hotspot detection inference on aerial and satellite imagery.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Path to an individual image file or image directory.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="artifacts/best_model.pth",
        help="Path to model checkpoint (.pth). Defaults to 'artifacts/best_model.pth'.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/predictions.csv",
        help="Path to destination predictions CSV file. Defaults to 'outputs/predictions.csv'.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Decision threshold override in [0.0, 1.0]. If omitted, uses checkpoint threshold or 0.50.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for folder inference. Default: 32.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Execution device ('cpu', 'cuda', 'cuda:0'). Default: auto-detect.",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable sliding-window tiling for large images.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for sliding-window tiled inference on large images (e.g. 256).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between sliding-window tiles (e.g. 200). Defaults to tile-size.",
    )
    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    )
    parser = build_parser()
    parsed = parser.parse_args(args)

    input_p = Path(parsed.input).resolve()
    if not input_p.exists():
        logger.error("Input path does not exist: %s", input_p)
        print(f"Error: Input path does not exist: {input_p}", file=sys.stderr)
        return 1

    try:
        if (parsed.tile or parsed.tile_size is not None) and input_p.is_file():
            ts = parsed.tile_size if parsed.tile_size is not None else 256
            st = parsed.stride if parsed.stride is not None else 200
            df = predict_image_tiled(
                image_path=input_p,
                checkpoint_path=parsed.checkpoint,
                output_csv=parsed.output,
                tile_size=ts,
                stride=st,
                threshold=parsed.threshold,
                batch_size=parsed.batch_size,
                device=parsed.device,
            )
        else:
            df = run_inference(
                input_path=input_p,
                checkpoint_path=parsed.checkpoint,
                output_path=parsed.output,
                threshold=parsed.threshold,
                batch_size=parsed.batch_size,
                device=parsed.device,
            )

        num_total = len(df)
        num_waste = int((df["status"] == "candidate_detection").sum()) if "status" in df else 0
        pct = (num_waste / num_total * 100.0) if num_total > 0 else 0.0

        print(f"\n========================================================")
        print(f"  Inference Completed Successfully")
        print(f"========================================================")
        print(f"  Input Source        : {input_p}")
        print(f"  Total Processed     : {num_total}")
        print(f"  Candidate Detections: {num_waste} ({pct:.1f}%)")
        print(f"  Predictions Saved   : {parsed.output}")
        print(f"========================================================\n")
        return 0

    except Exception as exc:
        logger.exception("Inference failed with unhandled exception: %s", exc)
        print(f"Inference execution error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
