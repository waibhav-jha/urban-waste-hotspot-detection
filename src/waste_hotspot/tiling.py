"""
Sliding-window tiling module for aerial and satellite imagery.

Provides overlapping sliding-window tile generation, boundary shift-back
coordinate clamping, pixel offset tracking, and support for large rasters.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("waste_hotspot.tiling")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _compute_1d_slices(dimension_size: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Computes 1D start and end coordinates with edge shift-back.

    Guarantees:
      - Full coverage from 0 to dimension_size.
      - Exact stride step between interior coordinates.
      - Final slice shifts back to clamp exactly to dimension_size.
      - If dimension_size <= tile_size, returns [(0, dimension_size)].

    Args:
        dimension_size: Full length along axis (width or height).
        tile_size: Desired window size.
        stride: Step stride between consecutive windows.

    Returns:
        List of (start, end) coordinate tuples.
    """
    if dimension_size <= tile_size:
        return [(0, dimension_size)]

    slices: List[Tuple[int, int]] = []
    pos = 0
    while pos + tile_size <= dimension_size:
        slices.append((pos, pos + tile_size))
        pos += stride

    # If the last slice didn't reach the boundary, shift back the last window
    if slices[-1][1] < dimension_size:
        shift_back_start = max(0, dimension_size - tile_size)
        if (shift_back_start, dimension_size) not in slices:
            slices.append((shift_back_start, dimension_size))

    # Deduplicate while preserving order
    seen = set()
    unique_slices = []
    for s in slices:
        if s not in seen:
            seen.add(s)
            unique_slices.append(s)

    return unique_slices


def load_image_as_array(image_source: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, str]:
    """
    Loads an image source into a NumPy array and extracts a display filename.

    Args:
        image_source: Path to file, PIL Image, or numpy array.

    Returns:
        Tuple of (numpy_array, source_name).
    """
    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        source_name = p.name
        # Try rasterio first if TIFF
        if p.suffix.lower() in {".tif", ".tiff"}:
            try:
                import rasterio
                with rasterio.open(p) as src:
                    arr = src.read()
                    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        arr = np.repeat(arr, 3, axis=2)
                    return arr, source_name
            except Exception:
                pass

        with Image.open(p) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            return np.array(pil_rgb), source_name

    elif isinstance(image_source, Image.Image):
        pil_rgb = image_source.convert("RGB")
        return np.array(pil_rgb), "in_memory_image"

    elif isinstance(image_source, np.ndarray):
        arr = image_source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return arr, "numpy_array"

    else:
        raise TypeError(f"Unsupported image_source type: {type(image_source)}")


def generate_tiles(
    image_path: Union[str, Path, np.ndarray, Image.Image],
    tile_size: int = 256,
    stride: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Yields sliding-window tiles and associated metadata for an image.

    Args:
        image_path: File path, PIL Image, or numpy array.
        tile_size: Window dimension in pixels (width and height).
        stride: Stride step between windows. Defaults to tile_size (no overlap).

    Yields:
        Tuple of (tile_array, metadata_dict).
    """
    img_arr, source_name = load_image_as_array(image_path)
    height, width = img_arr.shape[:2]
    step = stride if stride is not None and stride > 0 else tile_size

    x_slices = _compute_1d_slices(width, tile_size, step)
    y_slices = _compute_1d_slices(height, tile_size, step)

    tile_counter = 0
    stem = Path(source_name).stem if source_name != "numpy_array" else "tile"

    for y_min, y_max in y_slices:
        for x_min, x_max in x_slices:
            slice_h = y_max - y_min
            slice_w = x_max - x_min
            crop = img_arr[y_min:y_max, x_min:x_max]

            # If image dimension is smaller than tile_size, pad to tile_size
            if slice_h < tile_size or slice_w < tile_size:
                channels = img_arr.shape[2] if img_arr.ndim == 3 else 3
                padded = np.zeros((tile_size, tile_size, channels), dtype=img_arr.dtype)
                padded[:slice_h, :slice_w] = crop
                tile_arr = padded
            else:
                tile_arr = crop

            meta: Dict[str, Any] = {
                "source_image": source_name,
                "source_path": str(image_path) if isinstance(image_path, (str, Path)) else None,
                "tile_id": f"{stem}_tile_{tile_counter:04d}",
                "tile_index": tile_counter,
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
                "tile_w": int(slice_w),
                "tile_h": int(slice_h),
                "img_width": int(width),
                "img_height": int(height),
            }

            yield tile_arr, meta
            tile_counter += 1


def tile_image(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    tile_size: int = 256,
    stride: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generates tiles and optionally saves them to output_dir.

    Args:
        image_path: Source image path.
        output_dir: Optional destination directory for tile image files.
        tile_size: Window dimension.
        stride: Stride step.

    Returns:
        List of metadata dictionaries for all generated tiles.
    """
    out_dir_p = Path(output_dir) if output_dir is not None else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for tile_arr, meta in generate_tiles(image_path, tile_size=tile_size, stride=stride):
        if out_dir_p is not None:
            tile_filename = f"{meta['tile_id']}.png"
            tile_save_path = out_dir_p / tile_filename
            pil_img = Image.fromarray(tile_arr)
            pil_img.save(tile_save_path)
            meta["tile_path"] = str(tile_save_path)
        records.append(meta)

    return records


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI parser for tiling module."""
    parser = argparse.ArgumentParser(
        prog="python -m waste_hotspot.tiling",
        description="Tile high-resolution imagery into sliding-window patches.",
    )
    parser.add_argument("--input", "-i", required=True, type=str, help="Input image file path.")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for tile images.")
    parser.add_argument("--tile-size", "-s", type=int, default=256, help="Window size in pixels (default: 256).")
    parser.add_argument("--stride", type=int, default=None, help="Stride step in pixels (default: tile-size).")
    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    input_p = Path(parsed.input)
    if not input_p.is_file():
        logger.error("Input file does not exist: %s", input_p)
        print(f"Error: Input file does not exist: {input_p}", file=sys.stderr)
        return 1

    tiles = tile_image(
        image_path=input_p,
        output_dir=parsed.output_dir,
        tile_size=parsed.tile_size,
        stride=parsed.stride,
    )
    print(f"Generated {len(tiles)} tiles from {input_p.name} (tile_size={parsed.tile_size}, stride={parsed.stride or parsed.tile_size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
