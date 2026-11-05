"""Unit tests for sliding-window tiling module."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from waste_hotspot.tiling import generate_tiles, tile_image, _compute_1d_slices


def test_compute_1d_slices_exact():
    slices = _compute_1d_slices(512, 256, 256)
    assert slices == [(0, 256), (256, 512)]


def test_compute_1d_slices_with_overlap():
    slices = _compute_1d_slices(600, 256, 200)
    assert slices[0] == (0, 256)
    assert slices[1] == (200, 456)
    # Shift-back clamps to 600
    assert slices[-1] == (344, 600)
    assert slices[-1][1] == 600


def test_compute_1d_slices_smaller_than_tile():
    slices = _compute_1d_slices(100, 256, 256)
    assert slices == [(0, 100)]


def test_generate_tiles_from_array():
    arr = np.zeros((500, 500, 3), dtype=np.uint8)
    tiles = list(generate_tiles(arr, tile_size=256, stride=200))
    assert len(tiles) == 9  # 3x3 grid
    for tile_arr, meta in tiles:
        assert tile_arr.shape == (256, 256, 3)
        assert meta["x_max"] <= 500
        assert meta["y_max"] <= 500


def test_tile_image_saving(tmp_path: Path):
    img = Image.new("RGB", (300, 300), color=(50, 100, 150))
    img_path = tmp_path / "test_tile.png"
    img.save(img_path)

    out_dir = tmp_path / "tiles"
    records = tile_image(img_path, output_dir=out_dir, tile_size=200, stride=200)
    assert len(records) == 4
    for r in records:
        assert Path(r["tile_path"]).exists()

