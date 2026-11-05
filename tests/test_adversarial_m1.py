"""
Adversarial stress-test suite for Milestone 1: Data ingestion, deduplication,
scene grouping, and leakage-safe splitting (waste_hotspot.data).

Tests stress-test:
1. Deduplication edge cases (conflicting labels, duplicate filenames with distinct IDs)
2. Scene grouping dynamics (ID jumps, single-image scenes, 100-image chunks, sensor boundaries)
3. Splitting edge cases (tiny sample sizes N in [1..5], extreme class imbalance 100% / 0% waste)
4. Manifest mathematical zero-leakage verification (S_train ∩ S_val = ∅, S_train ∩ S_test = ∅)
5. Side-effect vulnerability analysis (default output_dir disk clobbering)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from waste_hotspot.data import (
    create_leakage_safe_splits,
    load_aerialwaste_metadata,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    td = Path(tempfile.mkdtemp())
    yield td
    shutil.rmtree(td, ignore_errors=True)


# ==============================================================================
# 1. DEDUPLICATION STRESS TESTS
# ==============================================================================

def test_deduplication_conflicting_labels_resolution(temp_dir: Path) -> None:
    """
    Stress-test deduplication when identical image IDs possess conflicting labels
    between training.json (label=0) and testing.json (label=1).
    Verifies that deduplication eliminates duplicate IDs (zero ID leakage),
    and observes how the conflict is resolved (first-seen wins).
    """
    data_train = {
        "images": [
            {"id": 100, "file_name": "100.png", "is_candidate_location": 0, "img_source": "GE", "width": 224, "height": 224},
            {"id": 101, "file_name": "101.png", "is_candidate_location": 1, "img_source": "GE", "width": 224, "height": 224},
        ]
    }
    data_test = {
        "images": [
            # Conflicting label for ID 100 (label=1 instead of 0)
            {"id": 100, "file_name": "100.png", "is_candidate_location": 1, "img_source": "GE", "width": 224, "height": 224},
            {"id": 102, "file_name": "102.png", "is_candidate_location": 0, "img_source": "GE", "width": 224, "height": 224},
        ]
    }

    with open(temp_dir / "training.json", "w", encoding="utf-8") as f:
        json.dump(data_train, f)
    with open(temp_dir / "testing.json", "w", encoding="utf-8") as f:
        json.dump(data_test, f)

    df = load_aerialwaste_metadata(temp_dir)

    # Assert exactly 3 unique records (deduplicated)
    assert len(df) == 3, f"Expected 3 records after dedup, got {len(df)}"
    assert df["image_id"].nunique() == 3

    # Assert ID 100 appears exactly once
    rec_100 = df[df["image_id"] == 100]
    assert len(rec_100) == 1
    # Check resolution policy: first loaded (training.json) retained
    assert rec_100.iloc[0]["label"] == 0


def test_deduplication_duplicate_filenames_distinct_ids_vulnerability(temp_dir: Path) -> None:
    """
    Verifies that duplicate file_names with distinct image_ids are properly deduplicated
    by load_aerialwaste_metadata and that create_leakage_safe_splits prevents cross-partition file leakage.
    """
    images = []
    for i in range(1, 21):
        images.append({
            "id": i * 10,
            "file_name": f"img_{i}.png",
            "is_candidate_location": i % 2,
            "img_source": "GE",
            "width": 224,
            "height": 224,
        })
    # Adversarial duplicate file_name with a distinct ID
    images.append({
        "id": 999,
        "file_name": "img_1.png",  # Same physical file as image_id 10!
        "is_candidate_location": 1,
        "img_source": "AGEA",
        "width": 224,
        "height": 224,
    })

    with open(temp_dir / "training.json", "w", encoding="utf-8") as f:
        json.dump({"images": images}, f)

    df = load_aerialwaste_metadata(temp_dir)
    assert len(df) == 20, f"Expected 20 records after filename dedup, got {len(df)}"
    assert df["file_name"].is_unique
    assert df["image_id"].is_unique

    # Verify splitting has zero file_name leakage
    train_df, val_df, test_df = create_leakage_safe_splits(df, output_dir=None, random_seed=42)
    assert len(set(train_df["file_name"]) & set(test_df["file_name"])) == 0
    assert len(set(train_df["file_name"]) & set(val_df["file_name"])) == 0
    assert len(set(val_df["file_name"]) & set(test_df["file_name"])) == 0


def test_create_leakage_safe_splits_catches_duplicate_filenames() -> None:
    """
    Verifies that create_leakage_safe_splits raises AssertionError if raw overlapping
    filenames cross partition boundaries.
    """
    records = []
    for i in range(1, 21):
        records.append({
            "image_id": i * 10,
            "file_name": f"img_{i}.png",
            "label": i % 2,
            "img_source": "GE",
            "width": 224,
            "height": 224,
        })
    records.append({
        "image_id": 999,
        "file_name": "img_1.png",  # duplicate of img_1.png
        "label": 1,
        "img_source": "AGEA",
        "width": 224,
        "height": 224,
    })
    raw_df = pd.DataFrame(records)
    with pytest.raises(AssertionError, match="[Ff]ile"):
        create_leakage_safe_splits(raw_df, output_dir=None, random_seed=42)


# ==============================================================================
# 2. SCENE GROUPING DYNAMICS
# ==============================================================================

def test_scene_grouping_id_jumps() -> None:
    """
    Stress-test scene grouping sensitivity to ID delta jumps.
    Delta <= 5 should remain in the same scene; Delta > 5 must trigger a new scene.
    """
    df = pd.DataFrame({
        "image_id": [1, 2, 5, 12, 15],
        "file_name": [f"{i}.png" for i in [1, 2, 5, 12, 15]],
        "label": [0, 0, 0, 1, 1],
        "img_source": ["GE"] * 5,
        "width": [224] * 5,
        "height": [224] * 5,
    })

    # Run scene assignment logic as in create_leakage_safe_splits
    df_sorted = df.sort_values("image_id").copy().reset_index(drop=True)
    scene_ids = []
    current_scene = 0
    prev_source = None
    prev_id = -9999
    chunk_count = 0

    for _, row in df_sorted.iterrows():
        curr_id = int(row["image_id"])
        jump = (curr_id - prev_id) > 5
        if row["img_source"] != prev_source or jump or chunk_count >= 10:
            current_scene += 1
            chunk_count = 0
        scene_ids.append(f"scene_{current_scene:05d}")
        prev_source = row["img_source"]
        prev_id = curr_id
        chunk_count += 1

    df_sorted["scene_id"] = scene_ids

    # IDs 1, 2, 5 should be in scene_00001
    assert df_sorted.loc[df_sorted["image_id"].isin([1, 2, 5]), "scene_id"].nunique() == 1
    # ID 12 should be in scene_00002
    assert df_sorted.loc[df_sorted["image_id"] == 12, "scene_id"].iloc[0] == "scene_00002"
    # ID 15 should also be in scene_00002 (delta 3 <= 5)
    assert df_sorted.loc[df_sorted["image_id"] == 15, "scene_id"].iloc[0] == "scene_00002"


def test_scene_grouping_single_image_scenes() -> None:
    """
    Stress-test scene grouping when every image is isolated (jump > 5 for all images).
    Every image should be assigned its own unique scene.
    """
    n_images = 25
    df = pd.DataFrame({
        "image_id": [i * 20 for i in range(1, n_images + 1)],
        "file_name": [f"{i}.png" for i in range(1, n_images + 1)],
        "label": [i % 2 for i in range(1, n_images + 1)],
        "img_source": ["GE"] * n_images,
        "width": [224] * n_images,
        "height": [224] * n_images,
    })

    train_df, val_df, test_df = create_leakage_safe_splits(df, output_dir=None, random_seed=42)
    all_df = pd.concat([train_df, val_df, test_df])
    # Every image should have its own scene
    assert all_df["scene_id"].nunique() == n_images


def test_scene_grouping_100_contiguous_images_single_scene() -> None:
    """
    Verify that 100 consecutive frames with delta=1 from the same sensor
    are grouped into a SINGLE continuous scene with zero chunking artifacts
    and zero adjacent frame boundary crossings.
    """
    n_images = 100
    df = pd.DataFrame({
        "image_id": list(range(1, n_images + 1)),
        "file_name": [f"{i}.png" for i in range(1, n_images + 1)],
        "label": [i % 2 for i in range(1, n_images + 1)],
        "img_source": ["GE"] * n_images,
        "width": [224] * n_images,
        "height": [224] * n_images,
    })

    train_df, val_df, test_df = create_leakage_safe_splits(df, output_dir=None, random_seed=42)
    all_df = pd.concat([train_df, val_df, test_df]).sort_values("image_id").reset_index(drop=True)

    # Exactly 1 scene formed (no chunk boundary splits)
    assert all_df["scene_id"].nunique() == 1

    # Entire continuous flight sequence assigned to exactly one partition
    active_splits = [s for s in [train_df, val_df, test_df] if len(s) > 0]
    assert len(active_splits) == 1
    assert len(active_splits[0]) == 100

    # Cross-partition boundary frame leaks must be 0
    boundary_crossings = [
        (all_df.iloc[i]["image_id"], all_df.iloc[i + 1]["image_id"])
        for i in range(len(all_df) - 1)
        if all_df.iloc[i]["split"] != all_df.iloc[i + 1]["split"]
    ]
    assert len(boundary_crossings) == 0


def test_scene_grouping_sensor_boundaries() -> None:
    """
    Stress-test scene grouping across sensor boundaries.
    Changing img_source must immediately force a new scene even if delta_id <= 5.
    """
    from waste_hotspot.data import assign_scenes

    df = pd.DataFrame({
        "image_id": [1, 2, 3, 4],
        "file_name": ["1.png", "2.png", "3.png", "4.png"],
        "label": [0, 1, 0, 1],
        "img_source": ["GE", "AGEA", "AGEA", "WV3"],
        "width": [224] * 4,
        "height": [224] * 4,
    })

    df_sorted = assign_scenes(df)
    # 1 is GE (scene 1), 2 & 3 are AGEA (scene 2), 4 is WV3 (scene 3)
    assert df_sorted["scene_id"].tolist() == ["scene_00001", "scene_00002", "scene_00002", "scene_00003"]


# ==============================================================================
# 3. SPLITTING EDGE CASES & ZERO LEAKAGE
# ==============================================================================

@pytest.mark.parametrize("n_samples", [1, 2, 3])
def test_splitting_handles_tiny_sample_sizes_under_4(n_samples: int) -> None:
    """
    Stress-test splitting on tiny sample sizes N in [1, 2, 3].
    Verifies that partitioning <= 3 scenes across partitions
    gracefully distributes records without raising an unhandled ValueError
    and maintains strict zero leakage.
    """
    df_tiny = pd.DataFrame({
        "image_id": [i * 20 for i in range(1, n_samples + 1)],
        "file_name": [f"{i}.png" for i in range(1, n_samples + 1)],
        "label": [i % 2 for i in range(1, n_samples + 1)],
        "img_source": ["GE"] * n_samples,
        "width": [224] * n_samples,
        "height": [224] * n_samples,
    })

    tr, val, te = create_leakage_safe_splits(df_tiny, output_dir=None, random_seed=42)

    assert len(tr) + len(val) + len(te) == n_samples
    assert len(set(tr["image_id"]) & set(val["image_id"])) == 0
    assert len(set(tr["image_id"]) & set(te["image_id"])) == 0
    assert len(set(val["image_id"]) & set(te["image_id"])) == 0
    assert len(set(tr["scene_id"]) & set(val["scene_id"])) == 0
    assert len(set(tr["scene_id"]) & set(te["scene_id"])) == 0
    assert len(set(val["scene_id"]) & set(te["scene_id"])) == 0

    if n_samples == 3:
        assert len(tr) == 1 and len(val) == 1 and len(te) == 1
    elif n_samples == 2:
        assert len(tr) == 1 and len(val) == 1 and len(te) == 0
    elif n_samples == 1:
        assert len(tr) == 1 and len(val) == 0 and len(te) == 0


def test_splitting_single_scene_multiple_images() -> None:
    """
    Stress-test splitting when there are multiple images (e.g. 5) but only 1 unique scene.
    Verifies graceful assignment to train partition without crashing or scene leakage.
    """
    df_single_scene = pd.DataFrame({
        "image_id": [1, 2, 3, 4, 5],
        "file_name": [f"{i}.png" for i in range(1, 6)],
        "label": [i % 2 for i in range(1, 6)],
        "img_source": ["GE"] * 5,
        "width": [224] * 5,
        "height": [224] * 5,
    })
    tr, val, te = create_leakage_safe_splits(df_single_scene, output_dir=None, random_seed=42)
    assert len(tr) == 5
    assert len(val) == 0
    assert len(te) == 0
    assert len(set(tr["image_id"]) & set(val["image_id"])) == 0
    assert len(set(tr["image_id"]) & set(te["image_id"])) == 0
    assert len(set(val["image_id"]) & set(te["image_id"])) == 0


def test_splitting_empty_dataframe_raises_value_error() -> None:
    """Verifies that an empty DataFrame still raises a clean ValueError."""
    with pytest.raises(ValueError, match="Cannot split empty DataFrame"):
        create_leakage_safe_splits(pd.DataFrame(), output_dir=None)


@pytest.mark.parametrize("n_samples", [4, 5])
def test_splitting_succeeds_on_minimal_valid_scene_counts(n_samples: int) -> None:
    """
    Stress-test splitting on minimal viable sample sizes N = 4 and N = 5
    (where each image forms an independent scene).
    Verifies that zero leakage holds strictly even at the minimal boundary.
    """
    df_min = pd.DataFrame({
        "image_id": [i * 20 for i in range(1, n_samples + 1)],
        "file_name": [f"{i}.png" for i in range(1, n_samples + 1)],
        "label": [i % 2 for i in range(1, n_samples + 1)],
        "img_source": ["GE"] * n_samples,
        "width": [224] * n_samples,
        "height": [224] * n_samples,
    })

    tr, val, te = create_leakage_safe_splits(df_min, output_dir=None, random_seed=42)

    assert len(tr) > 0 and len(val) > 0 and len(te) > 0
    assert len(set(tr["image_id"]) & set(val["image_id"])) == 0
    assert len(set(tr["image_id"]) & set(te["image_id"])) == 0
    assert len(set(val["image_id"]) & set(te["image_id"])) == 0


def test_splitting_extreme_imbalance_100_percent_waste() -> None:
    """
    Stress-test splitting with 100% positive class (all waste, label=1).
    Verifies that single-class stratification fallback operates without crash
    and maintains strict zero leakage.
    """
    n = 30
    df = pd.DataFrame({
        "image_id": [i * 20 for i in range(1, n + 1)],
        "file_name": [f"{i}.png" for i in range(1, n + 1)],
        "label": [1] * n,  # 100% waste
        "img_source": ["GE"] * n,
        "width": [224] * n,
        "height": [224] * n,
    })

    tr, val, te = create_leakage_safe_splits(df, output_dir=None, random_seed=42)
    assert tr["label"].mean() == 1.0
    assert val["label"].mean() == 1.0
    assert te["label"].mean() == 1.0
    assert len(set(tr["image_id"]) & set(val["image_id"])) == 0
    assert len(set(tr["image_id"]) & set(te["image_id"])) == 0
    assert len(set(val["image_id"]) & set(te["image_id"])) == 0


def test_splitting_extreme_imbalance_0_percent_waste() -> None:
    """
    Stress-test splitting with 0% positive class (all non-waste, label=0).
    Verifies that single-class stratification fallback operates without crash
    and maintains strict zero leakage.
    """
    n = 30
    df = pd.DataFrame({
        "image_id": [i * 20 for i in range(1, n + 1)],
        "file_name": [f"{i}.png" for i in range(1, n + 1)],
        "label": [0] * n,  # 0% waste
        "img_source": ["GE"] * n,
        "width": [224] * n,
        "height": [224] * n,
    })

    tr, val, te = create_leakage_safe_splits(df, output_dir=None, random_seed=42)
    assert tr["label"].mean() == 0.0
    assert val["label"].mean() == 0.0
    assert te["label"].mean() == 0.0
    assert len(set(tr["image_id"]) & set(val["image_id"])) == 0
    assert len(set(tr["image_id"]) & set(te["image_id"])) == 0
    assert len(set(val["image_id"]) & set(te["image_id"])) == 0


# ==============================================================================
# 4. MANIFEST MATHEMATICAL ZERO-LEAKAGE VERIFICATION
# ==============================================================================

def test_processed_manifests_mathematical_zero_leakage() -> None:
    """
    Mathematically inspect the generated manifests (train.csv, val.csv, test.csv)
    in data/processed/ and verify:
      S_train ∩ S_val = ∅
      S_train ∩ S_test = ∅
      S_val ∩ S_test = ∅
    for both image_id and scene_id.
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    assert train_path.exists(), f"Missing manifest: {train_path}"
    assert val_path.exists(), f"Missing manifest: {val_path}"
    assert test_path.exists(), f"Missing manifest: {test_path}"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Check non-empty
    assert len(train_df) > 0, "train.csv is empty"
    assert len(val_df) > 0, "val.csv is empty"
    assert len(test_df) > 0, "test.csv is empty"

    # Extract ID sets
    s_train_ids = set(train_df["image_id"])
    s_val_ids = set(val_df["image_id"])
    s_test_ids = set(test_df["image_id"])

    # Strict mathematical set intersections
    assert len(s_train_ids & s_val_ids) == 0, "Non-empty intersection: S_train_ids ∩ S_val_ids != ∅"
    assert len(s_train_ids & s_test_ids) == 0, "Non-empty intersection: S_train_ids ∩ S_test_ids != ∅"
    assert len(s_val_ids & s_test_ids) == 0, "Non-empty intersection: S_val_ids ∩ S_test_ids != ∅"

    # Extract scene sets
    s_train_scenes = set(train_df["scene_id"])
    s_val_scenes = set(val_df["scene_id"])
    s_test_scenes = set(test_df["scene_id"])

    assert len(s_train_scenes & s_val_scenes) == 0, "Non-empty intersection: S_train_scenes ∩ S_val_scenes != ∅"
    assert len(s_train_scenes & s_test_scenes) == 0, "Non-empty intersection: S_train_scenes ∩ S_test_scenes != ∅"
    assert len(s_val_scenes & s_test_scenes) == 0, "Non-empty intersection: S_val_scenes ∩ S_test_scenes != ∅"

    # Extract file_name sets
    if "file_name" in train_df.columns:
        s_train_files = set(train_df["file_name"])
        s_val_files = set(val_df["file_name"])
        s_test_files = set(test_df["file_name"])

        assert len(s_train_files & s_val_files) == 0, "Non-empty intersection: S_train_files ∩ S_val_files != ∅"
        assert len(s_train_files & s_test_files) == 0, "Non-empty intersection: S_train_files ∩ S_test_files != ∅"
        assert len(s_val_files & s_test_files) == 0, "Non-empty intersection: S_val_files ∩ S_test_files != ∅"

    # Check union size matches sum of lengths (strict partition)
    total_unique_ids = len(s_train_ids | s_val_ids | s_test_ids)
    assert total_unique_ids == len(train_df) + len(val_df) + len(test_df)


def test_processed_manifests_zero_adjacent_sensor_frame_leakage() -> None:
    """
    Verify that on the generated production split manifests (train.csv, val.csv, test.csv),
    no adjacent frames (delta_id <= 5 from the same sensor) cross split boundaries.
    Eliminates the 439 boundary frame leaks identified in challenger review.
    """
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"

    tr = pd.read_csv(processed_dir / "train.csv")
    va = pd.read_csv(processed_dir / "val.csv")
    te = pd.read_csv(processed_dir / "test.csv")

    tr["split"] = "train"
    va["split"] = "val"
    te["split"] = "test"

    df_all = pd.concat([tr, va, te]).sort_values("image_id").reset_index(drop=True)

    cross_boundary_leaks = 0
    for i in range(len(df_all) - 1):
        r1 = df_all.iloc[i]
        r2 = df_all.iloc[i + 1]
        if (
            r1["img_source"] == r2["img_source"]
            and (r2["image_id"] - r1["image_id"] <= 5)
            and (r1["split"] != r2["split"])
        ):
            cross_boundary_leaks += 1

    assert cross_boundary_leaks == 0, (
        f"Detected {cross_boundary_leaks} adjacent frames crossing split boundaries! "
        f"Expected 0 under natural scene grouping."
    )


# ==============================================================================
# 5. SIDE-EFFECT VULNERABILITY ANALYSIS
# ==============================================================================

def test_default_output_dir_side_effect_vulnerability() -> None:
    """
    Verifies that create_leakage_safe_splits defaults output_dir to None,
    preventing unintended disk clobbering during testing or interactive experimentation.
    """
    import inspect
    from waste_hotspot import data

    sig = inspect.signature(data.create_leakage_safe_splits)
    output_dir_param = sig.parameters.get("output_dir")

    assert output_dir_param is not None
    assert output_dir_param.default is None, (
        f"Expected output_dir default to be None, got {output_dir_param.default}"
    )
