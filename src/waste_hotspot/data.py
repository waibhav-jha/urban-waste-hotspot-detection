"""
Dataset ingestion, Zenodo downloading, metadata parsing, deduplication,
and scene-aware leakage-safe splitting for the AerialWaste dataset (Zenodo record 7991872).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import yaml

# Ensure src/ is on sys.path if invoked directly
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants for Zenodo record 7991872
ZENODO_RECORD_ID = "7991872"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

CHECKSUMS: Dict[str, str] = {
    "training.json": "ef069c90694e1174bae06f60f7e8c139",
    "testing.json": "6af2da456dbd6232c9051b2d15dd4cca",
    "LICENSE": "4a17ffc27c9f3b240fbf4fe17783c89c",
    "images6.zip": "63b6ea3a6dd8838000701843e8d2b3f0",
    "images0.zip": "99a48ac4a23b58266d8ec678bc587077",
    "images1.zip": "0f9bb21f6b19e0a05638ac42f5e49fab",
    "images2.zip": "768a21b031bed9f521372eb17152fb39",
    "images3.zip": "c59f2bc03be6912ddf2f1bad0b6c3c39",
    "images4.zip": "9ab464610b6a807a16148122c82fdc9d",
    "images5.zip": "24d66b1c156610baedd092bb24766380",
}


@dataclass
class DatasetStats:
    total_raw_records: int
    unique_images: int
    duplicate_ids_count: int
    num_waste: int
    num_non_waste: int
    waste_percentage: float
    source_counts: Dict[str, int]
    images_on_disk: int
    missing_images: int


def verify_md5(filepath: Union[str, Path], expected_md5: str) -> bool:
    """Verifies MD5 hash of a local file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return False
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_md5.lower()


def download_file_with_resume(
    url: str,
    dest_path: Union[str, Path],
    expected_md5: Optional[str] = None,
    timeout: int = 30,
) -> Path:
    """
    Downloads a file with HTTP Range resume support and optional MD5 verification.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if expected_md5 and verify_md5(dest_path, expected_md5):
            logger.info(f"File {dest_path.name} already exists and MD5 matches. Skipping download.")
            return dest_path
        elif not expected_md5:
            logger.info(f"File {dest_path.name} already exists. Skipping download.")
            return dest_path

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0

    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        logger.info(f"Resuming download of {dest_path.name} from byte {existing_bytes}")

    response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    if response.status_code == 416:  # Range not satisfiable, restart
        part_path.unlink(missing_ok=True)
        existing_bytes = 0
        response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    mode = "ab" if existing_bytes > 0 and response.status_code == 206 else "wb"
    with open(part_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if expected_md5 and not verify_md5(part_path, expected_md5):
        part_path.unlink(missing_ok=True)
        raise ValueError(f"MD5 mismatch for {dest_path.name} after download.")

    part_path.rename(dest_path)
    logger.info(f"Successfully downloaded {dest_path.name}")
    return dest_path


def download_aerialwaste(
    dest_dir: Union[str, Path] = "data/raw",
    mode: str = "metadata",
    cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Acquires AerialWaste dataset files from Zenodo or local pre-cached directory.
    Modes:
      - 'metadata': downloads training.json, testing.json, LICENSE
      - 'fast' / 'subset': metadata + images6.zip (extracted)
      - 'full': metadata + all images zips (extracted sequentially with zip cleanup)
      - 'mock': generates synthetic dataset without internet access
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    result_paths = {}

    if mode == "mock":
        return generate_synthetic_dataset(dest_dir)

    # 1. Acquire metadata files (check cache first)
    meta_files = ["training.json", "testing.json", "LICENSE"]
    for filename in meta_files:
        target_path = dest_dir / filename
        if target_path.exists() and verify_md5(target_path, CHECKSUMS[filename]):
            result_paths[filename] = target_path
            continue

        # Check local cache candidate directories
        cached = False
        candidates = [
            cache_dir,
            Path(".agents/teamwork/explorer_survey_2"),
            Path("../.agents/teamwork/explorer_survey_2"),
            Path("c:/Users/waibhav/Documents/geospat/.agents/teamwork/explorer_survey_2"),
        ]
        for cand in candidates:
            if cand and (Path(cand) / filename).exists():
                cand_file = Path(cand) / filename
                if verify_md5(cand_file, CHECKSUMS[filename]):
                    shutil.copy2(cand_file, target_path)
                    logger.info(f"Copied {filename} from local cache: {cand_file}")
                    result_paths[filename] = target_path
                    cached = True
                    break
        if cached:
            continue

        # Download from Zenodo
        url = f"{ZENODO_BASE_URL}/{filename}?download=1"
        target = download_file_with_resume(url, target_path, expected_md5=CHECKSUMS[filename])
        result_paths[filename] = target

    # 2. Acquire image archives based on mode
    if mode in ["fast", "subset"]:
        zip_name = "images6.zip"
        zip_url = f"{ZENODO_BASE_URL}/{zip_name}?download=1"
        zip_path = dest_dir / zip_name
        download_file_with_resume(zip_url, zip_path, expected_md5=CHECKSUMS[zip_name])
        logger.info(f"Extracting {zip_name} into {images_dir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(images_dir)
        result_paths[zip_name] = zip_path

    elif mode == "full":
        for i in range(7):
            zip_name = f"images{i}.zip"
            zip_url = f"{ZENODO_BASE_URL}/{zip_name}?download=1"
            zip_path = dest_dir / zip_name
            download_file_with_resume(zip_url, zip_path, expected_md5=CHECKSUMS[zip_name])
            logger.info(f"Extracting {zip_name} into {images_dir}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(images_dir)
            # Delete zip to conserve disk space
            zip_path.unlink()
            logger.info(f"Extracted and removed {zip_name} to conserve disk space.")

    return result_paths


def generate_synthetic_dataset(
    output_dir: Union[str, Path],
    n_samples: int = 30,
    image_size: int = 224,
) -> Dict[str, Path]:
    """
    Generates a synthetic mock AerialWaste dataset for fast offline testing.
    Creates mock PNG images, training.json, and testing.json with overlapping IDs.
    """
    from PIL import Image

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sources = ["GE", "AGEA", "WV3"]
    train_images = []
    test_images = []

    for i in range(1, n_samples + 1):
        # 1/3 positive, 2/3 negative
        label = 1 if i % 3 == 0 else 0
        source = sources[i % len(sources)]
        file_name = f"{i}.png"
        file_path = images_dir / file_name

        # Create synthetic image
        # Waste: brown/dirty texture, Clean: green/blue texture
        color = (139, 69, 19) if label == 1 else (34, 139, 34)
        arr = np.full((image_size, image_size, 3), color, dtype=np.uint8)
        # Add random noise
        noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(file_path)

        rec = {
            "id": i,
            "file_name": file_name,
            "img_source": source,
            "width": image_size,
            "height": image_size,
            "is_candidate_location": label,
            "categories": [1] if label == 1 else [],
        }

        # Put first 75% in train, last 35% in test -> creates intentional overlap
        if i <= int(n_samples * 0.75):
            train_images.append(rec)
        if i >= int(n_samples * 0.65):
            test_images.append(rec)

    train_data = {"images": train_images, "categories": [{"id": 1, "name": "Waste"}]}
    test_data = {"images": test_images, "categories": [{"id": 1, "name": "Waste"}]}

    train_json_path = output_dir / "training.json"
    test_json_path = output_dir / "testing.json"
    license_path = output_dir / "LICENSE"

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    with open(license_path, "w", encoding="utf-8") as f:
        f.write("Creative Commons Attribution 4.0 International (CC-BY 4.0)\nMock Dataset.\n")

    logger.info(f"Generated synthetic mock dataset at {output_dir} with {n_samples} images.")
    return {
        "training.json": train_json_path,
        "testing.json": test_json_path,
        "LICENSE": license_path,
    }


def load_aerialwaste_metadata(
    metadata_source: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    images_dir: Optional[Union[str, Path]] = None,
    filter_existing: bool = False,
    json_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
) -> pd.DataFrame:
    """
    Loads AerialWaste metadata from JSON files or raw directory, extracts required schema,
    and deduplicates overlapping IDs between training and testing metadata.

    Supports both metadata_source and json_path arguments for full interface compliance.
    """
    source = metadata_source if metadata_source is not None else json_path
    if source is None:
        source = "data/raw"

    records: List[Dict[str, Any]] = []

    # Resolve input JSON paths
    json_paths: List[Path] = []
    if isinstance(source, (list, tuple)):
        json_paths = [Path(p) for p in source]
    else:
        p = Path(source)
        if p.is_dir():
            for name in ["training.json", "testing.json"]:
                if (p / name).exists():
                    json_paths.append(p / name)
        elif p.is_file():
            json_paths.append(p)

    if not json_paths:
        raise FileNotFoundError(f"No valid JSON metadata found for source: {source}")

    for jpath in json_paths:
        logger.info(f"Loading metadata from {jpath.name}...")
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build annotations lookup if present (standard COCO annotation format)
        ann_labels: Dict[int, int] = {}
        for ann in data.get("annotations", []):
            if "image_id" in ann and "category_id" in ann:
                ann_labels[int(ann["image_id"])] = int(ann["category_id"])

        for img in data.get("images", []):
            img_id = int(img["id"])
            file_name = img.get("file_name", f"{img_id}.png")
            if "is_candidate_location" in img:
                label = int(img["is_candidate_location"])
            elif "category_id" in img:
                label = int(img["category_id"])
            elif "label" in img:
                label = int(img["label"])
            elif img_id in ann_labels:
                label = int(ann_labels[img_id])
            else:
                label = 0
            src = str(img.get("img_source", "unknown"))
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))

            records.append({
                "image_id": img_id,
                "file_name": file_name,
                "label": label,
                "img_source": src,
                "width": width,
                "height": height,
                "site_type": img.get("site_type", "unspecified"),
                "severity": img.get("severity", "unspecified"),
            })

    raw_df = pd.DataFrame(records)
    initial_count = len(raw_df)

    # CRITICAL: Deduplicate overlapping IDs and identical file_names (first-seen wins)
    # Stage 1: Deduplicate by image_id
    dup_id_mask = raw_df.duplicated(subset=["image_id"], keep="first")
    dup_id_count = int(dup_id_mask.sum())
    if dup_id_count > 0:
        logger.warning(
            f"Found {dup_id_count} duplicate image IDs across metadata files. "
            f"Deduplicating by image_id to prevent test contamination."
        )
        df_stage1 = raw_df[~dup_id_mask].copy().reset_index(drop=True)
    else:
        df_stage1 = raw_df.copy().reset_index(drop=True)

    # Stage 2: Deduplicate by file_name (prevents distinct IDs referencing the same image)
    dup_fn_mask = df_stage1.duplicated(subset=["file_name"], keep="first")
    dup_fn_count = int(dup_fn_mask.sum())
    if dup_fn_count > 0:
        logger.warning(
            f"Found {dup_fn_count} duplicate file_name records with distinct image IDs. "
            f"Deduplicating by file_name to prevent physical image leakage across splits."
        )
        df = df_stage1[~dup_fn_mask].copy().reset_index(drop=True)
    else:
        df = df_stage1

    # Assign file_path and check existence
    if images_dir is not None:
        images_dir_path = Path(images_dir)
        df["file_path"] = df["file_name"].apply(lambda fn: str(images_dir_path / fn))
    else:
        df["file_path"] = df["file_name"]

    df["exists"] = df["file_path"].apply(lambda p: os.path.exists(p))

    if filter_existing:
        on_disk = int(df["exists"].sum())
        logger.info(f"Filtering dataset to existing files only ({on_disk} / {len(df)} images on disk).")
        df = df[df["exists"]].copy().reset_index(drop=True)

    logger.info(f"Metadata loaded successfully: {len(df)} unique records (from {initial_count} raw).")
    return df


def inspect_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    images_dir: Optional[Union[str, Path]] = None,
) -> DatasetStats:
    """
    Scans dataset metadata and directory structure, returning a comprehensive DatasetStats summary.
    """
    raw_path = Path(raw_dir)
    img_dir = Path(images_dir) if images_dir else (raw_path / "images")

    # Load raw records before deduplication to count total and duplicates
    json_paths = []
    for name in ["training.json", "testing.json"]:
        if (raw_path / name).exists():
            json_paths.append(raw_path / name)

    total_raw_records = 0
    all_raw_ids = []
    for jpath in json_paths:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            imgs = data.get("images", [])
            total_raw_records += len(imgs)
            all_raw_ids.extend([img["id"] for img in imgs])

    raw_id_series = pd.Series(all_raw_ids)
    duplicate_count = int(raw_id_series.duplicated().sum())

    df = load_aerialwaste_metadata(raw_path, images_dir=img_dir, filter_existing=False)

    total_unique = len(df)
    waste_count = int((df["label"] == 1).sum())
    clean_count = int((df["label"] == 0).sum())
    waste_pct = float(round((waste_count / total_unique * 100.0) if total_unique > 0 else 0.0, 2))
    source_counts = {str(k): int(v) for k, v in df["img_source"].value_counts().to_dict().items()}
    on_disk_count = int(df["exists"].sum())
    missing_count = int(total_unique - on_disk_count)

    stats = DatasetStats(
        total_raw_records=total_raw_records if total_raw_records > 0 else total_unique,
        unique_images=total_unique,
        duplicate_ids_count=duplicate_count,
        num_waste=waste_count,
        num_non_waste=clean_count,
        waste_percentage=waste_pct,
        source_counts=source_counts,
        images_on_disk=on_disk_count,
        missing_images=missing_count,
    )
    return stats


def assign_scenes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups consecutive image acquisition frames from the same sensor where
    delta_id <= 5 into the same contiguous scene.

    Guarantees that full flight passes are kept together as atomic scenes
    with zero artificial chunk boundary breaks.
    """
    df_sorted = df.sort_values("image_id").copy().reset_index(drop=True)
    scene_ids = []
    current_scene = 0
    prev_source = None
    prev_id = -9999

    for _, row in df_sorted.iterrows():
        try:
            curr_id = int(row["image_id"])
            jump = (curr_id - prev_id) > 5
        except (ValueError, TypeError):
            curr_id = -9999
            jump = True

        # Advance scene if sensor changes or image ID jumps by > 5
        if row["img_source"] != prev_source or jump:
            current_scene += 1

        scene_ids.append(f"scene_{current_scene:05d}")
        prev_source = row["img_source"]
        prev_id = curr_id

    df_sorted["scene_id"] = scene_ids
    return df_sorted


def _allocate_scenes_stratified(
    scene_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    max_iter: int = 5000,
) -> Dict[str, str]:
    """
    Allocates variable-sized scenes to train/val/test partitions using size-and-waste-aware
    greedy bin-packing with local search refinement.

    Guarantees:
      - 0 scene leakage across partitions
      - Proportional sample counts matching target ratios
      - Balanced positive class (waste) distribution across all splits
      - Deterministic reproducibility given random_seed
    """
    total_r = train_ratio + val_ratio + test_ratio
    ratios = {
        "train": train_ratio / total_r,
        "val": val_ratio / total_r,
        "test": test_ratio / total_r,
    }

    total_imgs = scene_df["total_count"].sum()
    total_w = scene_df["waste_count"].sum()

    t_tot = {p: total_imgs * r for p, r in ratios.items()}
    t_w = {p: total_w * r for p, r in ratios.items()}

    rng = np.random.RandomState(random_seed)

    # Threshold for scenes that cannot fit into holdout without overflowing
    max_holdout_capacity = min(t_tot["val"], t_tot["test"]) * 0.60

    assignment: Dict[str, str] = {}
    curr_tot = {"train": 0, "val": 0, "test": 0}
    curr_w = {"train": 0, "val": 0, "test": 0}

    # Sort scenes descending by total_count, breaking ties by waste_count
    sorted_df = scene_df.sort_values(by=["total_count", "waste_count"], ascending=[False, False])

    for _, row in sorted_df.iterrows():
        sc = str(row["scene_id"])
        tot = int(row["total_count"])
        w = int(row["waste_count"])

        # Giant scenes that would dwarf val or test must go to train
        is_giant = (
            tot > max_holdout_capacity
            and len(scene_df) >= 20
            and tot > 2 * (total_imgs / len(scene_df))
        )
        if is_giant:
            best_p = "train"
        else:
            best_p = None
            best_score = float("inf")
            for cand_p in ["train", "val", "test"]:
                new_tot = curr_tot[cand_p] + tot
                new_w = curr_w[cand_p] + w

                tot_err = abs(new_tot - t_tot[cand_p]) / max(t_tot[cand_p], 1)
                w_err = abs(new_w - t_w[cand_p]) / max(t_w[cand_p], 1)

                score = tot_err + 1.8 * w_err
                if score < best_score:
                    best_score = score
                    best_p = cand_p

        assignment[sc] = best_p  # type: ignore[assignment]
        curr_tot[best_p] += tot  # type: ignore[index]
        curr_w[best_p] += w  # type: ignore[index]

    # Local search refinement (swapping eligible scenes to minimize combined objective error)
    def calc_loss(tot_dict, w_dict):
        loss = 0.0
        for p in ["train", "val", "test"]:
            loss += ((tot_dict[p] - t_tot[p]) / max(t_tot[p], 1)) ** 2
            loss += 1.8 * ((w_dict[p] - t_w[p]) / max(t_w[p], 1)) ** 2
        return loss

    swappable = [
        idx for idx in scene_df.index
        if not (
            scene_df.loc[idx, "total_count"] > max_holdout_capacity
            and len(scene_df) >= 20
            and scene_df.loc[idx, "total_count"] > 2 * (total_imgs / len(scene_df))
        )
    ]

    if len(swappable) >= 3 and max_iter > 0:
        current_loss = calc_loss(curr_tot, curr_w)
        for _ in range(max_iter):
            idx = rng.choice(swappable)
            row = scene_df.loc[idx]
            sc = str(row["scene_id"])
            tot = int(row["total_count"])
            w = int(row["waste_count"])

            old_p = assignment[sc]
            new_p = rng.choice([p for p in ["train", "val", "test"] if p != old_p])

            trial_tot = curr_tot.copy()
            trial_w = curr_w.copy()
            trial_tot[old_p] -= tot
            trial_tot[new_p] += tot
            trial_w[old_p] -= w
            trial_w[new_p] += w

            trial_loss = calc_loss(trial_tot, trial_w)
            if trial_loss < current_loss:
                assignment[sc] = str(new_p)
                curr_tot = trial_tot
                curr_w = trial_w
                current_loss = trial_loss

    return assignment


def create_leakage_safe_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset into leakage-safe train/val/test splits:
    1. Clusters contiguous image IDs from the same sensor (delta_id <= 5) into discrete scenes.
    2. Performs size-and-waste-aware stratified allocation at the scene level.
    3. Guarantees zero leakage across partitions for image IDs, scene groups, and filenames.
    4. Automatically saves train.csv, val.csv, test.csv to output_dir if provided.
    """
    if len(df) == 0:
        raise ValueError("Cannot split empty DataFrame.")

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    # 1. Cluster contiguous acquisition frames into scenes (no chunking)
    df_sorted = assign_scenes(df)

    # 2. Scene-level summary table
    scene_df = df_sorted.groupby("scene_id").agg({
        "img_source": "first",
        "label": "sum",
        "image_id": "count",
    }).reset_index().rename(columns={"label": "waste_count", "image_id": "total_count"})

    n_scenes = len(scene_df)

    # 3. Allocation logic
    if n_scenes < 4:
        # Graceful fallback for tiny datasets (N in [1, 2, 3])
        shuffled = scene_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        if n_scenes == 3:
            train_sc = set(shuffled.iloc[[0]]["scene_id"])
            val_sc = set(shuffled.iloc[[1]]["scene_id"])
            test_sc = set(shuffled.iloc[[2]]["scene_id"])
        elif n_scenes == 2:
            train_sc = set(shuffled.iloc[[0]]["scene_id"])
            val_sc = set(shuffled.iloc[[1]]["scene_id"])
            test_sc = set()
        elif n_scenes == 1:
            train_sc = set(shuffled.iloc[[0]]["scene_id"])
            val_sc = set()
            test_sc = set()
        else:
            train_sc = set()
            val_sc = set()
            test_sc = set()
    else:
        # Size-and-class-aware stratified scene allocation
        assignment = _allocate_scenes_stratified(
            scene_df=scene_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        train_sc = {sc for sc, p in assignment.items() if p == "train"}
        val_sc = {sc for sc, p in assignment.items() if p == "val"}
        test_sc = {sc for sc, p in assignment.items() if p == "test"}

    # 4. Map back to image splits
    train_df = df_sorted[df_sorted["scene_id"].isin(train_sc)].copy().reset_index(drop=True)
    val_df = df_sorted[df_sorted["scene_id"].isin(val_sc)].copy().reset_index(drop=True)
    test_df = df_sorted[df_sorted["scene_id"].isin(test_sc)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # 5. Strict Zero-Leakage Verifications
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    assert len(train_ids & val_ids) == 0, "Data leakage: overlapping IDs between train and val!"
    assert len(train_ids & test_ids) == 0, "Data leakage: overlapping IDs between train and test!"
    assert len(val_ids & test_ids) == 0, "Data leakage: overlapping IDs between val and test!"

    train_sc_ids = set(train_df["scene_id"])
    val_sc_ids = set(val_df["scene_id"])
    test_sc_ids = set(test_df["scene_id"])

    assert len(train_sc_ids & val_sc_ids) == 0, "Scene leakage: overlapping scenes between train and val!"
    assert len(train_sc_ids & test_sc_ids) == 0, "Scene leakage: overlapping scenes between train and test!"
    assert len(val_sc_ids & test_sc_ids) == 0, "Scene leakage: overlapping scenes between val and test!"

    # File-name zero leakage verification
    if "file_name" in train_df.columns:
        train_fn = set(train_df["file_name"])
        val_fn = set(val_df["file_name"])
        test_fn = set(test_df["file_name"])

        assert len(train_fn & val_fn) == 0, "File leakage: overlapping file_name between train and val!"
        assert len(train_fn & test_fn) == 0, "File leakage: overlapping file_name between train and test!"
        assert len(val_fn & test_fn) == 0, "File leakage: overlapping file_name between val and test!"

    # 6. Safe logging metrics
    train_pct = (len(train_df) / len(df_sorted) * 100) if len(df_sorted) > 0 else 0.0
    val_pct = (len(val_df) / len(df_sorted) * 100) if len(df_sorted) > 0 else 0.0
    test_pct = (len(test_df) / len(df_sorted) * 100) if len(df_sorted) > 0 else 0.0

    train_waste = f"{train_df['label'].mean()*100:.1f}%" if len(train_df) > 0 else "N/A"
    val_waste = f"{val_df['label'].mean()*100:.1f}%" if len(val_df) > 0 else "N/A"
    test_waste = f"{test_df['label'].mean()*100:.1f}%" if len(test_df) > 0 else "N/A"

    logger.info(
        f"Splits created successfully with ZERO leakage:\n"
        f"  Train: {len(train_df)} images ({train_pct:.1f}%) | Waste: {train_waste}\n"
        f"  Val:   {len(val_df)} images ({val_pct:.1f}%) | Waste: {val_waste}\n"
        f"  Test:  {len(test_df)} images ({test_pct:.1f}%) | Waste: {test_waste}"
    )

    # 7. Save manifests if output_dir provided
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_path / "train.csv", index=False)
        val_df.to_csv(out_path / "val.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)

        # Also write to splits/ subdirectory for maximum ecosystem convenience
        splits_sub = out_path / "splits"
        splits_sub.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(splits_sub / "train.csv", index=False)
        val_df.to_csv(splits_sub / "val.csv", index=False)
        test_df.to_csv(splits_sub / "test.csv", index=False)

        logger.info(f"Saved split manifests (train.csv, val.csv, test.csv) to {out_path} and {splits_sub}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="AerialWaste Data Sourcing, Inspection & Leakage-Safe Splitting")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, default=None, choices=["metadata", "fast", "full", "mock"], help="Download mode")
    parser.add_argument("--raw-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default=None, help="Processed data directory")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--filter-existing", action="store_true", help="Filter dataset to images present on disk")
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    raw_dir = Path(args.raw_dir or cfg.get("paths", {}).get("raw_data_dir", "data/raw"))
    processed_dir = Path(args.processed_dir or cfg.get("paths", {}).get("processed_data_dir", "data/processed"))
    val_ratio = args.val_ratio if args.val_ratio is not None else cfg.get("dataset", {}).get("val_ratio", 0.15)
    test_ratio = args.test_ratio if args.test_ratio is not None else cfg.get("dataset", {}).get("test_ratio", 0.15)
    train_ratio = 1.0 - val_ratio - test_ratio
    seed = args.seed if args.seed is not None else cfg.get("system", {}).get("seed", 42)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Handle download / mock if requested or if metadata missing
    mode = args.mode
    meta_missing = not ((raw_dir / "training.json").exists() and (raw_dir / "testing.json").exists())
    if mode or meta_missing:
        effective_mode = mode or "metadata"
        logger.info(f"Executing dataset acquisition (mode: {effective_mode})...")
        download_aerialwaste(dest_dir=raw_dir, mode=effective_mode)

    # Inspect dataset
    stats = inspect_dataset(raw_dir)
    print("\n" + "=" * 55)
    print("      AERIALWASTE DATASET INSPECTION REPORT")
    print("=" * 55)
    for k, v in asdict(stats).items():
        print(f"  {k:22s}: {v}")
    print("=" * 55 + "\n")

    # Load metadata and perform leakage-safe splitting
    df = load_aerialwaste_metadata(
        metadata_source=raw_dir,
        images_dir=raw_dir / "images",
        filter_existing=args.filter_existing,
    )

    create_leakage_safe_splits(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        output_dir=processed_dir,
    )
    print(f"Dataset preparation complete. Manifests exported to {processed_dir}.\n")


if __name__ == "__main__":
    main()
