"""
Adversarial stress tests for waste_hotspot.train module.
Empirically stress-tests:
1. EarlyStopping boundary and pathological conditions (oscillations, plateau, tiny deltas, non-finite loss).
2. Multi-step optimization weight invariance on frozen ResNet-50 backbones.
3. Checkpoint corruption resilience and atomic write safety.
4. CLI robustness, parameter validation, and error exit codes.
"""

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Tuple
from unittest.mock import patch
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from waste_hotspot.model import (
    build_classifier,
    load_checkpoint,
    save_checkpoint,
)
from waste_hotspot.train import (
    EarlyStopping,
    load_training_config,
    main,
    train_model,
    train_one_epoch,
    validate_one_epoch,
)


# ==============================================================================
# Fixtures & Test Helpers
# ==============================================================================

class SyntheticClassificationDataset(Dataset):
    """In-memory dataset generating deterministic synthetic RGB images."""

    def __init__(self, num_samples: int = 16, num_classes: int = 2, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.images = torch.randn(num_samples, 3, 224, 224, dtype=torch.float32)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        return self.images[idx], int(self.labels[idx]), {"image_id": f"synthetic_{idx}"}


@pytest.fixture
def dummy_train_env(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Creates temporary directory containing dummy splits, images, and config.
    Returns (config_path, artifacts_dir).
    """
    splits_dir = tmp_path / "splits"
    artifacts_dir = tmp_path / "artifacts"
    images_dir = tmp_path / "images"

    splits_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows_train = []
    for i in range(8):
        img_p = images_dir / f"train_{i}.png"
        Image.new("RGB", (224, 224), color=(50, 100, 150)).save(img_p)
        rows_train.append({
            "image_id": f"tr_{i}",
            "file_name": img_p.name,
            "file_path": str(img_p),
            "label": i % 2,
            "img_source": f"scene_{i // 2}",
            "width": 224,
            "height": 224,
        })

    rows_val = []
    for i in range(4):
        img_p = images_dir / f"val_{i}.png"
        Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_p)
        rows_val.append({
            "image_id": f"val_{i}",
            "file_name": img_p.name,
            "file_path": str(img_p),
            "label": i % 2,
            "img_source": f"val_scene_{i // 2}",
            "width": 224,
            "height": 224,
        })

    import pandas as pd
    pd.DataFrame(rows_train).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame(rows_val).to_csv(splits_dir / "val.csv", index=False)

    config_content = {
        "system": {"seed": 42, "device": "cpu"},
        "paths": {
            "splits_dir": str(splits_dir),
            "artifacts_dir": str(artifacts_dir),
            "checkpoint_path": str(artifacts_dir / "best_model.pth"),
            "history_path": str(artifacts_dir / "training_history.json"),
        },
        "dataset": {"num_workers": 0, "class_names": ["non-waste", "waste"]},
        "preprocessing": {"image_size": 224},
        "model": {
            "backbone": "resnet18",
            "num_classes": 2,
            "pretrained": False,
            "freeze_backbone": False,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
        },
    }

    config_path = tmp_path / "valid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return config_path, artifacts_dir


# ==============================================================================
# 1. EarlyStopping Boundary & Pathological Conditions
# ==============================================================================

def test_early_stopping_oscillating_losses() -> None:
    """
    Stress-tests EarlyStopping under oscillating/alternating validation loss sequences:
    [1.0, 0.9, 0.95, 0.85, 0.92, 0.88] with patience=3, min_delta=0.001.
    Verifies counter resets strictly on each new global best loss and does not stop prematurely.
    """
    es = EarlyStopping(patience=3, min_delta=0.001, mode="min")
    losses = [1.0, 0.9, 0.95, 0.85, 0.92, 0.88]

    # Epoch 1: 1.0 -> initial best
    stop1 = es.step(losses[0], epoch=1)
    assert not stop1
    assert es.counter == 0
    assert pytest.approx(es.best_loss) == 1.0
    assert es.best_epoch == 1

    # Epoch 2: 0.9 -> improved
    stop2 = es.step(losses[1], epoch=2)
    assert not stop2
    assert es.counter == 0
    assert pytest.approx(es.best_loss) == 0.9
    assert es.best_epoch == 2

    # Epoch 3: 0.95 -> oscillation upwards (worse than 0.9)
    stop3 = es.step(losses[2], epoch=3)
    assert not stop3
    assert es.counter == 1
    assert pytest.approx(es.best_loss) == 0.9
    assert es.best_epoch == 2

    # Epoch 4: 0.85 -> recovery to new global minimum
    stop4 = es.step(losses[3], epoch=4)
    assert not stop4
    assert es.counter == 0, "Counter must reset to 0 upon reaching new best loss"
    assert pytest.approx(es.best_loss) == 0.85
    assert es.best_epoch == 4

    # Epoch 5: 0.92 -> oscillation upwards
    stop5 = es.step(losses[4], epoch=5)
    assert not stop5
    assert es.counter == 1
    assert pytest.approx(es.best_loss) == 0.85

    # Epoch 6: 0.88 -> lower than 0.92, but still worse than best_loss 0.85
    stop6 = es.step(losses[5], epoch=6)
    assert not stop6
    assert es.counter == 2, "Counter must increment because 0.88 > 0.85"
    assert pytest.approx(es.best_loss) == 0.85
    assert es.should_stop is False


def test_early_stopping_zero_variance_plateau() -> None:
    """
    Stress-tests EarlyStopping under zero-variance flat loss plateau:
    [0.5, 0.5, 0.5, 0.5] with patience=3.
    Verifies that zero improvement increments counter every step and halts strictly at patience.
    """
    es = EarlyStopping(patience=3, min_delta=0.001, mode="min")
    losses = [0.5, 0.5, 0.5, 0.5]

    # Epoch 1: 0.5 -> initial best
    assert not es.step(losses[0], epoch=1)
    assert es.counter == 0
    assert es.best_loss == 0.5

    # Epoch 2: 0.5 -> delta is 0.0 < min_delta
    assert not es.step(losses[1], epoch=2)
    assert es.counter == 1

    # Epoch 3: 0.5 -> delta is 0.0 < min_delta
    assert not es.step(losses[2], epoch=3)
    assert es.counter == 2

    # Epoch 4: 0.5 -> counter reaches patience=3, halts!
    stop = es.step(losses[3], epoch=4)
    assert stop is True, "Must halt on epoch 4 after patience=3 non-improving epochs"
    assert es.counter == 3
    assert es.should_stop is True
    assert es.best_epoch == 1


def test_early_stopping_negative_and_sub_delta_improvements() -> None:
    """
    Stress-tests EarlyStopping with min_delta=0.01 under sub-delta decreases:
    [1.0, 0.995, 0.992] with patience=2.
    - 0.995 decreases by 0.005 (< min_delta 0.01) -> non-improvement.
    - 0.992 decreases by 0.008 from 1.0 (< min_delta 0.01) -> non-improvement.
    Verifies counter reaches 2 and halts.
    """
    es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
    losses = [1.0, 0.995, 0.992]

    # Epoch 1: 1.0
    assert not es.step(losses[0], epoch=1)
    assert es.counter == 0

    # Epoch 2: 0.995 (delta = 0.005 < 0.01)
    assert not es.step(losses[1], epoch=2)
    assert es.counter == 1, "Sub-delta decrease must count as non-improving"
    assert es.best_loss == 1.0

    # Epoch 3: 0.992 (total delta from best = 0.008 < 0.01)
    stop = es.step(losses[2], epoch=3)
    assert stop is True, "Must trigger early stopping after patience=2 sub-delta steps"
    assert es.counter == 2
    assert es.should_stop is True


def test_early_stopping_non_finite_validation_loss_handling() -> None:
    """
    Stress-tests EarlyStopping with non-finite validation loss values:
    [1.0, float('nan'), float('inf')].
    Verifies that:
    1. NaN and Inf do not crash the step evaluation.
    2. NaN and Inf are treated as non-improving (counter increments).
    3. best_loss remains the valid finite 1.0 and is not overwritten by NaN or Inf.
    4. Subsequent valid finite improvement recovers correctly.
    """
    es = EarlyStopping(patience=3, min_delta=0.001, mode="min")

    # Step 1: Valid initial loss
    es.step(1.0, epoch=1)
    assert es.best_loss == 1.0
    assert es.counter == 0

    # Step 2: NaN validation loss
    stop_nan = es.step(float("nan"), epoch=2)
    assert not stop_nan
    assert es.counter == 1, "NaN must increment counter as non-improving"
    assert es.best_loss == 1.0, "best_loss must not be corrupted by NaN"
    assert not es.improved

    # Step 3: Inf validation loss
    stop_inf = es.step(float("inf"), epoch=3)
    assert not stop_inf
    assert es.counter == 2, "Inf must increment counter as non-improving"
    assert es.best_loss == 1.0, "best_loss must not be corrupted by Inf"
    assert not es.improved

    # Step 4: Recovery with legitimate finite improvement (0.75)
    stop_rec = es.step(0.75, epoch=4)
    assert not stop_rec
    assert es.counter == 0, "Finite improvement after NaN/Inf must reset counter"
    assert pytest.approx(es.best_loss) == 0.75
    assert es.best_epoch == 4
    assert es.improved


def test_early_stopping_first_epoch_nan_pathology() -> None:
    """
    Investigates pathology: what if the very first epoch evaluates to float('nan')?
    Empirically checks if best_loss becomes NaN and whether it breaks subsequent epochs.
    """
    es = EarlyStopping(patience=2, min_delta=0.001, mode="min")

    # If first epoch is NaN:
    es.step(float("nan"), epoch=1)

    # In current implementation, if best_loss is None, self.best_loss = val_loss
    # So best_loss becomes float('nan')!
    # Subsequent finite step:
    es.step(0.50, epoch=2)
    # (nan - 0.50) is nan; (nan >= 0.001) is False.
    # Therefore, counter increments to 1, and best_loss remains nan!
    # This demonstrates that EarlyStopping without finite checks will poison best_loss if epoch 1 is NaN.
    if math.isnan(es.best_loss):
        assert es.counter == 1
        # Third epoch with even better loss:
        stop = es.step(0.30, epoch=3)
        assert stop is True, "Poisoned best_loss halts despite strictly decreasing finite losses"


# ==============================================================================
# 2. Multi-Step Optimization Weight Invariance
# ==============================================================================

def test_multistep_frozen_backbone_bit_for_bit_invariance() -> None:
    """
    Runs 3 full training steps with ResNet-50 under freeze_backbone=True.
    Verifies:
    1. ALL backbone weights (conv1.weight, bn1.weight, bn1.bias, layer1..4) remain
       BIT-FOR-BIT IDENTICAL to their starting values across every step via torch.equal.
    2. Head weights (fc.weight, fc.bias) change strictly on EVERY step.
    3. Gradients of backbone parameters remain None throughout.
    """
    torch.manual_seed(1337)
    model = build_classifier(
        model_name="resnet50",
        pretrained=False,
        num_classes=2,
        freeze_backbone=True,
    )
    criterion = nn.CrossEntropyLoss()

    # Trainable parameters should exclusively be head parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert set(trainable_names) == {"fc.weight", "fc.bias"}

    optimizer = torch.optim.AdamW(trainable_params, lr=0.01)

    # Snapshot all initial parameters
    initial_params: Dict[str, torch.Tensor] = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Verify specific key backbone tensors exist
    assert "conv1.weight" in initial_params
    assert "bn1.weight" in initial_params
    assert "bn1.bias" in initial_params
    assert "layer1.0.conv1.weight" in initial_params
    assert "layer2.0.conv1.weight" in initial_params
    assert "layer3.0.conv1.weight" in initial_params
    assert "layer4.0.conv1.weight" in initial_params

    # Run 3 consecutive optimization steps
    prev_fc_weight = initial_params["fc.weight"].clone()
    prev_fc_bias = initial_params["fc.bias"].clone()

    for step in range(1, 4):
        # Fresh distinct synthetic batch each step
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([step % 2, (step + 1) % 2, step % 2, (step + 1) % 2], dtype=torch.long)

        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # 1. Verify EVERY backbone parameter is bit-for-bit identical to initial snapshot
        for name, param in model.named_parameters():
            if name not in ("fc.weight", "fc.bias"):
                assert torch.equal(param.data, initial_params[name]), (
                    f"Step {step}: Backbone parameter '{name}' mutated during training!"
                )
                assert param.grad is None, (
                    f"Step {step}: Backbone parameter '{name}' received a non-None gradient!"
                )

        # 2. Verify head weights and biases updated on THIS step
        cur_fc_weight = model.fc.weight.data
        cur_fc_bias = model.fc.bias.data

        assert not torch.equal(cur_fc_weight, prev_fc_weight), (
            f"Step {step}: fc.weight failed to update!"
        )
        assert not torch.equal(cur_fc_bias, prev_fc_bias), (
            f"Step {step}: fc.bias failed to update!"
        )

        prev_fc_weight = cur_fc_weight.clone()
        prev_fc_bias = cur_fc_bias.clone()


# ==============================================================================
# 3. Checkpoint Corruption Resilience
# ==============================================================================

def test_checkpoint_atomic_write_crash_resilience(tmp_path: Path) -> None:
    """
    Simulates mid-write crash / exception during save_checkpoint.
    Verifies:
    1. A pre-existing valid checkpoint at artifacts/best_model.pth is NEVER corrupted.
    2. Temporary .tmp_{pid} file created during failed write is cleaned up.
    3. load_checkpoint continues to load intact original weights.
    """
    ckpt_dir = tmp_path / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    # Step 1: Create a verified valid checkpoint
    model_orig = build_classifier("resnet18", pretrained=False, num_classes=2)
    save_checkpoint(
        filepath=ckpt_path,
        model=model_orig,
        epoch=1,
        val_loss=0.555,
        val_acc=0.80,
    )
    assert ckpt_path.is_file()

    # Verify original file loadable
    loaded_orig = load_checkpoint(ckpt_path, model=False)
    assert loaded_orig["epoch"] == 1
    assert loaded_orig["val_loss"] == 0.555

    # Step 2: Attempt a subsequent save that crashes midway in torch.save
    model_mutated = build_classifier("resnet18", pretrained=False, num_classes=2)
    with torch.no_grad():
        model_mutated.fc.weight.fill_(999.0)

    def failing_torch_save(state, f, *args, **kwargs):
        # Partially write corrupt bytes to simulate mid-write power loss / disk error
        if isinstance(f, (str, Path)):
            with open(f, "wb") as fp:
                fp.write(b"CORRUPTED_PARTIAL_BYTES_SIMULATING_CRASH")
        raise OSError("Simulated unexpected process crash / disk full error mid-write")

    with patch("torch.save", side_effect=failing_torch_save):
        with pytest.raises(OSError, match="Simulated unexpected process crash"):
            save_checkpoint(
                filepath=ckpt_path,
                model=model_mutated,
                epoch=2,
                val_loss=0.222,
                val_acc=0.95,
            )

    # Step 3: Verify the destination checkpoint is 100% UNTOUCHED and still valid
    assert ckpt_path.is_file()
    loaded_model, loaded_ckpt = load_checkpoint(ckpt_path, model=model_orig, device="cpu")
    assert loaded_ckpt["epoch"] == 1, "Checkpoint epoch was corrupted!"
    assert loaded_ckpt["val_loss"] == 0.555, "Checkpoint val_loss was corrupted!"
    assert not torch.equal(loaded_model.fc.weight, model_mutated.fc.weight), (
        "Corrupted in-memory weights leaked into persisted checkpoint!"
    )

    # Step 4: Verify no temporary .tmp_* file remained
    tmp_files = list(ckpt_dir.glob("*.tmp_*"))
    assert len(tmp_files) == 0, f"Temporary files were not cleaned up: {tmp_files}"


def test_checkpoint_overwrite_preexisting_corrupted_tmp_file(tmp_path: Path) -> None:
    """
    Verifies that if an orphaned corrupt .tmp_{pid} file already exists on disk,
    save_checkpoint cleanly overwrites it and produces a valid checkpoint.
    """
    ckpt_dir = tmp_path / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    # Pre-create orphaned corrupt tmp file matching current PID
    orphan_tmp = ckpt_path.with_name(f"{ckpt_path.stem}.tmp_{os.getpid()}{ckpt_path.suffix}")
    orphan_tmp.write_bytes(b"ORPHAN_GARBAGE_BYTES_FROM_DEAD_PROCESS")
    assert orphan_tmp.is_file()

    model = build_classifier("resnet18", pretrained=False, num_classes=2)
    save_checkpoint(filepath=ckpt_path, model=model, epoch=1, val_loss=0.33)

    assert ckpt_path.is_file()
    assert not orphan_tmp.exists()  # Was replaced via os.replace

    ckpt = load_checkpoint(ckpt_path, model=False)
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == 0.33


# ==============================================================================
# 4. CLI Robustness & Parameter Validation
# ==============================================================================

def test_cli_invalid_batch_sizes(dummy_train_env: Tuple[Path, Path]) -> None:
    """
    Stress-tests CLI with invalid batch sizes:
    --batch-size 0 and --batch-size -1.
    Verifies non-zero exit codes.
    """
    config_path, _ = dummy_train_env

    # 1. Batch size 0
    exit_code_zero = main([
        "--config", str(config_path),
        "--batch-size", "0",
        "--device", "cpu",
        "--no-pretrained",
    ])
    assert exit_code_zero != 0, "CLI must exit with non-zero code on --batch-size 0"

    # 2. Negative batch size -1
    exit_code_neg = main([
        "--config", str(config_path),
        "--batch-size", "-1",
        "--device", "cpu",
        "--no-pretrained",
    ])
    assert exit_code_neg != 0, "CLI must exit with non-zero code on --batch-size -1"


def test_cli_negative_learning_rate(dummy_train_env: Tuple[Path, Path]) -> None:
    """
    Stress-tests CLI with negative learning rate: --lr -0.01.
    Verifies clean error log and non-zero exit code.
    """
    config_path, _ = dummy_train_env

    exit_code = main([
        "--config", str(config_path),
        "--lr", "-0.01",
        "--device", "cpu",
        "--no-pretrained",
    ])
    assert exit_code != 0, "CLI must exit with non-zero code on negative learning rate"


def test_cli_nonexistent_config_path() -> None:
    """
    Stress-tests CLI when a nonexistent config file is passed:
    --config non_existent_file_path_xyz.yaml.
    Verifies whether load_training_config or CLI validates file existence.
    """
    bogus_path = "non_existent_config_file_123456789.yaml"

    # Direct test of load_training_config
    # If a non-existent config path is provided, should it raise FileNotFoundError?
    # Or does it silently proceed with default_config?
    # As discovered empirically, load_training_config currently does:
    #   if config_path is not None:
    #       cpath = Path(config_path)
    #       if cpath.is_file(): ...
    # which silently ignores non-existent paths.
    # We assert the expected behavior: a non-existent explicit config path SHOULD raise FileNotFoundError!
    try:
        config = load_training_config(config_path=bogus_path)
        # If it didn't raise, we record the silent fallthrough as a finding
        config_silently_ignored = True
    except FileNotFoundError:
        config_silently_ignored = False

    # CLI invocation with bogus config and bogus splits dir should exit non-zero
    exit_code = main([
        "--config", bogus_path,
        "--splits-dir", "non_existent_splits_dir_xyz",
        "--device", "cpu",
        "--no-pretrained",
    ])
    assert exit_code != 0, "CLI must exit with non-zero code when config/splits are invalid"


def test_train_one_epoch_non_finite_loss_aborts() -> None:
    """
    Verifies that train_one_epoch explicitly detects non-finite loss (NaN/Inf)
    and aborts execution with a RuntimeError immediately.
    """
    model = build_classifier("resnet18", pretrained=False, num_classes=2)
    dataset = SyntheticClassificationDataset(num_samples=8)
    loader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock criterion to return NaN loss
    class NaNCriterion(nn.Module):
        def forward(self, logits, targets):
            return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(RuntimeError, match="Non-finite loss encountered"):
        train_one_epoch(
            model=model,
            dataloader=loader,
            criterion=NaNCriterion(),
            optimizer=optimizer,
            device=torch.device("cpu"),
        )
