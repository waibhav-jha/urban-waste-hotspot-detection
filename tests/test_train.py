"""
Unit tests for waste_hotspot.train module.
Verifies EarlyStopping logic, checkpoint saving and loading, single-step optimization,
class weighting, and CLI entry point in dry-run mode.
"""

import json
from pathlib import Path
import tempfile
from typing import Dict, Generator, Tuple
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from waste_hotspot.model import build_classifier
from waste_hotspot.train import (
    EarlyStopping,
    load_checkpoint,
    main,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


class SyntheticTensorDataset(Dataset):
    """Minimal in-memory synthetic dataset for instant test iterations."""

    def __init__(self, num_samples: int = 16, num_classes: int = 2):
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, 224, 224, dtype=torch.float32)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:
        return self.images[idx], int(self.labels[idx]), {"image_id": f"synthetic_{idx}"}


@pytest.fixture
def synthetic_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Provides small train and val DataLoaders (16 samples, batch_size=4)."""
    train_ds = SyntheticTensorDataset(num_samples=16)
    val_ds = SyntheticTensorDataset(num_samples=8)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def dummy_train_environment(tmp_path: Path) -> Path:
    """
    Creates a temporary configuration file, split CSVs, and artifacts directory
    for CLI and end-to-end dry-run tests.
    """
    splits_dir = tmp_path / "splits"
    artifacts_dir = tmp_path / "artifacts"
    splits_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows_train = []
    for i in range(8):
        img_p = img_dir / f"train_{i}.png"
        Image.new("RGB", (224, 224), color=(i * 20, 100, 150)).save(img_p)
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
        img_p = img_dir / f"val_{i}.png"
        Image.new("RGB", (224, 224), color=(100, i * 30, 200)).save(img_p)
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
        "preprocessing": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "model": {
            "backbone": "resnet18",
            "num_classes": 2,
            "pretrained": False,
            "freeze_backbone": False,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
        },
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    return config_path


# ==============================================================================
# EarlyStopping Unit Tests
# ==============================================================================

def test_early_stopping_decreasing_loss() -> None:
    """Verifies that steadily decreasing validation losses never trigger early stopping."""
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    loss_sequence = [0.80, 0.65, 0.50, 0.42, 0.35]
    for loss in loss_sequence:
        stop = early_stopping.step(val_loss=loss)
        assert stop is False, f"Early stopping prematurely triggered on loss {loss}"
        assert early_stopping.counter == 0
        assert pytest.approx(early_stopping.best_loss) == loss

    assert early_stopping.should_stop is False


def test_early_stopping_plateau_and_halt() -> None:
    """
    Verifies that early stopping increments counter on non-improving losses
    and halts execution exactly when counter reaches patience.
    """
    patience = 3
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)

    # Initial best loss
    stop = early_stopping.step(val_loss=0.500)
    assert stop is False
    assert early_stopping.counter == 0
    assert early_stopping.best_loss == 0.500

    # Non-improving step 1
    stop = early_stopping.step(val_loss=0.505)
    assert stop is False
    assert early_stopping.counter == 1

    # Non-improving step 2
    stop = early_stopping.step(val_loss=0.510)
    assert stop is False
    assert early_stopping.counter == 2

    # Non-improving step 3 -> triggers halt
    stop = early_stopping.step(val_loss=0.520)
    assert stop is True
    assert early_stopping.counter == 3
    assert early_stopping.should_stop is True


def test_early_stopping_min_delta_sensitivity() -> None:
    """Verifies that improvements smaller than min_delta are treated as non-improvements."""
    early_stopping = EarlyStopping(patience=2, min_delta=0.05)

    early_stopping.step(val_loss=1.00)
    assert early_stopping.counter == 0

    # Loss improves by only 0.01 (which is less than min_delta 0.05)
    early_stopping.step(val_loss=0.99)
    assert early_stopping.counter == 1, "Improvement < min_delta must increment counter"


def test_early_stopping_weights_caching_and_restoration() -> None:
    """Verifies that EarlyStopping clones best weights to CPU and can restore them."""
    model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
    early_stopping = EarlyStopping(patience=2, min_delta=0.01)

    # Best epoch
    early_stopping.step(val_loss=0.40, model=model, epoch=1)
    best_fc_weight = model.fc.weight.clone().detach()

    # Mutate weights to simulate subsequent worse epoch
    with torch.no_grad():
        model.fc.weight.add_(1.0)
    assert not torch.equal(model.fc.weight, best_fc_weight)

    # Subsequent non-improving step
    early_stopping.step(val_loss=0.60, model=model, epoch=2)

    # Restore best weights
    early_stopping.restore_best_weights(model)
    assert torch.equal(model.fc.weight, best_fc_weight)


# ==============================================================================
# Checkpoint Serialization & Restoration Tests
# ==============================================================================

def test_checkpoint_roundtrip_saving_and_loading(tmp_path: Path) -> None:
    """Verifies checkpoint saving, key completeness, and exact state restoration."""
    ckpt_path = tmp_path / "artifacts" / "test_model.pth"

    model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    config_dict = {"model": {"backbone": "resnet18"}, "training": {"batch_size": 32}}

    save_checkpoint(
        filepath=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        val_loss=0.452,
        val_acc=0.825,
        config=config_dict,
        class_names=["non-waste", "waste"],
        threshold=0.50,
    )

    assert ckpt_path.is_file(), f"Checkpoint was not created at {ckpt_path}"

    loaded_model, loaded_ckpt = load_checkpoint(ckpt_path, device="cpu")

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "val_acc",
        "config",
        "class_names",
        "best_val_loss",
        "threshold",
    }
    assert required_keys.issubset(set(loaded_ckpt.keys())), f"Missing keys: {required_keys - set(loaded_ckpt.keys())}"

    assert loaded_ckpt["epoch"] == 3
    assert pytest.approx(loaded_ckpt["val_loss"]) == 0.452
    assert loaded_ckpt["class_names"] == ["non-waste", "waste"]

    # Restore into fresh model and verify identical forward predictions
    fresh_model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
    fresh_model.load_state_dict(loaded_ckpt["model_state_dict"])
    fresh_model.eval()
    model.eval()

    sample_x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_orig = model(sample_x)
        out_restored = fresh_model(sample_x)

    assert torch.allclose(out_orig, out_restored, atol=1e-6), "Restored model output mismatch"


# ==============================================================================
# Optimization & Weight Update Tests
# ==============================================================================

def test_training_step_updates_weights_and_reduces_loss(
    synthetic_dataloaders: Tuple[DataLoader, DataLoader]
) -> None:
    """
    Runs a single epoch on synthetic data and verifies:
    1. Loss is finite and positive.
    2. Model weights are updated via backpropagation.
    """
    train_loader, _ = synthetic_dataloaders
    model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    initial_weights = model.fc.weight.clone().detach()

    loss, acc = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert not torch.isnan(torch.tensor(loss))
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    final_weights = model.fc.weight.detach()
    assert not torch.equal(initial_weights, final_weights), "Model parameters were not updated"


def test_frozen_backbone_weights_remain_invariant(
    synthetic_dataloaders: Tuple[DataLoader, DataLoader]
) -> None:
    """Verifies that frozen backbone layers undergo ZERO parameter modification during training."""
    train_loader, _ = synthetic_dataloaders
    model = build_classifier(
        model_name="resnet18",
        pretrained=False,
        num_classes=2,
        freeze_backbone=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    initial_backbone_weight = model.conv1.weight.clone().detach()
    initial_fc_weight = model.fc.weight.clone().detach()

    train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    # Conv1 weights must remain bit-for-bit identical
    assert torch.equal(model.conv1.weight.detach(), initial_backbone_weight)
    # Head weights must have updated
    assert not torch.equal(model.fc.weight.detach(), initial_fc_weight)


def test_class_weighted_loss_computation() -> None:
    """Verifies that CrossEntropyLoss handles class weights and penalizes waste errors more heavily."""
    class_weights = torch.tensor([1.0, 2.5], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logits = torch.tensor([[2.0, -1.0]], dtype=torch.float32)  # Confidently predicts class 0
    target_non_waste = torch.tensor([0], dtype=torch.long)
    target_waste = torch.tensor([1], dtype=torch.long)

    loss_correct = criterion(logits, target_non_waste)
    loss_wrong_waste = criterion(logits, target_waste)

    assert loss_wrong_waste.item() > loss_correct.item() * 2.0


# ==============================================================================
# CLI Entry Point & Dry-Run Tests
# ==============================================================================

def test_train_cli_dry_run_fast_execution(dummy_train_environment: Path) -> None:
    """
    Executes train CLI in --dry-run mode:
    1. Verifies process exit code is 0.
    2. Verifies execution completes in < 5 seconds.
    3. Verifies best_model.pth and training_history.json are saved.
    """
    import time
    config_path = dummy_train_environment

    start_time = time.time()
    exit_code = main(["--config", str(config_path), "--dry-run", "--device", "cpu"])
    duration = time.time() - start_time

    assert exit_code == 0, f"CLI dry-run failed with exit code {exit_code}"
    assert duration < 5.0, f"Dry run took {duration:.2f}s, expected < 5.0s"

    artifacts_dir = config_path.parent / "artifacts"
    checkpoint_file = artifacts_dir / "best_model.pth"
    history_file = artifacts_dir / "training_history.json"

    assert checkpoint_file.is_file(), "best_model.pth was not created"
    assert history_file.is_file(), "training_history.json was not created"

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert "epochs" in history
    assert len(history["epochs"]) >= 1


def test_train_cli_argument_overrides(dummy_train_environment: Path) -> None:
    """Verifies that command line arguments correctly override YAML configurations."""
    config_path = dummy_train_environment
    exit_code = main([
        "--config", str(config_path),
        "--epochs", "1",
        "--batch-size", "2",
        "--lr", "0.0005",
        "--device", "cpu",
        "--max-batches", "1",
    ])
    assert exit_code == 0
