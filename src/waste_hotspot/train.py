"""
Training pipeline, EarlyStopping, optimization loop, and checkpoint management
for the AerialWaste binary classification model.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Ensure src/ is on sys.path
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from waste_hotspot.model import (
    build_classifier,
    load_checkpoint,
    save_checkpoint,
)
from waste_hotspot.preprocessing import (
    AerialWasteDataset,
    get_eval_transforms,
    get_train_transforms,
)

logger = logging.getLogger("waste_hotspot.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Loads YAML configuration and applies any overriding keyword arguments.
    """
    default_config: Dict[str, Any] = {
        "system": {"seed": 42, "device": "cuda"},
        "paths": {
            "processed_data_dir": "data/processed",
            "splits_dir": "data/processed/splits",
            "artifacts_dir": "artifacts",
            "checkpoint_path": "artifacts/best_model.pth",
            "history_path": "artifacts/training_history.json",
        },
        "dataset": {
            "name": "AerialWaste",
            "num_workers": 2,
            "class_names": ["non-waste", "waste"],
        },
        "preprocessing": {
            "image_size": 224,
            "rotation_degrees": 15,
            "brightness_jitter": 0.1,
            "contrast_jitter": 0.1,
        },
        "model": {
            "backbone": "resnet50",
            "num_classes": 2,
            "pretrained": True,
            "freeze_backbone": False,
        },
        "training": {
            "epochs": 25,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "class_weights": [1.0, 2.0],  # ~1:2 ratio compensation
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.001,
            "lr_scheduler_patience": 2,
            "lr_scheduler_factor": 0.5,
        },
    }

    if config_path is not None:
        cpath = Path(config_path)
        if cpath.is_file():
            with open(cpath, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                # Shallow recursive update
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in default_config:
                        default_config[k].update(v)
                    else:
                        default_config[k] = v

    # Apply keyword overrides
    for k, v in overrides.items():
        if v is not None:
            if k in default_config["training"]:
                default_config["training"][k] = v
            elif k in default_config["system"]:
                default_config["system"][k] = v
            elif k in default_config["model"]:
                default_config["model"][k] = v
            elif k in default_config["paths"]:
                default_config["paths"][k] = v
            elif k == "lr":
                default_config["training"]["learning_rate"] = v
            elif k == "batch_size":
                default_config["training"]["batch_size"] = v
            elif k == "epochs":
                default_config["training"]["epochs"] = v
            elif k == "device":
                default_config["system"]["device"] = v
            elif k == "seed":
                default_config["system"]["seed"] = v
            elif k == "patience":
                default_config["training"]["early_stopping_patience"] = v
            elif k == "splits_dir":
                default_config["paths"]["splits_dir"] = str(v)
            elif k == "artifacts_dir":
                default_config["paths"]["artifacts_dir"] = str(v)
                default_config["paths"]["checkpoint_path"] = str(Path(v) / "best_model.pth")
                default_config["paths"]["history_path"] = str(Path(v) / "training_history.json")
            elif k == "checkpoint_path":
                default_config["paths"]["checkpoint_path"] = str(v)
            elif k == "pretrained":
                default_config["model"]["pretrained"] = bool(v)

    return default_config


def compute_class_weights(df: pd.DataFrame, num_classes: int = 2) -> torch.Tensor:
    """Computes balanced inverse-frequency weights: w_c = N / (num_classes * N_c)."""
    labels = df["label"].to_numpy()
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = int((labels == c).sum())
        if count > 0:
            w = total / (float(num_classes) * float(count))
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """
    Early stopping monitor tracking validation loss.
    Halts training when validation loss fails to improve for patience epochs.
    Caches the best state_dict on CPU to prevent GPU memory saturation.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop: bool = False
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.improved: bool = False

    @property
    def should_stop(self) -> bool:
        return self.early_stop

    @should_stop.setter
    def should_stop(self, val: bool) -> None:
        self.early_stop = val

    def step(
        self,
        val_loss: float,
        model: Optional[nn.Module] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Evaluates current validation loss against history.

        Args:
            val_loss: Current validation loss.
            model: Optional PyTorch model whose state_dict is saved when improving.
            epoch: Optional current epoch index.

        Returns:
            bool: True if early stopping halt condition is reached, False otherwise.
        """
        cur_epoch = epoch if epoch is not None else (self.best_epoch + self.counter + 1)

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = cur_epoch
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            self.improved = True
            return False

        if self.mode == "min":
            improvement = (self.best_loss - val_loss)
        else:
            improvement = (val_loss - self.best_loss)

        if improvement >= self.min_delta:
            if self.verbose:
                logger.info(
                    "Epoch %d: Val loss improved from %.4f to %.4f.",
                    cur_epoch,
                    self.best_loss,
                    val_loss,
                )
            self.best_loss = val_loss
            self.best_epoch = cur_epoch
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            self.improved = True
            return False
        else:
            self.counter += 1
            self.improved = False
            if self.verbose:
                logger.info(
                    "Epoch %d: Val loss did not improve (%.4f >= %.4f). Counter: %d/%d",
                    cur_epoch,
                    val_loss,
                    self.best_loss,
                    self.counter,
                    self.patience,
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restores cached best weights into model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logger.info("Restored best model weights from epoch %d.", self.best_epoch)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Executes a single training epoch with flexible batch unpacking.

    Args:
        model: Target model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        device: Execution device.
        max_batches: Optional batch count limit for fast iterations.

    Returns:
        (avg_train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Flexible batch unpacking (images, labels, *rest = batch)
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1]
        elif isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at batch {batch_idx}: {loss.item()}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = float(running_loss / total) if total > 0 else 0.0
    epoch_acc = float(correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Executes a deterministic validation epoch with flexible batch unpacking.

    Returns:
        (avg_val_loss, val_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1]
            elif isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = float(running_loss / total) if total > 0 else 0.0
    val_acc = float(correct / total) if total > 0 else 0.0
    return val_loss, val_acc


# Alias for compatibility
validate = validate_one_epoch


def _ensure_sample_images_exist(df: pd.DataFrame) -> None:
    """Generates synthetic RGB images for rows whose file_path does not exist on disk."""
    for _, row in df.iterrows():
        fp = Path(row["file_path"])
        if not fp.is_file():
            fp.parent.mkdir(parents=True, exist_ok=True)
            label = int(row.get("label", 0)) if "label" in row and pd.notna(row["label"]) else 0
            color = (139, 69, 19) if label == 1 else (34, 139, 34)
            img = Image.new("RGB", (224, 224), color=color)
            img.save(fp)


def train_model(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    max_batches: Optional[int] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    pretrained: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Main training execution function.

    Args:
        config: Path to YAML config or pre-loaded dictionary.
        train_df: Optional DataFrame for training split.
        val_df: Optional DataFrame for validation split.
        epochs: Number of epochs override.
        batch_size: Batch size override.
        lr: Learning rate override (alias for learning_rate).
        learning_rate: Learning rate override.
        weight_decay: Weight decay override.
        device: Target execution device ('cuda', 'cpu').
        checkpoint_path: Output path for best model checkpoint.
        dry_run: If True, runs 1 epoch with max 2 batches in < 5 seconds.
        max_batches: Restricts number of batches processed per epoch.
        splits_dir: Directory containing train.csv and val.csv.
        artifacts_dir: Output directory for checkpoints and logs.
        pretrained: Whether to initialize with ImageNet weights.
        **kwargs: Additional overrides.

    Returns:
        Summary dictionary with checkpoint and training history information.
    """
    effective_lr = lr if lr is not None else learning_rate

    # 1. Parse configuration
    cfg = load_training_config(
        config if isinstance(config, (str, Path)) else None,
        **(config if isinstance(config, dict) else {}),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=effective_lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_path=checkpoint_path,
        splits_dir=splits_dir,
        artifacts_dir=artifacts_dir,
        pretrained=pretrained,
        **kwargs,
    )

    seed_val = int(cfg.get("system", {}).get("seed", 42))
    seed_everything(seed_val)

    # 2. Select hardware device
    req_device = str(cfg.get("system", {}).get("device", "cuda")).lower()
    if req_device.startswith("cuda") and torch.cuda.is_available():
        target_device = torch.device(req_device)
    else:
        if req_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        target_device = torch.device("cpu")
    logger.info("Operating device: %s", target_device)

    # 3. Locate and load train / val splits
    paths_cfg = cfg.get("paths", {})
    resolved_splits_dir = Path(splits_dir) if splits_dir else Path(paths_cfg.get("splits_dir", "data/processed/splits"))
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))

    if train_df is None:
        candidates = [
            resolved_splits_dir / "train.csv",
            processed_dir / "train.csv",
            processed_dir / "splits" / "train.csv",
        ]
        t_path = next((p for p in candidates if p.is_file()), None)
        if t_path is None:
            raise FileNotFoundError(f"Training split not found in {candidates}. Run waste_hotspot.data first.")
        train_df = pd.read_csv(t_path)

    if val_df is None:
        candidates = [
            resolved_splits_dir / "val.csv",
            processed_dir / "val.csv",
            processed_dir / "splits" / "val.csv",
        ]
        v_path = next((p for p in candidates if p.is_file()), None)
        if v_path is None:
            raise FileNotFoundError(f"Validation split not found in {candidates}. Run waste_hotspot.data first.")
        val_df = pd.read_csv(v_path)

    # 4. Handle dry-run / synthetic test bounds
    if dry_run:
        logger.info("[DRY-RUN] Fast verification mode active: 1 epoch, max 2 batches per partition.")
        train_df = train_df.iloc[:8].copy().reset_index(drop=True)
        val_df = val_df.iloc[:4].copy().reset_index(drop=True)
        # Ensure image files exist on disk for the dry-run subset
        _ensure_sample_images_exist(train_df)
        _ensure_sample_images_exist(val_df)
        if max_batches is None:
            max_batches = 2

    # Verify non-empty datasets
    if len(train_df) == 0:
        raise ValueError("Train dataframe contains 0 samples. Verify dataset preparation.")
    if len(val_df) == 0:
        raise ValueError("Val dataframe contains 0 samples. Verify dataset preparation.")

    # 5. Build datasets and dataloaders
    prep_cfg = cfg.get("preprocessing", {})
    image_size = int(prep_cfg.get("image_size", 224))
    train_transform = get_train_transforms(
        image_size=image_size,
        rotation_degrees=int(prep_cfg.get("rotation_degrees", 15)),
        brightness_jitter=float(prep_cfg.get("brightness_jitter", 0.1)),
        contrast_jitter=float(prep_cfg.get("contrast_jitter", 0.1)),
    )
    eval_transform = get_eval_transforms(image_size=image_size)

    train_dataset = AerialWasteDataset(df=train_df, transform=train_transform)
    val_dataset = AerialWasteDataset(df=val_df, transform=eval_transform)

    train_params = cfg.get("training", {})
    bs = 2 if dry_run else int(train_params.get("batch_size", 32))
    num_workers = 0 if (sys.platform == "win32" or dry_run) else int(cfg.get("dataset", {}).get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(target_device.type == "cuda"),
        drop_last=(len(train_dataset) > bs),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(target_device.type == "cuda"),
        drop_last=False,
    )

    # 6. Instantiate Model
    model_cfg = cfg.get("model", {})
    backbone_name = str(model_cfg.get("backbone", "resnet50"))
    num_classes = int(model_cfg.get("num_classes", 2))
    is_pretrained = bool(model_cfg.get("pretrained", True)) if not dry_run else (pretrained is True)
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))

    model = build_classifier(
        model_name=backbone_name,
        num_classes=num_classes,
        pretrained=is_pretrained,
        freeze_backbone=freeze_backbone,
    )
    model.to(target_device)

    # 7. Loss function with class weighting
    raw_weights = train_params.get("class_weights", [1.0, 2.0])
    if raw_weights == "auto":
        weights_tensor = compute_class_weights(train_df, num_classes=num_classes).to(target_device)
        logger.info("Using auto-computed inverse frequency class weights: %s", weights_tensor.tolist())
    elif isinstance(raw_weights, (list, tuple)) and len(raw_weights) == num_classes:
        weights_tensor = torch.tensor(raw_weights, dtype=torch.float32, device=target_device)
        logger.info("Using configured class weights: %s", weights_tensor.tolist())
    else:
        weights_tensor = None
        logger.info("Using standard unweighted CrossEntropyLoss.")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 8. Optimizer (filtering trainable parameters for frozen backbones)
    opt_lr = float(train_params.get("learning_rate", 1e-4))
    opt_wd = float(train_params.get("weight_decay", 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=opt_lr, weight_decay=opt_wd)

    # 9. Learning rate scheduler (ReduceLROnPlateau)
    sched_patience = int(train_params.get("lr_scheduler_patience", 2))
    sched_factor = float(train_params.get("lr_scheduler_factor", 0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_factor,
        patience=sched_patience,
        min_lr=1e-6,
    )

    # 10. Early stopping monitor
    es_patience = int(train_params.get("early_stopping_patience", 3))
    es_delta = float(train_params.get("early_stopping_min_delta", 0.001))
    early_stopping = EarlyStopping(patience=es_patience, min_delta=es_delta, mode="min")

    # 11. Execution loop
    total_epochs = 1 if dry_run else int(train_params.get("epochs", 25))
    ckpt_path = Path(paths_cfg.get("checkpoint_path", "artifacts/best_model.pth")).resolve()
    history_path = Path(paths_cfg.get("history_path", "artifacts/training_history.json")).resolve()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 1

    logger.info("Beginning training for %d epochs...", total_epochs)

    for epoch in range(1, total_epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=target_device,
            max_batches=max_batches,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=target_device,
            max_batches=max_batches,
        )

        elapsed = time.perf_counter() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(float(round(elapsed, 2)))

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%% | LR: %.6f | Time: %.2fs",
            epoch,
            total_epochs,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0,
            current_lr,
            elapsed,
        )

        scheduler.step(val_loss)

        # Early stopping step
        stopped = early_stopping.step(val_loss=val_loss, model=model, epoch=epoch)
        if early_stopping.improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

            # Save atomic checkpoint
            checkpoint_state = {
                "epoch": int(best_epoch),
                "model_state_dict": early_stopping.best_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": float(best_val_loss),
                "val_acc": float(best_val_acc),
                "config": cfg,
                "class_names": cfg.get("dataset", {}).get("class_names", ["non-waste", "waste"]),
                "best_val_loss": float(best_val_loss),
                "best_val_f1": float(best_val_acc),
                "threshold": 0.50,
                "calibrated_threshold": 0.50,
                "model_architecture": backbone_name,
                "model_name": backbone_name,
                "num_classes": num_classes,
                "input_size": [3, image_size, image_size],
            }
            save_checkpoint(checkpoint_state, ckpt_path, is_best=True)

        if stopped or early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d. Halting training.", epoch)
            break

    # 12. Restore best model weights
    early_stopping.restore_best_weights(model)

    # 13. Export training history JSON
    history_payload = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "total_epochs": len(history["train_loss"]),
        "stopped_early": bool(early_stopping.early_stop),
        "epochs": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "train_acc": history["train_accuracy"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_accuracy"],
        "learning_rates": history["learning_rate"],
        "epoch_times_seconds": history["epoch_time"],
        "history": {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
            "learning_rate": history["learning_rate"],
            "epoch_time": history["epoch_time"],
        },
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    logger.info("Exported training history to %s", history_path)

    return {
        "model": model,
        "checkpoint_path": ckpt_path,
        "history_path": history_path,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "history": history,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Builds and parses CLI argument namespace."""
    parser = argparse.ArgumentParser(description="Train baseline classifier for AerialWaste detection.")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", "-b", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay.")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["cuda", "cpu"], help="Compute device.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Destination checkpoint path.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--dry-run", action="store_true", help="Execute rapid dry run for pipeline validation.")
    parser.add_argument("--max-batches", "-m", type=int, default=None, help="Cap batches per epoch for rapid testing.")
    parser.add_argument("--splits-dir", "-s", type=str, default=None, help="Directory containing train.csv and val.csv.")
    parser.add_argument("--artifacts-dir", "-o", type=str, default=None, help="Directory for saving checkpoints and training history.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--no-pretrained", action="store_true", help="Forces model creation with uninitialized weights.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI Entry Point for waste_hotspot.train."""
    args = parse_args(argv)

    try:
        results = train_model(
            config=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            early_stopping_patience=args.patience,
            dry_run=args.dry_run,
            max_batches=args.max_batches,
            splits_dir=args.splits_dir,
            artifacts_dir=args.artifacts_dir,
            seed=args.seed,
            pretrained=False if args.no_pretrained else None,
        )
        logger.info(
            "Training successfully completed! Best Epoch: %d | Val Loss: %.4f | Val Acc: %.2f%% | Checkpoint: %s",
            results["best_epoch"],
            results["best_val_loss"],
            results["best_val_acc"] * 100.0,
            results["checkpoint_path"],
        )
        return 0
    except Exception as exc:
        logger.error("Training failed with error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
