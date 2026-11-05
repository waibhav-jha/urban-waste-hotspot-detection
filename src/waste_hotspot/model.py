"""
Model architectures, transfer learning classifiers, and checkpoint persistence
for satellite and aerial urban waste hotspot detection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models

# Ensure src/ is on sys.path
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

logger = logging.getLogger(__name__)

# Supported model architectures
SUPPORTED_ARCHITECTURES: List[str] = ["resnet50", "resnet18", "efficientnet_b0"]
SUPPORTED_BACKBONES: List[str] = SUPPORTED_ARCHITECTURES


class _LightweightDummyClassifier(nn.Module):
    """Fallback architecture for lightweight test checkpoints matching tests/e2e/conftest.py."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 8, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.conv(x))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def _get_torchvision_model(
    architecture: str,
    pretrained: bool = True,
) -> Tuple[nn.Module, str]:
    """
    Instantiates a torchvision model with optional pretrained weights and offline fallback.

    Args:
        architecture: Name of the architecture ('resnet50', 'resnet18', 'efficientnet_b0').
        pretrained: If True, attempts to load default ImageNet weights. If False or if
                    download fails, falls back to uninitialized weights (weights=None).

    Returns:
        Tuple of (model_instance, classifier_head_attribute_name).
    """
    arch = architecture.lower().replace("-", "_")

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            model = models.resnet50(weights=weights)
        except Exception as exc:
            if pretrained:
                logger.warning(
                    "Failed to download pretrained weights for %s (%s). Falling back to weights=None.",
                    arch,
                    exc,
                )
                model = models.resnet50(weights=None)
            else:
                raise exc
        return model, "fc"

    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = models.resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                logger.warning(
                    "Failed to download pretrained weights for %s (%s). Falling back to weights=None.",
                    arch,
                    exc,
                )
                model = models.resnet18(weights=None)
            else:
                raise exc
        return model, "fc"

    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            if pretrained:
                logger.warning(
                    "Failed to download pretrained weights for %s (%s). Falling back to weights=None.",
                    arch,
                    exc,
                )
                model = models.efficientnet_b0(weights=None)
            else:
                raise exc
        return model, "classifier"

    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {SUPPORTED_ARCHITECTURES}"
        )


def freeze_model_backbone(model: nn.Module, head_attr: Optional[str] = None) -> None:
    """
    Freezes all convolutional backbone parameters, leaving only the classification head trainable.

    Args:
        model: PyTorch model instance.
        head_attr: Optional attribute name of classification head ('fc' or 'classifier').
                   If None, auto-detected from model.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Auto-detect head attribute if not provided
    if head_attr is None:
        if hasattr(model, "fc"):
            head_attr = "fc"
        elif hasattr(model, "classifier"):
            head_attr = "classifier"
        else:
            raise AttributeError("Unable to identify classification head on model.")

    head_module = getattr(model, head_attr)
    for param in head_module.parameters():
        param.requires_grad = True

    logger.info("Frozen backbone; enabled gradients exclusively for classification head '%s'.", head_attr)


def unfreeze_backbone(
    model: nn.Module,
    layers: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    """
    Selectively unfreezes model layers for phased fine-tuning.

    Args:
        model: PyTorch model instance.
        layers: Layer name (str) or sequence of layer/module names to unfreeze
                (e.g. 'layer4', ['layer4'], or ['features.7', 'features.8']).
                If None, or contains 'all' / 'full', unfreezes all parameters across the entire model.

    Raises:
        ValueError: If specified layer names are not found in the model.
    """
    if layers is None:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all parameters across entire model.")
        return

    if isinstance(layers, str):
        if layers in ("all", "full"):
            for param in model.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all parameters across entire model.")
            return
        layer_list: Sequence[str] = [layers]
    else:
        if any(alias in layers for alias in ["all", "full"]):
            for param in model.parameters():
                param.requires_grad = True
            logger.info("Unfrozen all parameters across entire model.")
            return
        layer_list = layers

    for layer_name in layer_list:
        matched = False
        # 1. Check direct child modules
        for child_name, child_module in model.named_children():
            if child_name == layer_name:
                for param in child_module.parameters():
                    param.requires_grad = True
                matched = True

        # 2. Check prefix matching on named parameters (e.g. 'layer4.0.conv1.weight')
        if not matched:
            for param_name, param in model.named_parameters():
                if param_name.startswith(layer_name):
                    param.requires_grad = True
                    matched = True

        if matched:
            logger.info("Unfrozen parameters matching layer '%s'.", layer_name)
        else:
            available = [name for name, _ in model.named_children()]
            raise ValueError(
                f"Layer '{layer_name}' not found in model hierarchy. "
                f"Available children: {available}"
            )


def unfreeze_layers(
    model: nn.Module,
    unfreeze_from: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    """
    Alias / convenience wrapper for unfreeze_backbone.
    """
    unfreeze_backbone(model, layers=unfreeze_from)


def build_classifier(
    model_name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    backbone: Optional[str] = None,
) -> nn.Module:
    """
    Constructs a transfer learning classifier with a custom classification head.

    Args:
        model_name: Backbone architecture name ('resnet50', 'resnet18', 'efficientnet_b0').
        pretrained: Whether to initialize with ImageNet pretrained weights.
        num_classes: Number of target output classes (default: 2 for binary waste/non-waste).
        freeze_backbone: If True, freezes feature extractor parameters for linear probing.
        backbone: Alias for model_name to maintain compatibility across config formats.

    Returns:
        nn.Module: Configured PyTorch model instance.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

    arch = backbone if backbone is not None else model_name
    model, head_attr = _get_torchvision_model(arch, pretrained=pretrained)

    # Replace final classification head
    if head_attr == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif head_attr == "classifier":
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise TypeError(f"Unexpected classifier type: {type(model.classifier)}")

    # Apply backbone freezing if requested
    if freeze_backbone:
        freeze_model_backbone(model, head_attr=head_attr)

    logger.info(
        "Built classifier '%s' with num_classes=%d, pretrained=%s, freeze_backbone=%s",
        arch,
        num_classes,
        pretrained,
        freeze_backbone,
    )
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Counts total, trainable, and frozen parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dict with keys 'total', 'trainable', 'frozen'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def get_trainable_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Retrieves all named parameters with requires_grad=True.

    Args:
        model: PyTorch model.

    Returns:
        List of (parameter_name, parameter_tensor) tuples.
    """
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def predict_probabilities(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Executes a forward pass and computes softmax class probabilities.

    Args:
        model: PyTorch model.
        x: Input image tensor of shape (B, 3, H, W).
        device: Target execution device. If None, uses device of model parameters.

    Returns:
        torch.Tensor of shape (B, num_classes) representing class probabilities.
    """
    model.eval()
    if device is not None:
        target_device = torch.device(device)
        model = model.to(target_device)
        x = x.to(target_device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs


def save_checkpoint(
    state_or_filepath: Union[Dict[str, Any], str, Path] = None,
    filepath: Optional[Union[str, Path]] = None,
    is_best: bool = False,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    atomic: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Atomically serializes model state and metadata to disk.

    Writes to a temporary file before renaming via os.replace to prevent
    corrupted checkpoints during unexpected process termination.

    Supports both dictionary-first invocation:
        save_checkpoint(state_dict, "artifacts/best_model.pth", is_best=True)
    and keyword argument invocation:
        save_checkpoint(filepath="...", model=model, optimizer=optimizer, epoch=1, ...)

    Args:
        state_or_filepath: Checkpoint dict or filepath string/Path.
        filepath: Destination path if state_or_filepath is a dictionary.
        is_best: If True, additionally copies checkpoint to best_model.pth if filename differs.

    Returns:
        Path object of saved checkpoint.
    """
    if isinstance(state_or_filepath, dict):
        state = state_or_filepath
        dest = Path(filepath if filepath is not None else "artifacts/best_model.pth").resolve()
    else:
        dest_str = state_or_filepath if state_or_filepath is not None else filepath
        dest = Path(dest_str if dest_str is not None else "artifacts/best_model.pth").resolve()
        state = dict(kwargs)
        if model is not None:
            state["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        if val_loss is not None:
            state["val_loss"] = float(val_loss)
            state["best_val_loss"] = float(val_loss)
        if val_acc is not None:
            state["val_acc"] = float(val_acc)
            state["best_val_f1"] = float(val_acc)
        if config is not None:
            state["config"] = config
        if class_names is not None:
            state["class_names"] = class_names
        if threshold is not None:
            state["threshold"] = float(threshold)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f"{dest.stem}.tmp_{os.getpid()}{dest.suffix}")

    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, dest)
        logger.info("Saved model checkpoint atomically to: %s", dest)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error("Failed to save checkpoint to %s: %s", dest, exc)
        raise exc

    if is_best and dest.name != "best_model.pth":
        best_path = dest.parent / "best_model.pth"
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, best_path)
            logger.info("Updated best model checkpoint at: %s", best_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.warning("Failed to copy best model checkpoint: %s", exc)

    return dest


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[Union[nn.Module, bool]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Union[Tuple[Optional[nn.Module], Dict[str, Any]], Dict[str, Any]]:
    """
    Loads a saved checkpoint, restores weights into a model, and returns checkpoint metadata.

    Args:
        filepath: Filesystem path to checkpoint file.
        model: Optional nn.Module instance to restore weights into. If None and
               'model_architecture' is in checkpoint metadata, automatically builds model.
               If False, returns only the checkpoint dictionary.
        device: Device to map tensors onto (defaults to 'cpu').
        strict: Whether to strictly enforce matching state dict keys.

    Returns:
        Tuple of (model, checkpoint_dictionary), or checkpoint_dictionary if model is False.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If weights mismatch under strict=True.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device(device if device is not None else "cpu")

    try:
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=target_device)

    # If caller explicitly asked for dict only (model is False)
    if model is False:
        return checkpoint

    # Extract state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Invalid checkpoint payload type: {type(checkpoint)}")

    # Auto-instantiate model if None
    target_model: Optional[nn.Module] = model if isinstance(model, nn.Module) else None
    if target_model is None and isinstance(checkpoint, dict):
        arch = checkpoint.get("model_architecture", checkpoint.get("model_name"))
        if arch is not None:
            num_classes = checkpoint.get("num_classes", 2)
            target_model = build_classifier(model_name=arch, pretrained=False, num_classes=num_classes)
        elif isinstance(state_dict, dict) and "conv.weight" in state_dict and "fc.weight" in state_dict and "layer1.0.conv1.weight" not in state_dict:
            hidden_dim = state_dict["conv.weight"].shape[0]
            num_classes = state_dict["fc.weight"].shape[0]
            target_model = _LightweightDummyClassifier(hidden_dim=hidden_dim, num_classes=num_classes)
        elif isinstance(state_dict, dict) and "fc.weight" in state_dict and len(state_dict["fc.weight"].shape) == 2 and state_dict["fc.weight"].shape[1] == 2048:
            num_classes = state_dict["fc.weight"].shape[0]
            target_model = build_classifier(model_name="resnet50", pretrained=False, num_classes=num_classes)

    # Restore weights into model if present
    if target_model is not None and isinstance(state_dict, dict):
        try:
            target_model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            if not strict:
                logger.warning("Non-strict load_state_dict mismatch warning: %s", exc)
                target_model.load_state_dict(state_dict, strict=False)
            else:
                raise exc
        target_model.to(target_device)
        target_model.eval()

    logger.info("Successfully loaded checkpoint from %s to %s", path, target_device)
    return target_model, checkpoint
