"""
Unit and integration tests for waste_hotspot.model architecture, freezing,
tensor shape contracts, and checkpoint serialization.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
import torchvision.models as models

from waste_hotspot.model import (
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_BACKBONES,
    build_classifier,
    count_parameters,
    freeze_model_backbone,
    get_trainable_parameters,
    load_checkpoint,
    predict_probabilities,
    save_checkpoint,
    unfreeze_backbone,
    unfreeze_layers,
)


@pytest.fixture
def synthetic_batch_tensor() -> torch.Tensor:
    """Creates a standard 4D batch tensor of shape (4, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224, dtype=torch.float32)


@pytest.fixture
def single_sample_tensor() -> torch.Tensor:
    """Creates a single sample tensor of shape (1, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)


class TestModelArchitectureContracts:
    """Verifies model building, head replacements, and tensor shape contracts."""

    @pytest.mark.parametrize("arch", ["resnet50", "resnet18", "efficientnet_b0"])
    def test_build_classifier_supported_architectures_shape_contract(self, arch: str):
        """Test build_classifier produces valid logits (B, 2) for input (B, 3, 224, 224)."""
        model = build_classifier(model_name=arch, pretrained=False, num_classes=2)
        assert isinstance(model, nn.Module)

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        logits = model(x)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
        assert logits.dtype == torch.float32
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_build_classifier_resnet50_head_dimensions(self):
        """Verifies ResNet-50 2-class head replacement dimensions."""
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2)
        assert hasattr(model, "fc")
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.in_features == 2048
        assert model.fc.out_features == 2

    def test_build_classifier_resnet18_head_dimensions(self):
        """Verifies ResNet-18 2-class head replacement dimensions."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        assert hasattr(model, "fc")
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.in_features == 512
        assert model.fc.out_features == 2

    def test_build_classifier_efficientnet_head_dimensions(self):
        """Verifies EfficientNet-B0 2-class head replacement dimensions."""
        model = build_classifier(model_name="efficientnet_b0", pretrained=False, num_classes=2)
        assert hasattr(model, "classifier")
        last_linear = model.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.in_features == 1280
        assert last_linear.out_features == 2

    def test_build_classifier_backbone_alias(self):
        """Test backbone kwarg works identically to model_name."""
        m1 = build_classifier(backbone="resnet18", pretrained=False, num_classes=2)
        assert hasattr(m1, "fc")
        assert m1.fc.out_features == 2

    def test_build_classifier_custom_num_classes(self):
        """Verifies arbitrary num_classes configuration."""
        for n_cls in [3, 5, 10]:
            m = build_classifier(model_name="resnet18", pretrained=False, num_classes=n_cls)
            assert m.fc.out_features == n_cls

    def test_forward_pass_single_sample_shape(self, single_sample_tensor: torch.Tensor):
        """Verifies forward pass on single sample (1, 3, 224, 224) produces logits (1, 2)."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        model.eval()
        with torch.no_grad():
            logits = model(single_sample_tensor)
        assert logits.shape == (1, 2)

    def test_build_classifier_unsupported_architecture_raises_value_error(self):
        """Test invalid architecture string raises ValueError with supported options listed."""
        with pytest.raises(ValueError, match="Unsupported architecture 'vgg16'"):
            build_classifier(model_name="vgg16", pretrained=False)

    def test_build_classifier_invalid_num_classes_raises_value_error(self):
        """Test non-positive num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            build_classifier(model_name="resnet18", pretrained=False, num_classes=0)


class TestBackboneFreezingAndUnfreezing:
    """Verifies gradient freezing and selective unfreezing mechanics."""

    def test_freeze_backbone_resnet50_trainable_parameters_count(self):
        """Test freeze_backbone=True leaves exactly 4098 trainable weights on ResNet-50."""
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2, freeze_backbone=True)

        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]

        assert set(trainable_names) == {"fc.weight", "fc.bias"}
        assert len(frozen_names) > 150

        param_counts = count_parameters(model)
        # fc.weight has 2 * 2048 = 4096 params, fc.bias has 2 params -> total 4098
        assert param_counts["trainable"] == 4098
        assert param_counts["frozen"] > 20_000_000

    def test_freeze_backbone_efficientnet(self):
        """Test freeze_backbone=True leaves only classifier layer trainable on EfficientNet."""
        model = build_classifier(model_name="efficientnet_b0", pretrained=False, num_classes=2, freeze_backbone=True)
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        assert set(trainable_names) == {"classifier.1.weight", "classifier.1.bias"}

    def test_unfreeze_backbone_selective_layer4(self):
        """Test unfreezing layer4 enables gradients on layer4 while keeping layer1-3 frozen."""
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2, freeze_backbone=True)
        unfreeze_backbone(model, layers=["layer4"])

        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        assert any(name.startswith("layer4") for name in trainable_names)
        assert not any(name.startswith("layer1") for name in trainable_names)
        assert not any(name.startswith("layer2") for name in trainable_names)
        assert not any(name.startswith("layer3") for name in trainable_names)
        assert "fc.weight" in trainable_names

    def test_unfreeze_layers_alias(self):
        """Test unfreeze_layers convenience wrapper."""
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2, freeze_backbone=True)
        unfreeze_layers(model, unfreeze_from="layer4")
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        assert any(name.startswith("layer4") for name in trainable_names)
        assert not any(name.startswith("layer1") for name in trainable_names)

    def test_unfreeze_backbone_all(self):
        """Test unfreezing all parameters enables gradients everywhere."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2, freeze_backbone=True)
        unfreeze_backbone(model, layers=None)

        all_params = list(model.parameters())
        assert all(p.requires_grad for p in all_params)

    def test_unfreeze_invalid_layer_raises_value_error(self):
        """Test unfreezing nonexistent layer raises ValueError."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        with pytest.raises(ValueError, match="Layer 'layer99' not found"):
            unfreeze_backbone(model, layers=["layer99"])

    def test_get_trainable_parameters(self):
        """Test get_trainable_parameters retrieves only active parameters."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2, freeze_backbone=True)
        trainable = get_trainable_parameters(model)
        trainable_names = [name for name, _ in trainable]
        assert set(trainable_names) == {"fc.weight", "fc.bias"}


class TestProbabilitySimplexAndInference:
    """Verifies probability generation and simplex constraints."""

    def test_predict_probabilities_simplex_constraint(self):
        """Test predicted probabilities satisfy simplex sum(p) == 1.0 and p in [0, 1]."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        x = torch.randn(5, 3, 224, 224)
        probs = predict_probabilities(model, x, device="cpu")

        assert probs.shape == (5, 2)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestCheckpointSerializationAndRestoration:
    """Verifies atomic save and load mechanisms."""

    def test_save_and_load_checkpoint_roundtrip(self, tmp_path: Path):
        """Test saving state dict and restoring weights produces identical predictions."""
        ckpt_file = tmp_path / "checkpoints" / "model.pth"
        model_orig = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)

        state = {
            "epoch": 7,
            "model_state_dict": model_orig.state_dict(),
            "best_val_loss": 0.28,
            "best_val_f1": 0.89,
            "threshold": 0.54,
            "model_architecture": "resnet18",
            "num_classes": 2,
        }
        saved_path = save_checkpoint(state, ckpt_file, is_best=True)
        assert saved_path.exists()
        assert (tmp_path / "checkpoints" / "best_model.pth").exists()

        # Load into fresh model
        model_restored = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        _, loaded_meta = load_checkpoint(saved_path, model=model_restored)

        assert loaded_meta["epoch"] == 7
        assert loaded_meta["best_val_loss"] == 0.28
        assert loaded_meta["threshold"] == 0.54

        # Compare outputs on deterministic input
        x = torch.randn(2, 3, 224, 224)
        model_orig.eval()
        model_restored.eval()
        with torch.no_grad():
            out_orig = model_orig(x)
            out_rest = model_restored(x)
        assert torch.allclose(out_orig, out_rest, atol=1e-6)

    def test_load_checkpoint_auto_instantiation(self, tmp_path: Path):
        """Test load_checkpoint with model=None automatically instantiates architecture from metadata."""
        ckpt_file = tmp_path / "auto_model.pth"
        model_orig = build_classifier(model_name="resnet50", pretrained=False, num_classes=2)

        save_checkpoint({
            "model_state_dict": model_orig.state_dict(),
            "model_architecture": "resnet50",
            "num_classes": 2,
        }, ckpt_file)

        model_loaded, _ = load_checkpoint(ckpt_file, model=None)
        assert isinstance(model_loaded, models.ResNet)
        assert model_loaded.fc.out_features == 2

    def test_load_checkpoint_missing_file_raises_not_found(self, tmp_path: Path):
        """Test loading nonexistent checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pth")
