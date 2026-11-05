"""
Adversarial and empirical stress test suite for waste_hotspot.model.

Investigates:
1. Odd and extreme batch sizes (B=1, B=3, B=17, B=64) in forward and predict_probabilities.
2. Strict gradient flow assertions under full backbone freezing (ResNet-50).
3. Selective unfreezing gradient flow assertions (layer4 vs layer1-3).
4. Softmax numerical stability under extreme logit regimes and extreme inputs.
5. Corrupted checkpoint handling in load_checkpoint (empty, truncated, invalid payload).
6. Tensor dimension and channel mismatch handling.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from waste_hotspot.model import (
    build_classifier,
    count_parameters,
    freeze_model_backbone,
    load_checkpoint,
    predict_probabilities,
    save_checkpoint,
    unfreeze_backbone,
    unfreeze_layers,
)


class TestOddAndExtremeBatchSizes:
    """Stress tests forward pass and predict_probabilities across odd and extreme batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 3, 17, 64])
    @pytest.mark.parametrize("arch", ["resnet50", "resnet18", "efficientnet_b0"])
    def test_predict_probabilities_extreme_batch_sizes(self, batch_size: int, arch: str):
        """Verifies predict_probabilities handles B=1, 3, 17, 64 cleanly for all backbones."""
        model = build_classifier(model_name=arch, pretrained=False, num_classes=2)
        # Even if model was in train mode, predict_probabilities must switch to eval and handle B=1
        model.train()

        torch.manual_seed(100 + batch_size)
        x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

        probs = predict_probabilities(model, x, device="cpu")

        # Verify output shape
        assert probs.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {probs.shape}"
        assert probs.dtype == torch.float32

        # Verify no NaN or Inf
        assert not torch.isnan(probs).any(), f"NaN found in probabilities for B={batch_size}, arch={arch}"
        assert not torch.isinf(probs).any(), f"Inf found in probabilities for B={batch_size}, arch={arch}"

        # Verify probability bounds [0, 1]
        assert (probs >= 0.0).all(), f"Probabilities below 0.0 for B={batch_size}, arch={arch}"
        assert (probs <= 1.0).all(), f"Probabilities above 1.0 for B={batch_size}, arch={arch}"

        # Verify simplex sum constraint: sum_k p_k == 1.0 +- 1e-6
        row_sums = probs.sum(dim=1)
        expected_sums = torch.ones(batch_size, dtype=torch.float32)
        assert torch.allclose(row_sums, expected_sums, atol=1e-6), (
            f"Row sums deviate from 1.0 for B={batch_size}: {row_sums}"
        )

    @pytest.mark.parametrize("batch_size", [1, 3, 17, 64])
    def test_forward_pass_eval_mode_batch_sizes(self, batch_size: int):
        """Verifies forward pass in eval mode produces valid logits across odd batch sizes."""
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2)
        model.eval()

        torch.manual_seed(200 + batch_size)
        x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_batch_size_1_train_and_eval_behavior(self):
        """Verifies ResNet forward pass and predict_probabilities behavior on B=1.

        Because spatial dimensions are (112, 112) after conv1, BatchNorm2d has
        H*W = 12544 values per channel, allowing both train and eval modes to succeed on B=1.
        """
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2)
        x_single = torch.randn(1, 3, 224, 224)

        # In train mode
        model.train()
        logits_train = model(x_single)
        assert logits_train.shape == (1, 2)

        # In predict_probabilities (which ensures eval mode)
        probs = predict_probabilities(model, x_single)
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


class TestBackboneFreezingGradientFlow:
    """Verifies gradient isolation and parameter immutability under backbone freezing."""

    def test_resnet50_backbone_freezing_strictly_isolates_gradients(self):
        """Forward pass batch (4, 3, 224, 224), compute loss, call loss.backward().

        Strictly verify:
        - model.fc.weight.grad is not None
        - model.fc.bias.grad is not None
        - model.conv1.weight.grad is None
        - model.layer1[0].conv1.weight.grad is None
        - model.layer2[0].conv1.weight.grad is None
        - model.layer3[0].conv1.weight.grad is None
        - model.layer4[0].conv1.weight.grad is None
        """
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2, freeze_backbone=True)
        model.train()

        torch.manual_seed(42)
        x = torch.randn(4, 3, 224, 224, dtype=torch.float32)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()

        # Classification head MUST have non-None gradients with non-zero norm
        assert model.fc.weight.grad is not None, "Head weight grad is None"
        assert model.fc.bias.grad is not None, "Head bias grad is None"
        assert model.fc.weight.grad.shape == (2, 2048)
        assert model.fc.bias.grad.shape == (2,)
        assert model.fc.weight.grad.abs().sum() > 0.0, "Head weight grad is all zeros"

        # Backbone convolutional layers MUST strictly have None gradients
        assert model.conv1.weight.grad is None, "conv1.weight.grad should be None"
        assert model.layer1[0].conv1.weight.grad is None, "layer1[0].conv1.weight.grad should be None"
        assert model.layer2[0].conv1.weight.grad is None, "layer2[0].conv1.weight.grad should be None"
        assert model.layer3[0].conv1.weight.grad is None, "layer3[0].conv1.weight.grad should be None"
        assert model.layer4[0].conv1.weight.grad is None, "layer4[0].conv1.weight.grad should be None"

        # Check every single parameter in the backbone to guarantee zero leakage
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                assert param.requires_grad is True
                assert param.grad is not None
            else:
                assert param.requires_grad is False, f"Parameter {name} unexpectedly has requires_grad=True"
                assert param.grad is None, f"Parameter {name} unexpectedly accumulated gradients"


class TestSelectiveUnfreezingGradientFlow:
    """Verifies gradient isolation when selectively unfreezing specific stages."""

    def test_selective_unfreeze_layer4_gradient_flow(self):
        """Call unfreeze_backbone(model, layers=['layer4']).

        Forward pass, backward pass.
        Strictly verify:
        - model.layer4[0].conv1.weight.grad is not None
        - model.fc.weight.grad is not None
        - model.layer3[0].conv1.weight.grad is None
        - model.layer2[0].conv1.weight.grad is None
        - model.layer1[0].conv1.weight.grad is None
        - model.conv1.weight.grad is None
        """
        model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2, freeze_backbone=True)
        unfreeze_backbone(model, layers=["layer4"])
        model.train()

        torch.manual_seed(42)
        x = torch.randn(4, 3, 224, 224, dtype=torch.float32)
        targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()

        # Unfrozen layer4 and fc MUST have non-None gradients
        assert model.fc.weight.grad is not None, "Head weight grad is None"
        assert model.fc.bias.grad is not None, "Head bias grad is None"
        assert model.layer4[0].conv1.weight.grad is not None, "layer4[0].conv1.weight.grad should NOT be None"
        assert model.layer4[0].conv1.weight.grad.abs().sum() > 0.0, "layer4[0].conv1 grad is all zeros"
        assert model.layer4[2].conv3.weight.grad is not None, "layer4 final conv grad should NOT be None"

        # Frozen layers (conv1, layer1, layer2, layer3) MUST strictly have None gradients
        assert model.conv1.weight.grad is None, "conv1.weight.grad should be None"
        assert model.layer1[0].conv1.weight.grad is None, "layer1[0].conv1.weight.grad should be None"
        assert model.layer2[0].conv1.weight.grad is None, "layer2[0].conv1.weight.grad should be None"
        assert model.layer3[0].conv1.weight.grad is None, "layer3[0].conv1.weight.grad should be None"

        # Comprehensive check across all named parameters
        for name, param in model.named_parameters():
            if name.startswith("layer4.") or name.startswith("fc."):
                assert param.requires_grad is True, f"{name} should have requires_grad=True"
                assert param.grad is not None, f"{name} should have non-None grad"
            else:
                assert param.requires_grad is False, f"{name} should have requires_grad=False"
                assert param.grad is None, f"{name} should have None grad"


class TestSoftmaxNumericalStability:
    """Adversarially tests softmax numerical stability under extreme logit and input regimes."""

    @pytest.mark.parametrize(
        "logits_input, expected_prob0, expected_prob1",
        [
            ([-1000.0, 1000.0], 0.0, 1.0),
            ([1000.0, -1000.0], 1.0, 0.0),
            ([500.0, 500.0], 0.5, 0.5),
            ([-500.0, -500.0], 0.5, 0.5),
            ([1e5, -1e5], 1.0, 0.0),
            ([-1e5, 1e5], 0.0, 1.0),
            ([0.0, 0.0], 0.5, 0.5),
        ],
    )
    def test_softmax_extreme_logits_numerical_stability(
        self, logits_input: list, expected_prob0: float, expected_prob1: float
    ):
        """Tests that extreme logit values compute exact stable probabilities without NaNs or Infs."""
        logits = torch.tensor([logits_input], dtype=torch.float32)
        probs = torch.softmax(logits, dim=1)

        assert not torch.isnan(probs).any(), f"NaN produced from logits {logits_input}"
        assert not torch.isinf(probs).any(), f"Inf produced from logits {logits_input}"
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) <= 1e-6, f"Probabilities sum={prob_sum} deviates from 1.0"
        assert abs(probs[0, 0].item() - expected_prob0) <= 1e-5
        assert abs(probs[0, 1].item() - expected_prob1) <= 1e-5

    def test_predict_probabilities_with_mock_extreme_logits(self):
        """Verifies predict_probabilities wrapper cleanly handles extreme model outputs."""
        class MockExtremeModel(nn.Module):
            def __init__(self, logits_tensor: torch.Tensor):
                super().__init__()
                self.logits = logits_tensor

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                return self.logits[:batch_size]

        extreme_logits = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0], [500.0, 500.0]])
        mock_model = MockExtremeModel(extreme_logits)

        dummy_x = torch.zeros(3, 3, 224, 224)
        probs = predict_probabilities(mock_model, dummy_x)

        assert probs.shape == (3, 2)
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)

        # Check row 0
        assert probs[0, 0].item() == 0.0
        assert probs[0, 1].item() == 1.0

        # Check row 1
        assert probs[1, 0].item() == 1.0
        assert probs[1, 1].item() == 0.0

        # Check row 2
        assert abs(probs[2, 0].item() - 0.5) <= 1e-5
        assert abs(probs[2, 1].item() - 0.5) <= 1e-5

    def test_predict_probabilities_extreme_input_tensor_values(self):
        """Verifies predict_probabilities handles inputs with huge or tiny values (out of normal range)."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)

        # Huge pixel values (e.g. unnormalized 1e4)
        huge_x = torch.full((2, 3, 224, 224), 1000.0, dtype=torch.float32)
        probs_huge = predict_probabilities(model, huge_x)

        assert probs_huge.shape == (2, 2)
        assert not torch.isnan(probs_huge).any()
        assert not torch.isinf(probs_huge).any()
        assert torch.allclose(probs_huge.sum(dim=1), torch.ones(2), atol=1e-5)

        # Extreme negative pixel values
        neg_x = torch.full((2, 3, 224, 224), -1000.0, dtype=torch.float32)
        probs_neg = predict_probabilities(model, neg_x)

        assert probs_neg.shape == (2, 2)
        assert not torch.isnan(probs_neg).any()
        assert not torch.isinf(probs_neg).any()
        assert torch.allclose(probs_neg.sum(dim=1), torch.ones(2), atol=1e-5)


class TestCorruptedCheckpointHandling:
    """Stress tests load_checkpoint resilience against corrupted, empty, or invalid checkpoint files."""

    def test_load_checkpoint_empty_file_raises_informative_error(self, tmp_path: Path):
        """Empty 0-byte file must raise EOFError or UnpicklingError cleanly without silent failure."""
        empty_file = tmp_path / "empty_model.pth"
        empty_file.write_bytes(b"")
        assert empty_file.stat().st_size == 0

        with pytest.raises((EOFError, Exception)) as exc_info:
            load_checkpoint(empty_file)

        # Verify an informative error was raised
        assert exc_info.type in (EOFError, RuntimeError, ValueError) or "pickle" in str(exc_info.type).lower()

    def test_load_checkpoint_truncated_file_raises_informative_error(self, tmp_path: Path):
        """Truncated checkpoint (partial byte stream) must raise unpickling/EOF error."""
        valid_file = tmp_path / "valid.pth"
        state = {
            "model_architecture": "resnet18",
            "model_state_dict": {},
            "epoch": 1,
        }
        torch.save(state, valid_file)

        # Truncate to first 30 bytes
        truncated_bytes = valid_file.read_bytes()[:30]
        corrupt_file = tmp_path / "corrupted_truncated.pth"
        corrupt_file.write_bytes(truncated_bytes)

        with pytest.raises(Exception) as exc_info:
            load_checkpoint(corrupt_file)

        # Must not succeed or exit cleanly
        assert exc_info.value is not None

    def test_load_checkpoint_invalid_payload_type_raises_value_error(self, tmp_path: Path):
        """Checkpoint containing a non-dict object (e.g. serialized integer or string)

        must raise ValueError with informative message.
        """
        invalid_file = tmp_path / "invalid_type.pth"
        torch.save("this is an invalid state dict string", invalid_file)

        with pytest.raises(ValueError, match="Invalid checkpoint payload type"):
            load_checkpoint(invalid_file)

    def test_load_checkpoint_mismatched_weights_strict_raises_runtime_error(self, tmp_path: Path):
        """Loading state dict with incompatible architecture under strict=True raises RuntimeError."""
        resnet18_model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        ckpt_file = tmp_path / "resnet18.pth"
        save_checkpoint({
            "model_architecture": "resnet18",
            "model_state_dict": resnet18_model.state_dict(),
        }, ckpt_file)

        # Attempt to load ResNet-18 weights into ResNet-50 with strict=True
        resnet50_model = build_classifier(model_name="resnet50", pretrained=False, num_classes=2)
        with pytest.raises(RuntimeError):
            load_checkpoint(ckpt_file, model=resnet50_model, strict=True)

    def test_load_checkpoint_nonexistent_file_raises_file_not_found(self, tmp_path: Path):
        """Nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_checkpoint(tmp_path / "does_not_exist_xyz.pth")


class TestTensorShapeAndDeviceAnomalies:
    """Stress tests input tensor dimensionalities and edge-case contracts."""

    def test_predict_probabilities_3d_input_raises_runtime_error(self):
        """Passing unbatched 3D tensor (3, 224, 224) instead of 4D tensor (1, 3, 224, 224)

        triggers informative PyTorch dimension error.
        """
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        x_3d = torch.randn(3, 224, 224)

        with pytest.raises((RuntimeError, ValueError)):
            predict_probabilities(model, x_3d)

    def test_predict_probabilities_incorrect_channel_count(self):
        """Passing 1-channel or 4-channel tensor raises RuntimeError in conv1."""
        model = build_classifier(model_name="resnet18", pretrained=False, num_classes=2)
        x_1ch = torch.randn(2, 1, 224, 224)
        x_4ch = torch.randn(2, 4, 224, 224)

        with pytest.raises(RuntimeError):
            predict_probabilities(model, x_1ch)

        with pytest.raises(RuntimeError):
            predict_probabilities(model, x_4ch)
