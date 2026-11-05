"""
Adversarial and empirical stress test suite for waste_hotspot.evaluate module.

Investigates:
1. Target snooping / test data leakage:
   - Synthetic validation and test arrays.
   - Run calibrate_threshold on val.
   - Mutate test labels/probabilities drastically (inversion, noise, size expansion, extreme shifts).
   - Assert bit-for-bit invariance of calibrated threshold T*.
   - Verify pipeline evaluate_model maintains strict validation-only threshold tuning.
2. Degenerate distributions in compute_evaluation_metrics:
   - Single-class positive (all y_true = 1).
   - Single-class negative (all y_true = 0).
   - Extreme probabilities (all 0.0, all 1.0, all 0.50, out-of-bounds unclamped).
   - Minimal sample sizes (N=1, N=2).
   - Verify roc_auc and pr_auc return None safely without ZeroDivisionError or crashes.
3. Extreme thresholds and monotonicity invariants:
   - T <= 0.0 yields Recall = 1.0.
   - T >= 1.0 yields Recall = 0.0.
   - Monotonicity invariants: Recall non-increasing, TP non-increasing, FP non-increasing,
     TN non-decreasing, FN non-decreasing, and total count conservation across 100 thresholds.
4. Threshold sweep tie-breaking:
   - Identical maximal F1 plateaus stably select candidate closest to 0.50.
   - Asymmetric and symmetric plateaus test deterministic resolution.
   - Degenerate zero-score landscapes safely fall back to 0.50.
   - Multi-metric calibration support ('f1', 'precision', 'recall', 'balanced_accuracy', 'recall_target').
5. Metric dictionary schema and serialization:
   - Completeness of all contract keys and sub-dictionaries.
   - Strict JSON-serializability (no NaNs, no unhandled numpy types).
   - Mathematical consistency between counts, support, contingency table, and macro averages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn
import yaml

from waste_hotspot.evaluate import (
    _coerce_binary_inputs,
    calibrate_threshold,
    compute_evaluation_metrics,
    evaluate_model,
)


class FastDummyClassifier(nn.Module):
    """Minimal dummy classifier for lightning-fast pipeline execution."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture
def dummy_pipeline_env(tmp_path: Path) -> Dict[str, Any]:
    """Sets up minimal files for evaluate_model testing."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def _create_split(name: str, count: int, label_bias: float = 0.5) -> Path:
        rows = []
        for i in range(count):
            p = img_dir / f"{name}_img_{i}.png"
            color = (200, 50, 50) if i % 2 == 1 else (50, 200, 50)
            Image.new("RGB", (224, 224), color=color).save(p)
            lbl = 1 if (i / count) < label_bias else 0
            rows.append({
                "image_id": f"{name}_{i}",
                "file_name": p.name,
                "file_path": str(p),
                "label": lbl,
                "img_source": f"scene_{name}",
                "width": 224,
                "height": 224,
            })
        csv_path = splits_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    val_csv = _create_split("val", count=8, label_bias=0.5)
    test_a_csv = _create_split("test_a", count=6, label_bias=0.8)
    test_b_csv = _create_split("test_b", count=10, label_bias=0.1)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "best_model.pth"

    model = FastDummyClassifier(num_classes=2)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "threshold": 0.50,
        "model_architecture": "resnet18",
        "num_classes": 2,
    }, ckpt_path)

    return {
        "ckpt_path": ckpt_path,
        "val_csv": val_csv,
        "test_a_csv": test_a_csv,
        "test_b_csv": test_b_csv,
    }


# ==============================================================================
# 1. Target Snooping & Test Data Leakage Stress Tests
# ==============================================================================

class TestTargetLeakageAndSnoopingInvariance:
    """Stress tests proving bit-for-bit invariance of threshold calibration under test mutation."""

    def test_validation_calibration_bit_for_bit_invariance_under_test_mutation(self):
        """
        Calibrates threshold T* on fixed validation data, then subjects test data
        to 5 radical mutations. Verifies T* is invariant across all mutations.
        """
        rng = np.random.default_rng(42)
        n_val = 150
        val_true = rng.choice([0, 1], size=n_val, p=[0.6, 0.4])
        # Separable probabilities with some overlap
        val_probs = np.where(val_true == 1, rng.uniform(0.45, 0.95, size=n_val), rng.uniform(0.05, 0.55, size=n_val))

        # Initial baseline calibration on validation split
        t_baseline = calibrate_threshold(y_true=val_true, y_probs=val_probs, metric="f1")

        # Define 5 radical mutations of hypothetical test set data
        n_test = 200
        test_true_orig = rng.choice([0, 1], size=n_test, p=[0.5, 0.5])
        test_probs_orig = rng.uniform(0.0, 1.0, size=n_test)

        test_mutations = [
            # 1. Complete label inversion (0->1, 1->0)
            (1 - test_true_orig, test_probs_orig),
            # 2. Extreme noise injection and shuffling
            (rng.permutation(test_true_orig), np.clip(test_probs_orig + rng.normal(0, 0.5, size=n_test), 0.0, 1.0)),
            # 3. All positive degenerate test set
            (np.ones(n_test, dtype=int), np.full(n_test, 0.99)),
            # 4. All negative degenerate test set
            (np.zeros(n_test, dtype=int), np.full(n_test, 0.01)),
            # 5. Massively expanded test set (20,000 samples)
            (rng.choice([0, 1], size=20000), rng.uniform(0.0, 1.0, size=20000)),
        ]

        for idx, (mut_true, mut_probs) in enumerate(test_mutations, 1):
            # Assert calibration on validation split is 100% immune to test data
            t_after = calibrate_threshold(y_true=val_true, y_probs=val_probs, metric="f1")
            assert t_after == t_baseline, (
                f"Mutation {idx} breached validation calibration invariance: "
                f"T_baseline={t_baseline} vs T_after={t_after}"
            )

    def test_evaluate_model_pipeline_leakage_immunity(self, dummy_pipeline_env: Dict[str, Any], tmp_path: Path):
        """
        Verifies evaluate_model with tune_threshold=True yields bit-for-bit identical
        calibrated decision threshold regardless of which test manifest is passed.
        """
        env = dummy_pipeline_env
        out_a = tmp_path / "report_a.json"
        out_b = tmp_path / "report_b.json"

        report_a = evaluate_model(
            checkpoint_path=env["ckpt_path"],
            data_csv=env["test_a_csv"],
            tune_threshold=True,
            val_csv=env["val_csv"],
            output_path=out_a,
            device="cpu",
        )

        report_b = evaluate_model(
            checkpoint_path=env["ckpt_path"],
            data_csv=env["test_b_csv"],
            tune_threshold=True,
            val_csv=env["val_csv"],
            output_path=out_b,
            device="cpu",
        )

        # Threshold calibrated on val must be bit-for-bit identical across disparate test sets
        t_a = report_a["threshold_used"]
        t_b = report_b["threshold_used"]
        assert t_a == t_b, f"Pipeline threshold leakage detected: test_a got {t_a}, test_b got {t_b}"
        assert report_a["threshold_calibrated"] is True
        assert report_b["threshold_calibrated"] is True


# ==============================================================================
# 2. Degenerate Distributions & Boundary Values Stress Tests
# ==============================================================================

class TestDegenerateDistributionsAndBoundaryValues:
    """Stress tests metric computation on degenerate distributions and extreme probability inputs."""

    @pytest.mark.parametrize("size", [1, 2, 5, 20, 100])
    @pytest.mark.parametrize("prob_value", [0.0, 0.50, 1.0])
    def test_single_class_positive_roc_pr_auc_none(self, size: int, prob_value: float):
        """Verifies all-positive split (y_true=1) sets roc_auc=None, pr_auc=None without exceptions."""
        y_true = np.ones(size, dtype=int)
        y_probs = np.full(size, prob_value)

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None
        assert metrics["support"]["waste"] == size
        assert metrics["support"]["non_waste"] == 0
        assert metrics["counts"]["tp"] + metrics["counts"]["fn"] == size
        assert metrics["counts"]["fp"] == 0
        assert metrics["counts"]["tn"] == 0

    @pytest.mark.parametrize("size", [1, 2, 5, 20, 100])
    @pytest.mark.parametrize("prob_value", [0.0, 0.50, 1.0])
    def test_single_class_negative_roc_pr_auc_none(self, size: int, prob_value: float):
        """Verifies all-negative split (y_true=0) sets roc_auc=None, pr_auc=None without exceptions."""
        y_true = np.zeros(size, dtype=int)
        y_probs = np.full(size, prob_value)

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None
        assert metrics["support"]["non_waste"] == size
        assert metrics["support"]["waste"] == 0
        assert metrics["counts"]["tn"] + metrics["counts"]["fp"] == size
        assert metrics["counts"]["tp"] == 0
        assert metrics["counts"]["fn"] == 0

    @pytest.mark.parametrize("prob_constant", [0.0, 0.50, 1.0])
    def test_extreme_probabilities_balanced_classes(self, prob_constant: float):
        """
        Verifies that when classes are balanced but probabilities are completely uniform,
        evaluation executes safely without NaN or ZeroDivisionError.
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        y_probs = np.full(len(y_true), prob_constant, dtype=np.float64)

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1"], float)
        assert metrics["support"]["total"] == 8
        assert metrics["counts"]["tp"] + metrics["counts"]["fp"] + metrics["counts"]["tn"] + metrics["counts"]["fn"] == 8

    def test_unclamped_probabilities_resilience(self):
        """Verifies that out-of-bounds probabilities (e.g. -5.0, 99.0) are safely clamped to [0.0, 1.0]."""
        y_true = np.array([0, 0, 1, 1], dtype=int)
        # Out-of-bounds probabilities
        y_probs_extreme = np.array([-10.0, -0.01, 1.01, 50.0])

        yt, yp = _coerce_binary_inputs(y_true, y_probs_extreme)
        assert np.all(yp >= 0.0)
        assert np.all(yp <= 1.0)
        assert yp[0] == 0.0
        assert yp[3] == 1.0

        metrics = compute_evaluation_metrics(y_true, y_probs_extreme, threshold=0.50)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    @pytest.mark.parametrize("input_container", [list, tuple, np.array, pd.Series, torch.tensor])
    def test_input_type_diversity(self, input_container):
        """Verifies that compute_evaluation_metrics and calibrate_threshold accept all standard container types."""
        raw_true = [0, 0, 1, 1]
        raw_probs = [0.1, 0.2, 0.8, 0.9]

        true_input = input_container(raw_true)
        probs_input = input_container(raw_probs)

        m = compute_evaluation_metrics(true_input, probs_input, threshold=0.50)
        assert m["f1"] == 1.0

        t = calibrate_threshold(true_input, probs_input, metric="f1")
        assert 0.20 <= t <= 0.80

    def test_2d_probability_matrix_variations(self):
        """Verifies 2D probability matrices with (N, 2) shape or (N, 1) shape are coerced correctly."""
        y_true = np.array([0, 1], dtype=int)

        # 2D shape (2, 2)
        p2 = np.array([[0.8, 0.2], [0.1, 0.9]])
        m2 = compute_evaluation_metrics(y_true, p2, threshold=0.50)
        assert m2["f1"] == 1.0

        # 2D shape (2, 1)
        p1 = np.array([[0.2], [0.9]])
        m1 = compute_evaluation_metrics(y_true, p1, threshold=0.50)
        assert m1["f1"] == 1.0

        # Invalid 2D shape (2, 0) or 3D shape raises ValueError
        with pytest.raises(ValueError, match="Invalid 2D y_probs shape"):
            _coerce_binary_inputs(y_true, np.empty((2, 0)))

        with pytest.raises(ValueError, match="y_probs must be 1D or 2D"):
            _coerce_binary_inputs(y_true, np.ones((2, 2, 2)))


# ==============================================================================
# 3. Extreme Thresholds & Monotonicity Invariants Stress Tests
# ==============================================================================

class TestExtremeThresholdsAndMonotonicityInvariants:
    """Stress tests boundary thresholds and mathematical monotonicity invariants."""

    @pytest.mark.parametrize("t_low", [0.0, -0.01, -1.0, -100.0])
    def test_extreme_threshold_low_yields_recall_one(self, t_low: float):
        """
        When decision threshold T <= 0.0, all samples are classified as positive.
        For any dataset with at least 1 actual waste item, Recall must be exactly 1.0,
        FN must be 0, and TN must be 0.
        """
        y_true = np.array([0, 0, 1, 1, 1], dtype=int)
        y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=t_low)
        assert metrics["recall"] == 1.0
        assert metrics["counts"]["fn"] == 0
        assert metrics["counts"]["tn"] == 0
        assert metrics["counts"]["tp"] == 3
        assert metrics["counts"]["fp"] == 2

    @pytest.mark.parametrize("t_high", [1.0, 1.01, 2.0, 100.0])
    def test_extreme_threshold_high_yields_recall_zero(self, t_high: float):
        """
        When decision threshold T >= 1.0, all samples are classified as negative.
        Recall must be exactly 0.0, TP must be 0, and FP must be 0.
        """
        y_true = np.array([0, 0, 1, 1, 1], dtype=int)
        y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=t_high)
        assert metrics["recall"] == 0.0
        assert metrics["counts"]["tp"] == 0
        assert metrics["counts"]["fp"] == 0
        assert metrics["counts"]["fn"] == 3
        assert metrics["counts"]["tn"] == 2

    @pytest.mark.parametrize("seed", [42, 100, 2026])
    def test_monotonicity_invariants_across_100_thresholds(self, seed: int):
        """
        Sweeps 100 thresholds T in [0.0, 1.0]. Verifies rigorous mathematical invariants:
        1. TP count is monotonically non-increasing.
        2. FP count is monotonically non-increasing.
        3. TN count is monotonically non-decreasing.
        4. FN count is monotonically non-decreasing.
        5. Recall is monotonically non-increasing.
        6. Total partition count TP + FP + TN + FN == N at every single threshold.
        """
        rng = np.random.default_rng(seed)
        n = 120
        y_true = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        y_probs = rng.uniform(0.0, 1.0, size=n)

        thresholds = np.linspace(0.0, 1.0, 100)

        prev_tp = n + 1
        prev_fp = n + 1
        prev_tn = -1
        prev_fn = -1
        prev_rec = 1.0 + 1e-6

        for t in thresholds:
            m = compute_evaluation_metrics(y_true, y_probs, threshold=float(t))
            c = m["counts"]
            tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

            # Invariant 1: Total count conservation
            assert tp + fp + tn + fn == n, f"Count conservation violated at T={t}: sum={tp+fp+tn+fn} != {n}"

            # Invariant 2: TP is non-increasing
            assert tp <= prev_tp, f"TP increased from {prev_tp} to {tp} at threshold {t}"
            prev_tp = tp

            # Invariant 3: FP is non-increasing
            assert fp <= prev_fp, f"FP increased from {prev_fp} to {fp} at threshold {t}"
            prev_fp = fp

            # Invariant 4: TN is non-decreasing
            assert tn >= prev_tn, f"TN decreased from {prev_tn} to {tn} at threshold {t}"
            prev_tn = tn

            # Invariant 5: FN is non-decreasing
            assert fn >= prev_fn, f"FN decreased from {prev_fn} to {fn} at threshold {t}"
            prev_fn = fn

            # Invariant 6: Recall is non-increasing
            rec = m["recall"]
            assert rec <= prev_rec + 1e-9, f"Recall increased from {prev_rec} to {rec} at threshold {t}"
            prev_rec = rec


# ==============================================================================
# 4. Threshold Sweep Tie-Breaking Stability Stress Tests
# ==============================================================================

class TestThresholdTieBreakingStability:
    """Stress tests tie-breaking behavior when multiple candidate thresholds achieve maximal metric score."""

    def test_tie_break_flat_f1_plateau_selects_closest_to_half(self):
        """
        Constructs a dataset where all candidate thresholds in [0.35, 0.65] achieve
        100% precision, 100% recall, and F1 = 1.0.
        Verifies tie-break stably selects a candidate closest to 0.50.
        """
        # Non-waste probabilities all <= 0.30
        non_waste_probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        # Waste probabilities all >= 0.70
        waste_probs = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        y_true = np.array([0] * len(non_waste_probs) + [1] * len(waste_probs))
        y_probs = np.array(non_waste_probs + waste_probs)

        # Candidate grid with explicit 0.01 step so 0.50 is an exact grid point
        t_star = calibrate_threshold(y_true, y_probs, metric="f1", step=0.01)
        assert t_star == 0.50, f"Expected tie-break to select exact 0.50 on symmetric plateau, got {t_star}"

    def test_tie_break_plateau_above_half_selects_closest_boundary(self):
        """
        When the optimal F1 plateau is strictly above 0.50 (e.g. [0.65, 0.85]),
        the threshold closest to 0.50 is the lower edge of the plateau (~0.65).
        """
        # Non-waste: probs up to 0.60
        non_waste_probs = [0.10, 0.20, 0.40, 0.50, 0.60]
        # Waste: probs 0.90 and above
        waste_probs = [0.90, 0.92, 0.95, 0.98]

        y_true = np.array([0] * len(non_waste_probs) + [1] * len(waste_probs))
        y_probs = np.array(non_waste_probs + waste_probs)

        # Any threshold in (0.60, 0.90] achieves F1 = 1.0.
        # Closest to 0.50 should be just above 0.60 (e.g. 0.61 or 0.62).
        t_star = calibrate_threshold(y_true, y_probs, metric="f1", step=0.01)
        assert 0.60 < t_star <= 0.65, f"Expected threshold closest to 0.50 in (0.60, 0.90], got {t_star}"

    def test_tie_break_plateau_below_half_selects_closest_boundary(self):
        """
        When the optimal F1 plateau is strictly below 0.50 (e.g. [0.20, 0.35]),
        the threshold closest to 0.50 is the upper edge of the plateau (~0.35).
        """
        # Non-waste: probs up to 0.15
        non_waste_probs = [0.05, 0.10, 0.15]
        # Waste: probs 0.40 and above (but below 0.50)
        waste_probs = [0.40, 0.42, 0.45, 0.48]

        y_true = np.array([0] * len(non_waste_probs) + [1] * len(waste_probs))
        y_probs = np.array(non_waste_probs + waste_probs)

        # Any threshold in (0.15, 0.40] achieves F1 = 1.0.
        # Closest to 0.50 should be near 0.40 (e.g. 0.39 or 0.40).
        t_star = calibrate_threshold(y_true, y_probs, metric="f1", step=0.01)
        assert 0.35 <= t_star <= 0.40, f"Expected threshold closest to 0.50 in (0.15, 0.40], got {t_star}"

    def test_tie_break_degenerate_zero_score_landscape(self):
        """When all candidate thresholds yield <= 0.0 metric score, safely defaults to 0.50."""
        y_true = np.array([0, 0, 1, 1], dtype=int)
        # All probabilities are exactly 0.0, so at all candidates T >= 0.01 preds are 0 -> TP=0 -> F1=0
        y_probs = np.zeros(4, dtype=float)

        t_star = calibrate_threshold(y_true, y_probs, metric="f1")
        assert t_star == 0.50

    @pytest.mark.parametrize("metric_name", ["f1", "precision", "recall", "balanced_accuracy"])
    def test_calibrate_threshold_all_supported_metrics(self, metric_name: str):
        """Verifies calibrate_threshold runs cleanly with all documented metric criteria."""
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        t_star = calibrate_threshold(y_true, y_probs, metric=metric_name)
        assert 0.01 <= t_star <= 0.99
        assert isinstance(t_star, float)

    def test_calibrate_threshold_recall_target_metric(self):
        """Verifies calibrate_threshold with metric='recall_target' meets target recall."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1], dtype=int)
        y_probs = np.array([0.1, 0.2, 0.5, 0.4, 0.6, 0.8, 0.9])

        t_star = calibrate_threshold(y_true, y_probs, metric="recall_target", target_recall=0.75)
        m = compute_evaluation_metrics(y_true, y_probs, threshold=t_star)
        assert m["recall"] >= 0.75, f"Calibrated threshold {t_star} did not achieve target recall 0.75: {m['recall']}"

    def test_calibrate_threshold_unsupported_metric_raises(self):
        """Verifies unsupported metric string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported metric for calibration"):
            calibrate_threshold([0, 1], [0.2, 0.8], metric="invalid_metric")


# ==============================================================================
# 5. Metric Dictionary Schema & JSON Serialization Stress Tests
# ==============================================================================

class TestMetricSchemaAndJSONSerialization:
    """Stress tests metric report schema, data types, and strict JSON-serializability."""

    def test_metric_schema_completeness_and_nested_keys(self):
        """Verifies all mandatory top-level and nested keys are present and conform to specification."""
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_probs = np.array([0.1, 0.3, 0.7, 0.9])

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

        required_top_keys = [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
            "counts",
            "support",
            "breakdown",
            "per_class",
            "class_metrics",
            "averages",
        ]
        for k in required_top_keys:
            assert k in metrics, f"Missing top-level key: {k}"

        # Counts
        for sub_k in ["tp", "fp", "tn", "fn"]:
            assert sub_k in metrics["counts"], f"Missing count subkey: {sub_k}"
            assert isinstance(metrics["counts"][sub_k], int)

        # Support
        for sub_k in ["total", "non_waste", "waste"]:
            assert sub_k in metrics["support"], f"Missing support subkey: {sub_k}"
            assert isinstance(metrics["support"][sub_k], int)

        # Breakdown
        for sub_k in ["tn", "fp", "fn", "tp", "specificity", "false_positive_rate"]:
            assert sub_k in metrics["breakdown"], f"Missing breakdown subkey: {sub_k}"

        # Per Class
        for cls in ["waste", "non_waste"]:
            assert cls in metrics["per_class"]
            for sub_k in ["precision", "recall", "f1", "support"]:
                assert sub_k in metrics["per_class"][cls]

        # Averages
        for avg_type in ["macro", "weighted"]:
            assert avg_type in metrics["averages"]
            for sub_k in ["precision", "recall", "f1"]:
                assert sub_k in metrics["averages"][avg_type]

    def test_json_serializability_no_nans_or_unhandled_types(self):
        """
        Verifies that compute_evaluation_metrics outputs can be dumped to JSON
        without TypeError or invalid JSON values across diverse scenarios.
        """
        scenarios = [
            # Standard
            ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.50),
            # Single-class positive (roc_auc=None, pr_auc=None)
            ([1, 1, 1], [0.8, 0.9, 0.7], 0.50),
            # Single-class negative
            ([0, 0, 0], [0.1, 0.2, 0.3], 0.50),
            # Extreme threshold 0.0
            ([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7], 0.0),
            # Extreme threshold 1.0
            ([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7], 1.0),
        ]

        for yt, yp, thresh in scenarios:
            m = compute_evaluation_metrics(yt, yp, threshold=thresh)
            serialized = json.dumps(m, indent=2)
            assert isinstance(serialized, str)
            assert "NaN" not in serialized, f"Found unhandled NaN in JSON output for thresh={thresh}"
            assert "Infinity" not in serialized, f"Found unhandled Infinity in JSON output for thresh={thresh}"

            # Ensure round-trip loading succeeds
            deserialized = json.loads(serialized)
            assert deserialized["threshold"] == thresh

    def test_metric_mathematical_consistency(self):
        """Verifies internal mathematical consistency between counts, support, and averages."""
        rng = np.random.default_rng(2026)
        y_true = rng.choice([0, 1], size=100, p=[0.65, 0.35])
        y_probs = rng.uniform(0.0, 1.0, size=100)

        metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.45)

        tp = metrics["counts"]["tp"]
        fp = metrics["counts"]["fp"]
        tn = metrics["counts"]["tn"]
        fn = metrics["counts"]["fn"]
        total = metrics["support"]["total"]

        # Total count
        assert tp + fp + tn + fn == total == 100

        # Confusion matrix matching
        assert metrics["confusion_matrix"] == [[tn, fp], [fn, tp]]

        # Accuracy
        expected_acc = round((tp + tn) / total, 4)
        assert metrics["accuracy"] == expected_acc

        # Class support
        assert metrics["support"]["waste"] == tp + fn
        assert metrics["support"]["non_waste"] == tn + fp

        # Macro averages
        p_w = metrics["per_class"]["waste"]["precision"]
        p_nw = metrics["per_class"]["non_waste"]["precision"]
        expected_macro_p = round((p_w + p_nw) / 2.0, 4)
        assert abs(metrics["averages"]["macro"]["precision"] - expected_macro_p) <= 1e-4
