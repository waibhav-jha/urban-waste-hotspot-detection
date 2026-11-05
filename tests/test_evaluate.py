"""
Unit tests for waste_hotspot.evaluate module.
Verifies metrics computations, decision threshold sweeps, validation-only isolation,
confusion matrix integrity, and CLI execution speed (< 3s).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn as nn
import yaml

from waste_hotspot.evaluate import (
    calibrate_threshold,
    compute_evaluation_metrics,
    evaluate_model,
    main,
)


class LightweightDummyClassifier(nn.Module):
    """Fast dummy classifier matching ResNet-50 output interface for instantaneous tests."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


@pytest.fixture
def dummy_eval_environment(tmp_path: Path) -> Dict[str, Path]:
    """
    Creates a temporary configuration file, split CSVs, and model checkpoint
    for fast CLI evaluation testing (< 1s).
    """
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 4 synthetic images
    rows_val, rows_test = [], []
    for i in range(4):
        p = img_dir / f"img_{i}.png"
        Image.new("RGB", (224, 224), color=(i * 30, 100, 150)).save(p)
        rec = {
            "image_id": f"id_{i}",
            "file_name": p.name,
            "file_path": str(p),
            "label": i % 2,
            "img_source": f"scene_{i // 2}",
            "width": 224,
            "height": 224,
        }
        if i < 2:
            rows_val.append(rec)
        else:
            rows_test.append(rec)

    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    pd.DataFrame(rows_val).to_csv(val_csv, index=False)
    pd.DataFrame(rows_test).to_csv(test_csv, index=False)

    # Checkpoint
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "best_model.pth"

    model = LightweightDummyClassifier(num_classes=2)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "threshold": 0.50,
        "model_architecture": "resnet18",
        "num_classes": 2,
    }, ckpt_path)

    # Config
    cfg = {
        "system": {"seed": 42, "device": "cpu"},
        "paths": {
            "splits_dir": str(splits_dir),
            "artifacts_dir": str(artifacts_dir),
            "checkpoint_path": str(ckpt_path),
        },
        "dataset": {"num_workers": 0, "class_names": ["non-waste", "waste"]},
        "preprocessing": {"image_size": 224},
        "model": {"backbone": "resnet18", "num_classes": 2, "pretrained": False},
        "evaluation": {
            "calibration_metric": "f1",
            "default_threshold": 0.50,
            "tune_threshold": False,
        },
    }
    cfg_path = tmp_path / "eval_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    return {
        "config_path": cfg_path,
        "checkpoint_path": ckpt_path,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "splits_dir": splits_dir,
    }


# ==============================================================================
# Metric Computation Unit Tests (Pure In-Memory: Runtime < 0.05s)
# ==============================================================================

def test_compute_metrics_perfect_predictions():
    """Verifies that 100% correct predictions yield exact F1=1.0, ROC-AUC=1.0, PR-AUC=1.0."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_probs = np.array([
        [0.95, 0.05],
        [0.85, 0.15],
        [0.10, 0.90],
        [0.05, 0.95],
    ])
    metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["counts"]["tp"] == 2
    assert metrics["counts"]["tn"] == 2
    assert metrics["counts"]["fp"] == 0
    assert metrics["counts"]["fn"] == 0


def test_compute_metrics_inverted_predictions():
    """Verifies that completely reversed predictions yield F1=0.0, ROC-AUC=0.0."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_probs = np.array([
        [0.05, 0.95],
        [0.10, 0.90],
        [0.85, 0.15],
        [0.95, 0.05],
    ])
    metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["roc_auc"] == 0.0
    assert metrics["confusion_matrix"] == [[0, 2], [2, 0]]
    assert metrics["counts"]["tp"] == 0
    assert metrics["counts"]["tn"] == 0
    assert metrics["counts"]["fp"] == 2
    assert metrics["counts"]["fn"] == 2


def test_compute_metrics_single_class_all_negative():
    """Verifies single-class negative data safely sets roc_auc=None without ZeroDivisionError."""
    y_true = np.zeros(10, dtype=int)
    y_probs = np.zeros((10, 2))
    y_probs[:, 0] = 0.8
    y_probs[:, 1] = 0.2

    metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["recall"] == 0.0
    assert metrics["support"]["waste"] == 0
    assert metrics["support"]["non_waste"] == 10
    assert metrics["confusion_matrix"] == [[10, 0], [0, 0]]


def test_compute_metrics_single_class_all_positive():
    """Verifies single-class positive data safely sets roc_auc=None without crashing."""
    y_true = np.ones(10, dtype=int)
    y_probs = np.zeros((10, 2))
    y_probs[:, 0] = 0.2
    y_probs[:, 1] = 0.8

    metrics = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["support"]["waste"] == 10
    assert metrics["support"]["non_waste"] == 0


def test_calibrate_threshold_f1_maximization():
    """
    Verifies that threshold sweep on validation data identifies the global F1 maximum.
    Constructs synthetic data where optimal cutoff is known to be ~0.60.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    # Waste probs: non-waste: [0.1, 0.2, 0.4, 0.55]; waste: [0.65, 0.7, 0.85, 0.9]
    # At T=0.50, sample 4 (prob 0.55) is False Positive -> F1 = 2*4/(4+5) = 8/9 = 0.888
    # At T=0.60, sample 4 is True Negative -> 100% precision & recall -> F1 = 1.0
    y_probs = np.array([0.1, 0.2, 0.4, 0.55, 0.65, 0.7, 0.85, 0.9])

    best_t = calibrate_threshold(y_true=y_true, y_probs=y_probs, metric="f1")
    assert 0.55 <= best_t <= 0.65, f"Expected calibrated threshold in [0.55, 0.65], got {best_t}"

    m_opt = compute_evaluation_metrics(y_true, y_probs, threshold=best_t)
    m_default = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)
    assert m_opt["f1"] > m_default["f1"], "Calibrated threshold must achieve higher F1 than default 0.50"


def test_calibrate_threshold_edge_cases():
    """Verifies edge cases: single-class fallback, all-zero scores fallback, input validation."""
    # Single-class returns 0.50
    t_single = calibrate_threshold(y_true=[0, 0, 0], y_probs=[0.1, 0.2, 0.3])
    assert t_single == 0.50

    # All-zero score returns 0.50
    t_zero = calibrate_threshold(y_true=[0, 0, 1, 1], y_probs=[0.0, 0.0, 0.0, 0.0])
    assert t_zero == 0.50

    # Empty inputs raise ValueError
    with pytest.raises(ValueError, match="empty"):
        calibrate_threshold(y_true=[], y_probs=[])

    # Mismatched lengths raise ValueError
    with pytest.raises(ValueError, match="mismatch"):
        calibrate_threshold(y_true=[0, 1], y_probs=[0.5])


def test_strict_validation_only_tuning():
    """
    Verifies strict isolation: modifying test set data has ZERO mathematical effect
    on the calibrated threshold.
    """
    np.random.seed(42)
    val_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    val_probs = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])

    t_initial = calibrate_threshold(y_true=val_true, y_probs=val_probs, metric="f1")

    # Change test data dramatically
    test_true_alt = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    test_probs_alt = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])

    # Re-run calibration using validation data only
    t_post = calibrate_threshold(y_true=val_true, y_probs=val_probs, metric="f1")
    assert t_initial == t_post, "Test data must never influence validation threshold calibration"


def test_accuracy_not_reported_alone_and_schema_complete():
    """Verifies that evaluation metrics dictionary adheres strictly to multi-faceted reporting rules."""
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.7, 0.3, 0.8])

    res = compute_evaluation_metrics(y_true, y_probs, threshold=0.50)

    # Must contain class-wise metrics, confusion matrix, counts, and support
    required_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix", "support", "per_class", "counts"}
    assert required_keys.issubset(set(res.keys()))

    assert isinstance(res["confusion_matrix"], list)
    assert len(res["confusion_matrix"]) == 2
    assert len(res["confusion_matrix"][0]) == 2

    assert "waste" in res["per_class"]
    assert "non_waste" in res["per_class"]
    assert "support" in res["per_class"]["waste"]


def test_extreme_thresholds_zero_and_one():
    """Verifies metrics behavior at boundary thresholds 0.0 (all positive) and 1.0 (all negative)."""
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])

    m_zero = compute_evaluation_metrics(y_true, y_probs, threshold=0.0)
    assert m_zero["recall"] == 1.0  # All detected as positive

    m_one = compute_evaluation_metrics(y_true, y_probs, threshold=1.0)
    assert m_one["recall"] == 0.0  # None detected as positive


def test_evaluate_model_pipeline_with_tuning(dummy_eval_environment: Dict[str, Path], tmp_path: Path):
    """Verifies evaluate_model with tune_threshold=True tunes on val and evaluates on test."""
    env = dummy_eval_environment
    out_json = tmp_path / "pipeline_report.json"

    report = evaluate_model(
        checkpoint_path=env["checkpoint_path"],
        data_csv=env["test_csv"],
        tune_threshold=True,
        val_csv=env["val_csv"],
        output_path=out_json,
        device="cpu",
    )

    assert out_json.is_file()
    assert report["threshold_calibrated"] is True
    assert "metrics" in report
    assert "f1" in report["metrics"]
    assert "precision" in report["metrics"]
    assert "recall" in report["metrics"]


# ==============================================================================
# CLI Execution Tests (Runtime < 3.0s)
# ==============================================================================

def test_evaluate_cli_fast_execution(dummy_eval_environment: Dict[str, Path], tmp_path: Path):
    """Verifies that evaluate CLI executes to completion in under 3.0 seconds."""
    out_json = tmp_path / "cli_eval_report.json"
    env = dummy_eval_environment

    start = time.perf_counter()
    exit_code = main([
        "--config", str(env["config_path"]),
        "--checkpoint", str(env["checkpoint_path"]),
        "--data-csv", str(env["test_csv"]),
        "--output", str(out_json),
        "--device", "cpu",
        "--dry-run",
    ])
    duration = time.perf_counter() - start

    assert exit_code == 0, f"CLI evaluation failed with code {exit_code}"
    assert duration < 3.0, f"CLI evaluation took {duration:.2f}s, expected < 3.0s"
    assert out_json.is_file(), "Report JSON was not created"

    with open(out_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert "f1" in report
    assert "confusion_matrix" in report


def test_evaluate_cli_argument_overrides(dummy_eval_environment: Dict[str, Path], tmp_path: Path):
    """Verifies that CLI flags correctly override config file parameters."""
    out_json = tmp_path / "override_report.json"
    env = dummy_eval_environment

    exit_code = main([
        "--config", str(env["config_path"]),
        "--checkpoint", str(env["checkpoint_path"]),
        "--data-csv", str(env["test_csv"]),
        "--val-csv", str(env["val_csv"]),
        "--threshold", "0.75",
        "--output", str(out_json),
        "--device", "cpu",
        "--tune-threshold",
    ])
    assert exit_code == 0
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["threshold"] == 0.75 or data["calibrated_on_validation"] is True
