"""Tests for the clinical metric bundle.

The recurring theme: verify that the metrics actually punish the failure mode
this project exists to prevent. A model that never predicts malignant scores
90.9% accuracy on BTXRD's distribution, and several tests below exist purely to
confirm that such a model looks as bad as it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from onnm.metrics import (
    bootstrap_ci,
    clinical_errors,
    compute_metrics,
    malignant_recall_fn,
    per_class_rates,
    threshold_for_sensitivity,
    with_confidence_intervals,
)


def _one_hot(labels: list[int], confidence: float = 0.9) -> np.ndarray:
    """Probability rows peaked on the given class."""
    probs = np.full((len(labels), 3), (1 - confidence) / 2)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


# ---------------------------------------------------------------------------
# The metric that matters
# ---------------------------------------------------------------------------
def test_perfect_prediction_scores_one() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    metrics = compute_metrics(y_true, _one_hot(list(y_true)))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["malignant_recall"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_never_predicting_malignant_is_exposed() -> None:
    """The 91%-accurate do-nothing model must score 0 on the headline metric."""
    y_true = np.array([0] * 50 + [1] * 41 + [2] * 9)
    y_pred = np.zeros(100, dtype=int)          # always "normal"
    metrics = compute_metrics(y_true, _one_hot(list(y_pred)), y_pred=y_pred)

    assert metrics["accuracy"] == pytest.approx(0.50)
    assert metrics["malignant_recall"] == pytest.approx(0.0)
    assert metrics["clinical_errors"]["malignant_called_normal"] == 9
    assert metrics["clinical_errors"]["malignant_called_normal_rate"] == pytest.approx(1.0)


def test_sensitivity_and_specificity_are_correct() -> None:
    # 2 malignant caught, 1 missed; 1 normal wrongly called malignant.
    y_true = np.array([2, 2, 2, 0, 0, 0, 0])
    y_pred = np.array([2, 2, 0, 2, 0, 0, 0])
    rates = per_class_rates(y_true, y_pred)

    assert rates["malignant"]["sensitivity"] == pytest.approx(2 / 3)
    assert rates["malignant"]["ppv"] == pytest.approx(2 / 3)
    # Specificity over the 4 non-malignant cases, 1 of which was flagged.
    assert rates["malignant"]["specificity"] == pytest.approx(3 / 4)
    assert rates["malignant"]["support"] == 3


# ---------------------------------------------------------------------------
# Clinical error separation
# ---------------------------------------------------------------------------
def test_missed_cancer_types_are_separated() -> None:
    """Called-normal and called-benign are different clinical events."""
    y_true = np.array([2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 2])
    errors = clinical_errors(y_true, y_pred)

    assert errors["malignant_called_normal"] == 2    # sent home
    assert errors["malignant_called_benign"] == 1    # still followed up
    assert errors["malignant_total"] == 4
    assert errors["malignant_called_normal_rate"] == pytest.approx(0.5)


def test_overcalling_normals_is_reported() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([2, 2, 0, 0])
    assert clinical_errors(y_true, y_pred)["normal_called_malignant"] == 2


# ---------------------------------------------------------------------------
# AUCs
# ---------------------------------------------------------------------------
def test_pr_auc_is_reported_alongside_roc() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [2] * 10)
    probs = rng.random((100, 3))
    probs /= probs.sum(axis=1, keepdims=True)

    metrics = compute_metrics(y_true, probs)
    assert "pr_auc" in metrics and "roc_auc" in metrics
    assert np.isfinite(metrics["pr_auc"]["malignant"])
    # Random scores on a 10% class: PR-AUC sits near prevalence, ROC near 0.5.
    assert metrics["pr_auc"]["malignant"] < 0.4


def test_absent_class_yields_nan_not_a_crash() -> None:
    """A small val fold missing a class must not abort a training run."""
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, _one_hot([0, 0, 1, 1]))
    assert np.isnan(metrics["roc_auc"]["malignant"])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(7)
    y_true = np.array([0] * 200 + [1] * 150 + [2] * 49)
    probs = rng.random((len(y_true), 3))
    probs[np.arange(len(y_true)), y_true] += 1.2      # a decent but imperfect model
    probs /= probs.sum(axis=1, keepdims=True)

    ci = bootstrap_ci(y_true, probs, malignant_recall_fn, n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_boot"] > 0


def test_bootstrap_is_stratified() -> None:
    """Every replicate must keep the malignant count, or the CI widens for
    reasons unrelated to model uncertainty."""
    y_true = np.array([0] * 100 + [2] * 5)
    probs = _one_hot(list(y_true))

    ci = bootstrap_ci(y_true, probs, malignant_recall_fn, n_boot=100)
    # A perfect classifier resampled within class stays perfect.
    assert ci["lo"] == pytest.approx(1.0)
    assert ci["hi"] == pytest.approx(1.0)


def test_confidence_interval_bundle_has_headline_metrics() -> None:
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 60 + [1] * 40 + [2] * 20)
    probs = rng.random((120, 3))
    probs /= probs.sum(axis=1, keepdims=True)

    cis = with_confidence_intervals(y_true, probs, n_boot=100)
    assert set(cis) == {"malignant_recall", "malignant_pr_auc", "malignant_roc_auc"}
    for ci in cis.values():
        assert ci["lo"] <= ci["hi"]


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
def test_threshold_reaches_target_sensitivity() -> None:
    y_true = np.array([0] * 50 + [2] * 10)
    probs = np.zeros((60, 3))
    probs[:50, 0] = 0.9
    probs[:50, 2] = 0.1
    probs[50:, 2] = np.linspace(0.3, 0.95, 10)   # a spread of malignant scores

    result = threshold_for_sensitivity(y_true, probs, target=0.90)
    assert result["sensitivity"] >= 0.90
    assert 0.0 <= result["threshold"] <= 1.0


def test_threshold_without_positives_is_nan() -> None:
    y_true = np.zeros(10, dtype=int)
    assert np.isnan(threshold_for_sensitivity(y_true, np.zeros((10, 3)))["threshold"])
