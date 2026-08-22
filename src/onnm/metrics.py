"""Clinical metrics for the 3-class triage task.

Accuracy is close to useless here. Predicting "not malignant" for every image
scores 90.9% on BTXRD's distribution while missing every cancer, so this module
reports the quantities that actually distinguish a useful model:

* **per-class sensitivity (recall)** -- malignant recall is the headline number
* **specificity, PPV, NPV** -- what a clinician needs to interpret a positive
* **PR-AUC alongside ROC-AUC** -- ROC-AUC is optimistic on a 9% class because the
  large true-negative pool flatters the false-positive rate; PR-AUC is not
* **bootstrap confidence intervals on everything** -- the test split holds 49
  malignant images, so a single point estimate is noise dressed as a result
* **the malignant->normal cell, called out separately** -- confusing a cancer for
  a benign lesion still triggers follow-up; calling it normal sends the patient
  home, and averaging those two together hides the difference that matters
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from . import CLASS_NAMES, MALIGNANT_INDEX

NORMAL_INDEX = 0


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def per_class_rates(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """One-vs-rest sensitivity, specificity, PPV and NPV for every class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    out: dict[str, dict[str, float]] = {}

    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)

        sensitivity = _safe_divide(tp, tp + fn)
        ppv = _safe_divide(tp, tp + fp)
        out[CLASS_NAMES[idx]] = {
            "sensitivity": sensitivity,
            "specificity": _safe_divide(tn, tn + fp),
            "ppv": ppv,
            "npv": _safe_divide(tn, tn + fn),
            # Harmonic mean of PPV and sensitivity. Reported because it is a
            # common early-stopping monitor, but note what it ignores: F1 has no
            # term for true negatives, so it says nothing about how many normal
            # films get flagged. For that, read specificity.
            "f1": _safe_divide(2 * ppv * sensitivity, ppv + sensitivity),
            "support": int(tp + fn),
        }
    return out


def auc_scores(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3) -> dict:
    """One-vs-rest ROC-AUC and PR-AUC per class, plus macro averages.

    Returns NaN for a class absent from ``y_true`` rather than raising, so a
    small validation fold cannot abort a training run.
    """
    roc: dict[str, float] = {}
    pr: dict[str, float] = {}

    for idx in range(num_classes):
        binary = (y_true == idx).astype(int)
        name = CLASS_NAMES[idx]
        if binary.sum() == 0 or binary.sum() == len(binary):
            roc[name] = float("nan")
            pr[name] = float("nan")
            continue
        roc[name] = float(roc_auc_score(binary, y_prob[:, idx]))
        pr[name] = float(average_precision_score(binary, y_prob[:, idx]))

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "roc_auc_macro": float(np.nanmean(list(roc.values()))),
        "pr_auc_macro": float(np.nanmean(list(pr.values()))),
    }


def clinical_errors(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """Break out the error types that carry different clinical consequences."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    n_malignant = int(cm[MALIGNANT_INDEX, :].sum())

    missed_as_normal = int(cm[MALIGNANT_INDEX, NORMAL_INDEX])
    missed_as_benign = int(cm[MALIGNANT_INDEX, 1])

    return {
        # The worst outcome: a cancer called normal sends the patient home.
        "malignant_called_normal": missed_as_normal,
        "malignant_called_normal_rate": _safe_divide(missed_as_normal, n_malignant),
        # Serious but recoverable: a benign call still triggers follow-up.
        "malignant_called_benign": missed_as_benign,
        "malignant_called_benign_rate": _safe_divide(missed_as_benign, n_malignant),
        "malignant_total": n_malignant,
        # Over-calling costs biopsies and anxiety, so it belongs in the report too.
        "normal_called_malignant": int(cm[NORMAL_INDEX, MALIGNANT_INDEX]),
    }


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> list[dict[str, float]]:
    """Per-bin confidence vs accuracy — the data behind a reliability diagram.

    Scalar ECE compresses the whole calibration story into one number; a model
    can be overconfident above 0.9 and underconfident below 0.5 and still post
    a flattering average. These rows expose where on the confidence axis the
    gap lives. Binning matches ``expected_calibration_error`` in
    :mod:`onnm.calibrate` (equal-width, left-open) so the diagram and the
    scalar are two views of the same computation, not two computations.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else confidence <= hi
        count = int(mask.sum())
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": count,
                "mean_confidence": float(confidence[mask].mean()) if count else float("nan"),
                "accuracy": float(correct[mask].mean()) if count else float("nan"),
                "gap": (
                    float(confidence[mask].mean() - correct[mask].mean())
                    if count
                    else float("nan")
                ),
            }
        )
    return rows


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strata: Sequence[str],
    normal_index: int = NORMAL_INDEX,
    min_support: int = 5,
) -> dict[str, dict[str, Any]]:
    """Lesion-level error rates broken out by a per-sample stratum label.

    ``strata`` is any per-sample grouping — anatomy region, tumour subtype,
    view. The complaint driving this is specific ("false positives on complex
    joint anatomy"), so the report answers a specific question: *which strata
    produce the false positives, and which hide the missed lesions?*

    Strata with fewer than ``min_support`` samples are still reported but
    flagged ``low_support`` — at n=3 a single error is a 33-point rate swing
    and must not be read as a finding.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    strata = np.asarray(list(strata), dtype=object)
    if not (len(y_true) == len(y_pred) == len(strata)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"strata={len(strata)}"
        )

    true_lesion = y_true != normal_index
    pred_lesion = y_pred != normal_index

    out: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(s) for s in strata}):
        mask = strata == stratum
        n = int(mask.sum())
        n_lesion = int(true_lesion[mask].sum())
        n_normal = n - n_lesion

        false_positives = int((pred_lesion & ~true_lesion & mask).sum())
        missed_lesions = int((~pred_lesion & true_lesion & mask).sum())
        malignant_mask = mask & (y_true == MALIGNANT_INDEX)
        n_malignant = int(malignant_mask.sum())

        out[stratum] = {
            "n": n,
            "n_lesion": n_lesion,
            "n_normal": n_normal,
            "sensitivity": _safe_divide(n_lesion - missed_lesions, n_lesion),
            "specificity": _safe_divide(n_normal - false_positives, n_normal),
            "false_positives": false_positives,
            "false_positive_rate": _safe_divide(false_positives, n_normal),
            "missed_lesions": missed_lesions,
            "n_malignant": n_malignant,
            "malignant_recall": _safe_divide(
                int((malignant_mask & (y_pred == MALIGNANT_INDEX)).sum()), n_malignant
            ),
            "low_support": n < min_support,
        }
    return out


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None
) -> dict[str, Any]:
    """Full metric bundle for one evaluation pass."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_pred is None:
        y_pred = y_prob.argmax(axis=1)
    y_pred = np.asarray(y_pred).astype(int)

    num_classes = y_prob.shape[1]
    rates = per_class_rates(y_true, y_pred, num_classes)

    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(
            np.nanmean([rates[c]["sensitivity"] for c in CLASS_NAMES[:num_classes]])
        ),
        "per_class": rates,
        # Promoted to the top level because it is the early-stopping criterion
        # and the number the whole project is judged on.
        "malignant_recall": rates[CLASS_NAMES[MALIGNANT_INDEX]]["sensitivity"],
        "malignant_ppv": rates[CLASS_NAMES[MALIGNANT_INDEX]]["ppv"],
        "malignant_f1": rates[CLASS_NAMES[MALIGNANT_INDEX]]["f1"],
        # Macro F1 weights all three classes equally, so the 244-image malignant
        # class counts as much as the 1342-image normal one. That is the point.
        "f1_macro": float(
            np.nanmean([rates[c]["f1"] for c in CLASS_NAMES[:num_classes]])
        ),
        **auc_scores(y_true, y_prob, num_classes),
        "clinical_errors": clinical_errors(y_true, y_pred, num_classes),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict[str, float]:
    """Stratified bootstrap confidence interval for a scalar metric.

    Resampling is done **within each class** so every replicate keeps the same
    class balance as the test set. Unstratified resampling of 49 malignant cases
    occasionally draws a replicate with far fewer, which widens the interval for
    a reason that has nothing to do with model uncertainty.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)

    by_class = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    values: list[float] = []

    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pool, size=len(pool), replace=True) for pool in by_class]
        )
        try:
            value = float(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)

    if not values:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}

    arr = np.asarray(values)
    return {
        "point": float(metric_fn(y_true, y_prob)),
        "lo": float(np.percentile(arr, 100 * alpha / 2)),
        "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_boot": len(values),
    }


def malignant_recall_fn(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pred = y_prob.argmax(axis=1)
    mask = y_true == MALIGNANT_INDEX
    return float((pred[mask] == MALIGNANT_INDEX).mean()) if mask.any() else float("nan")


def malignant_pr_auc_fn(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    binary = (y_true == MALIGNANT_INDEX).astype(int)
    if binary.sum() in (0, len(binary)):
        return float("nan")
    return float(average_precision_score(binary, y_prob[:, MALIGNANT_INDEX]))


def malignant_roc_auc_fn(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    binary = (y_true == MALIGNANT_INDEX).astype(int)
    if binary.sum() in (0, len(binary)):
        return float("nan")
    return float(roc_auc_score(binary, y_prob[:, MALIGNANT_INDEX]))


def with_confidence_intervals(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for the three headline malignant metrics."""
    return {
        name: bootstrap_ci(y_true, y_prob, fn, n_boot=n_boot, alpha=alpha)
        for name, fn in (
            ("malignant_recall", malignant_recall_fn),
            ("malignant_pr_auc", malignant_pr_auc_fn),
            ("malignant_roc_auc", malignant_roc_auc_fn),
        )
    }


def threshold_for_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target: float = 0.90
) -> dict[str, float]:
    """Lowest malignant-probability threshold reaching ``target`` sensitivity.

    Chosen on the **validation** split and then applied unchanged to test.
    Tuning it on test and reporting the same numbers is the most common way an
    honest-looking pipeline produces an inflated result.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(y_prob, dtype=np.float64)[:, MALIGNANT_INDEX]
    positive = y_true == MALIGNANT_INDEX

    if not positive.any():
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan")}

    for threshold in np.unique(np.round(scores, 4))[::-1]:
        predicted = scores >= threshold
        sensitivity = float(predicted[positive].mean())
        if sensitivity >= target:
            specificity = float((~predicted[~positive]).mean())
            return {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "specificity": specificity,
            }

    return {"threshold": 0.0, "sensitivity": 1.0, "specificity": 0.0}


def format_report(metrics: dict, cis: dict | None = None) -> str:
    """Render the metric bundle as a scannable text block."""
    lines = [
        "=" * 66,
        "CLINICAL METRICS",
        "=" * 66,
        f"  n = {metrics['n']}   accuracy = {metrics['accuracy']:.3f}   "
        f"balanced accuracy = {metrics['balanced_accuracy']:.3f}",
        "",
        f"  {'class':<12}{'sens':>8}{'spec':>8}{'PPV':>8}{'NPV':>8}"
        f"{'F1':>8}{'ROC':>8}{'PR':>8}{'n':>7}",
        "  " + "-" * 70,
    ]

    for name, rates in metrics["per_class"].items():
        lines.append(
            f"  {name:<12}{rates['sensitivity']:>8.3f}{rates['specificity']:>8.3f}"
            f"{rates['ppv']:>8.3f}{rates['npv']:>8.3f}{rates['f1']:>8.3f}"
            f"{metrics['roc_auc'].get(name, float('nan')):>8.3f}"
            f"{metrics['pr_auc'].get(name, float('nan')):>8.3f}{rates['support']:>7}"
        )

    lines += ["", f"  macro ROC-AUC {metrics['roc_auc_macro']:.3f}   "
                  f"macro PR-AUC {metrics['pr_auc_macro']:.3f}   "
                  f"macro F1 {metrics['f1_macro']:.3f}"]

    if cis:
        lines += ["", "  Bootstrap 95% CIs (stratified):"]
        for name, ci in cis.items():
            lines.append(
                f"    {name:<20}{ci['point']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]"
            )

    errors = metrics["clinical_errors"]
    lines += [
        "",
        "  Clinical error breakdown:",
        f"    malignant called NORMAL : {errors['malignant_called_normal']:>4} / "
        f"{errors['malignant_total']}  ({errors['malignant_called_normal_rate']:.1%})"
        "   <- patient sent home",
        f"    malignant called benign : {errors['malignant_called_benign']:>4} / "
        f"{errors['malignant_total']}  ({errors['malignant_called_benign_rate']:.1%})"
        "   <- still followed up",
        f"    normal called malignant : {errors['normal_called_malignant']:>4}"
        "         <- unnecessary workup",
        "",
        "  Confusion matrix (rows = true, cols = predicted):",
        f"    {'':<12}" + "".join(f"{n:>12}" for n in CLASS_NAMES),
    ]
    for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"], strict=False):
        lines.append(f"    {name:<12}" + "".join(f"{v:>12}" for v in row))

    return "\n".join(lines)
