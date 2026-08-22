"""Plots and a self-contained HTML report for one evaluation run.

Deliberately produces a single local HTML file with images inlined as base64.
Two reasons: it opens anywhere without a server, and BTXRD's CC BY-NC-ND licence
makes uploading derived radiographs to a host the wrong default.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from . import CLASS_NAMES
from .utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def plot_training_history(history: list[dict], out_path: Path | None = None):
    """Loss and malignant recall against epoch.

    The two panels usually diverge, which is the point: validation loss keeps
    falling while malignant recall plateaus, because loss is dominated by the
    91% of images that are not malignant.
    """
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_recall) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_recall.plot(epochs, [h["val_malignant_recall"] for h in history],
                   color="#E45756", label="malignant recall")
    pr = [h.get("val_pr_auc_malignant") for h in history]
    if any(v is not None for v in pr):
        ax_recall.plot(epochs, pr, color="#4C78A8", linestyle="--", label="malignant PR-AUC")
    ax_recall.set(xlabel="epoch", ylabel="score", title="Validation (clinical)", ylim=(0, 1))
    ax_recall.legend()
    ax_recall.grid(alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(ensure_dir(Path(out_path).parent) / Path(out_path).name, dpi=110)
    return fig


def plot_confusion_matrix(matrix: list[list[int]], out_path: Path | None = None):
    """Confusion matrix with the clinically critical cell outlined."""
    import matplotlib.pyplot as plt

    array = np.asarray(matrix, dtype=float)
    normalised = array / np.maximum(array.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, f"{int(array[i, j])}\n{normalised[i, j]:.1%}",
                    ha="center", va="center",
                    color="white" if normalised[i, j] > 0.5 else "black", fontsize=10)

    # malignant (row 2) predicted normal (col 0): the patient goes home.
    if array.shape[0] > 2:
        ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False,
                                   edgecolor="#E45756", linewidth=3))

    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xlabel="predicted", ylabel="true",
           title="Confusion matrix (red = missed cancer called normal)")
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path | None = None):
    """ROC and precision-recall curves, one line per class.

    Both are shown because they disagree in an informative way on a 9% class:
    ROC looks strong while PR reveals how much precision costs.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, name in enumerate(CLASS_NAMES[: y_prob.shape[1]]):
        binary = (y_true == idx).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, idx])
        ax_roc.plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, idx])
        ax_pr.plot(recall, precision, label=f"{name} (prev {binary.mean():.1%})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (one-vs-rest)")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    ax_pr.set(xlabel="recall", ylabel="precision",
              title="Precision-Recall (honest on rare classes)")
    ax_pr.legend()
    ax_pr.grid(alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig


def write_html_report(
    metrics: dict[str, Any],
    confidence_intervals: dict,
    output_dir: Path,
    history: list[dict] | None = None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    title: str = "ONNM evaluation",
) -> Path:
    """Write a single self-contained HTML file with metrics and inlined plots."""
    import matplotlib
    matplotlib.use("Agg")

    output_dir = ensure_dir(output_dir)
    figures: list[tuple[str, str]] = []

    if history:
        figures.append(("Training history", _figure_to_base64(plot_training_history(history))))
    figures.append(
        ("Confusion matrix", _figure_to_base64(plot_confusion_matrix(metrics["confusion_matrix"])))
    )
    if y_true is not None and y_prob is not None:
        figures.append(("ROC / PR curves", _figure_to_base64(plot_curves(y_true, y_prob))))

    errors = metrics["clinical_errors"]
    rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{r['sensitivity']:.3f}</td><td>{r['specificity']:.3f}</td>"
        f"<td>{r['ppv']:.3f}</td><td>{r['npv']:.3f}</td>"
        f"<td>{metrics['roc_auc'].get(name, float('nan')):.3f}</td>"
        f"<td>{metrics['pr_auc'].get(name, float('nan')):.3f}</td>"
        f"<td>{r['support']}</td></tr>"
        for name, r in metrics["per_class"].items()
    )
    ci_rows = "".join(
        f"<tr><td>{name}</td><td>{ci['point']:.3f}</td>"
        f"<td>[{ci['lo']:.3f}, {ci['hi']:.3f}]</td></tr>"
        for name, ci in confidence_intervals.items()
    )
    plots = "".join(
        f"<h2>{caption}</h2><img src='data:image/png;base64,{data}'/>"
        for caption, data in figures
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>
 body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1000px;
       margin:2rem auto;padding:0 1.5rem;color:#1a1a1a}}
 table{{border-collapse:collapse;margin:1rem 0;width:100%}}
 th,td{{border:1px solid #ddd;padding:.45rem .7rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f5f5f5}} img{{max-width:100%;margin:.5rem 0}}
 .warn{{background:#fff4f4;border-left:4px solid #E45756;padding:.8rem 1rem;margin:1rem 0}}
 code{{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px}}
</style></head><body>
<h1>{title}</h1>
<p>n = {metrics['n']} &middot; accuracy {metrics['accuracy']:.3f} &middot;
   balanced accuracy {metrics['balanced_accuracy']:.3f}</p>
<div class="warn"><b>Accuracy is not the headline.</b> Predicting "not malignant" for every
image scores about 90.9% on this distribution while missing every cancer. Read malignant
recall and the clinical error breakdown instead.</div>

<h2>Per-class</h2>
<table><tr><th>class</th><th>sens</th><th>spec</th><th>PPV</th><th>NPV</th>
<th>ROC-AUC</th><th>PR-AUC</th><th>n</th></tr>{rows}</table>

<h2>Bootstrap 95% confidence intervals</h2>
<table><tr><th>metric</th><th>point</th><th>95% CI</th></tr>{ci_rows}</table>
<p>Stratified resampling. With roughly 49 malignant test images, the interval — not the
point estimate — is the result.</p>

<h2>Clinical errors</h2>
<table>
<tr><th>error</th><th>count</th><th>rate</th></tr>
<tr><td>malignant called <b>normal</b> (patient sent home)</td>
    <td>{errors['malignant_called_normal']}</td>
    <td>{errors['malignant_called_normal_rate']:.1%}</td></tr>
<tr><td>malignant called benign (still followed up)</td>
    <td>{errors['malignant_called_benign']}</td>
    <td>{errors['malignant_called_benign_rate']:.1%}</td></tr>
<tr><td>normal called malignant (unnecessary workup)</td>
    <td>{errors['normal_called_malignant']}</td><td>&ndash;</td></tr>
</table>
{plots}
<hr><p><small>Research software, not a medical device. Data: BTXRD, CC BY-NC-ND 4.0 —
derived images must not be redistributed.</small></p>
</body></html>"""

    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    logger.info("wrote %s", path)
    return path
