"""Post-hoc calibration: fixing the probabilities, then fixing the decision.

These are two different problems and conflating them is the usual reason a model
"needs recalibrating" and never improves.

**The probabilities are wrong in scale.** A network trained with focal loss on a
9% class is systematically overconfident -- focal loss deliberately keeps pushing
on hard examples long after the easy ones are settled, and the logits grow to
match. Temperature scaling (Guo et al., 2017) divides every logit by one fitted
scalar ``T``. Because it is a single monotone transform applied to all classes,
it **cannot change any argmax**: accuracy, recall and AUC are bit-for-bit
identical before and after. It changes only how much the number should be
believed, which is the part a clinician reads.

**The decision boundary is arbitrary.** 0.50 on ``P(lesion)`` corresponds to no
clinical policy. The policy this project actually wants is "catch at least 95% of
lesions, then be as specific as possible", and the threshold implementing that is
whatever the validation set says it is -- usually nowhere near 0.50.

Both are fitted on **validation** and then applied unchanged. Fitting either on
test and reporting the resulting numbers is the most common way an honest-looking
pipeline produces an inflated result.

What this module cannot do
--------------------------
Calibration is a monotone rescaling. It cannot fix a model that has not learned
the task: if the ranking is wrong, every threshold on it is also wrong. Check
``val_malignant_recall`` and PR-AUC before trusting anything here -- a model near
chance will happily produce a confident-looking calibrated threshold that means
nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASS_NAMES
from .utils import get_logger, load_json, save_json

logger = get_logger(__name__)

CALIBRATION_FILENAME = "calibration.json"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """Everything needed to turn raw logits into a calibrated clinical decision."""

    temperature: float = 1.0
    lesion_threshold: float = 0.5
    malignant_threshold: float = 0.5

    target_sensitivity: float = 0.95
    min_specificity: float = 0.50
    mode: str = "sensitivity_floor"
    achieved_sensitivity: float = float("nan")
    achieved_specificity: float = float("nan")
    youden_threshold: float = float("nan")
    # The same sweep read under the opposite constraint, so the trade-off the
    # chosen mode implies is visible next to it rather than hidden by it.
    alternative_threshold: float = float("nan")
    alternative_sensitivity: float = float("nan")
    alternative_specificity: float = float("nan")

    ece_before: float = float("nan")
    ece_after: float = float("nan")
    nll_before: float = float("nan")
    nll_after: float = float("nan")

    n_val: int = 0
    fitted_on: str = "val"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = load_json(path)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def for_checkpoint(cls, checkpoint: str | Path) -> Calibration | None:
        """Load the calibration sitting beside a checkpoint, if one was fitted."""
        path = Path(checkpoint).parent / CALIBRATION_FILENAME
        if not path.is_file():
            return None
        try:
            return cls.load(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("%s is unreadable (%s); falling back to defaults", path, exc)
            return None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature-scale logits and return probabilities."""
        return torch.softmax(logits.float() / max(self.temperature, 1e-6), dim=-1)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """ECE: mean gap between confidence and accuracy, weighted by bin population.

    0 is perfect. A model reporting 90% confidence on predictions that are right
    70% of the time scores roughly 0.20, and that gap is exactly what a reader
    misled by a confident number is exposed to.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Left-open bins so a confidence of exactly 1.0 lands in the last one.
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else confidence <= hi
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0)
    chosen = probabilities[np.arange(len(y_true)), np.asarray(y_true).astype(int)]
    return float(-np.log(chosen).mean())


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_temperature: float = 10.0,
    max_iter: int = 100,
) -> float:
    """Fit the single scalar ``T`` minimising validation NLL.

    ``T > 1`` means the model was overconfident and is being softened, which is
    the expected direction after focal-loss training. ``T < 1`` means it was
    underconfident, which usually indicates heavy label smoothing.

    Why this is not just LBFGS
    --------------------------
    The textbook implementation runs LBFGS on ``log T`` and takes what it
    returns. On this objective that is not safe: without a line search LBFGS
    overshoots in log space and settles up to **25x** away from the true
    optimum, and it does so silently -- there is no exception, just a confident
    wrong temperature that then miscalibrates every probability the app reports.
    Measured here on synthetic logits: a true optimum of 0.160 came back as
    0.008.

    So the fit is guarded. A log-spaced grid brackets the optimum, ternary
    search refines it (NLL is unimodal in ``log T``), and LBFGS with a strong
    Wolfe line search gets a vote. Whichever candidate actually has the lowest
    NLL wins. The grid alone would be accurate enough; keeping LBFGS costs
    almost nothing and lets it win when it is genuinely better.
    """
    tensor_logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    tensor_labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    lo, hi = float(np.log(1.0 / max_temperature)), float(np.log(max_temperature))

    def nll(log_temperature: float) -> float:
        with torch.no_grad():
            scaled = tensor_logits / float(np.exp(log_temperature))
            return float(F.cross_entropy(scaled, tensor_labels))

    # -- 1. coarse grid, so the search starts inside the right basin ---------
    grid = np.linspace(lo, hi, 64)
    best_log = float(grid[int(np.argmin([nll(g) for g in grid]))])

    # -- 2. ternary search on the bracketing interval ------------------------
    step = (hi - lo) / 63.0
    left, right = max(lo, best_log - step), min(hi, best_log + step)
    for _ in range(60):
        if right - left < 1e-6:
            break
        third = (right - left) / 3.0
        if nll(left + third) < nll(right - third):
            right -= third
        else:
            left += third
    candidates = [(nll(0.5 * (left + right)), 0.5 * (left + right)), (nll(best_log), best_log)]

    # -- 3. LBFGS, with the line search that makes it trustworthy ------------
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(tensor_logits / log_t.exp(), tensor_labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)  # type: ignore[arg-type]
        proposal = float(log_t.detach())
        if np.isfinite(proposal) and lo <= proposal <= hi:
            candidates.append((nll(proposal), proposal))
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - optimiser guard
        logger.debug("LBFGS temperature step failed (%s); using the grid result", exc)

    best_log_temperature = min(candidates)[1]
    best = float(np.exp(best_log_temperature))
    if not np.isfinite(best) or best <= 0:
        logger.warning("temperature fit produced %s; falling back to 1.0", best)
        return 1.0

    # The search is bounded by construction, so the result can never exceed the
    # limits -- which means a clamp would be dead code. What does need saying is
    # when the optimum lands *on* a boundary: that is the search reporting that
    # the true optimum lies outside the range it was allowed to consider, and
    # the returned value is a limit rather than a fit.
    if min(abs(best_log_temperature - lo), abs(best_log_temperature - hi)) < 1e-6:
        logger.warning(
            "temperature fit settled on the edge of its [%.3g, %.3g] search range "
            "(T=%.4f). The model is miscalibrated further than calibrate.max_temperature "
            "allows, so this is a bound, not an optimum -- raise that bound, or check "
            "whether the validation logits are degenerate.",
            1.0 / max_temperature, max_temperature, best,
        )
    return best


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def lesion_scores(y_prob: np.ndarray, normal_index: int = 0) -> np.ndarray:
    """``P(lesion) = 1 - P(normal)``, the quantity the app thresholds."""
    return 1.0 - np.asarray(y_prob, dtype=np.float64)[:, normal_index]


def sweep_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, normal_index: int = 0
) -> list[dict[str, float]]:
    """Sensitivity/specificity at every distinct score, for the full ROC picture."""
    scores = lesion_scores(y_prob, normal_index)
    positive = np.asarray(y_true).astype(int) != normal_index

    rows: list[dict[str, float]] = []
    for threshold in np.unique(np.round(scores, 4)):
        predicted = scores >= threshold
        sensitivity = float(predicted[positive].mean()) if positive.any() else float("nan")
        specificity = float((~predicted[~positive]).mean()) if (~positive).any() else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
            }
        )
    return rows


def find_operating_point(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.95,
    min_specificity: float = 0.50,
    normal_index: int = 0,
    mode: str = "sensitivity_floor",
) -> dict[str, Any]:
    """Pick a decision threshold under a one-sided clinical constraint.

    Two modes, because "high sensitivity and high specificity" is not a
    specification until one of them is named as the binding constraint:

    ``sensitivity_floor`` (default)
        Sensitivity must be at least ``target_sensitivity``; among thresholds
        that qualify, take the most specific. This is the triage framing: the
        cost of a missed lesion sets a hard floor, and false alarms are
        minimised underneath it.

    ``specificity_floor``
        Specificity must be at least ``min_specificity``; among thresholds that
        qualify, take the most sensitive. This is the framing for a tool that
        will be ignored if it cries wolf -- an alert firing on a third of normal
        films gets switched off, and a switched-off model has zero sensitivity
        in practice.

    Both are reported either way, so the trade is visible rather than assumed.
    Whichever mode is chosen, an infeasible constraint is reported rather than
    resolved: silently returning a threshold of 0.0 would score 100% sensitivity
    by calling every film a lesion, which satisfies the letter of the request and
    is worse than no answer.
    """
    if mode not in ("sensitivity_floor", "specificity_floor"):
        raise ValueError(
            f"unknown mode {mode!r}; expected 'sensitivity_floor' or 'specificity_floor'"
        )

    rows = sweep_thresholds(y_true, y_prob, normal_index)
    warnings: list[str] = []
    empty = {"threshold": 0.5, "sensitivity": float("nan"),
             "specificity": float("nan"), "youden_j": float("nan")}

    if mode == "sensitivity_floor":
        feasible = [r for r in rows if r["sensitivity"] >= target_sensitivity]
        if feasible:
            chosen = max(feasible, key=lambda r: (r["specificity"], r["threshold"]))
            if chosen["specificity"] < min_specificity:
                warnings.append(
                    f"the {target_sensitivity:.0%}-sensitivity threshold "
                    f"({chosen['threshold']:.3f}) yields only "
                    f"{chosen['specificity']:.3f} specificity, below the "
                    f"{min_specificity:.2f} floor. At this operating point the model "
                    "flags most normal films; it is not yet good enough to deploy at "
                    "this sensitivity."
                )
        else:
            best = max(rows, key=lambda r: r["sensitivity"]) if rows else None
            warnings.append(
                f"no threshold reaches {target_sensitivity:.0%} sensitivity "
                f"(best achievable {best['sensitivity']:.3f} at {best['threshold']:.3f}); "
                "the model cannot support this operating point -- train longer before "
                "tuning the threshold"
                if best
                else "threshold sweep produced no candidates"
            )
            chosen = best or empty
    else:
        feasible = [r for r in rows if r["specificity"] >= min_specificity]
        if feasible:
            chosen = min(feasible, key=lambda r: (-r["sensitivity"], r["threshold"]))
            if chosen["sensitivity"] < target_sensitivity:
                warnings.append(
                    f"holding specificity at {min_specificity:.2f} caps sensitivity at "
                    f"{chosen['sensitivity']:.3f}, below the {target_sensitivity:.0%} "
                    "target. The two constraints cannot both be met on this ROC curve -- "
                    "decide which one is the real requirement, or improve the model."
                )
        else:
            best = max(rows, key=lambda r: r["specificity"]) if rows else None
            warnings.append(
                f"no threshold reaches {min_specificity:.0%} specificity "
                f"(best achievable {best['specificity']:.3f} at {best['threshold']:.3f}); "
                "the model flags too many normal films at every operating point"
                if best
                else "threshold sweep produced no candidates"
            )
            chosen = best or empty

    youden = max(rows, key=lambda r: r["youden_j"]) if rows else chosen
    return {
        "threshold": float(chosen["threshold"]),
        "sensitivity": float(chosen["sensitivity"]),
        "specificity": float(chosen["specificity"]),
        "mode": mode,
        "youden_threshold": float(youden["threshold"]),
        "youden_j": float(youden["youden_j"]),
        "sweep": rows,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_logits(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run a split and return raw (uncalibrated, fp32) logits with their labels.

    Logits, not probabilities: temperature scaling has to be fitted before the
    softmax, and recovering logits from probabilities loses the scale that is
    the entire object of the fit.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            # No autocast: bf16 logits carry ~3 decimal digits, which is fine
            # for an argmax and far too coarse for an NLL minimisation.
            logits = model(images).float()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def calibrate(
    logits: np.ndarray,
    labels: np.ndarray,
    cfg,
    normal_index: int = 0,
) -> Calibration:
    """Fit temperature and the decision threshold on one set of validation logits."""
    calibrate_cfg = cfg.get("calibrate", None)

    def setting(key: str, default: Any) -> Any:
        return default if calibrate_cfg is None else calibrate_cfg.get(key, default)

    probabilities_before = torch.softmax(torch.as_tensor(logits), dim=1).numpy()

    temperature = 1.0
    if bool(setting("temperature", True)):
        temperature = fit_temperature(
            logits, labels, max_temperature=float(setting("max_temperature", 10.0))
        )
    probabilities_after = (
        torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()
    )

    target = float(setting("target_sensitivity", 0.95))
    floor = float(setting("min_specificity", 0.50))
    mode = str(setting("mode", "sensitivity_floor"))

    operating_point = find_operating_point(
        labels, probabilities_after,
        target_sensitivity=target, min_specificity=floor,
        normal_index=normal_index, mode=mode,
    )
    other = "specificity_floor" if mode == "sensitivity_floor" else "sensitivity_floor"
    alternative = find_operating_point(
        labels, probabilities_after,
        target_sensitivity=target, min_specificity=floor,
        normal_index=normal_index, mode=other,
    )

    result = Calibration(
        temperature=temperature,
        lesion_threshold=operating_point["threshold"],
        malignant_threshold=float(
            find_operating_point(
                (np.asarray(labels) == len(CLASS_NAMES) - 1).astype(int),
                np.column_stack(
                    [
                        1.0 - probabilities_after[:, -1],
                        probabilities_after[:, -1],
                    ]
                ),
                target_sensitivity=target,
                min_specificity=0.0,
                normal_index=0,
            )["threshold"]
        ),
        target_sensitivity=target,
        min_specificity=floor,
        mode=mode,
        alternative_threshold=alternative["threshold"],
        alternative_sensitivity=alternative["sensitivity"],
        alternative_specificity=alternative["specificity"],
        achieved_sensitivity=operating_point["sensitivity"],
        achieved_specificity=operating_point["specificity"],
        youden_threshold=operating_point["youden_threshold"],
        ece_before=expected_calibration_error(labels, probabilities_before),
        ece_after=expected_calibration_error(labels, probabilities_after),
        nll_before=negative_log_likelihood(labels, probabilities_before),
        nll_after=negative_log_likelihood(labels, probabilities_after),
        n_val=int(len(labels)),
        warnings=list(operating_point["warnings"]),
    )

    if result.ece_after > result.ece_before + 1e-6:
        result.warnings.append(
            f"temperature scaling made calibration worse (ECE {result.ece_before:.4f} "
            f"-> {result.ece_after:.4f}); the validation set may be too small to fit on"
        )
    return result


def format_report(calibration: Calibration) -> str:
    """Render a calibration as a scannable text block."""
    direction = (
        "overconfident, softened" if calibration.temperature > 1.0
        else "underconfident, sharpened" if calibration.temperature < 1.0
        else "unchanged"
    )
    lines = [
        "=" * 66,
        "CALIBRATION  (fitted on validation)",
        "=" * 66,
        f"  n                     {calibration.n_val}",
        f"  temperature           {calibration.temperature:.4f}   ({direction})",
        f"  ECE                   {calibration.ece_before:.4f} -> {calibration.ece_after:.4f}",
        f"  NLL                   {calibration.nll_before:.4f} -> {calibration.nll_after:.4f}",
        "",
        "  Decision threshold on P(lesion) = 1 - P(normal):",
        f"    mode                {calibration.mode}",
        f"    target sensitivity  {calibration.target_sensitivity:.2f}"
        f"   |  min specificity  {calibration.min_specificity:.2f}",
        f"    threshold           {calibration.lesion_threshold:.4f}   (vs the naive 0.50)",
        f"    sensitivity         {calibration.achieved_sensitivity:.4f}",
        f"    specificity         {calibration.achieved_specificity:.4f}",
        "",
        "  Under the opposite constraint, for comparison:",
        f"    threshold           {calibration.alternative_threshold:.4f}",
        f"    sensitivity         {calibration.alternative_sensitivity:.4f}",
        f"    specificity         {calibration.alternative_specificity:.4f}",
        "",
        f"    Youden-J threshold  {calibration.youden_threshold:.4f}   "
        "(balanced, ignores the clinical asymmetry)",
        "",
        f"  Malignant threshold   {calibration.malignant_threshold:.4f}",
        "",
        "  Temperature scaling is monotone, so accuracy, recall and AUC are",
        "  unchanged by definition. Only the confidence numbers move.",
    ]
    if calibration.warnings:
        lines += ["", "  WARNINGS:"]
        lines += [f"    ! {w}" for w in calibration.warnings]
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_FILENAME",
    "Calibration",
    "calibrate",
    "collect_logits",
    "expected_calibration_error",
    "find_operating_point",
    "fit_temperature",
    "format_report",
    "lesion_scores",
    "negative_log_likelihood",
    "sweep_thresholds",
]
