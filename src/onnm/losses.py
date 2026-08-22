"""Loss functions for a heavily imbalanced 3-class problem.

BTXRD is 1879 normal / 1525 benign / 342 malignant, so malignant cases are 9.1%
of the data. Plain cross-entropy on that distribution converges happily to a
model that almost never predicts malignant: it is a cheap way to be right 91% of
the time and useless for the one call that matters.

Two corrections are available here. Apply exactly one of them. Combining a
weighted loss with a ``WeightedRandomSampler`` double-counts the imbalance -- the
rare class gets upweighted in the gradient *and* oversampled in the batch --
which in practice drives the model to over-predict malignant and destabilises
training. Over-calling normal controls is the visible symptom.

``onnm.train.build_sampler`` enforces the exclusivity rather than trusting the
config to be written correctly, because the failure is silent: the run trains,
the loss falls, and only specificity gives it away.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights easy examples by ``(1 - p_t) ** gamma`` so the gradient is
    dominated by the cases the model currently gets wrong. That matters here
    beyond raw class frequency: the *hard* examples are subtle early lesions,
    which is precisely the population this project exists to catch, while the
    easy ones are unremarkable normal films the model masters in one epoch.

    Args:
        alpha: Per-class weights, shape ``(C,)``. Typically inverse frequency
            from :func:`onnm.dataset.class_weights`.
        gamma: Focusing strength. 0 reduces to weighted cross-entropy; 2 is the
            standard choice and the configured default.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        label_smoothing: Optional smoothing applied to the underlying
            cross-entropy. Useful when subtype labels are noisy.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        # register_buffer so alpha follows the module across .to(device) and is
        # saved with the checkpoint -- a plain attribute would stay on the CPU
        # and raise a device mismatch on the first backward pass.
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # p_t recovered from the (already alpha-weighted) CE without a second
        # softmax: exp(-ce_unweighted). Use the unweighted log-prob for that.
        log_pt = F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - log_pt.exp()).pow(self.gamma)
        loss = focal_term * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        alpha = None if self.alpha is None else self.alpha.tolist()
        return f"gamma={self.gamma}, alpha={alpha}, label_smoothing={self.label_smoothing}"


def resolve_alpha(cfg, alpha: torch.Tensor | None) -> torch.Tensor | None:
    """Decide the final per-class weight vector from config and computed weights.

    Precedence, most explicit first:

    1. ``loss.alpha`` -- a hand-written list, e.g. ``[1.0, 1.0, 2.5]``. Use this
       when a validation sweep has produced a specific operating point and you
       want it pinned rather than re-derived.
    2. ``loss.auto_alpha: false`` -- no weighting at all. This is the correct
       setting when ``loader.balanced_sampler`` is on, because the sampler is
       already equalising the classes.
    3. otherwise the computed inverse-frequency weights, already tempered by
       ``loss.alpha_beta`` at the call site.
    """
    explicit = cfg.loss.get("alpha", None)
    if explicit is not None:
        vector = torch.as_tensor(list(explicit), dtype=torch.float32)
        expected = int(cfg.model.num_classes)
        if vector.numel() != expected:
            raise ValueError(
                f"loss.alpha has {vector.numel()} entries but the model has {expected} "
                f"classes; order must follow labels.classes"
            )
        if bool((vector < 0).any()):
            raise ValueError(f"loss.alpha must be non-negative, got {vector.tolist()}")
        return vector

    return alpha if bool(cfg.loss.get("auto_alpha", True)) else None


def build_loss(cfg, alpha: torch.Tensor | None = None) -> nn.Module:
    """Construct the configured loss.

    ``alpha`` should come from :func:`onnm.dataset.class_weights` computed on the
    **training split only**. Deriving it from the full dataset leaks test-set
    composition into training.
    """
    name = str(cfg.loss.name).lower()
    use_alpha = resolve_alpha(cfg, alpha)
    smoothing = float(cfg.loss.get("label_smoothing", 0.0))

    if name == "focal":
        return FocalLoss(
            alpha=use_alpha,
            gamma=float(cfg.loss.gamma),
            label_smoothing=smoothing,
        )
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=use_alpha, label_smoothing=smoothing)
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    raise ValueError(f"unknown loss {name!r}; expected one of: focal, weighted_ce, ce")
