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


class HardNegativeMiningLoss(nn.Module):
    """OHEM specialised to false positives on normal films.

    Classic OHEM ranks the whole batch by loss and backpropagates only the worst
    fraction. This is deliberately narrower: it magnifies the loss on samples
    that are **truly normal** and that the model calls a lesion with confidence.
    Those are the errors this project is trying to remove, and they are not the
    same population as "high loss" in general -- a badly-missed malignant case
    also has high loss, and down-weighting the rest of the batch to chase
    normals would trade the error that matters for the error that annoys.

    Relationship to focal loss
    --------------------------
    Focal already up-weights hard examples, symmetrically across classes. This
    adds a *class-asymmetric* term on top: normals specifically. The two compose,
    but they are not independent -- turning both up hard will make the loss
    surface jump around, because a sample can be amplified twice. If training
    destabilises, lower ``gamma`` before lowering the penalty here.

    Guards that matter
    ------------------
    ``warmup_epochs``
        Mining from step zero is close to useless. An untrained network calls a
        large share of normal films lesions, so on epoch 1 the criterion selects
        most of the normal class and the "penalty" becomes a constant class
        weight -- which is what alpha already does, only less clearly. Mining
        starts once the model has something to be wrong about.
    ``max_fraction``
        Caps how much of a batch can be amplified. Without it, a bad epoch
        amplifies nearly every normal sample at once and the effective learning
        rate spikes.
    ``normalize``
        Rescales so the batch's mean weight stays 1. The relative emphasis on
        hard negatives is preserved -- that is the whole mechanism -- while the
        loss magnitude stays comparable epoch to epoch, so the LR schedule and
        the plateau detector keep meaning the same thing.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        normal_index: int = 0,
        confidence: float = 0.5,
        penalty: float = 4.0,
        max_fraction: float = 0.5,
        warmup_epochs: int = 3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if penalty < 1.0:
            raise ValueError(f"penalty must be >= 1 (1 disables mining), got {penalty}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if not 0.0 < max_fraction <= 1.0:
            raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")
        if getattr(base_loss, "reduction", "none") != "none":
            raise ValueError(
                "base_loss must be constructed with reduction='none' so per-sample "
                "weights can be applied"
            )

        self.base_loss = base_loss
        self.normal_index = int(normal_index)
        self.confidence = float(confidence)
        self.penalty = float(penalty)
        self.max_fraction = float(max_fraction)
        self.warmup_epochs = int(warmup_epochs)
        self.normalize = bool(normalize)

        self.current_epoch = 0
        # Running counters, reset per epoch by the training loop, so the log can
        # show whether mining is finding anything at all.
        self.n_mined = 0
        self.n_normal = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.n_mined = 0
        self.n_normal = 0

    @property
    def active(self) -> bool:
        return self.current_epoch >= self.warmup_epochs and self.penalty > 1.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample = self.base_loss(logits, targets)
        is_normal = targets == self.normal_index
        self.n_normal += int(is_normal.sum())

        if not self.active or not bool(is_normal.any()):
            return per_sample.mean()

        with torch.no_grad():
            # P(lesion) = 1 - P(normal), the same quantity the app thresholds, so
            # "hard negative" here means the same thing a user would see.
            lesion_probability = 1.0 - torch.softmax(logits.float(), dim=1)[
                :, self.normal_index
            ]
            hard = is_normal & (lesion_probability >= self.confidence)

            budget = int(self.max_fraction * logits.size(0))
            if int(hard.sum()) > budget:
                # Over budget: keep only the most confidently wrong, which are
                # the most informative and the least likely to be label noise.
                scores = torch.where(hard, lesion_probability, torch.zeros_like(
                    lesion_probability
                ))
                keep = torch.topk(scores, budget).indices
                limited = torch.zeros_like(hard)
                limited[keep] = True
                hard = hard & limited

            self.n_mined += int(hard.sum())
            weights = torch.where(
                hard,
                torch.full_like(per_sample, self.penalty),
                torch.ones_like(per_sample),
            )
            if self.normalize:
                weights = weights / weights.mean().clamp_min(1e-8)

        return (per_sample * weights).mean()

    def extra_repr(self) -> str:
        return (
            f"confidence={self.confidence}, penalty={self.penalty}, "
            f"max_fraction={self.max_fraction}, warmup_epochs={self.warmup_epochs}, "
            f"normalize={self.normalize}"
        )


class LesionSupervisionLoss(nn.Module):
    """Pixel supervision for the lesion head: weighted BCE, plus Dice where there
    is something to overlap with.

    WHY BOTH TERMS
    --------------
    Lesion pixels are a measured 2.79% of the frame on average (median 1.56%,
    over 400 BTXRD polygons). Unweighted BCE on that distribution has an obvious
    minimum -- predict zero everywhere, be right 97% of the time, and never
    localise anything. ``pos_weight`` removes that shortcut; the default 35 is
    ``(1 - 0.0279) / 0.0279``, i.e. derived from the data rather than guessed.

    Dice then supplies the gradient BCE is worst at: it scores overlap as a
    ratio, so it cares about a small lesion as much as a large one, where BCE's
    per-pixel sum does not.

    WHY DICE IS SKIPPED ON NORMAL FILMS
    -----------------------------------
    On an all-zero target Dice is 0/0. Guarding it with an epsilon technically
    "works" and quietly rewards predicting nothing anywhere, which would undo
    ``pos_weight``. So Dice runs only on images that contain a lesion, and the
    normals are supervised by BCE alone -- which is the correct division of
    labour: "there is no lesion on this healthy joint" is a per-pixel statement,
    not an overlap one.

    DOWNSAMPLING THE TARGET
    -----------------------
    The head predicts at 64x64 and the mask arrives at the model input size, so
    the target is reduced with **max** pooling rather than area averaging. Area
    averaging followed by a 0.5 threshold erases a lesion that is small relative
    to one output cell; max pooling keeps it and slightly dilates it instead.
    That is the right direction for the error to run in a cancer detector.
    """

    def __init__(self, pos_weight: float = 35.0, dice: bool = True, eps: float = 1.0) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError(f"pos_weight must be > 0, got {pos_weight}")
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice = bool(dice)
        self.eps = float(eps)

    def forward(self, mask_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
            target = target.unsqueeze(1)
        target = target.to(mask_logits.dtype)

        if target.shape[-2:] != mask_logits.shape[-2:]:
            target = F.adaptive_max_pool2d(target, mask_logits.shape[-2:])

        loss = F.binary_cross_entropy_with_logits(
            mask_logits, target, pos_weight=self.pos_weight.to(mask_logits.dtype)
        )

        if not self.dice:
            return loss

        has_lesion = target.flatten(1).amax(dim=1) > 0.5
        if not bool(has_lesion.any()):
            return loss

        probability = torch.sigmoid(mask_logits[has_lesion]).flatten(1)
        truth = target[has_lesion].flatten(1)
        intersection = (probability * truth).sum(dim=1)
        union = probability.sum(dim=1) + truth.sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps))
        return loss + dice_loss.mean()

    def extra_repr(self) -> str:
        return f"pos_weight={float(self.pos_weight):.1f}, dice={self.dice}, eps={self.eps}"


def build_lesion_loss(cfg) -> LesionSupervisionLoss | None:
    """Construct the lesion-map loss, or ``None`` when the head is disabled.

    Deliberately separate from :func:`build_loss` rather than folded into it.
    ``build_loss`` decides the *classification* objective and carries the
    sampler-versus-alpha exclusivity that ``train.build_sampler`` enforces; the
    two are combined by the training loop, where the weighting between them is
    visible, rather than hidden inside a loss that also has to reason about class
    imbalance.
    """
    if not bool(cfg.model.get("lesion_head", False)):
        return None
    lesion = cfg.loss.get("lesion", None) or {}
    return LesionSupervisionLoss(
        pos_weight=float(lesion.get("pos_weight", 35.0)),
        dice=bool(lesion.get("dice", True)),
    )


def lesion_weight(cfg, epoch: int) -> float:
    """The multiplier on the lesion loss at ``epoch``, with a linear warm-up.

    The decoder starts random, so its output for the first epochs is noise.
    Weighting noise at full strength competes with the classification gradient
    for the shared backbone and destabilises both -- the same reasoning that
    gives :class:`HardNegativeMiningLoss` its ``warmup_epochs``, and the same
    remedy.
    """
    lesion = cfg.loss.get("lesion", None) or {}
    target = float(lesion.get("weight", 0.5))
    warmup = int(lesion.get("warmup_epochs", 3))
    if warmup <= 0 or epoch >= warmup:
        return target
    return target * (epoch + 1) / (warmup + 1)


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

    ohem = cfg.loss.get("ohem", None)
    mine = ohem is not None and bool(ohem.get("enabled", False))
    # With mining on, the base loss must produce one value per sample so it can
    # be reweighted; the reduction to a scalar happens inside the wrapper.
    reduction = "none" if mine else "mean"

    base: nn.Module
    if name == "focal":
        base = FocalLoss(
            alpha=use_alpha,
            gamma=float(cfg.loss.gamma),
            label_smoothing=smoothing,
            reduction=reduction,
        )
    elif name == "weighted_ce":
        base = nn.CrossEntropyLoss(
            weight=use_alpha, label_smoothing=smoothing, reduction=reduction
        )
    elif name == "ce":
        base = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction=reduction)
    else:
        raise ValueError(f"unknown loss {name!r}; expected one of: focal, weighted_ce, ce")

    if not mine:
        return base

    classes = [str(c) for c in cfg.labels.classes]
    return HardNegativeMiningLoss(
        base,
        normal_index=classes.index("normal") if "normal" in classes else 0,
        confidence=float(ohem.get("confidence", 0.5)),
        penalty=float(ohem.get("penalty", 4.0)),
        max_fraction=float(ohem.get("max_fraction", 0.5)),
        warmup_epochs=int(ohem.get("warmup_epochs", 3)),
        normalize=bool(ohem.get("normalize", True)),
    )
