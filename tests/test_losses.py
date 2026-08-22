"""Tests for the imbalance-aware losses."""

from __future__ import annotations

import pytest
import torch

from onnm.losses import FocalLoss, build_loss


def test_focal_reduces_to_cross_entropy_at_gamma_zero() -> None:
    """gamma=0 must be exactly weighted cross-entropy -- a useful ablation anchor."""
    logits = torch.randn(16, 3, generator=torch.Generator().manual_seed(0))
    targets = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))

    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    torch.testing.assert_close(focal, ce)


def test_focal_downweights_easy_examples() -> None:
    """The whole point: confident-correct cases must contribute far less loss."""
    easy = torch.tensor([[10.0, 0.0, 0.0]])
    hard = torch.tensor([[0.6, 0.5, 0.4]])
    target = torch.tensor([0])

    ce_ratio = (
        torch.nn.functional.cross_entropy(easy, target)
        / torch.nn.functional.cross_entropy(hard, target)
    )
    focal_ratio = FocalLoss(gamma=2.0)(easy, target) / FocalLoss(gamma=2.0)(hard, target)

    assert focal_ratio < ce_ratio, "focal loss did not suppress the easy example"


def test_alpha_upweights_the_rare_class() -> None:
    logits = torch.tensor([[0.5, 0.3, 0.2]])
    alpha = torch.tensor([0.3, 0.6, 3.0])       # malignant heavily weighted

    normal = FocalLoss(alpha=alpha)(logits, torch.tensor([0]))
    malignant = FocalLoss(alpha=alpha)(logits, torch.tensor([2]))
    assert malignant > normal


def test_alpha_is_a_buffer_and_follows_the_module() -> None:
    """A plain attribute would stay on the CPU and crash on the first backward."""
    loss = FocalLoss(alpha=torch.tensor([1.0, 2.0, 3.0]))
    assert "alpha" in dict(loss.named_buffers())
    assert "alpha" in loss.state_dict()


def test_gradients_flow() -> None:
    logits = torch.randn(8, 3, requires_grad=True)
    FocalLoss(gamma=2.0)(logits, torch.randint(0, 3, (8,))).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_stay_finite() -> None:
    """Saturated logits must not produce NaN -- bf16 training reaches them."""
    logits = torch.tensor([[100.0, -100.0, -100.0], [-100.0, -100.0, 100.0]])
    loss = FocalLoss(gamma=2.0)(logits, torch.tensor([0, 2]))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions(reduction: str) -> None:
    logits = torch.randn(6, 3)
    targets = torch.randint(0, 3, (6,))
    loss = FocalLoss(reduction=reduction)(logits, targets)

    assert loss.shape == (6,) if reduction == "none" else loss.ndim == 0


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="average")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_build_loss_returns_configured_type(cfg) -> None:
    alpha = torch.tensor([0.7, 0.8, 3.7])
    assert isinstance(build_loss(cfg, alpha=alpha), FocalLoss)

    cfg._data["loss"]["name"] = "weighted_ce"
    assert isinstance(build_loss(cfg, alpha=alpha), torch.nn.CrossEntropyLoss)

    cfg._data["loss"]["name"] = "ce"
    assert build_loss(cfg, alpha=alpha).weight is None


def test_build_loss_rejects_unknown_name(cfg) -> None:
    cfg._data["loss"]["name"] = "dice"
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss(cfg)


def test_auto_alpha_can_be_disabled(cfg) -> None:
    cfg._data["loss"]["auto_alpha"] = False
    assert build_loss(cfg, alpha=torch.tensor([1.0, 2.0, 3.0])).alpha is None
