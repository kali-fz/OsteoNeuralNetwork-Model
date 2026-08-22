"""Tests for class balancing, augmentation strength, and the foreground crop.

The theme is that the dangerous failures here are all silent. Double-correcting
the class imbalance still trains. A crop that breaks the Grad-CAM box geometry
still produces heatmaps. Both would be found weeks later, in a confusion matrix
or a localisation score, with nothing pointing at the cause -- so both are made
loud at the point they are configured.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from onnm.config import ConfigError
from onnm.dataset import build_sampler, build_transforms, class_weights, sample_weights
from onnm.losses import build_loss, resolve_alpha


@pytest.fixture
def imbalanced_records() -> list[dict]:
    """BTXRD's shape in miniature: 55% normal, 39% benign, 6% malignant."""
    return (
        [{"label": 0}] * 550 + [{"label": 1}] * 390 + [{"label": 2}] * 60
    )


# ---------------------------------------------------------------------------
# Tempered class weights
# ---------------------------------------------------------------------------
def test_beta_zero_disables_weighting(imbalanced_records: list[dict]) -> None:
    assert torch.allclose(
        class_weights(imbalanced_records, beta=0.0), torch.ones(3), atol=1e-6
    )


def test_beta_one_is_full_inverse_frequency(imbalanced_records: list[dict]) -> None:
    weights = class_weights(imbalanced_records, beta=1.0)
    assert weights[2] > weights[1] > weights[0]
    # 550 normal vs 60 malignant is a 9.2x ratio in the raw counts.
    assert float(weights[2] / weights[0]) == pytest.approx(550 / 60, rel=0.01)


def test_beta_interpolates_monotonically(imbalanced_records: list[dict]) -> None:
    """The knob has to move the trade-off smoothly to be tunable at all."""
    ratios = [
        float(class_weights(imbalanced_records, beta=b)[2] / class_weights(
            imbalanced_records, beta=b)[0])
        for b in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(a < b for a, b in zip(ratios, ratios[1:], strict=False))


def test_weights_normalise_to_mean_one(imbalanced_records: list[dict]) -> None:
    """Keeps the loss on the same scale, so the LR need not be retuned."""
    for beta in (0.0, 0.5, 1.0, 2.0):
        assert float(class_weights(imbalanced_records, beta=beta).mean()) == pytest.approx(1.0)


def test_negative_beta_rejected(imbalanced_records: list[dict]) -> None:
    with pytest.raises(ValueError, match="beta must be >= 0"):
        class_weights(imbalanced_records, beta=-1.0)


# ---------------------------------------------------------------------------
# Explicit alpha
# ---------------------------------------------------------------------------
def test_explicit_alpha_overrides_everything(cfg) -> None:
    cfg._data["loss"]["alpha"] = [1.0, 1.0, 3.0]
    resolved = resolve_alpha(cfg, torch.tensor([9.0, 9.0, 9.0]))
    assert torch.allclose(resolved, torch.tensor([1.0, 1.0, 3.0]))


def test_alpha_length_must_match_class_count(cfg) -> None:
    cfg._data["loss"]["alpha"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="order must follow labels.classes"):
        resolve_alpha(cfg, None)


def test_negative_alpha_rejected(cfg) -> None:
    cfg._data["loss"]["alpha"] = [1.0, -1.0, 2.0]
    with pytest.raises(ValueError, match="non-negative"):
        resolve_alpha(cfg, None)


def test_auto_alpha_off_means_no_weighting(cfg) -> None:
    cfg._data["loss"]["auto_alpha"] = False
    assert resolve_alpha(cfg, torch.tensor([1.0, 2.0, 3.0])) is None


def test_explicit_alpha_reaches_the_loss(cfg) -> None:
    cfg._data["loss"]["alpha"] = [1.0, 1.0, 4.0]
    loss = build_loss(cfg, alpha=None)
    assert torch.allclose(loss.alpha, torch.tensor([1.0, 1.0, 4.0]))


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------
def test_sample_weights_equalise_total_class_mass(imbalanced_records: list[dict]) -> None:
    """Each class must contribute the same total draw probability."""
    weights = sample_weights(imbalanced_records).numpy()
    labels = np.array([r["label"] for r in imbalanced_records])
    mass = [weights[labels == c].sum() for c in (0, 1, 2)]
    assert mass[0] == pytest.approx(mass[1]) == pytest.approx(mass[2])


def test_sampler_actually_balances_a_draw(cfg, imbalanced_records: list[dict]) -> None:
    """The end-to-end property: drawn batches are roughly equal across classes."""
    cfg._data["loader"]["balanced_sampler"] = True
    cfg._data["loss"]["auto_alpha"] = False

    sampler = build_sampler(cfg, imbalanced_records)
    torch.manual_seed(0)
    labels = np.array([imbalanced_records[i]["label"] for i in list(sampler)])
    counts = np.bincount(labels, minlength=3) / len(labels)

    # Sampling noise over 1000 draws is a couple of percent; 0.05 is comfortable
    # while still failing loudly if the weighting is wrong.
    assert np.allclose(counts, 1 / 3, atol=0.05), counts


def test_sampler_off_by_default(cfg, imbalanced_records: list[dict]) -> None:
    assert build_sampler(cfg, imbalanced_records) is None


def test_sampler_and_weighted_loss_together_is_refused(
    cfg, imbalanced_records: list[dict]
) -> None:
    """The silent failure this guard exists for.

    Both corrections applied at once makes the model over-predict malignant,
    which presents as normal films being called lesions -- the exact complaint
    that motivates balanced sampling in the first place. Training would run
    happily, so the conflict is caught at construction.
    """
    cfg._data["loader"]["balanced_sampler"] = True
    cfg._data["loss"]["auto_alpha"] = True

    with pytest.raises(ConfigError, match="both enabled"):
        build_sampler(cfg, imbalanced_records)


def test_explicit_alpha_also_trips_the_guard(cfg, imbalanced_records: list[dict]) -> None:
    cfg._data["loader"]["balanced_sampler"] = True
    cfg._data["loss"]["auto_alpha"] = False
    cfg._data["loss"]["alpha"] = [1.0, 1.0, 2.0]

    with pytest.raises(ConfigError):
        build_sampler(cfg, imbalanced_records)


def test_plain_ce_is_compatible_with_the_sampler(
    cfg, imbalanced_records: list[dict]
) -> None:
    """Unweighted CE applies no correction, so the sampler is free to."""
    cfg._data["loader"]["balanced_sampler"] = True
    cfg._data["loss"]["name"] = "ce"
    assert build_sampler(cfg, imbalanced_records) is not None


# ---------------------------------------------------------------------------
# Augmentation and cropping
# ---------------------------------------------------------------------------
def test_augmentation_only_applies_to_train(cfg) -> None:
    train = [type(t).__name__ for t in build_transforms(cfg, "train").transforms]
    test = [type(t).__name__ for t in build_transforms(cfg, "test").transforms]
    for name in ("RandFlipd", "RandRotated", "RandAdjustContrastd", "RandGaussianNoised"):
        assert name in train
        assert name not in test


def test_crop_is_off_by_default(cfg) -> None:
    names = [type(t).__name__ for t in build_transforms(cfg, "test").transforms]
    assert "CropForegroundd" not in names


def test_crop_is_inserted_before_the_resize(cfg) -> None:
    """Order matters: cropping after the resize would crop the padding."""
    cfg._data["data"]["crop_foreground"] = True
    names = [type(t).__name__ for t in build_transforms(cfg, "test").transforms]
    assert names.index("CropForegroundd") < names.index("Resized")
    assert names.index("ScaleIntensityRangePercentilesd") < names.index("CropForegroundd")


def test_crop_strips_a_black_border(cfg, tmp_path) -> None:
    """The point of the crop: collimation borders should not survive it."""
    from PIL import Image

    bordered = np.zeros((200, 200), dtype=np.uint8)
    bordered[60:140, 60:140] = 200          # the "bone", centred in black
    path = tmp_path / "bordered.png"
    Image.fromarray(bordered).save(path)

    cfg._data["data"]["crop_foreground"] = True
    cropped = build_transforms(cfg, "test")({"image": str(path), "label": 0})["image"]

    cfg._data["data"]["crop_foreground"] = False
    uncropped = build_transforms(cfg, "test")({"image": str(path), "label": 0})["image"]

    # Cropping to the bright square then resizing means it fills far more of
    # the frame, so the mean intensity rises.
    assert float(cropped.mean()) > float(uncropped.mean())


def test_crop_survives_an_all_black_image(cfg, tmp_path) -> None:
    """A blank film must not take the run down with an empty crop."""
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(path)

    cfg._data["data"]["crop_foreground"] = True
    result = build_transforms(cfg, "test")({"image": str(path), "label": 0})["image"]
    assert result.shape[-2:] == (int(cfg.data.image_size), int(cfg.data.image_size))


def test_localisation_refuses_to_score_under_a_crop(cfg) -> None:
    """Grad-CAM box scoring must not silently report through a changed geometry.

    `map_box_to_model_space` models exactly resize-then-pad. A crop inserts a
    per-image offset it does not know about, so every pointing-game hit would
    still be a number and all of them would be wrong.
    """
    from onnm.explainability import evaluate_localisation

    cfg._data["data"]["crop_foreground"] = True
    with pytest.raises(ValueError, match="crop_foreground"):
        evaluate_localisation(torch.nn.Identity(), cfg, [], torch.device("cpu"))
