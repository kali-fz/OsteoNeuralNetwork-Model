"""The mask must describe the same pixels the image does, always.

WHY THIS FILE IS THE MOST IMPORTANT NEW TEST
--------------------------------------------
Every other failure in the lesion-head work announces itself. A mis-threaded
mask does not. If the mask is warped differently from the image, or smoothed off
binary, or normalised with the ImageNet statistics, training still runs, the loss
still falls, and the result is a model taught to find lesions in the wrong place
-- which is indistinguishable from the problem the lesion head exists to fix.

Four specific silent failures are pinned here, all of them one keyword argument
away from happening:

* ``mode="bilinear"`` applied to a binary mask (measured: ~41 grey levels)
* ``RepeatChanneld`` making the mask ``(3, H, W)``, which a Dice loss happily
  broadcasts against ``(1, H, W)`` without complaint
* ``NormalizeIntensityd`` mapping {0, 1} to about {-2.1, 2.2}
* a geometric transform applied to the image but not the mask

Synthetic throughout: no BTXRD on disk is required, so this runs in CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from onnm.config import Config, ConfigError
from onnm.dataset import build_transforms


def _square_record(tmp_path, *, size=192, box=(60, 40, 120, 100), rotate=False):
    """An image whose bright square coincides exactly with its annotated polygon.

    Because the two start aligned, any geometric transform that moves one and not
    the other shows up as the mask drifting off the bright region -- which is
    what ``_alignment`` measures.
    """
    image = np.zeros((size, size), dtype=np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = 255
    path = tmp_path / "IMG.png"
    Image.fromarray(image).save(path)

    annotation = {
        "imageHeight": size,
        "imageWidth": size,
        "shapes": [
            {
                "label": "lesion",
                "shape_type": "polygon",
                "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            }
        ],
    }
    ann_path = tmp_path / "IMG.json"
    ann_path.write_text(json.dumps(annotation), encoding="utf-8")

    return {"image": str(path), "label": 1, "image_id": "IMG", "annotation": str(ann_path)}


def _cfg(base_cfg, **overrides):
    data = base_cfg.to_dict()
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config(data)


def _alignment(sample) -> float:
    """Share of mask area that lands on the image's bright square.

    1.0 means image and mask were transformed identically. A geometric transform
    applied to only one of them drives this toward 0.
    """
    image = sample["image"]
    mask = sample["mask"][0].numpy() if torch.is_tensor(sample["mask"]) else sample["mask"][0]
    plane = image[0].numpy() if torch.is_tensor(image) else image[0]

    lesion = mask > 0.5
    if not lesion.any():
        return float("nan")
    # The image has been percentile-scaled and normalised, so compare against its
    # own midpoint rather than an absolute intensity.
    bright = plane > (plane.min() + plane.max()) / 2.0
    return float((lesion & bright).sum() / lesion.sum())


# ---------------------------------------------------------------------------
# Value integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["test", "train"])
def test_mask_stays_binary(cfg, tmp_path, mode):
    """No interpolation, normalisation or noise may leave the mask non-binary."""
    transform = build_transforms(cfg, mode, with_mask=True)
    sample = transform(_square_record(tmp_path))
    values = torch.unique(sample["mask"])
    assert set(round(float(v), 6) for v in values) <= {0.0, 1.0}, (
        f"mask has {len(values)} distinct values: {values[:8]}. A geometric transform "
        "is interpolating it -- pass mode=[...,'nearest'], not a scalar mode."
    )


def test_mask_keeps_one_channel(cfg, tmp_path):
    """RepeatChanneld must not reach the mask.

    A (3, H, W) mask broadcasts silently against a (1, H, W) prediction.
    """
    sample = build_transforms(cfg, "train", with_mask=True)(_square_record(tmp_path))
    assert sample["mask"].shape[0] == 1, (
        f"mask has {sample['mask'].shape[0]} channels; RepeatChanneld is being applied "
        "to it. Dice would broadcast against this without raising."
    )
    assert sample["image"].shape[0] == int(cfg.data.in_channels)


def test_mask_escapes_intensity_normalisation(cfg, tmp_path):
    """NormalizeIntensityd would map {0, 1} to roughly {-2.1, 2.2}."""
    sample = build_transforms(cfg, "test", with_mask=True)(_square_record(tmp_path))
    assert float(sample["mask"].min()) >= 0.0
    assert float(sample["mask"].max()) <= 1.0


def test_normal_film_yields_an_empty_mask(cfg, tmp_path):
    """A record with no annotation file is supervision, not missing data.

    "No lesion anywhere on this healthy joint" is the lesson the false positives
    on complex anatomy need, and it is free for all 1879 normal images.
    """
    record = _square_record(tmp_path)
    record["annotation"] = str(tmp_path / "does-not-exist.json")
    record["label"] = 0
    sample = build_transforms(cfg, "test", with_mask=True)(record)
    assert float(sample["mask"].sum()) == 0.0


# ---------------------------------------------------------------------------
# Geometric co-registration -- the silent one
# ---------------------------------------------------------------------------
def test_mask_and_image_are_aligned_without_augmentation(cfg, tmp_path):
    sample = build_transforms(cfg, "test", with_mask=True)(_square_record(tmp_path))
    assert _alignment(sample) > 0.95


@pytest.mark.parametrize(
    "augment",
    [
        pytest.param({"hflip_prob": 1.0, "rotate_prob": 0.0, "zoom_prob": 0.0}, id="flip"),
        pytest.param({"hflip_prob": 0.0, "rotate_prob": 1.0, "zoom_prob": 0.0}, id="rotate"),
        pytest.param({"hflip_prob": 0.0, "rotate_prob": 0.0, "zoom_prob": 1.0}, id="zoom"),
        pytest.param(
            {"hflip_prob": 0.0, "rotate_prob": 0.0, "zoom_prob": 0.0, "affine_prob": 1.0},
            id="affine",
        ),
    ],
)
def test_every_geometric_transform_moves_both(cfg, tmp_path, augment):
    """Each geometry stage, forced on individually, must carry the mask with it.

    Parametrised one transform at a time on purpose: with all of them on, a
    single mis-threaded stage is masked by the others still being correct.
    """
    base = {"hflip_prob": 0.0, "rotate_prob": 0.0, "zoom_prob": 0.0, "affine_prob": 0.0,
            "contrast_prob": 0.0, "noise_prob": 0.0, "dropout_prob": 0.0,
            "histogram_prob": 0.0}
    base.update(augment)
    transform = build_transforms(_cfg(cfg, augment=base), "train", with_mask=True)
    transform.set_random_state(seed=1337)

    aligned = []
    for _ in range(6):
        sample = transform(_square_record(tmp_path))
        score = _alignment(sample)
        if not np.isnan(score):
            aligned.append(score)

    assert aligned, "every draw produced an empty mask"
    assert min(aligned) > 0.80, (
        f"mask drifted off the lesion (worst overlap {min(aligned):.2f}). The image and "
        "the mask are not being given to the same geometric transform."
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_crop_foreground_with_mask_is_refused(cfg, tmp_path):
    """CropForegroundd picks its box from the image alone; the mask would drift.

    evaluate_localisation already refuses to SCORE through this geometry. Nothing
    stopped a run from TRAINING through it, which is the more expensive mistake.
    """
    with pytest.raises(ConfigError, match="crop_foreground"):
        build_transforms(_cfg(cfg, data={"crop_foreground": True}), "train", with_mask=True)


def test_mask_is_absent_unless_requested(cfg, tmp_path):
    """with_mask=False must leave the chain byte-identical to what it was."""
    sample = build_transforms(cfg, "train", with_mask=False)(_square_record(tmp_path))
    assert "mask" not in sample
