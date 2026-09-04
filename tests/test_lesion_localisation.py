"""Scoring the lesion head's map, and proving it is the map the website serves.

WHY THIS FILE EXISTS
--------------------
``scripts/gradcam_report.py`` scored Grad-CAM on every checkpoint, lesion head or
not, because nothing outside training and serving ever read ``seg_head``. A sweep
run carrying the new head was therefore measured by the instrument it was built
to replace, and its pointing-game column would have read as "the head does not
work" while in fact measuring nothing about the head at all. That is a worse
failure than a wrong number, because a wrong number invites a second look and a
number about the wrong thing does not.

Three properties are pinned here, and each one is a specific way the scorer could
be wrong while still producing plausible output:

* **The map is not rescaled.** ``compute_cam`` min-max stretches a CAM, which is
  the only way to read an unbounded attribution. Doing the same to a sigmoid
  would turn "0.002 everywhere, nothing here" into a full-range heatmap claiming
  a lesion on a clean film -- scoring well against any box that happens to sit
  near the argmax, and painting a normal radiograph blue edge to edge in the UI.
* **The chance level is reported.** A pointing game is not a percentage out of
  100. These lesion boxes cover about a tenth of the frame, so a peak dropped at
  random scores about 0.10, and the project has already once read a number near
  chance as a result.
* **``return_mask`` is left as it was found.** MONAI's Grad-CAM indexes
  ``logits[:, class_idx]`` and fails on a tuple. A scorer that leaves the flag
  set breaks Grad-CAM for everything downstream of it -- including, if the same
  mistake reached ``onnm.inference``, ``/api/scan`` for every visitor.

Synthetic throughout: no BTXRD on disk is required, so this runs in CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from onnm.config import Config
from onnm.explainability import (
    compute_lesion_map,
    evaluate_lesion_localisation,
    evaluate_localisation,
    has_lesion_head,
    upsample_map,
)

#: The synthetic film is square and 192px, and the model input is 64px, so the
#: geometry is exactly a divide-by-three with no padding. That keeps every
#: expected value below hand-checkable, which is the point of choosing it.
ORIGINAL_SIZE = 192
MODEL_SIZE = 64
BOX = (60, 40, 120, 100)  # x_min, y_min, x_max, y_max in original coordinates

#: The lesion's share of the model-input frame, and therefore the pointing
#: game's chance level. Derived by hand rather than from the code under test:
#: scale = 64/192 = 1/3, so the box maps to x 20.0-40.0 and y 13.33-33.33;
#: `boxes_to_mask` floors the low edge and ceils the high one, giving columns
#: 20..39 (20 wide) and rows 13..33 (21 tall). 20 * 21 / 64**2 = 0.10254.
EXPECTED_CHANCE = 420 / 4096


def _annotated_film(tmp_path):
    """One record with a lesion annotation where the scorer expects to find it.

    ``evaluate_localisation`` resolves annotations through
    ``annotation_path_for``, which reads ``paths.data_root`` and
    ``paths.annotations_dirname`` -- NOT the record's own ``annotation`` key. A
    fixture that only sets the record key would score zero images and every
    assertion below would pass vacuously.
    """
    image = np.zeros((ORIGINAL_SIZE, ORIGINAL_SIZE), dtype=np.uint8)
    x0, y0, x1, y1 = BOX
    image[y0:y1, x0:x1] = 255
    image_path = tmp_path / "IMG.png"
    Image.fromarray(image).save(image_path)

    annotations = tmp_path / "annotations"
    annotations.mkdir(exist_ok=True)
    (annotations / "IMG.json").write_text(
        json.dumps(
            {
                "imageHeight": ORIGINAL_SIZE,
                "imageWidth": ORIGINAL_SIZE,
                "shapes": [
                    {
                        "label": "lesion",
                        "shape_type": "rectangle",
                        "points": [[x0, y0], [x1, y1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return {
        "image": str(image_path),
        "label": 2,
        "image_id": "IMG",
        "annotation": str(annotations / "IMG.json"),
    }


def _scoring_cfg(cfg, tmp_path, **extra):
    data = cfg.to_dict()
    data.setdefault("paths", {}).update(
        {"data_root": str(tmp_path), "annotations_dirname": "annotations"}
    )
    data.setdefault("data", {}).update({"image_size": MODEL_SIZE, "crop_foreground": False})
    data.setdefault("explain", {}).update({"cam_threshold": 0.5})
    for section, values in extra.items():
        data.setdefault(section, {}).update(values)
    return Config(data)


class _FixedLesionNet(torch.nn.Module):
    """A model whose lesion map is chosen by the test rather than learned.

    Only the *presence* of ``seg_head`` is what marks a checkpoint as carrying a
    head, so an ``Identity`` standing in for the decoder is enough here and keeps
    the expected map exact. The real decoder is exercised separately, at the
    bottom of this file, because a fake cannot catch a plumbing error in it.
    """

    def __init__(self, mask_logits: torch.Tensor) -> None:
        super().__init__()
        self.seg_head = torch.nn.Identity()
        self.return_mask = False
        self.register_buffer("mask_logits", mask_logits[None, None])

    def forward(self, x: torch.Tensor):
        logits = torch.zeros((x.shape[0], 3))
        if not self.return_mask:
            return logits
        return logits, self.mask_logits.expand(x.shape[0], -1, -1, -1)


def _mask_with_hot_cell(row: int, col: int, grid: int = 16) -> torch.Tensor:
    """A decoder-shaped logit map: one confident cell, confidently empty elsewhere.

    +/-6 rather than +/-inf so the sigmoid lands at 0.9975 and 0.0025 -- real
    probabilities a trained head could emit, which is what makes the
    "not rescaled" assertion below mean something.
    """
    logits = torch.full((grid, grid), -6.0)
    logits[row, col] = 6.0
    return logits


# ---------------------------------------------------------------------------
# What the score means
# ---------------------------------------------------------------------------
def test_the_chance_level_is_reported_and_is_the_lesion_box_area(cfg, tmp_path) -> None:
    """The baseline every pointing game has to beat, computed on these films.

    Not a remembered figure from another split: it is accumulated per image, so
    a report is always calibrated against the population it scored.
    """
    scoring_cfg = _scoring_cfg(cfg, tmp_path)
    model = _FixedLesionNet(_mask_with_hot_cell(6, 7))

    scores = evaluate_lesion_localisation(
        model, scoring_cfg, [_annotated_film(tmp_path)], torch.device("cpu")
    )

    assert scores["n_scored"] == 1
    assert scores["chance_pointing_game"] == pytest.approx(EXPECTED_CHANCE)
    assert scores["mean_lesion_fraction"] == pytest.approx(EXPECTED_CHANCE)


def test_a_peak_inside_the_lesion_is_a_hit_and_one_outside_is_not(cfg, tmp_path) -> None:
    """Pointing game, on the map the head produced, through the real geometry.

    Cell (6, 7) of a 16x16 decoder map upsamples to roughly (26, 30) in the 64px
    model frame, which is inside the lesion at rows 13..33, columns 20..39.
    Cell (0, 0) lands in the top-left corner, which is not.
    """
    scoring_cfg = _scoring_cfg(cfg, tmp_path)
    record = _annotated_film(tmp_path)

    inside = evaluate_lesion_localisation(
        _FixedLesionNet(_mask_with_hot_cell(6, 7)), scoring_cfg, [record], torch.device("cpu")
    )
    outside = evaluate_lesion_localisation(
        _FixedLesionNet(_mask_with_hot_cell(0, 0)), scoring_cfg, [record], torch.device("cpu")
    )

    assert inside["pointing_game_accuracy"] == 1.0
    assert outside["pointing_game_accuracy"] == 0.0
    assert inside["map_source"] == "lesion_head"


def test_a_confident_empty_map_stays_empty(cfg, tmp_path) -> None:
    """The single most important assertion in this file.

    A head that says "nothing here" emits about 0.0025 everywhere. Min-max
    rescaling it -- which is exactly what ``compute_cam`` does, correctly, to a
    Grad-CAM -- would stretch that to a full 0..1 heatmap: a lesion claimed on a
    clean film, an IoU computed against noise, and an overlay painted edge to
    edge in the browser. So the map must arrive as the model stated it.

    ``mean_max_value`` is the witness: near 0 here, and pinned at exactly 1.0 by
    any rescaling, on every film, forever.

    THE MAP MUST VARY, and the first version of this test got that wrong. A
    perfectly flat map is the one input min-max rescaling leaves alone -- every
    implementation guards ``hi > lo`` -- so a constant -6.0 passed this test with
    the rescale deliberately reintroduced. A gentle ramp between two confidently
    negative logits is both more realistic (no real decoder emits a constant) and
    the only version that actually fails when the bug comes back. Verified by
    reintroducing it: with the ramp, ``mean_max_value`` becomes exactly 1.0.
    """
    scoring_cfg = _scoring_cfg(cfg, tmp_path)
    ramp = torch.linspace(-6.0, -4.0, 16 * 16).reshape(16, 16)
    empty = _FixedLesionNet(ramp)

    scores = evaluate_lesion_localisation(
        empty, scoring_cfg, [_annotated_film(tmp_path)], torch.device("cpu")
    )

    # sigmoid(-4) = 0.018, and a rescale would put this at exactly 1.0.
    assert scores["mean_max_value"] < 0.05
    assert scores["n_below_threshold"] == 1
    assert scores["mean_iou"] == 0.0
    assert scores["mean_positive_fraction"] == 0.0


def test_the_threshold_can_be_overridden_without_touching_the_config(cfg, tmp_path) -> None:
    """``--lesion-threshold`` must not mutate the config it was handed.

    The config a report records is the config the run used; a scorer that edits
    it in place makes the report describe a threshold nobody chose, and the same
    object is what ``build_transforms`` reads for the next call.
    """
    scoring_cfg = _scoring_cfg(cfg, tmp_path)
    model = _FixedLesionNet(_mask_with_hot_cell(6, 7))

    scores = evaluate_lesion_localisation(
        model, scoring_cfg, [_annotated_film(tmp_path)], torch.device("cpu"), threshold=0.001
    )

    assert scores["cam_threshold"] == pytest.approx(0.001)
    assert float(scoring_cfg.explain.cam_threshold) == 0.5


# ---------------------------------------------------------------------------
# The map that is scored is the map that is served
# ---------------------------------------------------------------------------
def test_upsampling_matches_the_resize_the_website_uses() -> None:
    """``upsample_map`` and ``inference._resize_map`` must agree.

    The scorer claims to measure the map a visitor sees. That claim rests
    entirely on the two upsamplers agreeing -- one uses ``F.interpolate``, the
    other ``cv2.INTER_LINEAR`` -- so it is checked rather than asserted in a
    docstring. Both use half-pixel centres; if either ever stops, the numbers in
    a report quietly stop describing the overlay in the browser.
    """
    from onnm.inference import _resize_map

    rng = np.random.default_rng(0)
    small = rng.random((16, 16)).astype(np.float32)

    ours = upsample_map(small, MODEL_SIZE)
    theirs = _resize_map(small, (MODEL_SIZE, MODEL_SIZE))

    assert ours.shape == (MODEL_SIZE, MODEL_SIZE)
    assert np.allclose(ours, theirs, atol=1e-5)


def test_return_mask_is_restored_and_the_model_still_returns_a_tensor() -> None:
    """Scoring must not leave the model in two-output mode.

    ``build_cam`` hands the model to MONAI, which indexes ``logits[:, class_idx]``
    and raises on a tuple. ``gradcam_report.py`` scores the lesion map and then
    scores Grad-CAM on the same object, so a flag left set would break the second
    pass -- and the same mistake in ``onnm.inference`` would break every scan.
    """
    from onnm.lesion_head import build_densenet_with_lesion_head

    model = build_densenet_with_lesion_head("densenet121", None, 3, 0.0, decoder_width=8)
    model.eval()
    image = torch.zeros((1, 3, MODEL_SIZE, MODEL_SIZE))

    assert model.return_mask is False
    compute_lesion_map(model, image)
    assert model.return_mask is False
    assert torch.is_tensor(model(image))

    # And an outer caller that had it set keeps it set, rather than having it
    # forced back to the default underneath them.
    model.return_mask = True
    compute_lesion_map(model, image)
    assert model.return_mask is True


def test_a_real_lesion_head_scores_end_to_end(cfg, tmp_path) -> None:
    """The genuine decoder, not a stand-in, through the whole scoring path.

    Random weights, so the scores themselves mean nothing -- what is under test
    is that the real class produces a map of the right shape and range and that
    every metric comes back a number rather than a NaN or an exception.
    """
    from onnm.lesion_head import build_densenet_with_lesion_head

    model = build_densenet_with_lesion_head("densenet121", None, 3, 0.0, decoder_width=8)
    scoring_cfg = _scoring_cfg(cfg, tmp_path)

    assert has_lesion_head(model)
    scores = evaluate_lesion_localisation(
        model, scoring_cfg, [_annotated_film(tmp_path)], torch.device("cpu")
    )

    assert scores["n_scored"] == 1
    assert scores["map_source"] == "lesion_head"
    assert 0.0 <= scores["mean_max_value"] <= 1.0
    assert not np.isnan(scores["mean_iou"])
    assert not np.isnan(scores["mean_coverage"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_checkpoint_without_a_head_is_refused_not_guessed_at(cfg, tmp_path) -> None:
    """Better a clear error than a plausible number from the wrong output."""
    headless = torch.nn.Linear(4, 3)
    scoring_cfg = _scoring_cfg(cfg, tmp_path)

    assert not has_lesion_head(headless)
    with pytest.raises(ValueError, match="no lesion head"):
        evaluate_lesion_localisation(headless, scoring_cfg, [], torch.device("cpu"))
    with pytest.raises(ValueError, match="no lesion head"):
        compute_lesion_map(headless, torch.zeros((1, 3, MODEL_SIZE, MODEL_SIZE)))


def test_foreground_cropping_is_refused_for_the_lesion_map_too(cfg, tmp_path) -> None:
    """The geometry guard covers the new path as well as the old one.

    ``map_box_to_model_space`` models a resize and a symmetric pad and nothing
    else. Foreground cropping adds a per-image offset it does not know about, so
    every score would still be a number and all of them would be wrong. The
    lesion map is scored against the same boxes through the same mapping, so it
    inherits the same refusal -- and that has to be checked, not assumed, because
    the guard now lives in a helper either path could have skipped.
    """
    cropped = _scoring_cfg(cfg, tmp_path, data={"crop_foreground": True})
    model = _FixedLesionNet(_mask_with_hot_cell(6, 7))

    with pytest.raises(ValueError, match="crop_foreground"):
        evaluate_lesion_localisation(model, cropped, [], torch.device("cpu"))
    with pytest.raises(ValueError, match="crop_foreground"):
        evaluate_localisation(model, cropped, [], torch.device("cpu"))


# ---------------------------------------------------------------------------
# The Grad-CAM path, unchanged
# ---------------------------------------------------------------------------
class _TinyConvNet(torch.nn.Module):
    """conv -> ReLU -> global average pool -> linear, the shape Grad-CAM expects.

    Deliberately tiny, and deliberately a real module rather than a stub: the
    inverted-heatmap bug this project shipped in 2026-08 lived inside MONAI's
    default postprocessing, and no stubbed cam object would ever have reached it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(4, 3)
        with torch.no_grad():
            self.features[0].weight.fill_(0.05)
            self.classifier.weight.fill_(0.5)
            self.classifier.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.classifier(pooled)


def test_gradcam_scoring_still_works_and_now_says_which_map_it_used(cfg, tmp_path) -> None:
    """The refactor must not have changed what a Grad-CAM report contains.

    Both scorers share one engine so that the two explanations are measured
    identically. The risk in sharing it is that the Grad-CAM report quietly
    gains or loses a key that ``overnight_sweep.py`` and the version ledger read.
    So the old keys are pinned, the new ones are checked for, and the
    sigmoid-only diagnostics are checked to be ABSENT -- ``compute_cam`` rescales
    every map, so ``mean_max_value`` would be 1.0 by construction and reporting
    it would be inviting a comparison that means nothing.
    """
    scoring_cfg = _scoring_cfg(cfg, tmp_path, explain={"target_layer": "features"})
    scores = evaluate_localisation(
        _TinyConvNet(), scoring_cfg, [_annotated_film(tmp_path)], torch.device("cpu"),
        class_index=2,
    )

    assert scores["map_source"] == "gradcam"
    assert scores["n_scored"] == 1
    for key in ("pointing_game_accuracy", "mean_iou", "mean_coverage",
                "mean_peak_fraction", "cam_degenerate", "cam_threshold"):
        assert key in scores, f"{key} disappeared from the Grad-CAM report"
    assert scores["chance_pointing_game"] == pytest.approx(EXPECTED_CHANCE)
    for key in ("mean_max_value", "mean_positive_fraction", "n_below_threshold"):
        assert key not in scores
