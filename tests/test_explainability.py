"""Tests for annotation parsing, CAM scoring, and — most importantly — geometry.

``map_box_to_model_space`` re-implements the resize-and-pad arithmetic that
``build_transforms`` performs. Two independent implementations of the same
geometry will drift apart eventually, and when they do, every pointing-game and
IoU number silently becomes fiction while still looking entirely plausible.
``test_box_mapping_matches_actual_transform`` is what stops that: it puts a
bright square at known coordinates, pushes it through the real transform chain,
and checks the predicted box actually lands on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from onnm.dataset import build_transforms
from onnm.explainability import (
    DEGENERATE_PEAK_FRACTION,
    boxes_to_mask,
    cam_iou,
    coverage,
    load_annotation,
    map_box_to_model_space,
    overlay_cam,
    peak_fraction,
    pointing_game,
)


# ---------------------------------------------------------------------------
# Geometry: the one that matters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("orig_h", "orig_w"),
    [(3032, 1640), (1640, 3032), (512, 512), (2000, 900), (700, 2100)],
)
def test_box_mapping_matches_actual_transform(
    cfg, tmp_path: Path, orig_h: int, orig_w: int
) -> None:
    """The predicted box must contain the square the transform actually produced."""
    from PIL import Image

    # A bright square occupying a known region of the original image.
    x0, y0 = int(orig_w * 0.30), int(orig_h * 0.40)
    x1, y1 = int(orig_w * 0.55), int(orig_h * 0.60)

    arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
    arr[y0:y1, x0:x1] = 255
    path = tmp_path / "phantom.png"
    Image.fromarray(arr, mode="L").save(path)

    size = int(cfg.data.image_size)
    transformed = build_transforms(cfg, "val")(
        {"image": str(path), "label": 0, "image_id": "p", "patient_id": "p"}
    )["image"][0].numpy()

    # Where did the square actually end up?
    bright = transformed > (transformed.min() + 0.5 * (transformed.max() - transformed.min()))
    rows, cols = np.where(bright)
    assert rows.size > 0, "the square disappeared during transformation"

    predicted = map_box_to_model_space((x0, y0, x1, y1), orig_h, orig_w, size)
    px0, py0, px1, py1 = predicted

    # Allow two pixels of slack for interpolation and rounding at the edges.
    tol = 2.0
    assert px0 - tol <= cols.min() and cols.max() <= px1 + tol, (
        f"columns {cols.min()}..{cols.max()} outside predicted x {px0:.1f}..{px1:.1f}"
    )
    assert py0 - tol <= rows.min() and rows.max() <= py1 + tol, (
        f"rows {rows.min()}..{rows.max()} outside predicted y {py0:.1f}..{py1:.1f}"
    )


def test_box_mapping_preserves_aspect_ratio(cfg) -> None:
    """A square lesion on a 4:1 film must stay square in model space."""
    size = int(cfg.data.image_size)
    x0, y0, x1, y1 = 800.0, 200.0, 1000.0, 400.0  # 200x200 square
    mx0, my0, mx1, my1 = map_box_to_model_space((x0, y0, x1, y1), 800, 3200, size)

    width, height = mx1 - mx0, my1 - my0
    assert width == pytest.approx(height, rel=0.02)


def test_box_mapping_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="invalid original dimensions"):
        map_box_to_model_space((0, 0, 1, 1), 0, 100, 256)


def test_boxes_to_mask_covers_the_region() -> None:
    mask = boxes_to_mask([(10.0, 20.0, 30.0, 50.0)], 256)
    assert mask.shape == (256, 256)
    assert mask[25, 15] and mask[49, 29]
    assert not mask[19, 15] and not mask[25, 5]


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------
def _write_annotation(path: Path, shapes: list[dict], h: int = 3032, w: int = 1640) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "5.4.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": "IMG000002.jpeg",
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parses_rectangle_and_polygon(tmp_path: Path) -> None:
    """BTXRD stores a rectangle and a polygon per lesion; both must yield a box."""
    path = _write_annotation(
        tmp_path / "a.json",
        [
            {
                "label": "osteosarcoma",
                "points": [[885.9, 343.2], [1115.4, 1188.7]],
                "shape_type": "rectangle",
            },
            {
                "label": "osteosarcoma",
                "points": [[900.0, 400.0], [1100.0, 400.0], [1000.0, 1100.0]],
                "shape_type": "polygon",
            },
        ],
    )
    annotation = load_annotation(path)

    assert len(annotation["boxes"]) == 2
    assert annotation["height"] == 3032
    assert annotation["labels"] == ["osteosarcoma", "osteosarcoma"]
    # The polygon must collapse to its enclosing box.
    assert annotation["boxes"][1] == pytest.approx((900.0, 400.0, 1100.0, 1100.0))


def test_rectangle_corner_order_is_normalised(tmp_path: Path) -> None:
    """LabelMe does not guarantee top-left then bottom-right."""
    path = _write_annotation(
        tmp_path / "b.json",
        [{"label": "x", "points": [[500.0, 900.0], [100.0, 200.0]], "shape_type": "rectangle"}],
    )
    assert load_annotation(path)["boxes"][0] == pytest.approx((100.0, 200.0, 500.0, 900.0))


def test_unreadable_annotation_returns_empty(tmp_path: Path) -> None:
    """A corrupt annotation must degrade to 'no boxes', not kill the report."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    annotation = load_annotation(path)
    assert annotation["boxes"] == []


def test_missing_annotation_returns_empty(tmp_path: Path) -> None:
    assert load_annotation(tmp_path / "absent.json")["boxes"] == []


# ---------------------------------------------------------------------------
# CAM scoring
# ---------------------------------------------------------------------------
def test_pointing_game_detects_hit_and_miss() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True

    hit = np.zeros((64, 64), dtype=np.float32)
    hit[30, 30] = 1.0
    assert pointing_game(hit, mask) is True

    miss = np.zeros((64, 64), dtype=np.float32)
    miss[5, 5] = 1.0
    assert pointing_game(miss, mask) is False


def test_pointing_game_uses_the_plateau_centroid_not_raster_order() -> None:
    """The bug that made a broken measurement look like a broken model.

    ``np.argmax`` returns the FIRST maximal element, so a CAM with a large tie
    reports the top-left corner of the plateau every time -- background on
    essentially every radiograph. Scoring the pinned checkpoint that way gave a
    pointing-game accuracy of exactly 0.0000 across 267 images, which read as a
    devastating result about the model and was a statement about tie-breaking.

    Here the plateau is centred on the lesion, so the honest answer is a hit.
    """
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True

    cam = np.zeros((64, 64), dtype=np.float32)
    cam[20:44, 20:44] = 1.0          # a plateau centred on the lesion
    assert np.unravel_index(int(np.argmax(cam)), cam.shape) == (20, 20)
    assert not mask[20, 20], "argmax lands outside the lesion, which is the trap"
    assert pointing_game(cam, mask) is True


def test_pointing_game_still_misses_when_the_plateau_is_elsewhere() -> None:
    """The centroid must not turn every degenerate CAM into a hit."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[40:60, 40:60] = True

    cam = np.zeros((64, 64), dtype=np.float32)
    cam[0:20, 0:20] = 1.0
    assert pointing_game(cam, mask) is False


def test_peak_fraction_separates_a_peak_from_a_plateau() -> None:
    """The number that tells a bad model apart from an unusable measurement."""
    sharp = np.zeros((64, 64), dtype=np.float32)
    sharp[30, 30] = 1.0
    assert peak_fraction(sharp) == pytest.approx(1 / 4096)

    flat = np.ones((64, 64), dtype=np.float32)
    assert peak_fraction(flat) == pytest.approx(1.0)

    # What the pinned checkpoint actually produces: most of the frame tied.
    saturated = np.full((64, 64), 1.0, dtype=np.float32)
    saturated[0:16, 0:16] = 0.2
    assert peak_fraction(saturated) > DEGENERATE_PEAK_FRACTION


def test_pointing_game_without_ground_truth_is_false() -> None:
    assert pointing_game(np.ones((8, 8)), np.zeros((8, 8), dtype=bool)) is False


def test_cam_iou_perfect_and_disjoint() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True

    perfect = np.zeros((64, 64), dtype=np.float32)
    perfect[20:40, 20:40] = 1.0
    assert cam_iou(perfect, mask, 0.5) == pytest.approx(1.0)

    disjoint = np.zeros((64, 64), dtype=np.float32)
    disjoint[0:10, 0:10] = 1.0
    assert cam_iou(disjoint, mask, 0.5) == pytest.approx(0.0)


def test_coverage_measures_mass_inside_lesion() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, :] = True

    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0:5, :] = 1.0   # all mass inside
    assert coverage(cam, mask) == pytest.approx(1.0)

    cam[5:, :] = 1.0    # half the mass now outside
    assert coverage(cam, mask) == pytest.approx(0.5)


def test_overlay_produces_rgb_with_box() -> None:
    image = np.random.default_rng(0).random((64, 64))
    cam = np.zeros((64, 64))
    cam[20:40, 20:40] = 1.0

    out = overlay_cam(image, cam, boxes=[(10.0, 10.0, 50.0, 50.0)])
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    # The box is drawn pure green.
    assert (out[10, 10:50] == [0, 255, 0]).all()


# ---------------------------------------------------------------------------
# CAM polarity: the tests that would have caught the inverted heatmap
# ---------------------------------------------------------------------------
# These MUST construct a real model and go through `build_cam`, and that is the
# whole point of them.
#
# The bug they guard against lived in MONAI's default `postprocessing`, which
# maps (min, max) -> (1, 0) and therefore inverts the map. `compute_cam` was
# never at fault. So a test that feeds a synthetic array through a stubbed cam
# object -- `lambda x, class_idx: some_array` -- never invokes MONAI at all,
# never invokes the default normalizer, and passes identically whether the bug
# is present or fixed. Every pre-existing test in this file is synthetic, which
# is exactly why an exactly-inverted heatmap shipped unnoticed.
#
# A stub is only acceptable where the property under test genuinely belongs to
# `compute_cam` itself, as in the empty-evidence test at the end.


class _GapNet(torch.nn.Module):
    """Tiny conv -> ReLU -> global-average-pool -> linear net.

    Deliberately the same *shape* as DenseNet's head, which is what makes
    Grad-CAM behave the way it does here, and deliberately tiny so the test
    costs milliseconds and downloads nothing.

    The weights are set rather than random so the expected CAM is provable
    rather than hopeful: every conv weight and every classifier weight is
    positive, so the activation at each position is a positive multiple of the
    local input intensity, the gradient of the logit w.r.t. those activations is
    a positive constant, and the CAM is therefore monotonically increasing in
    input brightness. Bright input region => hot CAM. Nothing else is possible.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.features[0].weight.fill_(0.05)
            self.classifier.weight.fill_(0.5)
            self.classifier.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.classifier(pooled)


def _cam_cfg(cfg, method: str):
    """Point the config at _GapNet's only conv stage."""
    cfg._data["explain"]["target_layer"] = "features"
    cfg._data["explain"]["method"] = method
    return cfg


def _bright_corner_input() -> torch.Tensor:
    """A 32x32 input whose top-left quadrant is bright and rest is dark."""
    x = torch.full((1, 3, 32, 32), 0.01)
    x[:, :, :16, :16] = 1.0
    return x


@pytest.mark.parametrize("method", ["gradcam", "gradcampp"])
def test_the_cam_is_hot_where_the_evidence_is(cfg, method: str) -> None:
    """The regression test for the inverted heatmap.

    With this network the bright quadrant is provably the evidence, so a
    correctly-oriented CAM peaks inside it. Under MONAI's default postprocessing
    the peak lands in the dark region instead, and this fails.
    """
    from onnm.explainability import build_cam, compute_cam

    model = _GapNet().eval()
    cam = build_cam(model, _cam_cfg(cfg, method))
    result = compute_cam(cam, _bright_corner_input(), class_index=0)

    assert result.shape == (32, 32)
    assert result.min() >= 0.0 and result.max() <= 1.0

    bright = np.zeros((32, 32), dtype=bool)
    bright[:16, :16] = True
    assert pointing_game(result, bright), (
        "the CAM peak is outside the region that drove the prediction -- "
        "the heatmap is inverted"
    )
    assert result[:16, :16].mean() > result[16:, 16:].mean(), (
        "the evidence region must be hotter than the background"
    )


def test_build_cam_does_not_use_monai_default_postprocessing(cfg) -> None:
    """Pins the fix directly: our CAM must be the opposite of MONAI's default.

    MONAI's `default_normalizer` maps (min, max) -> (1, 0). If someone drops the
    explicit `postprocessing` argument from `build_cam`, this correlation flips
    from -1 to +1 and the test fails loudly rather than the overlay quietly
    going upside down again.
    """
    from monai.visualize import GradCAM
    from monai.visualize.class_activation_maps import default_normalizer

    from onnm.explainability import build_cam, compute_cam

    image = _bright_corner_input()
    ours = compute_cam(build_cam(_GapNet().eval(), _cam_cfg(cfg, "gradcam")), image, 0)
    monai_default = compute_cam(
        GradCAM(
            nn_module=_GapNet().eval(),
            target_layers="features",
            postprocessing=default_normalizer,
        ),
        image,
        0,
    )

    correlation = float(np.corrcoef(ours.ravel(), monai_default.ravel())[0, 1])
    assert correlation < -0.99, (
        f"expected our CAM to be the inverse of MONAI's default (corr ~ -1), got "
        f"{correlation:+.4f} -- if this is ~ +1, build_cam has lost its explicit "
        "postprocessing and the heatmap is inverted again"
    )


def test_a_cam_with_no_positive_evidence_reads_empty_not_full() -> None:
    """Grad-CAM ends in a ReLU, so a map can legitimately be all zeros.

    It must normalise to all zeros -- an honest "no evidence" -- rather than to
    all ones, which is what an inverting normaliser produces and which paints
    the entire film as maximum evidence.

    A stub is appropriate here: the property under test is `compute_cam`'s own
    degenerate-range guard, not the MONAI integration.
    """
    from onnm.explainability import compute_cam

    def all_zero_cam(x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        return torch.zeros(1, 1, 16, 16)

    result = compute_cam(all_zero_cam, torch.zeros(1, 3, 16, 16), class_index=0)
    assert result.shape == (16, 16)
    assert np.all(result == 0.0), "an empty CAM must not render as uniformly hot"
