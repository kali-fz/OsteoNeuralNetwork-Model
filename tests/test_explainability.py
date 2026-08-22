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

from onnm.dataset import build_transforms
from onnm.explainability import (
    boxes_to_mask,
    cam_iou,
    coverage,
    load_annotation,
    map_box_to_model_space,
    overlay_cam,
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
