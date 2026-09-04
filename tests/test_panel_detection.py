"""A gutter between two pasted views is not the same thing as a dark background.

Telling those apart is the whole job. The naive test -- "is there a dark column
in the middle" -- flagged 17.3% of BTXRD, because a single bone on a black field
produces exactly that. The shipped rule additionally requires the dark run to be
*bounded by content on both sides* and to split the frame into two panels of
comparable width, and requires a gutter column to be 97% dark rather than 90%.
Together those brought it to 1.3% while still catching both real composite
uploads.

These are synthetic so they run without BTXRD, and each one pins a specific way
the naive version was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from onnm.io_radiograph import detect_panels


def _panel(height: int, width: int, value: float = 0.8) -> np.ndarray:
    """A bright blob on black, standing in for a limb on a collimated field."""
    panel = np.zeros((height, width), dtype=np.float32)
    top, bottom = int(0.08 * height), int(0.92 * height)
    left, right = int(0.25 * width), int(0.75 * width)
    panel[top:bottom, left:right] = value
    return panel


def _composite(height=400, width=400, gutter=24) -> np.ndarray:
    """Two panels side by side with a black gutter, as an AP+lateral upload."""
    left, right = _panel(height, width), _panel(height, width)
    return np.concatenate(
        [left, np.zeros((height, gutter), dtype=np.float32), right], axis=1
    )


def test_a_two_panel_composite_is_detected():
    report = detect_panels(_composite())
    assert report["is_composite"] is True
    assert report["n_panels"] == 2
    assert len(report["split_at"]) == 1
    # The split should land in the gutter, near the middle.
    centre = report["split_at"][0] / (2 * 400 + 24)
    assert 0.45 < centre < 0.55


def test_a_single_film_is_not_flagged():
    assert detect_panels(_panel(400, 300))["is_composite"] is False


def test_a_dark_background_around_one_bone_is_not_a_gutter():
    """The exact false positive that made a column-mean test unusable.

    A single bone on a black field has fully dark columns either side of it,
    reaching the frame edge. Those are background. A gutter never touches the
    edge, because there is a panel on both sides of it.
    """
    image = np.zeros((400, 400), dtype=np.float32)
    image[40:360, 170:230] = 0.9  # one narrow bone, wide black margins
    assert detect_panels(image)["is_composite"] is False


def test_a_dark_anatomical_gap_is_not_a_gutter():
    """Two bones with a gap between them -- tibia and fibula, say.

    The gap is dark but it is interrupted: bone crosses it somewhere down its
    length, so no column in it is dark top-to-bottom. With dark_rows at 0.97 a
    column has to be at least 97% dark to count, so a bridge across even a tenth
    of the height disqualifies it -- which is the point.
    """
    image = np.zeros((400, 400), dtype=np.float32)
    image[40:360, 80:180] = 0.9
    image[40:360, 230:330] = 0.9
    image[180:220, 180:230] = 0.9  # the two bones meet partway down
    assert detect_panels(image)["is_composite"] is False


def test_a_lopsided_split_is_rejected():
    """An AP and a lateral of one limb are roughly equal in width.

    A dark band very close to one edge is far more likely to be collimation than
    a panel boundary, so the balance rule refuses it.
    """
    wide, narrow = _panel(400, 500), _panel(400, 80)
    image = np.concatenate(
        [wide, np.zeros((400, 24), dtype=np.float32), narrow], axis=1
    )
    assert detect_panels(image)["is_composite"] is False


def test_a_hairline_dark_column_is_not_a_gutter():
    """A one-pixel dark line is an artefact, not a panel boundary."""
    image = _panel(400, 400)
    image[:, 200] = 0.0
    assert detect_panels(image)["is_composite"] is False


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((0, 0), dtype=np.float32),
        np.zeros((10,), dtype=np.float32),
        np.zeros((50, 50), dtype=np.float32),
        np.full((50, 50), np.nan, dtype=np.float32),
    ],
    ids=["empty", "one-dimensional", "uniformly-black", "all-nan"],
)
def test_degenerate_input_is_answered_not_raised(array):
    """This runs on every upload, so it must never be the thing that throws."""
    report = detect_panels(array)
    assert report["is_composite"] is False
    assert report["n_panels"] == 1


def test_scaling_is_relative_so_bit_depth_does_not_matter():
    """A 12-bit DICOM and an 8-bit JPEG of the same film must agree.

    The thresholds are applied after percentile scaling for exactly this reason.
    """
    composite = _composite()
    assert detect_panels(composite)["is_composite"] is True
    assert detect_panels(composite * 4095.0)["is_composite"] is True
