"""InferenceResult.with_threshold must equal a real re-prediction, exactly.

This exists to justify an optimisation. The app used to re-run the model every
time the threshold slider moved -- a forward pass plus a Grad-CAM backward pass,
about half a second on CPU -- to produce probabilities and a heatmap that were
bit-identical to the ones it already had. Now it re-cuts the stored probability
instead.

That is only safe if the cheap path and the expensive path agree on every field
a reader sees. If they can ever disagree, the app shows one verdict while the
model believes another, and nothing would flag it. So the equivalence is pinned
here rather than argued in a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from onnm.inference import INCONCLUSIVE_LABEL, LESION_LABEL, NORMAL_LABEL, InferenceResult


def _result(normal: float, benign: float, malignant: float, threshold: float = 0.5):
    """Build a result the way predict() would, for a given softmax."""
    probabilities = {"normal": normal, "benign": benign, "malignant": malignant}
    lesion = 1.0 - normal
    is_lesion = lesion >= threshold
    return InferenceResult(
        label=LESION_LABEL if is_lesion else NORMAL_LABEL,
        confidence=100.0 * (lesion if is_lesion else 1.0 - lesion),
        lesion_probability=lesion,
        class_probabilities=probabilities,
        top_class=max(probabilities, key=probabilities.get),
        threshold=threshold,
        preprocessed_image=np.zeros((8, 8), dtype=np.float32),
        original_image=np.zeros((16, 16), dtype=np.float32),
        heatmap=np.full((8, 8), 0.25, dtype=np.float32),
        cam_class="malignant",
    )


# ---------------------------------------------------------------------------
# The decision moves; the evidence does not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("lesion_p", "threshold", "expected"),
    [
        (0.80, 0.50, LESION_LABEL),
        (0.80, 0.90, NORMAL_LABEL),
        (0.30, 0.50, NORMAL_LABEL),
        (0.30, 0.20, LESION_LABEL),
        (0.50, 0.50, LESION_LABEL),   # >= is inclusive, as in predict()
    ],
)
def test_verdict_follows_the_threshold(lesion_p, threshold, expected) -> None:
    base = _result(1.0 - lesion_p, lesion_p * 0.6, lesion_p * 0.4)
    assert base.with_threshold(threshold).label == expected


def test_probabilities_and_heatmap_are_untouched() -> None:
    """The whole justification: moving a cut cannot change what the model saw."""
    base = _result(0.2, 0.5, 0.3)
    moved = base.with_threshold(0.95)

    assert moved.class_probabilities == base.class_probabilities
    assert moved.lesion_probability == base.lesion_probability
    assert moved.top_class == base.top_class
    np.testing.assert_array_equal(moved.heatmap, base.heatmap)
    np.testing.assert_array_equal(moved.preprocessed_image, base.preprocessed_image)
    np.testing.assert_array_equal(moved.original_image, base.original_image)
    assert moved.cam_class == base.cam_class


def test_the_original_is_not_mutated() -> None:
    """One cached model output gets re-cut many times as the slider moves."""
    base = _result(0.2, 0.5, 0.3, threshold=0.5)
    base.with_threshold(0.99)
    assert base.threshold == 0.5
    assert base.label == LESION_LABEL


def test_confidence_backs_whichever_verdict_was_reached() -> None:
    base = _result(0.25, 0.45, 0.30)          # lesion probability 0.75
    assert base.with_threshold(0.5).confidence == pytest.approx(75.0)
    assert base.with_threshold(0.9).confidence == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# The uncertainty gate
# ---------------------------------------------------------------------------
def test_gate_withdraws_a_lesion_call_when_no_class_is_confident() -> None:
    base = _result(0.34, 0.33, 0.33)          # lesion 0.66, max prob 0.34
    gated = base.with_threshold(0.5, uncertainty_floor=0.65)
    assert gated.label == INCONCLUSIVE_LABEL
    assert gated.inconclusive is True


def test_gate_never_manufactures_a_lesion_call() -> None:
    """A Normal verdict stays Normal however uncertain it is -- the gate can
    only withdraw a positive call, never issue one."""
    base = _result(0.34, 0.33, 0.33)
    gated = base.with_threshold(0.9, uncertainty_floor=0.65)   # below threshold
    assert gated.label == NORMAL_LABEL
    assert gated.inconclusive is False


def test_gate_is_recomputed_not_inherited() -> None:
    """`inconclusive` is `is_lesion and defer`, so a stored False cannot tell
    you whether `defer` was False -- it must be re-derived from probabilities.

    Here the first cut is Normal (so inconclusive is False despite the model
    being uncertain); raising the cut into lesion territory must then surface
    the gate rather than trusting the stale False.
    """
    base = _result(0.34, 0.33, 0.33)
    normal_first = base.with_threshold(0.9, uncertainty_floor=0.65)
    assert normal_first.inconclusive is False

    now_lesion = normal_first.with_threshold(0.5, uncertainty_floor=0.65)
    assert now_lesion.inconclusive is True
    assert now_lesion.label == INCONCLUSIVE_LABEL


def test_no_gate_configured_is_a_no_op() -> None:
    base = _result(0.34, 0.33, 0.33)
    assert base.with_threshold(0.5).label == LESION_LABEL


# ---------------------------------------------------------------------------
# Equivalence with the expensive path
# ---------------------------------------------------------------------------
def test_matches_a_real_prediction_at_the_same_threshold(monkeypatch) -> None:
    """The claim being optimised on: re-cutting equals re-running.

    Rather than mock a model, this reproduces predict()'s decision arithmetic
    verbatim from its source values and asserts the cheap path agrees on every
    field the UI reads.
    """
    from onnm.ood import should_defer

    probabilities = {"normal": 0.28, "benign": 0.44, "malignant": 0.28}
    ordered = np.array(list(probabilities.values()), dtype=np.float64)
    floor, gate = 0.65, 0.90

    for threshold in (0.05, 0.25, 0.4959, 0.72, 0.95):
        # --- what predict() would compute from scratch ---
        lesion = 1.0 - probabilities["normal"]
        is_lesion = lesion >= threshold
        defer, max_p, entropy = should_defer(
            ordered, uncertainty_floor=floor, entropy_gate=gate
        )
        inconclusive = bool(is_lesion and defer)
        expected_label = (
            INCONCLUSIVE_LABEL if inconclusive else LESION_LABEL if is_lesion else NORMAL_LABEL
        )
        expected_conf = 100.0 * (lesion if is_lesion else 1.0 - lesion)

        # --- what the cheap path computes ---
        actual = _result(0.28, 0.44, 0.28).with_threshold(
            threshold, uncertainty_floor=floor, entropy_gate=gate
        )

        assert actual.label == expected_label, threshold
        assert actual.confidence == pytest.approx(expected_conf), threshold
        assert actual.inconclusive == inconclusive, threshold
        assert actual.max_probability == pytest.approx(max_p), threshold
        assert actual.predictive_entropy == pytest.approx(entropy), threshold
        assert actual.threshold == pytest.approx(threshold), threshold
