"""Tests for post-hoc calibration and threshold selection.

The properties pinned here are the ones whose violation would be invisible.
Temperature scaling that quietly changed a prediction, or a threshold search
that returned "call everything a lesion" to satisfy a sensitivity target, would
both produce plausible-looking output and a materially wrong clinical tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from onnm.calibrate import (
    Calibration,
    calibrate,
    expected_calibration_error,
    find_operating_point,
    fit_temperature,
    format_report,
    lesion_scores,
    negative_log_likelihood,
    sweep_thresholds,
)


@pytest.fixture
def overconfident_logits() -> tuple[np.ndarray, np.ndarray]:
    """A model right about 75% of the time that reports near-certainty.

    Overconfidence is a statement about the gap between confidence and accuracy,
    not about logit magnitude alone -- a model that is always right is *correctly*
    confident and wants T < 1. So the margin here is small enough to leave real
    errors, and the whole vector is then scaled up: right sometimes, certain
    always. That is the state focal loss leaves a network in.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 100)
    logits = rng.normal(0.0, 1.0, size=(len(labels), 3))
    logits[np.arange(len(labels)), labels] += 1.2
    return logits * 4.0, labels


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def test_temperature_never_changes_a_prediction(
    overconfident_logits: tuple[np.ndarray, np.ndarray],
) -> None:
    """The property the whole method rests on.

    Temperature scaling divides every logit by one positive scalar, which is
    monotone and therefore argmax-preserving. If this ever fails, calibration
    has silently changed the model's diagnoses -- so it is asserted rather than
    assumed, across temperatures spanning two orders of magnitude.
    """
    logits, _ = overconfident_logits
    baseline = logits.argmax(axis=1)
    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0):
        scaled = torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()
        assert np.array_equal(scaled.argmax(axis=1), baseline)


def test_fit_tracks_a_known_rescaling() -> None:
    """Divide the logits by k and the fitted temperature must divide by k too.

    The absolute optimum depends on the data, so asserting a specific number
    would only pin the fixture. The scale equivariance is the real property: it
    is what makes the fit invert whatever miscalibration training introduced,
    rather than converging to a constant.
    """
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 3, size=600)
    logits = rng.normal(0.0, 1.0, size=(600, 3))
    logits[np.arange(600), labels] += 2.5

    base = fit_temperature(logits, labels)
    # k values chosen so base/k stays inside the [0.1, 10] clamp; outside it the
    # returned value is a bound, not an optimum, which is covered separately.
    for k in (0.25, 0.5, 2.0):
        assert fit_temperature(logits / k, labels) == pytest.approx(base / k, rel=0.05)


def test_fit_matches_a_brute_force_optimum() -> None:
    """Guards the failure that motivated the guarded fit in the first place.

    A bare LBFGS run on this objective settles up to 25x from the true optimum
    without raising anything -- a silently wrong temperature that would
    miscalibrate every probability downstream. This pins the fit against an
    exhaustive search.
    """
    rng = np.random.default_rng(21)
    labels = rng.integers(0, 3, size=400)
    logits = rng.normal(0.0, 1.0, size=(400, 3))
    logits[np.arange(400), labels] += 1.5

    tensor_logits = torch.as_tensor(logits, dtype=torch.float32)
    tensor_labels = torch.as_tensor(labels, dtype=torch.long)
    grid = np.linspace(0.15, 8.0, 2000)
    losses = [
        float(torch.nn.functional.cross_entropy(tensor_logits / t, tensor_labels)) for t in grid
    ]
    assert fit_temperature(logits, labels) == pytest.approx(
        float(grid[int(np.argmin(losses))]), rel=0.05
    )


def test_boundary_fit_is_reported_not_silent(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temperature pinned to the bound is a limit, not a fit, and must say so."""
    import logging

    rng = np.random.default_rng(5)
    labels = rng.integers(0, 3, size=300)
    logits = rng.normal(0.0, 1.0, size=(300, 3))
    logits[np.arange(300), labels] += 2.5

    # onnm's logger sets propagate=False so records never reach caplog's root
    # handler. Re-enable it for this test rather than reaching into handlers.
    monkeypatch.setattr(logging.getLogger("onnm.calibrate"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="onnm.calibrate"):
        result = fit_temperature(logits / 8.0, labels, max_temperature=10.0)

    assert result == pytest.approx(0.1)
    assert any("edge of its" in record.getMessage() for record in caplog.records)


def test_overconfident_logits_get_softened(
    overconfident_logits: tuple[np.ndarray, np.ndarray],
) -> None:
    logits, labels = overconfident_logits
    assert fit_temperature(logits, labels) > 1.0


def test_temperature_improves_nll(overconfident_logits: tuple[np.ndarray, np.ndarray]) -> None:
    """NLL is what the fit minimises, so it must not get worse."""
    logits, labels = overconfident_logits
    temperature = fit_temperature(logits, labels)

    before = negative_log_likelihood(labels, torch.softmax(torch.as_tensor(logits), 1).numpy())
    after = negative_log_likelihood(
        labels, torch.softmax(torch.as_tensor(logits) / temperature, 1).numpy()
    )
    assert after <= before + 1e-6


def test_degenerate_fit_falls_back_to_one() -> None:
    """A single-class validation set must not return a nonsense temperature."""
    logits = np.tile([5.0, 0.0, 0.0], (20, 1))
    assert fit_temperature(logits, np.zeros(20, dtype=int)) > 0.0


# ---------------------------------------------------------------------------
# Reliability measures
# ---------------------------------------------------------------------------
def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    """100 predictions at 100% confidence, all correct -> perfectly calibrated."""
    y_prob = np.tile([1.0, 0.0, 0.0], (100, 1))
    assert expected_calibration_error(np.zeros(100, dtype=int), y_prob) == pytest.approx(0.0)


def test_ece_catches_confident_and_wrong() -> None:
    """The failure mode that matters: a confident number that is not earned."""
    y_prob = np.tile([0.99, 0.005, 0.005], (100, 1))
    assert expected_calibration_error(np.ones(100, dtype=int), y_prob) == pytest.approx(
        0.99, abs=0.02
    )


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def test_lesion_score_is_one_minus_normal() -> None:
    y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4]])
    assert np.allclose(lesion_scores(y_prob), [0.3, 0.9])


def test_picks_the_most_specific_threshold_meeting_the_target() -> None:
    """Sensitivity is the constraint; specificity is what gets maximised.

    A search that returned the *lowest* qualifying threshold would also satisfy
    the sensitivity target while needlessly flagging normal films -- which is
    the exact false-positive complaint this module exists to answer.
    """
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.5, 50), np.linspace(0.5, 1.0, 50)])
    y_prob = np.column_stack([1 - scores, scores, np.zeros(100)])

    result = find_operating_point(y_true, y_prob, target_sensitivity=0.90)
    assert result["sensitivity"] >= 0.90
    assert result["specificity"] > 0.80

    lower = [
        r for r in result["sweep"]
        if r["sensitivity"] >= 0.90 and r["threshold"] < result["threshold"]
    ]
    assert all(r["specificity"] <= result["specificity"] for r in lower)


def test_impossible_target_warns_instead_of_returning_a_useless_cut() -> None:
    """A model that cannot reach the target must say so.

    Returning threshold 0.0 would score 100% sensitivity by calling every image
    a lesion. That satisfies the letter of the request and is worse than no
    answer, so the search reports the conflict.
    """
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])  # ranking inverted
    y_prob = np.column_stack([1 - scores, scores, np.zeros(100)])

    result = find_operating_point(y_true, y_prob, target_sensitivity=0.95)
    assert result["warnings"]
    assert "cannot support" in result["warnings"][0] or "no threshold" in result["warnings"][0]


def test_low_specificity_at_target_is_flagged() -> None:
    """Reaching the sensitivity target by flagging everything is called out."""
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 100 + [1] * 100)
    scores = rng.uniform(0, 1, 200)  # no signal at all
    y_prob = np.column_stack([1 - scores, scores, np.zeros(200)])

    result = find_operating_point(y_true, y_prob, target_sensitivity=0.95, min_specificity=0.5)
    assert any("specificity" in w for w in result["warnings"])


def test_sweep_is_monotone_in_the_right_directions() -> None:
    """Raising the threshold can only lose sensitivity and gain specificity."""
    rng = np.random.default_rng(11)
    y_true = rng.integers(0, 2, 200)
    scores = rng.uniform(0, 1, 200)
    y_prob = np.column_stack([1 - scores, scores, np.zeros(200)])

    rows = sweep_thresholds(y_true, y_prob)
    sensitivities = [r["sensitivity"] for r in rows]
    specificities = [r["specificity"] for r in rows]
    # strict=False: an offset slice is one shorter by construction.
    assert all(a >= b - 1e-9 for a, b in zip(sensitivities, sensitivities[1:], strict=False))
    assert all(a <= b + 1e-9 for a, b in zip(specificities, specificities[1:], strict=False))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_calibrate_produces_a_usable_result(
    cfg, overconfident_logits: tuple[np.ndarray, np.ndarray]
) -> None:
    logits, labels = overconfident_logits
    result = calibrate(logits, labels, cfg)

    assert result.temperature > 0
    assert 0.0 <= result.lesion_threshold <= 1.0
    assert result.n_val == len(labels)
    assert result.fitted_on == "val"
    assert "CALIBRATION" in format_report(result)


def test_calibration_round_trips_through_json(tmp_path: Path) -> None:
    """Every field survives save/load, including ones added after this was written.

    Compared field-by-field with NaN treated as equal to NaN, rather than with
    ``==`` on the dataclass: several defaults are NaN, which never compares equal
    to itself, and listing the fields explicitly would silently stop covering any
    field added later.
    """
    import math

    original = Calibration(
        temperature=1.7, lesion_threshold=0.31, malignant_threshold=0.62,
        target_sensitivity=0.95, min_specificity=0.80, mode="specificity_floor",
        achieved_sensitivity=0.96, achieved_specificity=0.71,
        n_val=535, warnings=["careful"],
    )
    restored = Calibration.load(original.save(tmp_path / "calibration.json"))

    for name in Calibration.__dataclass_fields__:
        a, b = getattr(original, name), getattr(restored, name)
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), name
        else:
            assert a == b, name


def test_unknown_fields_are_ignored_on_load(tmp_path: Path) -> None:
    """A calibration written by a future version must not crash an older app."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": 2.0, "some_future_field": 1}), encoding="utf-8")
    assert Calibration.load(path).temperature == 2.0


def test_for_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    """An uncalibrated checkpoint is a normal state, not an error."""
    assert Calibration.for_checkpoint(tmp_path / "best.pt") is None


def test_corrupt_calibration_degrades_to_none(tmp_path: Path) -> None:
    (tmp_path / "calibration.json").write_text("{not json", encoding="utf-8")
    assert Calibration.for_checkpoint(tmp_path / "best.pt") is None


def test_apply_matches_manual_scaling() -> None:
    calibration = Calibration(temperature=2.5)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(
        calibration.apply(logits), torch.softmax(logits / 2.5, dim=-1)
    )


# ---------------------------------------------------------------------------
# Specificity-floor mode
# ---------------------------------------------------------------------------
@pytest.fixture
def graded_scores() -> tuple[np.ndarray, np.ndarray]:
    """Overlapping distributions, so neither constraint is free."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 200 + [1] * 200)
    scores = np.clip(
        np.concatenate([rng.normal(0.35, 0.18, 200), rng.normal(0.65, 0.18, 200)]), 0, 1
    )
    return y_true, np.column_stack([1 - scores, scores, np.zeros(400)])


def test_specificity_floor_respects_its_constraint(
    graded_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """The mode the >=80% specificity requirement needs."""
    y_true, y_prob = graded_scores
    result = find_operating_point(
        y_true, y_prob, min_specificity=0.80, mode="specificity_floor"
    )
    assert result["specificity"] >= 0.80
    assert result["mode"] == "specificity_floor"


def test_specificity_floor_maximises_sensitivity_under_it(
    graded_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """Among thresholds meeting the floor, none may be more sensitive."""
    y_true, y_prob = graded_scores
    result = find_operating_point(
        y_true, y_prob, min_specificity=0.80, mode="specificity_floor"
    )
    qualifying = [r for r in result["sweep"] if r["specificity"] >= 0.80]
    assert result["sensitivity"] == pytest.approx(max(r["sensitivity"] for r in qualifying))


def test_the_two_modes_bracket_the_trade_off(
    graded_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """Demanding sensitivity costs specificity and vice versa -- that is the trade."""
    y_true, y_prob = graded_scores
    sens_first = find_operating_point(
        y_true, y_prob, target_sensitivity=0.95, mode="sensitivity_floor"
    )
    spec_first = find_operating_point(
        y_true, y_prob, min_specificity=0.80, mode="specificity_floor"
    )
    assert sens_first["sensitivity"] >= spec_first["sensitivity"]
    assert spec_first["specificity"] >= sens_first["specificity"]
    assert spec_first["threshold"] >= sens_first["threshold"]


def test_conflicting_constraints_are_reported(
    graded_scores: tuple[np.ndarray, np.ndarray],
) -> None:
    """Both floors at once is usually infeasible, and must be said so."""
    y_true, y_prob = graded_scores
    result = find_operating_point(
        y_true, y_prob, target_sensitivity=0.99, min_specificity=0.95,
        mode="specificity_floor",
    )
    assert result["warnings"]


def test_unreachable_specificity_floor_warns() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])  # inverted ranking
    y_prob = np.column_stack([1 - scores, scores, np.zeros(100)])

    result = find_operating_point(y_true, y_prob, min_specificity=0.95, mode="specificity_floor")
    assert any("specificity" in w for w in result["warnings"])


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        find_operating_point(np.array([0, 1]), np.array([[0.5, 0.5, 0.0]] * 2), mode="whatever")


def test_calibration_records_both_operating_points(
    cfg, overconfident_logits: tuple[np.ndarray, np.ndarray]
) -> None:
    """The unchosen mode is reported too, so the trade stays visible."""
    logits, labels = overconfident_logits
    result = calibrate(logits, labels, cfg)
    assert np.isfinite(result.alternative_threshold)
    assert "opposite constraint" in format_report(result)
