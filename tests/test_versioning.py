"""Tests for the model version ledger and the bone-versus-misc metric.

The ledger exists because the community loop retrains on data the model helped
collect, and feedback loops drift. What has to hold is not that versions are
recorded -- that part is hard to get wrong -- but that **a bad version cannot
take over serving**. Every test below is ultimately about that.

The gate metric is tested separately because it is the one number that says
whether the project is getting better at the thing the daily loop is supposed to
improve: telling a bone from a hotdog.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from onnm.ood_eval import evaluate_gate
from onnm.versioning import (
    GUARDED_METRICS,
    REGRESSION_TOLERANCE,
    Version,
    bump,
    load_registry,
    parse_version,
    register,
    render_markdown,
    save_registry,
    serving,
    should_promote,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

GOOD = {"macro_roc_auc": 0.89, "malignant_recall": 0.63, "misc_rejection": 0.80}


def _seed() -> list[Version]:
    versions, _ = register([], run="run-a", metrics=dict(GOOD))
    return versions


# ---------------------------------------------------------------------------
# Version numbers
# ---------------------------------------------------------------------------
def test_a_data_only_retrain_is_a_patch() -> None:
    """v1.0.0 -> v1.0.1 after a day of approvals, which is the stated cadence."""
    assert bump("v1.0.0", "patch") == "v1.0.1"
    assert bump("v1.0.9", "patch") == "v1.0.10"


def test_a_recipe_change_resets_the_patch_count() -> None:
    """A minor bump means the comparison to the previous version is not like for like."""
    assert bump("v1.0.7", "minor") == "v1.1.0"
    assert bump("v1.3.7", "major") == "v2.0.0"


def test_a_malformed_version_raises_rather_than_guessing() -> None:
    for bad in ("1.0", "v1.0.x", "", "latest"):
        with pytest.raises(ValueError):
            parse_version(bad)


# ---------------------------------------------------------------------------
# The promotion guard — the part that protects the model
# ---------------------------------------------------------------------------
def test_the_first_version_is_always_promoted() -> None:
    ok, reason = should_promote(GOOD, None)
    assert ok
    assert "first" in reason


def test_a_regression_blocks_promotion() -> None:
    """The whole safety net: a worse model does not get to serve.

    A drop in ranking quality is the case that matters most, because ROC-AUC is
    threshold-independent -- it cannot be recovered by re-tuning the operating
    point later, so a fall here is a genuinely worse model rather than a
    differently-tuned one.
    """
    worse = {**GOOD, "macro_roc_auc": 0.85}
    ok, reason = should_promote(worse, GOOD)
    assert not ok
    assert "macro_roc_auc" in reason


def test_getting_worse_at_bone_versus_misc_also_blocks_promotion() -> None:
    """The second thing the project is trying to improve, and the easier one to lose.

    A run can gain lesion accuracy while becoming more willing to diagnose a
    photograph. Nothing in the lesion metrics would show it.
    """
    worse = {**GOOD, "misc_rejection": 0.50, "macro_roc_auc": 0.92}
    ok, reason = should_promote(worse, GOOD)
    assert not ok
    assert "misc_rejection" in reason


def test_noise_within_tolerance_does_not_block_promotion() -> None:
    """A zero-tolerance gate would block every version and be overridden by habit.

    Bootstrap noise on 536 test images moves malignant recall by more than a
    point between identical runs, so the tolerance exists to distinguish a
    collapse from a wobble.
    """
    wobble = {**GOOD, "malignant_recall": GOOD["malignant_recall"] - REGRESSION_TOLERANCE / 2}
    ok, _ = should_promote(wobble, GOOD)
    assert ok


def test_a_flat_retrain_is_promoted() -> None:
    """The daily loop accumulates data; it need not prove a gain to be allowed to exist."""
    ok, _ = should_promote(dict(GOOD), dict(GOOD))
    assert ok


def test_a_metric_the_incumbent_never_measured_is_skipped_not_assumed() -> None:
    """Scoring it as "no change" would let a worse run through on a technicality."""
    incumbent = {"macro_roc_auc": 0.89, "malignant_recall": 0.63}
    candidate = {"macro_roc_auc": 0.89, "malignant_recall": 0.63, "misc_rejection": 0.1}
    ok, reason = should_promote(candidate, incumbent)
    assert ok, "an unmeasurable metric must not block"
    assert "misc_rejection" not in reason


def test_nothing_comparable_at_all_refuses_rather_than_waves_through() -> None:
    ok, reason = should_promote({"something_else": 1.0}, GOOD)
    assert not ok
    assert "no guarded metric" in reason


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_a_held_version_leaves_the_incumbent_serving() -> None:
    """The rollback *is* the default: a bad run changes nothing about what serves."""
    versions = _seed()
    versions, added = register(
        versions, run="run-b", metrics={**GOOD, "macro_roc_auc": 0.70}
    )
    assert added.status == "held"
    assert added.held_because
    still = serving(versions)
    assert still is not None
    assert still.run == "run-a", "the incumbent must keep serving"


def test_a_promoted_version_supersedes_the_previous_one() -> None:
    versions = _seed()
    versions, added = register(versions, run="run-b", metrics={**GOOD, "macro_roc_auc": 0.91})
    assert added.status == "serving"
    assert [v.status for v in versions] == ["superseded", "serving"]


def test_a_held_version_is_still_recorded() -> None:
    """"We tried more data and it got worse" is exactly what a ledger is for."""
    versions = _seed()
    versions, added = register(versions, run="run-b", metrics={**GOOD, "misc_rejection": 0.1})
    assert len(versions) == 2
    assert added.version == "v1.0.1"
    assert added.parent == "v1.0.0"


def test_registering_the_same_number_twice_raises() -> None:
    versions = _seed()
    with pytest.raises(ValueError, match="already registered"):
        register(versions, run="run-b", metrics=dict(GOOD), version="v1.0.0")


def test_the_registry_round_trips(tmp_path: Path) -> None:
    versions = _seed()
    versions, _ = register(versions, run="run-b", metrics={**GOOD, "macro_roc_auc": 0.90},
                           community={"lesion_rows_total": 12}, note="daily")
    path = tmp_path / "model_versions.json"
    save_registry(versions, path)
    restored = load_registry(path)
    assert [v.as_dict() for v in restored] == [v.as_dict() for v in versions]


def test_versions_load_in_order_regardless_of_file_order(tmp_path: Path) -> None:
    """v1.0.10 sorts after v1.0.9, which string ordering gets wrong."""
    versions = _seed()
    for index in range(2, 12):
        versions, _ = register(versions, run=f"run-{index}", metrics=dict(GOOD))
    path = tmp_path / "r.json"
    save_registry(versions, path)
    assert [v.version for v in load_registry(path)][-2:] == ["v1.0.9", "v1.0.10"]


# ---------------------------------------------------------------------------
# The rendered ledger
# ---------------------------------------------------------------------------
def test_the_markdown_names_what_is_serving_and_why_anything_was_held() -> None:
    versions = _seed()
    versions, _ = register(versions, run="run-b", metrics={**GOOD, "macro_roc_auc": 0.5})
    text = render_markdown(versions)
    assert "v1.0.0 (`run-a`)" in text, "the reader must be able to see what is live"
    assert "Not promoted:" in text
    assert "macro_roc_auc" in text


def test_the_committed_ledger_is_in_step_with_its_json() -> None:
    """ONN.md is generated. A ledger that disagrees with itself is worse than none.

    Regenerate with:  python scripts/version_model.py render
    """
    markdown = REPO_ROOT / "ONN.md"
    if not markdown.is_file():
        pytest.skip("no ledger committed yet")
    assert render_markdown(load_registry()) == markdown.read_text(encoding="utf-8")


def test_the_ledger_does_not_headline_accuracy() -> None:
    """"Never malignant" scores 90.9% here, so accuracy is not a guarded metric."""
    assert "accuracy" not in GUARDED_METRICS
    assert not any("accuracy" in name for name in GUARDED_METRICS)


# ---------------------------------------------------------------------------
# Bone versus misc
# ---------------------------------------------------------------------------
def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _manifest(path: Path, images: list[Path]) -> None:
    rows = ["image,image_id,is_radiograph,bucket,source"]
    rows += [f"{image.as_posix()},{image.stem},0,misc,community" for image in images]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_the_gate_is_scored_against_reviewed_misuse(tmp_path: Path) -> None:
    """Noise is what the entropy heuristic is built to catch, so it should be caught."""
    rng = np.random.default_rng(0)
    noisy = tmp_path / "images" / "hotdog.png"
    _write_png(noisy, rng.integers(0, 256, (256, 256), dtype=np.uint8))
    manifest = tmp_path / "ood_manifest.csv"
    _manifest(manifest, [noisy])

    report = evaluate_gate(manifest)
    assert report.misc_total == 1
    assert report.misc_rejection == 1.0
    assert report.greyscale_lower_bound, "the stored copy is greyscale; say so"


def test_a_non_radiograph_the_gate_accepts_is_named_not_just_counted(
    tmp_path: Path,
) -> None:
    """Each miss reached the classifier and got a clinical-sounding verdict.

    A rate alone would say "80%" and leave the interesting twenty percent
    anonymous; the review console needs to know *which* images to look at.
    """
    smooth = tmp_path / "images" / "wall.png"
    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    _write_png(smooth, gradient)
    manifest = tmp_path / "ood_manifest.csv"
    _manifest(manifest, [smooth])

    report = evaluate_gate(manifest)
    assert report.misc_total == 1
    if report.misc_rejection == 0.0:
        assert report.misses == ["wall"]


def test_no_reviewed_misuse_yet_is_not_an_error(tmp_path: Path) -> None:
    """Day one. The pipeline must not fail because nobody has approved anything."""
    report = evaluate_gate(tmp_path / "does-not-exist.csv")
    assert report.misc_total == 0
    assert report.misc_rejection is None
    assert report.as_dict()["misc_rejection"] is None


def test_rejection_and_acceptance_are_reported_separately(tmp_path: Path) -> None:
    """A gate that rejects everything scores 1.0 on one and 0.0 on the other.

    Folding them into a single number would hide exactly the failure worth
    catching, which is why the ledger guards the rejection rate and prints
    acceptance beside it rather than combining them.
    """
    rng = np.random.default_rng(1)
    noisy = tmp_path / "images" / "noise.png"
    _write_png(noisy, rng.integers(0, 256, (256, 256), dtype=np.uint8))
    manifest = tmp_path / "m.csv"
    _manifest(manifest, [noisy])

    report = evaluate_gate(manifest, [noisy])
    assert report.misc_rejection == 1.0
    assert report.bone_acceptance == 0.0
    assert "misc_rejection" in report.as_dict()
    assert "bone_acceptance" in report.as_dict()


def test_a_manifest_row_whose_image_vanished_is_not_counted(tmp_path: Path) -> None:
    """Otherwise an unmounted Drive reads as a gate that suddenly got worse."""
    missing = tmp_path / "images" / "gone.png"
    manifest = tmp_path / "m.csv"
    _manifest(manifest, [missing])
    report = evaluate_gate(manifest)
    assert report.misc_total == 0
    assert report.misc_rejection is None
