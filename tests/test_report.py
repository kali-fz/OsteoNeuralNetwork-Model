"""Tests for the standalone HTML case-report builder (torch-free)."""

from __future__ import annotations

import base64

from report import build_html_report

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAA"
    "AABJRU5ErkJggg=="
)


def _build(**overrides):
    kwargs = dict(
        filename="femur_ap.png",
        verdict="Potential Bone Lesion",
        confidence_pct=87.3,
        class_probabilities={"normal": 0.127, "benign": 0.61, "malignant": 0.263},
        lesion_probability=0.873,
        threshold=0.496,
        calibrated=True,
        temperature=1.41,
        inconclusive=False,
        max_probability=0.61,
        predictive_entropy=0.41,
        checkpoint_name="full-20260822-041653",
        app_version="0.1.0",
        disclaimer="Research prototype. Not a medical device.",
    )
    kwargs.update(overrides)
    return build_html_report(**kwargs)


def test_report_contains_verdict_probabilities_and_disclaimer() -> None:
    html_doc = _build()
    assert html_doc.startswith("<!DOCTYPE html>")
    assert "Potential Bone Lesion" in html_doc
    assert "87.3% confidence" in html_doc
    assert "full-20260822-041653" in html_doc
    assert "Not a medical device." in html_doc
    assert "T=1.41" in html_doc


def test_report_escapes_hostile_filenames() -> None:
    html_doc = _build(filename="<script>alert(1)</script>.png")
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;script&gt;" in html_doc


def test_report_embeds_images_as_data_uris() -> None:
    html_doc = _build(original_png=_PNG_1PX, overlay_png=_PNG_1PX, cam_class="malignant")
    assert html_doc.count("data:image/png;base64,") == 2
    assert "malignant" in html_doc


def test_report_omits_imagery_section_without_images() -> None:
    html_doc = _build()
    assert "data:image/png" not in html_doc
    assert "<h2>Imagery</h2>" not in html_doc


def test_inconclusive_report_carries_uncertainty_note() -> None:
    html_doc = _build(
        verdict="Non-Diagnostic / Inconclusive",
        inconclusive=True,
        max_probability=0.42,
        predictive_entropy=0.95,
    )
    assert "Uncertainty gate" in html_doc
    assert "42.0%" in html_doc


def test_uncalibrated_report_says_so() -> None:
    html_doc = _build(calibrated=False, temperature=1.0)
    assert "UNCALIBRATED" in html_doc
