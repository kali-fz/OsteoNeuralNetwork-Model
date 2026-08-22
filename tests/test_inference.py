"""Tests for the app-facing inference layer.

The checkpoint is built here rather than loaded from ``reports/``: that
directory is gitignored, so a suite that depended on it would silently skip on a
clean clone -- which is the same as not having the tests. A randomly-initialised
DenseNet-121 predicts nonsense, but nothing below asserts anything about *which*
class comes out. What is asserted is that the three input paths agree, that
DICOM conventions survive the byte round-trip, and that the threshold and the
verdict cannot disagree.

The property that matters most is the last one: the app's headline is a binary
verdict derived from a 3-class head, and a sign error there would read as
plausible on every image.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from onnm.inference import (
    LESION_LABEL,
    NORMAL_LABEL,
    UPLOAD_TYPES,
    RadiographClassifier,
    find_checkpoints,
    render_overlay,
    to_display_uint8,
)
from onnm.io_radiograph import RadiographReadError, UnsupportedFormatError


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real checkpoint on disk, with an embedded config, built from scratch.

    ``pretrained`` is forced off so the fixture needs no network. The saved
    config keeps ``pretrained: true`` on purpose -- that is what a real training
    run writes, and loading it back is how we check the classifier overrides it.
    """
    from onnm.config import load_config
    from onnm.model import build_model

    cfg = load_config("configs/base.yaml")
    saved = cfg.to_dict()
    cfg._data["model"]["pretrained"] = False

    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save(
        {"model": build_model(cfg).state_dict(), "epoch": 3, "config": saved,
         "malignant_recall": 0.42},
        path,
    )
    return path


@pytest.fixture(scope="module")
def classifier(checkpoint: Path) -> RadiographClassifier:
    return RadiographClassifier(checkpoint, device=torch.device("cpu"), warmup=False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_pretrained_is_forced_off(classifier: RadiographClassifier) -> None:
    """No ImageNet download at load time; the state dict replaces everything.

    Without this the app needs internet on a cold torch cache, which breaks the
    offline guarantee it is built around.
    """
    assert classifier.cfg.model.pretrained is False


def test_config_comes_from_the_checkpoint(classifier: RadiographClassifier) -> None:
    assert classifier.image_size == 256
    assert classifier.class_names == ["normal", "benign", "malignant"]
    assert classifier.normal_index == 0
    assert classifier.lesion_indices == [1, 2]


def test_describe_reports_checkpoint_metrics(classifier: RadiographClassifier) -> None:
    info = classifier.describe()
    assert info["architecture"] == "densenet121"
    assert info["trained_epochs"] == 4                # epoch is 0-based on disk
    assert info["malignant_recall"] == pytest.approx(0.42)


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        RadiographClassifier("does/not/exist.pt")


def test_find_checkpoints_tolerates_missing_reports_dir(tmp_path: Path) -> None:
    assert find_checkpoints(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
def test_all_three_input_forms_agree(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """A path, raw bytes and a file object must produce identical probabilities.

    Streamlit hands over bytes; every script hands over a path. If those diverge
    the app is not showing what the CLI would.
    """
    payload = jpeg_image.read_bytes()
    from_path = classifier.predict(jpeg_image)
    from_bytes = classifier.predict(payload, filename=jpeg_image.name)
    from_file = classifier.predict(io.BytesIO(payload), filename=jpeg_image.name)

    assert from_path.class_probabilities == from_bytes.class_probabilities
    assert from_path.class_probabilities == from_file.class_probabilities


def test_dicom_bytes_honour_photometric_interpretation(
    classifier: RadiographClassifier, mono1_dicom: Path, mono2_dicom: Path
) -> None:
    """A MONOCHROME1 upload must not reach the model as a negative.

    This is the failure ``io_radiograph`` exists to prevent, re-checked through
    the byte path the app actually uses -- the temp-file round-trip is exactly
    where a lost header would go unnoticed.
    """
    inverted = classifier.predict(mono1_dicom.read_bytes(), filename="a.dcm")
    upright = classifier.predict(mono2_dicom.read_bytes(), filename="b.dcm")

    assert inverted.source_meta["inverted"] is True
    assert upright.source_meta["inverted"] is False
    assert np.allclose(
        list(inverted.class_probabilities.values()),
        list(upright.class_probabilities.values()),
        atol=1e-4,
    )


def test_extensionless_dicom_is_sniffed(
    classifier: RadiographClassifier, mono2_dicom: Path
) -> None:
    """De-identified exports routinely lose the extension; the preamble decides."""
    payload = mono2_dicom.read_bytes()
    sniffed = classifier.predict(payload, filename="anonymised_export")
    named = classifier.predict(payload, filename="b.dcm")
    assert sniffed.class_probabilities == named.class_probabilities


def test_temp_file_is_cleaned_up(classifier: RadiographClassifier, jpeg_image: Path) -> None:
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("*.jpeg"))
    classifier.predict(jpeg_image.read_bytes(), filename="scan.jpeg")
    assert set(Path(tempfile.gettempdir()).glob("*.jpeg")) == before


def test_source_meta_hides_the_temp_path(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    result = classifier.predict(jpeg_image.read_bytes(), filename="patient_scan.jpeg")
    assert result.source_meta["filename"] == "patient_scan.jpeg"
    assert "filename_or_obj" not in result.source_meta


@pytest.mark.parametrize(
    ("payload", "name", "expected"),
    [
        (b"", "x.png", RadiographReadError),
        (b"\x89PNG\r\n\x1a\n garbage", "x.png", RadiographReadError),
        (b"some text", "notes.txt", UnsupportedFormatError),
        (b"no extension and no preamble", "mystery", UnsupportedFormatError),
    ],
)
def test_bad_uploads_raise(
    classifier: RadiographClassifier, payload: bytes, name: str, expected: type
) -> None:
    """Every rejection path must raise, so the app can render a message.

    Returning a default prediction for an undecodable file would be the worst
    possible behaviour here: a confident verdict on an image nobody read.
    """
    with pytest.raises(expected):
        classifier.predict(payload, filename=name)


# ---------------------------------------------------------------------------
# The binary verdict
# ---------------------------------------------------------------------------
def test_lesion_probability_is_one_minus_normal(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    result = classifier.predict(jpeg_image)
    probabilities = result.class_probabilities
    assert result.lesion_probability == pytest.approx(
        probabilities["benign"] + probabilities["malignant"], abs=1e-6
    )
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_verdict_and_threshold_never_disagree(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """The headline label must be exactly ``lesion_probability >= threshold``."""
    for threshold in (0.05, 0.25, 0.5, 0.75, 0.95):
        result = classifier.predict(jpeg_image, threshold=threshold, with_heatmap=False)
        expected = LESION_LABEL if result.lesion_probability >= threshold else NORMAL_LABEL
        assert result.label == expected
        assert result.is_lesion == (expected is LESION_LABEL)


def test_confidence_backs_the_reported_label(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """Confidence must describe the verdict shown, not always the lesion class.

    Reporting P(lesion) under a "Normal" headline would invert the meaning of
    the single number a reader is most likely to quote.
    """
    for threshold in (0.05, 0.5, 0.95):
        result = classifier.predict(jpeg_image, threshold=threshold, with_heatmap=False)
        expected = result.lesion_probability if result.is_lesion else 1 - result.lesion_probability
        assert result.confidence == pytest.approx(100 * expected, abs=1e-4)
        assert 0.0 <= result.confidence <= 100.0


def test_result_serialises_to_json(classifier: RadiographClassifier, jpeg_image: Path) -> None:
    payload = json.loads(json.dumps(classifier.predict(jpeg_image).as_dict()))
    assert payload["label"] in (NORMAL_LABEL, LESION_LABEL)
    assert set(payload["class_probabilities"]) == {"normal", "benign", "malignant"}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def test_heatmap_shape_and_range(classifier: RadiographClassifier, jpeg_image: Path) -> None:
    result = classifier.predict(jpeg_image)
    assert result.heatmap is not None
    assert result.heatmap.shape == (classifier.image_size, classifier.image_size)
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0


def test_heatmap_can_be_skipped(classifier: RadiographClassifier, jpeg_image: Path) -> None:
    result = classifier.predict(jpeg_image, with_heatmap=False)
    assert result.heatmap is None and result.cam_class is None


def test_auto_cam_targets_a_lesion_class(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """`auto` must never explain "normal" -- "where would it have been?" is the question."""
    result = classifier.predict(jpeg_image, cam_class="auto")
    assert result.cam_class in ("benign", "malignant")


def test_explicit_cam_class_is_honoured(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    assert classifier.predict(jpeg_image, cam_class="normal").cam_class == "normal"


def test_unknown_cam_class_raises(classifier: RadiographClassifier) -> None:
    with pytest.raises(ValueError, match="unknown cam_class"):
        classifier.resolve_cam_index(np.array([0.2, 0.3, 0.5]), "sarcoma")


def test_model_stays_in_eval_mode_after_gradcam(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """Grad-CAM's backward pass must not leave the model in train mode.

    If it did, the next prediction would run dropout and updated BatchNorm
    statistics, and two clicks on the same file would return different numbers.
    """
    classifier.predict(jpeg_image)
    assert not classifier.model.training

    first = classifier.predict(jpeg_image).class_probabilities
    second = classifier.predict(jpeg_image).class_probabilities
    assert first == second


def test_parameters_are_unchanged_by_prediction(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """Inference must not mutate weights, even though Grad-CAM backpropagates."""
    before = classifier.model.classifier[1].weight.detach().clone()
    classifier.predict(jpeg_image)
    assert torch.equal(before, classifier.model.classifier[1].weight.detach())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_to_display_uint8_handles_a_flat_image() -> None:
    """A blank film must not divide by zero on its way to the screen."""
    out = to_display_uint8(np.full((8, 8), 3.5))
    assert out.dtype == np.uint8 and out.max() == 0


def test_overlay_shape_and_dtype(classifier: RadiographClassifier, jpeg_image: Path) -> None:
    result = classifier.predict(jpeg_image)
    overlay = render_overlay(result.preprocessed_image, result.heatmap)
    assert overlay.shape == (classifier.image_size, classifier.image_size, 3)
    assert overlay.dtype == np.uint8


def test_zero_alpha_leaves_the_radiograph_untouched(
    classifier: RadiographClassifier, jpeg_image: Path
) -> None:
    """The opacity slider at 0 must show the film, not a tinted version of it."""
    result = classifier.predict(jpeg_image)
    overlay = render_overlay(result.preprocessed_image, result.heatmap, alpha=0.0)
    grey = to_display_uint8(result.preprocessed_image)
    assert np.array_equal(overlay, np.stack([grey] * 3, axis=-1))


@pytest.mark.parametrize("colormap", ["jet", "turbo", "inferno", "magma", "viridis", "hot"])
def test_every_offered_colormap_renders(
    classifier: RadiographClassifier, jpeg_image: Path, colormap: str
) -> None:
    result = classifier.predict(jpeg_image)
    overlay = render_overlay(result.preprocessed_image, result.heatmap, colormap=colormap)
    assert overlay.shape[-1] == 3


@pytest.mark.parametrize(("alpha", "floor"), [(0.0, 0.0), (1.0, 0.0), (0.0, 0.9), (1.0, 0.9)])
def test_slider_extremes_are_safe(
    classifier: RadiographClassifier, jpeg_image: Path, alpha: float, floor: float
) -> None:
    result = classifier.predict(jpeg_image)
    overlay = render_overlay(
        result.preprocessed_image, result.heatmap, alpha=alpha, threshold=floor
    )
    assert np.isfinite(overlay).all()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def test_upload_types_carry_no_leading_dot() -> None:
    """Streamlit's file_uploader rejects extensions written with a dot."""
    assert all(not t.startswith(".") for t in UPLOAD_TYPES)
    assert {"dcm", "png", "jpg", "jpeg"} <= set(UPLOAD_TYPES)


def test_shim_reexports_the_public_api() -> None:
    """``src/inference.py`` is the path the app spec named; keep it working."""
    import inference

    assert inference.RadiographClassifier is RadiographClassifier
    assert inference.render_overlay is render_overlay
