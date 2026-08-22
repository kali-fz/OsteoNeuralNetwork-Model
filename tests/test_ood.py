"""OOD gate tests: non-radiograph inputs must be rejected, not classified.

The closed-set softmax will happily assign a hotdog photo ~50% "benign", so the
pipeline's first line of defense is :mod:`onnm.ood`. Everything here is
synthetic and torch-free: the validator is pure numpy/PIL/pydicom by design so
this suite runs without the GPU stack.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from onnm.ood import (
    NonRadiographError,
    ensure_radiograph,
    predictive_entropy,
    should_defer,
    validate_image,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Fixtures: a plausible radiograph and a rogue's gallery of non-radiographs
# ---------------------------------------------------------------------------
def bone_phantom(height: int = 256, width: int = 192) -> np.ndarray:
    """A synthetic femur-like radiograph: dark air background, bright shaft.

    Deliberately built with the statistics the validator measures -- a peaked
    dark background, a smooth wide-range bone plateau, soft cortical edges --
    rather than the ramp phantom in conftest, whose uniform histogram is
    exactly the photographic signature the entropy check exists to reject.
    """
    rng = np.random.default_rng(7)
    image = rng.normal(18.0, 4.0, (height, width))

    yy, xx = np.mgrid[0:height, 0:width]
    center = width / 2 + 12.0 * np.sin(yy / 37.0)      # a gently curved shaft
    distance = np.abs(xx - center)
    half_width = 22.0
    shaft = np.clip((half_width - distance) / half_width, 0.0, 1.0)
    image += 190.0 * shaft ** 0.7
    image += 25.0 * np.exp(-((distance - half_width) ** 2) / 18.0)  # cortex

    return np.clip(image, 0, 255).astype(np.uint8)


def png_bytes(array: np.ndarray, mode: str = "L") -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def hotdog_photo(height: int = 240, width: int = 320) -> np.ndarray:
    """A colorful photographic stand-in: smooth warm-toned blobs, full RGB."""
    yy, xx = np.mgrid[0:height, 0:width]
    red = 170 + 60 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    green = 110 + 50 * np.sin(xx / 17.0 + 1.0)
    blue = 50 + 40 * np.cos(yy / 13.0)
    rng = np.random.default_rng(3)
    stack = np.stack([red, green, blue], axis=-1) + rng.normal(0, 6, (height, width, 3))
    return np.clip(stack, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 1: image-statistics validation
# ---------------------------------------------------------------------------
def test_synthetic_radiograph_passes() -> None:
    report = validate_image(bone_phantom())
    assert report.is_radiograph, [c.detail for c in report.failures]


def test_radiograph_png_payload_passes() -> None:
    report = validate_payload(png_bytes(bone_phantom()), "femur.png")
    assert report.is_radiograph, [c.detail for c in report.failures]


def test_radiograph_dicom_payload_passes(tmp_path) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    pixels = (bone_phantom().astype(np.uint16)) * 12  # a 12-bit-ish range

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("phantom.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()

    path = tmp_path / "phantom.dcm"
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:
        ds.save_as(str(path), write_like_original=False)

    report = validate_payload(path.read_bytes(), "phantom.dcm")
    assert report.is_radiograph, [c.detail for c in report.failures]


def test_color_photograph_is_rejected() -> None:
    report = validate_payload(png_bytes(hotdog_photo(), mode="RGB"), "hotdog.png")
    assert not report.is_radiograph
    assert any(check.name == "grayscale" for check in report.failures)


def test_uniform_noise_is_rejected() -> None:
    rng = np.random.default_rng(11)
    noise = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    report = validate_image(noise)
    assert not report.is_radiograph
    failed = {check.name for check in report.failures}
    assert failed & {"histogram_entropy", "edge_density"}


def test_blank_images_are_rejected() -> None:
    for value in (0, 128, 255):
        blank = np.full((256, 256), value, dtype=np.uint8)
        report = validate_image(blank)
        assert not report.is_radiograph
        assert any(check.name == "dynamic_range" for check in report.failures)


def test_tiny_image_is_rejected() -> None:
    report = validate_image(bone_phantom(32, 32))
    assert not report.is_radiograph
    assert any(check.name == "dimensions" for check in report.failures)


def test_undecodable_payloads_are_rejected() -> None:
    for payload, name in ((b"", "empty.png"), (b"definitely not an image", "junk.png")):
        report = validate_payload(payload, name)
        assert not report.is_radiograph
        assert any(check.name == "decodable" for check in report.failures)


def test_grayscale_reencoded_as_rgb_is_not_penalised() -> None:
    """A grayscale film saved as 3-channel PNG has zero channel spread."""
    rgb = np.stack([bone_phantom()] * 3, axis=-1)
    report = validate_payload(png_bytes(rgb, mode="RGB"), "scan.png")
    assert report.is_radiograph, [c.detail for c in report.failures]


def test_ensure_radiograph_raises_with_report() -> None:
    with pytest.raises(NonRadiographError) as raised:
        ensure_radiograph(b"not an image", "junk.png")
    assert "not recognized as a valid musculoskeletal radiograph" in str(raised.value)
    assert raised.value.report.failures


# ---------------------------------------------------------------------------
# Stage 2: softmax uncertainty gating
# ---------------------------------------------------------------------------
def test_predictive_entropy_bounds() -> None:
    assert predictive_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)
    assert predictive_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(1.0, abs=1e-9)


def test_uniform_softmax_is_deferred() -> None:
    """The hotdog signature: probability smeared across every class."""
    defer, max_prob, entropy = should_defer(
        np.array([0.34, 0.33, 0.33]),
        uncertainty_floor=0.65,
        entropy_gate=0.90,
    )
    assert defer
    assert max_prob < 0.65
    assert entropy > 0.99


def test_confident_prediction_is_not_deferred() -> None:
    defer, max_prob, entropy = should_defer(
        np.array([0.02, 0.08, 0.90]),
        uncertainty_floor=0.65,
        entropy_gate=0.90,
    )
    assert not defer
    assert max_prob == pytest.approx(0.90)
    assert entropy < 0.90


def test_gate_disabled_by_default() -> None:
    defer, _, _ = should_defer(np.array([0.34, 0.33, 0.33]))
    assert not defer


def test_floor_and_entropy_criteria_are_independent() -> None:
    # Fails the floor but not the entropy gate.
    skewed = np.array([0.60, 0.35, 0.05])
    assert should_defer(skewed, uncertainty_floor=0.65)[0]
    assert not should_defer(skewed, entropy_gate=0.90)[0]
