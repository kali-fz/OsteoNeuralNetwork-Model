"""Tests for radiograph decoding.

The centrepiece is ``test_mono1_matches_mono2``. Every other failure mode in
this file is loud -- an exception, a wrong shape. Photometric inversion is the
one that is silent: the array is valid, training converges, the metrics look
plausible, and the model has learned a negative of the world. Nothing
downstream can detect it. So it gets an exact-equality assertion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onnm.io_radiograph import (
    RadiographReadError,
    UnsupportedFormatError,
    read_dicom,
    read_radiograph,
    read_standard_image,
)


# ---------------------------------------------------------------------------
# The silent failure
# ---------------------------------------------------------------------------
def test_mono1_matches_mono2(mono1_dicom: Path, mono2_dicom: Path) -> None:
    """MONOCHROME1 must decode to the same image as its MONOCHROME2 twin."""
    mono1, meta1 = read_dicom(mono1_dicom)
    mono2, meta2 = read_dicom(mono2_dicom)

    assert meta1["inverted"] is True, "MONOCHROME1 was not inverted"
    assert meta2["inverted"] is False, "MONOCHROME2 must not be inverted"
    np.testing.assert_allclose(mono1, mono2, atol=1e-4)


def test_mono1_without_inversion_would_be_negative(
    mono1_dicom: Path, mono2_dicom: Path
) -> None:
    """Guard the guard: confirm the two files really do differ before inversion.

    Without this, a fixture bug that wrote identical pixel data to both files
    would make the test above pass for entirely the wrong reason.
    """
    import pydicom

    raw1 = pydicom.dcmread(str(mono1_dicom)).pixel_array.astype(np.float64)
    raw2 = pydicom.dcmread(str(mono2_dicom)).pixel_array.astype(np.float64)

    assert not np.allclose(raw1, raw2), "fixtures are identical; the test proves nothing"
    # Anti-correlated, as a negative should be.
    assert np.corrcoef(raw1.ravel(), raw2.ravel())[0, 1] < -0.99


# ---------------------------------------------------------------------------
# DICOM header handling
# ---------------------------------------------------------------------------
def test_voi_lut_applied(windowed_dicom: Path) -> None:
    arr, meta = read_dicom(windowed_dicom)
    assert meta["voi_lut_applied"] is True
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()


def test_voi_lut_can_be_disabled(windowed_dicom: Path) -> None:
    windowed, _ = read_dicom(windowed_dicom, apply_lut=True)
    raw, meta = read_dicom(windowed_dicom, apply_lut=False)
    assert meta["voi_lut_applied"] is False
    assert not np.allclose(windowed, raw), "windowing had no effect on the pixels"


def test_rescale_applied(rescaled_dicom: Path, mono2_dicom: Path) -> None:
    """RescaleSlope=2, RescaleIntercept=-1024 must actually be applied."""
    rescaled, meta = read_dicom(rescaled_dicom)
    plain, _ = read_dicom(mono2_dicom)

    assert meta["rescale_applied"] is True
    np.testing.assert_allclose(rescaled, plain * 2.0 - 1024.0, atol=1e-3)


def test_missing_transfer_syntax_recovers(headerless_dicom: Path) -> None:
    """A stripped file-meta header should cost a warning, not a dead run."""
    arr, meta = read_dicom(headerless_dicom)
    assert arr.ndim == 2
    assert arr.dtype == np.float32
    assert meta["format"] == "dicom"


def test_truncated_dicom_raises(truncated_dicom: Path) -> None:
    """Unreadable pixel data must raise -- never return a half-image."""
    with pytest.raises(RadiographReadError):
        read_dicom(truncated_dicom)


def test_extensionless_dicom_is_sniffed(tmp_path: Path, mono2_dicom: Path) -> None:
    """DICOM files routinely arrive with no extension; fall back to magic bytes."""
    target = tmp_path / "IMG000123"
    target.write_bytes(mono2_dicom.read_bytes())

    arr, meta = read_radiograph(target)
    assert meta["format"] == "dicom"
    assert arr.ndim == 2


# ---------------------------------------------------------------------------
# Standard image formats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture_name", ["jpeg_image", "png_image"])
def test_standard_formats(request: pytest.FixtureRequest, fixture_name: str) -> None:
    path = request.getfixturevalue(fixture_name)
    arr, meta = read_standard_image(path)

    assert arr.ndim == 2, "must be single-channel; BTXRD is 8-bit grayscale"
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    assert meta["photometric_interpretation"] == "MONOCHROME2"


def test_rgb_is_flattened_to_grayscale(tmp_path: Path) -> None:
    """A scanned-film RGB export must not change the channel count downstream."""
    from PIL import Image

    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8), mode="RGB").save(path)

    arr, meta = read_standard_image(path)
    assert arr.shape == (32, 48)
    assert meta["pil_mode"] == "RGB"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_corrupt_image_raises(corrupt_file: Path) -> None:
    with pytest.raises(RadiographReadError):
        read_radiograph(corrupt_file)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RadiographReadError, match="does not exist"):
        read_radiograph(tmp_path / "nope.jpeg")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    with pytest.raises(UnsupportedFormatError):
        read_radiograph(path)


def test_dispatch_is_format_blind(jpeg_image: Path, mono2_dicom: Path) -> None:
    """Both formats must yield the same contract: 2-D, float32, finite."""
    for path in (jpeg_image, mono2_dicom):
        arr, meta = read_radiograph(path)
        assert arr.ndim == 2
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert "filename_or_obj" in meta
