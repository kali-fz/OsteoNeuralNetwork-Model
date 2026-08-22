"""Shared fixtures: synthetic radiographs in every format the loader must handle.

Everything here is generated, not sampled from BTXRD, so the suite runs on a
clean clone with no dataset present -- and so the DICOM edge cases (MONOCHROME1,
VOI LUT, a truncated file, a missing TransferSyntaxUID) can be constructed
exactly rather than hoped for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Deliberately non-square: a square phantom would hide any aspect-ratio bug in
# the resize/pad stage, which is exactly what we want these tests to catch.
PHANTOM_H, PHANTOM_W = 96, 64


@pytest.fixture(scope="session")
def phantom() -> np.ndarray:
    """A deterministic 16-bit phantom with min exactly 0.

    Min-0 matters: the MONOCHROME1 round-trip is only an exact identity when the
    array's floor is zero, so pinning it keeps the inversion test unambiguous.
    """
    yy, xx = np.mgrid[0:PHANTOM_H, 0:PHANTOM_W]
    arr = (yy * 37 + xx * 11).astype(np.float64)
    arr[20:40, 15:35] += 900          # a bright "lesion"
    arr[60:80, 25:55] -= 300          # a dark "lytic" region
    arr -= arr.min()                  # floor at exactly 0
    arr = arr / arr.max() * 3000.0    # a plausible 12-bit-ish range
    return arr.astype(np.uint16)


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    photometric: str = "MONOCHROME2",
    window: tuple[float, float] | None = None,
    rescale: tuple[float, float] | None = None,
    drop_transfer_syntax: bool = False,
) -> Path:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST^001"
    ds.Modality = "DX"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = float(window[0]), float(window[1])
    if rescale is not None:
        ds.RescaleSlope, ds.RescaleIntercept = float(rescale[0]), float(rescale[1])
    ds.PixelData = pixels.astype(np.uint16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    # pydicom 3.x renamed the strict-write flag; support both major versions.
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:
        ds.save_as(str(path), write_like_original=False)

    if drop_transfer_syntax:
        # Strip the 128-byte preamble and "DICM" magic so the file parses only
        # under force=True with no file_meta -- a real pattern in sloppily
        # de-identified research exports.
        raw = path.read_bytes()
        body = raw[132:]
        marker = body.find(b"\x08\x00")
        path.write_bytes(body[marker:] if marker > 0 else body)

    return path


@pytest.fixture
def mono2_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    """A conventional radiograph: higher stored value == brighter."""
    return _write_dicom(tmp_path / "mono2.dcm", phantom, "MONOCHROME2")


@pytest.fixture
def mono1_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    """The same image stored under the inverted MONOCHROME1 convention.

    Reading this without inverting yields a photographic negative -- valid
    floats, plausible training curves, and a model that has learned an inverted
    world. This fixture is the reason ``read_dicom`` handles photometric
    interpretation at all.
    """
    inverted = (int(phantom.max()) - phantom.astype(np.int32)).astype(np.uint16)
    return _write_dicom(tmp_path / "mono1.dcm", inverted, "MONOCHROME1")


@pytest.fixture
def windowed_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    return _write_dicom(tmp_path / "windowed.dcm", phantom, "MONOCHROME2", window=(1500.0, 2000.0))


@pytest.fixture
def rescaled_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    return _write_dicom(tmp_path / "rescaled.dcm", phantom, "MONOCHROME2", rescale=(2.0, -1024.0))


@pytest.fixture
def headerless_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    return _write_dicom(
        tmp_path / "headerless.dcm", phantom, "MONOCHROME2", drop_transfer_syntax=True
    )


@pytest.fixture
def truncated_dicom(tmp_path: Path, phantom: np.ndarray) -> Path:
    """Header intact, pixel data cut in half -- must fail loudly, not silently."""
    full = _write_dicom(tmp_path / "_full.dcm", phantom, "MONOCHROME2")
    raw = full.read_bytes()
    path = tmp_path / "truncated.dcm"
    path.write_bytes(raw[: len(raw) // 2])
    return path


@pytest.fixture
def jpeg_image(tmp_path: Path, phantom: np.ndarray) -> Path:
    from PIL import Image

    eight_bit = (phantom.astype(np.float64) / phantom.max() * 255).astype(np.uint8)
    path = tmp_path / "radiograph.jpeg"
    Image.fromarray(eight_bit, mode="L").save(path, quality=100)
    return path


@pytest.fixture
def png_image(tmp_path: Path, phantom: np.ndarray) -> Path:
    from PIL import Image

    eight_bit = (phantom.astype(np.float64) / phantom.max() * 255).astype(np.uint8)
    path = tmp_path / "radiograph.png"
    Image.fromarray(eight_bit, mode="L").save(path)
    return path


@pytest.fixture
def wide_jpeg(tmp_path: Path) -> Path:
    """A 4:1 image with a centred square, for asserting aspect ratio survives.

    If the pipeline squashed this to 256x256 instead of resizing-and-padding,
    the square would come out as a 4:1 rectangle.
    """
    from PIL import Image

    arr = np.zeros((64, 256), dtype=np.uint8)
    arr[16:48, 112:144] = 255  # a 32x32 square, centred
    path = tmp_path / "wide.png"
    Image.fromarray(arr, mode="L").save(path)
    return path


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.jpeg"
    path.write_bytes(b"this is definitely not a JPEG")
    return path


@pytest.fixture
def cfg():
    """Project config with caching off, so tests never build a 3 GB cache."""
    from onnm.config import load_config

    config = load_config("configs/base.yaml")
    config._data["loader"]["cache_rate"] = 0.0
    config._data["loader"]["num_workers"] = 0
    config._data["loader"]["batch_size"] = 4
    return config


@pytest.fixture
def records(jpeg_image: Path, png_image: Path) -> list[dict]:
    """A minimal record list covering all three classes."""
    return [
        {"image": str(jpeg_image), "label": 0, "image_id": "a", "patient_id": "p1"},
        {"image": str(png_image), "label": 1, "image_id": "b", "patient_id": "p2"},
        {"image": str(jpeg_image), "label": 2, "image_id": "c", "patient_id": "p3"},
        {"image": str(png_image), "label": 0, "image_id": "d", "patient_id": "p4"},
    ]
