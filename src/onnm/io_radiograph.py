"""Radiograph I/O: a MONAI transform that reads DICOM and standard images alike.

Why this exists instead of ``monai.transforms.LoadImaged``
----------------------------------------------------------
MONAI's loader hands back raw stored pixel values. For radiographs that is not
enough, because two DICOM conventions change what the pixels *mean*:

1. **VOI LUT / Window Center-Width.** Stored values are frequently 12- or
   16-bit with a windowing transform carried in the header. Ignoring it yields
   a flat, low-contrast image where subtle lytic lesions vanish into the
   background -- precisely the lesions this project exists to catch.

2. **PhotometricInterpretation = MONOCHROME1.** In MONOCHROME1, *lower* stored
   values are *brighter* -- the opposite of MONOCHROME2 and of every JPEG. Read
   naively, the image is a photographic negative. The failure is silent: the
   array is valid, training converges, metrics look plausible, and the model has
   learned an inverted world. A mixed-convention dataset teaches it nothing at
   all.

So the modality LUT, VOI LUT and photometric inversion are applied here, in the
order the DICOM standard specifies, and every fallback is logged rather than
swallowed. ``tests/test_io_radiograph.py`` asserts a synthetic MONOCHROME1 file
round-trips to the same array as its MONOCHROME2 twin.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from monai.transforms import MapTransform

from .utils import get_logger

logger = get_logger(__name__)

DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
STANDARD_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RadiographReadError(RuntimeError):
    """A radiograph could not be decoded."""


class UnsupportedFormatError(RadiographReadError):
    """The file extension is not a format this project knows how to read."""


# ---------------------------------------------------------------------------
# pydicom moved apply_voi_lut in 3.0. Support both so the project is not pinned
# to one major version of a dependency it barely uses.
# ---------------------------------------------------------------------------
def _get_apply_voi_lut():
    try:
        from pydicom.pixels import apply_voi_lut  # pydicom >= 3.0
        return apply_voi_lut
    except ImportError:
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut  # pydicom 2.x
            return apply_voi_lut
        except ImportError:
            return None


def _to_single_channel(arr: np.ndarray, path: Path) -> np.ndarray:
    """Reduce whatever DICOM handed us to a single 2-D grayscale plane."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Either an RGB radiograph (scanned film, secondary capture) or a
        # multi-frame object. Distinguish by the trailing axis.
        if arr.shape[-1] == 3:
            return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        logger.warning("%s: %d-frame image, using frame 0", path.name, arr.shape[0])
        return arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 3:
        logger.warning("%s: multi-frame RGB, using frame 0", path.name)
        rgb = arr[0]
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise RadiographReadError(f"{path}: unexpected pixel array shape {arr.shape}")


def read_dicom(path: str | Path, apply_lut: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a DICOM radiograph into a float32 2-D array.

    Applies, in the order mandated by the standard: modality LUT (Rescale
    Slope/Intercept), VOI LUT (or Window Center/Width), then MONOCHROME1
    inversion. Every stage degrades gracefully and logs when it does, because a
    header that is merely incomplete should cost one warning, not a training run.
    """
    import pydicom

    path = Path(path)
    try:
        # force=True lets us open files with a missing or damaged preamble,
        # which is common in de-identified research exports.
        ds = pydicom.dcmread(str(path), force=True)
    except Exception as exc:  # noqa: BLE001 - any pydicom failure is a read failure
        raise RadiographReadError(f"{path}: could not parse DICOM ({exc})") from exc

    # Without a TransferSyntaxUID pydicom refuses to decode. force=True skips
    # the check that would normally populate it, so supply the standard default.
    if not hasattr(ds, "file_meta") or "TransferSyntaxUID" not in ds.file_meta:
        logger.warning("%s: missing TransferSyntaxUID, assuming ImplicitVRLittleEndian", path.name)
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian

        if not hasattr(ds, "file_meta"):
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    try:
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001
        raise RadiographReadError(f"{path}: pixel data unreadable ({exc})") from exc

    arr = _to_single_channel(arr, path).astype(np.float32)

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()
    meta: dict[str, Any] = {
        "filename_or_obj": str(path),
        "format": "dicom",
        "photometric_interpretation": photometric,
        "inverted": False,
        "voi_lut_applied": False,
        "rescale_applied": False,
        "original_shape": tuple(int(v) for v in arr.shape),
    }

    # -- 1. Modality LUT ----------------------------------------------------
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    if slope is not None or intercept is not None:
        try:
            arr = arr * float(slope if slope is not None else 1.0) + float(
                intercept if intercept is not None else 0.0
            )
            meta["rescale_applied"] = True
        except (TypeError, ValueError):
            logger.warning("%s: unusable RescaleSlope/Intercept, skipping modality LUT", path.name)

    # -- 2. VOI LUT / windowing --------------------------------------------
    has_voi = "VOILUTSequence" in ds or "WindowCenter" in ds
    if apply_lut and has_voi:
        apply_voi_lut = _get_apply_voi_lut()
        if apply_voi_lut is None:
            logger.warning("%s: pydicom has no apply_voi_lut, using raw values", path.name)
        else:
            try:
                windowed = apply_voi_lut(arr, ds)
                arr = np.asarray(windowed, dtype=np.float32)
                meta["voi_lut_applied"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: VOI LUT failed (%s), using raw values", path.name, exc)

    # -- 3. Photometric inversion ------------------------------------------
    # apply_voi_lut deliberately does NOT do this; it is a presentation-stage
    # concern. If we skip it, MONOCHROME1 films arrive as negatives.
    if photometric.upper() == "MONOCHROME1":
        arr = float(np.nanmax(arr)) - arr
        meta["inverted"] = True

    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        logger.warning("%s: %d non-finite pixels, replacing with 0", path.name, n_bad)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return np.ascontiguousarray(arr, dtype=np.float32), meta


def read_standard_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a JPEG/PNG/TIFF radiograph into a float32 2-D array.

    BTXRD ships 8-bit grayscale JPEG, but images are forced to mode ``L``
    anyway so a stray RGB or palette file cannot change the channel count
    downstream.
    """
    from PIL import Image, UnidentifiedImageError

    path = Path(path)
    try:
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            arr = np.asarray(img.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RadiographReadError(f"{path}: could not decode image ({exc})") from exc

    if arr.size == 0:
        raise RadiographReadError(f"{path}: decoded to an empty array")

    meta: dict[str, Any] = {
        "filename_or_obj": str(path),
        "format": path.suffix.lower().lstrip("."),
        "photometric_interpretation": "MONOCHROME2",
        "inverted": False,
        "voi_lut_applied": False,
        "rescale_applied": False,
        "pil_mode": mode,
        "original_shape": tuple(int(v) for v in arr.shape),
    }
    return np.ascontiguousarray(arr, dtype=np.float32), meta


def read_radiograph(path: str | Path, apply_lut: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Read any supported radiograph, dispatching on file extension."""
    path = Path(path)
    if not path.is_file():
        raise RadiographReadError(f"{path}: file does not exist")

    suffix = path.suffix.lower()
    if suffix in DICOM_SUFFIXES:
        return read_dicom(path, apply_lut=apply_lut)
    if suffix in STANDARD_SUFFIXES:
        return read_standard_image(path)
    if suffix == "":
        # DICOM files are routinely extensionless. Sniff the magic bytes.
        try:
            with path.open("rb") as fh:
                fh.seek(128)
                if fh.read(4) == b"DICM":
                    return read_dicom(path, apply_lut=apply_lut)
        except OSError as exc:
            raise RadiographReadError(f"{path}: unreadable ({exc})") from exc

    raise UnsupportedFormatError(
        f"{path}: unsupported extension {suffix!r}; "
        f"expected one of {sorted(DICOM_SUFFIXES | STANDARD_SUFFIXES)}"
    )


class LoadRadiographd(MapTransform):
    """MONAI dictionary transform wrapping :func:`read_radiograph`.

    Emits a 2-D ``(H, W)`` float32 array, so the next transform in the chain
    must be ``EnsureChannelFirstd(..., channel_dim="no_channel")``. Being
    explicit here beats letting MONAI infer a channel axis, which guesses wrong
    on square images.

    Read failures raise :class:`RadiographReadError`. Resilience is handled one
    level up by ``onnm.dataset.RobustDataset``, which keeps this transform pure
    and makes the substitution policy visible where it belongs.
    """

    def __init__(
        self,
        keys: str | list[str],
        meta_key_postfix: str = "meta_dict",
        apply_lut: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_key_postfix = meta_key_postfix
        self.apply_lut = apply_lut

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            arr, meta = read_radiograph(d[key], apply_lut=self.apply_lut)
            d[key] = arr
            if self.meta_key_postfix:
                d[f"{key}_{self.meta_key_postfix}"] = meta
        return d
