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


def detect_panels(
    gray: np.ndarray,
    *,
    darkness: float = 0.05,
    dark_rows: float = 0.97,
    min_panel: float = 0.18,
    min_gap: float = 0.01,
    balance: float = 1.8,
) -> dict[str, Any]:
    """Detect a multi-panel composite -- two views pasted into one image.

    WHY THIS MATTERS MORE THAN IT LOOKS
    -----------------------------------
    BTXRD is single-view, single-panel. A visitor who uploads an AP and a lateral
    side by side hands the model something it has never seen, and the damage is
    not merely novelty: ``build_transforms`` resizes the LONGEST side, so a
    double-width composite is scaled to roughly half the linear size a single film
    would get. A lesion occupying 10% of one frame becomes about 5% of the
    composite -- and the black gutter between the panels lands mid-frame, which is
    exactly where the reported heatmaps peaked.

    WHAT ACTUALLY DISCRIMINATES
    ---------------------------
    Not "is there a dark column". Measured over 300 BTXRD films, 17% have a fully
    dark interior column and 15-22% have one that is dark down its whole height --
    because a single bone on a black background produces exactly that. Those are
    background, not gutters.

    A gutter is a dark stripe *bounded by real content on both sides*. So a run of
    dark columns counts only when it

      * never touches the frame edge (otherwise it is background),
      * leaves at least ``min_panel`` of the width on each side,
      * has mostly-bright columns on both sides, and
      * splits the frame into two panels of comparable width (``balance``),
        because an AP and a lateral of the same limb are roughly equal.

    Measured with these defaults over 300 BTXRD films plus the four real uploads:
    both composites detected, neither single-panel upload flagged, and 1.3% of
    BTXRD flagged -- against 17.3% for a naive column-mean test.

    ``dark_rows`` is tuned rather than guessed. The two real composites measure
    1.000 and 0.991, so 0.97 keeps both; raising it to 0.98 loses one, and
    lowering it to 0.90 admits enough anatomical gaps to nearly double the false
    positives.

    ADVISORY, NEVER A REJECTION
    ---------------------------
    A composite is still a radiograph and the model should still answer. This only
    supports telling the visitor that one view at a time reads more reliably, so a
    false positive costs a sentence of advice rather than a refused scan. That is
    why 2.3% is acceptable here and would not be for the OOD gate.
    """
    empty = {"is_composite": False, "n_panels": 1, "split_at": [], "gutter_darkness": None}
    if gray.ndim != 2 or gray.size == 0:
        return empty

    finite = np.isfinite(gray)
    if not finite.any():
        return empty

    # Percentile scaling, matching the training transform, so `darkness` means
    # the same thing on a 12-bit DICOM and an 8-bit JPEG.
    lo, hi = np.percentile(gray[finite], (1.0, 99.0))
    if hi <= lo:
        return empty
    scaled = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)

    width = scaled.shape[1]
    # Per column: the share of its ROWS that are dark. A pasted gutter is dark
    # top to bottom; an anatomical gap is interrupted by bone somewhere.
    column_is_dark = (scaled < darkness).mean(axis=0) >= dark_rows

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(width + 1):
        dark = bool(column_is_dark[index]) if index < width else False
        if dark and start is None:
            start = index
        elif not dark and start is not None:
            runs.append((start, index))
            start = None

    split_at: list[int] = []
    for left, right in runs:
        if left == 0 or right == width:
            continue  # touches the frame edge: background, not a gutter
        if (right - left) < max(int(min_gap * width), 2):
            continue  # a thin dark line, not a gutter
        if (left / width) < min_panel or ((width - right) / width) < min_panel:
            continue  # one "panel" is a sliver
        if float((~column_is_dark[:left]).mean()) < 0.5:
            continue  # nothing solid to the left
        if float((~column_is_dark[right:]).mean()) < 0.5:
            continue  # nothing solid to the right
        if not (1.0 / balance) <= (left / max(width - right, 1)) <= balance:
            continue  # lopsided: two views of one limb are roughly equal
        split_at.append(int((left + right) // 2))

    return {
        "is_composite": bool(split_at),
        "n_panels": len(split_at) + 1,
        "split_at": split_at,
        "gutter_darkness": float(scaled.mean(axis=0).min()),
    }


class LoadLesionMaskd(MapTransform):
    """Rasterise a record's lesion polygon into a mask aligned with its image.

    RASTERISED HERE, NOT IN THE RECORD
    ----------------------------------
    It would be simpler to attach a mask array to each record in
    ``dataset.build_records``. It would also be a ~20 GB mistake: records live in
    ``CacheDataset.data`` for the whole run at *original* resolution, and BTXRD
    films run to 2397x3213, so one mask per record is ~30 MB x 2675 training
    images. Rasterising inside the chain means the cache holds the mask only
    after it has been resized to the model input, which is 256 KB.

    The mask emerges as a 2-D ``(H, W)`` float32 array in {0.0, 1.0}, matching
    :class:`LoadRadiographd`'s output shape so the same
    ``EnsureChannelFirstd(channel_dim="no_channel")`` handles both.

    A record with no annotation file -- every normal film -- yields an all-zero
    mask. That is not a placeholder for missing data. It is the supervision that
    matters most here: "there is no lesion anywhere on this healthy joint" is
    exactly the lesson the false positives on complex anatomy need, and it is
    available for all 1879 normal images at no labelling cost.
    """

    def __init__(
        self,
        keys: str | list[str],
        annotation_key: str = "annotation",
        image_key: str = "image",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.annotation_key = annotation_key
        self.image_key = image_key

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        import cv2

        from .explainability import load_polygons

        d = dict(data)
        image = d[self.image_key]
        # Runs before EnsureChannelFirstd, so the image is still (H, W).
        height, width = int(image.shape[-2]), int(image.shape[-1])

        # `self.keys`, not `self.key_iterator(d)`: this transform CREATES its
        # key rather than modifying one, and key_iterator raises on a key that is
        # not already in the dict.
        for key in self.keys:
            mask = np.zeros((height, width), dtype=np.float32)
            path = d.get(self.annotation_key)
            if path and Path(path).is_file():
                annotation = load_polygons(path)
                # The annotation records the dimensions it was drawn against.
                # If they disagree with the decoded image the polygon would land
                # somewhere arbitrary, so scale rather than assume. (Verified
                # equal across BTXRD, but a community-exported annotation is
                # written against a 256px preprocessed image, not the original.)
                src_h = int(annotation["height"]) or height
                src_w = int(annotation["width"]) or width
                scale_y, scale_x = height / src_h, width / src_w
                contours = []
                for polygon in annotation["polygons"]:
                    scaled = polygon.copy()
                    scaled[:, 0] *= scale_x
                    scaled[:, 1] *= scale_y
                    contours.append(np.round(scaled).astype(np.int32))
                if contours:
                    cv2.fillPoly(mask, contours, 1.0)
            d[key] = mask
        return d


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
