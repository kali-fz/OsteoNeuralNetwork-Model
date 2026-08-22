"""Out-of-distribution gating for uploads: reject non-radiographs, defer on doubt.

Why this module exists
----------------------
The classifier is a closed-set softmax over {normal, benign, malignant}: its
three probabilities sum to 1.0 for *any* input, including a photograph of a
hotdog. Without a gate, out-of-domain uploads are silently forced into the
nearest class -- typically a ~50% "benign" call that looks like a diagnosis.

Two independent defenses are implemented here, both torch-free (numpy + PIL +
pydicom only) so they can be unit-tested without the GPU stack:

1. **Pre-inference input validation** (:func:`validate_payload`,
   :func:`validate_image`). Cheap image-statistics heuristics that ask "does
   this even look like a musculoskeletal radiograph?" before the network runs:
   colorfulness (X-rays are single-channel), dynamic range (blank / constant
   frames), histogram entropy (photographic texture), and edge density
   (salt-and-pepper noise). Every check is named and reported so a rejection
   is explainable, not a black box inside another black box.

2. **Post-inference uncertainty gating** (:func:`should_defer`). When the
   softmax itself is ambivalent -- low max probability or high predictive
   entropy -- a lesion call is downgraded to "Non-Diagnostic / Inconclusive"
   rather than shown as a finding. The gate only ever *withdraws* a positive
   verdict; it never upgrades one, and it never touches the calibrated
   threshold, which remains fitted on validation data only.

These are heuristics, not a learned OOD detector. They catch the flagrant
failure modes (photos, noise, blanks); a grayscale photograph with X-ray-like
statistics can still pass stage 1, which is why stage 2 exists and why the
medical disclaimer still applies to everything downstream.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

REJECTION_MESSAGE = (
    "Invalid Image: Uploaded file is not recognized as a valid musculoskeletal radiograph."
)

INCONCLUSIVE_LABEL = "Non-Diagnostic / Inconclusive"

#: Suffixes duplicated from io_radiograph to keep this module import-light.
_DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}

# -- Heuristic thresholds, named so reports and docs can reference them ------
MIN_DIMENSION = 48            # px; anything smaller carries no diagnostic signal
MAX_PIXELS = 100_000_000      # decompression-bomb ceiling, matches storage.py
MIN_GRAY_LEVELS = 8           # fewer distinct levels = blank or synthetic shape
MAX_CHANNEL_SPREAD = 0.08     # mean |R-G|,|G-B| above this = a color photograph
MAX_HISTOGRAM_ENTROPY = 7.5   # bits over 256 bins; photos/noise sit near 8.0
MAX_EDGE_DENSITY = 0.45       # fraction of strong-gradient pixels; noise ~0.9

# -- Uncertainty gate defaults (the app passes these to predict) -------------
DEFAULT_CONFIDENCE_FLOOR = 0.65   # max softmax prob below this = defer
DEFAULT_ENTROPY_GATE = 0.90       # normalized predictive entropy at/above = defer


class NonRadiographError(ValueError):
    """The uploaded image failed pre-inference radiograph validation."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(REJECTION_MESSAGE)
        self.report = report


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    value: float
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    is_radiograph: bool
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[ValidationCheck]:
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict:
        return {
            "is_radiograph": self.is_radiograph,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "value": round(check.value, 4),
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


def _failed(name: str, detail: str, value: float = 0.0) -> ValidationReport:
    return ValidationReport(False, [ValidationCheck(name, False, value, detail)])


# ---------------------------------------------------------------------------
# Stage 1 -- image statistics
# ---------------------------------------------------------------------------
def validate_image(gray: np.ndarray, rgb: np.ndarray | None = None) -> ValidationReport:
    """Run every radiograph heuristic against a decoded image.

    Args:
        gray: 2-D luminance plane, any numeric dtype and range.
        rgb: optional ``(H, W, 3)`` array from the *original* encoding, used
            for the colorfulness check. Grayscale sources should pass ``None``
            -- a channel check against replicated luminance proves nothing.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or gray.size == 0:
        return _failed("decodable", f"expected a 2-D image, got shape {gray.shape}")
    if not np.isfinite(gray).all():
        gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)

    checks: list[ValidationCheck] = []
    height, width = gray.shape

    smallest = float(min(height, width))
    checks.append(
        ValidationCheck(
            "dimensions",
            smallest >= MIN_DIMENSION and height * width <= MAX_PIXELS,
            smallest,
            f"{width}x{height} px (min side must be >= {MIN_DIMENSION}, "
            f"total <= {MAX_PIXELS} px)",
        )
    )

    lo, hi = float(gray.min()), float(gray.max())
    span = hi - lo
    if span > 0:
        scaled = (gray - lo) / span
        levels = int(np.unique((scaled * 255).astype(np.uint8)).size)
    else:
        scaled = np.zeros_like(gray)
        levels = 1
    checks.append(
        ValidationCheck(
            "dynamic_range",
            levels >= MIN_GRAY_LEVELS,
            float(levels),
            f"{levels} distinct gray levels (a radiograph needs >= {MIN_GRAY_LEVELS}; "
            "blank or flat frames fail here)",
        )
    )

    if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] == 3:
        channels = np.asarray(rgb, dtype=np.float64) / 255.0
        spread = float(
            np.mean(np.abs(channels[..., 0] - channels[..., 1]))
            + np.mean(np.abs(channels[..., 1] - channels[..., 2]))
        ) / 2.0
        checks.append(
            ValidationCheck(
                "grayscale",
                spread <= MAX_CHANNEL_SPREAD,
                spread,
                f"mean channel disagreement {spread:.3f} "
                f"(radiographs are single-channel; > {MAX_CHANNEL_SPREAD} "
                "indicates a color photograph)",
            )
        )
    else:
        checks.append(
            ValidationCheck("grayscale", True, 0.0, "single-channel source")
        )

    histogram, _ = np.histogram(scaled, bins=256, range=(0.0, 1.0))
    frequencies = histogram[histogram > 0] / max(float(gray.size), 1.0)
    entropy_bits = float(-(frequencies * np.log2(frequencies)).sum()) if frequencies.size else 0.0
    checks.append(
        ValidationCheck(
            "histogram_entropy",
            entropy_bits <= MAX_HISTOGRAM_ENTROPY,
            entropy_bits,
            f"intensity entropy {entropy_bits:.2f} bits "
            f"(photographic texture and noise approach 8.0; limit {MAX_HISTOGRAM_ENTROPY})",
        )
    )

    if smallest >= 2:
        gy, gx = np.gradient(scaled)
        magnitude = np.hypot(gx, gy)
        edge_density = float(np.mean(magnitude > 0.10))
    else:
        edge_density = 1.0
    checks.append(
        ValidationCheck(
            "edge_density",
            edge_density <= MAX_EDGE_DENSITY,
            edge_density,
            f"strong-gradient fraction {edge_density:.3f} "
            f"(salt-and-pepper noise approaches 1.0; limit {MAX_EDGE_DENSITY})",
        )
    )

    report = ValidationReport(all(check.passed for check in checks), checks)
    if not report.is_radiograph:
        logger.info(
            "input rejected as non-radiograph: %s",
            ", ".join(f"{c.name}={c.value:.3f}" for c in report.failures),
        )
    return report


def _looks_like_dicom(payload: bytes, suffix: str) -> bool:
    if suffix in _DICOM_SUFFIXES:
        return True
    return len(payload) >= 132 and payload[128:132] == b"DICM"


def _decode_dicom_gray(payload: bytes) -> np.ndarray:
    import pydicom

    dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
    if not hasattr(dataset, "file_meta") or "TransferSyntaxUID" not in dataset.file_meta:
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian

        if not hasattr(dataset, "file_meta"):
            dataset.file_meta = FileMetaDataset()
        dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    array = np.asarray(dataset.pixel_array, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    elif array.ndim == 3:
        array = array[0]
    return array


def validate_payload(payload: bytes, filename: str | None = None) -> ValidationReport:
    """Decode raw upload bytes and run :func:`validate_image` on the result.

    DICOM payloads skip the colorfulness check (the pixel data is already
    single-channel by modality); standard images keep their original channels
    so a colorful photograph saved as ``.png`` is still caught.

    Statistics are computed on stored pixel values without the VOI window or
    MONOCHROME1 inversion: entropy, gray-level count, and edge density are
    invariant to those presentation transforms, and skipping them keeps this
    validator independent of the decoding pipeline it guards.
    """
    if not payload:
        return _failed("decodable", "uploaded file is empty")

    suffix = Path(filename or "").suffix.lower()

    if _looks_like_dicom(payload, suffix):
        try:
            gray = _decode_dicom_gray(payload)
        except Exception as exc:  # noqa: BLE001 - any decode failure is a rejection
            return _failed("decodable", f"DICOM pixel data unreadable ({exc})")
        return validate_image(gray, rgb=None)

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.width * image.height > MAX_PIXELS:
                return _failed(
                    "dimensions",
                    f"{image.width}x{image.height} px exceeds the {MAX_PIXELS}-pixel limit",
                    float(image.width * image.height),
                )
            source_mode = image.mode
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
            gray = np.asarray(image.convert("L"), dtype=np.float64)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return _failed("decodable", f"could not decode image ({exc})")

    # A source that was stored single-channel carries no color evidence.
    color_source = source_mode not in {"L", "I", "I;16", "F", "1"}
    return validate_image(gray, rgb=rgb if color_source else None)


def ensure_radiograph(payload: bytes, filename: str | None = None) -> ValidationReport:
    """Validate and raise :class:`NonRadiographError` on failure."""
    report = validate_payload(payload, filename)
    if not report.is_radiograph:
        raise NonRadiographError(report)
    return report


# ---------------------------------------------------------------------------
# Stage 2 -- softmax uncertainty
# ---------------------------------------------------------------------------
def predictive_entropy(probabilities: np.ndarray) -> float:
    """Normalized Shannon entropy of a probability vector, in [0, 1].

    0.0 is a one-hot (fully confident) distribution; 1.0 is uniform across all
    classes -- the closed-set softmax's signature on an input it has never
    seen. Normalizing by ``log(K)`` keeps the gate threshold meaningful if the
    class count ever changes.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    if p.size < 2:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def should_defer(
    probabilities: np.ndarray,
    *,
    uncertainty_floor: float | None = None,
    entropy_gate: float | None = None,
) -> tuple[bool, float, float]:
    """Decide whether a verdict is too uncertain to present as a finding.

    Returns ``(defer, max_probability, normalized_entropy)``. ``defer`` is True
    when the max class probability falls below ``uncertainty_floor`` or the
    normalized predictive entropy reaches ``entropy_gate``. Passing ``None``
    disables the corresponding criterion, so the default is a no-op -- callers
    opt in, and the trained pipeline's behavior is unchanged unless asked.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    max_probability = float(p.max()) if p.size else 0.0
    entropy = predictive_entropy(p)

    defer = False
    if uncertainty_floor is not None and max_probability < float(uncertainty_floor):
        defer = True
    if entropy_gate is not None and entropy >= float(entropy_gate):
        defer = True
    return defer, max_probability, entropy


__all__ = [
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_ENTROPY_GATE",
    "INCONCLUSIVE_LABEL",
    "REJECTION_MESSAGE",
    "NonRadiographError",
    "ValidationCheck",
    "ValidationReport",
    "ensure_radiograph",
    "predictive_entropy",
    "should_defer",
    "validate_image",
    "validate_payload",
]
