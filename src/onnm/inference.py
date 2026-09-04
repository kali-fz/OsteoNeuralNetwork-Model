"""Single-image inference for the local Streamlit app.

This is the thin layer between an uploaded file and a verdict. Almost everything
it does is already implemented and tested elsewhere in the package -- the point
of this module is to guarantee the *app* runs the identical pipeline the model
was trained under, because a UI that quietly preprocesses differently is worse
than no UI at all.

Three properties are load-bearing:

1. **The preprocessing comes from the checkpoint, not from disk.** ``best.pt``
   embeds the exact config it was trained with. Reading ``configs/base.yaml``
   instead would let an edit to the intensity percentiles or the image size
   silently desynchronise inference from training months after the fact. The
   YAML is only a fallback for checkpoints old enough to lack the block.

2. **The backbone is built with ``pretrained=False``.** Every weight is
   overwritten by the checkpoint a line later, so fetching ImageNet weights
   would be pure waste -- and, more importantly, a network call. This app is
   specified to run offline; it now does, on a cold torch cache.

3. **Grad-CAM runs outside ``no_grad``.** MONAI's ``GradCAM`` backpropagates to
   the hooked layer, so wrapping it the way the forward pass is wrapped would
   break it. The forward pass used for probabilities is separate and *is* under
   ``no_grad``.

Three-class model, two-class display
------------------------------------
The trained network is a 3-way classifier (normal / benign / malignant). The app
presents "Normal" vs "Potential Bone Lesion", which is ``P(benign) +
P(malignant)`` -- computed here as ``1 - P(normal)`` so it stays correct if the
class list ever changes. The full 3-way breakdown rides along in
:class:`InferenceResult` and the app shows it, because collapsing benign and
malignant into one bucket discards the distinction that actually changes
management.
"""

from __future__ import annotations

import contextlib
import copy
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from .calibrate import CALIBRATION_FILENAME, Calibration
from .config import REPO_ROOT, Config, load_config
from .dataset import build_transforms
from .explainability import build_cam, compute_cam
from .io_radiograph import (
    DICOM_SUFFIXES,
    STANDARD_SUFFIXES,
    RadiographReadError,
    UnsupportedFormatError,
    read_radiograph,
)
from .model import build_model
from .ood import INCONCLUSIVE_LABEL, ensure_radiograph, should_defer
from .utils import get_device, get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES: tuple[str, ...] = tuple(sorted(DICOM_SUFFIXES | STANDARD_SUFFIXES))

#: Streamlit's uploader wants extensions without the leading dot.
UPLOAD_TYPES: list[str] = [s.lstrip(".") for s in SUPPORTED_SUFFIXES]

NORMAL_LABEL = "Normal"
LESION_LABEL = "Potential Bone Lesion"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Everything one prediction produced, in display-ready form.

    ``heatmap`` and ``preprocessed_image`` are deliberately returned raw rather
    than pre-blended: the app re-renders the overlay whenever the opacity slider
    moves, and re-running a GPU forward+backward pass for a cosmetic change
    would be absurd.
    """

    label: str                                  # NORMAL_LABEL or LESION_LABEL
    confidence: float                           # percentage backing `label`, 0-100
    lesion_probability: float                   # P(not normal), 0-1
    class_probabilities: dict[str, float]       # normal/benign/malignant -> 0-1
    top_class: str                              # argmax of the 3-way head
    threshold: float                            # decision threshold applied

    preprocessed_image: np.ndarray              # (S, S) float, model-space grayscale
    original_image: np.ndarray                  # (H, W) float, as decoded
    heatmap: np.ndarray | None = None           # (S, S) float in [0, 1]
    cam_class: str | None = None                # class the CAM was taken against
    #: "lesion_map" (supervised, class-agnostic) or "gradcam" (attribution), or
    #: None when no heatmap could be produced. The UI caption switches on this;
    #: without it a lesion map would be captioned as being "taken against" a
    #: class, which is a Grad-CAM concept that does not apply to it.
    heatmap_kind: str | None = None

    temperature: float = 1.0                    # 1.0 = uncalibrated logits
    calibrated: bool = False                    # a fitted calibration.json was used

    max_probability: float = 0.0                # max of the 3-way softmax, 0-1
    predictive_entropy: float = 0.0             # normalized softmax entropy, 0-1
    inconclusive: bool = False                  # uncertainty gate withdrew a lesion call

    source_meta: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    device: str = "cpu"

    def with_threshold(
        self,
        threshold: float,
        *,
        uncertainty_floor: float | None = None,
        entropy_gate: float | None = None,
    ) -> InferenceResult:
        """Re-derive the verdict at a different threshold, without the model.

        A threshold is a cut on an already-computed probability: it cannot
        change what the network produced. Re-running a forward and a Grad-CAM
        backward pass to move it costs roughly half a second on CPU and returns
        bit-identical probabilities and a bit-identical heatmap.

        So the decision is recomputed here instead, from values this result
        already carries. The uncertainty gate is re-evaluated rather than
        reused, because ``inconclusive`` is ``is_lesion and defer`` -- a False
        does not tell you which half was False, and ``should_defer`` is a pure
        function of the probabilities, so recomputing it is exact and free.

        The returned result is a copy; the original is left untouched, so a
        cached model output can be re-cut at many thresholds.
        """
        ordered = np.array(
            [self.class_probabilities[name] for name in self.class_probabilities],
            dtype=np.float64,
        )
        defer, max_probability, entropy = should_defer(
            ordered, uncertainty_floor=uncertainty_floor, entropy_gate=entropy_gate
        )

        is_lesion = self.lesion_probability >= threshold
        inconclusive = bool(is_lesion and defer)
        if inconclusive:
            label = INCONCLUSIVE_LABEL
        elif is_lesion:
            label = LESION_LABEL
        else:
            label = NORMAL_LABEL

        return replace(
            self,
            label=label,
            confidence=100.0
            * (self.lesion_probability if is_lesion else 1.0 - self.lesion_probability),
            threshold=float(threshold),
            max_probability=max_probability,
            predictive_entropy=entropy,
            inconclusive=inconclusive,
        )

    @property
    def is_lesion(self) -> bool:
        return self.label == LESION_LABEL

    @property
    def malignant_probability(self) -> float:
        return float(self.class_probabilities.get("malignant", 0.0))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary, for the app's download button."""
        return {
            "label": self.label,
            "confidence_pct": round(self.confidence, 2),
            "lesion_probability": round(self.lesion_probability, 6),
            "class_probabilities": {k: round(v, 6) for k, v in self.class_probabilities.items()},
            "top_class": self.top_class,
            "decision_threshold": self.threshold,
            "cam_class": self.cam_class,
            "heatmap_kind": self.heatmap_kind,
            "temperature": round(self.temperature, 4),
            "calibrated": self.calibrated,
            "max_probability": round(self.max_probability, 6),
            "predictive_entropy": round(self.predictive_entropy, 6),
            "inconclusive": self.inconclusive,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "device": self.device,
            "source": self.source_meta,
        }


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _suffix_for(filename: str | None, payload: bytes) -> str:
    """Pick the extension to write a temp file under.

    ``read_radiograph`` dispatches on suffix, so an upload whose name was lost
    or that arrived extensionless has to be sniffed. The DICOM preamble puts
    ``DICM`` at byte 128, which is the cheap and unambiguous check; anything
    else is left to PIL.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix in DICOM_SUFFIXES or suffix in STANDARD_SUFFIXES:
        return suffix
    if len(payload) >= 132 and payload[128:132] == b"DICM":
        return ".dcm"
    if suffix:
        raise UnsupportedFormatError(
            f"unsupported extension {suffix!r}; expected one of {list(SUPPORTED_SUFFIXES)}"
        )
    raise UnsupportedFormatError(
        "cannot determine file type: no usable extension and no DICOM preamble"
    )


def _coerce_to_bytes(data: bytes | bytearray | memoryview | BinaryIO | str | Path) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, (str, Path)):
        return Path(data).read_bytes()
    read = getattr(data, "read", None)
    if read is None:
        raise TypeError(f"cannot read image data from {type(data).__name__}")
    if hasattr(data, "seek"):
        # Streamlit hands the same handle back on every rerun, so rewind rather
        # than trusting the caller. Not every file-like object is seekable.
        with contextlib.suppress(OSError, ValueError):
            data.seek(0)
    return bytes(read())


class _MaterialisedUpload:
    """Context manager putting uploaded bytes on disk under a correct suffix.

    The whole reason for the round-trip is that ``io_radiograph`` is path-based,
    and that module is where the DICOM handling that actually matters lives --
    the modality LUT, the VOI window, and the MONOCHROME1 inversion that turns a
    film into its own negative when skipped. Duplicating that logic for a
    byte-stream variant would mean two implementations of the one thing in this
    project it is least safe to get subtly wrong. A ~1 ms temp-file write is the
    cheaper trade.

    ``delete=False`` plus an explicit unlink is required on Windows, where a
    still-open ``NamedTemporaryFile`` cannot be reopened by another handle.
    """

    def __init__(self, payload: bytes, suffix: str) -> None:
        self._payload = payload
        self._suffix = suffix
        self.path: Path | None = None

    def __enter__(self) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=self._suffix, delete=False)
        try:
            handle.write(self._payload)
        finally:
            handle.close()
        self.path = Path(handle.name)
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self.path is not None:
            try:
                os.unlink(self.path)
            except OSError:  # a virus scanner still holding the handle is not fatal
                logger.debug("could not remove temp file %s", self.path)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
#: Filenames a training run may have written its best checkpoint under. ``best.pt``
#: is what ``onnm.train`` writes; ``best_model.pt`` is accepted too so a run named
#: by the other common convention is still discovered by the app.
CHECKPOINT_NAMES: tuple[str, ...] = ("best.pt", "best_model.pt")


def find_checkpoints(reports_dir: str | Path = "reports") -> list[Path]:
    """List every best-checkpoint under the reports tree, newest first."""
    root = Path(reports_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not root.is_dir():
        return []

    found: dict[Path, float] = {}
    for name in CHECKPOINT_NAMES:
        for path in root.glob(f"*/{name}"):
            found[path] = path.stat().st_mtime
    return sorted(found, key=lambda p: found[p], reverse=True)


#: Marker file naming the pinned production run. Contains one line: the run
#: directory name under ``reports/`` (e.g. ``full-20260822-041653``). Written
#: by the operator, read by the app so a fresh experimental run never silently
#: becomes the default model.
PRODUCTION_MARKER = "PRODUCTION"

#: Run-directory name prefixes that are throwaway by convention and should be
#: hidden from end-user checkpoint pickers (they remain on disk and reachable
#: by explicit path for debugging).
THROWAWAY_RUN_PREFIXES: tuple[str, ...] = ("smoke-", "tmp-", "debug-")


def production_checkpoint(reports_dir: str | Path = "reports") -> Path | None:
    """Resolve the pinned production checkpoint, or ``None`` when unpinned.

    Reads ``reports/PRODUCTION`` (first non-empty, non-comment line = run
    directory name) and returns that run's checkpoint. Returns ``None`` when
    the marker is absent; raises ``FileNotFoundError`` when the marker names a
    run whose checkpoint does not exist — a stale pin should be loud, not a
    silent fallback to an arbitrary newer run.
    """
    root = Path(reports_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    marker = root / PRODUCTION_MARKER
    if not marker.is_file():
        return None

    run_name = ""
    for line in marker.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            run_name = line
            break
    if not run_name:
        return None

    for name in CHECKPOINT_NAMES:
        candidate = root / run_name / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"reports/{PRODUCTION_MARKER} pins run '{run_name}' but no checkpoint "
        f"({' or '.join(CHECKPOINT_NAMES)}) exists under {root / run_name}. "
        "Fix or delete the marker file."
    )


def is_throwaway_run(checkpoint: Path) -> bool:
    """True when a checkpoint belongs to a throwaway run (``smoke-*`` etc.)."""
    return checkpoint.parent.name.startswith(THROWAWAY_RUN_PREFIXES)


def default_checkpoint(reports_dir: str | Path = "reports") -> Path | None:
    """The checkpoint the app should load: the production pin, else newest.

    Preference order: the run pinned in ``reports/PRODUCTION``; otherwise the
    newest non-throwaway checkpoint; otherwise the newest checkpoint of any
    kind (so a repo containing only a smoke run still loads something).
    """
    pinned = production_checkpoint(reports_dir)
    if pinned is not None:
        return pinned
    found = find_checkpoints(reports_dir)
    kept = [p for p in found if not is_throwaway_run(p)]
    if kept:
        return kept[0]
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class RadiographClassifier:
    """A loaded model plus the exact transform chain it was trained under.

    Construct once and reuse -- in the app it lives behind ``st.cache_resource``.
    Building it per rerun would re-register Grad-CAM's forward and backward hooks
    on a fresh module every time and pay the ROCm kernel-compilation cost on
    every click.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        config_path: str | Path = "configs/base.yaml",
        device: torch.device | None = None,
        warmup: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint}. Train one with "
                "`python scripts/train.py --override configs/densenet121_3class.yaml`."
            )

        self.checkpoint_path = checkpoint
        self.device = device or get_device()
        # Grad-CAM mutates module state through hooks, so two Streamlit sessions
        # hitting one cached classifier concurrently must not interleave.
        self._lock = threading.Lock()

        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.cfg = self._config_from(state, config_path)

        self.class_names: list[str] = [str(c) for c in self.cfg.labels.classes]
        self.normal_index = (
            self.class_names.index("normal") if "normal" in self.class_names else 0
        )
        self.lesion_indices = [i for i in range(len(self.class_names)) if i != self.normal_index]
        self.image_size = int(self.cfg.data.image_size)

        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.transform = build_transforms(self.cfg, "test", keep_meta=True)

        # A heatmap is a decision aid. Refusing to classify because the
        # explanation could not be built is the wrong trade, and until now it was
        # the behaviour: build_cam -> get_cam_layer RAISES on an unknown layer
        # name, this constructor is called by ScanService, and inference/main.py
        # constructs ScanService AT IMPORT. So one renamed module took uvicorn
        # down before it bound a port, and with max_instances: 1 in
        # wrangler.jsonc that is the whole site's /api/scan, for everyone.
        try:
            self.cam = build_cam(self.model, self.cfg)
        except Exception as exc:  # noqa: BLE001 - any CAM failure is non-fatal
            self.cam = None
            logger.error(
                "Grad-CAM unavailable (%s). Predictions will be served WITHOUT a "
                "heatmap. Check explain.target_layer against this checkpoint.", exc,
            )

        # Set when the checkpoint carries a trained lesion decoder. Its map is
        # preferred over Grad-CAM: it is a supervised output at 64x64 rather than
        # an 8x8 after-the-fact gradient attribution, which is why the CAM could
        # never resolve anything smaller than a joint.
        self.has_lesion_head = hasattr(self.model, "seg_head")
        if self.has_lesion_head:
            logger.info("checkpoint carries a lesion head; it supplies the heatmap")

        # calibration.json is written next to the checkpoint by
        # scripts/calibrate.py. Absent, the model runs uncalibrated at a naive
        # 0.50 cut -- which is the state every freshly trained model is in, and
        # a state worth surfacing rather than papering over.
        self.calibration = Calibration.for_checkpoint(checkpoint)
        if self.calibration is None:
            logger.warning(
                "no %s beside %s: probabilities are uncalibrated and the decision "
                "threshold defaults to 0.50, which corresponds to no clinical policy. "
                "Run `python scripts/calibrate.py --checkpoint %s`.",
                CALIBRATION_FILENAME, checkpoint.name, checkpoint,
            )
        else:
            logger.info(
                "calibration: T=%.3f, lesion threshold=%.3f (%.0f%% target sensitivity)",
                self.calibration.temperature,
                self.calibration.lesion_threshold,
                100 * self.calibration.target_sensitivity,
            )

        self.trained_epoch = state.get("epoch")
        self.checkpoint_metrics = {
            k: v for k, v in state.items() if k not in {"model", "config", "epoch"}
        }

        logger.info("loaded %s (%s) on %s", checkpoint.name, self.cfg.model.name, self.device)
        if warmup:
            self._warmup()

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _config_from(state: dict[str, Any], config_path: str | Path) -> Config:
        """Prefer the checkpoint's embedded config; fall back to the YAML.

        ``pretrained`` is forced off either way. The state dict about to be
        loaded replaces every parameter, so downloading ImageNet weights would
        buy nothing and would make a first run require internet access.
        """
        embedded = state.get("config")
        if isinstance(embedded, dict):
            data = copy.deepcopy(embedded)
        else:
            logger.warning(
                "checkpoint has no embedded config; falling back to %s. Verify that its "
                "data.* block still matches what this model was trained with.",
                config_path,
            )
            data = load_config(config_path).to_dict()

        data.setdefault("model", {})["pretrained"] = False
        return Config(data)

    def _warmup(self) -> None:
        """Burn the first-call cost on a dummy tensor rather than a real upload.

        On ROCm the first convolution of a process triggers MIOpen kernel
        compilation, which can take several seconds. Paying it at load time keeps
        the app's first real prediction from looking broken.
        """
        dummy = torch.zeros(
            1, int(self.cfg.data.in_channels), self.image_size, self.image_size,
            device=self.device,
        )
        try:
            with torch.no_grad():
                self.model(dummy)
            compute_cam(self.cam, dummy, self.lesion_indices[-1] if self.lesion_indices else 0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort by definition
            logger.warning("warmup pass failed (%s); first prediction may be slow", exc)

    @property
    def default_threshold(self) -> float:
        """The fitted decision threshold, or 0.50 when nothing has been fitted."""
        return 0.5 if self.calibration is None else self.calibration.lesion_threshold

    @property
    def temperature(self) -> float:
        return 1.0 if self.calibration is None else self.calibration.temperature

    # -- introspection -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Facts about the loaded model, for the app's sidebar."""
        info: dict[str, Any] = {
            "checkpoint": str(self.checkpoint_path),
            "run": self.checkpoint_path.parent.name,
            "architecture": str(self.cfg.model.name),
            "classes": self.class_names,
            "image_size": self.image_size,
            "cam_method": str(self.cfg.explain.get("method", "gradcam")),
            "cam_layer": str(self.cfg.explain.get("target_layer", "")),
            "device": str(self.device),
            "calibrated": self.calibration is not None,
            "temperature": self.temperature,
            "default_threshold": self.default_threshold,
        }
        if self.calibration is not None:
            info["mode"] = self.calibration.mode
            info["target_sensitivity"] = self.calibration.target_sensitivity
            info["min_specificity"] = self.calibration.min_specificity
            info["val_sensitivity"] = self.calibration.achieved_sensitivity
            info["val_specificity"] = self.calibration.achieved_specificity
            info["calibration_warnings"] = list(self.calibration.warnings)
        if self.trained_epoch is not None:
            info["trained_epochs"] = int(self.trained_epoch) + 1
        info.update(
            {k: v for k, v in self.checkpoint_metrics.items() if isinstance(v, (int, float))}
        )
        return info

    def resolve_cam_index(self, probabilities: np.ndarray, cam_class: str = "auto") -> int:
        """Choose which class the heatmap explains.

        ``auto`` targets the most likely *lesion* class even when the verdict is
        Normal. That is deliberate: a reader looking at a negative result still
        wants "and where would it have been?", whereas a CAM taken against the
        normal class answers a question nobody asked.
        """
        if cam_class in ("auto", "", None):
            if not self.lesion_indices:
                return int(np.argmax(probabilities))
            return max(self.lesion_indices, key=lambda i: float(probabilities[i]))
        if cam_class == "predicted":
            return int(np.argmax(probabilities))
        if isinstance(cam_class, str):
            if cam_class not in self.class_names:
                raise ValueError(
                    f"unknown cam_class {cam_class!r}; expected one of "
                    f"{self.class_names + ['auto', 'predicted']}"
                )
            return self.class_names.index(cam_class)
        return int(cam_class)

    # -- inference ---------------------------------------------------------
    def predict(
        self,
        data: bytes | bytearray | memoryview | BinaryIO | str | Path,
        filename: str | None = None,
        with_heatmap: bool = True,
        threshold: float | None = None,
        cam_class: str = "auto",
        validate_input: bool = False,
        uncertainty_floor: float | None = None,
        entropy_gate: float | None = None,
    ) -> InferenceResult:
        """Classify one radiograph.

        Args:
            data: Raw bytes, an open binary file object, or a path.
            filename: Original name, used to pick the decoder. Required when
                ``data`` is bytes without a DICOM preamble.
            with_heatmap: Compute the Grad-CAM map. Roughly doubles the cost.
            threshold: Lesion probability at or above which the verdict is
                "Potential Bone Lesion". Defaults to the threshold fitted on
                validation by ``scripts/calibrate.py``, falling back to 0.50
                when no calibration has been fitted. Lower it to trade
                specificity for sensitivity.
            cam_class: ``auto``, ``predicted``, or an explicit class name.
            validate_input: Run the :mod:`onnm.ood` radiograph heuristics
                before inference and raise ``NonRadiographError`` on failure.
                Off by default so scripted evaluation of the curated dataset
                is byte-identical with and without this layer.
            uncertainty_floor: If set, a lesion verdict whose max class
                probability falls below this is downgraded to
                "Non-Diagnostic / Inconclusive" instead of being shown as a
                finding. ``None`` (default) disables the gate.
            entropy_gate: If set, a lesion verdict whose normalized predictive
                entropy reaches this is likewise downgraded. The two gates
                only ever withdraw a positive call -- a Normal verdict is
                never touched, and neither gate moves the calibrated
                threshold.

        Raises:
            RadiographReadError: the file could not be decoded.
            UnsupportedFormatError: the file type is not supported.
            onnm.ood.NonRadiographError: ``validate_input`` was set and the
                image failed radiograph validation.
        """
        threshold = self.default_threshold if threshold is None else float(threshold)

        payload = _coerce_to_bytes(data)
        if not payload:
            raise RadiographReadError("uploaded file is empty")

        if filename is None and isinstance(data, (str, Path)):
            filename = str(data)
        suffix = _suffix_for(filename, payload)

        if validate_input:
            ensure_radiograph(payload, filename)

        started = time.perf_counter()
        with _MaterialisedUpload(payload, suffix) as path:
            # Decoded twice on purpose: once here for the "as uploaded" panel at
            # native resolution, once inside the transform chain the model sees.
            # Threading one array through both would mean bypassing
            # LoadRadiographd, and the app's guarantee is that the model's input
            # came out of the real training pipeline, untouched.
            original, meta = read_radiograph(path)

            # `label` is a dummy: build_transforms ends with EnsureTyped on it,
            # so the key has to exist even though nothing here uses it.
            sample = self.transform({"image": str(path), "label": 0})

        tensor = sample["image"].unsqueeze(0).to(self.device)
        preprocessed = np.asarray(sample["image"][0].detach().cpu(), dtype=np.float32)

        with self._lock:
            lesion_map = None
            with torch.no_grad():
                if self.has_lesion_head:
                    # One forward pass yields both heads. `return_mask` is
                    # restored immediately so nothing else in this process --
                    # MONAI's Grad-CAM especially -- receives a tuple where it
                    # indexes a tensor.
                    self.model.return_mask = True
                    try:
                        raw_logits, mask_logits = self.model(tensor)
                    finally:
                        self.model.return_mask = False
                    logits = raw_logits.float()
                    lesion_map = torch.sigmoid(mask_logits.float())[0, 0].cpu().numpy()
                else:
                    logits = self.model(tensor).float()
                # Temperature scaling is monotone, so this cannot move the
                # argmax -- the predicted class is identical either way. What
                # changes is the confidence number, which is the one a reader
                # actually quotes.
                probabilities = (
                    torch.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()
                )

            heatmap = None
            cam_name = None
            heatmap_kind = None
            if with_heatmap and lesion_map is not None:
                # The lesion map is class-agnostic: it answers "where is the
                # lesion", not "what moved the benign logit". That removes the
                # caption that read "Heat map taken against the Benign class"
                # under a Normal verdict, which explained nothing to anyone.
                heatmap = _resize_map(lesion_map, preprocessed.shape)
                heatmap_kind = "lesion_map"
            elif with_heatmap and self.cam is not None:
                index = self.resolve_cam_index(probabilities, cam_class)
                cam_name = self.class_names[index]
                try:
                    # Deliberately outside no_grad: Grad-CAM needs the backward
                    # pass through the hooked convolutional block.
                    heatmap = compute_cam(self.cam, tensor, index)
                    heatmap_kind = "gradcam"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Grad-CAM failed (%s); returning prediction only", exc)
                    cam_name = None

            if self.device.type == "cuda":
                torch.cuda.synchronize()

        lesion_probability = float(1.0 - probabilities[self.normal_index])
        is_lesion = lesion_probability >= threshold

        # The uncertainty gate can only withdraw a positive call, never issue
        # one: a Normal verdict below threshold stays Normal, and gating never
        # moves the calibrated threshold itself.
        defer, max_probability, entropy = should_defer(
            probabilities,
            uncertainty_floor=uncertainty_floor,
            entropy_gate=entropy_gate,
        )
        inconclusive = bool(is_lesion and defer)
        if inconclusive:
            logger.info(
                "lesion call withdrawn as inconclusive: max_prob=%.3f entropy=%.3f",
                max_probability, entropy,
            )

        if inconclusive:
            label = INCONCLUSIVE_LABEL
        elif is_lesion:
            label = LESION_LABEL
        else:
            label = NORMAL_LABEL

        source_meta = dict(meta)
        source_meta["filename"] = Path(filename).name if filename else "upload"
        source_meta.pop("filename_or_obj", None)  # a temp path helps nobody

        return InferenceResult(
            label=label,
            confidence=100.0 * (lesion_probability if is_lesion else 1.0 - lesion_probability),
            lesion_probability=lesion_probability,
            class_probabilities={
                name: float(probabilities[i]) for i, name in enumerate(self.class_names)
            },
            top_class=self.class_names[int(np.argmax(probabilities))],
            threshold=float(threshold),
            temperature=self.temperature,
            calibrated=self.calibration is not None,
            max_probability=max_probability,
            predictive_entropy=entropy,
            inconclusive=inconclusive,
            preprocessed_image=preprocessed,
            original_image=original,
            heatmap=heatmap,
            cam_class=cam_name,
            heatmap_kind=heatmap_kind,
            source_meta=source_meta,
            elapsed_ms=1000.0 * (time.perf_counter() - started),
            device=str(self.device),
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max an arbitrary float plane to uint8 for display.

    Applied to the model-space tensor this is visually a no-op: the z-score
    normalisation before it is affine, so rescaling inverts it exactly.
    """
    array = np.asarray(image, dtype=np.float64)
    lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    scaled = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _resize_map(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Bring a decoder output up to the display grid.

    The lesion head predicts at 64x64 to stay cheap on half a vCPU, while
    ``render_overlay`` blends against the 256px model input and does no resizing
    of its own -- so without this the two shapes simply fail to broadcast.

    Bilinear rather than nearest: this is for display, and a nearest-neighbour
    upsample of a 64x64 map draws 4px blocks that read as structure the model
    never predicted.
    """
    target = (int(shape[0]), int(shape[1]))
    if array.shape == target:
        return array
    import cv2

    return cv2.resize(array, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)


def render_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.40,
    colormap: str = "jet",
    threshold: float = 0.0,
) -> np.ndarray:
    """Blend a CAM over a grayscale plane, returning uint8 RGB.

    Generalises ``explainability.overlay_cam`` on the two axes the UI needs and
    the reporting script does not: a selectable colormap, and a floor below which
    the heatmap fades out entirely. Without that floor a jet map paints the whole
    film deep blue, which reads as information when it is really just the bottom
    of the colour scale.
    """
    import matplotlib

    grey = to_display_uint8(image).astype(np.float64) / 255.0
    rgb = np.stack([grey] * 3, axis=-1)

    heat = np.clip(np.asarray(cam, dtype=np.float64), 0.0, 1.0)
    coloured = matplotlib.colormaps[colormap](heat)[..., :3]

    # Fade opacity in above the threshold rather than cutting at it: a hard edge
    # in the alpha channel is a contour the model never drew.
    weight = (
        np.zeros_like(heat)
        if threshold >= 1.0
        else np.clip((heat - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    )
    weight = (alpha * weight)[..., None]

    blended = (1.0 - weight) * rgb + weight * coloured
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def predict_file(
    path: str | Path,
    checkpoint: str | Path | None = None,
    **kwargs: Any,
) -> InferenceResult:
    """One-shot convenience wrapper, for scripts and smoke tests.

    Loads a fresh model every call, so never use it in a loop -- construct a
    :class:`RadiographClassifier` and reuse it.
    """
    checkpoint = checkpoint or default_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("no checkpoint found under reports/; train a model first")
    return RadiographClassifier(checkpoint, warmup=False).predict(path, **kwargs)


__all__ = [
    "INCONCLUSIVE_LABEL",
    "LESION_LABEL",
    "NORMAL_LABEL",
    "SUPPORTED_SUFFIXES",
    "CHECKPOINT_NAMES",
    "UPLOAD_TYPES",
    "InferenceResult",
    "RadiographClassifier",
    "default_checkpoint",
    "find_checkpoints",
    "predict_file",
    "render_overlay",
    "to_display_uint8",
]
