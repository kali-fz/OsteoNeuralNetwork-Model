"""The one inference contract, shared by every entry point.

WHAT THIS IS
------------
``run_scan`` is the whole of ONNM's model-facing behaviour reduced to a single
function: bytes in, a JSON-serialisable dict out. The Cloudflare Container's HTTP
layer (``main.py``) is a thin shell around it, and ``check_inference_parity.py``
can drive it directly.

WHY IT IS A SEPARATE MODULE FROM THE HTTP LAYER
-----------------------------------------------
The migration off Streamlit replaces the *caller*, not the model. Keeping the
call sequence in one place that has no idea whether it is being invoked by
FastAPI, a test, or a parity harness is what makes it checkable that the hosted
path does exactly what ``app.py`` did. The ordering below deliberately mirrors
``app.py:render_scanner`` step for step:

    1. The OOD gate runs first, on the raw upload.
    2. Rejections return immediately and NEVER reach the model.
    3. Only a validated radiograph is predicted, with the uncertainty gate wired
       to the same two constants the app used.

Step 2 is the one that matters clinically. In the Streamlit app a rejected file
could not reach ``classifier.predict`` because the ``continue`` came first; here
the early return is that same guarantee.

WHAT IS DELIBERATELY NOT DECIDED HERE
-------------------------------------
Nothing in this module talks to D1, checks a session, counts a quota, or decides
whether a scan is allowed. Those are authorisation and accounting questions and
they belong to the Pages Function that calls this. A container answering them
itself would be a second place where the daily cap is enforced, and two
enforcement points that must agree forever is exactly the shape this codebase
avoids elsewhere -- see the note in ``cloudflare/src/worker.js`` on why the
Worker deliberately cannot verify a password.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
from typing import Any

import numpy as np

from community import encode_image_for_sharing
from onnm.inference import InferenceResult, RadiographClassifier, render_overlay
from onnm.io_radiograph import detect_panels
from onnm.ood import DEFAULT_CONFIDENCE_FLOOR, DEFAULT_ENTROPY_GATE, validate_payload

#: Grad-CAM overlay defaults, matching the initial widget values in
#: ``app.py:render_scanner`` so a scan looks the same after the migration as it
#: did before it. The frontend may re-render at other opacities client-side, but
#: the server default has to agree with the old one or every archived screenshot
#: and saved report silently stops matching.
DEFAULT_OVERLAY_ALPHA = 0.40
DEFAULT_OVERLAY_COLORMAP = "jet"
DEFAULT_OVERLAY_FLOOR = 0.0

#: Fade-in floor for a LESION-HEAD map, kept separate from the Grad-CAM floor
#: above rather than replacing it -- so every archived Grad-CAM screenshot still
#: reproduces exactly, which is what the note above asks for.
#:
#: A separate constant is necessary, not tidiness. ``compute_cam`` rescales every
#: Grad-CAM to put its minimum at 0 and maximum at 1, so a floor of 0.0 is
#: harmless there. A sigmoid has no such guarantee: a confident "nothing here"
#: map sits near 0.02 everywhere, and ``render_overlay`` gives every pixel above
#: the floor a non-zero alpha -- so at 0.0 a clean normal film would be painted
#: deep blue from edge to edge, which reads as evidence rather than as the bottom
#: of a colour scale.
LESION_MAP_FLOOR = 0.35


def _png_b64(array: np.ndarray) -> tuple[str, int]:
    """Encode a uint8 image as a base64 PNG, returning the payload and its size.

    PNG rather than JPEG because a Grad-CAM overlay is a decision aid: JPEG
    ringing around a lesion boundary would be an artefact a reader could mistake
    for signal, and at 256 px the size saving is not worth that risk.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), len(payload)


def _rejection(filename: str, digest: str, report: Any, elapsed_ms: float) -> dict[str, Any]:
    """Shape an OOD refusal so it is obviously not a prediction.

    There is no ``prediction`` key at all rather than a neutral-looking one. A
    caller that forgets to check ``is_radiograph`` gets a KeyError on the very
    next line instead of a plausible verdict it can render.
    """
    return {
        "ok": True,
        "is_radiograph": False,
        "filename": filename,
        "image_sha256": digest,
        "ood": {
            "is_radiograph": False,
            "checks": [
                {
                    "name": check.name,
                    "passed": bool(check.passed),
                    "value": float(check.value),
                    "detail": check.detail,
                }
                for check in report.checks
            ],
        },
        "elapsed_ms": round(elapsed_ms, 1),
    }


class ScanService:
    """Holds the loaded model for the life of the process.

    The container is billed by wall-clock runtime, so the checkpoint is loaded
    once at construction and the process then serves many scans before it
    sleeps. Constructing this per request would add roughly a second of model
    load to every scan and, under a metered budget, would be paid for directly.
    """

    def __init__(self, checkpoint: str, *, warmup: bool = True) -> None:
        self.checkpoint = checkpoint
        self.classifier = RadiographClassifier(checkpoint, warmup=warmup)

    @property
    def default_threshold(self) -> float:
        """The calibrated operating point, or 0.50 when nothing was fitted.

        Exposed so the frontend seeds its threshold slider from the model rather
        than hardcoding a number that would silently stop matching the
        checkpoint after a promotion through ``scripts/version_model.py``.
        """
        return float(self.classifier.default_threshold)

    def describe(self) -> dict[str, Any]:
        """Static facts about the loaded model, for ``GET /health``.

        Delegates to ``RadiographClassifier.describe``, which already assembles
        the architecture, calibration mode, validation sensitivity/specificity
        and checkpoint metrics that the Streamlit sidebar showed. Rebuilding
        that dict here would be a second version to keep in step, and would get
        ``calibrated`` wrong: there is no such attribute on the classifier, it
        is derived from whether a calibration was found.
        """
        info = dict(self.classifier.describe())
        # The two uncertainty-gate constants are properties of the *serving*
        # policy rather than of the checkpoint, so they are added here rather
        # than pushed down into the classifier.
        info["confidence_floor"] = DEFAULT_CONFIDENCE_FLOOR
        info["entropy_gate"] = DEFAULT_ENTROPY_GATE
        return info

    def run_scan(
        self,
        payload: bytes,
        filename: str,
        *,
        threshold: float | None = None,
        cam_class: str = "auto",
        with_heatmap: bool = True,
        want_preprocessed: bool = True,
    ) -> dict[str, Any]:
        """Validate, classify and explain one radiograph.

        Args:
            payload: The uploaded file, exactly as received.
            filename: Original name. ``predict`` needs it to choose a decoder
                for anything without a DICOM preamble.
            threshold: Lesion probability at or above which the verdict is a
                finding. ``None`` uses the checkpoint's calibrated threshold.
            cam_class: ``auto``, ``predicted``, or an explicit class name.
            with_heatmap: Compute Grad-CAM. Roughly doubles the cost, because it
                needs a backward pass as well as a forward one.
            want_preprocessed: Also return the 256 px model input as a base64
                PNG. This is what ``submissions.image_b64`` stores when a user
                consents to share, so it is encoded here with the same function
                the Streamlit app used rather than re-derived.

        Returns:
            A JSON-serialisable dict. ``is_radiograph`` is False for an OOD
            rejection, in which case no model output is present at all.
        """
        started = time.perf_counter()
        digest = hashlib.sha256(payload).hexdigest()

        # The OOD gate runs first and its refusals never reach the model. This
        # ordering is the contract, not an optimisation.
        report = validate_payload(payload, filename)
        if not report.is_radiograph:
            return _rejection(
                filename, digest, report, (time.perf_counter() - started) * 1000.0
            )

        result: InferenceResult = self.classifier.predict(
            payload,
            filename=filename,
            with_heatmap=with_heatmap,
            threshold=threshold,
            cam_class=cam_class,
            uncertainty_floor=DEFAULT_CONFIDENCE_FLOOR,
            entropy_gate=DEFAULT_ENTROPY_GATE,
        )

        response: dict[str, Any] = {
            "ok": True,
            "is_radiograph": True,
            "filename": filename,
            "image_sha256": digest,
            "prediction": result.as_dict(),
            # Raw probabilities travel to the browser so the threshold slider
            # can re-cut the verdict client-side, exactly as
            # InferenceResult.with_threshold does server-side. Moving the slider
            # must not cost a request, and under a metered container budget that
            # stops being merely a nicety.
            "class_probabilities": {
                name: float(value) for name, value in result.class_probabilities.items()
            },
            "lesion_probability": float(result.lesion_probability),
            "default_threshold": self.default_threshold,
            "confidence_floor": DEFAULT_CONFIDENCE_FLOOR,
            "entropy_gate": DEFAULT_ENTROPY_GATE,
            "ood": {"is_radiograph": True, "checks": []},
        }

        # Advisories are notes about the INPUT, not about the finding. They run
        # after the OOD gate, so a rejection still returns with no `prediction`
        # key at all, and they never change the verdict -- a composite is still a
        # radiograph and still gets an answer.
        #
        # Shipped as a list so the browser can render nothing at all when it is
        # empty, and so a cached older frontend that ignores the key still works.
        advisories: list[dict[str, Any]] = []
        panels = detect_panels(result.original_image)
        if panels["is_composite"]:
            advisories.append(
                {
                    "code": "multi_panel",
                    "message": (
                        f"This looks like {panels['n_panels']} views combined into one "
                        "image. The model was trained on single views, and combining "
                        "them shrinks each one by about half, which makes a small "
                        "lesion harder to see. Uploading each view on its own will "
                        "give a more reliable result."
                    ),
                }
            )
        if advisories:
            response["advisories"] = advisories

        if result.heatmap is not None:
            is_lesion_map = result.heatmap_kind == "lesion_map"
            floor = LESION_MAP_FLOOR if is_lesion_map else DEFAULT_OVERLAY_FLOOR
            overlay = render_overlay(
                result.preprocessed_image,
                result.heatmap,
                alpha=DEFAULT_OVERLAY_ALPHA,
                colormap=DEFAULT_OVERLAY_COLORMAP,
                threshold=floor,
            )
            overlay_b64, overlay_bytes = _png_b64(overlay)
            response["overlay"] = {
                "png_b64": overlay_b64,
                "bytes": overlay_bytes,
                # `kind` tells the browser which caption is true. A lesion map is
                # class-agnostic -- "where is the lesion" -- so captioning it as
                # taken "against" a class would describe a Grad-CAM property it
                # does not have. cam_class stays None for it.
                "kind": result.heatmap_kind or "gradcam",
                "cam_class": result.cam_class,
                "alpha": DEFAULT_OVERLAY_ALPHA,
                "colormap": DEFAULT_OVERLAY_COLORMAP,
                "floor": floor,
            }

        if want_preprocessed:
            # Reuses the community encoder rather than reimplementing it, so the
            # bytes stored in D1 are byte-identical to what the Streamlit app
            # would have stored for the same upload.
            image_b64, image_sha256, image_bytes = encode_image_for_sharing(
                result.preprocessed_image
            )
            response["preprocessed"] = {
                "png_b64": image_b64,
                "sha256": image_sha256,
                "bytes": image_bytes,
            }

        response["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return response


__all__ = [
    "DEFAULT_OVERLAY_ALPHA",
    "DEFAULT_OVERLAY_COLORMAP",
    "DEFAULT_OVERLAY_FLOOR",
    "ScanService",
]
