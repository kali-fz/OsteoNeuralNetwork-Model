"""Client for the ONNM community API (Cloudflare Worker + D1).

WHY THIS EXISTS
---------------
The Streamlit app runs on Hugging Face Spaces, whose filesystem is ephemeral --
a restart wipes it, and persistent storage is a paid add-on. So the SQLite
database in ``database.py`` cannot hold accounts for a hosted deployment. This
module is the same data layer over a network instead of a file.

It deliberately mirrors ``database.py``'s interface for the user functions
(``create_user``, ``get_user_by_email``, returning the same ``User`` dataclass),
so ``auth.py`` works against either backend unchanged. Password hashing stays in
``auth.py``: only the encoded PBKDF2 string ever crosses the network, and the
Worker cannot verify a password even if it wanted to.

DEGRADED MODE
-------------
Every call fails soft. If the API is unreachable, misconfigured, or over its
cap, the caller gets ``None``/``False`` and a logged warning rather than an
exception. Inference is local and must keep working when the community
features do not -- a network blip should not stop someone reading a
radiograph.

CONFIGURATION
-------------
Read from the environment, which is how Hugging Face Spaces secrets arrive:

    ONNM_COMMUNITY_URL   https://onnm-community.<subdomain>.workers.dev
    ONNM_COMMUNITY_KEY   the app key  (read/write ordinary rows)
    ONNM_ADMIN_KEY       the admin key (review queue, approvals, export)

Absent ONNM_COMMUNITY_URL the client is simply disabled, which is the correct
behaviour for a local run.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0
# Health checks get their own, much shorter budget. They run to decorate the UI,
# not to do work, so a slow or unreachable API must cost a moment rather than
# stall the page. The long timeout stays for calls that carry real payloads.
HEALTH_TIMEOUT = 3.0
# Mirrors MAX_IMAGE_B64_BYTES in cloudflare/src/worker.js. Checked here too so
# an oversized image is rejected before it is uploaded rather than after.
MAX_IMAGE_B64_BYTES = 600_000

VALID_LABELS = ("normal", "benign", "malignant")


@dataclass(frozen=True)
class User:
    """Mirrors database.User so auth.py can use either backend."""

    user_id: str
    email: str
    password_hash: str
    created_at: str
    tos_accepted_at: str
    is_admin: bool = False


class CommunityError(RuntimeError):
    """A community API call failed in a way the caller should know about."""


class DuplicateEmailError(CommunityError):
    """An account already exists for an email address."""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def encode_image_for_sharing(image: Any) -> tuple[str, str, int]:
    """Encode a preprocessed radiograph as base64 PNG for community storage.

    Takes the *model input* (256px, single channel, already windowed and with
    any DICOM metadata long since discarded by the loading path) rather than
    the uploaded original. Three reasons, in order of importance:

    1. It is what retraining actually consumes, so storing the original would
       cost far more to hold data the pipeline would only downsample again.
    2. A DICOM original carries patient identifiers in its headers. The
       preprocessed array is pixels alone.
    3. ~30 KB rather than several MB keeps the whole thing inside D1's free
       tier without ever needing R2, and therefore without a payment method.

    Returns ``(base64_png, sha256_hex, byte_length)``.
    """
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 3:
        # (C, H, W) -> (H, W) by taking the first channel; the three channels
        # are an ImageNet-shaped copy of one grayscale plane.
        array = array[0] if array.shape[0] <= 4 else array[..., 0]

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise CommunityError("image contains no finite pixels")
    low, high = float(finite.min()), float(finite.max())
    scaled = np.zeros_like(array, dtype=np.uint8) if high <= low else (
        ((array - low) / (high - low) * 255.0).clip(0, 255).astype(np.uint8)
    )

    buffer = io.BytesIO()
    Image.fromarray(scaled, mode="L").save(buffer, format="PNG", optimize=True)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > MAX_IMAGE_B64_BYTES:
        raise CommunityError(
            f"encoded image is {len(encoded)} bytes, over the {MAX_IMAGE_B64_BYTES} limit"
        )
    return encoded, hashlib.sha256(raw).hexdigest(), len(encoded)


def decode_shared_image(encoded: str) -> Any:
    """Inverse of :func:`encode_image_for_sharing`, for review and retraining."""
    import numpy as np
    from PIL import Image

    return np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L"))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class CommunityClient:
    """Talks to the Cloudflare Worker. Every method fails soft."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        admin_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ONNM_COMMUNITY_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("ONNM_COMMUNITY_KEY", "")
        self.admin_key = admin_key or os.environ.get("ONNM_ADMIN_KEY", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        """False when unconfigured -- the app then runs purely locally."""
        return bool(self.base_url and self.api_key)

    @property
    def admin_enabled(self) -> bool:
        return bool(self.base_url and self.admin_key)

    # -- transport ---------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        admin: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """Return ``(status, body)``. Never raises for network or HTTP errors."""
        if not self.base_url:
            return 0, {"error": "community API not configured"}
        key = self.admin_key if admin else self.api_key
        if not key:
            return 0, {"error": "admin key not configured" if admin else "api key not configured"}

        url = f"{self.base_url}{path}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urllib.parse.urlencode(clean)}"

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("authorization", f"Bearer {key}")
        if data is not None:
            request.add_header("content-type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body or "{}")
            except json.JSONDecodeError:
                parsed = {"error": body[:500]}
            # 4xx are expected control flow (duplicate email, rate limit) and
            # are not logged as failures; 5xx are genuine faults.
            if exc.code >= 500:
                logger.warning("community API %s %s -> %s", method, path, exc.code)
            return exc.code, parsed
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("community API unreachable (%s %s): %s", method, path, exc)
            return 0, {"error": str(exc)}
        except json.JSONDecodeError as exc:
            logger.warning("community API returned non-JSON (%s %s): %s", method, path, exc)
            return 0, {"error": "invalid JSON from API"}

    # -- health ------------------------------------------------------------
    def health(self) -> dict[str, Any] | None:
        """Liveness and capacity. Uses the short timeout -- see HEALTH_TIMEOUT."""
        previous, self.timeout = self.timeout, min(self.timeout, HEALTH_TIMEOUT)
        try:
            status, body = self._request("GET", "/health")
        finally:
            self.timeout = previous
        return body if status == 200 else None

    # -- accounts (mirrors database.py) ------------------------------------
    def create_user(
        self, email: str, password_hash: str, *, is_admin: bool = False
    ) -> User:
        """Create an account. Raises DuplicateEmailError, matching database.py."""
        user_id = str(uuid.uuid4())
        status, body = self._request(
            "POST",
            "/users",
            {
                "user_id": user_id,
                "email": email,
                "password_hash": password_hash,
                "is_admin": bool(is_admin),
            },
        )
        if status == 201:
            return User(
                user_id=user_id,
                email=email,
                password_hash=password_hash,
                created_at=body.get("created_at", ""),
                tos_accepted_at=body.get("created_at", ""),
                is_admin=bool(is_admin),
            )
        if status == 409:
            raise DuplicateEmailError("an account already exists for that email")
        raise CommunityError(body.get("error", f"could not create account (status {status})"))

    def get_user_by_email(self, email: str) -> User | None:
        status, body = self._request("GET", "/users/by-email", params={"email": email})
        if status != 200:
            return None
        return User(
            user_id=body["user_id"],
            email=body["email"],
            password_hash=body["password_hash"],
            created_at=body.get("created_at", ""),
            tos_accepted_at=body.get("tos_accepted_at", ""),
            is_admin=bool(body.get("is_admin", 0)),
        )

    # -- submissions -------------------------------------------------------
    def create_submission(
        self,
        user_id: str,
        result: Any,
        *,
        shared: bool = False,
        image_b64: str | None = None,
        image_sha256: str | None = None,
        ood_flagged: bool = False,
        ood_score: float | None = None,
        checkpoint: str | None = None,
    ) -> str | None:
        """Record one prediction. Returns the submission id, or None on failure.

        ``image_b64`` is sent only when ``shared`` is true. The Worker discards
        an image sent without consent as well, so a bug here cannot cause
        silent retention -- but not sending it is the first line of defence.
        """
        submission_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "submission_id": submission_id,
            "user_id": user_id,
            "model_label": getattr(result, "label", str(result)),
            "lesion_probability": float(getattr(result, "lesion_probability", 0.0)),
            "class_probabilities": dict(getattr(result, "class_probabilities", {}) or {}),
            "threshold": float(getattr(result, "threshold", 0.5)),
            "calibrated": bool(getattr(result, "calibrated", False)),
            "checkpoint": checkpoint,
            "ood_flagged": bool(ood_flagged),
            "ood_score": ood_score,
            "shared": bool(shared),
        }
        if shared and image_b64:
            payload["image_b64"] = image_b64
            payload["image_sha256"] = image_sha256

        status, body = self._request("POST", "/submissions", payload)
        if status == 201:
            return submission_id
        logger.warning("submission not recorded (%s): %s", status, body.get("error"))
        return None

    def submit_feedback(
        self,
        submission_id: str,
        user_id: str,
        *,
        says_wrong: bool,
        suggested_label: str | None = None,
        comment: str | None = None,
    ) -> bool:
        """Record what the user thinks. This is a signal, never a label.

        Nothing written here can reach a training set: the Worker writes only
        the untrusted columns, and the export query reads only ``admin_label``.
        """
        if suggested_label and suggested_label not in VALID_LABELS:
            raise ValueError(f"suggested_label must be one of {VALID_LABELS}")
        status, _ = self._request(
            "POST",
            f"/submissions/{urllib.parse.quote(submission_id)}/feedback",
            {
                "user_id": user_id,
                "says_wrong": bool(says_wrong),
                "suggested_label": suggested_label,
                "comment": comment,
            },
        )
        return status == 200

    def list_user_submissions(self, user_id: str, limit: int = 25) -> list[dict[str, Any]]:
        status, body = self._request(
            "GET", "/submissions", params={"user_id": user_id, "limit": limit}
        )
        return body.get("submissions", []) if status == 200 else []

    # -- admin -------------------------------------------------------------
    def pending_review(self, limit: int = 25, with_images: bool = True) -> list[dict[str, Any]]:
        status, body = self._request(
            "GET",
            "/admin/pending",
            params={"limit": limit, "images": "1" if with_images else "0"},
            admin=True,
        )
        return body.get("pending", []) if status == 200 else []

    def review_submission(
        self,
        submission_id: str,
        *,
        decision: str,
        admin_label: str | None = None,
        note: str | None = None,
        reviewed_by: str = "admin",
    ) -> tuple[bool, str]:
        """Approve or reject. Approving requires a ground-truth label.

        Returns ``(ok, message)`` so a UI can show why a rejection happened.
        """
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        if decision == "approved" and admin_label not in VALID_LABELS:
            raise ValueError(f"approving requires admin_label in {VALID_LABELS}")
        status, body = self._request(
            "POST",
            f"/admin/review/{urllib.parse.quote(submission_id)}",
            {
                "decision": decision,
                "admin_label": admin_label,
                "note": note,
                "reviewed_by": reviewed_by,
            },
            admin=True,
        )
        return (status == 200), body.get("error", "ok")

    def export_batch(
        self, batch_id: str | None = None, note: str | None = None,
        limit: int = 100, dry_run: bool = False,
    ) -> dict[str, Any]:
        """Claim approved rows into a training batch.

        Rows already carrying a ``batch_id`` are excluded, so the same example
        cannot silently enter two generations of training.
        """
        status, body = self._request(
            "POST",
            "/admin/export",
            {"batch_id": batch_id, "note": note, "limit": limit, "dry_run": dry_run},
            admin=True,
        )
        if status == 200:
            return body
        return {"batch_id": None, "count": 0, "rows": [], "error": body.get("error")}


_client: CommunityClient | None = None


def get_client() -> CommunityClient:
    """Process-wide client, built from the environment on first use."""
    global _client
    if _client is None:
        _client = CommunityClient()
    return _client
