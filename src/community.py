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

# Cloudflare's edge refuses the default urllib User-Agent ("Python-urllib/3.x")
# with HTTP 403 and a plain-text "error code: 1010" body -- a browser-signature
# ban applied before the request ever reaches the Worker. Nothing in the Worker,
# the token or the schema is involved, which is what makes it so misleading: the
# API looks broken while curl and a browser both get a clean 200. Sending an
# honest, identifiable agent string is the whole fix, and is better manners than
# impersonating a browser would be.
USER_AGENT = "ONNM-Streamlit/1.0 (+https://github.com/kali-fz/OsteoNeuralNetwork-Model)"

DEFAULT_TIMEOUT = 15.0
# Health checks get their own, much shorter budget. They run to decorate the UI,
# not to do work, so a slow or unreachable API must cost a moment rather than
# stall the page. The long timeout stays for calls that carry real payloads.
HEALTH_TIMEOUT = 3.0
# Mirrors MAX_IMAGE_B64_BYTES in cloudflare/src/worker.js. Checked here too so
# an oversized image is rejected before it is uploaded rather than after.
MAX_IMAGE_B64_BYTES = 600_000

VALID_LABELS = ("normal", "benign", "malignant")

#: What a reviewer may write into ``admin_label``. ``misc`` means "not a bone
#: radiograph at all" -- a genuine training target for the OOD detector, which
#: today has only hand-written heuristics and no negatives to learn from, but
#: not a diagnosis. The Worker and the D1 schema carry the same four values.
MISC_LABEL = "misc"
REVIEW_LABELS = (*VALID_LABELS, MISC_LABEL)

#: The three triage buckets. See :func:`classify_bucket`.
BUCKET_VALID_BONE = "valid_bone"
BUCKET_MISC = "misc"
BUCKET_CONTRADICTION = "contradiction"
BUCKETS = (BUCKET_VALID_BONE, BUCKET_MISC, BUCKET_CONTRADICTION)

#: Human-readable bucket names, for the review UI and the export summary.
BUCKET_TITLES = {
    BUCKET_VALID_BONE: "Valid bone radiographs",
    BUCKET_MISC: "Misc / misuse",
    BUCKET_CONTRADICTION: "Mislabelled — the system contradicted itself",
}

#: Mirrors ``DEFAULT_CONFIDENCE_FLOOR`` in ``onnm.ood`` and ``CONFIDENT_PROB``
#: in the Worker. Imported rather than re-derived would be nicer, but ``ood``
#: pulls in numpy and this module is deliberately dependency-light so that
#: ``backend.py`` can import it without the inference stack.
CONFIDENT_PROB = 0.65

# ---------------------------------------------------------------------------
# Who may review
# ---------------------------------------------------------------------------
#: The single account permitted to see the review queue or approve anything.
#:
#: Hardcoded in three places on purpose -- here, in ``cloudflare/src/worker.js``,
#: and as a CHECK constraint in ``cloudflare/schema.sql``. Review is the only
#: path by which any data reaches training, so "who may review" is a property of
#: the deployment rather than a setting: an environment variable that could be
#: mistyped, or a database flag that a future endpoint could grant, would both
#: be weaker than a constant that requires a code change and a migration to
#: move. Tests assert the three copies agree.
ADMIN_USER_ID = "c2c5a209-4aaa-4eb9-b112-b2929b6dbe12"
ADMIN_EMAIL = "kzfhero@gmail.com"


def is_admin(user_id: str | None, email: str | None = None) -> bool:
    """True only for the one account allowed to review submissions.

    Matches on the user id, and on the email address as well when one is given.
    The id is the real check -- it is what the Worker and the schema pin -- and
    the email is a second, independent statement of the same fact, so that a
    session carrying a mismatched pair is refused rather than resolved.
    """
    if user_id != ADMIN_USER_ID:
        return False
    return email is None or str(email).strip().lower() == ADMIN_EMAIL


def classify_bucket(
    *,
    ood_flagged: bool,
    max_probability: float = 0.0,
    user_says_wrong: bool = False,
    user_suggested_label: str | None = None,
) -> tuple[str, str]:
    """Sort one submission into a triage bucket. Returns ``(bucket, reason)``.

    The Python mirror of ``triageBucket()`` in ``cloudflare/src/worker.js``.
    The Worker is authoritative -- it is what actually writes the column -- and
    this exists so the rule can be unit-tested without a network, and so the
    app can show a user which queue their submission joined.

    The three buckets:

    ``valid_bone``
        The OOD gate accepted the image and the classifier ran normally. These
        retrain the lesion head and need a clinical label.

    ``misc``
        The gate rejected it: a hotdog, a screenshot, a photograph of a wall.
        Misuse is data. These retrain the OOD detector as negatives, and must
        never carry a diagnosis, because they have none.

    ``contradiction``
        The system disagrees with itself. Either the gate rejected an image the
        user insists is a radiograph -- a false rejection nobody but the user
        can witness, since inference never ran -- or it accepted one the user
        says is not a radiograph at all while the classifier confidently
        diagnosed it. Worth the most per row: each is a demonstrated failure of
        the gate with the image still attached.

    Note what is *not* a contradiction: a user disputing the grade ("you said
    malignant, I think benign") on an accepted radiograph. That is a labelling
    disagreement for the reviewer, not evidence that the gate misfired, and it
    stays in ``valid_bone``.
    """
    user_says_not_radiograph = user_suggested_label == MISC_LABEL
    if ood_flagged:
        if user_says_wrong and not user_says_not_radiograph:
            return BUCKET_CONTRADICTION, "gate rejected it; the user says it is a radiograph"
        if max_probability >= CONFIDENT_PROB:
            return (
                BUCKET_CONTRADICTION,
                f"gate rejected it but the classifier was {max_probability:.2f} confident",
            )
        return BUCKET_MISC, "the out-of-distribution gate rejected it"
    if user_says_not_radiograph:
        return BUCKET_CONTRADICTION, "gate accepted it; the user says it is not a radiograph"
    return BUCKET_VALID_BONE, "the out-of-distribution gate accepted it"


@dataclass(frozen=True)
class User:
    """Mirrors database.User so auth.py can use either backend."""

    user_id: str
    email: str
    password_hash: str | None
    created_at: str
    tos_accepted_at: str
    is_admin: bool = False
    auth_provider: str = "password"
    provider_subject: str | None = None
    display_name: str | None = None
    profile_picture_url: str | None = None
    public_contributor_profile: bool = False


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


def encode_payload_for_sharing(payload: bytes, max_side: int = 256) -> tuple[str, str, int]:
    """Encode a *rejected* upload -- the raw file bytes -- for community storage.

    :func:`encode_image_for_sharing` takes the model input, which only exists
    once inference has run. An image the OOD gate turned away never reaches the
    model, and those are exactly the images the gate needs as negatives: it
    currently learns from no data at all, only hand-written thresholds. Without
    this the "misc" bucket would be a queue of rows with nothing in them.

    Re-encoding through Pillow to a grayscale PNG is also the de-identification
    step. It is the same treatment ``storage.py`` gives a standard image: the
    pixels survive and every scrap of container metadata -- EXIF, GPS, camera
    make, colour profiles -- is discarded, because a new single-channel PNG is
    written from the pixel array rather than the original file being copied.

    DICOM is deliberately not handled here. Its identifiers live in headers that
    Pillow cannot read and therefore cannot be shown to have stripped, and the
    de-identification path that does handle them runs later, after the gate. A
    rejected DICOM is stored as a row with no image; the alternative is a code
    path that could put patient details in a shared table.

    Returns ``(base64_png, sha256_hex, byte_length)``.
    """
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("L")
        # Thumbnail rather than resize: the aspect ratio carries information
        # about what the misuse actually was, and a squashed hotdog is a worse
        # negative than a small one.
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        array = np.asarray(image)

    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG", optimize=True)
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
        """Whether *this process* can call the admin routes at all.

        Holding the key is necessary but not sufficient: the Worker also
        requires the request to name the one account allowed to review. Call
        :func:`is_admin` on the signed-in session before showing any review UI,
        because this property answers "is a key configured", not "is this
        person allowed".
        """
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
        request.add_header("user-agent", USER_AGENT)
        if admin:
            # Says which account is asking, alongside the key that says the
            # caller is trusted software. The Worker checks both, so a process
            # holding the admin key still cannot review as somebody else.
            request.add_header("x-onnm-admin-user", ADMIN_USER_ID)
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
                # A non-JSON body means something in front of the Worker
                # answered -- the Cloudflare edge, a proxy, a captive portal.
                # Say which, because "error code: 1010" alone sends you
                # debugging the Worker, where the problem is not.
                parsed = {
                    "error": f"the API gateway refused the request (HTTP {exc.code}): "
                    f"{body.strip()[:200]}"
                }
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

    # -- globe ---------------------------------------------------------------
    def globe(self) -> dict[str, Any] | None:
        """Aggregated country counts for the landing-page globe.

        Returns country codes and integers only -- never a user, a submission,
        a timestamp or a coordinate. See the ``/globe`` handler in
        ``cloudflare/src/worker.js`` for what the Worker refuses to include,
        and ``src/geo.py`` for the step that attaches coordinates locally.

        Called by the Streamlit **server**, not by the visitor's browser: this
        route is behind the API key like every other one, and the key must
        never reach a page. Fails soft like ``health()`` -- a decorative globe
        is not a reason for a landing page to error.
        """
        previous, self.timeout = self.timeout, min(self.timeout, HEALTH_TIMEOUT)
        try:
            status, body = self._request("GET", "/globe")
        finally:
            self.timeout = previous
        return body if status == 200 else None

    def contributors(self) -> list[dict[str, Any]]:
        """Public, explicitly opted-in Google contributor profiles only."""
        status, body = self._request("GET", "/contributors")
        if status != 200:
            return []
        rows = body.get("contributors")
        return rows if isinstance(rows, list) else []

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

    def create_oauth_user(
        self,
        email: str,
        provider_subject: str,
        *,
        auth_provider: str = "google",
        display_name: str | None = None,
        profile_picture_url: str | None = None,
    ) -> User:
        """Create a federated account. Sends no password hash, and cannot.

        The Worker rejects a ``password_hash`` on a federated account outright,
        so a bug that tried to attach one fails loudly rather than creating an
        account that could be logged into by two different routes.
        """
        user_id = str(uuid.uuid4())
        status, body = self._request(
            "POST",
            "/users",
            {
                "user_id": user_id,
                "email": email,
                "auth_provider": auth_provider,
                "provider_subject": provider_subject,
                "display_name": display_name,
                "profile_picture_url": profile_picture_url,
            },
        )
        if status == 201:
            return User(
                user_id=user_id,
                email=email,
                password_hash=None,
                created_at=body.get("created_at", ""),
                tos_accepted_at=body.get("created_at", ""),
                auth_provider=auth_provider,
                provider_subject=provider_subject,
                display_name=display_name,
                profile_picture_url=profile_picture_url,
            )
        if status == 409:
            raise DuplicateEmailError("an account already exists for that email")
        raise CommunityError(body.get("error", f"could not create account (status {status})"))

    def _user_from_body(self, body: dict[str, Any]) -> User:
        return User(
            user_id=body["user_id"],
            email=body["email"],
            password_hash=body.get("password_hash"),
            created_at=body.get("created_at", ""),
            tos_accepted_at=body.get("tos_accepted_at", ""),
            is_admin=bool(body.get("is_admin", 0)),
            auth_provider=body.get("auth_provider") or "password",
            provider_subject=body.get("provider_subject"),
            display_name=body.get("display_name"),
            profile_picture_url=body.get("profile_picture_url"),
            public_contributor_profile=bool(body.get("public_contributor_profile", 0)),
        )

    def get_user_by_email(self, email: str) -> User | None:
        status, body = self._request("GET", "/users/by-email", params={"email": email})
        return self._user_from_body(body) if status == 200 else None

    def get_user_by_subject(self, provider_subject: str) -> User | None:
        """Look a federated account up by the provider's stable subject claim."""
        status, body = self._request(
            "GET", "/users/by-subject", params={"subject": provider_subject}
        )
        return self._user_from_body(body) if status == 200 else None

    def update_contributor_profile(
        self,
        user_id: str,
        provider_subject: str,
        *,
        display_name: str | None,
        profile_picture_url: str | None,
        public_profile: bool | None = None,
    ) -> bool:
        """Refresh trusted Google fields and optionally change public visibility."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "provider_subject": provider_subject,
            "display_name": display_name,
            "profile_picture_url": profile_picture_url,
        }
        if public_profile is not None:
            payload["public_profile"] = bool(public_profile)
        status, _ = self._request("POST", "/users/profile", payload)
        return status == 200

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

    def create_rejected_submission(
        self,
        user_id: str,
        *,
        shared: bool,
        image_b64: str | None = None,
        image_sha256: str | None = None,
        ood_score: float | None = None,
    ) -> str | None:
        """Record an upload the OOD gate refused, before inference ever ran.

        These rows are the entire content of the ``misc`` bucket, and the reason
        the OOD detector can eventually be retrained on evidence rather than on
        more hand-tuned thresholds. There is no prediction to store, so
        ``model_label`` is ``'rejected'`` and the probability map is empty --
        which is also what keeps the row out of the lesion manifest: it has no
        class, and the export refuses to invent one.
        """
        submission_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "submission_id": submission_id,
            "user_id": user_id,
            "model_label": "rejected",
            "lesion_probability": 0.0,
            "class_probabilities": {},
            "ood_flagged": True,
            "ood_score": ood_score,
            "shared": bool(shared),
        }
        if shared and image_b64:
            payload["image_b64"] = image_b64
            payload["image_sha256"] = image_sha256

        status, body = self._request("POST", "/submissions", payload)
        if status == 201:
            return submission_id
        logger.warning("rejection not recorded (%s): %s", status, body.get("error"))
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
        if suggested_label and suggested_label not in REVIEW_LABELS:
            raise ValueError(f"suggested_label must be one of {REVIEW_LABELS}")
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
    def pending_review(
        self, limit: int = 25, with_images: bool = True, bucket: str | None = None
    ) -> list[dict[str, Any]]:
        """The review queue, optionally narrowed to one triage bucket.

        Filtering server-side rather than in the UI matters here: with images
        attached each row is ~30 KB, so fetching all three buckets in order to
        render one tab would move megabytes to display a third of them.
        """
        if bucket is not None and bucket not in BUCKETS:
            raise ValueError(f"bucket must be one of {BUCKETS}")
        status, body = self._request(
            "GET",
            "/admin/pending",
            params={
                "limit": limit,
                "images": "1" if with_images else "0",
                "bucket": bucket,
            },
            admin=True,
        )
        return body.get("pending", []) if status == 200 else []

    def review_submission(
        self,
        submission_id: str,
        *,
        decision: str,
        admin_label: str | None = None,
        admin_bucket: str | None = None,
        note: str | None = None,
        reviewed_by: str = ADMIN_USER_ID,
    ) -> tuple[bool, str]:
        """Approve or reject. Approving requires a ground truth *and* a bucket.

        The bucket says what the row is for -- retraining the lesion head, or
        hardening the OOD gate -- and the label says what the image is. Both are
        required because the export sorts on the first and trains on the second,
        so a row missing either would be silently dropped rather than raise.

        The pairing is checked here, again in the Worker, and a third time by a
        schema trigger. Two of those are redundant on any given call; the one
        that is not is whichever the current bug is in.

        Returns ``(ok, message)`` so a UI can show why a rejection happened.
        """
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        if decision == "approved":
            if admin_label not in REVIEW_LABELS:
                raise ValueError(f"approving requires admin_label in {REVIEW_LABELS}")
            if admin_bucket not in BUCKETS:
                raise ValueError(f"approving requires admin_bucket in {BUCKETS}")
            if admin_bucket == BUCKET_MISC and admin_label != MISC_LABEL:
                raise ValueError("a misc row has no diagnosis: label it 'misc'")
            if admin_bucket == BUCKET_VALID_BONE and admin_label == MISC_LABEL:
                raise ValueError("a bone radiograph needs a clinical label, not 'misc'")
        status, body = self._request(
            "POST",
            f"/admin/review/{urllib.parse.quote(submission_id)}",
            {
                "decision": decision,
                "admin_label": admin_label,
                "admin_bucket": admin_bucket,
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
        return {
            "batch_id": None, "count": 0, "lesion_rows": 0, "ood_rows": 0,
            "rows": [], "error": body.get("error"),
        }


_client: CommunityClient | None = None


def get_client() -> CommunityClient:
    """Process-wide client, built from the environment on first use."""
    global _client
    if _client is None:
        _client = CommunityClient()
    return _client
