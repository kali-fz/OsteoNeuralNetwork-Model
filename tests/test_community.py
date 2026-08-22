"""Tests for the community API client and the review-gate invariant.

Two things are worth testing here, and they are not the HTTP plumbing.

The first is that the client **fails soft**. This runs on Hugging Face Spaces
talking to a Cloudflare Worker across the public internet, and inference is
local. A network blip must not stop someone reading a radiograph, so every
call has to degrade to "no community features" rather than raise.

The second is the invariant the whole design rests on: a user saying "this was
wrong" is a signal, never a label. Anyone can write the untrusted columns.
Only a human review can write ``admin_label``, and only ``admin_label`` reaches
training. That is enforced in three places and tested at the database level
here, because it is the one failure that would silently poison the model
instead of raising.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3

import numpy as np
import pytest

from community import (
    MAX_IMAGE_B64_BYTES,
    CommunityClient,
    CommunityError,
    DuplicateEmailError,
    User,
    decode_shared_image,
    encode_image_for_sharing,
)

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "cloudflare" / "schema.sql"


# ---------------------------------------------------------------------------
# The review gate, tested against the real schema
# ---------------------------------------------------------------------------
@pytest.fixture
def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.execute(
        "INSERT INTO users VALUES ('u1','a@b.c','pbkdf2_sha256$1$aa$bb','t','t',0)"
    )
    return con


def _submit(con, sid, *, shared=1, image="IMG", says_wrong=None, suggested=None):
    con.execute(
        """INSERT INTO submissions
           (submission_id,user_id,created_at,model_label,lesion_probability,
            class_probabilities,shared,image_b64,user_says_wrong,user_suggested_label)
           VALUES (?,?,'t','malignant',0.9,'{}',?,?,?,?)""",
        (sid, "u1", shared, image if shared else None, says_wrong, suggested),
    )


def test_cannot_approve_without_a_ground_truth_label(db) -> None:
    """The hotdog case.

    Someone uploads a hotdog, the model says malignant, they click "wrong".
    If that alone could reach the training set, the next generation trains on
    a hotdog labelled normal. Approval must require a human to state the label.
    """
    _submit(db, "s1", says_wrong=1, suggested="normal")
    with pytest.raises(sqlite3.IntegrityError, match="admin_label"):
        db.execute("UPDATE submissions SET review_status='approved' WHERE submission_id='s1'")


def test_cannot_approve_a_submission_the_user_did_not_share(db) -> None:
    """Consent is opt-in, and approval cannot manufacture it after the fact."""
    _submit(db, "s2", shared=0)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "UPDATE submissions SET review_status='approved', admin_label='normal' "
            "WHERE submission_id='s2'"
        )


def test_admin_label_is_constrained_to_the_three_classes(db) -> None:
    _submit(db, "s3")
    with pytest.raises(sqlite3.IntegrityError, match="admin_label"):
        db.execute("UPDATE submissions SET admin_label='hotdog' WHERE submission_id='s3'")


def test_export_uses_the_admin_label_not_the_user_suggestion(db) -> None:
    """The user guessed 'benign'; the reviewer determined 'normal'.

    What ships to training must be the reviewer's call.
    """
    _submit(db, "s4", says_wrong=1, suggested="benign")
    db.execute(
        "UPDATE submissions SET review_status='approved', admin_label='normal' "
        "WHERE submission_id='s4'"
    )
    row = db.execute(
        """SELECT admin_label, user_suggested_label FROM submissions
           WHERE review_status='approved' AND admin_label IS NOT NULL
             AND shared=1 AND image_b64 IS NOT NULL"""
    ).fetchone()
    assert row == ("normal", "benign")


def test_pending_and_rejected_rows_never_export(db) -> None:
    _submit(db, "s5")                      # pending
    _submit(db, "s6")
    db.execute("UPDATE submissions SET review_status='rejected' WHERE submission_id='s6'")
    exported = db.execute(
        """SELECT COUNT(*) FROM submissions
           WHERE review_status='approved' AND admin_label IS NOT NULL AND shared=1"""
    ).fetchone()[0]
    assert exported == 0


def test_a_row_cannot_enter_two_batches(db) -> None:
    """batch_id is claimed on export, and the export query skips claimed rows."""
    _submit(db, "s7")
    db.execute(
        "UPDATE submissions SET review_status='approved', admin_label='benign' "
        "WHERE submission_id='s7'"
    )
    db.execute("UPDATE submissions SET batch_id='batch-1' WHERE submission_id='s7'")
    remaining = db.execute(
        """SELECT COUNT(*) FROM submissions
           WHERE review_status='approved' AND admin_label IS NOT NULL
             AND shared=1 AND batch_id IS NULL"""
    ).fetchone()[0]
    assert remaining == 0


# ---------------------------------------------------------------------------
# Failing soft
# ---------------------------------------------------------------------------
def test_client_is_disabled_without_configuration(monkeypatch) -> None:
    for var in ("ONNM_COMMUNITY_URL", "ONNM_COMMUNITY_KEY", "ONNM_ADMIN_KEY"):
        monkeypatch.delenv(var, raising=False)
    client = CommunityClient()
    assert client.enabled is False
    assert client.admin_enabled is False


def test_unreachable_api_returns_empty_rather_than_raising() -> None:
    """The whole point: a dead API must not take the radiograph viewer with it."""
    client = CommunityClient(base_url="http://127.0.0.1:9", api_key="k", timeout=0.4)
    assert client.health() is None
    assert client.list_user_submissions("u1") == []
    assert client.create_submission("u1", _FakeResult()) is None
    assert client.submit_feedback("s1", "u1", says_wrong=True) is False
    assert client.pending_review() == []


def test_create_user_raises_on_duplicate(monkeypatch) -> None:
    client = CommunityClient(base_url="https://x", api_key="k")
    monkeypatch.setattr(client, "_request", lambda *a, **k: (409, {"error": "dupe"}))
    with pytest.raises(DuplicateEmailError):
        client.create_user("a@b.c", "pbkdf2_sha256$1$a$b")


def test_get_user_by_email_returns_none_when_absent(monkeypatch) -> None:
    client = CommunityClient(base_url="https://x", api_key="k")
    monkeypatch.setattr(client, "_request", lambda *a, **k: (404, {}))
    assert client.get_user_by_email("nobody@example.com") is None


def test_user_dataclass_matches_database_module() -> None:
    """auth.py imports User from either backend; the fields must line up."""
    import database

    shared = {"user_id", "email", "password_hash", "created_at", "tos_accepted_at"}
    assert shared <= set(User.__dataclass_fields__)
    assert shared <= set(database.User.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Consent is enforced client-side too
# ---------------------------------------------------------------------------
class _FakeResult:
    label = "benign"
    lesion_probability = 0.71
    class_probabilities = {"normal": 0.29, "benign": 0.61, "malignant": 0.10}
    threshold = 0.5
    calibrated = True


def test_image_is_not_sent_when_sharing_is_off(monkeypatch) -> None:
    captured: dict = {}

    def fake(method, path, payload=None, params=None, admin=False):
        captured.update(payload or {})
        return 201, {}

    client = CommunityClient(base_url="https://x", api_key="k")
    monkeypatch.setattr(client, "_request", fake)
    client.create_submission("u1", _FakeResult(), shared=False, image_b64="SECRET")

    assert captured["shared"] is False
    assert "image_b64" not in captured, "an image must never be sent without consent"


def test_image_is_sent_when_sharing_is_on(monkeypatch) -> None:
    captured: dict = {}
    client = CommunityClient(base_url="https://x", api_key="k")
    monkeypatch.setattr(
        client,
        "_request",
        lambda m, p, payload=None, params=None, admin=False: (
            captured.update(payload or {}), (201, {})
        )[1],
    )
    client.create_submission("u1", _FakeResult(), shared=True, image_b64="IMG", image_sha256="abc")
    assert captured["image_b64"] == "IMG"
    assert captured["shared"] is True


def test_rejects_an_invalid_suggested_label() -> None:
    client = CommunityClient(base_url="https://x", api_key="k")
    with pytest.raises(ValueError, match="suggested_label"):
        client.submit_feedback("s1", "u1", says_wrong=True, suggested_label="hotdog")


def test_approving_without_a_label_is_refused_client_side() -> None:
    client = CommunityClient(base_url="https://x", api_key="k", admin_key="a")
    with pytest.raises(ValueError, match="admin_label"):
        client.review_submission("s1", decision="approved")


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def test_round_trips_a_greyscale_image() -> None:
    original = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    encoded, digest, size = encode_image_for_sharing(original)
    assert len(digest) == 64
    assert size == len(encoded) <= MAX_IMAGE_B64_BYTES
    restored = decode_shared_image(encoded)
    assert restored.shape == (256, 256)
    # Monotone ramp survives the uint8 quantisation.
    assert restored[0, 0] < restored[-1, -1]


def test_encodes_a_three_channel_model_input() -> None:
    """The model input is (3, H, W); the channels are one grey plane repeated."""
    encoded, _, _ = encode_image_for_sharing(np.zeros((3, 256, 256), dtype=np.float32))
    assert decode_shared_image(encoded).shape == (256, 256)


def test_a_shared_image_is_small_enough_to_stay_on_the_free_tier() -> None:
    """~30 KB per image is what keeps this in D1 and out of R2 (and off a card)."""
    rng = np.random.default_rng(0)
    encoded, _, size = encode_image_for_sharing(rng.random((256, 256)).astype(np.float32))
    assert size < 200_000, f"{size} bytes is larger than expected for a 256px PNG"


def test_all_black_image_does_not_divide_by_zero() -> None:
    encoded, _, _ = encode_image_for_sharing(np.zeros((256, 256), dtype=np.float32))
    assert decode_shared_image(encoded).max() == 0


def test_rejects_an_image_with_no_finite_pixels() -> None:
    with pytest.raises(CommunityError, match="finite"):
        encode_image_for_sharing(np.full((16, 16), np.nan, dtype=np.float32))


def test_worker_and_client_agree_on_the_image_cap() -> None:
    """A mismatch means uploads rejected server-side after the bytes are spent."""
    worker = (SCHEMA.parent / "src" / "worker.js").read_text(encoding="utf-8")
    for line in worker.splitlines():
        if line.startswith("const MAX_IMAGE_B64_BYTES"):
            value = int(line.split("=")[1].split(";")[0].strip().replace("_", ""))
            assert value == MAX_IMAGE_B64_BYTES
            return
    pytest.fail("MAX_IMAGE_B64_BYTES not found in worker.js")


def test_schema_and_client_agree_on_the_label_set() -> None:
    from community import VALID_LABELS

    schema = SCHEMA.read_text(encoding="utf-8")
    for label in VALID_LABELS:
        assert f"'{label}'" in schema
    assert "'hotdog'" not in schema


def test_class_probabilities_survive_json_encoding() -> None:
    """They cross the wire as a JSON string and come back parsed."""
    payload = json.dumps(_FakeResult.class_probabilities)
    assert json.loads(payload)["malignant"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_backend_defaults_to_local_sqlite(monkeypatch) -> None:
    """Unconfigured, nothing about the existing local behaviour changes."""
    import backend
    import community

    monkeypatch.delenv("ONNM_COMMUNITY_URL", raising=False)
    monkeypatch.delenv("ONNM_COMMUNITY_KEY", raising=False)
    monkeypatch.setattr(community, "_client", None)
    assert backend.using_community() is False
    assert backend.backend_name() == "local SQLite"


def test_backend_switches_when_configured(monkeypatch) -> None:
    """Set the two variables and accounts move to Cloudflare, with no code change."""
    import backend
    import community

    monkeypatch.setenv("ONNM_COMMUNITY_URL", "https://example.workers.dev")
    monkeypatch.setenv("ONNM_COMMUNITY_KEY", "k")
    monkeypatch.setattr(community, "_client", None)
    assert backend.using_community() is True
    assert backend.backend_name() == "Cloudflare D1"


def test_duplicate_email_is_translated_to_one_exception_type(monkeypatch) -> None:
    """auth.py catches database.DuplicateEmailError; the remote backend must
    raise that same type rather than its own, or signup errors escape as 500s."""
    import backend
    import community
    import database

    monkeypatch.setenv("ONNM_COMMUNITY_URL", "https://example.workers.dev")
    monkeypatch.setenv("ONNM_COMMUNITY_KEY", "k")
    client = community.CommunityClient(base_url="https://x", api_key="k")
    monkeypatch.setattr(client, "_request", lambda *a, **k: (409, {"error": "dupe"}))
    monkeypatch.setattr(community, "_client", client)

    with pytest.raises(database.DuplicateEmailError):
        backend.create_user("a@b.c", "pbkdf2_sha256$1$a$b")


def test_auth_imports_from_backend_not_database() -> None:
    """The whole point of the indirection -- if this regresses, the hosted app
    silently writes accounts to a filesystem that is wiped on restart."""
    import auth

    assert auth.create_user.__module__ == "backend"
    assert auth.get_user_by_email.__module__ == "backend"
