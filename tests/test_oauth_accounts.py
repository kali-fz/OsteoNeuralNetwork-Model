"""Google Sign-In: the account model, and the bug that made accounts fail.

Two things are pinned here.

**The regression.** Account creation on the hosted app failed with an opaque
``CommunityError``. The cause was not in the Worker, the token or the schema:
Cloudflare's edge bans the default ``Python-urllib/3.x`` User-Agent with HTTP
403 and a plain-text "error code: 1010" body, so the request never arrived.
Nothing in a normal test would catch that -- the client was correct, the server
was correct -- so the header itself is asserted.

**The account model.** A federated account must have no password, and a
password account must have no provider subject. If those can ever be mixed, an
account becomes reachable by two different proofs of identity, which is an
authentication bypass rather than a data-quality problem. The constraint is
therefore asserted against the real ``schema.sql`` rather than trusted to the
Worker that writes through it.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = ROOT / "cloudflare" / "schema.sql"

import auth  # noqa: E402
import backend  # noqa: E402
import community  # noqa: E402
import database  # noqa: E402


# ---------------------------------------------------------------------------
# The regression: the User-Agent Cloudflare's edge will accept
# ---------------------------------------------------------------------------
class _FakeResponse:
    status = 200

    def read(self):
        return b'{"ok": true}'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_requests_send_an_explicit_user_agent(monkeypatch) -> None:
    """The default urllib agent is refused at the edge before the Worker sees it."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["ua"] = request.get_header("User-agent")
        return _FakeResponse()

    monkeypatch.setattr(community.urllib.request, "urlopen", fake_urlopen)
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    client.health()

    assert captured["ua"], "no User-Agent was sent"
    assert "Python-urllib" not in captured["ua"]
    assert captured["ua"] == community.USER_AGENT


def test_a_non_json_error_body_names_the_gateway(monkeypatch) -> None:
    """"error code: 1010" alone sends you debugging the Worker, wrongly."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.invalid/health", 403, "Forbidden", {}, None
        )

    monkeypatch.setattr(community.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        urllib.error.HTTPError, "read", lambda self: b"error code: 1010\n", raising=False
    )
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    status, body = client._request("GET", "/health")

    assert status == 403
    assert "gateway refused" in body["error"]
    assert "1010" in body["error"]


# ---------------------------------------------------------------------------
# The account model, against the real D1 schema
# ---------------------------------------------------------------------------
@pytest.fixture
def d1() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    return con


def _insert(con, **kw):
    row = {
        "user_id": "u1",
        "email": "a@b.c",
        "password_hash": None,
        "auth_provider": "password",
        "provider_subject": None,
        "created_at": "t",
        "tos_accepted_at": "t",
        **kw,
    }
    con.execute(
        """INSERT INTO users
             (user_id, email, password_hash, auth_provider, provider_subject,
              created_at, tos_accepted_at)
           VALUES (:user_id, :email, :password_hash, :auth_provider,
                   :provider_subject, :created_at, :tos_accepted_at)""",
        row,
    )


def test_a_password_account_needs_a_hash(d1) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(d1, auth_provider="password", password_hash=None)


def test_a_google_account_must_not_carry_a_password(d1) -> None:
    """The bypass this guards: one account, two ways to prove you are its owner."""
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            d1,
            auth_provider="google",
            provider_subject="sub-1",
            password_hash="pbkdf2_sha256$1$aa$bb",
        )


def test_a_google_account_needs_a_subject(d1) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(d1, auth_provider="google", provider_subject=None)


def test_a_password_account_must_not_carry_a_subject(d1) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            d1,
            auth_provider="password",
            password_hash="pbkdf2_sha256$1$aa$bb",
            provider_subject="sub-1",
        )


def test_unknown_providers_are_refused(d1) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(d1, auth_provider="facebook", provider_subject="sub-1")


def test_one_account_per_google_identity(d1) -> None:
    _insert(d1, user_id="u1", email="a@b.c", auth_provider="google", provider_subject="sub-1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(d1, user_id="u2", email="d@e.f", auth_provider="google", provider_subject="sub-1")


def test_password_accounts_do_not_collide_on_null_subject(d1) -> None:
    """The partial unique index must not treat two NULLs as a duplicate."""
    _insert(d1, user_id="u1", email="a@b.c", password_hash="pbkdf2_sha256$1$aa$bb")
    _insert(d1, user_id="u2", email="d@e.f", password_hash="pbkdf2_sha256$1$cc$dd")
    assert d1.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Local SQLite backend
# ---------------------------------------------------------------------------
@pytest.fixture
def local_db(tmp_path) -> Path:
    path = tmp_path / "users.db"
    database.initialize_database(path)
    return path


def test_oauth_user_round_trips(local_db) -> None:
    created = database.create_oauth_user("a@b.c", "sub-1", path=local_db)
    assert created.password_hash is None
    assert created.auth_provider == "google"

    found = database.get_user_by_subject("sub-1", path=local_db)
    assert found is not None
    assert found.user_id == created.user_id
    assert found.password_hash is None


def test_unknown_subject_is_none(local_db) -> None:
    assert database.get_user_by_subject("nobody", path=local_db) is None


def test_the_migration_preserves_existing_password_accounts(tmp_path) -> None:
    """Anyone who already has an account must still have it afterwards."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tos_accepted_at TEXT NOT NULL
        );
        INSERT INTO users VALUES ('u1', 'old@b.c', 'pbkdf2_sha256$1$aa$bb', 't0', 't0');
        """
    )
    con.commit()
    con.close()

    database.initialize_database(path)

    survivor = database.get_user_by_email("old@b.c", path=path)
    assert survivor is not None
    assert survivor.user_id == "u1"
    assert survivor.password_hash == "pbkdf2_sha256$1$aa$bb"
    assert survivor.auth_provider == "password"
    assert survivor.provider_subject is None

    # And the migrated table now accepts a federated account.
    database.create_oauth_user("new@b.c", "sub-1", path=path)
    assert database.get_user_by_subject("sub-1", path=path) is not None


def test_the_migration_is_idempotent(local_db) -> None:
    database.create_oauth_user("a@b.c", "sub-1", path=local_db)
    database.initialize_database(local_db)
    database.initialize_database(local_db)
    assert database.get_user_by_subject("sub-1", path=local_db) is not None


# ---------------------------------------------------------------------------
# Password login must not reach a federated account
# ---------------------------------------------------------------------------
def test_verify_password_refuses_a_null_hash() -> None:
    assert auth.verify_password("anything at all", None) is False


def test_a_google_account_cannot_be_password_logged_in(local_db, monkeypatch) -> None:
    """The whole point of storing NULL: there is no password to guess."""
    monkeypatch.setenv("ONNM_DATABASE_PATH", str(local_db))
    database.create_oauth_user("a@b.c", "sub-1", path=local_db)

    for attempt in ("", "password12345", "sub-1", "None"):
        assert auth.authenticate_user("a@b.c", attempt, database=local_db) is None


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------
@pytest.fixture
def local_backend(local_db, monkeypatch):
    """Force backend dispatch to local SQLite, not Cloudflare."""
    monkeypatch.setattr(backend, "using_community", lambda: False)
    return local_db


def test_the_account_is_created_once(local_backend) -> None:
    first = backend.get_or_create_oauth_user("a@b.c", "sub-1", path=local_backend)
    second = backend.get_or_create_oauth_user("a@b.c", "sub-1", path=local_backend)
    assert first.user_id == second.user_id


def test_identity_follows_the_subject_not_the_email(local_backend) -> None:
    """Someone who changes their Google address keeps their scan history."""
    first = backend.get_or_create_oauth_user("old@b.c", "sub-1", path=local_backend)
    renamed = backend.get_or_create_oauth_user("new@b.c", "sub-1", path=local_backend)
    assert renamed.user_id == first.user_id
    assert renamed.email == "old@b.c"  # the stored row is not rewritten


def test_a_password_account_is_not_silently_converted(local_backend) -> None:
    """Proving control of an address must not take over a password account."""
    existing = database.create_user(
        "a@b.c", "pbkdf2_sha256$1$aa$bb", path=local_backend
    )
    resolved = backend.get_or_create_oauth_user("a@b.c", "sub-1", path=local_backend)

    assert resolved.user_id == existing.user_id
    assert resolved.auth_provider == "password"
    assert resolved.provider_subject is None


# ---------------------------------------------------------------------------
# The client sends what the Worker expects
# ---------------------------------------------------------------------------
def test_create_oauth_user_sends_no_password_hash(monkeypatch) -> None:
    sent = {}

    def fake_request(self, method, path, payload=None, params=None, admin=False):
        sent.update({"method": method, "path": path, "payload": payload})
        return 201, {"created_at": "t"}

    monkeypatch.setattr(community.CommunityClient, "_request", fake_request)
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    user = client.create_oauth_user("a@b.c", "sub-1")

    assert sent["path"] == "/users"
    assert "password_hash" not in sent["payload"]
    assert sent["payload"]["auth_provider"] == "google"
    assert sent["payload"]["provider_subject"] == "sub-1"
    assert user.password_hash is None


def test_duplicate_email_is_translated(monkeypatch) -> None:
    monkeypatch.setattr(
        community.CommunityClient,
        "_request",
        lambda *a, **k: (409, {"error": "exists"}),
    )
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    with pytest.raises(community.DuplicateEmailError):
        client.create_oauth_user("a@b.c", "sub-1")


def test_worker_json_shape_matches_the_client(monkeypatch) -> None:
    """A Worker row deserialises into the client's User without loss."""
    row = {
        "user_id": "u1",
        "email": "a@b.c",
        "password_hash": None,
        "auth_provider": "google",
        "provider_subject": "sub-1",
        "created_at": "t",
        "tos_accepted_at": "t",
        "is_admin": 0,
    }
    monkeypatch.setattr(
        community.CommunityClient, "_request", lambda *a, **k: (200, json.loads(json.dumps(row)))
    )
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    user = client.get_user_by_subject("sub-1")

    assert user is not None
    assert user.auth_provider == "google"
    assert user.provider_subject == "sub-1"
    assert user.password_hash is None
