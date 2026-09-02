"""Public contributor profiles stay opt-in and approved-only."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import community  # noqa: E402


def test_profile_migration_keeps_existing_accounts_private() -> None:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE users (user_id TEXT PRIMARY KEY, email TEXT NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO users VALUES ('u1', 'person@example.com');
        INSERT INTO meta VALUES ('schema_version', '4');
        """
    )
    migration = (
        ROOT / "cloudflare" / "migrations" / "0005_public_contributor_profiles.sql"
    ).read_text(encoding="utf-8")
    db.executescript(migration)

    row = db.execute(
        "SELECT display_name, profile_picture_url, public_contributor_profile FROM users"
    ).fetchone()
    assert row == (None, None, 0)
    assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "5"


def test_worker_contributor_roll_is_opt_in_and_approved_only() -> None:
    worker = (ROOT / "cloudflare" / "src" / "worker.js").read_text(encoding="utf-8")
    start = worker.index("async function listContributors")
    end = worker.index("async function createSubmission")
    handler = worker[start:end]

    assert "u.public_contributor_profile = 1" in handler
    assert "s.review_status = 'approved'" in handler
    assert "COUNT(s.submission_id) AS approved_contributions" in handler
    assert "u.email" not in handler


def test_existing_approved_rows_gain_a_country_after_browser_capture() -> None:
    worker = (ROOT / "cloudflare" / "src" / "worker.js").read_text(encoding="utf-8")
    globe = worker[worker.index("async function globe"):worker.index("async function createUser")]
    capture = worker[
        worker.index("async function captureBrowserCountry"):
        worker.index("function cleanDisplayName")
    ]

    assert "COALESCE(s.origin_country, u.signup_country)" in globe
    assert "SET signup_country = ?, country_captured_at = ?" in capture
    assert "UPDATE submissions SET origin_country = ?" in capture
    assert "country_captured_at = ?" in capture
    assert 'path === "/location/capture"' in worker


def test_client_profile_toggle_sends_google_identity(monkeypatch) -> None:
    sent: dict = {}

    def fake_request(self, method, path, payload=None, params=None, admin=False):
        sent.update(method=method, path=path, payload=payload)
        return 200, {"ok": True}

    monkeypatch.setattr(community.CommunityClient, "_request", fake_request)
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")

    assert client.update_contributor_profile(
        "u1",
        "google-subject",
        display_name="Ada Lovelace",
        profile_picture_url="https://lh3.googleusercontent.com/photo",
        public_profile=True,
    )
    assert sent == {
        "method": "POST",
        "path": "/users/profile",
        "payload": {
            "user_id": "u1",
            "provider_subject": "google-subject",
            "display_name": "Ada Lovelace",
            "profile_picture_url": "https://lh3.googleusercontent.com/photo",
            "public_profile": True,
        },
    }


def test_client_contributor_list_reports_unavailable(monkeypatch) -> None:
    client = community.CommunityClient(base_url="https://example.invalid", api_key="k")
    monkeypatch.setattr(
        community.CommunityClient,
        "_request",
        lambda *_a, **_k: (200, {"contributors": [{"name": "Ada"}]}),
    )
    assert client.contributors() == [{"name": "Ada"}]

    monkeypatch.setattr(
        community.CommunityClient,
        "_request",
        lambda *_a, **_k: (503, {"error": "offline"}),
    )
    assert client.contributors() is None


def _site_worker() -> str:
    return (ROOT / "worker" / "index.js").read_text(encoding="utf-8")


def _visibility_handler() -> str:
    worker = _site_worker()
    start = worker.index("async function setProfileVisibility")
    return worker[start : worker.index("// Scanning", start)]


def test_visibility_route_exists_and_requires_a_session() -> None:
    """Anonymous callers must not be able to publish or unpublish anybody."""
    worker = _site_worker()
    start = worker.index('path === "/api/profile/visibility"')
    route = worker[start : start + 220]

    assert 'method === "POST"' in worker[start - 60 : start]
    assert 'fail(401, "sign in first")' in route
    assert "setProfileVisibility(request, env, session)" in route


def test_visibility_takes_identity_from_the_session_not_the_request_body() -> None:
    """A body-supplied user_id would let anyone rewrite someone else's profile.

    The storage Worker re-checks the Google subject as well, so this is the
    second of two locks; it is asserted here because the first one is the only
    thing standing between a signed-in user and every other account's listing.
    """
    handler = _visibility_handler()

    assert "user_id: session.uid" in handler
    assert "provider_subject: session.sub" in handler
    # The only value taken from the request is the flag itself.
    assert "body.user_id" not in handler
    assert "body.provider_subject" not in handler


def test_visibility_requires_an_explicit_boolean() -> None:
    """A missing or misspelled field must not read as "make me private".

    Truthiness would turn a typo into a silent unpublish, which looks identical
    to the user asking for it.
    """
    handler = _visibility_handler()

    assert 'typeof body?.public_profile !== "boolean"' in handler
    assert 'fail(400, "public_profile must be true or false")' in handler


def test_session_reports_visibility_from_the_account_row() -> None:
    """The cookie lasts eight hours; the answer must not.

    Carrying this in the session would keep somebody listed after they asked to
    be removed, for as long as their cookie survived.
    """
    worker = _site_worker()
    start = worker.index('path === "/api/session"')
    handler = worker[start : worker.index('path === "/api/terms/accept"', start)]

    assert "public_profile: account?.public_contributor_profile === 1" in handler
    assert "accountFor(request, env, session)" in handler


def test_opting_out_clears_the_stored_name_and_photo() -> None:
    """Turning the toggle off deletes the data, rather than only hiding it.

    The Privacy notice offers this as a withdrawal of consent, so leaving the
    name and picture in the row would make the promise untrue.
    """
    worker = (ROOT / "cloudflare" / "src" / "worker.js").read_text(encoding="utf-8")
    start = worker.index("async function updateContributorProfile")
    handler = worker[start : worker.index("async function listContributors", start)]

    assert "const storedName = publicProfile === 0 ? null : cleanDisplayName(display_name);" in handler
    assert (
        "const storedPicture = publicProfile === 0 ? null : cleanGooglePicture(profile_picture_url);"
        in handler
    )


def test_the_checkbox_defaults_to_unchecked() -> None:
    """Opt-in is the whole point: the box must never start ticked by default."""
    page = (ROOT / "web" / "src" / "pages" / "profile.js").read_text(encoding="utf-8")

    assert "state.session?.public_profile === true" in page
    assert '${isPublic ? "checked" : ""}' in page
    # A failed save must not leave the box showing a state the server rejected.
    assert "visibility.checked = !wanted;" in page
