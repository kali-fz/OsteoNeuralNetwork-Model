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
