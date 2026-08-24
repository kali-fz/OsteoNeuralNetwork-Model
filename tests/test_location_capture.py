"""Guards for privacy-safe browser-edge country capture."""

from __future__ import annotations

import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORKER = (ROOT / "cloudflare" / "src" / "worker.js").read_text(encoding="utf-8")
SCHEMA = ROOT / "cloudflare" / "schema.sql"
MIGRATION = ROOT / "cloudflare" / "migrations" / "0006_browser_country_capture.sql"


def test_capture_component_exposes_only_the_one_use_token() -> None:
    from components.location_capture import _capture_html

    html = _capture_html("https://community.example", "one-use-token")
    assert "https://community.example/location/capture" in html
    assert "one-use-token" in html
    assert "ONNM_COMMUNITY_KEY" not in html
    assert "API_KEY" not in html
    assert "email" not in html.lower()
    assert "geolocation" not in html.lower()
    assert "latitude" not in html.lower()
    assert "longitude" not in html.lower()


def test_client_mints_location_token_server_side(monkeypatch) -> None:
    from community import CommunityClient

    sent = {}

    def fake_request(method, path, payload=None, params=None, admin=False):
        sent.update(method=method, path=path, payload=payload, admin=admin)
        return 200, {"ok": True, "token": "opaque", "expires_at": "soon"}

    client = CommunityClient("https://community.example", "private-app-key")
    monkeypatch.setattr(client, "_request", fake_request)
    result = client.location_token("u1")

    assert result and result["token"] == "opaque"
    assert sent == {
        "method": "POST",
        "path": "/location/token",
        "payload": {"user_id": "u1"},
        "admin": False,
    }


def test_worker_never_accepts_a_client_claimed_country() -> None:
    capture = WORKER[
        WORKER.index("async function captureBrowserCountry"):
        WORKER.index("function cleanDisplayName")
    ]
    assert "const country = countryOf(request)" in capture
    assert "await readJson(request)" not in capture
    assert "body.country" not in capture
    assert "request.cf && request.cf.country" in WORKER


def test_capture_is_one_use_and_can_repair_old_server_country() -> None:
    capture = WORKER[
        WORKER.index("async function captureBrowserCountry"):
        WORKER.index("function cleanDisplayName")
    ]
    assert "used_at IS NULL AND expires_at >= ?" in capture
    assert "country_captured_at IS NULL" in capture
    assert "UPDATE submissions SET origin_country = ?" in capture
    assert "const results = await db.batch([" in capture
    assert "results[0].meta.changes !== 1" in capture
    assert 'country === "XX"' in capture
    assert 'country === "T1"' in capture


def test_capture_schema_is_country_only_and_migration_reaches_v6() -> None:
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    user_columns = {row[1] for row in con.execute("PRAGMA table_info(users)")}
    token_columns = {
        row[1] for row in con.execute("PRAGMA table_info(location_capture_tokens)")
    }
    schema_source = SCHEMA.read_text(encoding="utf-8")
    token_table = schema_source.split(
        "CREATE TABLE IF NOT EXISTS location_capture_tokens", 1
    )[1].split(");", 1)[0]
    assert "country_captured_at" in user_columns
    assert token_columns == {
        "token_hash", "user_id", "expires_at", "used_at", "used_nonce"
    }
    assert "latitude" not in token_table.lower()
    assert "longitude" not in token_table.lower()
    assert "schema_version', '6" in schema_source
    assert "schema_version = '6'" not in MIGRATION.read_text(encoding="utf-8")
    assert "SET value = '6'" in MIGRATION.read_text(encoding="utf-8")


def test_v6_migration_preserves_existing_users_and_submissions() -> None:
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            signup_country TEXT
        );
        CREATE TABLE submissions (
            submission_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            origin_country TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO users VALUES ('u1', 'US');
        INSERT INTO submissions VALUES ('s1', 'u1', 'US');
        INSERT INTO meta VALUES ('schema_version', '5');
        """
    )
    con.executescript(MIGRATION.read_text(encoding="utf-8"))

    assert con.execute("SELECT signup_country FROM users").fetchone()[0] == "US"
    assert con.execute("SELECT origin_country FROM submissions").fetchone()[0] == "US"
    assert con.execute("SELECT country_captured_at FROM users").fetchone()[0] is None
    assert con.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()[0] == "6"
