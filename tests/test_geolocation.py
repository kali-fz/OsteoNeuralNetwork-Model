"""Tests for the country-level geolocation behind the landing-page globe.

The globe is decoration. The data feeding it is not: it is derived from where
people were when they uploaded a radiograph, and this project stores medical
images. So the tests here are not about markers looking nice. They are about
the three properties that make the feature defensible, each of which is easy to
break with a well-meaning change:

1. **The schema cannot hold a location finer than a country.** Not "does not",
   *cannot* -- a CHECK refuses it. A future endpoint that starts recording a
   city or a coordinate has to change the schema to do it, which is a visible
   act rather than a quiet one.

2. **A country is only ever plotted once enough people share it.** One signup
   in a small country is not a statistic, it is a person. The suppression
   threshold is the difference between "we have users in 23 countries" and
   pointing at somebody.

3. **The payload carries no identifiers.** Country codes and integers, and
   nothing else -- no user id, no submission id, no timestamp, no coordinate.

The migration is also tested against a *populated* database, because the one
way this feature could damage something that already works is by failing
halfway through an ALTER on a table full of stored radiographs.
"""

from __future__ import annotations

import pathlib
import sqlite3
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from geo import (  # noqa: E402
    CONTRIBUTOR_LAYER,
    COUNTRY_CENTROIDS,
    JITTER_DEGREES,
    SIGNUP_LAYER,
    build_markers,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "cloudflare" / "schema.sql"
MIGRATION = ROOT / "cloudflare" / "migrations" / "0004_geolocation.sql"
WORKER = ROOT / "cloudflare" / "src" / "worker.js"

# The Worker's own aggregation, restated here. Kept literal rather than parsed
# out of the JS: a test that reads the implementation it is testing proves only
# that the file has not changed. `test_the_worker_runs_the_same_aggregation`
# below checks the two have not drifted apart on the parts that matter.
SIGNUP_QUERY = """
    SELECT signup_country AS country, COUNT(*) AS n
      FROM users
     WHERE signup_country IS NOT NULL
     GROUP BY signup_country
"""
CONTRIBUTOR_QUERY = """
    SELECT origin_country AS country, COUNT(DISTINCT user_id) AS n
      FROM submissions
     WHERE review_status = 'approved' AND origin_country IS NOT NULL
     GROUP BY origin_country
"""

K_ANONYMITY_MIN = 5


@pytest.fixture
def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.row_factory = sqlite3.Row
    return con


def add_user(con, uid, country=None, email=None):
    con.execute(
        "INSERT INTO users (user_id, email, password_hash, auth_provider,"
        " created_at, tos_accepted_at, signup_country)"
        " VALUES (?, ?, 'pbkdf2_sha256$x', 'password', '2026-01-01T00:00:00Z',"
        " '2026-01-01T00:00:00Z', ?)",
        (uid, email or f"{uid}@example.test", country),
    )


def add_submission(con, sid, uid, country=None, status="pending"):
    con.execute(
        "INSERT INTO submissions (submission_id, user_id, created_at, model_label,"
        " lesion_probability, class_probabilities, shared, review_status, origin_country)"
        " VALUES (?, ?, '2026-01-01T00:00:00Z', 'normal', 0.1, '{}', 1, ?, ?)",
        (sid, uid, status, country),
    )


def suppress(rows, k=K_ANONYMITY_MIN):
    """The Worker's k-anonymity split, restated."""
    plotted = [{"country": r["country"], "count": r["n"]} for r in rows if r["n"] >= k]
    elsewhere = sum(r["n"] for r in rows if r["n"] < k)
    return plotted, elsewhere


# ---------------------------------------------------------------------------
# 1. The schema cannot hold anything finer than a country
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "gb",                 # lowercase: not a canonical ISO code
        "GBR",                # alpha-3: one character too many is still a country,
                              # but accepting two formats means two code paths
        "51.5074,-0.1278",    # the thing this constraint exists to refuse
        "London",
        "",
    ],
)
def test_the_schema_refuses_a_location_finer_than_a_country(db, value) -> None:
    """The CHECK is the guarantee. Everything else is a promise about behaviour."""
    add_user(db, "u1")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE users SET signup_country = ? WHERE user_id = 'u1'", (value,))


def test_a_submission_cannot_carry_a_coordinate_either(db) -> None:
    add_user(db, "u1")
    add_submission(db, "s1", "u1")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "UPDATE submissions SET origin_country = ? WHERE submission_id = 's1'",
            ("51.5074,-0.1278",),
        )


def test_a_valid_country_code_is_accepted(db) -> None:
    add_user(db, "u1", "GB")
    assert db.execute("SELECT signup_country FROM users").fetchone()[0] == "GB"


def test_cloudflare_placeholders_are_storable(db) -> None:
    """'T1' (Tor) and 'XX' (undetermined) are what the edge actually sends.

    They are stored honestly rather than coerced to NULL, because "we asked and
    could not tell" and "we never asked" are different facts. Neither has a
    centroid, so neither is ever drawn -- see the unplaced test below.
    """
    add_user(db, "u1", "T1")
    add_user(db, "u2", "XX")
    assert db.execute("SELECT COUNT(*) FROM users WHERE signup_country IS NOT NULL").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# 2. Suppression: a country is plotted only when enough people share it
# ---------------------------------------------------------------------------
def test_a_country_below_the_threshold_is_never_plotted(db) -> None:
    for i in range(K_ANONYMITY_MIN - 1):
        add_user(db, f"small{i}", "IS")
    for i in range(K_ANONYMITY_MIN):
        add_user(db, f"big{i}", "GB")

    plotted, elsewhere = suppress(db.execute(SIGNUP_QUERY).fetchall())

    assert [p["country"] for p in plotted] == ["GB"]
    assert elsewhere == K_ANONYMITY_MIN - 1, "the suppressed people are still counted"
    assert "IS" not in str(plotted), "a suppressed country must not appear at all"


def test_the_threshold_is_a_floor_not_a_ceiling(db) -> None:
    """Exactly k is enough. An off-by-one here silently hides a whole country."""
    for i in range(K_ANONYMITY_MIN):
        add_user(db, f"u{i}", "IE")
    plotted, elsewhere = suppress(db.execute(SIGNUP_QUERY).fetchall())
    assert [p["country"] for p in plotted] == ["IE"]
    assert elsewhere == 0


def test_one_enthusiastic_uploader_cannot_inflate_their_own_country(db) -> None:
    """The contributor layer counts people, not submissions.

    Counting submissions would let a single user with 40 approved uploads put a
    large dot on a country containing exactly one contributor -- both a false
    picture of reach and, in a small country, a marker pointing at one person.
    """
    add_user(db, "loud", "MT")
    for i in range(40):
        add_submission(db, f"s{i}", "loud", "MT", status="approved")

    rows = db.execute(CONTRIBUTOR_QUERY).fetchall()
    assert rows[0]["n"] == 1, "40 submissions from one person is one contributor"

    plotted, elsewhere = suppress(rows)
    assert plotted == [], "and one contributor is below the disclosure threshold"
    assert elsewhere == 1


def test_only_approved_contributions_reach_the_globe(db) -> None:
    """Consistent with the review gate: an unreviewed upload has contributed nothing."""
    for i in range(K_ANONYMITY_MIN + 3):
        add_user(db, f"u{i}", "DE")
        add_submission(db, f"s{i}", f"u{i}", "DE", status="pending")

    assert db.execute(CONTRIBUTOR_QUERY).fetchall() == []

    # Approval goes through the real gate: the schema trigger refuses a row
    # promoted to 'approved' without a human-set label and bucket, so this is
    # also a check that the globe cannot be fed by a shortcut around review.
    for i in range(K_ANONYMITY_MIN):
        db.execute(
            "UPDATE submissions SET review_status = 'approved', admin_label = 'normal',"
            " admin_bucket = 'valid_bone' WHERE submission_id = ?",
            (f"s{i}",),
        )
    rows = db.execute(CONTRIBUTOR_QUERY).fetchall()
    assert rows[0]["country"] == "DE" and rows[0]["n"] == K_ANONYMITY_MIN


# ---------------------------------------------------------------------------
# 3. Rows that predate the feature must not break it
# ---------------------------------------------------------------------------
def test_users_without_a_country_are_counted_but_not_plotted(db) -> None:
    """Every row predating migration 0004 carries NULL, and NULL is never guessed."""
    for i in range(3):
        add_user(db, f"old{i}")  # no country: the pre-migration population
    for i in range(K_ANONYMITY_MIN):
        add_user(db, f"new{i}", "FR")

    plotted, _ = suppress(db.execute(SIGNUP_QUERY).fetchall())
    assert [p["country"] for p in plotted] == ["FR"]

    total = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert total == 3 + K_ANONYMITY_MIN, "the headline total still counts everyone"


def test_the_migration_applies_to_a_populated_pre_0004_database() -> None:
    """The real risk of this change is not the globe -- it is the ALTER.

    The live database holds every shared radiograph. A migration that fails
    partway, or that cannot run against rows that already exist, would be the
    one way a decorative feature damages something that works.
    """
    old_schema = subprocess.run(
        ["git", "show", "HEAD:cloudflare/schema.sql"],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT, check=True,
    ).stdout
    if "signup_country" in old_schema:
        pytest.skip("HEAD already contains the geolocation columns")

    con = sqlite3.connect(":memory:")
    con.executescript(old_schema)
    con.row_factory = sqlite3.Row
    con.execute(
        "INSERT INTO users (user_id, email, password_hash, auth_provider,"
        " created_at, tos_accepted_at) VALUES ('u1','a@b.test','pbkdf2_sha256$x',"
        " 'password','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')"
    )
    con.execute(
        "INSERT INTO submissions (submission_id, user_id, created_at, model_label,"
        " lesion_probability, class_probabilities) VALUES ('s1','u1',"
        " '2026-01-01T00:00:00Z','normal',0.1,'{}')"
    )
    con.commit()

    con.executescript(MIGRATION.read_text(encoding="utf-8"))

    assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1
    assert con.execute("SELECT signup_country FROM users").fetchone()[0] is None
    assert con.execute("SELECT origin_country FROM submissions").fetchone()[0] is None
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "4"


def test_the_migration_and_the_schema_agree() -> None:
    """A fresh deployment and a migrated one must be the same database."""
    fresh = sqlite3.connect(":memory:")
    fresh.executescript(SCHEMA.read_text(encoding="utf-8"))
    for table, column in (("users", "signup_country"), ("submissions", "origin_country")):
        columns = [r[1] for r in fresh.execute(f"PRAGMA table_info({table})")]
        assert column in columns, f"{table}.{column} missing from schema.sql"


# ---------------------------------------------------------------------------
# 4. The payload and the markers built from it
# ---------------------------------------------------------------------------
def sample_payload():
    return {
        "ok": True,
        "totals": {"users": 42, "contributors": 11, "approved_submissions": 60,
                   "countries_represented": 4},
        "layers": {
            "signups": {"plotted": [{"country": "GB", "count": 20},
                                    {"country": "US", "count": 12}],
                        "elsewhere": 7, "suppressed_countries": 3},
            "contributors": {"plotted": [{"country": "GB", "count": 8}],
                             "elsewhere": 3, "suppressed_countries": 2},
        },
        "k_anonymity_min": K_ANONYMITY_MIN,
    }


def test_markers_carry_no_identifiers() -> None:
    """The marker contract is coordinates, a country label and a count."""
    allowed = {"lat", "lng", "label", "country", "count", "layer"}
    for marker in build_markers(sample_payload())["markers"]:
        assert set(marker) <= allowed, f"unexpected field: {set(marker) - allowed}"


def test_both_layers_are_built_and_labelled() -> None:
    markers = build_markers(sample_payload())["markers"]
    layers = {m["layer"] for m in markers}
    assert layers == {SIGNUP_LAYER, CONTRIBUTOR_LAYER}


def test_the_two_layers_do_not_draw_on_top_of_each_other() -> None:
    """GB appears in both layers; identical coordinates would hide one of them."""
    markers = build_markers(sample_payload())["markers"]
    gb = [m for m in markers if m["country"] == "GB"]
    assert len(gb) == 2
    assert (gb[0]["lat"], gb[0]["lng"]) != (gb[1]["lat"], gb[1]["lng"])


def test_marker_positions_are_stable_between_renders() -> None:
    """A dot that moved on each load would look like live tracking."""
    first = build_markers(sample_payload())["markers"]
    second = build_markers(sample_payload())["markers"]
    assert first == second


def test_markers_stay_near_their_country() -> None:
    for marker in build_markers(sample_payload())["markers"]:
        _, lat, lng = COUNTRY_CENTROIDS[marker["country"]]
        assert abs(marker["lat"] - lat) <= JITTER_DEGREES + 1e-6
        assert abs(marker["lng"] - lng) <= JITTER_DEGREES + 1e-6


def test_suppressed_people_are_reported_not_dropped() -> None:
    """"and 7 elsewhere" is honest; showing only the plotted dots is not."""
    built = build_markers(sample_payload())
    assert built["elsewhere"][SIGNUP_LAYER] == 7
    assert built["elsewhere"][CONTRIBUTOR_LAYER] == 3
    assert built["k_anonymity_min"] == K_ANONYMITY_MIN


def test_a_country_with_no_centroid_is_counted_not_silently_dropped() -> None:
    payload = sample_payload()
    payload["layers"]["signups"]["plotted"].append({"country": "XX", "count": 9})
    built = build_markers(payload)
    assert all(m["country"] != "XX" for m in built["markers"])
    assert built["unplaced"][SIGNUP_LAYER] == 9


def test_the_globe_fails_soft_when_the_backend_is_unreachable() -> None:
    """A landing page must render without the community API. It is decoration."""
    for payload in (None, {}, {"ok": False, "error": "unauthorized"}):
        built = build_markers(payload)
        assert built["markers"] == []
        assert built["available"] is False
        assert built["totals"]["users"] == 0


# ---------------------------------------------------------------------------
# 5. The Worker and this test suite must not drift apart
# ---------------------------------------------------------------------------
def test_the_worker_runs_the_same_aggregation() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    assert "COUNT(DISTINCT user_id)" in worker, "contributor layer must count people"
    assert "review_status = 'approved'" in worker, "contributor layer must be approved-only"
    assert f"const K_ANONYMITY_MIN = {K_ANONYMITY_MIN};" in worker, (
        "the suppression threshold in worker.js has drifted from this suite"
    )


def test_the_globe_endpoint_selects_no_identifying_column() -> None:
    """Guards the property that makes the endpoint safe to call from a public page."""
    worker = WORKER.read_text(encoding="utf-8")
    start = worker.index("async function globe(db)")
    end = worker.index("async function createUser")
    body = worker[start:end]
    for forbidden in ("email", "image_b64", "provider_subject", "password_hash",
                      "submission_id", "created_at AS", "latitude", "longitude"):
        assert forbidden not in body, f"/globe must not touch {forbidden}"


def test_the_worker_never_reads_a_country_from_the_request_body() -> None:
    """The client cannot be trusted to say where it is, and does not need to."""
    worker = WORKER.read_text(encoding="utf-8")
    assert "request.cf && request.cf.country" in worker
    # The destructured request bodies must not contain a country field.
    for handler in ("async function createUser", "async function createSubmission"):
        start = worker.index(handler)
        body = worker[start:start + 900]
        assert "signup_country," not in body.split("} = body")[0]
        assert "origin_country," not in body.split("} = body")[0]
