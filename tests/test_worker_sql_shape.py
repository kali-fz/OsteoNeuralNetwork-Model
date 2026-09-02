"""Every INSERT in the community Worker binds as many values as it names columns.

**The regression.** New Google sign-ups failed with an opaque
``Your account could not be opened. Please try again.`` while every existing
account signed in normally. The cause was one character of SQL: the commit that
recorded which version of the Terms an account agreed to added ``tos_version``
to the column list of ``createUser``'s INSERT, and added the matching value to
``.bind()``, but left the ``VALUES (...)`` list at eleven placeholders. SQLite
rejects the statement with "11 values for 12 columns", which is not a UNIQUE
violation, so it escaped the one ``catch`` in that function and surfaced as a
500.

Nothing caught it because nothing ran it. The JavaScript tests stub
``createUser`` at the resolver boundary and the Python tests write their own
INSERT statements, so the Worker's real statement was never executed by a test
in any suite. A column/placeholder mismatch is invisible to review precisely
because the three lists are far apart on the screen and each one looks right on
its own.

So the shape is asserted directly against the source, for every INSERT in the
Worker rather than only the one that broke, and ``createUser``'s statement is
additionally executed against the real ``schema.sql``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "cloudflare" / "src" / "worker.js"
SCHEMA = ROOT / "cloudflare" / "schema.sql"

# `INSERT INTO <table> (<columns>) VALUES (<placeholders>)`, across the line
# breaks the Worker's template literals wrap these in. Statements whose column
# list is implicit, or whose values come from a generated list rather than a
# literal one, are not matched and are reported separately below.
INSERT = re.compile(
    r"INSERT\s+INTO\s+(?P<table>\w+)\s*\((?P<columns>[^)]*)\)\s*"
    r"VALUES\s*\((?P<values>[^)]*)\)",
    re.S | re.I,
)


def worker_source() -> str:
    return WORKER.read_text(encoding="utf-8")


def parsed_inserts():
    return list(INSERT.finditer(worker_source()))


def split_terms(raw: str) -> list[str]:
    """Split a column or value list on its commas.

    Value lists are counted as terms rather than as ``?`` characters, because a
    literal is a perfectly good value: ``rate_limit`` seeds its counter with
    ``VALUES (?, ?, 1)`` and binds two parameters for three columns quite
    correctly. Counting placeholders alone would report that as a defect.
    """
    return [term.strip() for term in raw.split(",") if term.strip()]


@pytest.mark.parametrize("match", parsed_inserts(), ids=lambda m: m.group("table"))
def test_insert_binds_one_value_per_column(match) -> None:
    """A named column list and its value list must be the same length."""
    columns = split_terms(match.group("columns"))
    values = split_terms(match.group("values"))

    assert len(values) == len(columns), (
        f"INSERT INTO {match.group('table')} names {len(columns)} columns "
        f"but supplies {len(values)} values: {columns}"
    )


def test_every_insert_in_the_worker_was_checked() -> None:
    """The regex must not silently stop covering a statement it used to cover.

    A parser that matches nothing passes vacuously, which is the failure mode
    that would let this bug come back. Pinning the count means reformatting a
    statement out of the regex's reach breaks this test rather than quietly
    reducing coverage.
    """
    assert len(parsed_inserts()) == worker_source().count("INSERT INTO") == 5


def users_table_ddl() -> str:
    schema = SCHEMA.read_text(encoding="utf-8")
    match = re.search(r"CREATE TABLE[^;]*?\busers\b[^;]*?\(.*?\n\);", schema, re.S)
    assert match, "users table not found in schema.sql"
    return match.group(0)


def test_create_user_statement_executes_against_the_real_schema() -> None:
    """The statement the Worker actually sends, run against the real table.

    The shape test above would pass on a statement that named a column the
    table does not have, so the statement is executed here as well.
    """
    statement = next(
        match for match in parsed_inserts() if match.group("table") == "users"
    )
    columns = split_terms(statement.group("columns"))

    db = sqlite3.connect(":memory:")
    db.executescript(users_table_ddl())
    db.execute(
        f"INSERT INTO users ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})",
        (
            "user-1",
            "person@example.com",
            None,  # password_hash: a Google account carries none
            "google",
            "google-subject-1",
            "2026-09-02T00:00:00Z",
            "2026-09-02T00:00:00Z",
            "2026-08-30",
            0,
            None,  # signup_country, filled in later by /location/capture
            "A Person",
            None,
        ),
    )

    stored = db.execute("SELECT tos_version FROM users WHERE user_id = 'user-1'").fetchone()
    assert stored == ("2026-08-30",)
