"""The suite must never touch the production backend.

``backend.create_user`` dispatches on ``using_community()``. When the community
client is configured it calls Cloudflare and discards the ``path`` argument, so
a test that passes a throwaway ``tmp_path`` database silently registers a real
account in live D1 instead. Three ``@example.com`` accounts reached production
that way before it was noticed.

``tests/conftest.py`` installs an autouse fixture that clears the three
environment variables and resets the memoised client. These tests exist so that
if the fixture is ever removed, renamed, or defeated by an import-order change,
something fails loudly here rather than quietly writing to production.
"""

from __future__ import annotations

import os

import backend
import community
from auth import register_user
from conftest import COMMUNITY_ENV_VARS


def test_community_environment_is_cleared() -> None:
    """No community variable survives into a test process."""
    for name in COMMUNITY_ENV_VARS:
        assert os.environ.get(name) is None, (
            f"{name} is set during tests; the suite would write to production"
        )


def test_backend_dispatches_to_local_sqlite() -> None:
    """The dispatch itself, not just the environment it reads."""
    assert backend.using_community() is False
    assert backend.backend_name() == "local SQLite"


def test_memoised_client_is_not_configured() -> None:
    """``get_client`` caches process-wide, so the cache must be clear too."""
    assert community.get_client().enabled is False


def test_register_user_writes_to_the_database_it_was_given(tmp_path) -> None:
    """The property that was actually violated.

    The temp file existing afterwards is the whole assertion: under the
    community backend it was never created, because ``path`` was ignored.
    """
    database = tmp_path / "users.db"
    assert not database.exists()

    user = register_user(
        "isolation@example.test",
        "a strong password 123",
        accepted_terms=True,
        database=database,
    )

    assert database.exists(), "register_user ignored the database it was given"
    assert user.email == "isolation@example.test"
