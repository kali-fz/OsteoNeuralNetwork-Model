"""Chooses where accounts live: Cloudflare D1 when configured, local SQLite otherwise.

WHY
---
``database.py`` stores accounts in a SQLite file. That is right for a local run
and wrong for Hugging Face Spaces, whose filesystem is wiped on every restart --
users would silently lose their accounts each time the Space slept.

Rather than fork ``auth.py`` into local and hosted variants, this module presents
one interface and dispatches. ``auth.py`` imports from here and does not care
which backend answered; password hashing stays in ``auth.py`` either way, so the
tested PBKDF2 implementation remains the only one.

Selection is by configuration, not by a flag: set ``ONNM_COMMUNITY_URL`` and
``ONNM_COMMUNITY_KEY`` and accounts go to Cloudflare; leave them unset and
nothing changes from the local behaviour that exists today.
"""

from __future__ import annotations

from pathlib import Path

import database
from community import DuplicateEmailError as CommunityDuplicateEmailError
from community import get_client

# Re-exported so callers catch one exception type regardless of backend.
DatabaseError = database.DatabaseError
DuplicateEmailError = database.DuplicateEmailError
User = database.User


def using_community() -> bool:
    """True when accounts are stored in Cloudflare rather than on local disk."""
    return get_client().enabled


def backend_name() -> str:
    return "Cloudflare D1" if using_community() else "local SQLite"


def initialize_database(path: str | Path | None = None):
    """No-op for the hosted backend; the D1 schema is applied at deploy time."""
    if using_community():
        return None
    return database.initialize_database(path)


def create_user(email: str, password_hash: str, *, path: str | Path | None = None) -> User:
    """Create an account, raising DuplicateEmailError on a clash either way.

    The community backend raises its own DuplicateEmailError, which is
    translated here so ``auth.py`` keeps catching exactly one type.
    """
    if using_community():
        try:
            remote = get_client().create_user(email, password_hash)
        except CommunityDuplicateEmailError as exc:
            raise database.DuplicateEmailError(str(exc)) from exc
        return User(
            user_id=remote.user_id,
            email=remote.email,
            password_hash=remote.password_hash,
            created_at=remote.created_at,
            tos_accepted_at=remote.tos_accepted_at,
        )
    return database.create_user(email, password_hash, path=path)


def get_user_by_email(email: str, path: str | Path | None = None) -> User | None:
    if using_community():
        remote = get_client().get_user_by_email(email)
        if remote is None:
            return None
        return User(
            user_id=remote.user_id,
            email=remote.email,
            password_hash=remote.password_hash,
            created_at=remote.created_at,
            tos_accepted_at=remote.tos_accepted_at,
        )
    return database.get_user_by_email(email, path=path)
