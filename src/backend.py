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
        return _local(remote)
    return database.create_user(email, password_hash, path=path)


def _local(remote) -> User:
    """Convert a community User into the database User the app passes around."""
    return User(
        user_id=remote.user_id,
        email=remote.email,
        password_hash=remote.password_hash,
        created_at=remote.created_at,
        tos_accepted_at=remote.tos_accepted_at,
        auth_provider=remote.auth_provider,
        provider_subject=remote.provider_subject,
    )


def get_user_by_email(email: str, path: str | Path | None = None) -> User | None:
    if using_community():
        remote = get_client().get_user_by_email(email)
        return _local(remote) if remote is not None else None
    return database.get_user_by_email(email, path=path)


def get_user_by_subject(
    provider_subject: str, path: str | Path | None = None
) -> User | None:
    if using_community():
        remote = get_client().get_user_by_subject(provider_subject)
        return _local(remote) if remote is not None else None
    return database.get_user_by_subject(provider_subject, path=path)


def create_oauth_user(
    email: str,
    provider_subject: str,
    *,
    auth_provider: str = "google",
    path: str | Path | None = None,
) -> User:
    if using_community():
        try:
            remote = get_client().create_oauth_user(
                email, provider_subject, auth_provider=auth_provider
            )
        except CommunityDuplicateEmailError as exc:
            raise database.DuplicateEmailError(str(exc)) from exc
        return _local(remote)
    return database.create_oauth_user(
        email, provider_subject, auth_provider=auth_provider, path=path
    )


def get_or_create_oauth_user(
    email: str,
    provider_subject: str,
    *,
    auth_provider: str = "google",
    path: str | Path | None = None,
) -> User:
    """Resolve a signed-in Google identity to an account, creating it once.

    Lookup is by ``provider_subject`` first and email only as a fallback, which
    is the order that matters: ``sub`` is the stable identifier, so a user who
    changes the address on their Google account keeps the same ONNM account and
    the same submission history rather than silently acquiring a second one.

    The email fallback exists for the reverse case -- an account created by a
    provider that did not supply a subject, or a password account being
    upgraded -- and deliberately does *not* rewrite the stored provider. An
    existing password account is returned as-is rather than converted, because
    silently turning a password login into a Google login would let anyone who
    can prove control of the address take over that account.
    """
    existing = get_user_by_subject(provider_subject, path=path)
    if existing is not None:
        return existing
    existing = get_user_by_email(email, path=path)
    if existing is not None:
        return existing
    return create_oauth_user(
        email, provider_subject, auth_provider=auth_provider, path=path
    )
