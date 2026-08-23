"""Password authentication and Streamlit session helpers for ONNM."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

# Imported from `backend`, not `database`, so accounts land in Cloudflare D1
# when the app is hosted and in local SQLite otherwise. The interface is
# identical either way; password hashing below is unaffected and remains the
# single implementation.
from backend import (
    DuplicateEmailError,
    User,
    create_user,
    get_user_by_email,
    initialize_database,
)

PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DUMMY_SALT = bytes(SALT_BYTES)
_DUMMY_DIGEST = hashlib.pbkdf2_hmac(
    "sha256", b"constant-time-dummy1", _DUMMY_SALT, PBKDF2_ITERATIONS
)
_DUMMY_HASH = (
    f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_DUMMY_SALT.hex()}${_DUMMY_DIGEST.hex()}"
)


class AuthenticationError(ValueError):
    """Authentication input or credentials are invalid."""


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise AuthenticationError("Enter a valid email address.")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < 12:
        raise AuthenticationError("Password must contain at least 12 characters.")
    if len(password) > 1024:
        raise AuthenticationError("Password is too long.")
    if not any(char.isalpha() for char in password) or not any(char.isdigit() for char in password):
        raise AuthenticationError("Password must contain at least one letter and one number.")


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    validate_password(password)
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str | None) -> bool:
    # A federated account stores NULL here. Rejecting a non-string outright
    # means "this account has no password" can only ever answer False, rather
    # than raising an AttributeError that some caller might mistake for a bug
    # worth working around.
    if not isinstance(encoded_hash, str):
        return False
    try:
        algorithm, iterations_text, salt_hex, expected_hex = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        if iterations < 100_000 or iterations > 10_000_000:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
    except (TypeError, ValueError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def register_user(
    email: str,
    password: str,
    *,
    accepted_terms: bool,
    database: str | Path | None = None,
) -> User:
    if not accepted_terms:
        raise AuthenticationError("You must accept the legal terms to create an account.")
    normalized = normalize_email(email)
    initialize_database(database)
    try:
        return create_user(normalized, hash_password(password), path=database)
    except DuplicateEmailError as exc:
        raise AuthenticationError(str(exc)) from exc


def authenticate_user(
    email: str,
    password: str,
    *,
    database: str | Path | None = None,
) -> User | None:
    try:
        normalized = normalize_email(email)
    except AuthenticationError:
        normalized = "invalid@example.invalid"

    initialize_database(database)
    user = get_user_by_email(normalized, path=database)
    # A federated account has no password to check. Treat it exactly like an
    # unknown account -- same dummy hash, same wasted work -- so that "this
    # address signs in with Google" is not something an attacker can learn by
    # timing the login form.
    federated = user is not None and getattr(user, "auth_provider", "password") != "password"
    # Perform the same expensive operation for unknown accounts to reduce timing leakage.
    encoded = user.password_hash if (user and not federated) else _DUMMY_HASH
    password_is_bounded = len(password) <= 1024
    valid = verify_password(password if password_is_bounded else "", encoded)
    return user if user and not federated and password_is_bounded and valid else None


def initialize_session(state: MutableMapping[str, Any]) -> None:
    state.setdefault("authenticated", False)
    state.setdefault("user_id", None)
    state.setdefault("user_email", None)
    state.setdefault("user_name", None)
    state.setdefault("user_picture", None)
    state.setdefault("user_subject", None)
    state.setdefault("public_contributor_profile", False)


def login_session(
    state: MutableMapping[str, Any],
    user: User,
    *,
    display_name: str | None = None,
    picture: str | None = None,
    subject: str | None = None,
) -> None:
    state["authenticated"] = True
    state["user_id"] = user.user_id
    state["user_email"] = user.email
    state["user_name"] = (
        display_name
        or getattr(user, "display_name", None)
        or user.email.split("@", 1)[0]
    )
    state["user_picture"] = picture or getattr(user, "profile_picture_url", None)
    state["user_subject"] = subject or getattr(user, "provider_subject", None)
    state["public_contributor_profile"] = bool(
        getattr(user, "public_contributor_profile", False)
    )


def logout_session(state: MutableMapping[str, Any]) -> None:
    state.clear()
    initialize_session(state)


def google_sign_in_required(*, hosted: bool, oidc_available: bool) -> bool:
    """Keep hosted/D1 account creation on Google even after config failures."""
    return hosted or oidc_available
