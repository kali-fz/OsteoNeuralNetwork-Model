from __future__ import annotations

import sqlite3

import pytest

from auth import (
    AuthenticationError,
    authenticate_user,
    hash_password,
    login_session,
    logout_session,
    register_user,
    verify_password,
)
from database import create_upload, list_user_uploads


def test_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("correct horse 123")
    second = hash_password("correct horse 123")

    assert first != second
    assert "correct horse" not in first
    assert verify_password("correct horse 123", first)
    assert not verify_password("wrong password 123", first)


def test_registration_authentication_and_consent_timestamp(tmp_path) -> None:
    database = tmp_path / "users.db"
    user = register_user(
        " USER@Example.com ",
        "a strong password 123",
        accepted_terms=True,
        database=database,
    )

    assert user.email == "user@example.com"
    assert user.tos_accepted_at
    assert authenticate_user(
        "user@example.com", "a strong password 123", database=database
    ) == user
    assert authenticate_user("user@example.com", "incorrect 1234", database=database) is None


def test_registration_requires_terms_and_unique_email(tmp_path) -> None:
    database = tmp_path / "users.db"
    with pytest.raises(AuthenticationError, match="accept"):
        register_user("user@example.com", "a strong password 123", accepted_terms=False)

    register_user(
        "user@example.com",
        "a strong password 123",
        accepted_terms=True,
        database=database,
    )
    with pytest.raises(AuthenticationError, match="already exists"):
        register_user(
            "USER@example.com",
            "a different password 456",
            accepted_terms=True,
            database=database,
        )


def test_scan_history_is_scoped_to_user(tmp_path) -> None:
    database = tmp_path / "users.db"
    first = register_user(
        "first@example.com", "a strong password 123", accepted_terms=True, database=database
    )
    second = register_user(
        "second@example.com", "a strong password 456", accepted_terms=True, database=database
    )
    create_upload(
        user_id=first.user_id,
        filename="scan.png",
        file_path=tmp_path / "stored.png",
        model_verdict="Normal",
        confidence_score=92.5,
        path=database,
    )

    assert len(list_user_uploads(first.user_id, path=database)) == 1
    assert list_user_uploads(second.user_id, path=database) == []


def test_foreign_key_rejects_unknown_user(tmp_path) -> None:
    database = tmp_path / "users.db"
    register_user(
        "user@example.com", "a strong password 123", accepted_terms=True, database=database
    )
    with pytest.raises(Exception) as raised:
        create_upload(
            user_id="00000000-0000-0000-0000-000000000000",
            filename="scan.png",
            file_path=tmp_path / "stored.png",
            model_verdict="Normal",
            confidence_score=50,
            path=database,
        )
    assert isinstance(raised.value.__cause__, sqlite3.IntegrityError)


def test_session_login_and_logout() -> None:
    state = {"cache_key": "secret", "unrelated": "kept"}
    user = type("UserLike", (), {"user_id": "id", "email": "user@example.com"})()

    login_session(state, user)
    assert state["authenticated"] is True
    logout_session(state)

    assert state["authenticated"] is False
    assert state["user_id"] is None
    assert "cache_key" not in state
    assert "unrelated" not in state
