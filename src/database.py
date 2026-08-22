"""Local SQLite persistence for ONNM users and scan history."""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "users.db"


class DatabaseError(RuntimeError):
    """A database operation could not be completed."""


class DuplicateEmailError(DatabaseError):
    """An account already exists for an email address."""


@dataclass(frozen=True)
class User:
    user_id: str
    email: str
    password_hash: str
    created_at: str
    tos_accepted_at: str


@dataclass(frozen=True)
class UploadRecord:
    upload_id: str
    user_id: str
    filename: str
    file_path: str
    upload_timestamp: str
    model_verdict: str
    confidence_score: float


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def database_path(path: str | Path | None = None) -> Path:
    configured = os.environ.get("ONNM_DATABASE_PATH")
    return Path(path or configured or DEFAULT_DATABASE_PATH).resolve()


def _secure_permissions(path: Path, mode: int) -> None:
    # Windows ACLs remain authoritative; chmod is best-effort hardening.
    with suppress(OSError):
        path.chmod(mode)


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = database_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _secure_permissions(db_path.parent, 0o700)

    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        yield connection
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise DatabaseError("The local account database operation failed.") from exc
    finally:
        connection.close()

    if db_path.exists():
        _secure_permissions(db_path, 0o600)


def initialize_database(path: str | Path | None = None) -> Path:
    db_path = database_path(path)
    with connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tos_accepted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                upload_timestamp TEXT NOT NULL,
                model_verdict TEXT NOT NULL,
                confidence_score REAL NOT NULL
                    CHECK (confidence_score >= 0 AND confidence_score <= 100),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_uploads_user_timestamp
                ON uploads(user_id, upload_timestamp DESC);
            """
        )
    return db_path


def create_user(
    email: str,
    password_hash: str,
    *,
    tos_accepted_at: str | None = None,
    path: str | Path | None = None,
) -> User:
    user = User(
        user_id=str(uuid.uuid4()),
        email=email,
        password_hash=password_hash,
        created_at=utc_now(),
        tos_accepted_at=tos_accepted_at or utc_now(),
    )
    try:
        with connect(path) as connection:
            connection.execute(
                """
                INSERT INTO users
                    (user_id, email, password_hash, created_at, tos_accepted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.email,
                    user.password_hash,
                    user.created_at,
                    user.tos_accepted_at,
                ),
            )
    except DatabaseError as exc:
        if isinstance(exc.__cause__, sqlite3.IntegrityError):
            raise DuplicateEmailError("An account already exists for this email.") from exc
        raise
    return user


def get_user_by_email(email: str, path: str | Path | None = None) -> User | None:
    with connect(path) as connection:
        row = connection.execute(
            """
            SELECT user_id, email, password_hash, created_at, tos_accepted_at
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
    return User(**dict(row)) if row else None


def create_upload(
    *,
    user_id: str,
    filename: str,
    file_path: str | Path,
    model_verdict: str,
    confidence_score: float,
    upload_id: str | None = None,
    path: str | Path | None = None,
) -> UploadRecord:
    record = UploadRecord(
        upload_id=upload_id or str(uuid.uuid4()),
        user_id=user_id,
        filename=filename,
        file_path=str(Path(file_path).resolve()),
        upload_timestamp=utc_now(),
        model_verdict=model_verdict,
        confidence_score=float(confidence_score),
    )
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO uploads
                (upload_id, user_id, filename, file_path, upload_timestamp,
                 model_verdict, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.upload_id,
                record.user_id,
                record.filename,
                record.file_path,
                record.upload_timestamp,
                record.model_verdict,
                record.confidence_score,
            ),
        )
    return record


def list_user_uploads(
    user_id: str,
    *,
    limit: int = 100,
    path: str | Path | None = None,
) -> list[UploadRecord]:
    safe_limit = max(1, min(int(limit), 500))
    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT upload_id, user_id, filename, file_path, upload_timestamp,
                   model_verdict, confidence_score
            FROM uploads
            WHERE user_id = ?
            ORDER BY upload_timestamp DESC
            LIMIT ?
            """,
            (user_id, safe_limit),
        ).fetchall()
    return [UploadRecord(**dict(row)) for row in rows]


def update_upload_result(
    upload_id: str,
    user_id: str,
    *,
    model_verdict: str,
    confidence_score: float,
    path: str | Path | None = None,
) -> None:
    with connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE uploads
            SET model_verdict = ?, confidence_score = ?
            WHERE upload_id = ? AND user_id = ?
            """,
            (model_verdict, float(confidence_score), upload_id, user_id),
        )
        if cursor.rowcount != 1:
            raise DatabaseError("The scan history record was not found.")
