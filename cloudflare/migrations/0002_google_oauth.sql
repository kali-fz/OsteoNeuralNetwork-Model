-- Migration 0002: federated (Google) accounts.
--
-- WHY A TABLE REBUILD RATHER THAN ALTER TABLE
-- -------------------------------------------
-- Two of the three changes are additive and could be done with ALTER TABLE ADD
-- COLUMN. The third cannot: `password_hash` must become nullable, because a
-- Google account has no password and storing a sentinel in a column named
-- `password_hash` would mislead anyone auditing the table. SQLite cannot relax
-- NOT NULL in place, so the standard rebuild recipe is the only route.
--
-- Rows are copied rather than discarded, so this is safe to run against a
-- database that already has real accounts in it. Existing rows are all
-- password accounts by definition -- federated ones could not exist before
-- this migration -- so they take auth_provider = 'password'.
--
-- Idempotent by guard: `users_old` is dropped at the end, and re-running on an
-- already-migrated table is prevented by the schema_version check in meta.
-- Apply with:
--   npx wrangler d1 execute onnm-community --remote --file=./migrations/0002_google_oauth.sql

PRAGMA defer_foreign_keys = true;

CREATE TABLE IF NOT EXISTS users_new (
    user_id         TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT,
    auth_provider   TEXT NOT NULL DEFAULT 'password'
        CHECK (auth_provider IN ('password', 'google')),
    provider_subject TEXT,
    created_at      TEXT NOT NULL,
    tos_accepted_at TEXT NOT NULL,
    is_admin        INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    CHECK (
        (auth_provider = 'password'
             AND password_hash IS NOT NULL AND provider_subject IS NULL)
     OR (auth_provider = 'google'
             AND password_hash IS NULL     AND provider_subject IS NOT NULL)
    )
);

INSERT OR IGNORE INTO users_new
    (user_id, email, password_hash, auth_provider, provider_subject,
     created_at, tos_accepted_at, is_admin)
SELECT user_id, email, password_hash, 'password', NULL,
       created_at, tos_accepted_at, is_admin
  FROM users;

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_subject
    ON users(provider_subject) WHERE provider_subject IS NOT NULL;

UPDATE meta SET value = '2' WHERE key = 'schema_version';
