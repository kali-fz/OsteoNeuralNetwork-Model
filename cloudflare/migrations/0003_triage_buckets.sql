-- Migration 0003: three-bucket triage, an expanded label vocabulary, and a
-- database-level pin on who may review.
--
-- WHAT CHANGES
-- ------------
-- 1. `submissions` gains `triage_bucket` / `triage_reason` (the automatic
--    guess), `admin_bucket` (the human's confirmation), and `admin_label`
--    gains 'misc' as a fourth value meaning "not a bone radiograph at all".
-- 2. `users` gains a CHECK pinning `is_admin = 1` to a single account id.
-- 3. `batches` gains per-target row counts.
--
-- WHY TWO TABLE REBUILDS AND ONE ALTER
-- ------------------------------------
-- Adding a column is additive and ALTER TABLE handles it. Changing a CHECK
-- constraint is not: SQLite stores the constraint as part of the table's
-- declared SQL and offers no way to amend it in place, so widening
-- `admin_label` to accept 'misc' and narrowing `users.is_admin` to one id both
-- require the standard rebuild recipe -- create the new shape, copy the rows,
-- drop the old table, rename. `batches` only gains columns, so it is an ALTER.
--
-- The rebuild is also the only honest way to do this. An alternative would be
-- to leave the old CHECK alone and enforce 'misc' in the Worker, but the whole
-- point of these constraints is that they hold when the Worker is wrong.
--
-- EXISTING ROWS
-- -------------
-- Every existing submission predates triage, so it is backfilled from the one
-- signal that was already recorded: `ood_flagged`. A flagged row becomes
-- 'misc', an unflagged one 'valid_bone'. No existing row becomes
-- 'contradiction' -- that classification needs the confidence and dispute
-- signals this migration starts recording, and inventing it retroactively
-- would put rows in the highest-value queue on no evidence.
--
-- Already-approved rows keep their approval and are given an `admin_bucket`
-- matching their existing clinical label, which is always a bone bucket: the
-- old schema could not express anything else.
--
-- Apply with:
--   npx wrangler d1 execute onnm-community --remote --file=./migrations/0003_triage_buckets.sql

PRAGMA foreign_keys = OFF;

-- ---------------------------------------------------------------------------
-- users: pin the admin flag to one account id.
-- ---------------------------------------------------------------------------
CREATE TABLE users_new (
    user_id         TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT,
    auth_provider   TEXT NOT NULL DEFAULT 'password'
        CHECK (auth_provider IN ('password', 'google')),
    provider_subject TEXT,
    created_at      TEXT NOT NULL,
    tos_accepted_at TEXT NOT NULL,
    is_admin        INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    CHECK (is_admin = 0 OR user_id = 'c2c5a209-4aaa-4eb9-b112-b2929b6dbe12'),
    CHECK (
        (auth_provider = 'password'
             AND password_hash IS NOT NULL AND provider_subject IS NULL)
     OR (auth_provider = 'google'
             AND password_hash IS NULL     AND provider_subject IS NOT NULL)
    )
);

-- Any stray admin flag on another account is cleared rather than allowed to
-- abort the copy. Losing a privilege that this migration exists to revoke is
-- the intended outcome; failing halfway through a rebuild is not.
INSERT INTO users_new (user_id, email, password_hash, auth_provider,
                       provider_subject, created_at, tos_accepted_at, is_admin)
SELECT user_id, email, password_hash, auth_provider, provider_subject,
       created_at, tos_accepted_at,
       CASE WHEN user_id = 'c2c5a209-4aaa-4eb9-b112-b2929b6dbe12'
            THEN is_admin ELSE 0 END
  FROM users;

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_subject
    ON users(provider_subject) WHERE provider_subject IS NOT NULL;

-- ---------------------------------------------------------------------------
-- submissions: triage columns and the widened label vocabulary.
-- ---------------------------------------------------------------------------
DROP TRIGGER IF EXISTS approved_rows_need_a_label;

CREATE TABLE submissions_new (
    submission_id   TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    model_label     TEXT NOT NULL,
    lesion_probability REAL NOT NULL
        CHECK (lesion_probability >= 0 AND lesion_probability <= 1),
    class_probabilities TEXT NOT NULL,
    checkpoint      TEXT,
    threshold       REAL,
    calibrated      INTEGER NOT NULL DEFAULT 0 CHECK (calibrated IN (0, 1)),
    ood_flagged     INTEGER NOT NULL DEFAULT 0 CHECK (ood_flagged IN (0, 1)),
    ood_score       REAL,
    shared          INTEGER NOT NULL DEFAULT 0 CHECK (shared IN (0, 1)),
    consent_at      TEXT,
    image_b64       TEXT,
    image_sha256    TEXT,
    image_bytes     INTEGER NOT NULL DEFAULT 0,
    user_says_wrong      INTEGER CHECK (user_says_wrong IN (0, 1)),
    user_suggested_label TEXT,
    user_comment         TEXT,
    feedback_at          TEXT,
    triage_bucket   TEXT NOT NULL DEFAULT 'valid_bone'
        CHECK (triage_bucket IN ('valid_bone', 'misc', 'contradiction')),
    triage_reason   TEXT,
    review_status   TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    admin_bucket    TEXT CHECK (admin_bucket IN ('valid_bone', 'misc', 'contradiction')),
    admin_label     TEXT CHECK (admin_label IN ('normal', 'benign', 'malignant', 'misc')),
    admin_note      TEXT,
    reviewed_at     TEXT,
    reviewed_by     TEXT,
    batch_id        TEXT,
    exported_at     TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

INSERT INTO submissions_new (
    submission_id, user_id, created_at, model_label, lesion_probability,
    class_probabilities, checkpoint, threshold, calibrated, ood_flagged, ood_score,
    shared, consent_at, image_b64, image_sha256, image_bytes,
    user_says_wrong, user_suggested_label, user_comment, feedback_at,
    triage_bucket, triage_reason,
    review_status, admin_bucket, admin_label, admin_note, reviewed_at, reviewed_by,
    batch_id, exported_at
)
SELECT
    submission_id, user_id, created_at, model_label, lesion_probability,
    class_probabilities, checkpoint, threshold, calibrated, ood_flagged, ood_score,
    shared, consent_at, image_b64, image_sha256, image_bytes,
    user_says_wrong, user_suggested_label, user_comment, feedback_at,
    CASE WHEN ood_flagged = 1 THEN 'misc' ELSE 'valid_bone' END,
    'backfilled by migration 0003 from ood_flagged',
    review_status,
    -- Approved rows already carry a clinical label, which the old schema only
    -- allowed for bone images, so their bucket is not in doubt.
    CASE WHEN review_status = 'approved' THEN 'valid_bone' ELSE NULL END,
    admin_label, admin_note, reviewed_at, reviewed_by,
    batch_id, exported_at
FROM submissions;

DROP TABLE submissions;
ALTER TABLE submissions_new RENAME TO submissions;

CREATE TRIGGER IF NOT EXISTS approved_rows_need_a_label
BEFORE UPDATE OF review_status ON submissions
WHEN NEW.review_status = 'approved'
     AND (NEW.admin_label IS NULL OR NEW.admin_bucket IS NULL OR NEW.shared = 0)
BEGIN
    SELECT RAISE(ABORT,
        'cannot approve: an approved row needs admin_label, admin_bucket and shared = 1');
END;

CREATE TRIGGER IF NOT EXISTS bucket_and_label_must_agree
BEFORE UPDATE OF admin_label, admin_bucket ON submissions
WHEN NEW.admin_bucket IS NOT NULL AND NEW.admin_label IS NOT NULL
     AND ((NEW.admin_bucket = 'misc'       AND NEW.admin_label != 'misc')
       OR (NEW.admin_bucket = 'valid_bone' AND NEW.admin_label = 'misc'))
BEGIN
    SELECT RAISE(ABORT,
        'cannot review: admin_bucket and admin_label disagree (misc takes the misc label; valid_bone takes a clinical label)');
END;

CREATE INDEX IF NOT EXISTS idx_submissions_user
    ON submissions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_review
    ON submissions(review_status, created_at);
CREATE INDEX IF NOT EXISTS idx_submissions_queue
    ON submissions(created_at) WHERE shared = 1 AND review_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_submissions_triage
    ON submissions(triage_bucket, created_at) WHERE shared = 1 AND review_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_submissions_approved_bucket
    ON submissions(admin_bucket, created_at) WHERE review_status = 'approved';
CREATE INDEX IF NOT EXISTS idx_submissions_batch
    ON submissions(batch_id) WHERE batch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_submissions_sha
    ON submissions(image_sha256) WHERE image_sha256 IS NOT NULL;

-- ---------------------------------------------------------------------------
-- batches: additive, so no rebuild.
-- ---------------------------------------------------------------------------
ALTER TABLE batches ADD COLUMN lesion_rows INTEGER NOT NULL DEFAULT 0;
ALTER TABLE batches ADD COLUMN ood_rows INTEGER NOT NULL DEFAULT 0;

PRAGMA foreign_keys = ON;

-- Grant the review flag to the pinned account, if it is already present.
--
-- A no-op when the account has not signed in yet: the CHECK above means the row
-- can be created later with is_admin = 1, and no other row can ever hold it.
-- If this leaves nothing updated, the account id in the app has diverged from
-- the one hardcoded here -- see cloudflare/README.md, "Who can review".
UPDATE users SET is_admin = 1 WHERE user_id = 'c2c5a209-4aaa-4eb9-b112-b2929b6dbe12';

UPDATE meta SET value = '3' WHERE key = 'schema_version';
