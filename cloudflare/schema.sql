-- ONNM community database (Cloudflare D1 / SQLite dialect).
--
-- Apply with:
--   npx wrangler d1 execute onnm-community --remote --file=./schema.sql
--
-- ---------------------------------------------------------------------------
-- THE CENTRAL INVARIANT
-- ---------------------------------------------------------------------------
-- A user saying "this result was wrong" is a SIGNAL, never a LABEL. Anonymous
-- self-reported labels cannot enter a training set: someone uploads a hotdog,
-- calls the malignant verdict wrong, and if that flowed through unchecked the
-- next generation trains on a hotdog labelled "normal bone".
--
-- So user feedback and ground truth live in two separate columns that are never
-- copied into one another:
--
--   user_says_wrong / user_suggested_label   <- untrusted, anyone can write it
--   admin_label                              <- trusted, only set during review
--
-- Nothing is exportable for training unless a human set `admin_label` and moved
-- `review_status` to 'approved'. That gate is enforced in the export query, in
-- the Worker, and again in scripts/export_batch.py -- three places, because
-- this is the one mistake in the whole design that would silently poison the
-- model rather than raise an error.

-- ---------------------------------------------------------------------------
-- Accounts.
--
-- Password hashing happens in Python (src/auth.py, PBKDF2-HMAC-SHA256 at
-- 600k iterations) and only the encoded hash is ever sent here. The Worker
-- never sees a plaintext password and cannot verify one -- it is a store, not
-- an authenticator. That keeps the tested Python implementation authoritative
-- rather than re-implementing crypto in a second language.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,

    -- NULL for a federated account. Google holds those credentials; we never
    -- receive a password and must not pretend to store one. A sentinel string
    -- here would be worse than NULL: it reads like a hash to anyone auditing
    -- the table, and the CHECK below could not then rule out a password login.
    password_hash   TEXT,

    -- 'password' for a local account, 'google' for Google Sign-In. The pairing
    -- with password_hash is enforced rather than assumed, because the failure
    -- it prevents -- a federated account that can also be logged into with a
    -- password someone set -- would be an authentication bypass, not a bug.
    auth_provider   TEXT NOT NULL DEFAULT 'password'
        CHECK (auth_provider IN ('password', 'google')),
    -- Google's `sub` claim: stable for the life of the account, and unlike the
    -- email address it never changes hands. Identity is keyed on this.
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

-- One account per Google identity. Partial, so the many NULLs on password
-- accounts do not collide (SQLite treats NULLs as distinct in a UNIQUE index,
-- but stating the intent keeps the index small and the meaning obvious).
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_subject
    ON users(provider_subject) WHERE provider_subject IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Submissions: one row per image a user ran through the model.
--
-- The image is stored ONLY when the user opted in (shared = 1). It is the
-- 256px preprocessed array as PNG base64 -- about 30 KB, and precisely what
-- retraining consumes, so keeping the original would cost far more storage to
-- hold data the pipeline would only downsample again. It also means a stored
-- image has already been stripped of DICOM headers by the preprocessing path.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS submissions (
    submission_id   TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,

    -- What the model said.
    model_label     TEXT NOT NULL,
    lesion_probability REAL NOT NULL
        CHECK (lesion_probability >= 0 AND lesion_probability <= 1),
    class_probabilities TEXT NOT NULL,          -- JSON object
    checkpoint      TEXT,                        -- which model generation
    threshold       REAL,
    calibrated      INTEGER NOT NULL DEFAULT 0 CHECK (calibrated IN (0, 1)),

    -- Out-of-distribution gate (src/onnm/ood.py). A hotdog should be caught
    -- here and never receive a clinical-sounding verdict at all.
    ood_flagged     INTEGER NOT NULL DEFAULT 0 CHECK (ood_flagged IN (0, 1)),
    ood_score       REAL,

    -- Consent. Default 0: sharing is opt-in, never assumed.
    shared          INTEGER NOT NULL DEFAULT 0 CHECK (shared IN (0, 1)),
    consent_at      TEXT,
    image_b64       TEXT,                        -- NULL unless shared = 1
    image_sha256    TEXT,                        -- dedupe identical resubmissions
    image_bytes     INTEGER NOT NULL DEFAULT 0,

    -- Untrusted user feedback. NULL means "user said nothing".
    user_says_wrong      INTEGER CHECK (user_says_wrong IN (0, 1)),
    user_suggested_label TEXT,
    user_comment         TEXT,
    feedback_at          TEXT,

    -- Trusted human review. This is the gate.
    review_status   TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    admin_label     TEXT CHECK (admin_label IN ('normal', 'benign', 'malignant')),
    admin_note      TEXT,
    reviewed_at     TEXT,
    reviewed_by     TEXT,

    -- Set when a row is pulled into a training batch, so batches are
    -- reproducible and a row cannot silently be trained on twice.
    batch_id        TEXT,
    exported_at     TEXT,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Never approve a row without a ground-truth label. Enforced by the database
-- rather than trusted to the Worker, because this is the invariant that
-- protects the training set.
CREATE TRIGGER IF NOT EXISTS approved_rows_need_a_label
BEFORE UPDATE OF review_status ON submissions
WHEN NEW.review_status = 'approved'
     AND (NEW.admin_label IS NULL OR NEW.shared = 0)
BEGIN
    SELECT RAISE(ABORT,
        'cannot approve: an approved row needs admin_label set and shared = 1');
END;

CREATE INDEX IF NOT EXISTS idx_submissions_user
    ON submissions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_review
    ON submissions(review_status, created_at);
-- Partial index: the review queue only ever asks for shared rows, and this
-- keeps it small as unshared submissions accumulate.
CREATE INDEX IF NOT EXISTS idx_submissions_queue
    ON submissions(created_at) WHERE shared = 1 AND review_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_submissions_batch
    ON submissions(batch_id) WHERE batch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_submissions_sha
    ON submissions(image_sha256) WHERE image_sha256 IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Training batches: a named, frozen set of approved rows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batches (
    batch_id    TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    note        TEXT,
    row_count   INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Counters for the spend guard.
--
-- The account carries no payment method, so overage cannot be billed -- it
-- returns errors instead. These caps exist so the failure is a clear refusal
-- from our own code rather than an opaque D1 quota error, and so the free
-- tier is never actually approached. Kept as running totals because
-- SUM(length(image_b64)) across the table on every insert is exactly the
-- O(n) query that would burn the read quota it is meant to protect.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('bytes_stored', '0');
INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');

-- Per-user, per-day write counter. Bounds one account's ability to consume
-- the shared free tier, whether by enthusiasm or malice.
CREATE TABLE IF NOT EXISTS rate_limit (
    user_id   TEXT NOT NULL,
    day       TEXT NOT NULL,          -- YYYY-MM-DD (UTC)
    submissions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
