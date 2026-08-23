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
--
-- ---------------------------------------------------------------------------
-- THE THREE BUCKETS
-- ---------------------------------------------------------------------------
-- Every shared submission is triaged into exactly one of three buckets, and the
-- bucket decides what the row can teach:
--
--   'valid_bone'    the OOD gate accepted it and the classifier ran normally.
--                   Retrains the lesion classifier. Needs a clinical label.
--   'misc'          the OOD gate rejected it -- a hotdog, a screenshot, a photo
--                   of a wall. Retrains the OOD detector as a negative. It must
--                   never receive a clinical label, because it has none.
--   'contradiction' the system disagrees with itself: the gate rejected an image
--                   the user insists is a radiograph, or it accepted one that is
--                   plainly not. These are the rows worth the most per example,
--                   because each one is a demonstrated failure of the gate.
--
-- As with labels, the bucket exists twice. `triage_bucket` is computed from the
-- model's own signals and is therefore only as good as the gate that is being
-- corrected; `admin_bucket` is what a human confirmed. Export reads the admin
-- column, and falls back to nothing at all rather than to the guess.

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

    -- Admin is one specific person, and the database says so.
    --
    -- The review queue is the only path by which anything reaches training, so
    -- "who is an admin" is not a preference to be configured -- it is a fixed
    -- property of this deployment. Writing the id into a CHECK means no code
    -- path, no misconfigured environment variable and no future endpoint can
    -- grant the flag to a second account: the INSERT simply fails. Moving the
    -- privilege requires a migration, which is the correct amount of friction.
    is_admin        INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    CHECK (is_admin = 0 OR user_id = 'c2c5a209-4aaa-4eb9-b112-b2929b6dbe12'),

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

    -- Automatic triage. Computed by the Worker from ood_flagged, the softmax
    -- confidence and any user dispute -- see triageBucket() in worker.js, and
    -- classify_bucket() in src/community.py, which must agree.
    --
    -- Recomputed when feedback arrives, because a user disputing a rejection is
    -- exactly the evidence that moves a row from 'misc' to 'contradiction'.
    -- Untrusted in the same sense the user columns are: it is the guess of the
    -- system being corrected, so it orders the queue and nothing more.
    triage_bucket   TEXT NOT NULL DEFAULT 'valid_bone'
        CHECK (triage_bucket IN ('valid_bone', 'misc', 'contradiction')),
    triage_reason   TEXT,

    -- Trusted human review. This is the gate.
    review_status   TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),

    -- The bucket a human confirmed. NULL until reviewed. Distinct from
    -- triage_bucket for the same reason admin_label is distinct from
    -- user_suggested_label: the automatic value is the thing under correction
    -- and cannot be its own ground truth. Export reads only this column.
    admin_bucket    TEXT CHECK (admin_bucket IN ('valid_bone', 'misc', 'contradiction')),

    -- 'misc' joins the three clinical classes here, and means "not a bone
    -- radiograph". It is a real training target -- the OOD detector needs
    -- negatives and currently has none but hand-written heuristics -- but it is
    -- not a diagnosis, so the trigger below stops it being paired with a
    -- diagnostic bucket and stops a diagnosis being pinned to a non-radiograph.
    admin_label     TEXT CHECK (admin_label IN ('normal', 'benign', 'malignant', 'misc')),
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
--
-- The bucket is required for the same reason the label is: export sorts rows by
-- admin_bucket into two different training targets, and a row with no bucket
-- would be silently dropped from both rather than raise.
CREATE TRIGGER IF NOT EXISTS approved_rows_need_a_label
BEFORE UPDATE OF review_status ON submissions
WHEN NEW.review_status = 'approved'
     AND (NEW.admin_label IS NULL OR NEW.admin_bucket IS NULL OR NEW.shared = 0)
BEGIN
    SELECT RAISE(ABORT,
        'cannot approve: an approved row needs admin_label, admin_bucket and shared = 1');
END;

-- A label and a bucket that contradict each other would poison whichever
-- training target believed it.
--
-- A 'misc' row carries no diagnosis: calling a hotdog 'benign' is precisely the
-- hotdog case one level up, dressed as a bucket assignment. And a row a human
-- put in a bone bucket cannot be labelled 'misc', because the lesion manifest
-- has no index for it and it would land in training as class -1 or be silently
-- skipped. 'contradiction' accepts either: the bucket records that the gate got
-- it wrong, and the label records what the image actually was, which is what
-- makes those rows the useful ones.
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
-- Partial index: the review queue only ever asks for shared rows, and this
-- keeps it small as unshared submissions accumulate.
CREATE INDEX IF NOT EXISTS idx_submissions_queue
    ON submissions(created_at) WHERE shared = 1 AND review_status = 'pending';
-- The review UI shows one bucket at a time, so the queue is always filtered on
-- it; without this the tab a bucket is empty in still scans the whole queue.
CREATE INDEX IF NOT EXISTS idx_submissions_triage
    ON submissions(triage_bucket, created_at) WHERE shared = 1 AND review_status = 'pending';
-- Export walks approved rows bucket by bucket -- lesion rows to one manifest,
-- OOD negatives to another.
CREATE INDEX IF NOT EXISTS idx_submissions_approved_bucket
    ON submissions(admin_bucket, created_at) WHERE review_status = 'approved';
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
    row_count   INTEGER NOT NULL DEFAULT 0,
    -- Split by what the rows retrain, because a batch of 40 that was 39 OOD
    -- negatives and one bone film is not the same event as the reverse, and
    -- "which generation saw what" is unanswerable later from a single total.
    lesion_rows INTEGER NOT NULL DEFAULT 0,
    ood_rows    INTEGER NOT NULL DEFAULT 0
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
INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '3');

-- Per-user, per-day write counter. Bounds one account's ability to consume
-- the shared free tier, whether by enthusiasm or malice.
CREATE TABLE IF NOT EXISTS rate_limit (
    user_id   TEXT NOT NULL,
    day       TEXT NOT NULL,          -- YYYY-MM-DD (UTC)
    submissions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
