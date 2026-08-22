/**
 * ONNM community API — Cloudflare Worker over D1.
 *
 * WHAT THIS IS
 * ------------
 * A thin, guarded data layer. The Streamlit app on Hugging Face Spaces is the
 * only client; it holds a shared secret and does all the interesting work
 * (inference, Grad-CAM, password hashing). This Worker stores rows and enforces
 * two things the client cannot be trusted to enforce on itself:
 *
 *   1. The review gate — user feedback never becomes a training label.
 *   2. The spend guard — hard caps so the free tier is never approached.
 *
 * WHY THE WORKER DOES NOT HASH PASSWORDS
 * --------------------------------------
 * src/auth.py already implements PBKDF2-HMAC-SHA256 at 600k iterations with a
 * constant-time dummy comparison for unknown accounts, and it is tested.
 * Re-implementing that in JS would mean two implementations that must agree
 * forever. Instead Python hashes and this stores the encoded string. The
 * Worker therefore cannot verify a password, which is the correct capability
 * for a store.
 *
 * BILLING
 * -------
 * Everything here is free-tier: Workers + D1 only. No R2, no KV, no Durable
 * Objects, no Queues, no paid bindings of any kind. With no payment method on
 * the account, overage cannot bill — it fails closed. The caps below exist so
 * that failure is an intelligible refusal from our own code long before any
 * platform limit is reached.
 */

// --- Spend guard ------------------------------------------------------------
// Sized for the free tier with a very wide margin. D1 free allows 5 GB; we stop
// at 200 MB, which at ~30 KB per shared image is roughly 6,600 images — far
// more than a test needs, and ~2.5% of the limit.
const MAX_BODY_BYTES = 1_500_000; // 1.5 MB request ceiling
const MAX_IMAGE_B64_BYTES = 600_000; // ~450 KB of image after base64
const MAX_TOTAL_BYTES_STORED = 200_000_000; // 200 MB of images, then refuse
const MAX_SUBMISSIONS_PER_USER_PER_DAY = 50;
const MAX_USERS = 500; // a test deployment, not a product launch
const MAX_PAGE_SIZE = 100;

const VALID_LABELS = new Set(["normal", "benign", "malignant"]);

// --- Small helpers ----------------------------------------------------------
const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });

const fail = (status, message, extra = {}) => json({ error: message, ...extra }, status);

const nowIso = () => new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
const today = () => nowIso().slice(0, 10);

/** Constant-time string compare, so the API key cannot be probed byte by byte. */
function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function bearer(request) {
  const header = request.headers.get("authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7) : "";
}

async function readJson(request) {
  const declared = Number(request.headers.get("content-length") || "0");
  if (declared > MAX_BODY_BYTES) {
    throw { status: 413, message: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  const text = await request.text();
  // content-length is client-supplied, so re-check what actually arrived.
  if (text.length > MAX_BODY_BYTES) {
    throw { status: 413, message: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    throw { status: 400, message: "body is not valid JSON" };
  }
}

async function getMetaInt(db, key) {
  const row = await db.prepare("SELECT value FROM meta WHERE key = ?").bind(key).first();
  return row ? Number(row.value) : 0;
}

// --- Route handlers ---------------------------------------------------------

async function health(db) {
  const users = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  const subs = await db.prepare("SELECT COUNT(*) AS n FROM submissions").first();
  const pending = await db
    .prepare(
      "SELECT COUNT(*) AS n FROM submissions WHERE shared = 1 AND review_status = 'pending'"
    )
    .first();
  const approved = await db
    .prepare("SELECT COUNT(*) AS n FROM submissions WHERE review_status = 'approved'")
    .first();
  const bytes = await getMetaInt(db, "bytes_stored");
  return json({
    ok: true,
    users: users.n,
    submissions: subs.n,
    pending_review: pending.n,
    approved: approved.n,
    bytes_stored: bytes,
    // Surfaced so the app can warn before anything starts refusing writes.
    capacity_used: Number((bytes / MAX_TOTAL_BYTES_STORED).toFixed(4)),
    limits: {
      max_total_bytes_stored: MAX_TOTAL_BYTES_STORED,
      max_submissions_per_user_per_day: MAX_SUBMISSIONS_PER_USER_PER_DAY,
      max_users: MAX_USERS,
    },
  });
}

async function createUser(db, body) {
  const { user_id, email, password_hash, tos_accepted_at, is_admin } = body;
  if (!user_id || !email || !password_hash) {
    return fail(400, "user_id, email and password_hash are required");
  }
  // The Worker must never be handed a plaintext password. This does not prove
  // the value is a real PBKDF2 hash, but it catches the client bug where the
  // raw password is passed by mistake, which is the failure worth catching.
  if (!String(password_hash).startsWith("pbkdf2_sha256$")) {
    return fail(400, "password_hash must be a pbkdf2_sha256 encoded hash, not a password");
  }
  const count = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  if (count.n >= MAX_USERS) return fail(507, `user cap reached (${MAX_USERS})`);

  const created = nowIso();
  try {
    await db
      .prepare(
        `INSERT INTO users (user_id, email, password_hash, created_at, tos_accepted_at, is_admin)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
      .bind(
        user_id,
        String(email).toLowerCase(),
        password_hash,
        created,
        tos_accepted_at || created,
        is_admin ? 1 : 0
      )
      .run();
  } catch (err) {
    if (String(err).includes("UNIQUE")) return fail(409, "an account already exists for that email");
    throw err;
  }
  return json({ user_id, email, created_at: created }, 201);
}

async function getUserByEmail(db, email) {
  if (!email) return fail(400, "email query parameter is required");
  const row = await db
    .prepare(
      `SELECT user_id, email, password_hash, created_at, tos_accepted_at, is_admin
       FROM users WHERE email = ? COLLATE NOCASE`
    )
    .bind(String(email).toLowerCase())
    .first();
  return row ? json(row) : fail(404, "no such user");
}

async function createSubmission(db, body) {
  const {
    submission_id,
    user_id,
    model_label,
    lesion_probability,
    class_probabilities,
    checkpoint,
    threshold,
    calibrated,
    ood_flagged,
    ood_score,
    shared,
    image_b64,
    image_sha256,
  } = body;

  if (!submission_id || !user_id || !model_label) {
    return fail(400, "submission_id, user_id and model_label are required");
  }
  const user = await db.prepare("SELECT user_id FROM users WHERE user_id = ?").bind(user_id).first();
  if (!user) return fail(404, "no such user");

  // Per-user daily cap.
  const day = today();
  const rl = await db
    .prepare("SELECT submissions FROM rate_limit WHERE user_id = ? AND day = ?")
    .bind(user_id, day)
    .first();
  if (rl && rl.submissions >= MAX_SUBMISSIONS_PER_USER_PER_DAY) {
    return fail(429, `daily submission limit reached (${MAX_SUBMISSIONS_PER_USER_PER_DAY})`);
  }

  // Consent decides whether an image is stored at all. A client that sends an
  // image without shared=1 is discarded rather than trusted, so an app-side bug
  // cannot silently retain data the user did not agree to share.
  const isShared = shared ? 1 : 0;
  let storedImage = null;
  let imageBytes = 0;
  if (isShared && image_b64) {
    if (image_b64.length > MAX_IMAGE_B64_BYTES) {
      return fail(413, `image exceeds ${MAX_IMAGE_B64_BYTES} base64 bytes`);
    }
    const bytesStored = await getMetaInt(db, "bytes_stored");
    if (bytesStored + image_b64.length > MAX_TOTAL_BYTES_STORED) {
      return fail(507, "community storage cap reached; contact the maintainer", {
        bytes_stored: bytesStored,
        cap: MAX_TOTAL_BYTES_STORED,
      });
    }
    storedImage = image_b64;
    imageBytes = image_b64.length;
  }

  const created = nowIso();
  const statements = [
    db
      .prepare(
        `INSERT INTO submissions (
            submission_id, user_id, created_at, model_label, lesion_probability,
            class_probabilities, checkpoint, threshold, calibrated,
            ood_flagged, ood_score, shared, consent_at, image_b64, image_sha256, image_bytes
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        submission_id,
        user_id,
        created,
        model_label,
        Number(lesion_probability ?? 0),
        JSON.stringify(class_probabilities ?? {}),
        checkpoint ?? null,
        threshold ?? null,
        calibrated ? 1 : 0,
        ood_flagged ? 1 : 0,
        ood_score ?? null,
        isShared,
        isShared ? created : null,
        storedImage,
        storedImage ? (image_sha256 ?? null) : null,
        imageBytes
      ),
    db
      .prepare(
        `INSERT INTO rate_limit (user_id, day, submissions) VALUES (?, ?, 1)
         ON CONFLICT(user_id, day) DO UPDATE SET submissions = submissions + 1`
      )
      .bind(user_id, day),
  ];
  if (imageBytes > 0) {
    statements.push(
      db
        .prepare(
          "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = 'bytes_stored'"
        )
        .bind(imageBytes)
    );
  }
  // batch() is atomic in D1, so the counter cannot drift from the rows.
  await db.batch(statements);

  return json({ submission_id, created_at: created, shared: isShared === 1 }, 201);
}

/**
 * User feedback. Deliberately cannot touch review_status or admin_label —
 * this endpoint writes only to the untrusted columns.
 */
async function submitFeedback(db, submissionId, body) {
  const { user_id, says_wrong, suggested_label, comment } = body;
  if (!user_id) return fail(400, "user_id is required");
  if (suggested_label && !VALID_LABELS.has(suggested_label)) {
    return fail(400, `suggested_label must be one of ${[...VALID_LABELS].join(", ")}`);
  }
  const row = await db
    .prepare("SELECT user_id, review_status FROM submissions WHERE submission_id = ?")
    .bind(submissionId)
    .first();
  if (!row) return fail(404, "no such submission");
  // A user may only annotate their own submission.
  if (row.user_id !== user_id) return fail(403, "not your submission");
  // Once reviewed, the record is evidence; late edits would rewrite history.
  if (row.review_status !== "pending") {
    return fail(409, "this submission has already been reviewed");
  }

  await db
    .prepare(
      `UPDATE submissions
         SET user_says_wrong = ?, user_suggested_label = ?, user_comment = ?, feedback_at = ?
       WHERE submission_id = ?`
    )
    .bind(
      says_wrong ? 1 : 0,
      suggested_label ?? null,
      comment ? String(comment).slice(0, 2000) : null,
      nowIso(),
      submissionId
    )
    .run();
  return json({ submission_id: submissionId, recorded: true });
}

async function listUserSubmissions(db, userId, limit) {
  if (!userId) return fail(400, "user_id is required");
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  const { results } = await db
    .prepare(
      `SELECT submission_id, created_at, model_label, lesion_probability, checkpoint,
              ood_flagged, shared, user_says_wrong, user_suggested_label,
              review_status, admin_label
         FROM submissions WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?`
    )
    .bind(userId, n)
    .all();
  return json({ submissions: results ?? [] });
}

// --- Admin ------------------------------------------------------------------

async function pendingReview(db, limit, includeImages) {
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  const columns = `submission_id, user_id, created_at, model_label, lesion_probability,
                   class_probabilities, ood_flagged, ood_score,
                   user_says_wrong, user_suggested_label, user_comment${
                     includeImages ? ", image_b64" : ""
                   }`;
  const { results } = await db
    .prepare(
      `SELECT ${columns} FROM submissions
        WHERE shared = 1 AND review_status = 'pending'
        ORDER BY user_says_wrong DESC, created_at ASC
        LIMIT ?`
    )
    .bind(n)
    .all();
  // Ordered so disputed results surface first — those are the ones with new
  // information in them; agreeing predictions teach the model least.
  return json({ pending: results ?? [] });
}

async function reviewSubmission(db, submissionId, body) {
  const { decision, admin_label, note, reviewed_by } = body;
  if (!["approved", "rejected"].includes(decision)) {
    return fail(400, "decision must be 'approved' or 'rejected'");
  }
  if (decision === "approved") {
    if (!VALID_LABELS.has(admin_label)) {
      return fail(400, `approving requires admin_label in ${[...VALID_LABELS].join(", ")}`);
    }
  }
  const row = await db
    .prepare("SELECT shared, review_status FROM submissions WHERE submission_id = ?")
    .bind(submissionId)
    .first();
  if (!row) return fail(404, "no such submission");
  if (row.review_status !== "pending") return fail(409, "already reviewed");
  if (decision === "approved" && row.shared !== 1) {
    return fail(400, "cannot approve a submission the user did not share");
  }

  await db
    .prepare(
      `UPDATE submissions
          SET review_status = ?, admin_label = ?, admin_note = ?, reviewed_at = ?, reviewed_by = ?
        WHERE submission_id = ?`
    )
    .bind(
      decision,
      decision === "approved" ? admin_label : null,
      note ? String(note).slice(0, 2000) : null,
      nowIso(),
      reviewed_by ?? "admin",
      submissionId
    )
    .run();
  return json({ submission_id: submissionId, review_status: decision, admin_label: admin_label ?? null });
}

/**
 * Export approved rows as a training batch.
 *
 * The WHERE clause is the third and last place the review gate is enforced
 * (schema trigger, review endpoint, here). `admin_label IS NOT NULL` is
 * redundant given the trigger — deliberately so. A redundant guard on the
 * query that feeds the training set is worth more than the line it costs.
 */
async function exportBatch(db, body) {
  const { batch_id, note, limit, dry_run } = body;
  const n = Math.min(Number(limit) || MAX_PAGE_SIZE, MAX_PAGE_SIZE);

  const { results } = await db
    .prepare(
      `SELECT submission_id, created_at, admin_label, image_b64, image_sha256,
              model_label, lesion_probability, checkpoint, user_says_wrong
         FROM submissions
        WHERE review_status = 'approved'
          AND admin_label IS NOT NULL
          AND shared = 1
          AND image_b64 IS NOT NULL
          AND batch_id IS NULL
        ORDER BY created_at ASC
        LIMIT ?`
    )
    .bind(n)
    .all();

  const rows = results ?? [];
  if (dry_run || rows.length === 0) {
    return json({ batch_id: null, count: rows.length, rows: dry_run ? rows : [], dry_run: true });
  }

  const id = batch_id || `batch-${nowIso().replace(/[:.]/g, "").replace("Z", "")}`;
  const stamp = nowIso();
  await db.batch([
    db
      .prepare("INSERT INTO batches (batch_id, created_at, note, row_count) VALUES (?, ?, ?, ?)")
      .bind(id, stamp, note ?? null, rows.length),
    ...rows.map((r) =>
      db
        .prepare("UPDATE submissions SET batch_id = ?, exported_at = ? WHERE submission_id = ?")
        .bind(id, stamp, r.submission_id)
    ),
  ]);
  return json({ batch_id: id, count: rows.length, rows });
}

// --- Router -----------------------------------------------------------------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (!env.DB) return fail(500, "D1 binding 'DB' is not configured");

    // Every route is authenticated. There are no public endpoints: this API
    // stores health data and must not be readable or writable by strangers.
    const key = bearer(request);
    const isAdmin = env.ADMIN_KEY ? timingSafeEqual(key, env.ADMIN_KEY) : false;
    const isApp = env.API_KEY ? timingSafeEqual(key, env.API_KEY) : false;
    if (!isAdmin && !isApp) return fail(401, "unauthorized");
    if (path.startsWith("/admin") && !isAdmin) return fail(403, "admin key required");

    const db = env.DB;
    const method = request.method.toUpperCase();

    try {
      if (method === "GET" && path === "/health") return await health(db);

      if (method === "POST" && path === "/users") return await createUser(db, await readJson(request));
      if (method === "GET" && path === "/users/by-email") {
        return await getUserByEmail(db, url.searchParams.get("email"));
      }

      if (method === "POST" && path === "/submissions") {
        return await createSubmission(db, await readJson(request));
      }
      if (method === "GET" && path === "/submissions") {
        return await listUserSubmissions(db, url.searchParams.get("user_id"), url.searchParams.get("limit"));
      }

      const feedback = path.match(/^\/submissions\/([^/]+)\/feedback$/);
      if (method === "POST" && feedback) {
        return await submitFeedback(db, feedback[1], await readJson(request));
      }

      if (method === "GET" && path === "/admin/pending") {
        return await pendingReview(
          db,
          url.searchParams.get("limit"),
          url.searchParams.get("images") === "1"
        );
      }
      const review = path.match(/^\/admin\/review\/([^/]+)$/);
      if (method === "POST" && review) {
        return await reviewSubmission(db, review[1], await readJson(request));
      }
      if (method === "POST" && path === "/admin/export") {
        return await exportBatch(db, await readJson(request));
      }

      return fail(404, `no route for ${method} ${path}`);
    } catch (err) {
      if (err && err.status) return fail(err.status, err.message);
      return fail(500, "internal error", { detail: String(err && err.message ? err.message : err) });
    }
  },
};
