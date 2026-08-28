var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// cloudflare/src/worker.js
var MAX_BODY_BYTES = 15e5;
var MAX_IMAGE_B64_BYTES = 6e5;
var MAX_TOTAL_BYTES_STORED = 2e8;
var MAX_SUBMISSIONS_PER_USER_PER_DAY = 50;
var MAX_USERS = 500;
var MAX_PAGE_SIZE = 100;
var LOCATION_TOKEN_TTL_MS = 5 * 60 * 1e3;
var K_ANONYMITY_MIN = 1;
var VALID_LABELS = /* @__PURE__ */ new Set(["normal", "benign", "malignant"]);
var REVIEW_LABELS = /* @__PURE__ */ new Set([...VALID_LABELS, "misc"]);
var BUCKETS = /* @__PURE__ */ new Set(["valid_bone", "misc", "contradiction"]);
var AUTH_PROVIDERS = /* @__PURE__ */ new Set(["password", "google"]);
var CONFIDENT_PROB = 0.65;
var ADMIN_USER_ID = "c2c5a209-4aaa-4eb9-b112-b2929b6dbe12";
var ADMIN_EMAIL = "kzfhero@gmail.com";
var json = /* @__PURE__ */ __name((data, status = 200) => new Response(JSON.stringify(data), {
  status,
  headers: { "content-type": "application/json; charset=utf-8" }
}), "json");
var fail = /* @__PURE__ */ __name((status, message, extra = {}) => json({ error: message, ...extra }, status), "fail");
var nowIso = /* @__PURE__ */ __name(() => (/* @__PURE__ */ new Date()).toISOString().replace(/\.\d{3}Z$/, "Z"), "nowIso");
var today = /* @__PURE__ */ __name(() => nowIso().slice(0, 10), "today");
function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
__name(timingSafeEqual, "timingSafeEqual");
function countryOf(request) {
  const code = request && request.cf && request.cf.country;
  if (typeof code !== "string") return null;
  const upper = code.toUpperCase();
  return /^[A-Z]{2}$/.test(upper) || upper === "T1" ? upper : null;
}
__name(countryOf, "countryOf");
function bearer(request) {
  const header = request.headers.get("authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7) : "";
}
__name(bearer, "bearer");
async function readJson(request) {
  const declared = Number(request.headers.get("content-length") || "0");
  if (declared > MAX_BODY_BYTES) {
    throw { status: 413, message: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) {
    throw { status: 413, message: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    throw { status: 400, message: "body is not valid JSON" };
  }
}
__name(readJson, "readJson");
function triageBucket({ oodFlagged, maxProb, userSaysWrong, userSuggestedLabel }) {
  const userSaysNotRadiograph = userSuggestedLabel === "misc";
  if (oodFlagged) {
    if (userSaysWrong && !userSaysNotRadiograph) {
      return { bucket: "contradiction", reason: "gate rejected it; the user says it is a radiograph" };
    }
    if (maxProb >= CONFIDENT_PROB) {
      return {
        bucket: "contradiction",
        reason: `gate rejected it but the classifier was ${maxProb.toFixed(2)} confident`
      };
    }
    return { bucket: "misc", reason: "the out-of-distribution gate rejected it" };
  }
  if (userSaysNotRadiograph) {
    return { bucket: "contradiction", reason: "gate accepted it; the user says it is not a radiograph" };
  }
  return { bucket: "valid_bone", reason: "the out-of-distribution gate accepted it" };
}
__name(triageBucket, "triageBucket");
function maxProbability(probabilities) {
  if (!probabilities || typeof probabilities !== "object") return 0;
  const values = Object.values(probabilities).map(Number).filter((v) => Number.isFinite(v));
  return values.length ? Math.max(...values) : 0;
}
__name(maxProbability, "maxProbability");
async function getMetaInt(db, key) {
  const row = await db.prepare("SELECT value FROM meta WHERE key = ?").bind(key).first();
  return row ? Number(row.value) : 0;
}
__name(getMetaInt, "getMetaInt");
async function health(db) {
  const users = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  const subs = await db.prepare("SELECT COUNT(*) AS n FROM submissions").first();
  const pending = await db.prepare(
    "SELECT COUNT(*) AS n FROM submissions WHERE shared = 1 AND review_status = 'pending'"
  ).first();
  const approved = await db.prepare("SELECT COUNT(*) AS n FROM submissions WHERE review_status = 'approved'").first();
  const bytes = await getMetaInt(db, "bytes_stored");
  const { results: byBucket } = await db.prepare(
    `SELECT triage_bucket, COUNT(*) AS n FROM submissions
        WHERE shared = 1 AND review_status = 'pending' GROUP BY triage_bucket`
  ).all();
  const buckets = { valid_bone: 0, misc: 0, contradiction: 0 };
  for (const row of byBucket ?? []) buckets[row.triage_bucket] = row.n;
  return json({
    ok: true,
    users: users.n,
    submissions: subs.n,
    pending_review: pending.n,
    pending_by_bucket: buckets,
    approved: approved.n,
    bytes_stored: bytes,
    // Surfaced so the app can warn before anything starts refusing writes.
    capacity_used: Number((bytes / MAX_TOTAL_BYTES_STORED).toFixed(4)),
    limits: {
      max_total_bytes_stored: MAX_TOTAL_BYTES_STORED,
      max_submissions_per_user_per_day: MAX_SUBMISSIONS_PER_USER_PER_DAY,
      max_users: MAX_USERS
    }
  });
}
__name(health, "health");
async function globe(db) {
  const { results: signupRows } = await db.prepare(
    `SELECT signup_country AS country, COUNT(*) AS n
         FROM users
        WHERE signup_country IS NOT NULL
        GROUP BY signup_country`
  ).all();
  const { results: contributorRows } = await db.prepare(
    `SELECT CASE WHEN u.country_captured_at IS NOT NULL THEN u.signup_country
                   ELSE COALESCE(s.origin_country, u.signup_country) END AS country,
              COUNT(DISTINCT s.user_id) AS n
         FROM submissions s
         JOIN users u ON u.user_id = s.user_id
        WHERE s.review_status = 'approved'
          AND (CASE WHEN u.country_captured_at IS NOT NULL THEN u.signup_country
                    ELSE COALESCE(s.origin_country, u.signup_country) END) IS NOT NULL
        GROUP BY CASE WHEN u.country_captured_at IS NOT NULL THEN u.signup_country
                      ELSE COALESCE(s.origin_country, u.signup_country) END`
  ).all();
  const split = /* @__PURE__ */ __name((rows) => {
    const plotted = [];
    let elsewhere = 0;
    let suppressed = 0;
    for (const row of rows ?? []) {
      if (row.n >= K_ANONYMITY_MIN) plotted.push({ country: row.country, count: row.n });
      else {
        elsewhere += row.n;
        suppressed += 1;
      }
    }
    plotted.sort((a, b) => b.count - a.count || a.country.localeCompare(b.country));
    return { plotted, elsewhere, suppressed_countries: suppressed };
  }, "split");
  const totalUsers = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  const totalContributors = await db.prepare(
    `SELECT COUNT(DISTINCT user_id) AS n FROM submissions WHERE review_status = 'approved'`
  ).first();
  const totalApproved = await db.prepare("SELECT COUNT(*) AS n FROM submissions WHERE review_status = 'approved'").first();
  const signups = split(signupRows);
  const contributors = split(contributorRows);
  return json({
    ok: true,
    // Headline figures for the landing page. Whole-population counts, so no
    // suppression applies and none is needed.
    totals: {
      users: totalUsers.n,
      contributors: totalContributors.n,
      approved_submissions: totalApproved.n,
      countries_represented: signups.plotted.length + signups.suppressed_countries
    },
    layers: { signups, contributors },
    // Stated in the payload so the client can label the map honestly rather
    // than implying the plotted dots are everyone.
    k_anonymity_min: K_ANONYMITY_MIN,
    generated_at: nowIso()
  });
}
__name(globe, "globe");
async function createUser(db, body) {
  const {
    user_id,
    email,
    password_hash,
    tos_accepted_at,
    is_admin,
    auth_provider,
    provider_subject,
    display_name,
    profile_picture_url
  } = body;
  const provider = auth_provider || "password";
  if (!user_id || !email) return fail(400, "user_id and email are required");
  if (!AUTH_PROVIDERS.has(provider)) {
    return fail(400, `auth_provider must be one of ${[...AUTH_PROVIDERS].join(", ")}`);
  }
  if (provider === "password") {
    if (!password_hash) return fail(400, "password_hash is required for a password account");
    if (!String(password_hash).startsWith("pbkdf2_sha256$")) {
      return fail(400, "password_hash must be a pbkdf2_sha256 encoded hash, not a password");
    }
    if (provider_subject) return fail(400, "a password account must not carry a provider_subject");
  } else {
    if (!provider_subject) return fail(400, `provider_subject is required for a ${provider} account`);
    if (password_hash) return fail(400, `a ${provider} account must not carry a password_hash`);
  }
  if (is_admin && user_id !== ADMIN_USER_ID) {
    return fail(403, "is_admin is pinned to a single account and cannot be granted here");
  }
  if (is_admin && String(email).toLowerCase() !== ADMIN_EMAIL) {
    return fail(403, `the admin account is ${ADMIN_EMAIL}`);
  }
  const count = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  if (count.n >= MAX_USERS) return fail(507, `user cap reached (${MAX_USERS})`);
  const created = nowIso();
  try {
    await db.prepare(
      `INSERT INTO users (user_id, email, password_hash, auth_provider, provider_subject,
                            created_at, tos_accepted_at, is_admin, signup_country,
                            display_name, profile_picture_url)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      user_id,
      String(email).toLowerCase(),
      provider === "password" ? password_hash : null,
      provider,
      provider === "password" ? null : String(provider_subject),
      created,
      tos_accepted_at || created,
      is_admin ? 1 : 0,
      // Filled only after the signed-in browser reaches /location/capture.
      null,
      provider === "google" ? cleanDisplayName(display_name) : null,
      provider === "google" ? cleanGooglePicture(profile_picture_url) : null
    ).run();
  } catch (err) {
    if (String(err).includes("UNIQUE")) return fail(409, "an account already exists for that email");
    throw err;
  }
  return json({ user_id, email, created_at: created, auth_provider: provider }, 201);
}
__name(createUser, "createUser");
async function getUserBySubject(db, subject) {
  if (!subject) return fail(400, "subject query parameter is required");
  const row = await db.prepare(
    `SELECT user_id, email, password_hash, auth_provider, provider_subject,
              created_at, tos_accepted_at, is_admin, display_name,
              profile_picture_url, public_contributor_profile
         FROM users WHERE provider_subject = ?`
  ).bind(String(subject)).first();
  return row ? json(row) : fail(404, "no such user");
}
__name(getUserBySubject, "getUserBySubject");
async function getUserByEmail(db, email) {
  if (!email) return fail(400, "email query parameter is required");
  const row = await db.prepare(
    `SELECT user_id, email, password_hash, auth_provider, provider_subject,
              created_at, tos_accepted_at, is_admin, display_name,
              profile_picture_url, public_contributor_profile
       FROM users WHERE email = ? COLLATE NOCASE`
  ).bind(String(email).toLowerCase()).first();
  return row ? json(row) : fail(404, "no such user");
}
__name(getUserByEmail, "getUserByEmail");
function withLocationCors(response) {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "POST, OPTIONS");
  headers.set("access-control-allow-headers", "authorization, content-type");
  headers.set("access-control-max-age", "600");
  return new Response(response.body, { status: response.status, headers });
}
__name(withLocationCors, "withLocationCors");
async function tokenHash(token) {
  const bytes = new TextEncoder().encode(token);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
__name(tokenHash, "tokenHash");
async function issueLocationToken(db, body) {
  const userId = String(body.user_id || "");
  if (!userId) return fail(400, "user_id is required");
  const user = await db.prepare("SELECT user_id FROM users WHERE user_id = ?").bind(userId).first();
  if (!user) return fail(404, "no such user");
  const token = crypto.randomUUID();
  const hash = await tokenHash(token);
  const issuedAt = (/* @__PURE__ */ new Date()).toISOString();
  const expiresAt = new Date(Date.now() + LOCATION_TOKEN_TTL_MS).toISOString();
  await db.batch([
    db.prepare("DELETE FROM location_capture_tokens WHERE expires_at < ? OR used_at IS NOT NULL").bind(issuedAt),
    db.prepare(
      `INSERT INTO location_capture_tokens (token_hash, user_id, expires_at)
         VALUES (?, ?, ?)`
    ).bind(hash, userId, expiresAt)
  ]);
  return json({ ok: true, token, expires_at: expiresAt });
}
__name(issueLocationToken, "issueLocationToken");
async function captureBrowserCountry(db, request) {
  const country = countryOf(request);
  if (!country || country === "XX" || country === "T1") {
    return withLocationCors(fail(422, "country could not be resolved at the Cloudflare edge"));
  }
  const token = bearer(request);
  if (!token) return withLocationCors(fail(401, "location token required"));
  const hash = await tokenHash(token);
  const capturedAt = (/* @__PURE__ */ new Date()).toISOString();
  const captureAttempt = crypto.randomUUID();
  const row = await db.prepare(
    `SELECT user_id FROM location_capture_tokens
        WHERE token_hash = ? AND used_at IS NULL AND expires_at >= ?`
  ).bind(hash, capturedAt).first();
  if (!row) return withLocationCors(fail(401, "location token is invalid, expired, or used"));
  const results = await db.batch([
    db.prepare(
      `UPDATE location_capture_tokens SET used_at = ?, used_nonce = ?
          WHERE token_hash = ? AND used_at IS NULL`
    ).bind(capturedAt, captureAttempt, hash),
    db.prepare(
      `UPDATE users
            SET signup_country = ?, country_captured_at = ?
          WHERE user_id = ? AND country_captured_at IS NULL
            AND EXISTS (
              SELECT 1 FROM location_capture_tokens
               WHERE token_hash = ? AND used_nonce = ?
            )`
    ).bind(country, capturedAt, String(row.user_id), hash, captureAttempt),
    // Repair historical submissions that inherited the Streamlit server's
    // country. New submissions copy the already-captured account country.
    db.prepare(
      `UPDATE submissions SET origin_country = ?
          WHERE user_id = ?
            AND EXISTS (
              SELECT 1 FROM users
               WHERE user_id = ? AND country_captured_at = ?
            )
            AND EXISTS (
              SELECT 1 FROM location_capture_tokens
               WHERE token_hash = ? AND used_nonce = ?
            )`
    ).bind(
      country,
      String(row.user_id),
      String(row.user_id),
      capturedAt,
      hash,
      captureAttempt
    )
  ]);
  if (!results[0].meta || results[0].meta.changes !== 1) {
    return withLocationCors(fail(409, "location token has already been used"));
  }
  return withLocationCors(json({ ok: true, country_recorded: true }));
}
__name(captureBrowserCountry, "captureBrowserCountry");
function cleanDisplayName(value) {
  const name = String(value || "").trim();
  return name ? name.slice(0, 80) : null;
}
__name(cleanDisplayName, "cleanDisplayName");
function cleanGooglePicture(value) {
  const raw = String(value || "").trim();
  if (!raw || raw.length > 2048) return null;
  try {
    const parsed = new URL(raw);
    const host = parsed.hostname.toLowerCase();
    if (parsed.protocol !== "https:") return null;
    if (host !== "googleusercontent.com" && !host.endsWith(".googleusercontent.com")) return null;
    return parsed.href;
  } catch (_) {
    return null;
  }
}
__name(cleanGooglePicture, "cleanGooglePicture");
async function updateContributorProfile(db, body) {
  const { user_id, provider_subject, display_name, profile_picture_url } = body;
  if (!user_id || !provider_subject) return fail(400, "user_id and provider_subject are required");
  const user = await db.prepare("SELECT auth_provider, provider_subject FROM users WHERE user_id = ?").bind(String(user_id)).first();
  if (!user) return fail(404, "no such user");
  if (user.auth_provider !== "google" || !timingSafeEqual(user.provider_subject || "", String(provider_subject))) {
    return fail(403, "profile identity does not match the Google account");
  }
  const publicProfile = Object.prototype.hasOwnProperty.call(body, "public_profile") ? body.public_profile ? 1 : 0 : null;
  if (publicProfile === null) {
    return json({ ok: true });
  }
  const storedName = publicProfile === 0 ? null : cleanDisplayName(display_name);
  const storedPicture = publicProfile === 0 ? null : cleanGooglePicture(profile_picture_url);
  await db.prepare(
    `UPDATE users
          SET display_name = ?, profile_picture_url = ?,
              public_contributor_profile = ?
        WHERE user_id = ?`
  ).bind(
    storedName,
    storedPicture,
    publicProfile,
    String(user_id)
  ).run();
  return json({ ok: true, public_contributor_profile: publicProfile });
}
__name(updateContributorProfile, "updateContributorProfile");
async function listContributors(db) {
  const { results } = await db.prepare(
    `SELECT u.display_name AS name, u.profile_picture_url AS picture,
              COUNT(s.submission_id) AS approved_contributions
         FROM users u
         JOIN submissions s ON s.user_id = u.user_id
        WHERE u.public_contributor_profile = 1
          AND u.display_name IS NOT NULL
          AND s.review_status = 'approved'
        GROUP BY u.user_id, u.display_name, u.profile_picture_url
        ORDER BY approved_contributions DESC, u.display_name COLLATE NOCASE ASC
        LIMIT 24`
  ).all();
  return json({ contributors: results ?? [] });
}
__name(listContributors, "listContributors");
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
    image_sha256
  } = body;
  if (!submission_id || !user_id || !model_label) {
    return fail(400, "submission_id, user_id and model_label are required");
  }
  const user = await db.prepare("SELECT user_id, signup_country FROM users WHERE user_id = ?").bind(user_id).first();
  if (!user) return fail(404, "no such user");
  const day = today();
  const rl = await db.prepare("SELECT submissions FROM rate_limit WHERE user_id = ? AND day = ?").bind(user_id, day).first();
  if (rl && rl.submissions >= MAX_SUBMISSIONS_PER_USER_PER_DAY) {
    return fail(429, `daily submission limit reached (${MAX_SUBMISSIONS_PER_USER_PER_DAY})`);
  }
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
        cap: MAX_TOTAL_BYTES_STORED
      });
    }
    storedImage = image_b64;
    imageBytes = image_b64.length;
  }
  const triage = triageBucket({
    oodFlagged: Boolean(ood_flagged),
    maxProb: maxProbability(class_probabilities),
    userSaysWrong: false,
    userSuggestedLabel: null
  });
  const created = nowIso();
  const statements = [
    db.prepare(
      `INSERT INTO submissions (
            submission_id, user_id, created_at, model_label, lesion_probability,
            class_probabilities, checkpoint, threshold, calibrated,
            ood_flagged, ood_score, shared, consent_at, image_b64, image_sha256, image_bytes,
            triage_bucket, triage_reason, origin_country
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
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
      storedImage ? image_sha256 ?? null : null,
      imageBytes,
      triage.bucket,
      triage.reason,
      user.signup_country ?? null
    ),
    db.prepare(
      `INSERT INTO rate_limit (user_id, day, submissions) VALUES (?, ?, 1)
         ON CONFLICT(user_id, day) DO UPDATE SET submissions = submissions + 1`
    ).bind(user_id, day)
  ];
  if (imageBytes > 0) {
    statements.push(
      db.prepare(
        "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = 'bytes_stored'"
      ).bind(imageBytes)
    );
  }
  await db.batch(statements);
  return json(
    {
      submission_id,
      created_at: created,
      shared: isShared === 1,
      triage_bucket: triage.bucket,
      triage_reason: triage.reason
    },
    201
  );
}
__name(createSubmission, "createSubmission");
async function submitFeedback(db, submissionId, body) {
  const { user_id, says_wrong, suggested_label, comment } = body;
  if (!user_id) return fail(400, "user_id is required");
  if (suggested_label && !REVIEW_LABELS.has(suggested_label)) {
    return fail(400, `suggested_label must be one of ${[...REVIEW_LABELS].join(", ")}`);
  }
  const row = await db.prepare(
    `SELECT user_id, review_status, ood_flagged, class_probabilities
         FROM submissions WHERE submission_id = ?`
  ).bind(submissionId).first();
  if (!row) return fail(404, "no such submission");
  if (row.user_id !== user_id) return fail(403, "not your submission");
  if (row.review_status !== "pending") {
    return fail(409, "this submission has already been reviewed");
  }
  let parsedProbabilities = {};
  try {
    parsedProbabilities = JSON.parse(row.class_probabilities || "{}");
  } catch {
    parsedProbabilities = {};
  }
  const triage = triageBucket({
    oodFlagged: Boolean(row.ood_flagged),
    maxProb: maxProbability(parsedProbabilities),
    userSaysWrong: Boolean(says_wrong),
    userSuggestedLabel: suggested_label ?? null
  });
  await db.prepare(
    `UPDATE submissions
         SET user_says_wrong = ?, user_suggested_label = ?, user_comment = ?, feedback_at = ?,
             triage_bucket = ?, triage_reason = ?
       WHERE submission_id = ?`
  ).bind(
    says_wrong ? 1 : 0,
    suggested_label ?? null,
    comment ? String(comment).slice(0, 2e3) : null,
    nowIso(),
    triage.bucket,
    triage.reason,
    submissionId
  ).run();
  return json({
    submission_id: submissionId,
    recorded: true,
    triage_bucket: triage.bucket,
    triage_reason: triage.reason
  });
}
__name(submitFeedback, "submitFeedback");
async function listUserSubmissions(db, userId, limit) {
  if (!userId) return fail(400, "user_id is required");
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  const { results } = await db.prepare(
    `SELECT submission_id, created_at, model_label, lesion_probability, checkpoint,
              ood_flagged, shared, user_says_wrong, user_suggested_label,
              triage_bucket, review_status, admin_bucket, admin_label
         FROM submissions WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?`
  ).bind(userId, n).all();
  return json({ submissions: results ?? [] });
}
__name(listUserSubmissions, "listUserSubmissions");
async function pendingReview(db, limit, includeImages, bucket) {
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  if (bucket && !BUCKETS.has(bucket)) {
    return fail(400, `bucket must be one of ${[...BUCKETS].join(", ")}`);
  }
  const columns = `submission_id, user_id, created_at, model_label, lesion_probability,
                   class_probabilities, ood_flagged, ood_score,
                   triage_bucket, triage_reason,
                   user_says_wrong, user_suggested_label, user_comment${includeImages ? ", image_b64" : ""}`;
  const { results } = await db.prepare(
    `SELECT ${columns} FROM submissions
        WHERE shared = 1 AND review_status = 'pending'
          AND (? IS NULL OR triage_bucket = ?)
        ORDER BY user_says_wrong DESC, created_at ASC
        LIMIT ?`
  ).bind(bucket ?? null, bucket ?? null, n).all();
  return json({ pending: results ?? [], bucket: bucket ?? null });
}
__name(pendingReview, "pendingReview");
async function reviewSubmission(db, submissionId, body) {
  const { decision, admin_label, admin_bucket, note, reviewed_by } = body;
  if (!["approved", "rejected"].includes(decision)) {
    return fail(400, "decision must be 'approved' or 'rejected'");
  }
  if (decision === "approved") {
    if (!REVIEW_LABELS.has(admin_label)) {
      return fail(400, `approving requires admin_label in ${[...REVIEW_LABELS].join(", ")}`);
    }
    if (!BUCKETS.has(admin_bucket)) {
      return fail(400, `approving requires admin_bucket in ${[...BUCKETS].join(", ")}`);
    }
    if (admin_bucket === "misc" && admin_label !== "misc") {
      return fail(400, "a misc row has no diagnosis: label it 'misc'");
    }
    if (admin_bucket === "valid_bone" && admin_label === "misc") {
      return fail(400, "a bone radiograph needs a clinical label, not 'misc'");
    }
  }
  const row = await db.prepare("SELECT shared, review_status, image_b64 FROM submissions WHERE submission_id = ?").bind(submissionId).first();
  if (!row) return fail(404, "no such submission");
  if (row.review_status !== "pending") return fail(409, "already reviewed");
  if (decision === "approved" && row.shared !== 1) {
    return fail(400, "cannot approve a submission the user did not share");
  }
  if (decision === "approved" && !row.image_b64) {
    return fail(400, "cannot approve a row with no image: there is nothing to train on");
  }
  await db.batch([
    db.prepare(
      `UPDATE submissions SET admin_label = ?, admin_bucket = ? WHERE submission_id = ?`
    ).bind(
      decision === "approved" ? admin_label : null,
      decision === "approved" ? admin_bucket : null,
      submissionId
    ),
    db.prepare(
      `UPDATE submissions
            SET review_status = ?, admin_note = ?, reviewed_at = ?, reviewed_by = ?
          WHERE submission_id = ?`
    ).bind(
      decision,
      note ? String(note).slice(0, 2e3) : null,
      nowIso(),
      reviewed_by || ADMIN_USER_ID,
      submissionId
    )
  ]);
  return json({
    submission_id: submissionId,
    review_status: decision,
    admin_label: decision === "approved" ? admin_label : null,
    admin_bucket: decision === "approved" ? admin_bucket : null
  });
}
__name(reviewSubmission, "reviewSubmission");
async function exportBatch(db, body) {
  const { batch_id, note, limit, dry_run } = body;
  const n = Math.min(Number(limit) || MAX_PAGE_SIZE, MAX_PAGE_SIZE);
  const { results } = await db.prepare(
    `SELECT submission_id, created_at, admin_bucket, admin_label, admin_note,
              image_b64, image_sha256, model_label, lesion_probability, checkpoint,
              ood_flagged, ood_score, triage_bucket, user_says_wrong, user_suggested_label
         FROM submissions
        WHERE review_status = 'approved'
          AND admin_label IS NOT NULL
          AND admin_bucket IS NOT NULL
          AND shared = 1
          AND image_b64 IS NOT NULL
          AND batch_id IS NULL
        ORDER BY created_at ASC
        LIMIT ?`
  ).bind(n).all();
  const rows = results ?? [];
  const lesionRows = rows.filter((r) => r.admin_label !== "misc").length;
  const oodRows = rows.length - lesionRows;
  if (dry_run || rows.length === 0) {
    return json({
      batch_id: null,
      count: rows.length,
      lesion_rows: lesionRows,
      ood_rows: oodRows,
      rows: dry_run ? rows : [],
      dry_run: true
    });
  }
  const id = batch_id || `batch-${nowIso().replace(/[:.]/g, "").replace("Z", "")}`;
  const stamp = nowIso();
  await db.batch([
    db.prepare(
      `INSERT INTO batches (batch_id, created_at, note, row_count, lesion_rows, ood_rows)
         VALUES (?, ?, ?, ?, ?, ?)`
    ).bind(id, stamp, note ?? null, rows.length, lesionRows, oodRows),
    ...rows.map(
      (r) => db.prepare("UPDATE submissions SET batch_id = ?, exported_at = ? WHERE submission_id = ?").bind(id, stamp, r.submission_id)
    )
  ]);
  return json({ batch_id: id, count: rows.length, lesion_rows: lesionRows, ood_rows: oodRows, rows });
}
__name(exportBatch, "exportBatch");
var worker = {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    if (!env.DB) return fail(500, "D1 binding 'DB' is not configured");
    if (path === "/location/capture" && request.method.toUpperCase() === "OPTIONS") {
      return withLocationCors(new Response(null, { status: 204 }));
    }
    if (path === "/location/capture" && request.method.toUpperCase() === "POST") {
      try {
        return await captureBrowserCountry(env.DB, request);
      } catch (_) {
        return withLocationCors(fail(500, "internal error"));
      }
    }
    const key = bearer(request);
    const isAdmin = env.ADMIN_KEY ? timingSafeEqual(key, env.ADMIN_KEY) : false;
    const isApp = env.API_KEY ? timingSafeEqual(key, env.API_KEY) : false;
    if (!isAdmin && !isApp) return fail(401, "unauthorized");
    if (path.startsWith("/admin")) {
      if (!isAdmin) return fail(403, "admin key required");
      const actor = request.headers.get("x-onnm-admin-user") || "";
      if (!timingSafeEqual(actor, ADMIN_USER_ID)) {
        return fail(403, "review is restricted to the owning account");
      }
    }
    const db = env.DB;
    const method = request.method.toUpperCase();
    try {
      if (method === "GET" && path === "/health") return await health(db);
      if (method === "GET" && path === "/globe") return await globe(db);
      if (method === "GET" && path === "/contributors") return await listContributors(db);
      if (method === "POST" && path === "/location/token") {
        return await issueLocationToken(db, await readJson(request));
      }
      if (method === "POST" && path === "/users") {
        return await createUser(db, await readJson(request));
      }
      if (method === "GET" && path === "/users/by-email") {
        return await getUserByEmail(db, url.searchParams.get("email"));
      }
      if (method === "GET" && path === "/users/by-subject") {
        return await getUserBySubject(db, url.searchParams.get("subject"));
      }
      if (method === "POST" && path === "/users/profile") {
        return await updateContributorProfile(db, await readJson(request));
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
          url.searchParams.get("images") === "1",
          url.searchParams.get("bucket") || null
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
  }
};
var handleApiRequest = /* @__PURE__ */ __name((request, env) => worker.fetch(request, env), "handleApiRequest");

// node_modules/@cloudflare/containers/dist/lib/helpers.js
function generateId(length = 9) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  let result = "";
  for (let i = 0; i < length; i++) {
    result += alphabet[bytes[i] % alphabet.length];
  }
  return result;
}
__name(generateId, "generateId");
function parseTimeExpression(timeExpression) {
  if (typeof timeExpression === "number") {
    return timeExpression;
  }
  if (typeof timeExpression === "string") {
    const match = timeExpression.match(/^(\d+)([smh])$/);
    if (!match) {
      throw new Error(`invalid time expression ${timeExpression}`);
    }
    const value = parseInt(match[1]);
    const unit = match[2];
    switch (unit) {
      case "s":
        return value;
      case "m":
        return value * 60;
      case "h":
        return value * 60 * 60;
      default:
        throw new Error(`unknown time unit ${unit}`);
    }
  }
  throw new Error(`invalid type for a time expression: ${typeof timeExpression}`);
}
__name(parseTimeExpression, "parseTimeExpression");

// node_modules/@cloudflare/containers/dist/lib/container.js
import { DurableObject } from "cloudflare:workers";
var NO_CONTAINER_INSTANCE_ERROR = "there is no container instance that can be provided to this durable object";
var RUNTIME_SIGNALLED_ERROR = "runtime signalled the container to exit:";
var UNEXPECTED_EXIT_ERROR = "container exited with unexpected exit code:";
var NOT_LISTENING_ERROR = "the container is not listening";
var CONTAINER_STATE_KEY = "__CF_CONTAINER_STATE";
var MAX_ALARM_RETRIES = 3;
var PING_TIMEOUT_MS = 5e3;
var DEFAULT_SLEEP_AFTER = "10m";
var INSTANCE_POLL_INTERVAL_MS = 300;
var TIMEOUT_TO_GET_CONTAINER_MS = 8e3;
var TIMEOUT_TO_GET_PORTS_MS = 2e4;
var FALLBACK_PORT_TO_CHECK = 33;
var signalToNumbers = {
  SIGINT: 2,
  SIGTERM: 15,
  SIGKILL: 9
};
function isErrorOfType(e, matchingString) {
  const errorString = e instanceof Error ? e.message : String(e);
  return errorString.toLowerCase().includes(matchingString);
}
__name(isErrorOfType, "isErrorOfType");
var isNoInstanceError = /* @__PURE__ */ __name((error) => isErrorOfType(error, NO_CONTAINER_INSTANCE_ERROR), "isNoInstanceError");
var isRuntimeSignalledError = /* @__PURE__ */ __name((error) => isErrorOfType(error, RUNTIME_SIGNALLED_ERROR), "isRuntimeSignalledError");
var isNotListeningError = /* @__PURE__ */ __name((error) => isErrorOfType(error, NOT_LISTENING_ERROR), "isNotListeningError");
var isContainerExitNonZeroError = /* @__PURE__ */ __name((error) => isErrorOfType(error, UNEXPECTED_EXIT_ERROR), "isContainerExitNonZeroError");
function getExitCodeFromError(error) {
  if (!(error instanceof Error)) {
    return null;
  }
  if (isRuntimeSignalledError(error)) {
    return +error.message.toLowerCase().slice(error.message.toLowerCase().indexOf(RUNTIME_SIGNALLED_ERROR) + RUNTIME_SIGNALLED_ERROR.length + 1);
  }
  if (isContainerExitNonZeroError(error)) {
    return +error.message.toLowerCase().slice(error.message.toLowerCase().indexOf(UNEXPECTED_EXIT_ERROR) + UNEXPECTED_EXIT_ERROR.length + 1);
  }
  return null;
}
__name(getExitCodeFromError, "getExitCodeFromError");
function addTimeoutSignal(existingSignal, timeoutMs) {
  const controller = new AbortController();
  if (existingSignal?.aborted) {
    controller.abort();
    return controller.signal;
  }
  existingSignal?.addEventListener("abort", () => controller.abort());
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  controller.signal.addEventListener("abort", () => clearTimeout(timeoutId));
  return controller.signal;
}
__name(addTimeoutSignal, "addTimeoutSignal");
var ContainerState = class {
  static {
    __name(this, "ContainerState");
  }
  storage;
  status;
  constructor(storage) {
    this.storage = storage;
  }
  async setRunning() {
    await this.setStatusAndupdate("running");
  }
  async setHealthy() {
    await this.setStatusAndupdate("healthy");
  }
  async setStopping() {
    await this.setStatusAndupdate("stopping");
  }
  async setStopped() {
    await this.setStatusAndupdate("stopped");
  }
  async setStoppedWithCode(exitCode) {
    this.status = { status: "stopped_with_code", lastChange: Date.now(), exitCode };
    await this.update();
  }
  async getState() {
    if (!this.status) {
      const state = await this.storage.get(CONTAINER_STATE_KEY);
      if (!state) {
        this.status = {
          status: "stopped",
          lastChange: Date.now()
        };
        await this.update();
      } else {
        this.status = state;
      }
    }
    return this.status;
  }
  async setStatusAndupdate(status) {
    this.status = { status, lastChange: Date.now() };
    await this.update();
  }
  async update() {
    if (!this.status)
      throw new Error("status should be init");
    await this.storage.put(CONTAINER_STATE_KEY, this.status);
  }
};
var Container = class extends DurableObject {
  static {
    __name(this, "Container");
  }
  // =========================
  //     Public Attributes
  // =========================
  // Default port for the container (undefined means no default port)
  defaultPort;
  // Required ports that should be checked for availability during container startup
  // Override this in your subclass to specify ports that must be ready
  requiredPorts;
  // Timeout after which the container will sleep if no activity
  // The signal sent to the container by default is a SIGTERM.
  // The container won't get a SIGKILL if this threshold is triggered.
  sleepAfter = DEFAULT_SLEEP_AFTER;
  // Container configuration properties
  // Set these properties directly in your container instance
  envVars = {};
  entrypoint;
  enableInternet = true;
  // pingEndpoint is the host and path value that the class will use to send a request to the container and check if the
  // instance is ready.
  //
  // The user does not have to implement this route by any means,
  // but it's still useful if you want to control the path that
  // the Container class uses to send HTTP requests to.
  pingEndpoint = "ping";
  // =========================
  //     PUBLIC INTERFACE
  // =========================
  constructor(ctx, env, options) {
    super(ctx, env);
    if (ctx.container === void 0) {
      throw new Error("Containers have not been enabled for this Durable Object class. Have you correctly setup your Wrangler config? More info: https://developers.cloudflare.com/containers/get-started/#configuration");
    }
    this.state = new ContainerState(this.ctx.storage);
    this.ctx.blockConcurrencyWhile(async () => {
      this.renewActivityTimeout();
      await this.scheduleNextAlarm();
    });
    this.container = ctx.container;
    if (options) {
      if (options.defaultPort !== void 0)
        this.defaultPort = options.defaultPort;
      if (options.sleepAfter !== void 0)
        this.sleepAfter = options.sleepAfter;
    }
    this.sql`
      CREATE TABLE IF NOT EXISTS container_schedules (
        id TEXT PRIMARY KEY NOT NULL DEFAULT (randomblob(9)),
        callback TEXT NOT NULL,
        payload TEXT,
        type TEXT NOT NULL CHECK(type IN ('scheduled', 'delayed')),
        time INTEGER NOT NULL,
        delayInSeconds INTEGER,
        created_at INTEGER DEFAULT (unixepoch())
      )
    `;
    if (this.container.running) {
      this.monitor = this.container.monitor();
      this.setupMonitorCallbacks();
    }
  }
  /**
   * Gets the current state of the container
   * @returns Promise<State>
   */
  async getState() {
    return { ...await this.state.getState() };
  }
  // ==========================
  //     CONTAINER STARTING
  // ==========================
  /**
   * Start the container if it's not running and set up monitoring and lifecycle hooks,
   * without waiting for ports to be ready.
   *
   * It will automatically retry if the container fails to start, using the specified waitOptions
   *
   *
   * @example
   * await this.start({
   *   envVars: { DEBUG: 'true', NODE_ENV: 'development' },
   *   entrypoint: ['npm', 'run', 'dev'],
   *   enableInternet: false
   * });
   *
   * @param startOptions - Override `envVars`, `entrypoint` and `enableInternet` on a per-instance basis
   * @param waitOptions - Optional wait configuration with abort signal for cancellation. Default ~8s timeout.
   * @returns A promise that resolves when the container start command has been issued
   * @throws Error if no container context is available or if all start attempts fail
   */
  async start(startOptions, waitOptions) {
    const portToCheck = waitOptions?.portToCheck ?? this.defaultPort ?? (this.requiredPorts ? this.requiredPorts[0] : FALLBACK_PORT_TO_CHECK);
    const pollInterval = waitOptions?.waitInterval ?? INSTANCE_POLL_INTERVAL_MS;
    await this.startContainerIfNotRunning({
      signal: waitOptions?.signal,
      waitInterval: pollInterval,
      retries: waitOptions?.retries ?? Math.ceil(TIMEOUT_TO_GET_CONTAINER_MS / pollInterval),
      portToCheck
    }, startOptions);
    this.setupMonitorCallbacks();
    await this.ctx.blockConcurrencyWhile(async () => {
      await this.onStart();
    });
  }
  async startAndWaitForPorts(portsOrArgs, cancellationOptions, startOptions) {
    let ports;
    let resolvedCancellationOptions = {};
    let resolvedStartOptions = {};
    if (typeof portsOrArgs === "object" && portsOrArgs !== null && !Array.isArray(portsOrArgs)) {
      ports = portsOrArgs.ports;
      resolvedCancellationOptions = portsOrArgs.cancellationOptions;
      resolvedStartOptions = portsOrArgs.startOptions;
    } else {
      ports = portsOrArgs;
      resolvedCancellationOptions = cancellationOptions;
      resolvedStartOptions = startOptions;
    }
    const portsToCheck = await this.getPortsToCheck(ports);
    await this.syncPendingStoppedEvents();
    resolvedCancellationOptions ??= {};
    const containerGetTimeout = resolvedCancellationOptions.instanceGetTimeoutMS ?? TIMEOUT_TO_GET_CONTAINER_MS;
    const pollInterval = resolvedCancellationOptions.waitInterval ?? INSTANCE_POLL_INTERVAL_MS;
    let containerGetRetries = Math.ceil(containerGetTimeout / pollInterval);
    const waitOptions = {
      signal: resolvedCancellationOptions.abort,
      retries: containerGetRetries,
      waitInterval: pollInterval,
      portToCheck: portsToCheck[0]
    };
    const triesUsed = await this.startContainerIfNotRunning(waitOptions, resolvedStartOptions);
    const totalPortReadyTries = Math.ceil((resolvedCancellationOptions.portReadyTimeoutMS ?? TIMEOUT_TO_GET_PORTS_MS) / pollInterval);
    let triesLeft = totalPortReadyTries - triesUsed;
    for (const port of portsToCheck) {
      triesLeft = await this.waitForPort({
        signal: resolvedCancellationOptions.abort,
        waitInterval: pollInterval,
        retries: triesLeft,
        portToCheck: port
      });
    }
    this.setupMonitorCallbacks();
    await this.ctx.blockConcurrencyWhile(async () => {
      await this.state.setHealthy();
      await this.onStart();
    });
  }
  /**
   *
   * Waits for a specified port to be ready
   *
   * Returns the number of tries used to get the port, or throws if it couldn't get the port within the specified retry limits.
   *
   * @param waitOptions -
   * - `portToCheck`: The port number to check
   * - `abort`: Optional AbortSignal to cancel waiting
   * - `retries`: Number of retries before giving up (default: TRIES_TO_GET_PORTS)
   * - `waitInterval`: Interval between retries in milliseconds (default: INSTANCE_POLL_INTERVAL_MS)
   */
  async waitForPort(waitOptions) {
    const port = waitOptions.portToCheck;
    const tcpPort = this.container.getTcpPort(port);
    const abortedSignal = new Promise((res) => {
      waitOptions.signal?.addEventListener("abort", () => {
        res(true);
      });
    });
    const pollInterval = waitOptions.waitInterval ?? INSTANCE_POLL_INTERVAL_MS;
    let tries = waitOptions.retries ?? Math.ceil(TIMEOUT_TO_GET_PORTS_MS / pollInterval);
    for (let i = 0; i < tries; i++) {
      try {
        const combinedSignal = addTimeoutSignal(waitOptions.signal, PING_TIMEOUT_MS);
        await tcpPort.fetch(`http://${this.pingEndpoint}`, { signal: combinedSignal });
        console.log(`Port ${port} is ready`);
        break;
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        console.debug(`Error checking ${port}: ${errorMessage}`);
        if (!this.container.running) {
          try {
            await this.onError(new Error(`Container crashed while checking for ports, did you start the container and setup the entrypoint correctly?`));
          } catch {
          }
          throw e;
        }
        if (i === tries - 1) {
          try {
            await this.onError(`Failed to verify port ${port} is available after ${(i + 1) * pollInterval}ms, last error: ${errorMessage}`);
          } catch {
          }
          throw e;
        }
        await Promise.any([
          new Promise((resolve) => setTimeout(resolve, waitOptions.waitInterval)),
          abortedSignal
        ]);
        if (waitOptions.signal?.aborted) {
          throw new Error("Container request aborted.");
        }
      }
    }
    return tries;
  }
  // =======================
  //     LIFECYCLE HOOKS
  // =======================
  /**
   * Send a signal to the container.
   * @param signal - The signal to send to the container (default: 15 for SIGTERM)
   */
  async stop(signal = "SIGTERM") {
    if (this.container.running) {
      this.container.signal(typeof signal === "string" ? signalToNumbers[signal] : signal);
    }
    await this.syncPendingStoppedEvents();
  }
  /**
   * Destroys the container with a SIGKILL. Triggers onStop.
   */
  async destroy() {
    await this.container.destroy();
  }
  /**
   * Lifecycle method called when container starts successfully
   * Override this method in subclasses to handle container start events
   */
  onStart() {
  }
  /**
   * Lifecycle method called when container shuts down
   * Override this method in subclasses to handle Container stopped events
   * @param params - Object containing exitCode and reason for the stop
   */
  onStop(_) {
  }
  /**
   * Lifecycle method called when the container is running, and the activity timeout
   * expiration (set by `sleepAfter`) has been reached.
   *
   * If you want to shutdown the container, you should call this.stop() here
   *
   * By default, this method calls `this.stop()`
   */
  async onActivityExpired() {
    if (!this.container.running) {
      return;
    }
    await this.stop();
  }
  /**
   * Error handler for container errors
   * Override this method in subclasses to handle container errors
   * @param error - The error that occurred
   * @returns Can return any value or throw the error
   */
  onError(error) {
    console.error("Container error:", error);
    throw error;
  }
  /**
   * Renew the container's activity timeout
   *
   * Call this method whenever there is activity on the container
   */
  renewActivityTimeout() {
    const timeoutInMs = parseTimeExpression(this.sleepAfter) * 1e3;
    this.sleepAfterMs = Date.now() + timeoutInMs;
  }
  // ==================
  //     SCHEDULING
  // ==================
  /**
   * Schedule a task to be executed in the future.
   *
   * We strongly recommend using this instead of the `alarm` handler.
   *
   * @template T Type of the payload data
   * @param when When to execute the task (Date object or number of seconds delay)
   * @param callback Name of the method to call
   * @param payload Data to pass to the callback
   * @returns Schedule object representing the scheduled task
   */
  async schedule(when, callback, payload) {
    const id = generateId(9);
    if (typeof callback !== "string") {
      throw new Error("Callback must be a string (method name)");
    }
    if (typeof this[callback] !== "function") {
      throw new Error(`this.${callback} is not a function`);
    }
    if (when instanceof Date) {
      const timestamp = Math.floor(when.getTime() / 1e3);
      this.sql`
        INSERT OR REPLACE INTO container_schedules (id, callback, payload, type, time)
        VALUES (${id}, ${callback}, ${JSON.stringify(payload)}, 'scheduled', ${timestamp})
      `;
      await this.scheduleNextAlarm();
      return {
        taskId: id,
        callback,
        payload,
        time: timestamp,
        type: "scheduled"
      };
    }
    if (typeof when === "number") {
      const time = Math.floor(Date.now() / 1e3 + when);
      this.sql`
        INSERT OR REPLACE INTO container_schedules (id, callback, payload, type, delayInSeconds, time)
        VALUES (${id}, ${callback}, ${JSON.stringify(payload)}, 'delayed', ${when}, ${time})
      `;
      await this.scheduleNextAlarm();
      return {
        taskId: id,
        callback,
        payload,
        delayInSeconds: when,
        time,
        type: "delayed"
      };
    }
    throw new Error("Invalid schedule type. 'when' must be a Date or number of seconds");
  }
  // ============
  //     HTTP
  // ============
  /**
   * Send a request to the container (HTTP or WebSocket) using standard fetch API signature
   *
   * This method handles HTTP requests to the container.
   *
   * WebSocket requests done outside the DO won't work until https://github.com/cloudflare/workerd/issues/2319 is addressed.
   * Until then, please use `switchPort` + `fetch()`.
   *
   * Method supports multiple signatures to match standard fetch API:
   * - containerFetch(request: Request, port?: number)
   * - containerFetch(url: string | URL, init?: RequestInit, port?: number)
   *
   * Starts the container if not already running, and waits for the target port to be ready.
   *
   * @returns A Response from the container
   */
  async containerFetch(requestOrUrl, portOrInit, portParam) {
    let { request, port } = this.requestAndPortFromContainerFetchArgs(requestOrUrl, portOrInit, portParam);
    const state = await this.state.getState();
    if (!this.container.running || state.status !== "healthy") {
      try {
        await this.startAndWaitForPorts(port, { abort: request.signal });
      } catch (e) {
        if (isNoInstanceError(e)) {
          return new Response("There is no Container instance available at this time.\nThis is likely because you have reached your max concurrent instance count (set in wrangler config) or are you currently provisioning the Container.\nIf you are deploying your Container for the first time, check your dashboard to see provisioning status, this may take a few minutes.", { status: 503 });
        } else {
          return new Response(`Failed to start container: ${e instanceof Error ? e.message : String(e)}`, { status: 500 });
        }
      }
    }
    const tcpPort = this.container.getTcpPort(port);
    const containerUrl = request.url.replace("https:", "http:");
    try {
      this.renewActivityTimeout();
      const res = await tcpPort.fetch(containerUrl, request);
      return res;
    } catch (e) {
      if (!(e instanceof Error)) {
        throw e;
      }
      if (e.message.includes("Network connection lost.")) {
        return new Response("Container suddenly disconnected, try again", { status: 500 });
      }
      console.error(`Error proxying request to container ${this.ctx.id}:`, e);
      return new Response(`Error proxying request to container: ${e instanceof Error ? e.message : String(e)}`, { status: 500 });
    }
  }
  /**
   *
   * Fetch handler on the Container class.
   * By default this forwards all requests to the container by calling `containerFetch`.
   * Use `switchPort` to specify which port on the container to target, or this will use `defaultPort`.
   * @param request The request to handle
   */
  async fetch(request) {
    if (this.defaultPort === void 0 && !request.headers.has("cf-container-target-port")) {
      throw new Error("No port configured for this container. Set the `defaultPort` in your Container subclass, or specify a port with `container.fetch(switchPort(request, port))`.");
    }
    let portValue = this.defaultPort;
    if (request.headers.has("cf-container-target-port")) {
      const portFromHeaders = parseInt(request.headers.get("cf-container-target-port") ?? "");
      if (isNaN(portFromHeaders)) {
        throw new Error("port value from switchPort is not a number");
      } else {
        portValue = portFromHeaders;
      }
    }
    return await this.containerFetch(request, portValue);
  }
  // ===============================
  // ===============================
  //     PRIVATE METHODS & ATTRS
  // ===============================
  // ===============================
  // ==========================
  //     PRIVATE ATTRIBUTES
  // ==========================
  container;
  // onStopCalled will be true when we are in the middle of an onStop call
  onStopCalled = false;
  state;
  monitor;
  monitorSetup = false;
  sleepAfterMs = 0;
  // ==========================
  //     GENERAL HELPERS
  // ==========================
  /**
   * Execute SQL queries against the Container's database
   */
  sql(strings, ...values) {
    let query = "";
    query = strings.reduce((acc, str, i) => acc + str + (i < values.length ? "?" : ""), "");
    return [...this.ctx.storage.sql.exec(query, ...values)];
  }
  requestAndPortFromContainerFetchArgs(requestOrUrl, portOrInit, portParam) {
    let request;
    let port;
    if (requestOrUrl instanceof Request) {
      request = requestOrUrl;
      port = typeof portOrInit === "number" ? portOrInit : void 0;
    } else {
      const url = typeof requestOrUrl === "string" ? requestOrUrl : requestOrUrl.toString();
      const init = typeof portOrInit === "number" ? {} : portOrInit || {};
      port = typeof portOrInit === "number" ? portOrInit : typeof portParam === "number" ? portParam : void 0;
      request = new Request(url, init);
    }
    port ??= this.defaultPort;
    if (port === void 0) {
      throw new Error("No port specified for container fetch. Set defaultPort or specify a port parameter.");
    }
    return { request, port };
  }
  /**
   *
   * The method prioritizes port sources in this order:
   * 1. Ports specified directly in the method call
   * 2. `requiredPorts` class property (if set)
   * 3. `defaultPort` (if neither of the above is specified)
   * 4. Falls back to port 33 if none of the above are set
   */
  async getPortsToCheck(overridePorts) {
    let portsToCheck = [];
    if (overridePorts !== void 0) {
      portsToCheck = Array.isArray(overridePorts) ? overridePorts : [overridePorts];
    } else if (this.requiredPorts && this.requiredPorts.length > 0) {
      portsToCheck = [...this.requiredPorts];
    } else {
      portsToCheck = [this.defaultPort ?? FALLBACK_PORT_TO_CHECK];
    }
    return portsToCheck;
  }
  // ===========================================
  //     CONTAINER INTERACTION & MONITORING
  // ===========================================
  /**
   * Tries to start a container if it's not already running
   * Returns the number of tries used
   */
  async startContainerIfNotRunning(waitOptions, options) {
    if (this.container.running) {
      if (!this.monitor) {
        this.monitor = this.container.monitor();
      }
      return 0;
    }
    const abortedSignal = new Promise((res) => {
      waitOptions.signal?.addEventListener("abort", () => {
        res(true);
      });
    });
    const pollInterval = waitOptions.waitInterval ?? INSTANCE_POLL_INTERVAL_MS;
    const totalTries = waitOptions.retries ?? Math.ceil(TIMEOUT_TO_GET_CONTAINER_MS / pollInterval);
    await this.state.setRunning();
    for (let tries = 0; tries < totalTries; tries++) {
      const envVars = options?.envVars ?? this.envVars;
      const entrypoint = options?.entrypoint ?? this.entrypoint;
      const enableInternet = options?.enableInternet ?? this.enableInternet;
      const startConfig = {
        enableInternet
      };
      if (envVars && Object.keys(envVars).length > 0)
        startConfig.env = envVars;
      if (entrypoint)
        startConfig.entrypoint = entrypoint;
      this.renewActivityTimeout();
      const handleError = /* @__PURE__ */ __name(async () => {
        const err = await this.monitor?.catch((err2) => err2);
        if (typeof err === "number") {
          const toThrow = new Error(`Container exited before we could determine the container health, exit code: ${err}`);
          try {
            await this.onError(toThrow);
          } catch {
          }
          throw toThrow;
        } else if (!isNoInstanceError(err)) {
          try {
            await this.onError(err);
          } catch {
          }
          throw err;
        }
      }, "handleError");
      if (tries > 0 && !this.container.running) {
        await handleError();
      }
      await this.scheduleNextAlarm();
      if (!this.container.running) {
        this.container.start(startConfig);
        this.monitor = this.container.monitor();
      } else {
        await this.scheduleNextAlarm();
      }
      this.renewActivityTimeout();
      const port = this.container.getTcpPort(waitOptions.portToCheck);
      try {
        const combinedSignal = addTimeoutSignal(waitOptions.signal, PING_TIMEOUT_MS);
        await port.fetch("http://containerstarthealthcheck", { signal: combinedSignal });
        return tries;
      } catch (error) {
        if (isNotListeningError(error) && this.container.running) {
          return tries;
        }
        if (!this.container.running && isNotListeningError(error)) {
          await handleError();
        }
        console.debug("Error checking if container is ready:", error instanceof Error ? error.message : String(error));
        await Promise.any([
          new Promise((res) => setTimeout(res, waitOptions.waitInterval)),
          abortedSignal
        ]);
        if (waitOptions.signal?.aborted) {
          throw new Error("Aborted waiting for container to start as we received a cancellation signal");
        }
        if (totalTries === tries + 1) {
          if (error instanceof Error && error.message.includes("Network connection lost")) {
            this.ctx.abort();
          }
          throw new Error(NO_CONTAINER_INSTANCE_ERROR);
        }
        continue;
      }
    }
    throw new Error(`Container did not start after ${totalTries * pollInterval}ms`);
  }
  setupMonitorCallbacks() {
    if (this.monitorSetup) {
      return;
    }
    this.monitorSetup = true;
    this.monitor?.then(async () => {
      await this.ctx.blockConcurrencyWhile(async () => {
        await this.state.setStoppedWithCode(0);
      });
    }).catch(async (error) => {
      if (isNoInstanceError(error)) {
        return;
      }
      const exitCode = getExitCodeFromError(error);
      if (exitCode !== null) {
        await this.state.setStoppedWithCode(exitCode);
        this.monitorSetup = false;
        this.monitor = void 0;
        return;
      }
      try {
        await this.onError(error);
      } catch {
      }
    }).finally(() => {
      this.monitorSetup = false;
      if (this.timeout) {
        if (this.resolve)
          this.resolve();
        clearTimeout(this.timeout);
      }
    });
  }
  deleteSchedules(name) {
    this.sql`DELETE FROM container_schedules WHERE callback = ${name}`;
  }
  // ============================
  //     ALARMS AND SCHEDULES
  // ============================
  /**
   * Method called when an alarm fires
   * Executes any scheduled tasks that are due
   */
  async alarm(alarmProps) {
    if (alarmProps.isRetry && alarmProps.retryCount > MAX_ALARM_RETRIES) {
      const scheduleCount = Number(this.sql`SELECT COUNT(*) as count FROM container_schedules`[0]?.count) || 0;
      const hasScheduledTasks = scheduleCount > 0;
      if (hasScheduledTasks || this.container.running) {
        await this.scheduleNextAlarm();
      }
      return;
    }
    const prevAlarm = Date.now();
    await this.ctx.storage.setAlarm(prevAlarm);
    await this.ctx.storage.sync();
    const result = this.sql`
         SELECT * FROM container_schedules;
       `;
    let minTime = Date.now() + 3 * 60 * 1e3;
    const now = Date.now() / 1e3;
    for (const row of result) {
      if (row.time > now) {
        continue;
      }
      const callback = this[row.callback];
      if (!callback || typeof callback !== "function") {
        console.error(`Callback ${row.callback} not found or is not a function`);
        continue;
      }
      const schedule = this.getSchedule(row.id);
      try {
        const payload = row.payload ? JSON.parse(row.payload) : void 0;
        await callback.call(this, payload, await schedule);
      } catch (e) {
        console.error(`Error executing scheduled callback "${row.callback}":`, e);
      }
      this.sql`DELETE FROM container_schedules WHERE id = ${row.id}`;
    }
    const resultForMinTime = this.sql`
         SELECT * FROM container_schedules;
       `;
    const minTimeFromSchedules = Math.min(...resultForMinTime.map((r) => r.time * 1e3));
    if (!this.container.running) {
      await this.syncPendingStoppedEvents();
      if (resultForMinTime.length == 0) {
        await this.ctx.storage.deleteAlarm();
      } else {
        await this.ctx.storage.setAlarm(minTimeFromSchedules);
      }
      return;
    }
    if (this.isActivityExpired()) {
      await this.onActivityExpired();
      this.renewActivityTimeout();
      return;
    }
    minTime = Math.min(minTimeFromSchedules, minTime, this.sleepAfterMs);
    const timeout = Math.max(0, minTime - Date.now());
    await new Promise((resolve) => {
      this.resolve = resolve;
      if (!this.container.running) {
        resolve();
        return;
      }
      this.timeout = setTimeout(() => {
        resolve();
      }, timeout);
    });
    await this.ctx.storage.setAlarm(Date.now());
  }
  timeout;
  resolve;
  // synchronises container state with the container source of truth to process events
  async syncPendingStoppedEvents() {
    const state = await this.state.getState();
    if (!this.container.running && state.status === "healthy") {
      await this.callOnStop({ exitCode: 0, reason: "exit" });
      return;
    }
    if (!this.container.running && state.status === "stopped_with_code") {
      await this.callOnStop({ exitCode: state.exitCode ?? 0, reason: "exit" });
      return;
    }
  }
  async callOnStop(onStopParams) {
    if (this.onStopCalled) {
      return;
    }
    this.onStopCalled = true;
    const promise = this.onStop(onStopParams);
    if (promise instanceof Promise) {
      await promise.finally(() => {
        this.onStopCalled = false;
      });
    } else {
      this.onStopCalled = false;
    }
    await this.state.setStopped();
  }
  /**
   * Schedule the next alarm based on upcoming tasks
   */
  async scheduleNextAlarm(ms = 1e3) {
    const nextTime = ms + Date.now();
    if (this.timeout) {
      if (this.resolve)
        this.resolve();
      clearTimeout(this.timeout);
    }
    await this.ctx.storage.setAlarm(nextTime);
    await this.ctx.storage.sync();
  }
  async listSchedules(name) {
    const result = this.sql`
      SELECT * FROM container_schedules WHERE callback = ${name} LIMIT 1
    `;
    if (!result || result.length === 0) {
      return [];
    }
    return result.map(this.toSchedule);
  }
  toSchedule(schedule) {
    let payload;
    try {
      payload = JSON.parse(schedule.payload);
    } catch (e) {
      console.error(`Error parsing payload for schedule ${schedule.id}:`, e);
      payload = void 0;
    }
    if (schedule.type === "delayed") {
      return {
        taskId: schedule.id,
        callback: schedule.callback,
        payload,
        type: "delayed",
        time: schedule.time,
        delayInSeconds: schedule.delayInSeconds
      };
    }
    return {
      taskId: schedule.id,
      callback: schedule.callback,
      payload,
      type: "scheduled",
      time: schedule.time
    };
  }
  /**
   * Get a scheduled task by ID
   * @template T Type of the payload data
   * @param id ID of the scheduled task
   * @returns The Schedule object or undefined if not found
   */
  async getSchedule(id) {
    const result = this.sql`
      SELECT * FROM container_schedules WHERE id = ${id} LIMIT 1
    `;
    if (!result || result.length === 0) {
      return void 0;
    }
    const schedule = result[0];
    return this.toSchedule(schedule);
  }
  isActivityExpired() {
    return this.sleepAfterMs <= Date.now();
  }
};

// node_modules/@cloudflare/containers/dist/lib/utils.js
var singletonContainerId = "cf-singleton-container";
function getContainer(binding, name = singletonContainerId) {
  const objectId = binding.idFromName(name);
  return binding.get(objectId);
}
__name(getContainer, "getContainer");

// worker/lib/breaker.js
var MONTHLY_BUDGET_SECONDS = 5 * 60 * 60;
var WARN_FRACTION = 0.8;
var STOP_FRACTION = 1;
function monthKey(now = /* @__PURE__ */ new Date()) {
  return `container_seconds_${now.getUTCFullYear()}_${String(now.getUTCMonth() + 1).padStart(2, "0")}`;
}
__name(monthKey, "monthKey");
async function usedSeconds(db, now = /* @__PURE__ */ new Date()) {
  const row = await db.prepare("SELECT value FROM meta WHERE key = ?").bind(monthKey(now)).first();
  if (!row) return 0;
  const value = Number.parseFloat(row.value);
  return Number.isFinite(value) && value >= 0 ? value : 0;
}
__name(usedSeconds, "usedSeconds");
async function addSeconds(db, seconds, now = /* @__PURE__ */ new Date()) {
  if (!Number.isFinite(seconds) || seconds <= 0) return;
  const key = monthKey(now);
  await db.batch([
    db.prepare("INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')").bind(key),
    db.prepare(
      "UPDATE meta SET value = CAST(CAST(value AS REAL) + ? AS TEXT) WHERE key = ?"
    ).bind(seconds, key)
  ]);
}
__name(addSeconds, "addSeconds");
async function budgetStatus(db, now = /* @__PURE__ */ new Date()) {
  let used;
  try {
    used = await usedSeconds(db, now);
  } catch (error) {
    return {
      ok: false,
      allowed: false,
      reason: "meter_unavailable",
      message: "Scanning is paused because the capacity meter could not be read.",
      usedSeconds: null,
      budgetSeconds: MONTHLY_BUDGET_SECONDS,
      fraction: null,
      detail: String(error?.message || error)
    };
  }
  const fraction = used / MONTHLY_BUDGET_SECONDS;
  const allowed = fraction < STOP_FRACTION;
  return {
    ok: true,
    allowed,
    warn: fraction >= WARN_FRACTION,
    reason: allowed ? null : "budget_exhausted",
    message: allowed ? null : "Monthly scanning capacity has been reached. Scanning resumes at the start of next month.",
    usedSeconds: Math.round(used),
    budgetSeconds: MONTHLY_BUDGET_SECONDS,
    remainingSeconds: Math.max(0, Math.round(MONTHLY_BUDGET_SECONDS - used)),
    fraction: Number(fraction.toFixed(4))
  };
}
__name(budgetStatus, "budgetStatus");

// worker/container.js
var STARTED_AT = "container_started_at";
var MAX_PLAUSIBLE_PERIOD_SECONDS = 10 * 60;
var InferenceContainer = class extends Container {
  static {
    __name(this, "InferenceContainer");
  }
  /** Matches EXPOSE and the uvicorn bind in inference/Dockerfile. */
  defaultPort = 8080;
  /**
   * Ninety seconds.
   *
   * Long enough that a visitor scanning several films in a row does not pay a
   * cold start between each, short enough that an abandoned session stops
   * costing money almost immediately. Every second here is billed at roughly
   * $0.0000206, so the default of "10m" would cost about seven times more per
   * visit for no benefit a user would notice.
   */
  sleepAfter = "90s";
  /**
   * The container makes no outbound requests, so it is not permitted any.
   *
   * The model and its calibration are baked into the image, and
   * `RadiographClassifier` forces `pretrained` off, so nothing needs to be
   * downloaded at runtime. Turning the network off makes that a structural
   * guarantee rather than an assumption: a process that cannot open a socket
   * cannot exfiltrate a radiograph, whatever a future dependency decides to do.
   */
  enableInternet = false;
  /**
   * Secrets reach the container as environment variables at start.
   *
   * A getter rather than a plain field because `this.env` is only populated once
   * the Durable Object is constructed, and a class field initialiser would
   * capture `undefined`.
   */
  get envVars() {
    return {
      INFERENCE_KEY: this.env.INFERENCE_KEY || "",
      ONNM_CHECKPOINT: this.env.ONNM_CHECKPOINT || "/opt/onnm/best.pt"
    };
  }
  async onStart() {
    const now = Date.now();
    const stale = await this.ctx.storage.get(STARTED_AT);
    if (typeof stale === "number" && stale > 0) {
      const orphaned = Math.min((now - stale) / 1e3, MAX_PLAUSIBLE_PERIOD_SECONDS);
      console.warn(
        `container: stale start marker found; billing ${orphaned.toFixed(1)}s from an unclean stop`
      );
      await this.#bill(orphaned);
    }
    await this.ctx.storage.put(STARTED_AT, now);
    console.log("container: started");
  }
  async onStop({ exitCode, reason }) {
    const startedAt = await this.ctx.storage.get(STARTED_AT);
    await this.ctx.storage.delete(STARTED_AT);
    if (typeof startedAt !== "number" || startedAt <= 0) {
      console.warn("container: stopped with no start marker; runtime not metered");
      return;
    }
    const elapsed = (Date.now() - startedAt) / 1e3;
    await this.#bill(elapsed);
    console.log(
      `container: stopped after ${elapsed.toFixed(1)}s (reason=${reason}, exit=${exitCode})`
    );
  }
  onError(error) {
    console.error("container: error", error);
    throw error;
  }
  /**
   * Write elapsed runtime to the monthly meter.
   *
   * Never allowed to throw. A metering failure must not turn into a failed
   * scan for the user -- but it is logged loudly, because a silently broken
   * meter is the one failure that costs money.
   */
  async #bill(seconds) {
    try {
      if (!this.env.DB) {
        console.error("container: no DB binding; runtime NOT metered");
        return;
      }
      await addSeconds(this.env.DB, seconds);
    } catch (error) {
      console.error("container: failed to record runtime", error);
    }
  }
};
function inferenceStub(env) {
  return getContainer(env.ONNM_INFERENCE, "singleton");
}
__name(inferenceStub, "inferenceStub");

// worker/lib/google.js
var AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth";
var TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";
var VALID_ISSUERS = /* @__PURE__ */ new Set(["https://accounts.google.com", "accounts.google.com"]);
var SCOPES = "openid email profile";
var ENCODER = new TextEncoder();
function base64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
__name(base64url, "base64url");
function randomToken(bytes = 32) {
  return base64url(crypto.getRandomValues(new Uint8Array(bytes)));
}
__name(randomToken, "randomToken");
async function codeChallenge(verifier) {
  const digest = await crypto.subtle.digest("SHA-256", ENCODER.encode(verifier));
  return base64url(new Uint8Array(digest));
}
__name(codeChallenge, "codeChallenge");
function authorizationUrl({ clientId, redirectUri, state, challenge }) {
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: SCOPES,
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
    access_type: "online",
    prompt: "select_account"
  });
  return `${AUTH_ENDPOINT}?${params.toString()}`;
}
__name(authorizationUrl, "authorizationUrl");
async function exchangeCode({ clientId, clientSecret, redirectUri, code, verifier }) {
  const response = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
      code,
      code_verifier: verifier
    })
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.error_description || payload?.error || `HTTP ${response.status}`;
    throw new Error(`token exchange failed: ${detail}`);
  }
  if (!payload?.id_token) throw new Error("token exchange returned no id_token");
  return payload;
}
__name(exchangeCode, "exchangeCode");
function decodeJwtPayload(token) {
  const parts = String(token).split(".");
  if (parts.length !== 3) throw new Error("id_token is not a JWT");
  const padded = parts[1].replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - padded.length % 4) % 4));
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0))));
}
__name(decodeJwtPayload, "decodeJwtPayload");
function claimsFromIdToken(idToken, clientId) {
  const claims = decodeJwtPayload(idToken);
  if (!VALID_ISSUERS.has(claims.iss)) {
    throw new Error(`unexpected issuer ${claims.iss}`);
  }
  if (claims.aud !== clientId) {
    throw new Error("id_token audience does not match this client");
  }
  if (typeof claims.exp !== "number" || claims.exp < Math.floor(Date.now() / 1e3)) {
    throw new Error("id_token has expired");
  }
  if (!claims.sub) {
    throw new Error("id_token carries no subject");
  }
  if (claims.email_verified === false) {
    throw new Error("google account email is not verified");
  }
  if (!claims.email) {
    throw new Error("id_token carries no email address");
  }
  return claims;
}
__name(claimsFromIdToken, "claimsFromIdToken");
function identityProfile(claims) {
  const name = String(claims.name || "").trim().slice(0, 80);
  let picture = String(claims.picture || "").trim();
  try {
    const parsed = new URL(picture);
    const host = parsed.hostname.toLowerCase();
    const allowed = parsed.protocol === "https:" && (host === "googleusercontent.com" || host.endsWith(".googleusercontent.com"));
    if (!allowed) picture = "";
  } catch {
    picture = "";
  }
  return {
    name,
    picture: picture.slice(0, 2048),
    subject: String(claims.sub),
    email: String(claims.email).trim().toLowerCase()
  };
}
__name(identityProfile, "identityProfile");

// worker/lib/session.js
var ENCODER2 = new TextEncoder();
var DECODER = new TextDecoder();
var SESSION_COOKIE = "__Host-onnm_session";
var SESSION_TTL_SECONDS = 8 * 60 * 60;
function base64urlEncode(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
__name(base64urlEncode, "base64urlEncode");
function base64urlDecode(text) {
  const padded = text.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - padded.length % 4) % 4));
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}
__name(base64urlDecode, "base64urlDecode");
async function hmacKey(secret) {
  return crypto.subtle.importKey(
    "raw",
    ENCODER2.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}
__name(hmacKey, "hmacKey");
function timingSafeEqual2(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
__name(timingSafeEqual2, "timingSafeEqual");
async function signSession(payload, secret, ttlSeconds = SESSION_TTL_SECONDS) {
  const now = Math.floor(Date.now() / 1e3);
  const body = { ...payload, iat: now, exp: now + ttlSeconds };
  const encoded = base64urlEncode(ENCODER2.encode(JSON.stringify(body)));
  const key = await hmacKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, ENCODER2.encode(encoded));
  return `${encoded}.${base64urlEncode(new Uint8Array(signature))}`;
}
__name(signSession, "signSession");
async function verifySession(token, secret) {
  if (typeof token !== "string" || !token.includes(".")) return null;
  const [encoded, signature] = token.split(".", 2);
  if (!encoded || !signature) return null;
  let valid = false;
  try {
    const key = await hmacKey(secret);
    valid = await crypto.subtle.verify(
      "HMAC",
      key,
      base64urlDecode(signature),
      ENCODER2.encode(encoded)
    );
  } catch {
    return null;
  }
  if (!valid) return null;
  try {
    const payload = JSON.parse(DECODER.decode(base64urlDecode(encoded)));
    if (typeof payload?.exp !== "number" || payload.exp < Math.floor(Date.now() / 1e3)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}
__name(verifySession, "verifySession");
function readCookie(request, name) {
  const header = request.headers.get("cookie") || "";
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return null;
}
__name(readCookie, "readCookie");
async function currentSession(request, env) {
  if (!env.SESSION_SECRET) return null;
  const token = readCookie(request, SESSION_COOKIE);
  if (!token) return null;
  return verifySession(token, env.SESSION_SECRET);
}
__name(currentSession, "currentSession");
function sessionCookieHeader(token, { maxAge = SESSION_TTL_SECONDS } = {}) {
  return `${SESSION_COOKIE}=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`;
}
__name(sessionCookieHeader, "sessionCookieHeader");
function clearSessionCookieHeader() {
  return `${SESSION_COOKIE}=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0`;
}
__name(clearSessionCookieHeader, "clearSessionCookieHeader");
var OAUTH_COOKIE = "__Host-onnm_oauth";
var OAUTH_TTL_SECONDS = 10 * 60;
function oauthCookieHeader(token) {
  return `${OAUTH_COOKIE}=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${OAUTH_TTL_SECONDS}`;
}
__name(oauthCookieHeader, "oauthCookieHeader");
function clearOauthCookieHeader() {
  return `${OAUTH_COOKIE}=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0`;
}
__name(clearOauthCookieHeader, "clearOauthCookieHeader");

// worker/index.js
var json2 = /* @__PURE__ */ __name((data, status = 200, headers = {}) => new Response(JSON.stringify(data), {
  status,
  headers: { "content-type": "application/json; charset=utf-8", ...headers }
}), "json");
var fail2 = /* @__PURE__ */ __name((status, message, extra = {}) => json2({ error: message, ...extra }, status), "fail");
async function forward(request, env, path, { method, body, search } = {}) {
  const url = new URL(request.url);
  url.pathname = path;
  url.search = search ? `?${new URLSearchParams(search).toString()}` : "";
  const headers = new Headers();
  headers.set("authorization", `Bearer ${env.API_KEY}`);
  if (body !== void 0) headers.set("content-type", "application/json");
  const init = { method: method || request.method, headers, cf: request.cf };
  if (body !== void 0) init.body = JSON.stringify(body);
  return handleApiRequest(new Request(url.toString(), init), env);
}
__name(forward, "forward");
async function forwardJson(request, env, path, options) {
  const response = await forward(request, env, path, options);
  const payload = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, payload };
}
__name(forwardJson, "forwardJson");
function redirectUriFor(request) {
  return new URL("/api/auth/google/callback", new URL(request.url).origin).toString();
}
__name(redirectUriFor, "redirectUriFor");
async function authStart(request, env) {
  if (!env.GOOGLE_CLIENT_ID || !env.GOOGLE_CLIENT_SECRET || !env.SESSION_SECRET) {
    return fail2(503, "Google sign-in is not configured on this deployment");
  }
  const state = randomToken();
  const verifier = randomToken(48);
  const challenge = await codeChallenge(verifier);
  const token = await signSession({ state, verifier }, env.SESSION_SECRET, OAUTH_TTL_SECONDS);
  const url = authorizationUrl({
    clientId: env.GOOGLE_CLIENT_ID,
    redirectUri: redirectUriFor(request),
    state,
    challenge
  });
  return new Response(null, {
    status: 302,
    headers: { location: url, "set-cookie": oauthCookieHeader(token) }
  });
}
__name(authStart, "authStart");
async function authCallback(request, env) {
  const url = new URL(request.url);
  const origin = url.origin;
  const bounce = /* @__PURE__ */ __name((code2) => new Response(null, {
    status: 302,
    headers: { location: `${origin}/?auth_error=${code2}`, "set-cookie": clearOauthCookieHeader() }
  }), "bounce");
  if (url.searchParams.get("error")) return bounce("declined");
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  if (!code || !state) return bounce("missing_code");
  const pending = await verifySession(readCookie(request, OAUTH_COOKIE), env.SESSION_SECRET);
  if (!pending) return bounce("expired");
  if (!timingSafeEqual2(String(state), String(pending.state))) return bounce("state_mismatch");
  let claims;
  try {
    const tokens = await exchangeCode({
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
      redirectUri: redirectUriFor(request),
      code,
      verifier: pending.verifier
    });
    claims = claimsFromIdToken(tokens.id_token, env.GOOGLE_CLIENT_ID);
  } catch (error) {
    console.error("oauth: exchange failed", error);
    const reason = String(error?.message || "").includes("not verified") ? "email_unverified" : "exchange_failed";
    return bounce(reason);
  }
  const profile = identityProfile(claims);
  const account = await forwardJson(request, env, "/users/by-subject", {
    method: "GET",
    search: { subject: profile.subject }
  });
  const existing = account.ok ? account.payload?.user : null;
  let userId = existing?.user_id || null;
  if (!userId) {
    const created = await forwardJson(request, env, "/users", {
      method: "POST",
      body: {
        email: profile.email,
        auth_provider: "google",
        provider_subject: profile.subject,
        display_name: profile.name,
        profile_picture_url: profile.picture
      }
    });
    if (!created.ok) {
      console.error("oauth: account creation failed", created.status, created.payload);
      return bounce(created.status === 403 ? "registration_closed" : "account_failed");
    }
    userId = created.payload?.user_id || created.payload?.user?.user_id || null;
  }
  if (!userId) return bounce("account_failed");
  await forward(request, env, "/users/profile", {
    method: "POST",
    body: {
      user_id: userId,
      display_name: profile.name,
      profile_picture_url: profile.picture
    }
  }).catch(() => {
  });
  const token = await signSession(
    { uid: userId, sub: profile.subject, email: profile.email, name: profile.name, pic: profile.picture },
    env.SESSION_SECRET
  );
  return new Response(null, {
    status: 302,
    headers: [
      ["location", `${origin}/`],
      ["set-cookie", clearOauthCookieHeader()],
      ["set-cookie", sessionCookieHeader(token)]
    ]
  });
}
__name(authCallback, "authCallback");
async function warmup(request, env) {
  const session = await currentSession(request, env);
  if (!session) return fail2(401, "sign in first");
  const budget = await budgetStatus(env.DB);
  if (!budget.allowed) return json2({ warming: false, budget }, 200);
  try {
    const response = await inferenceStub(env).containerFetch(
      new Request("http://container/health", { method: "GET" })
    );
    return json2({ warming: true, ready: response.ok, budget });
  } catch (error) {
    console.error("warmup failed", error);
    return json2({ warming: false, ready: false, budget });
  }
}
__name(warmup, "warmup");
async function scan(request, env) {
  const session = await currentSession(request, env);
  if (!session) return fail2(401, "sign in first");
  const budget = await budgetStatus(env.DB);
  if (!budget.allowed) {
    return json2({ error: budget.message, reason: budget.reason, budget }, 429);
  }
  const form = await request.formData().catch(() => null);
  const file = form?.get("file");
  if (!file || typeof file === "string") return fail2(400, "no file uploaded");
  const outbound = new FormData();
  outbound.set("file", file, file.name || "upload");
  const threshold = form.get("threshold");
  if (threshold) outbound.set("threshold", String(threshold));
  outbound.set("cam_class", String(form.get("cam_class") || "auto"));
  outbound.set("with_heatmap", "true");
  outbound.set("want_preprocessed", "true");
  let result;
  try {
    const response = await inferenceStub(env).containerFetch(
      new Request("http://container/infer", {
        method: "POST",
        headers: { authorization: `Bearer ${env.INFERENCE_KEY}` },
        body: outbound
      })
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      return json2({ error: payload?.detail || "inference failed" }, response.status);
    }
    result = payload;
  } catch (error) {
    console.error("scan: container call failed", error);
    return fail2(502, "the model is temporarily unavailable");
  }
  if (!result.is_radiograph) {
    return json2({ ...result, budget });
  }
  const shared = String(form.get("share_consent") || "") === "true";
  const prediction = result.prediction || {};
  const recorded = await forwardJson(request, env, "/submissions", {
    method: "POST",
    body: {
      user_id: session.uid,
      model_label: prediction.label,
      lesion_probability: result.lesion_probability,
      class_probabilities: result.class_probabilities,
      checkpoint: prediction.source?.run || null,
      threshold: prediction.decision_threshold,
      calibrated: prediction.calibrated,
      ood_flagged: false,
      shared,
      image_b64: shared ? result.preprocessed?.png_b64 : null,
      image_sha256: result.preprocessed?.sha256 || null,
      image_bytes: result.preprocessed?.bytes || null
    }
  });
  return json2({
    ...result,
    // The stored image never travels back to the browser; it has the overlay
    // and the original already, and this keeps the response small.
    preprocessed: void 0,
    submission_id: recorded.ok ? recorded.payload?.submission_id : null,
    stored: recorded.ok,
    budget
  });
}
__name(scan, "scan");
var index_default = {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const method = request.method.toUpperCase();
    if (!path.startsWith("/api/")) {
      return env.ASSETS ? env.ASSETS.fetch(request) : fail2(404, "not found");
    }
    if (!env.DB) return fail2(500, "D1 binding 'DB' is not configured");
    try {
      if (method === "GET" && path === "/api/auth/google/start") return authStart(request, env);
      if (method === "GET" && path === "/api/auth/google/callback") {
        return authCallback(request, env);
      }
      if (method === "GET" && path === "/api/globe") return forward(request, env, "/globe");
      if (method === "GET" && path === "/api/contributors") {
        return forward(request, env, "/contributors");
      }
      if (method === "GET" && path === "/api/stats") return forward(request, env, "/health");
      if (method === "GET" && path === "/api/session") {
        const session2 = await currentSession(request, env);
        return json2({
          signed_in: Boolean(session2),
          user: session2 ? { user_id: session2.uid, email: session2.email, name: session2.name, picture: session2.pic } : null,
          budget: await budgetStatus(env.DB)
        });
      }
      if (method === "POST" && path === "/api/auth/signout") {
        return json2({ signed_out: true }, 200, { "set-cookie": clearSessionCookieHeader() });
      }
      const session = await currentSession(request, env);
      if (method === "GET" && path === "/api/submissions") {
        if (!session) return fail2(401, "sign in first");
        const forwarded = new URL(request.url);
        forwarded.pathname = "/submissions";
        forwarded.search = `?user_id=${encodeURIComponent(session.uid)}&limit=50`;
        return handleApiRequest(
          new Request(forwarded.toString(), {
            method: "GET",
            headers: { authorization: `Bearer ${env.API_KEY}` },
            cf: request.cf
          }),
          env
        );
      }
      if (method === "POST" && path === "/api/warmup") return warmup(request, env);
      if (method === "POST" && path === "/api/scan") return scan(request, env);
      return fail2(404, `no route for ${method} ${path}`);
    } catch (error) {
      console.error("unhandled", error);
      return fail2(500, "internal error");
    }
  }
};
export {
  InferenceContainer,
  index_default as default
};
//# sourceMappingURL=index.js.map
