/**
 * ONNM community API — Cloudflare Worker over D1.
 *
 * WHAT THIS IS
 * ------------
 * A thin, guarded data layer. The Streamlit app on Hugging Face Spaces is the
 * main client; it holds a shared secret and does all the interesting work
 * (inference, Grad-CAM, password hashing). A signed-in browser makes one
 * token-scoped country-capture request. This Worker stores rows and enforces
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
const LOCATION_TOKEN_TTL_MS = 5 * 60 * 1000;

// --- Country display rule ---------------------------------------------------
// Product decision: show a country as soon as one account is recorded there.
// The endpoint remains country-level and never returns coordinates, timestamps,
// account ids, or diagnoses. A one-account country can still reveal that an
// account exists there, so this value is a display minimum, not anonymity.
// Keep the response field name for client compatibility.
const K_ANONYMITY_MIN = 1;

// The three clinical classes the lesion classifier predicts.
const VALID_LABELS = new Set(["normal", "benign", "malignant"]);
// What a reviewer may write into admin_label. 'misc' means "not a bone
// radiograph at all" -- a real training target for the OOD detector, which
// today has only hand-written heuristics and no negatives, but not a diagnosis.
// The schema carries the same four values; tests assert the two agree.
const REVIEW_LABELS = new Set([...VALID_LABELS, "misc"]);
const BUCKETS = new Set(["valid_bone", "misc", "contradiction"]);
const AUTH_PROVIDERS = new Set(["password", "google"]);

// Mirrors DEFAULT_CONFIDENCE_FLOOR in src/onnm/ood.py. Below this the app
// already refuses to present a lesion call, so a flagged-but-confident row is
// the only shape in which the two stages of the gate genuinely contradict each
// other; a flagged-and-hesitant one is just the gate working.
const CONFIDENT_PROB = 0.65;

// ---------------------------------------------------------------------------
// Who may review.
//
// One account, written here and again as a CHECK constraint in schema.sql.
// The admin key authenticates the *caller*; this pins the *account*, so a
// leaked or shared key still cannot review on behalf of somebody else, and a
// signed-in ordinary user cannot reach these routes even if the app process
// they are talking to holds the key. Two different questions, two answers.
//
// This is not defence in depth against a stolen key -- anyone holding the key
// can also send the header. It is what stops the far likelier failure: a UI
// bug, a shared deployment, or a future endpoint quietly serving the review
// queue to the wrong signed-in account.
// ---------------------------------------------------------------------------
const ADMIN_USER_ID = "c2c5a209-4aaa-4eb9-b112-b2929b6dbe12";
const ADMIN_EMAIL = "kzfhero@gmail.com";

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

/**
 * The country the request came from, at ISO 3166-1 alpha-2 resolution.
 *
 * Cloudflare resolves this at the edge from the connecting address and hands
 * it over already reduced, so nothing here ever sees or stores an IP. That is
 * the whole reason it is done this way rather than with the browser
 * Geolocation API, which would prompt the visitor and return GPS precision the
 * schema deliberately cannot hold.
 *
 * Returns null rather than a placeholder when Cloudflare offers nothing --
 * `wrangler dev` without `--remote` has no `cf` object at all, and a local run
 * writing a fake country would be worse than writing none. 'T1' (Tor) and 'XX'
 * (undetermined) are passed through honestly; they have no centroid, so the
 * globe drops them at aggregation time.
 */
function countryOf(request) {
  const code = request && request.cf && request.cf.country;
  if (typeof code !== "string") return null;
  const upper = code.toUpperCase();
  return /^[A-Z]{2}$/.test(upper) || upper === "T1" ? upper : null;
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

/**
 * Sort one submission into a triage bucket.
 *
 * Three buckets, and the rule is deliberately readable rather than clever,
 * because the thing being classified is the failure of a classifier:
 *
 *   valid_bone    the gate accepted it. Retrains the lesion head.
 *   misc          the gate rejected it. Retrains the OOD detector as a
 *                 negative -- misuse is data, not noise.
 *   contradiction the two halves of the system disagree, or the user disagrees
 *                 with the gate. Highest value per row: each one is a
 *                 demonstrated gate failure with an image attached.
 *
 * Two shapes count as a contradiction:
 *
 *   1. The gate rejected the image but the user insists it is a radiograph.
 *      That is a false rejection -- the "user uploads a bone, the OOD gate
 *      turns it away" case -- and the user is the only witness to it, because
 *      inference never ran.
 *   2. The gate accepted the image and the classifier produced a confident
 *      call, but the user says it is not a radiograph at all. That is the
 *      hotdog-called-a-bone case, caught from the other side.
 *
 * A flagged image with no confident call and no dispute is simply misuse, and
 * a clean prediction the user merely disagrees with about *grade* stays in
 * valid_bone: "you said malignant, I think benign" is a labelling dispute for
 * the reviewer, not evidence that the gate misfired.
 *
 * The result is a guess by the system under correction, so it is written to
 * `triage_bucket`, never to `admin_bucket`. It orders the queue; the human
 * decides. Kept in step with classify_bucket() in src/community.py.
 */
function triageBucket({ oodFlagged, maxProb, userSaysWrong, userSuggestedLabel }) {
  const userSaysNotRadiograph = userSuggestedLabel === "misc";
  if (oodFlagged) {
    if (userSaysWrong && !userSaysNotRadiograph) {
      return { bucket: "contradiction", reason: "gate rejected it; the user says it is a radiograph" };
    }
    if (maxProb >= CONFIDENT_PROB) {
      return {
        bucket: "contradiction",
        reason: `gate rejected it but the classifier was ${maxProb.toFixed(2)} confident`,
      };
    }
    return { bucket: "misc", reason: "the out-of-distribution gate rejected it" };
  }
  if (userSaysNotRadiograph) {
    return { bucket: "contradiction", reason: "gate accepted it; the user says it is not a radiograph" };
  }
  return { bucket: "valid_bone", reason: "the out-of-distribution gate accepted it" };
}

/** Largest value in the class-probability map, or 0 when it is absent/unusable. */
function maxProbability(probabilities) {
  if (!probabilities || typeof probabilities !== "object") return 0;
  const values = Object.values(probabilities)
    .map(Number)
    .filter((v) => Number.isFinite(v));
  return values.length ? Math.max(...values) : 0;
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
  // Per-bucket, because "42 awaiting review" says nothing about whether the
  // queue in front of you is 42 ordinary films or 42 demonstrated gate failures.
  const { results: byBucket } = await db
    .prepare(
      `SELECT triage_bucket, COUNT(*) AS n FROM submissions
        WHERE shared = 1 AND review_status = 'pending' GROUP BY triage_bucket`
    )
    .all();
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
      max_users: MAX_USERS,
    },
  });
}

/**
 * Aggregated country counts for the landing-page globe.
 *
 * WHAT THIS DELIBERATELY DOES NOT RETURN
 * --------------------------------------
 * No user_id, no email, no submission_id, no timestamps, no coordinates, and
 * no row that could be traced to a person. The response is a list of country
 * codes and integers, and that is the whole of it. If a future change makes
 * this endpoint capable of returning anything else, the tests that assert the
 * shape of this payload should fail -- they exist for that reason.
 *
 * It does not emit latitude or longitude either. Turning a country code into a
 * dot is the client's job (src/geo.py), so the API never carries a coordinate
 * at all and cannot leak a precision it does not possess.
 *
 * TWO LAYERS
 * ----------
 *   signups      -- accounts created, per country.
 *   contributors -- DISTINCT users with at least one APPROVED submission, per
 *                   country. Distinct users rather than submission count, so
 *                   one enthusiastic uploader cannot inflate their own dot;
 *                   approved-only, because an unreviewed submission has not
 *                   contributed anything yet, and consistency with the review
 *                   gate matters more than a busier-looking map.
 *
 * DISPLAY THRESHOLD
 * -----------------
 * The configured minimum is one account, so every known country is plotted.
 * `elsewhere` remains part of the stable API contract in case the threshold is
 * raised again. Coordinates are still fixed country centroids; no precise
 * location or identifying account field is selected by this endpoint.
 */
async function globe(db) {
  const { results: signupRows } = await db
    .prepare(
      `SELECT signup_country AS country, COUNT(*) AS n
         FROM users
        WHERE signup_country IS NOT NULL
        GROUP BY signup_country`
    )
    .all();

  const { results: contributorRows } = await db
    .prepare(
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
    )
    .all();

  // Rows predating migration 0004 carry NULL. Their approved contributions use
  // the account country once that account next signs in; until then they stay
  // in the totals without inventing a location.
  const split = (rows) => {
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
  };

  const totalUsers = await db.prepare("SELECT COUNT(*) AS n FROM users").first();
  const totalContributors = await db
    .prepare(
      `SELECT COUNT(DISTINCT user_id) AS n FROM submissions WHERE review_status = 'approved'`
    )
    .first();
  const totalApproved = await db
    .prepare("SELECT COUNT(*) AS n FROM submissions WHERE review_status = 'approved'")
    .first();

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
      countries_represented: signups.plotted.length + signups.suppressed_countries,
    },
    layers: { signups, contributors },
    // Stated in the payload so the client can label the map honestly rather
    // than implying the plotted dots are everyone.
    k_anonymity_min: K_ANONYMITY_MIN,
    generated_at: nowIso(),
  });
}

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
    profile_picture_url,
  } = body;
  const provider = auth_provider || "password";
  if (!user_id || !email) return fail(400, "user_id and email are required");
  if (!AUTH_PROVIDERS.has(provider)) {
    return fail(400, `auth_provider must be one of ${[...AUTH_PROVIDERS].join(", ")}`);
  }

  // Each provider has its own required shape, and the wrong one is rejected
  // rather than coerced. Accepting a password_hash on a Google account would
  // create a federated identity that can *also* be logged into with a password
  // — an authentication bypass, not a data-quality problem.
  if (provider === "password") {
    if (!password_hash) return fail(400, "password_hash is required for a password account");
    // The Worker must never be handed a plaintext password. This does not prove
    // the value is a real PBKDF2 hash, but it catches the client bug where the
    // raw password is passed by mistake, which is the failure worth catching.
    if (!String(password_hash).startsWith("pbkdf2_sha256$")) {
      return fail(400, "password_hash must be a pbkdf2_sha256 encoded hash, not a password");
    }
    if (provider_subject) return fail(400, "a password account must not carry a provider_subject");
  } else {
    if (!provider_subject) return fail(400, `provider_subject is required for a ${provider} account`);
    if (password_hash) return fail(400, `a ${provider} account must not carry a password_hash`);
  }

  // Admin is one hardcoded account. The schema refuses the row anyway, but a
  // constraint failure surfaces as an opaque 500; saying so here names the
  // reason. Nothing in the app ever sets this flag -- it exists so the seed row
  // can, and so an attempt to grant it elsewhere is a visible refusal rather
  // than a silently-honoured request.
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
    await db
      .prepare(
        `INSERT INTO users (user_id, email, password_hash, auth_provider, provider_subject,
                            created_at, tos_accepted_at, is_admin, signup_country,
                            display_name, profile_picture_url)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
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
      )
      .run();
  } catch (err) {
    if (String(err).includes("UNIQUE")) return fail(409, "an account already exists for that email");
    throw err;
  }
  return json({ user_id, email, created_at: created, auth_provider: provider }, 201);
}

/**
 * Look an account up by Google's `sub` claim rather than by email.
 *
 * Email is the wrong key for a federated identity: a Google Workspace address
 * can be reassigned to a different person after an account is closed, and a
 * user can change the address on their own account. `sub` is stable and unique
 * for the lifetime of the Google account, so it is what identity is keyed on.
 */
async function getUserBySubject(db, subject) {
  if (!subject) return fail(400, "subject query parameter is required");
  const row = await db
    .prepare(
      `SELECT user_id, email, password_hash, auth_provider, provider_subject,
              created_at, tos_accepted_at, is_admin, display_name,
              profile_picture_url, public_contributor_profile
         FROM users WHERE provider_subject = ?`
    )
    .bind(String(subject))
    .first();
  return row ? json(row) : fail(404, "no such user");
}

async function getUserByEmail(db, email) {
  if (!email) return fail(400, "email query parameter is required");
  const row = await db
    .prepare(
      `SELECT user_id, email, password_hash, auth_provider, provider_subject,
              created_at, tos_accepted_at, is_admin, display_name,
              profile_picture_url, public_contributor_profile
       FROM users WHERE email = ? COLLATE NOCASE`
    )
    .bind(String(email).toLowerCase())
    .first();
  return row ? json(row) : fail(404, "no such user");
}

function withLocationCors(response) {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "POST, OPTIONS");
  headers.set("access-control-allow-headers", "authorization, content-type");
  headers.set("access-control-max-age", "600");
  return new Response(response.body, { status: response.status, headers });
}

async function tokenHash(token) {
  const bytes = new TextEncoder().encode(token);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function issueLocationToken(db, body) {
  const userId = String(body.user_id || "");
  if (!userId) return fail(400, "user_id is required");
  const user = await db.prepare("SELECT user_id FROM users WHERE user_id = ?").bind(userId).first();
  if (!user) return fail(404, "no such user");

  const token = crypto.randomUUID();
  const hash = await tokenHash(token);
  const issuedAt = new Date().toISOString();
  const expiresAt = new Date(Date.now() + LOCATION_TOKEN_TTL_MS)
    .toISOString();
  await db.batch([
    db
      .prepare("DELETE FROM location_capture_tokens WHERE expires_at < ? OR used_at IS NOT NULL")
      .bind(issuedAt),
    db
      .prepare(
        `INSERT INTO location_capture_tokens (token_hash, user_id, expires_at)
         VALUES (?, ?, ?)`
      )
      .bind(hash, userId, expiresAt),
  ]);
  return json({ ok: true, token, expires_at: expiresAt });
}

async function captureBrowserCountry(db, request) {
  const country = countryOf(request);
  if (!country || country === "XX" || country === "T1") {
    return withLocationCors(fail(422, "country could not be resolved at the Cloudflare edge"));
  }
  const token = bearer(request);
  if (!token) return withLocationCors(fail(401, "location token required"));

  const hash = await tokenHash(token);
  const capturedAt = new Date().toISOString();
  const captureAttempt = crypto.randomUUID();
  const row = await db
    .prepare(
      `SELECT user_id FROM location_capture_tokens
        WHERE token_hash = ? AND used_at IS NULL AND expires_at >= ?`
    )
    .bind(hash, capturedAt)
    .first();
  if (!row) return withLocationCors(fail(401, "location token is invalid, expired, or used"));

  // The first browser-edge capture is authoritative. It replaces any country
  // incorrectly written by the old server-to-Worker path, then stays fixed so
  // travel or a later sign-in cannot rewrite an account's country. The batch
  // is atomic: a failed repair cannot consume the one-use token by itself.
  const results = await db.batch([
    db
      .prepare(
        `UPDATE location_capture_tokens SET used_at = ?, used_nonce = ?
          WHERE token_hash = ? AND used_at IS NULL`
      )
      .bind(capturedAt, captureAttempt, hash),
    db
      .prepare(
        `UPDATE users
            SET signup_country = ?, country_captured_at = ?
          WHERE user_id = ? AND country_captured_at IS NULL
            AND EXISTS (
              SELECT 1 FROM location_capture_tokens
               WHERE token_hash = ? AND used_nonce = ?
            )`
      )
      .bind(country, capturedAt, String(row.user_id), hash, captureAttempt),
    // Repair historical submissions that inherited the Streamlit server's
    // country. New submissions copy the already-captured account country.
    db
      .prepare(
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
      )
      .bind(
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

function cleanDisplayName(value) {
  const name = String(value || "").trim();
  return name ? name.slice(0, 80) : null;
}

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

async function updateContributorProfile(db, body) {
  const { user_id, provider_subject, display_name, profile_picture_url } = body;
  if (!user_id || !provider_subject) return fail(400, "user_id and provider_subject are required");
  const user = await db
    .prepare("SELECT auth_provider, provider_subject FROM users WHERE user_id = ?")
    .bind(String(user_id))
    .first();
  if (!user) return fail(404, "no such user");
  if (user.auth_provider !== "google" || !timingSafeEqual(user.provider_subject || "", String(provider_subject))) {
    return fail(403, "profile identity does not match the Google account");
  }

  const publicProfile = Object.prototype.hasOwnProperty.call(body, "public_profile")
    ? (body.public_profile ? 1 : 0)
    : null;
  if (publicProfile === null) {
    return json({ ok: true });
  }
  const storedName = publicProfile === 0 ? null : cleanDisplayName(display_name);
  const storedPicture = publicProfile === 0 ? null : cleanGooglePicture(profile_picture_url);
  await db
    .prepare(
      `UPDATE users
          SET display_name = ?, profile_picture_url = ?,
              public_contributor_profile = ?
        WHERE user_id = ?`
    )
    .bind(
      storedName,
      storedPicture,
      publicProfile,
      String(user_id)
    )
    .run();
  return json({ ok: true, public_contributor_profile: publicProfile });
}

async function listContributors(db) {
  const { results } = await db
    .prepare(
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
    )
    .all();
  return json({ contributors: results ?? [] });
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
  const user = await db
    .prepare("SELECT user_id, signup_country FROM users WHERE user_id = ?")
    .bind(user_id)
    .first();
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

  // Triage on arrival, from the model's own signals. There is no user feedback
  // yet, so a row can only land in valid_bone or misc here; contradiction needs
  // either a confident call the gate rejected, or a dispute that arrives later.
  const triage = triageBucket({
    oodFlagged: Boolean(ood_flagged),
    maxProb: maxProbability(class_probabilities),
    userSaysWrong: false,
    userSuggestedLabel: null,
  });

  const created = nowIso();
  const statements = [
    db
      .prepare(
        `INSERT INTO submissions (
            submission_id, user_id, created_at, model_label, lesion_probability,
            class_probabilities, checkpoint, threshold, calibrated,
            ood_flagged, ood_score, shared, consent_at, image_b64, image_sha256, image_bytes,
            triage_bucket, triage_reason, origin_country
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
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
        imageBytes,
        triage.bucket,
        triage.reason,
        user.signup_country ?? null
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

  return json(
    {
      submission_id,
      created_at: created,
      shared: isShared === 1,
      triage_bucket: triage.bucket,
      triage_reason: triage.reason,
    },
    201
  );
}

/**
 * User feedback. Deliberately cannot touch review_status or admin_label —
 * this endpoint writes only to the untrusted columns.
 */
async function submitFeedback(db, submissionId, body) {
  const { user_id, says_wrong, suggested_label, comment } = body;
  if (!user_id) return fail(400, "user_id is required");
  // 'misc' is allowed here as well as the three classes, and means "this is not
  // a radiograph at all". It is the only user statement that can move a row
  // between buckets, and it still cannot label anything: it is read as evidence
  // about the gate, not as ground truth about the image.
  if (suggested_label && !REVIEW_LABELS.has(suggested_label)) {
    return fail(400, `suggested_label must be one of ${[...REVIEW_LABELS].join(", ")}`);
  }
  const row = await db
    .prepare(
      `SELECT user_id, review_status, ood_flagged, class_probabilities
         FROM submissions WHERE submission_id = ?`
    )
    .bind(submissionId)
    .first();
  if (!row) return fail(404, "no such submission");
  // A user may only annotate their own submission.
  if (row.user_id !== user_id) return fail(403, "not your submission");
  // Once reviewed, the record is evidence; late edits would rewrite history.
  if (row.review_status !== "pending") {
    return fail(409, "this submission has already been reviewed");
  }

  // Re-triage. A dispute is the second half of the evidence: "the gate turned
  // my radiograph away" and "the model diagnosed my lunch" are both only
  // visible once the user says so, and both belong in the contradiction queue
  // rather than wherever the row landed on arrival.
  let parsedProbabilities = {};
  try {
    parsedProbabilities = JSON.parse(row.class_probabilities || "{}");
  } catch {
    parsedProbabilities = {}; // a malformed blob must not block feedback
  }
  const triage = triageBucket({
    oodFlagged: Boolean(row.ood_flagged),
    maxProb: maxProbability(parsedProbabilities),
    userSaysWrong: Boolean(says_wrong),
    userSuggestedLabel: suggested_label ?? null,
  });

  await db
    .prepare(
      `UPDATE submissions
         SET user_says_wrong = ?, user_suggested_label = ?, user_comment = ?, feedback_at = ?,
             triage_bucket = ?, triage_reason = ?
       WHERE submission_id = ?`
    )
    .bind(
      says_wrong ? 1 : 0,
      suggested_label ?? null,
      comment ? String(comment).slice(0, 2000) : null,
      nowIso(),
      triage.bucket,
      triage.reason,
      submissionId
    )
    .run();
  return json({
    submission_id: submissionId,
    recorded: true,
    triage_bucket: triage.bucket,
    triage_reason: triage.reason,
  });
}

async function listUserSubmissions(db, userId, limit) {
  if (!userId) return fail(400, "user_id is required");
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  const { results } = await db
    .prepare(
      `SELECT submission_id, created_at, model_label, lesion_probability, checkpoint,
              ood_flagged, shared, user_says_wrong, user_suggested_label,
              triage_bucket, review_status, admin_bucket, admin_label
         FROM submissions WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?`
    )
    .bind(userId, n)
    .all();
  return json({ submissions: results ?? [] });
}

// --- Admin ------------------------------------------------------------------

async function pendingReview(db, limit, includeImages, bucket) {
  const n = Math.min(Number(limit) || 25, MAX_PAGE_SIZE);
  if (bucket && !BUCKETS.has(bucket)) {
    return fail(400, `bucket must be one of ${[...BUCKETS].join(", ")}`);
  }
  const columns = `submission_id, user_id, created_at, model_label, lesion_probability,
                   class_probabilities, ood_flagged, ood_score,
                   triage_bucket, triage_reason,
                   user_says_wrong, user_suggested_label, user_comment${
                     includeImages ? ", image_b64" : ""
                   }`;
  // One bucket at a time. The review decision differs by bucket — a bone film
  // needs a diagnosis, a hotdog needs confirming as a hotdog — and mixing them
  // in one list is how a reviewer ends up assigning a clinical class to a
  // photograph of a car park out of sheer momentum.
  const { results } = await db
    .prepare(
      `SELECT ${columns} FROM submissions
        WHERE shared = 1 AND review_status = 'pending'
          AND (? IS NULL OR triage_bucket = ?)
        ORDER BY user_says_wrong DESC, created_at ASC
        LIMIT ?`
    )
    .bind(bucket ?? null, bucket ?? null, n)
    .all();
  // Ordered so disputed results surface first — those are the ones with new
  // information in them; agreeing predictions teach the model least.
  return json({ pending: results ?? [], bucket: bucket ?? null });
}

/**
 * The gate. Approving requires the reviewer to state *both* what the image is
 * (admin_label) and what it is for (admin_bucket), and the two must agree.
 *
 * The bucket is not defaulted from `triage_bucket`, and that is the whole point
 * of the endpoint: the automatic bucket is the guess of the gate being
 * retrained, so accepting it unchallenged would feed the gate its own output
 * and teach it nothing but confidence in what it already believed. The reviewer
 * confirms it, which in the ordinary case means re-typing the same value — and
 * in the interesting case means noticing that it is wrong.
 */
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
    // Mirrors the bucket_and_label_must_agree trigger. Checked here so the
    // reviewer gets a sentence rather than a constraint failure, and enforced
    // there so it holds when this check is the thing that is wrong.
    if (admin_bucket === "misc" && admin_label !== "misc") {
      return fail(400, "a misc row has no diagnosis: label it 'misc'");
    }
    if (admin_bucket === "valid_bone" && admin_label === "misc") {
      return fail(400, "a bone radiograph needs a clinical label, not 'misc'");
    }
  }
  const row = await db
    .prepare("SELECT shared, review_status, image_b64 FROM submissions WHERE submission_id = ?")
    .bind(submissionId)
    .first();
  if (!row) return fail(404, "no such submission");
  if (row.review_status !== "pending") return fail(409, "already reviewed");
  if (decision === "approved" && row.shared !== 1) {
    return fail(400, "cannot approve a submission the user did not share");
  }
  if (decision === "approved" && !row.image_b64) {
    return fail(400, "cannot approve a row with no image: there is nothing to train on");
  }

  // Two statements, batched: the label/bucket pair is written first so the
  // approval trigger sees it, and both land atomically so a row can never be
  // approved with only half of its ground truth.
  await db.batch([
    db
      .prepare(
        `UPDATE submissions SET admin_label = ?, admin_bucket = ? WHERE submission_id = ?`
      )
      .bind(
        decision === "approved" ? admin_label : null,
        decision === "approved" ? admin_bucket : null,
        submissionId
      ),
    db
      .prepare(
        `UPDATE submissions
            SET review_status = ?, admin_note = ?, reviewed_at = ?, reviewed_by = ?
          WHERE submission_id = ?`
      )
      .bind(
        decision,
        note ? String(note).slice(0, 2000) : null,
        nowIso(),
        reviewed_by || ADMIN_USER_ID,
        submissionId
      ),
  ]);
  return json({
    submission_id: submissionId,
    review_status: decision,
    admin_label: decision === "approved" ? admin_label : null,
    admin_bucket: decision === "approved" ? admin_bucket : null,
  });
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
    )
    .bind(n)
    .all();

  const rows = results ?? [];
  // Split by what each row retrains. A 'misc' row is an OOD negative and has no
  // lesion class; a bone row is a lesion example. A 'contradiction' row is
  // whichever its label says it is — the bucket records that the gate got it
  // wrong, and the label records what the image actually was.
  const lesionRows = rows.filter((r) => r.admin_label !== "misc").length;
  const oodRows = rows.length - lesionRows;

  if (dry_run || rows.length === 0) {
    return json({
      batch_id: null,
      count: rows.length,
      lesion_rows: lesionRows,
      ood_rows: oodRows,
      rows: dry_run ? rows : [],
      dry_run: true,
    });
  }

  const id = batch_id || `batch-${nowIso().replace(/[:.]/g, "").replace("Z", "")}`;
  const stamp = nowIso();
  await db.batch([
    db
      .prepare(
        `INSERT INTO batches (batch_id, created_at, note, row_count, lesion_rows, ood_rows)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
      .bind(id, stamp, note ?? null, rows.length, lesionRows, oodRows),
    ...rows.map((r) =>
      db
        .prepare("UPDATE submissions SET batch_id = ?, exported_at = ? WHERE submission_id = ?")
        .bind(id, stamp, r.submission_id)
    ),
  ]);
  return json({ batch_id: id, count: rows.length, lesion_rows: lesionRows, ood_rows: oodRows, rows });
}

// --- Router -----------------------------------------------------------------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (!env.DB) return fail(500, "D1 binding 'DB' is not configured");

    // This is the one browser-facing route. It accepts only a short-lived,
    // one-use token minted by the authenticated app route below. Cloudflare
    // can therefore resolve the visitor's country without exposing API_KEY,
    // accepting a client-claimed country, or storing an IP address.
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

    // Every data route is authenticated. /location/capture above is narrowly
    // token-gated because it must be called by the browser itself.
    const key = bearer(request);
    const isAdmin = env.ADMIN_KEY ? timingSafeEqual(key, env.ADMIN_KEY) : false;
    const isApp = env.API_KEY ? timingSafeEqual(key, env.API_KEY) : false;
    if (!isAdmin && !isApp) return fail(401, "unauthorized");

    // /admin needs two things, because they answer two different questions.
    // The key says the caller is trusted software; the header says which
    // account is asking. Only one account may review, and it is hardcoded --
    // so a signed-in ordinary user cannot be served the review queue by an app
    // process that happens to hold the admin key.
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
      // Aggregated country counts for the landing-page globe. Still behind the
      // API key like every ordinary data route. The Streamlit server calls it
      // and renders the result, so the key never leaves the server.
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
  },
};
