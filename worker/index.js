/**
 * ONNM: the whole application, as one Worker.
 *
 * Static assets are served by the platform (see `assets` in wrangler.jsonc);
 * this script runs for `/api/*` and nothing else.
 *
 * WHY THIS IS NOT A TRANSPARENT PROXY TO worker.js
 * ------------------------------------------------
 * The existing `cloudflare/src/worker.js` was designed around one caller: the
 * Streamlit *server*, which held the API key and decided what to ask for. Its
 * routes trust their caller accordingly -- `/users/by-email` will return any
 * account, and `/submissions?user_id=` will list anyone's history.
 *
 * The browser is now the caller. Forwarding `/api/*` straight through would
 * hand every visitor the authority Streamlit had. So this file exposes a
 * curated set of routes, each of which derives the acting user from the signed
 * session cookie and never from anything the client sent. `user_id` is not a
 * parameter here; it is a conclusion.
 *
 * The storage work is still done by the handlers in worker.js, reached through
 * `forward()` with the API key attached server-side. Routing, the review gate
 * and the spend guards therefore keep exactly one implementation.
 */

import {
  ADMIN_USER_ID,
  handleApiRequest,
  purgeRejectedImages,
} from "../cloudflare/src/worker.js";
import { InferenceContainer, inferenceStub } from "./container.js";
import { budgetStatus } from "./lib/breaker.js";
import { resolveGoogleAccount } from "./lib/account.js";
import { bucketFor, isAdminSession } from "./lib/review.js";
import {
  acceptedVersionFromToken,
  clearTermsCookieHeader,
  hasAcceptedTerms,
  signTermsToken,
  TERMS_COOKIE,
  TERMS_VERSION,
  termsCookieHeader,
} from "./lib/terms.js";
import { buildMarkers } from "./lib/geo.js";
import {
  authorizationUrl,
  claimsFromIdToken,
  codeChallenge,
  exchangeCode,
  identityProfile,
  randomToken,
} from "./lib/google.js";
import {
  clearOauthCookieHeader,
  clearSessionCookieHeader,
  currentSession,
  OAUTH_COOKIE,
  OAUTH_TTL_SECONDS,
  oauthCookieHeader,
  readCookie,
  sessionCookieHeader,
  signSession,
  timingSafeEqual,
  verifySession,
} from "./lib/session.js";

export { InferenceContainer };

const json = (data, status = 200, headers = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...headers },
  });

const fail = (status, message, extra = {}) => json({ error: message, ...extra }, status);

/**
 * Call a `worker.js` route with the app API key attached.
 *
 * The original Request is passed through rather than rebuilt so that the `cf`
 * object survives. That matters for exactly one route: `/location/capture`
 * resolves the visitor's country from `request.cf.country`, which is how this
 * application records geography without ever seeing an IP address.
 */
async function forward(request, env, path, { method, body, search } = {}) {
  const url = new URL(request.url);
  url.pathname = path;
  // Cleared by default rather than inherited: the browser's query string must
  // never reach a backend route by accident. Callers pass what they mean.
  url.search = search ? `?${new URLSearchParams(search).toString()}` : "";

  const headers = new Headers();
  headers.set("authorization", `Bearer ${env.API_KEY}`);
  if (body !== undefined) headers.set("content-type", "application/json");

  const init = { method: method || request.method, headers, cf: request.cf };
  if (body !== undefined) init.body = JSON.stringify(body);

  return handleApiRequest(new Request(url.toString(), init), env);
}

/** Read a forwarded response as JSON, preserving the upstream failure if there was one. */
async function forwardJson(request, env, path, options) {
  const response = await forward(request, env, path, options);
  const payload = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, payload };
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

function redirectUriFor(request) {
  return new URL("/api/auth/google/callback", new URL(request.url).origin).toString();
}

/**
 * Mint a one-use location token and immediately spend it on this request.
 *
 * Never throws. A country is a decoration on a globe; failing to record one
 * must not cost somebody their sign-in. Failures are logged, not surfaced.
 */
async function captureCountry(request, env, userId) {
  try {
    const minted = await forwardJson(request, env, "/location/token", {
      method: "POST",
      body: { user_id: userId },
    });
    if (!minted.ok || !minted.payload?.token) return;

    const url = new URL(request.url);
    url.pathname = "/location/capture";
    url.search = "";

    // The original request's `cf` is passed through deliberately: it is the only
    // thing in this call that knows the country, and it arrived already reduced
    // to a two-letter code by Cloudflare.
    await handleApiRequest(
      new Request(url.toString(), {
        method: "POST",
        headers: { authorization: `Bearer ${minted.payload.token}` },
        cf: request.cf,
      }),
      env,
    );
  } catch (error) {
    console.error("country capture failed", error);
  }
}

/**
 * Begin sign-in.
 *
 * `state` and the PKCE verifier are carried in a short-lived signed cookie
 * rather than in D1. They are per-browser, single-use and expire in minutes;
 * a database row would mean one write per abandoned sign-in and a cleanup job
 * to match.
 */
async function authStart(request, env) {
  if (!env.GOOGLE_CLIENT_ID || !env.GOOGLE_CLIENT_SECRET || !env.SESSION_SECRET) {
    return fail(503, "Google sign-in is not configured on this deployment");
  }

  // The gate. Without a valid acceptance cookie no account can be created,
  // whatever the page that sent the visitor here happened to render. Bouncing
  // back to /terms rather than returning an error page, because this URL is
  // reached by a top-level navigation and whatever it returns is what is seen.
  const accepted = await acceptedVersionFromToken(
    readCookie(request, TERMS_COOKIE),
    env.SESSION_SECRET,
  );
  if (!accepted) {
    const origin = new URL(request.url).origin;
    return new Response(null, {
      status: 302,
      headers: { location: `${origin}/terms?auth_error=terms_required` },
    });
  }

  const state = randomToken();
  const verifier = randomToken(48);
  const challenge = await codeChallenge(verifier);

  const token = await signSession({ state, verifier }, env.SESSION_SECRET, OAUTH_TTL_SECONDS);
  const url = authorizationUrl({
    clientId: env.GOOGLE_CLIENT_ID,
    redirectUri: redirectUriFor(request),
    state,
    challenge,
  });

  return new Response(null, {
    status: 302,
    headers: { location: url, "set-cookie": oauthCookieHeader(token) },
  });
}

/**
 * Finish sign-in.
 *
 * Failures redirect to the app with an error code rather than rendering an API
 * error page: this URL is reached by a top-level browser navigation back from
 * Google, so whatever it returns is what the visitor sees.
 */
async function authCallback(request, env) {
  const url = new URL(request.url);
  const origin = url.origin;
  const bounce = (code) =>
    new Response(null, {
      status: 302,
      headers: { location: `${origin}/?auth_error=${code}`, "set-cookie": clearOauthCookieHeader() },
    });

  if (url.searchParams.get("error")) return bounce("declined");

  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  if (!code || !state) return bounce("missing_code");

  const acceptedVersion = await acceptedVersionFromToken(
    readCookie(request, TERMS_COOKIE),
    env.SESSION_SECRET,
  );

  const pending = await verifySession(readCookie(request, OAUTH_COOKIE), env.SESSION_SECRET);
  if (!pending) return bounce("expired");
  // Constant-time, and compared against the cookie we signed rather than
  // anything else in the request. This is the CSRF control for the flow.
  if (!timingSafeEqual(String(state), String(pending.state))) return bounce("state_mismatch");

  let claims;
  try {
    const tokens = await exchangeCode({
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
      redirectUri: redirectUriFor(request),
      code,
      verifier: pending.verifier,
    });
    claims = claimsFromIdToken(tokens.id_token, env.GOOGLE_CLIENT_ID);
  } catch (error) {
    console.error("oauth: exchange failed", error);
    // An unverified email lands here too, and is the one failure worth naming
    // to the user, because it is fixable on their side.
    const reason = String(error?.message || "").includes("not verified")
      ? "email_unverified"
      : "exchange_failed";
    return bounce(reason);
  }

  const profile = identityProfile(claims);

  // Prefer Google `sub`, then retain the legacy email fallback for accounts
  // created before a provider subject was stored, and only then create. New
  // identities still key on the subject, so later Google email changes are safe.
  const account = await resolveGoogleAccount(profile, {
    lookupBySubject: (subject) =>
      forwardJson(request, env, "/users/by-subject", {
        method: "GET",
        search: { subject },
      }),
    lookupByEmail: (email) =>
      forwardJson(request, env, "/users/by-email", {
        method: "GET",
        search: { email },
      }),
    createUser: (body) =>
      forwardJson(request, env, "/users", {
        method: "POST",
        // The version agreed to on the way in. authStart refused to start this
        // flow without it, so by here it is known to exist and to be ours.
        body: { ...body, tos_version: acceptedVersion || TERMS_VERSION },
      }),
  });
  if (!account.ok) {
    console.error("oauth: account resolution failed", account.status, account.payload);
    return bounce(account.status === 403 ? "registration_closed" : "account_failed");
  }
  const userId = account.userId;

  // Refresh the display name and avatar on every sign-in, mirroring the
  // profile_sync the Streamlit app did on each rerun.
  // provider_subject is REQUIRED: updateContributorProfile re-checks it against
  // the stored subject in constant time, so that holding the API key is not by
  // itself enough to rewrite somebody else's public profile. Omitting it made
  // this a silent 400 behind the catch below.
  await forward(request, env, "/users/profile", {
    method: "POST",
    body: {
      user_id: userId,
      provider_subject: profile.subject,
      display_name: profile.name,
      profile_picture_url: profile.picture,
    },
  }).catch(() => {});

  // Record the country, once, from Cloudflare's edge resolution of THIS request.
  //
  // The Streamlit app needed a zero-height iframe and a one-use token for this,
  // because the page could not be trusted with the API key and the Worker was on
  // a different origin. Neither is true any more: this Worker is the edge, and
  // request.cf.country is already resolved and already coarsened to two letters.
  // No IP address is seen here, and the browser Geolocation API is still never
  // called.
  //
  // The token dance is kept rather than bypassed because captureBrowserCountry
  // carries the atomicity and the "first capture wins" guarantee that stop an
  // account's country being rewritten by later travel. Running it on every
  // sign-in is therefore safe: the second attempt is a no-op by construction.
  await captureCountry(request, env, userId);

  const token = await signSession(
    { uid: userId, sub: profile.subject, email: profile.email, name: profile.name, pic: profile.picture },
    env.SESSION_SECRET,
  );

  return new Response(null, {
    status: 302,
    headers: [
      ["location", `${origin}/`],
      ["set-cookie", clearOauthCookieHeader()],
      // Spent. It exists only to carry the acceptance across the trip to Google,
      // and the account row now holds it durably.
      ["set-cookie", clearTermsCookieHeader()],
      ["set-cookie", sessionCookieHeader(token)],
    ],
  });
}

// ---------------------------------------------------------------------------
// The Terms gate
// ---------------------------------------------------------------------------

/**
 * The account row behind a session, or null.
 *
 * Looked up by Google subject rather than by the `uid` in the cookie, because the
 * subject is the identity Google vouched for and the one `users` is keyed on for
 * OAuth accounts.
 */
async function accountFor(request, env, session) {
  if (!session?.sub) return null;
  const found = await forwardJson(request, env, "/users/by-subject", {
    method: "GET",
    search: { subject: session.sub },
  });
  return found.ok ? found.payload : null;
}

/**
 * Has the acting account accepted the Terms?
 *
 * A database read on every call rather than a claim carried in the session
 * cookie. The cookie is issued for eight hours; if acceptance lived in it, an
 * account that agreed would keep being asked until it expired, and worse, an
 * account whose acceptance was withdrawn would keep scanning. The same reasoning
 * already keeps `is_admin` out of the cookie -- see lib/session.js.
 */
async function sessionHasAcceptedTerms(request, env, session) {
  if (!session) return false;
  return hasAcceptedTerms(await accountFor(request, env, session));
}

/**
 * Record acceptance. Works signed in and signed out; the server decides which.
 *
 * Signed in, it writes to the account row. Signed out, there is no row yet, so it
 * mints the short-lived signed cookie that `authStart` requires -- which is what
 * makes the tick box on the page consequential rather than decorative.
 *
 * The recorded version is always TERMS_VERSION, never the value the client sent.
 * A client that could choose its own version string could choose to have agreed
 * to something else.
 */
async function acceptTerms(request, env) {
  if (!env.SESSION_SECRET) return fail(503, "sign-in is not configured on this deployment");

  const session = await currentSession(request, env);

  if (session) {
    const recorded = await forwardJson(request, env, "/users/terms", {
      method: "POST",
      body: { user_id: session.uid, version: TERMS_VERSION },
    });
    if (!recorded.ok) {
      console.error("terms: could not record acceptance", recorded.status, recorded.payload);
      return fail(502, "your agreement could not be saved; please try again");
    }
    return json({ accepted: true, version: TERMS_VERSION, signed_in: true });
  }

  const token = await signTermsToken(env.SESSION_SECRET);
  return json({ accepted: true, version: TERMS_VERSION, signed_in: false }, 200, {
    "set-cookie": termsCookieHeader(token),
  });
}

/**
 * Appear on the public contributors list, or stop appearing.
 *
 * The Terms say a name and photo are shown publicly only if the person chooses
 * to appear as a contributor, and the Privacy notice promises the choice is
 * reversible at any time. This route is where both of those become true; the
 * column has defaulted to 0 since migration 0005, so silence stays private.
 *
 * The identity is taken from the session cookie and never from the body. A
 * client-supplied user_id or provider_subject would let anyone publish -- or
 * unpublish -- somebody else's profile. updateContributorProfile re-checks the
 * subject against the stored one regardless, so this is the second of two locks
 * rather than the only one.
 *
 * The name and picture are re-sent from the session because turning the setting
 * off nulls both columns. Turning it back on has to restore them, and the
 * session already holds what Google last told us.
 */
async function setProfileVisibility(request, env, session) {
  const body = await request.json().catch(() => ({}));
  // Explicitly boolean rather than truthy: a missing or misspelled field would
  // otherwise read as false and quietly unpublish somebody.
  if (typeof body?.public_profile !== "boolean") {
    return fail(400, "public_profile must be true or false");
  }

  const updated = await forwardJson(request, env, "/users/profile", {
    method: "POST",
    body: {
      user_id: session.uid,
      provider_subject: session.sub,
      public_profile: body.public_profile,
      display_name: session.name,
      profile_picture_url: session.pic,
    },
  });
  if (!updated.ok) {
    console.error("profile: visibility update failed", updated.status, updated.payload);
    return fail(502, "your preference could not be saved; please try again");
  }
  return json({ public_profile: body.public_profile });
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/**
 * Wake the container without running a scan.
 *
 * Called the moment a visitor picks a file. Container cold start plus torch's
 * import is 5-8 seconds, and this spends it while they are still looking at the
 * file picker rather than after they press Analyse. It costs the same runtime
 * the scan would have cost anyway, because the wake period is billed either way.
 */
async function warmup(request, env) {
  const session = await currentSession(request, env);
  if (!session) return fail(401, "sign in first");

  // Waking a container that bills by the second, for an account that is not
  // allowed to scan, would be paying for a refusal.
  if (!(await sessionHasAcceptedTerms(request, env, session))) {
    return json({ warming: false, reason: "terms_required" }, 200);
  }

  const budget = await budgetStatus(env.DB);
  if (!budget.allowed) return json({ warming: false, budget }, 200);

  try {
    const response = await inferenceStub(env).containerFetch(
      new Request("http://container/health", { method: "GET" }),
    );
    return json({ warming: true, ready: response.ok, budget });
  } catch (error) {
    console.error("warmup failed", error);
    return json({ warming: false, ready: false, budget });
  }
}

async function scan(request, env) {
  const session = await currentSession(request, env);
  if (!session) return fail(401, "sign in first");

  // Routing sends an unaccepted account to /terms, but routing is a convention
  // any client can ignore. This is the part that holds.
  if (!(await sessionHasAcceptedTerms(request, env, session))) {
    return fail(403, "please accept the Terms of use before scanning", {
      reason: "terms_required",
    });
  }

  // The spend guard runs before any container work, so an exhausted budget
  // costs nothing to refuse.
  const budget = await budgetStatus(env.DB);
  if (!budget.allowed) {
    return json({ error: budget.message, reason: budget.reason, budget }, 429);
  }

  const form = await request.formData().catch(() => null);
  const file = form?.get("file");
  if (!file || typeof file === "string") return fail(400, "no file uploaded");

  // Rebuilt rather than forwarded so that only the fields the container accepts
  // reach it, and so the browser cannot set `want_preprocessed` itself.
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
        body: outbound,
      }),
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      return json({ error: payload?.detail || "inference failed" }, response.status);
    }
    result = payload;
  } catch (error) {
    console.error("scan: container call failed", error);
    return fail(502, "the model is temporarily unavailable");
  }

  // An OOD rejection is recorded but never predicted on, matching app.py.
  if (!result.is_radiograph) {
    return json({ ...result, budget });
  }

  const shared = String(form.get("share_consent") || "") === "true";
  const prediction = result.prediction || {};
  const recorded = await forwardJson(request, env, "/submissions", {
    method: "POST",
    body: {
      // Required by createSubmission, and generated here rather than in the
      // container so that the id belongs to the record, not to the inference.
      submission_id: crypto.randomUUID(),
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
      image_bytes: result.preprocessed?.bytes || null,
    },
  });

  return json({
    ...result,
    // The stored image never travels back to the browser; it has the overlay
    // and the original already, and this keeps the response small.
    preprocessed: undefined,
    submission_id: recorded.ok ? recorded.payload?.submission_id : null,
    stored: recorded.ok,
    budget,
  });
}

// ---------------------------------------------------------------------------
// Review (the owner's account only)
// ---------------------------------------------------------------------------

/**
 * Call an `/admin/*` route on worker.js with both credentials it demands.
 *
 * Two separate questions, deliberately: ADMIN_KEY says the caller is trusted
 * software, and `x-onnm-admin-user` says which account is asking. Neither
 * value is reachable from the browser -- the key is a Worker secret and the id
 * is written here, not read from the request -- so the only way to reach these
 * routes is to have already passed isAdminSession() in lib/review.js.
 *
 * Kept separate from forward() so that no ordinary route can reach /admin/* by
 * accident: forward() attaches API_KEY, which worker.js refuses there.
 */
async function forwardAdmin(request, env, path, { method, body, search } = {}) {
  const url = new URL(request.url);
  url.pathname = path;
  url.search = search ? `?${new URLSearchParams(search).toString()}` : "";

  const headers = new Headers();
  headers.set("authorization", `Bearer ${env.ADMIN_KEY}`);
  headers.set("x-onnm-admin-user", ADMIN_USER_ID);
  if (body !== undefined) headers.set("content-type", "application/json");

  const init = { method: method || request.method, headers, cf: request.cf };
  if (body !== undefined) init.body = JSON.stringify(body);

  return handleApiRequest(new Request(url.toString(), init), env);
}

/** The queue, with images, plus the counts that head the page. */
async function adminQueue(request, env, bucket) {
  // `bucket` is omitted rather than sent as null when absent: URLSearchParams
  // would render it as the string "null", which is not one of the three and
  // would be refused as a bad bucket rather than read as "all of them".
  const search = { limit: "24", images: "1" };
  if (bucket) search.bucket = bucket;

  const [counts, queue] = await Promise.all([
    // The counts come through forward() and the queue through forwardAdmin():
    // /health is an ordinary route and /admin/pending is not, and each is
    // called with exactly the credential worker.js demands for it.
    forwardJson(request, env, "/health"),
    forwardAdmin(request, env, "/admin/pending", { method: "GET", search }),
  ]);

  const payload = await queue.json().catch(() => ({}));
  if (!queue.ok) return json(payload, queue.status);
  return json({
    bucket: bucket || null,
    pending: payload?.pending ?? [],
    counts: counts.ok ? counts.payload : null,
  });
}

/**
 * Approve or reject one submission.
 *
 * `ood_flagged` is read from the row rather than taken from the request, so the
 * bucket is derived from what the gate actually did and not from what a client
 * claims it did. The queue response carries the same field only so the page can
 * show the reviewer where a decision will file the row before they make it.
 */
async function adminReview(request, env, submissionId) {
  const body = await request.json().catch(() => ({}));
  const decision = body?.decision;

  if (decision === "rejected") {
    return forwardAdmin(request, env, `/admin/review/${submissionId}`, {
      method: "POST",
      body: { decision: "rejected", note: body?.note || null, reviewed_by: ADMIN_USER_ID },
    });
  }
  if (decision !== "approved") return fail(400, "decision must be 'approved' or 'rejected'");

  const label = String(body?.admin_label || "");
  const row = await env.DB.prepare(
    "SELECT ood_flagged FROM submissions WHERE submission_id = ?",
  )
    .bind(submissionId)
    .first();
  if (!row) return fail(404, "no such submission");

  return forwardAdmin(request, env, `/admin/review/${submissionId}`, {
    method: "POST",
    body: {
      decision: "approved",
      admin_label: label,
      admin_bucket: bucketFor(label, Number(row.ood_flagged) === 1),
      note: body?.note || null,
      reviewed_by: ADMIN_USER_ID,
    },
  });
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const method = request.method.toUpperCase();

    if (!path.startsWith("/api/")) {
      // Should be unreachable: `run_worker_first` only routes /api/* here.
      return env.ASSETS ? env.ASSETS.fetch(request) : fail(404, "not found");
    }
    if (!env.DB) return fail(500, "D1 binding 'DB' is not configured");

    try {
      // -- public ---------------------------------------------------------
      if (method === "GET" && path === "/api/auth/google/start") return authStart(request, env);
      if (method === "GET" && path === "/api/auth/google/callback") {
        return authCallback(request, env);
      }
      if (method === "GET" && path === "/api/globe") {
        // Country codes and counts come from D1; the conversion to coordinates
        // happens in lib/geo.js so that one function still describes the whole
        // of what the map is able to reveal.
        const upstream = await forwardJson(request, env, "/globe");
        if (!upstream.ok) return json(upstream.payload, upstream.status);
        return json(await buildMarkers(upstream.payload));
      }
      if (method === "GET" && path === "/api/contributors") {
        return forward(request, env, "/contributors");
      }
      if (method === "GET" && path === "/api/stats") return forward(request, env, "/health");

      // -- session --------------------------------------------------------
      if (method === "GET" && path === "/api/session") {
        const session = await currentSession(request, env);
        // One read of the account row, shared by both answers below. Asking for
        // it twice would be two round trips for the same row on every page load.
        const account = session ? await accountFor(request, env, session) : null;
        return json({
          signed_in: Boolean(session),
          user: session
            ? { user_id: session.uid, email: session.email, name: session.name, picture: session.pic }
            : null,
          // Recomputed here on every request rather than carried in the cookie,
          // so it cannot be forged and cannot drift. The frontend uses it only
          // to decide whether to draw the Admin link; every /api/admin/* route
          // re-derives it and would refuse regardless of what the page shows.
          is_admin: isAdminSession(session),
          // Read from the account row, not from the cookie, so that agreeing
          // takes effect on the next request rather than the next sign-in.
          terms_accepted: hasAcceptedTerms(account),
          terms_version: TERMS_VERSION,
          // Whether this account has chosen to appear on the public contributors
          // list. Read from the row rather than the cookie for the same reason as
          // is_admin: it can change without a new sign-in, and a stale "yes"
          // carried in an eight-hour cookie would keep somebody listed after they
          // asked not to be.
          public_profile: account?.public_contributor_profile === 1,
          budget: await budgetStatus(env.DB),
        });
      }
      if (method === "POST" && path === "/api/terms/accept") {
        return acceptTerms(request, env);
      }
      if (method === "POST" && path === "/api/auth/signout") {
        return json({ signed_out: true }, 200, { "set-cookie": clearSessionCookieHeader() });
      }

      // -- authenticated --------------------------------------------------
      const session = await currentSession(request, env);

      if (method === "POST" && path === "/api/profile/visibility") {
        if (!session) return fail(401, "sign in first");
        return setProfileVisibility(request, env, session);
      }

      if (method === "GET" && path === "/api/submissions") {
        if (!session) return fail(401, "sign in first");
        // The user id comes from the cookie. A client-supplied one is ignored,
        // which is the whole reason this route is not a pass-through.
        const forwarded = new URL(request.url);
        forwarded.pathname = "/submissions";
        forwarded.search = `?user_id=${encodeURIComponent(session.uid)}&limit=50`;
        return handleApiRequest(
          new Request(forwarded.toString(), {
            method: "GET",
            headers: { authorization: `Bearer ${env.API_KEY}` },
            cf: request.cf,
          }),
          env,
        );
      }

      // Withdrawing your own shared image. The submission id comes from the
      // URL but the account does not: it is taken from the session cookie, and
      // the storage layer re-checks the row against it, so naming somebody
      // else's submission gets a 404 rather than deleting their image.
      const withdraw = path.match(/^\/api\/submissions\/([^/]+)\/withdraw$/);
      if (method === "POST" && withdraw) {
        if (!session) return fail(401, "sign in first");
        return forward(request, env, `/submissions/${withdraw[1]}/withdraw`, {
          method: "POST",
          body: { user_id: session.uid },
        });
      }

      if (method === "POST" && path === "/api/warmup") return warmup(request, env);
      if (method === "POST" && path === "/api/scan") return scan(request, env);

      // -- review, one account --------------------------------------------
      //
      // The guard is here, once, in front of every /api/admin/* path, rather
      // than repeated per route: a route added below this line is closed by
      // default, which is the failure mode worth having. 404 rather than 403
      // for a signed-in stranger, so the queue's existence is not confirmed to
      // an account that may not have it.
      if (path.startsWith("/api/admin/")) {
        if (!isAdminSession(session)) {
          return session ? fail(404, `no route for ${method} ${path}`) : fail(401, "sign in first");
        }
        if (!env.ADMIN_KEY) {
          return fail(503, "ADMIN_KEY is not set on this deployment, so review is unavailable");
        }

        if (method === "GET" && path === "/api/admin/queue") {
          return adminQueue(request, env, url.searchParams.get("bucket") || null);
        }
        const decide = path.match(/^\/api\/admin\/review\/([^/]+)$/);
        if (method === "POST" && decide) return adminReview(request, env, decide[1]);
      }

      return fail(404, `no route for ${method} ${path}`);
    } catch (error) {
      console.error("unhandled", error);
      return fail(500, "internal error");
    }
  },

  /**
   * The retention job. Runs daily; deletes rejected images older than a week.
   *
   * WHY A CRON HERE IS NOT THE CRON THIS PROJECT FORBIDS
   * ----------------------------------------------------
   * wrangler.jsonc warns that a cron trigger is "the single most expensive
   * thing that could be added here". That warning is about a *keep-warm*
   * heartbeat: something that periodically reaches the container, which bills
   * by wall-clock runtime and would therefore never be allowed to sleep. At
   * roughly $0.074/hour, a container kept awake by a schedule is about $53 a
   * month.
   *
   * This handler touches D1 and nothing else. It never resolves the container
   * binding, never calls inferenceStub, and cannot wake anything that bills by
   * time. A scheduled invocation is charged as one request, so a daily run is
   * about 30 requests a month against an allowance of millions.
   *
   * Keep it that way. If this function ever needs the model, it does not need
   * the model -- it needs to become a route somebody triggers deliberately.
   *
   * Daily rather than weekly, with a seven-day window, so a rejected image is
   * gone within about a week of rejection rather than up to a fortnight.
   */
  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      (async () => {
        try {
          const result = await purgeRejectedImages(env.DB);
          if (result.purged) {
            console.log(`retention: deleted ${result.purged} rejected image(s) older than ${result.cutoff}`);
          }
        } catch (error) {
          // Logged, never thrown. A failed purge is worth knowing about, but
          // retrying it in a minute would not help and the next run is a day
          // away, which is soon enough for a seven-day window.
          console.error("retention: purge failed", error);
        }
      })(),
    );
  },
};
