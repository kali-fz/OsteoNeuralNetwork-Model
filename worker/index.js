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

import { handleApiRequest } from "../cloudflare/src/worker.js";
import { InferenceContainer, inferenceStub } from "./container.js";
import { budgetStatus } from "./lib/breaker.js";
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

  // Look up by Google `sub`, then create. Keying on the subject rather than the
  // email is what makes an address change at the Google end harmless.
  const account = await forwardJson(request, env, "/users/by-subject", {
    method: "GET",
    search: { subject: profile.subject },
  });
  // getUserBySubject returns the user row itself -- `json(row)` -- not a
  // wrapper object. Reading `.user` here found undefined for an account that
  // existed, so sign-in fell through to account creation, hit the UNIQUE
  // constraint on email, and reported "account could not be opened" to a user
  // whose account was in the database the whole time.
  const existing = account.ok ? account.payload : null;

  let userId = existing?.user_id || null;
  if (!userId) {
    const created = await forwardJson(request, env, "/users", {
      method: "POST",
      body: {
        email: profile.email,
        auth_provider: "google",
        provider_subject: profile.subject,
        display_name: profile.name,
        profile_picture_url: profile.picture,
      },
    });
    if (!created.ok) {
      console.error("oauth: account creation failed", created.status, created.payload);
      return bounce(created.status === 403 ? "registration_closed" : "account_failed");
    }
    userId = created.payload?.user_id || created.payload?.user?.user_id || null;
  }

  if (!userId) return bounce("account_failed");

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
      ["set-cookie", sessionCookieHeader(token)],
    ],
  });
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
        return json({
          signed_in: Boolean(session),
          user: session
            ? { user_id: session.uid, email: session.email, name: session.name, picture: session.pic }
            : null,
          budget: await budgetStatus(env.DB),
        });
      }
      if (method === "POST" && path === "/api/auth/signout") {
        return json({ signed_out: true }, 200, { "set-cookie": clearSessionCookieHeader() });
      }

      // -- authenticated --------------------------------------------------
      const session = await currentSession(request, env);

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

      if (method === "POST" && path === "/api/warmup") return warmup(request, env);
      if (method === "POST" && path === "/api/scan") return scan(request, env);

      return fail(404, `no route for ${method} ${path}`);
    } catch (error) {
      console.error("unhandled", error);
      return fail(500, "internal error");
    }
  },
};
