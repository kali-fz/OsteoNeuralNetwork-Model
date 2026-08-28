/**
 * Signed session cookies for the Pages frontend.
 *
 * WHAT THIS REPLACES
 * ------------------
 * Streamlit's `st.login()` / `st.user`, which kept the signed-in identity in a
 * cookie managed by the Streamlit server. There is no Streamlit server any more,
 * so the Pages Functions layer has to hold the session itself.
 *
 * WHY A COOKIE AND NOT A TOKEN IN localStorage
 * --------------------------------------------
 * The site and its API share one origin (`onnm.pages.dev`), which makes an
 * httpOnly cookie both possible and clearly correct. A token in localStorage is
 * readable by any script that ends up on the page, and this application renders
 * user-supplied display names and Google avatar URLs. httpOnly removes that
 * entire class of theft: script on the page cannot read the session at all.
 *
 * This is also the reason the API is served from Pages Functions rather than
 * from the existing `*.workers.dev` Worker. `pages.dev` and `workers.dev` are
 * different registrable domains, so a cookie set by the Worker would simply
 * never be sent to the site.
 *
 * TOKEN FORMAT
 * ------------
 * `base64url(JSON payload) + "." + base64url(HMAC-SHA256)`. Deliberately not a
 * JWT: there is exactly one issuer, one consumer and one algorithm here, and a
 * JWT library's flexibility -- `alg` negotiation above all -- is a liability
 * rather than a feature at this size. Nothing is encrypted, only signed; the
 * payload holds no secret, and the client is never meant to parse it anyway.
 *
 * WHAT IS DELIBERATELY NOT IN THE PAYLOAD
 * ---------------------------------------
 * No `is_admin` flag. Administrative authority is pinned to one hardcoded
 * account id in `cloudflare/src/worker.js` and again as a CHECK constraint in
 * `schema.sql`; putting a boolean in a cookie would create a third place that
 * has to agree, and the only one an attacker could influence.
 */

const ENCODER = new TextEncoder();
const DECODER = new TextDecoder();

/** Cookie name. Prefixed so a browser refuses it if it is ever sent without Secure. */
export const SESSION_COOKIE = "__Host-onnm_session";

/** Eight hours. Long enough to finish a sitting, short enough that a shared machine forgets. */
export const SESSION_TTL_SECONDS = 8 * 60 * 60;

function base64urlEncode(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecode(text) {
  const padded = text.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

async function hmacKey(secret) {
  return crypto.subtle.importKey(
    "raw",
    ENCODER.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

/**
 * Constant-time comparison of two byte arrays.
 *
 * `crypto.subtle.verify` is already constant-time and is what actually guards
 * the signature; this exists for the places where two strings must be compared
 * directly, such as the OAuth `state` round trip.
 */
export function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * Sign a payload into a session token.
 *
 * `iat` and `exp` are added here rather than by the caller so that every token
 * in the system has an expiry and none can be minted without one.
 */
export async function signSession(payload, secret, ttlSeconds = SESSION_TTL_SECONDS) {
  const now = Math.floor(Date.now() / 1000);
  const body = { ...payload, iat: now, exp: now + ttlSeconds };
  const encoded = base64urlEncode(ENCODER.encode(JSON.stringify(body)));
  const key = await hmacKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, ENCODER.encode(encoded));
  return `${encoded}.${base64urlEncode(new Uint8Array(signature))}`;
}

/**
 * Verify a token and return its payload, or null.
 *
 * Returns null for every failure mode -- malformed, bad signature, expired --
 * rather than distinguishing them. A caller cannot do anything useful with the
 * difference, and reporting it back would tell an attacker which half of a
 * forgery attempt was wrong.
 */
export async function verifySession(token, secret) {
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
      ENCODER.encode(encoded),
    );
  } catch {
    return null;
  }
  if (!valid) return null;

  try {
    const payload = JSON.parse(DECODER.decode(base64urlDecode(encoded)));
    if (typeof payload?.exp !== "number" || payload.exp < Math.floor(Date.now() / 1000)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

/** Read one cookie out of a request. */
export function readCookie(request, name) {
  const header = request.headers.get("cookie") || "";
  for (const part of header.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return null;
}

/**
 * The signed-in account for this request, or null.
 *
 * Every browser-facing route funnels through this. Nothing downstream should
 * ever take a user id from a query string or request body -- that is precisely
 * the mistake this function exists to make unnecessary.
 */
export async function currentSession(request, env) {
  if (!env.SESSION_SECRET) return null;
  const token = readCookie(request, SESSION_COOKIE);
  if (!token) return null;
  return verifySession(token, env.SESSION_SECRET);
}

/**
 * Build the Set-Cookie header for a session.
 *
 * `__Host-` forces Secure, Path=/ and no Domain, which together mean the cookie
 * cannot be set by, or leak to, a sibling subdomain. SameSite=Lax rather than
 * Strict because the OAuth callback is a top-level cross-site navigation back
 * from Google, and Strict would drop the cookie on exactly that redirect.
 */
export function sessionCookieHeader(token, { maxAge = SESSION_TTL_SECONDS } = {}) {
  return `${SESSION_COOKIE}=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`;
}

/** Expire the session cookie. Same attributes, or the browser will not match it. */
export function clearSessionCookieHeader() {
  return `${SESSION_COOKIE}=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0`;
}

/**
 * A short-lived signed cookie for values that must survive the trip to Google
 * and back: the OAuth `state` and the PKCE `code_verifier`.
 *
 * Kept in a cookie rather than in D1 because it is per-browser, single-use and
 * expires in minutes. Writing it to the database would mean a row per abandoned
 * sign-in attempt and a cleanup job to match.
 */
export const OAUTH_COOKIE = "__Host-onnm_oauth";
export const OAUTH_TTL_SECONDS = 10 * 60;

export function oauthCookieHeader(token) {
  return `${OAUTH_COOKIE}=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${OAUTH_TTL_SECONDS}`;
}

export function clearOauthCookieHeader() {
  return `${OAUTH_COOKIE}=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0`;
}
