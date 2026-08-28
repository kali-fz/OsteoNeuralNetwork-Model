/**
 * Google Sign-In, as an OIDC authorization-code flow with PKCE.
 *
 * WHAT THIS REPLACES
 * ------------------
 * `st.login()`. Streamlit implemented this flow internally using Authlib; with
 * Streamlit gone the flow has to exist explicitly. The externally visible
 * behaviour is deliberately identical, including the two rules that matter:
 *
 *   1. An account is keyed on Google's `sub`, never on the email address. An
 *      email can be reassigned by a Workspace administrator; `sub` cannot. This
 *      matches `src/oauth.py:resolve_account` and the partial unique index
 *      `idx_users_subject` in `cloudflare/schema.sql`.
 *   2. An unverified email is refused outright rather than signed in.
 *
 * ON ID TOKEN SIGNATURE VERIFICATION
 * ----------------------------------
 * The ID token here is not checked against Google's JWKS, and that is a
 * considered decision rather than an omission. The token is received as the
 * direct TLS response to a server-to-server POST to Google's token endpoint,
 * authenticated with the client secret. Google's own documentation states that
 * a token obtained this way can be trusted without local signature validation,
 * because TLS already establishes that Google produced it. Fetching JWKS on
 * every sign-in would add a network dependency, and a cache for it would add
 * state, in exchange for re-proving something TLS has already proven.
 *
 * The claims that do NOT follow from transport are still checked below: issuer,
 * audience, expiry and `email_verified`. Those are properties of the token's
 * contents, not of how it arrived.
 */

/** Google's endpoints. Pinned rather than discovered: they have been stable for years,
 *  and a discovery fetch is one more thing that can fail during a sign-in. */
const AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";

/** The only issuers Google uses. Both spellings are legitimate. */
const VALID_ISSUERS = new Set(["https://accounts.google.com", "accounts.google.com"]);

/** Matches the Streamlit deployment exactly. No extra scopes: nothing else is used. */
const SCOPES = "openid email profile";

const ENCODER = new TextEncoder();

function base64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Cryptographically random URL-safe string, for `state` and the PKCE verifier. */
export function randomToken(bytes = 32) {
  return base64url(crypto.getRandomValues(new Uint8Array(bytes)));
}

/** PKCE S256 challenge for a verifier. */
export async function codeChallenge(verifier) {
  const digest = await crypto.subtle.digest("SHA-256", ENCODER.encode(verifier));
  return base64url(new Uint8Array(digest));
}

/**
 * The URL to send the browser to.
 *
 * `prompt=select_account` rather than the default, so a visitor on a shared
 * machine is offered the account chooser instead of being silently signed in as
 * whoever used the browser last. For a site that stores medical images against
 * an identity, silently reusing a session is the wrong default.
 */
export function authorizationUrl({ clientId, redirectUri, state, challenge }) {
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: SCOPES,
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
    access_type: "online",
    prompt: "select_account",
  });
  return `${AUTH_ENDPOINT}?${params.toString()}`;
}

/** Exchange the authorization code for tokens. Throws with a readable message on failure. */
export async function exchangeCode({ clientId, clientSecret, redirectUri, code, verifier }) {
  const response = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
      code,
      code_verifier: verifier,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    // Google's error bodies are genuinely useful (`redirect_uri_mismatch` above
    // all, which is the single most common setup failure), so the description is
    // surfaced rather than swallowed into a generic failure.
    const detail = payload?.error_description || payload?.error || `HTTP ${response.status}`;
    throw new Error(`token exchange failed: ${detail}`);
  }
  if (!payload?.id_token) throw new Error("token exchange returned no id_token");
  return payload;
}

/** Decode a JWT payload without verifying it. See the module note on why that is safe here. */
function decodeJwtPayload(token) {
  const parts = String(token).split(".");
  if (parts.length !== 3) throw new Error("id_token is not a JWT");
  const padded = parts[1].replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0))));
}

/**
 * Validate the ID token's claims and return them.
 *
 * Throws rather than returning null, because every failure here is either a
 * misconfiguration or an attack, and both deserve to be logged with a reason.
 */
export function claimsFromIdToken(idToken, clientId) {
  const claims = decodeJwtPayload(idToken);

  if (!VALID_ISSUERS.has(claims.iss)) {
    throw new Error(`unexpected issuer ${claims.iss}`);
  }
  if (claims.aud !== clientId) {
    throw new Error("id_token audience does not match this client");
  }
  if (typeof claims.exp !== "number" || claims.exp < Math.floor(Date.now() / 1000)) {
    throw new Error("id_token has expired");
  }
  if (!claims.sub) {
    throw new Error("id_token carries no subject");
  }
  // Mirrors src/oauth.py:resolve_account. An unverified address must not be
  // allowed to claim an account, because anyone can put any address on an
  // unverified Google profile.
  if (claims.email_verified !== true) {
    throw new Error("google account email is not verified");
  }
  if (!claims.email) {
    throw new Error("id_token carries no email address");
  }
  return claims;
}

/**
 * Display-only profile fields, with the avatar URL allow-listed.
 *
 * Ported from `src/oauth.py:identity_profile`. The allow-list matters because
 * this URL is rendered into an `<img src>` on the public contributor roll: an
 * arbitrary URL there would let a display picture become a tracking pixel
 * pointed at every visitor to the landing page.
 */
export function identityProfile(claims) {
  const name = String(claims.name || "").trim().slice(0, 80);
  let picture = String(claims.picture || "").trim();
  try {
    const parsed = new URL(picture);
    const host = parsed.hostname.toLowerCase();
    const allowed =
      parsed.protocol === "https:" &&
      (host === "googleusercontent.com" || host.endsWith(".googleusercontent.com"));
    if (!allowed) picture = "";
  } catch {
    picture = "";
  }
  return {
    name,
    picture: picture.slice(0, 2048),
    subject: String(claims.sub),
    email: String(claims.email).trim().toLowerCase(),
  };
}
