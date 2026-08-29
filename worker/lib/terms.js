/**
 * The Terms gate, server side.
 *
 * The browser is not trusted with any of this. `web/src/pages/terms.js` draws a
 * checkbox, but a checkbox is a drawing; what actually stops an account being
 * created is that `authStart` refuses to talk to Google without a signed cookie
 * that only this module mints, and what actually stops a scan is a database read.
 */

import { signSession, verifySession } from "./session.js";

/**
 * The version recorded against every acceptance.
 *
 * A date rather than a counter, so a stored value names the text without needing
 * a changelog to decode it. It must match `TERMS_VERSION` in
 * `web/src/pages/terms.js`; the two are asserted equal in terms.test.js, because
 * a silent drift would record agreement to a wording nobody was shown.
 *
 * The server records THIS value, never the one the client posted. A stale cached
 * bundle can send an old string, and honouring it would let a client choose what
 * it is deemed to have agreed to.
 */
export const TERMS_VERSION = "2026-08-30";

/**
 * The oldest acceptance still worth anything.
 *
 * The Terms promise, in the "Changes" clause, that a *material* change requires
 * agreeing again. Enforcing that by comparing against TERMS_VERSION would be too
 * blunt: fixing a typo would bump the version and re-prompt everybody, which
 * trains people to click through the thing we most want them to read.
 *
 * So materiality is stated explicitly. An acceptance counts only if it names this
 * version or a later one, and moving this date is a deliberate decision that the
 * text changed enough to need fresh agreement. A wording fix bumps TERMS_VERSION
 * alone and nobody is disturbed.
 *
 * Set to 2026-08-30 because that revision added a minimum age, an indemnity, a
 * prohibited-use section and a governing law clause. Consent to the previous text
 * is not consent to those.
 */
export const TERMS_MATERIAL_SINCE = "2026-08-30";

/**
 * A short-lived cookie proving the Terms were accepted before any account exists.
 *
 * A signed-out visitor has no row to write to, so the acceptance has to survive
 * the round trip to Google and back in the only place available: a cookie the
 * client cannot forge. This is the same mechanism the OAuth `state` and PKCE
 * verifier already use, for the same reason.
 *
 * `__Host-` forces Secure, Path=/ and no Domain. SameSite=Lax matches the session
 * cookie, and matters here too: the callback that consumes this is a top-level
 * cross-site navigation back from Google, which Strict would drop.
 */
export const TERMS_COOKIE = "__Host-onnm_terms";

/** Fifteen minutes: long enough to sign in, short enough not to linger. */
export const TERMS_TTL_SECONDS = 15 * 60;

export function termsCookieHeader(token) {
  return `${TERMS_COOKIE}=${token}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${TERMS_TTL_SECONDS}`;
}

export function clearTermsCookieHeader() {
  return `${TERMS_COOKIE}=; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=0`;
}

/** Mint the pre-account acceptance token. */
export function signTermsToken(secret, version = TERMS_VERSION) {
  return signSession({ terms: version }, secret, TERMS_TTL_SECONDS);
}

/**
 * The version a valid token attests to, or null.
 *
 * Null for every failure -- absent, malformed, expired, wrong signature -- because
 * a caller can do nothing useful with the difference and reporting it would tell
 * a forger which half of the attempt was wrong.
 */
export async function acceptedVersionFromToken(token, secret) {
  if (!secret) return null;
  const payload = await verifySession(token, secret);
  const version = payload?.terms;
  return typeof version === "string" && version.length > 0 && version.length <= 64
    ? version
    : null;
}

/**
 * Has this account agreed to the Terms as they currently stand?
 *
 * A row with no `tos_version` has agreed to nothing recorded. That is every
 * account created before this gate existed, and treating it as accepted would
 * defeat the point of adding the column.
 *
 * A row whose version predates TERMS_MATERIAL_SINCE agreed to text that has since
 * changed materially, so it is treated as not accepted and the account is asked
 * again. This is what makes the "Changes" clause in the Terms true rather than
 * aspirational: the document promises re-agreement on a material change, and this
 * is the line that delivers it.
 *
 * The comparison is a plain string compare, which is correct because versions are
 * ISO dates and ISO dates sort lexicographically. That is the whole reason the
 * version is a date and not a counter.
 */
export function hasAcceptedTerms(userRow, materialSince = TERMS_MATERIAL_SINCE) {
  const version = userRow?.tos_version;
  if (typeof version !== "string" || version.length === 0) return false;
  return version >= materialSince;
}
