/**
 * Tests for the Terms gate.
 *
 * The gate's whole value is that it cannot be talked around from the browser, so
 * what is tested here is refusal: a forged token, an expired one, one signed with
 * the wrong secret, and an account row that never agreed.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  acceptedVersionFromToken,
  clearTermsCookieHeader,
  hasAcceptedTerms,
  signTermsToken,
  TERMS_COOKIE,
  TERMS_TTL_SECONDS,
  TERMS_VERSION,
  termsCookieHeader,
} from "./terms.js";
import { signSession } from "./session.js";

const SECRET = "a-test-secret-that-is-long-enough";

// ---------------------------------------------------------------------------
// The two halves must agree
// ---------------------------------------------------------------------------

const termsPage = readFileSync(
  fileURLToPath(new URL("../../web/src/pages/terms.js", import.meta.url)),
  "utf8",
);

test("the page and the server record the same Terms version", () => {
  // If these drift, every acceptance is recorded against a version string that
  // does not describe the text the person actually read. Nothing else in the
  // system would notice, which is exactly why this is asserted.
  const match = termsPage.match(/export const TERMS_VERSION = "([^"]+)"/);
  assert.ok(match, "web/src/pages/terms.js must export TERMS_VERSION");
  assert.equal(match[1], TERMS_VERSION);
});

test("the operator is named before the Terms can ship", () => {
  // A contract needs a named counterparty and Article 13(1)(a) needs a named
  // controller. This test is what stops a placeholder reaching production: it is
  // not a reminder, it is a build failure.
  //
  // The assertion reads the value out of the OPERATOR object rather than
  // scanning the whole file. Scanning matched the marker wherever it appeared,
  // including inside the code that checked for it, so the test failed even once
  // the name was filled in. Assert on the datum, not on the text around it.
  const match = termsPage.match(/export const OPERATOR = \{\s*name: "([^"]*)"/);
  assert.ok(match, "web/src/pages/terms.js must export OPERATOR with a name");

  const name = match[1].trim();
  assert.ok(name.length > 0, "OPERATOR.name is empty");
  assert.ok(
    !name.includes("TODO"),
    `OPERATOR.name is still a placeholder (${name}). The Terms and the Privacy ` +
      "notice both name the operator, and neither can be published until the " +
      "controller's legal name and capacity are confirmed.",
  );
});

test("the Privacy notice and the Terms move together", () => {
  // They cross-reference each other and the Terms require agreement to both, so
  // a Privacy notice older than the Terms means people agreed to a notice that
  // no longer describes what happens. Dates, so this is a string comparison.
  const privacyPage = readFileSync(
    fileURLToPath(new URL("../../web/src/pages/privacy.js", import.meta.url)),
    "utf8",
  );
  const match = privacyPage.match(/export const PRIVACY_VERSION = "([^"]+)"/);
  assert.ok(match, "web/src/pages/privacy.js must export PRIVACY_VERSION");
  assert.ok(
    match[1] >= TERMS_VERSION,
    `the Privacy notice (${match[1]}) predates the Terms (${TERMS_VERSION})`,
  );
});

// ---------------------------------------------------------------------------
// Who counts as having accepted
// ---------------------------------------------------------------------------

test("an account with a recorded version has accepted", () => {
  assert.equal(hasAcceptedTerms({ tos_version: TERMS_VERSION }), true);
});

test("an account predating the gate has not accepted", () => {
  // The shape of every row created before this work: a timestamp copied from
  // created_at, and no version, because there was no text to agree to.
  assert.equal(
    hasAcceptedTerms({ tos_accepted_at: "2026-08-01T00:00:00Z", tos_version: null }),
    false,
  );
});

test("missing, empty and non-string versions are all refusals", () => {
  for (const row of [{}, null, undefined, { tos_version: "" }, { tos_version: 1 }]) {
    assert.equal(hasAcceptedTerms(row), false, `should refuse ${JSON.stringify(row)}`);
  }
});

test("an older recorded version still counts as accepted", () => {
  // The owner chose not to re-prompt when the text changes. Storing the version
  // keeps that option open; it must not silently start enforcing it.
  assert.equal(hasAcceptedTerms({ tos_version: "2020-01-01" }), true);
});

// ---------------------------------------------------------------------------
// The pre-account token
// ---------------------------------------------------------------------------

test("a freshly minted token attests to the current version", async () => {
  const token = await signTermsToken(SECRET);
  assert.equal(await acceptedVersionFromToken(token, SECRET), TERMS_VERSION);
});

test("no token means no acceptance", async () => {
  for (const value of [null, undefined, "", "not-a-token"]) {
    assert.equal(await acceptedVersionFromToken(value, SECRET), null);
  }
});

test("a token signed with a different secret is refused", async () => {
  const token = await signTermsToken("some-other-secret-entirely");
  assert.equal(await acceptedVersionFromToken(token, SECRET), null);
});

test("a tampered token is refused", async () => {
  const token = await signTermsToken(SECRET);
  const [payload, signature] = token.split(".");
  // Same signature, different payload: the forgery a signed cookie exists to stop.
  const forged = `${payload.slice(0, -2)}XX.${signature}`;
  assert.equal(await acceptedVersionFromToken(forged, SECRET), null);
});

test("an expired token is refused", async () => {
  const expired = await signSession({ terms: TERMS_VERSION }, SECRET, -1);
  assert.equal(await acceptedVersionFromToken(expired, SECRET), null);
});

test("a validly signed token carrying no version is refused", async () => {
  // A session cookie is signed with the same secret. It must not be usable here.
  const sessionShaped = await signSession({ uid: "someone" }, SECRET, 3600);
  assert.equal(await acceptedVersionFromToken(sessionShaped, SECRET), null);
});

test("no secret means no acceptance, rather than an exception", async () => {
  const token = await signTermsToken(SECRET);
  assert.equal(await acceptedVersionFromToken(token, ""), null);
});

// ---------------------------------------------------------------------------
// Cookie attributes
// ---------------------------------------------------------------------------

test("the cookie is host-scoped, httpOnly and short-lived", () => {
  const header = termsCookieHeader("token-value");
  assert.ok(header.startsWith(`${TERMS_COOKIE}=token-value`));
  for (const attribute of ["HttpOnly", "Secure", "SameSite=Lax", "Path=/"]) {
    assert.ok(header.includes(attribute), `missing ${attribute}`);
  }
  assert.ok(header.includes(`Max-Age=${TERMS_TTL_SECONDS}`));
  assert.ok(TERMS_TTL_SECONDS <= 3600, "the acceptance token should not linger");
});

test("clearing uses the same attributes, or the browser will not match it", () => {
  const cleared = clearTermsCookieHeader();
  for (const attribute of ["HttpOnly", "Secure", "SameSite=Lax", "Path=/", "Max-Age=0"]) {
    assert.ok(cleared.includes(attribute), `missing ${attribute}`);
  }
});
