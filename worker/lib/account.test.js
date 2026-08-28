import assert from "node:assert/strict";
import test from "node:test";

import { resolveGoogleAccount } from "./account.js";
import { claimsFromIdToken } from "./google.js";

const PROFILE = {
  subject: "google-subject-123",
  email: "person@example.com",
  name: "Person Example",
  picture: "https://lh3.googleusercontent.com/avatar",
};

test("an existing Google subject reuses its user id without creating an account", async () => {
  const calls = [];
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async (subject) => {
      calls.push(["lookup", subject]);
      return { ok: true, status: 200, payload: { user_id: "existing-user" } };
    },
    createUser: async (body) => {
      calls.push(["create", body]);
      throw new Error("creation must not run for an existing subject");
    },
  });

  assert.deepEqual(result, { ok: true, userId: "existing-user", created: false });
  assert.deepEqual(calls, [["lookup", PROFILE.subject]]);
});

test("a new Google subject sends the complete legacy /users request shape", async () => {
  const generatedUserId = "9abdb6ff-a632-4d60-b0ac-36d2d214ced7";
  let createBody;
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async (subject) => {
      assert.equal(subject, PROFILE.subject);
      return { ok: false, status: 404, payload: { error: "no such user" } };
    },
    lookupByEmail: async (email) => {
      assert.equal(email, PROFILE.email);
      return { ok: false, status: 404, payload: { error: "no such user" } };
    },
    createUser: async (body) => {
      createBody = body;
      return { ok: true, status: 201, payload: { user_id: body.user_id } };
    },
    makeUserId: () => generatedUserId,
  });

  assert.deepEqual(createBody, {
    user_id: generatedUserId,
    email: PROFILE.email,
    auth_provider: "google",
    provider_subject: PROFILE.subject,
    display_name: PROFILE.name,
    profile_picture_url: PROFILE.picture,
  });
  assert.deepEqual(result, { ok: true, userId: generatedUserId, created: true });
});

test("new accounts receive an application-owned UUID by default", async () => {
  let generatedUserId;
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async () => ({ ok: false, status: 404, payload: {} }),
    lookupByEmail: async () => ({ ok: false, status: 404, payload: {} }),
    createUser: async (body) => {
      generatedUserId = body.user_id;
      return { ok: true, status: 201, payload: { user_id: body.user_id } };
    },
  });

  assert.match(
    generatedUserId,
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
  assert.equal(result.userId, generatedUserId);
});

test("a lookup failure other than not-found never becomes account creation", async () => {
  let created = false;
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async () => ({
      ok: false,
      status: 500,
      payload: { error: "database unavailable" },
    }),
    createUser: async () => {
      created = true;
    },
  });

  assert.equal(created, false);
  assert.deepEqual(result, {
    ok: false,
    status: 500,
    payload: { error: "database unavailable" },
  });
});

test("a mismatched account-service user id is rejected", async () => {
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async () => ({ ok: false, status: 404, payload: {} }),
    lookupByEmail: async () => ({ ok: false, status: 404, payload: {} }),
    createUser: async () => ({ ok: true, status: 201, payload: { user_id: "different-id" } }),
    makeUserId: () => "expected-id",
  });

  assert.equal(result.ok, false);
  assert.equal(result.status, 502);
});

test("an existing email is reused without rewriting its provider identity", async () => {
  const calls = [];
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async (subject) => {
      calls.push(["subject", subject]);
      return { ok: false, status: 404, payload: { error: "no such user" } };
    },
    lookupByEmail: async (email) => {
      calls.push(["email", email]);
      return {
        ok: true,
        status: 200,
        payload: {
          user_id: "password-era-user",
          auth_provider: "password",
          provider_subject: null,
        },
      };
    },
    createUser: async () => {
      throw new Error("creation must not run when the email already exists");
    },
  });

  assert.deepEqual(result, {
    ok: true,
    userId: "password-era-user",
    created: false,
  });
  assert.deepEqual(calls, [
    ["subject", PROFILE.subject],
    ["email", PROFILE.email],
  ]);
});

test("an email lookup failure other than not-found never becomes account creation", async () => {
  let created = false;
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async () => ({ ok: false, status: 404, payload: {} }),
    lookupByEmail: async () => ({
      ok: false,
      status: 503,
      payload: { error: "database unavailable" },
    }),
    createUser: async () => {
      created = true;
    },
  });

  assert.equal(created, false);
  assert.deepEqual(result, {
    ok: false,
    status: 503,
    payload: { error: "database unavailable" },
  });
});

test("a create conflict recovers only when the same subject now exists", async () => {
  let subjectLookups = 0;
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async (subject) => {
      assert.equal(subject, PROFILE.subject);
      subjectLookups += 1;
      return subjectLookups === 1
        ? { ok: false, status: 404, payload: {} }
        : { ok: true, status: 200, payload: { user_id: "race-winner" } };
    },
    lookupByEmail: async () => ({ ok: false, status: 404, payload: {} }),
    createUser: async () => ({
      ok: false,
      status: 409,
      payload: { error: "an account already exists for that email" },
    }),
    makeUserId: () => "race-loser",
  });

  assert.equal(subjectLookups, 2);
  assert.deepEqual(result, {
    ok: true,
    userId: "race-winner",
    created: false,
  });
});

test("a create conflict is rejected when the subject still does not exist", async () => {
  let subjectLookups = 0;
  const conflict = {
    ok: false,
    status: 409,
    payload: { error: "an account already exists for that email" },
  };
  const result = await resolveGoogleAccount(PROFILE, {
    lookupBySubject: async () => {
      subjectLookups += 1;
      return { ok: false, status: 404, payload: {} };
    },
    lookupByEmail: async () => ({ ok: false, status: 404, payload: {} }),
    createUser: async () => conflict,
    makeUserId: () => "unsafe-duplicate",
  });

  assert.equal(subjectLookups, 2);
  assert.deepEqual(result, {
    ok: false,
    status: 409,
    payload: conflict.payload,
  });
});

function idToken(claims) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(claims)}.`;
}

function googleClaims(emailVerified) {
  const claims = {
    iss: "https://accounts.google.com",
    aud: "client-id",
    exp: Math.floor(Date.now() / 1000) + 300,
    sub: PROFILE.subject,
    email: PROFILE.email,
  };
  if (emailVerified !== undefined) claims.email_verified = emailVerified;
  return claims;
}

test("Google claims require email_verified to be exactly true", () => {
  assert.equal(
    claimsFromIdToken(idToken(googleClaims(true)), "client-id").email_verified,
    true,
  );
  assert.throws(
    () => claimsFromIdToken(idToken(googleClaims(false)), "client-id"),
    /email is not verified/,
  );
  assert.throws(
    () => claimsFromIdToken(idToken(googleClaims(undefined)), "client-id"),
    /email is not verified/,
  );
  assert.throws(
    () => claimsFromIdToken(idToken(googleClaims("true")), "client-id"),
    /email is not verified/,
  );
});
