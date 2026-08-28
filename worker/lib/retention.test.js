/**
 * Tests for the two paths that DELETE a stored radiograph.
 *
 * Both are exercised through `handleApiRequest`, not by calling the functions
 * directly, so the route, its authentication and its guards are covered too --
 * a purge reachable without the admin key, or a withdrawal that accepted
 * somebody else's submission id, would be a far worse bug than a wrong SQL
 * predicate and neither is visible from a unit call.
 *
 * The D1 mock answers by SQL shape rather than executing SQL. What is worth
 * defending here is the control flow: which rows are chosen, that the image is
 * cleared rather than the row dropped, that the byte counter is put back in
 * step, and who is allowed to ask. SQLite's own behaviour is not under test.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { handleApiRequest, purgeRejectedImages } from "../../cloudflare/src/worker.js";

const ADMIN_USER_ID = "c2c5a209-4aaa-4eb9-b112-b2929b6dbe12";

/** A D1 stand-in that records every statement and answers from fixtures. */
function fakeDb({ rows = [], first = null } = {}) {
  const executed = [];
  const db = {
    executed,
    prepare(sql) {
      const statement = { sql, params: [] };
      const api = {
        bind(...params) {
          statement.params = params;
          return api;
        },
        async all() {
          executed.push(statement);
          return { results: typeof rows === "function" ? rows(sql) : rows };
        },
        async first() {
          executed.push(statement);
          return typeof first === "function" ? first(sql) : first;
        },
        async run() {
          executed.push(statement);
          return { success: true };
        },
      };
      return api;
    },
    batch: async (statements) => statements,
  };
  return db;
}

const sqlOf = (db) => db.executed.map((s) => s.sql.replace(/\s+/g, " ").trim());
const anyMatching = (db, pattern) => sqlOf(db).some((sql) => pattern.test(sql));

// ---------------------------------------------------------------------------
// The scheduled purge
// ---------------------------------------------------------------------------

test("the purge clears the image and never deletes the row", async () => {
  const db = fakeDb({ rows: [{ submission_id: "a" }, { submission_id: "b" }] });
  const result = await purgeRejectedImages(db);

  assert.equal(result.purged, 2);
  assert.ok(anyMatching(db, /UPDATE submissions SET image_b64 = NULL/));
  assert.ok(
    !anyMatching(db, /DELETE FROM submissions/),
    "the row is the record that a human refused this upload and must survive",
  );
});

test("only rejected rows are eligible", async () => {
  const db = fakeDb({ rows: [{ submission_id: "a" }] });
  await purgeRejectedImages(db);

  for (const sql of sqlOf(db).filter((s) => s.includes("submissions"))) {
    if (/SELECT submission_id|UPDATE submissions SET image_b64/.test(sql)) {
      assert.match(sql, /review_status = 'rejected'/);
    }
  }
});

test("the cutoff is the requested number of days back", async () => {
  const db = fakeDb({ rows: [{ submission_id: "a" }] });
  const before = Date.now();
  const { cutoff } = await purgeRejectedImages(db, 7);

  const age = before - Date.parse(cutoff);
  const day = 86400 * 1000;
  // Seven days back, give or take the time the call itself took.
  assert.ok(Math.abs(age - 7 * day) < 60_000, `cutoff was ${cutoff}`);
});

test("a pending or approved image is never touched by age alone", async () => {
  // Nothing matches the predicate, which is what a table of pending and
  // approved rows looks like to this query.
  const db = fakeDb({ rows: [] });
  const result = await purgeRejectedImages(db);

  assert.equal(result.purged, 0);
  assert.ok(!anyMatching(db, /UPDATE submissions SET image_b64 = NULL/));
});

test("a run with nothing to do writes nothing at all", async () => {
  const db = fakeDb({ rows: [] });
  await purgeRejectedImages(db);
  assert.ok(
    !anyMatching(db, /UPDATE meta/),
    "the byte counter should not be rewritten when no image was removed",
  );
});

test("removing images puts the storage meter back in step", async () => {
  const db = fakeDb({ rows: [{ submission_id: "a" }] });
  await purgeRejectedImages(db);

  const recompute = sqlOf(db).find((sql) => sql.includes("UPDATE meta"));
  assert.ok(recompute, "bytes_stored must be corrected after a purge");
  assert.match(recompute, /SUM\(LENGTH\(image_b64\)\)/);
  assert.ok(
    !/value AS INTEGER\) -/.test(recompute),
    "recompute from the table rather than decrementing, so drift self-heals",
  );
});

// ---------------------------------------------------------------------------
// The purge route
// ---------------------------------------------------------------------------

const call = (path, { method = "POST", key, actor, body } = {}) => {
  const headers = new Headers();
  if (key) headers.set("authorization", `Bearer ${key}`);
  if (actor) headers.set("x-onnm-admin-user", actor);
  if (body !== undefined) headers.set("content-type", "application/json");
  return new Request(`https://example.test${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
};

const ENV = (db) => ({ DB: db, API_KEY: "app-key", ADMIN_KEY: "admin-key" });

test("the purge route refuses the app key", async () => {
  const db = fakeDb({ rows: [] });
  const response = await handleApiRequest(
    call("/admin/purge", { key: "app-key", actor: ADMIN_USER_ID }),
    ENV(db),
  );
  assert.equal(response.status, 403);
});

test("the purge route refuses the admin key without the owning account", async () => {
  const db = fakeDb({ rows: [] });
  const response = await handleApiRequest(
    call("/admin/purge", { key: "admin-key", actor: "somebody-else" }),
    ENV(db),
  );
  assert.equal(response.status, 403);
});

test("the purge route runs for the admin key and the owning account", async () => {
  const db = fakeDb({ rows: [{ submission_id: "a" }] });
  const response = await handleApiRequest(
    call("/admin/purge", { key: "admin-key", actor: ADMIN_USER_ID }),
    ENV(db),
  );
  assert.equal(response.status, 200);
  assert.equal((await response.json()).purged, 1);
});

// ---------------------------------------------------------------------------
// Withdrawal
// ---------------------------------------------------------------------------

const OWNER = "user-1";

function withdrawalDb(row) {
  return fakeDb({ first: row });
}

test("a user may withdraw their own pending image", async () => {
  const db = withdrawalDb({
    user_id: OWNER,
    review_status: "pending",
    shared: 1,
    image_b64: "png",
  });
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    submission_id: "s1",
    withdrawn: true,
    already: false,
  });
  assert.ok(anyMatching(db, /SET image_b64 = NULL, shared = 0/));
  assert.ok(anyMatching(db, /UPDATE meta/), "the storage meter must be corrected");
});

test("an approved image can no longer be withdrawn", async () => {
  const db = withdrawalDb({
    user_id: OWNER,
    review_status: "approved",
    shared: 1,
    image_b64: "png",
  });
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );

  assert.equal(response.status, 409);
  assert.match((await response.json()).error, /already been approved/);
  assert.ok(
    !anyMatching(db, /SET image_b64 = NULL/),
    "nothing may be cleared once the image is in the training set",
  );
});

test("somebody else's submission is a 404, not a 403", async () => {
  const db = withdrawalDb({
    user_id: "a-different-user",
    review_status: "pending",
    shared: 1,
    image_b64: "png",
  });
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );

  // 403 would confirm the id exists, which is enough to enumerate submissions.
  assert.equal(response.status, 404);
  assert.ok(!anyMatching(db, /SET image_b64 = NULL/));
});

test("a rejected image can still be withdrawn before the purge reaches it", async () => {
  const db = withdrawalDb({
    user_id: OWNER,
    review_status: "rejected",
    shared: 1,
    image_b64: "png",
  });
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );
  assert.equal(response.status, 200);
});

test("withdrawing twice succeeds without touching anything", async () => {
  const db = withdrawalDb({
    user_id: OWNER,
    review_status: "pending",
    shared: 0,
    image_b64: null,
  });
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );

  assert.equal(response.status, 200);
  assert.equal((await response.json()).already, true);
  assert.ok(!anyMatching(db, /SET image_b64 = NULL/));
});

test("a missing submission is a 404", async () => {
  const db = withdrawalDb(null);
  const response = await handleApiRequest(
    call("/submissions/s1/withdraw", { key: "app-key", body: { user_id: OWNER } }),
    ENV(db),
  );
  assert.equal(response.status, 404);
});
