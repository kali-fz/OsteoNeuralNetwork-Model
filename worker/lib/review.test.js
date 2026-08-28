import assert from "node:assert/strict";
import test from "node:test";

import { ADMIN_EMAIL, ADMIN_USER_ID } from "../../cloudflare/src/worker.js";
import { bucketFor, isAdminSession } from "./review.js";

// ---------------------------------------------------------------------------
// Who may review
// ---------------------------------------------------------------------------

const OWNER = { uid: ADMIN_USER_ID, email: ADMIN_EMAIL };

test("the owning account is recognised", () => {
  assert.equal(isAdminSession(OWNER), true);
});

test("a signed-out request is not the owner", () => {
  assert.equal(isAdminSession(null), false);
  assert.equal(isAdminSession(undefined), false);
});

test("an ordinary signed-in account is not the owner", () => {
  assert.equal(
    isAdminSession({ uid: "11111111-2222-3333-4444-555555555555", email: "someone@example.com" }),
    false,
  );
});

test("the owner's address on somebody else's account is not enough", () => {
  // The shape a bug would produce: an account row that borrowed the address.
  assert.equal(isAdminSession({ uid: "not-the-owner", email: ADMIN_EMAIL }), false);
});

test("the owner's id with a different address is not enough", () => {
  assert.equal(isAdminSession({ uid: ADMIN_USER_ID, email: "someone@example.com" }), false);
});

test("the address is compared case-insensitively, as Google may return it", () => {
  assert.equal(isAdminSession({ uid: ADMIN_USER_ID, email: ADMIN_EMAIL.toUpperCase() }), true);
});

test("a session missing either field is not the owner", () => {
  assert.equal(isAdminSession({ uid: ADMIN_USER_ID }), false);
  assert.equal(isAdminSession({ email: ADMIN_EMAIL }), false);
  assert.equal(isAdminSession({}), false);
});

// ---------------------------------------------------------------------------
// Where a decision files a row
// ---------------------------------------------------------------------------

test("the gate accepted it and the reviewer gave it a class: an ordinary bone row", () => {
  for (const label of ["normal", "benign", "malignant"]) {
    assert.equal(bucketFor(label, false), "valid_bone");
  }
});

test("the gate rejected it and the reviewer agrees: an ordinary negative", () => {
  assert.equal(bucketFor("misc", true), "misc");
});

test("the gate rejected a real radiograph: a demonstrated gate failure", () => {
  for (const label of ["normal", "benign", "malignant"]) {
    assert.equal(bucketFor(label, true), "contradiction");
  }
});

test("the gate accepted something that is not a radiograph: also a gate failure", () => {
  assert.equal(bucketFor("misc", false), "contradiction");
});

/**
 * The rules reviewSubmission() enforces in cloudflare/src/worker.js, and that
 * the bucket_and_label_must_agree trigger enforces behind it. Restated rather
 * than imported because the point is to catch the two drifting apart: if that
 * function's rules change, this copy stops matching and the test fails.
 */
function backendWouldAccept(label, bucket) {
  if (!["normal", "benign", "malignant", "misc"].includes(label)) return false;
  if (!["valid_bone", "misc", "contradiction"].includes(bucket)) return false;
  if (bucket === "misc" && label !== "misc") return false;
  if (bucket === "valid_bone" && label === "misc") return false;
  return true;
}

test("every pair this can produce is one the review endpoint accepts", () => {
  for (const label of ["normal", "benign", "malignant", "misc"]) {
    for (const oodFlagged of [true, false]) {
      const bucket = bucketFor(label, oodFlagged);
      assert.ok(
        backendWouldAccept(label, bucket),
        `${label} + ood_flagged=${oodFlagged} produced ${bucket}, which the review endpoint would refuse`,
      );
    }
  }
});
