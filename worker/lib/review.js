/**
 * The two decisions the review console makes on the server's side.
 *
 * Kept out of worker/index.js so they can be tested directly. Both are pure
 * functions of their arguments, and both answer a question the browser must
 * never be allowed to answer for itself.
 */

import { ADMIN_EMAIL, ADMIN_USER_ID } from "../../cloudflare/src/worker.js";
import { timingSafeEqual } from "./session.js";

/**
 * Is the browser on the other end of this request the owning account?
 *
 * Answered from the signed session cookie and the two pinned constants, and
 * from nothing else. There is deliberately no `is_admin` claim inside the
 * cookie -- see the note at the top of lib/session.js -- because a boolean in a
 * token would be a third place that has to agree with the Worker and the D1
 * CHECK constraint, and the only one an attacker could ever influence.
 * Deriving it per request from the id is one fact, checked in one place.
 *
 * Both the id and the address must match. They cannot disagree today, since
 * `users` carries a CHECK pinning the flag to this id and the OAuth callback
 * writes the address from Google's verified claim; requiring both means that a
 * future change breaking that link fails closed rather than open.
 */
export function isAdminSession(session) {
  if (!session) return false;
  return (
    timingSafeEqual(String(session.uid || ""), ADMIN_USER_ID) &&
    timingSafeEqual(String(session.email || "").toLowerCase(), ADMIN_EMAIL)
  );
}

/**
 * Which of the three queues a decision files a row into.
 *
 * The buckets record what a row is *for*, and that is a consequence of two
 * facts rather than a third judgement: what the gate did with the image, and
 * what the reviewer says the image actually is. When those agree the row is an
 * ordinary example of whichever kind; when they disagree the row is a
 * demonstrated gate failure, which is exactly what 'contradiction' means.
 *
 * This is why the browser sends a label and never a bucket. The Streamlit
 * console asked for both, so that a tired reviewer could not wave through the
 * model's own guess -- but that protection lives in the *label*, which is still
 * never preselected. Making someone re-type a value that follows mechanically
 * from their own answer bought nothing and cost a click on every row.
 *
 * Every pair this can return satisfies the agreement rules that
 * reviewSubmission() enforces in cloudflare/src/worker.js and that the
 * bucket_and_label_must_agree trigger enforces behind it. review.test.js pins
 * that, because the two must not be allowed to drift apart.
 */
export function bucketFor(label, oodFlagged) {
  const gateSaysRadiograph = !oodFlagged;
  const reviewerSaysRadiograph = label !== "misc";
  if (gateSaysRadiograph !== reviewerSaysRadiograph) return "contradiction";
  return reviewerSaysRadiograph ? "valid_bone" : "misc";
}
