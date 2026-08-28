/**
 * The browser's view of the API.
 *
 * Every call is same-origin and relies on the `__Host-onnm_session` cookie,
 * which is httpOnly and therefore invisible to this file. That is the point:
 * there is no token here to steal, and no code path in the frontend can read,
 * copy or leak the session.
 *
 * There is also no API key here. The Streamlit deployment held one because the
 * server was the client; the browser is the client now, so the key stays in the
 * Worker and this file simply cannot address the storage routes directly.
 */

/** Throw a useful message for a failed response, or return its JSON. */
async function unwrap(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const error = new Error(payload?.error || `request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function get(path) {
  return unwrap(await fetch(path, { credentials: "same-origin" }));
}

/** Who is signed in, plus the scanning budget. Never throws for "nobody". */
export const getSession = () => get("/api/session");

/** Country-level markers for the globe. Public. */
export const getGlobe = () => get("/api/globe");

/** Headline counts for the landing page. Public. */
export const getStats = () => get("/api/stats");

/** This account's scan history. The user id comes from the cookie, not from here. */
export const getSubmissions = () => get("/api/submissions");

export async function signOut() {
  return unwrap(
    await fetch("/api/auth/signout", { method: "POST", credentials: "same-origin" }),
  );
}

/**
 * Wake the inference container.
 *
 * Called when a file is chosen rather than when Analyse is pressed, so the
 * container's cold start happens while the visitor is still looking at their
 * own file picker. Failure is not surfaced: this is an optimisation, and a scan
 * that follows will start the container anyway.
 */
export async function warmup() {
  try {
    return await unwrap(
      await fetch("/api/warmup", { method: "POST", credentials: "same-origin" }),
    );
  } catch {
    return null;
  }
}

/**
 * Run one scan.
 *
 * `shareConsent` is passed per upload and defaults to false. The server decides
 * what to store; this flag only ever grants, never assumes.
 */
export async function scan(file, { threshold, camClass = "auto", shareConsent = false } = {}) {
  const body = new FormData();
  body.set("file", file, file.name);
  if (threshold !== undefined && threshold !== null) body.set("threshold", String(threshold));
  body.set("cam_class", camClass);
  body.set("share_consent", shareConsent ? "true" : "false");

  return unwrap(
    await fetch("/api/scan", { method: "POST", body, credentials: "same-origin" }),
  );
}

/**
 * Re-cut a verdict at a new threshold, in the browser.
 *
 * A faithful port of `InferenceResult.with_threshold` in
 * src/onnm/inference.py. A threshold is a cut on an already-computed
 * probability: it cannot change what the network produced, so re-running the
 * model would return bit-identical numbers at the cost of a container wake.
 * Under a metered runtime budget that is not merely wasteful, it is billable.
 *
 * The uncertainty gate is recomputed rather than reused, because `inconclusive`
 * is `is_lesion and defer` -- a False does not tell you which half was False.
 */
export function withThreshold(result, threshold, { confidenceFloor, entropyGate }) {
  const probs = result.class_probabilities || {};
  const values = Object.values(probs);
  const maxProbability = values.length ? Math.max(...values) : 0;

  // Normalised predictive entropy, matching onnm.ood.predictive_entropy.
  const n = values.length || 1;
  let entropy = 0;
  for (const p of values) if (p > 0) entropy -= p * Math.log(p);
  const normalised = n > 1 ? entropy / Math.log(n) : 0;

  const defer =
    (confidenceFloor != null && maxProbability < confidenceFloor) ||
    (entropyGate != null && normalised >= entropyGate);

  const isLesion = result.lesion_probability >= threshold;
  const inconclusive = Boolean(isLesion && defer);

  const label = inconclusive
    ? "Non-Diagnostic / Inconclusive"
    : isLesion
      ? "Potential Bone Lesion"
      : "Normal";

  return {
    label,
    isLesion,
    inconclusive,
    confidence:
      100 * (isLesion ? result.lesion_probability : 1 - result.lesion_probability),
    threshold,
    maxProbability,
    predictiveEntropy: normalised,
  };
}
