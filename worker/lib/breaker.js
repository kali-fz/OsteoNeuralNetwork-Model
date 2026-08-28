/**
 * The container-runtime circuit breaker.
 *
 * WHY THIS EXISTS
 * ---------------
 * Until this migration the Cloudflare account had no payment method, so overage
 * could not bill: the platform failed closed, and `cloudflare/wrangler.toml`
 * documents that as a deliberate safety property. Workers Paid removes it. A
 * card is now on file and Containers bill by wall-clock runtime, so a container
 * that stays awake is a container that spends money.
 *
 * The owner of this project has stated that the $5/month subscription is the
 * only expense available, ever. This module is what makes that a property of
 * the system rather than a hope.
 *
 * THE ARITHMETIC
 * --------------
 * Instance type `standard-1` is 1/2 vCPU, 4 GiB memory, 8 GB disk. At the
 * published rates that is:
 *
 *   memory  4    GiB x $0.0000025 /GiB-s  = $0.036 /hour
 *   cpu     0.5 vCPU x $0.000020 /vCPU-s  = $0.036 /hour
 *   disk    8     GB x $0.00000007 /GB-s  = $0.002 /hour
 *                                          ---------------
 *                                           $0.074 /hour
 *
 * The budget below is 5 hours per calendar month, so the worst case this module
 * permits is about $0.37 on top of the $5 subscription. Cloudflare's docs also
 * describe an included allowance (25 GiB-hours, which for this instance is 6.25
 * hours) that would make it $0.00, but the budget deliberately does not depend
 * on that allowance existing.
 *
 * WHY IT METERS RUNTIME AND NOT REQUESTS
 * --------------------------------------
 * Requests are free; time is not. A container is billed from the moment it
 * starts until it sleeps, whether it is working or idling out its 90-second
 * timer. Counting scans would therefore measure the wrong thing: ten scans in
 * one sitting cost roughly the same as one, and one scan every two minutes for
 * an hour costs twelve times as much as ten scans back to back.
 *
 * FAIL-CLOSED
 * -----------
 * Every failure path here refuses rather than allows. If the counter cannot be
 * read, the scanner is disabled. An unreadable meter is indistinguishable from
 * an exhausted one, and only one of those two guesses can cost money.
 */

/** Five hours per month, in seconds. See the arithmetic above. */
export const MONTHLY_BUDGET_SECONDS = 5 * 60 * 60;

/** Warn the user (and the admin) from here on, so exhaustion is never a surprise. */
export const WARN_FRACTION = 0.8;

/** Refuse new scans at this point, leaving headroom for scans already in flight. */
export const STOP_FRACTION = 1.0;

/** `meta` keys are per calendar month so the budget resets without a cron job. */
export function monthKey(now = new Date()) {
  return `container_seconds_${now.getUTCFullYear()}_${String(now.getUTCMonth() + 1).padStart(2, "0")}`;
}

/**
 * Accumulated container runtime this month, in seconds.
 *
 * Throws rather than returning 0 when the row cannot be read. A missing row is
 * legitimately zero and returns 0; a database error is not, and must not be
 * mistaken for an unused budget.
 */
export async function usedSeconds(db, now = new Date()) {
  const row = await db
    .prepare("SELECT value FROM meta WHERE key = ?")
    .bind(monthKey(now))
    .first();
  if (!row) return 0;
  const value = Number.parseFloat(row.value);
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

/**
 * Add elapsed runtime to this month's total.
 *
 * Uses the same read-modify-write-in-SQL idiom as the `bytes_stored` counter in
 * `worker.js`, so the increment is atomic within D1 rather than a racy
 * read-then-write from two concurrent container lifecycles.
 */
export async function addSeconds(db, seconds, now = new Date()) {
  if (!Number.isFinite(seconds) || seconds <= 0) return;
  const key = monthKey(now);
  await db.batch([
    db.prepare("INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')").bind(key),
    db
      .prepare(
        "UPDATE meta SET value = CAST(CAST(value AS REAL) + ? AS TEXT) WHERE key = ?",
      )
      .bind(seconds, key),
  ]);
}

/**
 * Whether a scan may start, and the numbers behind that answer.
 *
 * The shape is deliberately reportable: the frontend shows the remaining budget
 * rather than only discovering it is gone at the moment a scan is refused. A
 * capacity limit a user can see coming is a constraint; one that appears without
 * warning is a bug, as far as they are concerned.
 */
export async function budgetStatus(db, now = new Date()) {
  let used;
  try {
    used = await usedSeconds(db, now);
  } catch (error) {
    // Fail closed. See the module note.
    return {
      ok: false,
      allowed: false,
      reason: "meter_unavailable",
      message: "Scanning is paused because the capacity meter could not be read.",
      usedSeconds: null,
      budgetSeconds: MONTHLY_BUDGET_SECONDS,
      fraction: null,
      detail: String(error?.message || error),
    };
  }

  const fraction = used / MONTHLY_BUDGET_SECONDS;
  const allowed = fraction < STOP_FRACTION;
  return {
    ok: true,
    allowed,
    warn: fraction >= WARN_FRACTION,
    reason: allowed ? null : "budget_exhausted",
    message: allowed
      ? null
      : "Monthly scanning capacity has been reached. Scanning resumes at the start of next month.",
    usedSeconds: Math.round(used),
    budgetSeconds: MONTHLY_BUDGET_SECONDS,
    remainingSeconds: Math.max(0, Math.round(MONTHLY_BUDGET_SECONDS - used)),
    fraction: Number(fraction.toFixed(4)),
  };
}
