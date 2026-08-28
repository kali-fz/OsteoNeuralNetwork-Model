/**
 * The inference container, and the meter that keeps it affordable.
 *
 * WHY A DURABLE OBJECT AT ALL
 * ---------------------------
 * Cloudflare Containers are addressed through a Durable Object: the DO is the
 * thing that starts the container, holds the connection to it, and decides when
 * it sleeps. That is also why this cannot live in a Pages project -- Pages
 * cannot define Durable Object classes -- and therefore why the whole site is a
 * single Worker with static assets rather than Pages plus Functions.
 *
 * THE METER
 * ---------
 * `onStart` and `onStop` bracket every period the container is awake, and the
 * elapsed time is added to a monthly total in D1. `lib/breaker.js` reads that
 * total and refuses new scans once the budget is spent.
 *
 * This is the only place that knows how long the container actually ran. A
 * per-scan estimate would be wrong in both directions: ten scans in one sitting
 * share a single wake period, while one scan every two minutes for an hour keeps
 * it awake the whole time. Billing follows wall-clock, so the meter must too.
 *
 * The accounting is deliberately conservative. If the process is killed without
 * a clean `onStop`, the next `onStart` finds the stale marker and bills the gap
 * before resetting it. Over-counting costs the user some capacity;
 * under-counting costs them money.
 */

import { Container, getContainer } from "@cloudflare/containers";

import { addSeconds } from "./lib/breaker.js";

/** DO storage key holding the timestamp of the current wake period. */
const STARTED_AT = "container_started_at";

/**
 * An upper bound on any single wake period, in seconds.
 *
 * Used only when reconciling a stale marker left by an unclean shutdown. Without
 * it, a marker orphaned for a week would bill a week of runtime and permanently
 * exhaust the budget. `sleepAfter` is 90 seconds, so a genuine period cannot
 * plausibly exceed a few minutes; ten is generous and still bounded.
 */
const MAX_PLAUSIBLE_PERIOD_SECONDS = 10 * 60;

export class InferenceContainer extends Container {
  /** Matches EXPOSE and the uvicorn bind in inference/Dockerfile. */
  defaultPort = 8080;

  /**
   * Ninety seconds.
   *
   * Long enough that a visitor scanning several films in a row does not pay a
   * cold start between each, short enough that an abandoned session stops
   * costing money almost immediately. Every second here is billed at roughly
   * $0.0000206, so the default of "10m" would cost about seven times more per
   * visit for no benefit a user would notice.
   */
  sleepAfter = "90s";

  /**
   * The container makes no outbound requests, so it is not permitted any.
   *
   * The model and its calibration are baked into the image, and
   * `RadiographClassifier` forces `pretrained` off, so nothing needs to be
   * downloaded at runtime. Turning the network off makes that a structural
   * guarantee rather than an assumption: a process that cannot open a socket
   * cannot exfiltrate a radiograph, whatever a future dependency decides to do.
   */
  enableInternet = false;

  /**
   * Secrets reach the container as environment variables at start.
   *
   * `Container` creates an own `envVars` field during `super()`. A prototype
   * getter here would therefore be shadowed and never called, silently leaving
   * the container with an empty environment. Set that inherited field only
   * after the Durable Object receives its Worker environment instead.
   */
  constructor(ctx, env) {
    super(ctx, env);
    this.envVars = {
      INFERENCE_KEY: env.INFERENCE_KEY || "",
      ONNM_CHECKPOINT: env.ONNM_CHECKPOINT || "/opt/onnm/best.pt",
    };
  }

  async onStart() {
    const now = Date.now();

    // Reconcile an unclean previous shutdown before starting a new period.
    const stale = await this.ctx.storage.get(STARTED_AT);
    if (typeof stale === "number" && stale > 0) {
      const orphaned = Math.min((now - stale) / 1000, MAX_PLAUSIBLE_PERIOD_SECONDS);
      console.warn(
        `container: stale start marker found; billing ${orphaned.toFixed(1)}s from an unclean stop`,
      );
      await this.#bill(orphaned);
    }

    await this.ctx.storage.put(STARTED_AT, now);
    console.log("container: started");
  }

  async onStop({ exitCode, reason }) {
    const startedAt = await this.ctx.storage.get(STARTED_AT);
    await this.ctx.storage.delete(STARTED_AT);

    if (typeof startedAt !== "number" || startedAt <= 0) {
      console.warn("container: stopped with no start marker; runtime not metered");
      return;
    }

    const elapsed = (Date.now() - startedAt) / 1000;
    await this.#bill(elapsed);
    console.log(
      `container: stopped after ${elapsed.toFixed(1)}s (reason=${reason}, exit=${exitCode})`,
    );
  }

  onError(error) {
    // Logged rather than swallowed, then re-thrown so the caller sees a failed
    // scan instead of a hung request. Observability is enabled on this Worker,
    // so this is the line that will explain a container that will not start.
    console.error("container: error", error);
    throw error;
  }

  /**
   * Write elapsed runtime to the monthly meter.
   *
   * Never allowed to throw. A metering failure must not turn into a failed
   * scan for the user -- but it is logged loudly, because a silently broken
   * meter is the one failure that costs money.
   */
  async #bill(seconds) {
    try {
      if (!this.env.DB) {
        console.error("container: no DB binding; runtime NOT metered");
        return;
      }
      await addSeconds(this.env.DB, seconds);
    } catch (error) {
      console.error("container: failed to record runtime", error);
    }
  }
}

/**
 * The single container instance.
 *
 * A fixed name rather than `getRandom`, because `max_instances` is 1 and the
 * point is that exactly one container can ever be awake. Routing to a pool
 * would multiply the runtime bill by the pool size for a project whose entire
 * budget is $5 a month.
 */
export function inferenceStub(env) {
  return getContainer(env.ONNM_INFERENCE, "singleton");
}
