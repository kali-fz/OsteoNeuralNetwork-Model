/**
 * The review console: the only path a shared radiograph has into training.
 *
 * WHAT THIS REPLACED
 * ------------------
 * A separate review tool run from a local checkout with the admin key exported
 * into the shell. That arrangement existed because the review path could not be
 * trusted to the public deployment: any guard inside it would have been a guard
 * running in a process strangers were talking to, and its real protection was
 * that the code simply was not deployed there.
 *
 * That reasoning does not carry over. The queue is not guarded here at all --
 * this file is in the bundle every visitor downloads, and reading it grants
 * nothing. `/api/admin/*` re-derives the acting account from the signed session
 * cookie on every single request, refuses anyone who is not the pinned owner,
 * and holds the admin key server-side where no page can reach it. Behind that,
 * `cloudflare/src/worker.js` checks the account again and a CHECK constraint in
 * D1 checks it a third time. Hiding this page would add a fourth check in the
 * one place an attacker controls, which is the least useful place to put one.
 *
 * WHAT IS DELIBERATELY NOT AUTOMATED
 * ----------------------------------
 * No "approve as-is". The label starts unselected on every card, because a
 * control already sitting on the model's own guess is a rubber stamp wearing a
 * disguise: approving it would feed the classifier its own output and teach it
 * nothing except more confidence in what it already believed.
 *
 * The bucket, by contrast, is not asked for. It follows from the label the
 * reviewer picked and what the gate did with the image, so asking would be
 * asking someone to re-type their own answer. It is shown, not chosen.
 */

import { getAdminQueue, reviewSubmission } from "../api.js";

/**
 * The three queues, in the order they are worked.
 *
 * Contradictions first. Those are the rows where the system disagreed with
 * itself and the image is still attached, which makes them the only rows that
 * carry information the model does not already have.
 */
const BUCKETS = [
  {
    id: "contradiction",
    title: "Disputed",
    blurb:
      "The system disagreed with itself: the gate turned away something the user says is a radiograph, or accepted something that is not one. Each of these is a demonstrated gate failure with the image still attached.",
  },
  {
    id: "valid_bone",
    title: "Bone radiographs",
    blurb:
      "The gate accepted these as radiographs. Assign the class you can defend from the image; these retrain the lesion classifier.",
  },
  {
    id: "misc",
    title: "Not radiographs",
    blurb:
      "The gate rejected these as non-radiographs. Confirming one keeps it as a negative example for the out-of-distribution detector, which today learns from no data at all and relies on hand-written thresholds.",
  },
];

/**
 * What a reviewer may say an image is.
 *
 * `misc` is a real training target, not a way of saying "discard this": it is
 * an honest negative for the gate. Throwing the row away is the Reject button,
 * and the two are kept visibly apart because they mean opposite things.
 */
const LABELS = [
  { id: "normal", title: "Normal", hint: "A bone radiograph with no lesion." },
  { id: "benign", title: "Benign", hint: "A lesion, not malignant." },
  { id: "malignant", title: "Malignant", hint: "A malignant lesion." },
  {
    id: "misc",
    title: "Not a radiograph",
    hint: "Not a bone X-ray at all. Kept as a negative example for the gate.",
  },
];

const MODEL_LABELS = { normal: "Normal", benign: "Benign", malignant: "Malignant" };

function escapeHtml(text) {
  return String(text ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function formatDate(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return String(iso);
  }
}

/**
 * Where a decision will file this row, for display only.
 *
 * A mirror of bucketFor() in worker/index.js, which is the one that decides.
 * This exists so the reviewer can see the consequence of a label before they
 * commit to it; it is never sent, and if the two ever disagreed the server's
 * answer is the one that would be written.
 */
function destinationFor(label, oodFlagged) {
  const gateSaysRadiograph = !oodFlagged;
  const reviewerSaysRadiograph = label !== "misc";
  if (gateSaysRadiograph !== reviewerSaysRadiograph) {
    return {
      title: "Disputed",
      why: gateSaysRadiograph
        ? "the gate accepted this and you say it is not a radiograph"
        : "the gate rejected this and you say it is a radiograph",
    };
  }
  return reviewerSaysRadiograph
    ? { title: "Bone radiographs", why: "the gate accepted it and you gave it a class" }
    : { title: "Not radiographs", why: "the gate rejected it and you agree" };
}

function renderCard(item) {
  const sid = String(item.submission_id);
  const short = escapeHtml(sid.slice(0, 8));
  const rejected = item.model_label === "rejected";
  const probability =
    typeof item.lesion_probability === "number"
      ? `${(item.lesion_probability * 100).toFixed(1)}%`
      : null;

  const verdict = rejected
    ? "<strong>The model did not run.</strong> The gate rejected this upload."
    : `Model said <strong>${escapeHtml(MODEL_LABELS[item.model_label] || item.model_label || "-")}</strong>${
        probability ? ` · P(lesion) ${probability}` : ""
      }`;

  const context = [
    item.triage_reason ? `Triaged here because ${escapeHtml(item.triage_reason)}.` : "",
    item.user_says_wrong ? "<strong>The user says this result is wrong.</strong>" : "",
    item.user_suggested_label
      ? `They suggested <em>${escapeHtml(item.user_suggested_label)}</em>.`
      : "",
    item.user_comment ? `“${escapeHtml(item.user_comment)}”` : "",
  ]
    .filter(Boolean)
    .map((line) => `<p class="onnm-review-context">${line}</p>`)
    .join("");

  const image = item.image_b64
    ? `<img class="onnm-review-image" alt="Submitted radiograph, as stored"
            src="data:image/png;base64,${escapeHtml(item.image_b64)}" />
       <p class="onnm-caption">As stored: 256 pixels, greyscale, metadata stripped.</p>`
    : `<div class="onnm-review-noimage">
         No image was kept for this row, so there is nothing here to train on.
         It can only be rejected.
       </div>`;

  const choices = LABELS.map(
    (label) => `
      <label class="onnm-review-choice">
        <input type="radio" name="label-${escapeHtml(sid)}" value="${label.id}" />
        <span>
          <strong>${label.title}</strong>
          <small>${label.hint}</small>
        </span>
      </label>`,
  ).join("");

  return `
    <article class="onnm-review-card" data-card="${escapeHtml(sid)}"
             data-ood="${item.ood_flagged ? "1" : "0"}"
             data-hasimage="${item.image_b64 ? "1" : "0"}">
      <header class="onnm-review-head">
        <span class="onnm-review-id">${short}</span>
        <span class="onnm-review-when">${escapeHtml(formatDate(item.created_at))}</span>
        ${item.user_says_wrong ? '<span class="onnm-review-flag">Disputed</span>' : ""}
      </header>

      <div class="onnm-review-body">
        <div class="onnm-review-figure">${image}</div>

        <div class="onnm-review-decision">
          <p class="onnm-review-verdict">${verdict}</p>
          ${context}
          <p class="onnm-caption">
            Treat anything the user said as context, not as evidence. Assign the
            label you can defend from the image itself.
          </p>

          <fieldset class="onnm-review-labels">
            <legend>Ground truth</legend>
            ${choices}
          </fieldset>

          <p class="onnm-review-destination" data-destination>
            Pick a label to see which set this joins.
          </p>

          <label class="onnm-review-note">
            <span>Reviewer note (optional)</span>
            <input type="text" data-note maxlength="2000"
                   placeholder="Anything worth recording about this decision" />
          </label>

          <div class="onnm-review-actions">
            <button class="onnm-btn onnm-btn-primary" type="button" data-approve disabled>
              Approve for training
            </button>
            <button class="onnm-btn" type="button" data-reject>Reject</button>
          </div>
          <p class="onnm-status" data-cardstatus role="status" aria-live="polite"></p>
        </div>
      </div>
    </article>`;
}

export async function renderAdmin(main) {
  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-panel">
      <h1>Review queue</h1>
      <p class="onnm-hero-lede">
        Shared radiographs waiting for a decision. Approving one records the
        ground truth and releases the row into the approved set that training
        batches are drawn from; rejecting one takes it out of consideration for
        good. Nothing a user does on their own can put a label in here.
      </p>
      <div class="onnm-review-counts" id="counts"></div>
      <div id="capacity"></div>
    </section>

    <section class="onnm-panel">
      <div class="onnm-review-tabs" role="tablist" aria-label="Review queues">
        ${BUCKETS.map(
          (bucket) => `
          <button class="onnm-review-tab" role="tab" type="button"
                  id="tab-${bucket.id}" data-bucket="${bucket.id}"
                  aria-selected="false" aria-controls="queue">
            ${bucket.title} <span data-tabcount>·</span>
          </button>`,
        ).join("")}
      </div>
      <p class="onnm-caption" id="blurb"></p>
      <div id="queue" role="tabpanel">
        <p class="onnm-muted">Loading…</p>
      </div>
    </section>`,
  );

  const counts = main.querySelector("#counts");
  const capacity = main.querySelector("#capacity");
  const blurb = main.querySelector("#blurb");
  const queue = main.querySelector("#queue");
  const tabs = [...main.querySelectorAll(".onnm-review-tab")];

  let active = BUCKETS[0].id;
  /** The last counts seen, adjusted in place as decisions are made. */
  let totals = null;

  function paintCounts() {
    if (!totals) return;
    const used = 100 * Number(totals.capacity_used || 0);
    const cells = [
      ["Submissions", totals.submissions ?? 0],
      ["Awaiting review", totals.pending_review ?? 0],
      ["Approved", totals.approved ?? 0],
      ["Storage used", `${used.toFixed(1)}%`],
    ];
    counts.innerHTML = cells
      .map(
        ([label, value]) => `
        <div class="onnm-review-count">
          <span class="onnm-review-count-value">${escapeHtml(value)}</span>
          <span class="onnm-review-count-label">${label}</span>
        </div>`,
      )
      .join("");

    const byBucket = totals.pending_by_bucket || {};
    for (const tab of tabs) {
      tab.querySelector("[data-tabcount]").textContent = byBucket[tab.dataset.bucket] ?? 0;
    }

    // Replaced rather than appended, or every refresh would stack another copy.
    capacity.innerHTML =
      used > 80
        ? `<p class="onnm-metric-warning">Community storage is over 80% of its cap; new shares will be refused before long.</p>`
        : "";
  }

  async function load(bucket) {
    active = bucket;
    for (const tab of tabs) {
      tab.setAttribute("aria-selected", tab.dataset.bucket === bucket ? "true" : "false");
    }
    queue.setAttribute("aria-labelledby", `tab-${bucket}`);
    blurb.textContent = BUCKETS.find((b) => b.id === bucket).blurb;
    queue.innerHTML = `<p class="onnm-muted">Loading…</p>`;

    let payload;
    try {
      payload = await getAdminQueue(bucket);
    } catch (error) {
      queue.innerHTML = `<div class="onnm-banner onnm-banner-error" role="alert">${escapeHtml(error.message)}</div>`;
      return;
    }
    // A slow request for a tab the reviewer has since left must not paint over
    // the tab they are now looking at.
    if (active !== bucket) return;

    if (payload.counts) {
      totals = payload.counts;
      paintCounts();
    }

    const rows = payload.pending || [];
    if (!rows.length) {
      queue.innerHTML = `<div class="onnm-empty-state">
        <strong>Nothing waiting here.</strong>
        <span>Every shared submission in this queue has been decided.</span>
      </div>`;
      return;
    }
    queue.innerHTML = rows.map(renderCard).join("");
  }

  /**
   * Decide one card, then retire it in place.
   *
   * The card stays on the page rather than being removed. Removing it would
   * pull everything below it upwards under the cursor, which is how the next
   * row gets a decision nobody read it for.
   */
  async function decide(card, decision) {
    const status = card.querySelector("[data-cardstatus]");
    const buttons = [...card.querySelectorAll("button")];
    const chosen = card.querySelector("input[type=radio]:checked");

    if (decision === "approved") {
      if (!chosen) {
        status.textContent = "Choose the ground-truth label before approving.";
        return;
      }
      if (card.dataset.hasimage !== "1") {
        status.textContent = "There is no image on this row, so there is nothing to train on.";
        return;
      }
    }

    for (const button of buttons) button.disabled = true;
    status.textContent = decision === "approved" ? "Approving…" : "Rejecting…";

    let result;
    try {
      result = await reviewSubmission(card.dataset.card, {
        decision,
        admin_label: chosen ? chosen.value : null,
        note: card.querySelector("[data-note]").value || null,
      });
    } catch (error) {
      for (const button of buttons) button.disabled = false;
      status.textContent = error.message;
      return;
    }

    card.classList.add("onnm-review-card-done");
    card.querySelector(".onnm-review-body").innerHTML = `
      <p class="onnm-review-outcome">
        ${
          result.review_status === "approved"
            ? `Approved as <strong>${escapeHtml(
                LABELS.find((l) => l.id === result.admin_label)?.title || result.admin_label,
              )}</strong>. It is in the approved set now, and the next training batch will draw it.`
            : "Rejected. This row will not be used."
        }
      </p>`;

    // The headline numbers are stale by one now. They are corrected here rather
    // than by refetching: the queue carries every image with it, so a refresh to
    // move one counter would pull a few hundred kilobytes back down the wire and
    // throw away the reviewer's place in the list to do it.
    if (totals) {
      totals.pending_review = Math.max(0, (totals.pending_review || 0) - 1);
      if (totals.pending_by_bucket) {
        totals.pending_by_bucket[active] = Math.max(
          0,
          (totals.pending_by_bucket[active] || 0) - 1,
        );
      }
      if (result.review_status === "approved") totals.approved = (totals.approved || 0) + 1;
      paintCounts();
    }
  }

  // One delegated listener for the whole queue, so cards can be replaced freely.
  queue.addEventListener("click", (event) => {
    const approve = event.target.closest("[data-approve]");
    const reject = event.target.closest("[data-reject]");
    if (!approve && !reject) return;
    const card = event.target.closest("[data-card]");
    if (card) decide(card, approve ? "approved" : "rejected");
  });

  queue.addEventListener("change", (event) => {
    const radio = event.target.closest("input[type=radio]");
    if (!radio) return;
    const card = radio.closest("[data-card]");
    const destination = destinationFor(radio.value, card.dataset.ood === "1");
    card.querySelector("[data-destination]").innerHTML =
      `Files into <strong>${escapeHtml(destination.title)}</strong>: ${escapeHtml(destination.why)}.`;
    card.querySelector("[data-approve]").disabled = card.dataset.hasimage !== "1";
  });

  for (const tab of tabs) {
    tab.addEventListener("click", () => load(tab.dataset.bucket));
  }

  await load(active);

  // Open on the work rather than on the first tab. Disputed rows are worth the
  // most and so come first, but landing on an empty queue while eleven rows sit
  // one tab over is a worse start than a tab that moved once.
  if (totals && !queue.querySelector("[data-card]")) {
    const next = BUCKETS.find((b) => (totals.pending_by_bucket || {})[b.id] > 0);
    if (next) await load(next.id);
  }

  return null;
}
