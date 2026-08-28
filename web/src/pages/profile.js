/**
 * The account page: who you are signed in as, and what you have scanned.
 *
 * The history comes from /api/submissions, which derives the account from the
 * session cookie. There is no user id in the request and none would be honoured
 * if there were -- see the note at the top of worker/index.js on why that route
 * is not a pass-through to the storage Worker.
 */

import { getSubmissions, withdrawSubmission } from "../api.js";

function escapeHtml(text) {
  return String(text ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

const REVIEW_TEXT = {
  pending: "Awaiting review",
  approved: "Approved for training",
  rejected: "Not used",
};

/**
 * Whether this row's image can still be withdrawn.
 *
 * Two conditions, and both are about what is actually possible rather than
 * about policy. There has to be a stored image to delete -- an unshared scan
 * never kept one -- and it has to be unapproved, because approval is the point
 * at which the picture enters a training corpus and, after a run, weights that
 * cannot be unlearned. The server enforces both again; this only decides
 * whether to draw the button.
 */
function canWithdraw(row) {
  return Boolean(row.shared) && row.review_status !== "approved";
}

export async function renderProfile(main, state) {
  const user = state.session?.user || {};

  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-panel">
      <h1>Your account</h1>
      <div class="onnm-account">
        ${
          user.picture
            ? `<img class="onnm-avatar-lg" src="${escapeHtml(user.picture)}" alt="" width="56" height="56" referrerpolicy="no-referrer" />`
            : ""
        }
        <div>
          <div class="onnm-profile-account-name">${escapeHtml(user.name || "Signed in")}</div>
          <div class="onnm-muted">${escapeHtml(user.email || "")}</div>
        </div>
      </div>
      <p class="onnm-caption">
        ONNM never receives your Google password. Your email address and Google
        account identifier are used to link saved scans to this account, and
        your name and photo are not shown publicly unless you choose to appear
        as a contributor.
      </p>
    </section>

    <section class="onnm-panel">
      <h2>Your scans</h2>
      <div id="history"><p class="onnm-muted">Loading…</p></div>
    </section>`,
  );

  const host = main.querySelector("#history");

  let payload;
  try {
    payload = await getSubmissions();
  } catch (error) {
    host.innerHTML = `<div class="onnm-banner onnm-banner-error" role="alert">${escapeHtml(error.message)}</div>`;
    return null;
  }

  // listUserSubmissions in cloudflare/src/worker.js returns { submissions: [...] }.
  const rows = payload?.submissions || [];
  if (!rows.length) {
    host.innerHTML = `<p class="onnm-muted">
      You have not scanned anything yet. Scans you run will be listed here.
    </p>`;
    return null;
  }

  host.innerHTML = `
    <table class="onnm-table">
      <thead>
        <tr>
          <th scope="col">When</th>
          <th scope="col">Result</th>
          <th scope="col">Lesion probability</th>
          <th scope="col">Shared</th>
          <th scope="col">Review</th>
          <th scope="col">Your image</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
          <tr>
            <td>${escapeHtml(formatDate(row.created_at))}</td>
            <td>${escapeHtml(row.model_label || "—")}</td>
            <td>${
              typeof row.lesion_probability === "number"
                ? (row.lesion_probability * 100).toFixed(1) + "%"
                : "—"
            }</td>
            <td>${row.shared ? "Yes" : "No"}</td>
            <td>${escapeHtml(REVIEW_TEXT[row.review_status] || "—")}</td>
            <td class="onnm-history-action">${
              canWithdraw(row)
                ? `<button class="onnm-btn onnm-btn-quiet" type="button"
                           data-withdraw="${escapeHtml(row.submission_id)}">Delete image</button>`
                : row.shared
                  ? `<span class="onnm-muted onnm-history-locked">In the training set</span>`
                  : `<span class="onnm-muted">—</span>`
            }</td>
          </tr>`,
          )
          .join("")}
      </tbody>
    </table>
    <p class="onnm-caption">
      "Approved for training" means a human reviewer confirmed the image and
      assigned a label. Your own feedback on a result is recorded as a signal
      for review and never becomes a training label on its own.
    </p>
    <p class="onnm-caption">
      <strong>Deleting a shared image</strong> removes the stored copy and takes
      it out of the review queue, and you can do that at any point up until a
      reviewer approves it. After approval the image has been added to the
      training set, and once a model has trained on it there is no way to remove
      it from what that model learned — so approval is the point of no return,
      not a policy we could choose to relax. Deleting the image does not remove
      the row below; your own record of the scan and its result stays.
    </p>`;

  // One delegated listener: rows are re-rendered, individual handlers are not.
  host.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-withdraw]");
    if (!button) return;

    const cell = button.closest(".onnm-history-action");
    button.disabled = true;
    button.textContent = "Deleting…";
    try {
      await withdrawSubmission(button.dataset.withdraw);
      // The row stays; only its shared state changes. Rewriting the two cells
      // in place avoids refetching the whole history and losing the reader's
      // place in a long list.
      const row = button.closest("tr");
      row.children[3].textContent = "No";
      cell.innerHTML = `<span class="onnm-muted">Deleted</span>`;
    } catch (error) {
      button.disabled = false;
      button.textContent = "Delete image";
      cell.insertAdjacentHTML(
        "beforeend",
        `<span class="onnm-history-error">${escapeHtml(error.message)}</span>`,
      );
    }
  });

  return null;
}
