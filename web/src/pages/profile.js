/**
 * The account page: who you are signed in as, and what you have scanned.
 *
 * The history comes from /api/submissions, which derives the account from the
 * session cookie. There is no user id in the request and none would be honoured
 * if there were -- see the note at the top of worker/index.js on why that route
 * is not a pass-through to the storage Worker.
 */

import { getSubmissions } from "../api.js";

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
          <div class="onnm-account-name">${escapeHtml(user.name || "Signed in")}</div>
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
          </tr>`,
          )
          .join("")}
      </tbody>
    </table>
    <p class="onnm-caption">
      "Approved for training" means a human reviewer confirmed the image and
      assigned a label. Your own feedback on a result is recorded as a signal
      for review and never becomes a training label on its own.
    </p>`;

  return null;
}
