/**
 * Public landing page.
 *
 * Every number, contributor and country marker comes from a same-origin Worker
 * API, and the globe keeps its canvas implementation rather than re-drawing in
 * the DOM, because it animates.
 */

import heroMoss from "../../../assets/hero-moss.svg?url";
import globeMarkup from "../globe/markup.html?raw";
import world from "../globe/countries-110m.json";
import { mountGlobe } from "../globe/globe.js";
import { getContributors, getGlobe, getStats } from "../api.js";

/**
 * The public release version, shown in the eyebrow, the lede and the call to action.
 *
 * This is the *release* line, not the model ledger. The two are deliberately
 * separate: `model_versions.json` tracks trained checkpoints and currently serves
 * `v1.0.0`, promoted only by `scripts/version_model.py` when no guarded metric has
 * regressed. This number moves when what a visitor gets changes, which includes
 * changes that touch no weights at all.
 *
 * 0.1.1 is the first such move: the model is unchanged, and the release is the
 * Terms gate, the Privacy notice and the governance work behind them.
 */
const MODEL_VERSION = "0.1.1";

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
}

function safeImageUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" ? escapeHtml(url.href) : "";
  } catch {
    return "";
  }
}

function formatCount(value) {
  if (value === null || value === undefined || value === "") return "-";
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "-";
}

function statTile(value, label) {
  return `
    <div class="onnm-globe-stat">
      <span class="onnm-globe-stat-value">${formatCount(value)}</span>
      <span class="onnm-globe-stat-label">${label}</span>
    </div>`;
}

function contributorCard(contributor) {
  const name = String(contributor?.name || "Contributor");
  const safeName = escapeHtml(name);
  const count = Math.max(0, Number(contributor?.approved_contributions) || 0);
  const picture = safeImageUrl(contributor?.picture);
  const initials = escapeHtml(
    name
      .split(/\s+/)
      .slice(0, 2)
      .map((part) => part[0] || "")
      .join("")
      .toUpperCase() || "C",
  );
  const avatar = picture
    ? `<img class="onnm-contributor-avatar" src="${picture}" alt="" referrerpolicy="no-referrer" />`
    : `<span class="onnm-contributor-avatar onnm-account-initials" aria-hidden="true">${initials}</span>`;
  const noun = count === 1 ? "contribution" : "contributions";

  return `
    <article class="onnm-contributor-card">
      ${avatar}
      <span><strong>${safeName}</strong><small>${formatCount(count)} approved ${noun}</small></span>
    </article>`;
}

export async function renderLanding(main, state) {
  const signedIn = Boolean(state.session?.signed_in);
  const cta = signedIn
    ? `<a class="onnm-home-cta onnm-btn onnm-btn-primary" href="/scanner" data-link>Test ONNM v${MODEL_VERSION}</a>`
    : `<a class="onnm-home-cta onnm-btn onnm-btn-cream" href="/terms" data-link>Create a free account with Google</a>`;

  main.insertAdjacentHTML(
    "beforeend",
    `
    <div class="onnm-home-grid">
      <div class="onnm-home-mission">
        <section class="onnm-home-hero" style="--hero-img: url('${heroMoss}')">
          <p class="onnm-hero-eyebrow">Open bone X-ray research · ONNM v${MODEL_VERSION}</p>
          <h1 class="onnm-hero-title">Test the current model and help us train the next one.</h1>
          <p class="onnm-hero-subtitle">
            ONNM v${MODEL_VERSION} is a trained model that screens bone X-rays for
            possible lesions, and creating an account to use it is free. It's
            community-driven: every image someone chooses to share helps train the next
            version, and what we're working towards next is telling apart the five main
            types of bone cancer, and more!
          </p>
          <div class="onnm-hero-points" aria-label="Account and contribution highlights">
            <span>Free Google account</span>
            <span>Optional, human-reviewed contributions</span>
          </div>
        </section>
        ${cta}
        <p class="onnm-home-tool-note">This is a research tool only, not a medical device, and its output is not medical advice.</p>
      </div>

      <section class="onnm-home-community" aria-labelledby="community-heading">
        <div class="onnm-globe-heading">
          <span>Community reach</span>
          <strong id="community-heading">Country-level activity</strong>
        </div>
        <div class="onnm-globe-host" id="globe-host">${globeMarkup}</div>
        <p class="onnm-globe-note" id="globe-note" aria-live="polite">Loading community activity…</p>
        <div class="onnm-globe-stats" id="stats" aria-label="Project totals">
          ${statTile(null, "Registered users")}
          ${statTile(null, "Reviewed shared scans")}
          ${statTile(null, "Countries represented")}
        </div>
      </section>
    </div>

    <section class="onnm-metric-band" aria-labelledby="scan-benefits-heading">
      <h2 id="scan-benefits-heading">What you receive after a scan</h2>
      <div class="onnm-metric-grid">
        <div class="onnm-metric-item">
          <span class="onnm-metric-num">3-way</span>
          <span class="onnm-metric-ci">Normal · benign · malignant</span>
          <div class="onnm-metric-desc">Calibrated probabilities for each model class.</div>
        </div>
        <div class="onnm-metric-item">
          <span class="onnm-metric-num">Grad-CAM</span>
          <span class="onnm-metric-ci">Visual explanation</span>
          <div class="onnm-metric-desc">A heatmap shows which part of the image influenced the result.</div>
        </div>
        <div class="onnm-metric-item">
          <span class="onnm-metric-num">Private</span>
          <span class="onnm-metric-ci">Consent before sharing</span>
          <div class="onnm-metric-desc">You decide whether each processed image can be reviewed for research.</div>
        </div>
      </div>
      <p class="onnm-metric-warning">
        Every result must be reviewed by a qualified clinician, because this prototype
        has no FDA, CE or MHRA clearance and must never direct patient care.
      </p>
    </section>

    <section class="onnm-terms" id="terms" aria-labelledby="terms-heading">
      <div class="onnm-section-heading">
        <span>Terms of use</span>
        <strong id="terms-heading">What happens to an image you upload</strong>
      </div>

      <div class="onnm-terms-notice">
        <h3>You can withdraw a shared image until it is approved</h3>
        <p>
          Sharing is optional and is asked separately for every image. While a
          shared image is still waiting to be reviewed, you can delete it at any
          time from <a href="/profile" data-link>your account page</a>, and it
          leaves the review queue immediately.
        </p>
        <p>
          <strong>That self-service window closes when a reviewer approves the
          image</strong>, but your rights do not. After approval you can still ask
          us and we will delete the stored copy. What cannot be undone is the
          training itself: once a model has learned from an image, no technique
          removes that image's contribution from the weights, and we will not
          claim otherwise. Approval is the last point at which the decision is
          entirely yours, so if you are unsure, delete it before then.
        </p>
      </div>

      <ol class="onnm-terms-list">
        <li>
          <strong>Research and education only.</strong> ONNM is not a medical
          device, has no FDA, CE or MHRA clearance, and has not been clinically
          validated. Never use its output to make, confirm or delay a clinical
          decision.
        </li>
        <li>
          <strong>Only upload images you are entitled to use.</strong> Do not
          upload an identifiable patient radiograph. If an image shows a name, a
          hospital number or any burned-in identifier, it does not belong here.
        </li>
        <li>
          <strong>Nothing is shared unless you tick the box.</strong> Sharing is
          off by default and is asked per image, because agreeing once says
          nothing about the next file you open. An unshared scan is analysed and
          the image is never written down.
        </li>
        <li>
          <strong>A shared image is stored as a 256-pixel processed copy.</strong>
          Never your original file, and never its metadata. DICOM tags are
          stripped before anything is stored.
        </li>
        <li>
          <strong>Only three parties ever meet a shared image:</strong> you, the
          single reviewer account that assigns its label, and the model it is
          used to train. It is not published, sold or passed to anyone else.
        </li>
        <li>
          <strong>Refused images are deleted automatically.</strong> If a
          reviewer rejects an upload, its stored copy is erased within seven
          days.
        </li>
        <li>
          <strong>Location is recorded at country level only.</strong> No IP
          address is stored, and the map shows aggregated countries rather than
          places.
        </li>
      </ol>
    </section>

    <section class="onnm-home-contributors" aria-labelledby="contributors-heading">
      <div class="onnm-section-heading">
        <span>Contributors</span>
        <strong id="contributors-heading">People who shared approved research images</strong>
      </div>
      <div id="contributors" aria-live="polite">
        <p class="onnm-muted">Loading approved contributors…</p>
      </div>
    </section>`,
  );

  // Each public data surface fails independently. A problem with contributor
  // cards must never remove the globe, and a map outage must not hide totals.
  const [stats, globe, contributors] = await Promise.allSettled([
    getStats(),
    getGlobe(),
    getContributors(),
  ]);

  const statsHost = main.querySelector("#stats");
  const globeTotals = globe.status === "fulfilled" ? globe.value?.totals || {} : {};
  if (stats.status === "fulfilled") {
    statsHost.innerHTML = [
      statTile(stats.value?.users, "Registered users"),
      statTile(stats.value?.approved, "Reviewed shared scans"),
      statTile(globeTotals.countries_represented, "Countries represented"),
    ].join("");
  } else {
    statsHost.innerHTML = `<p class="onnm-muted">Project totals are temporarily unavailable.</p>`;
  }

  const contributorsHost = main.querySelector("#contributors");
  if (contributors.status === "fulfilled") {
    const people = contributors.value?.contributors || [];
    contributorsHost.innerHTML = people.length
      ? `<div class="onnm-contributor-grid">${people.map(contributorCard).join("")}</div>`
      : `<p class="onnm-muted">Approved contributors can choose to show their Google name and photo here from My Profile, and no account appears here on its own.</p>`;
  } else {
    contributorsHost.innerHTML = `<p class="onnm-muted">Contributor profiles are temporarily unavailable, but Google profiles stay private by default either way.</p>`;
  }

  const note = main.querySelector("#globe-note");
  const payload = globe.status === "fulfilled"
    ? globe.value
    : { markers: [], unplaced: { signups: 0, contributors: 0 } };
  const unplaced =
    (payload.unplaced?.signups || 0) + (payload.unplaced?.contributors || 0);
  if (globe.status !== "fulfilled") {
    note.textContent =
      "Community activity could not be loaded, though the globe is still here for orientation and currently shows no live markers.";
  } else if (!payload.markers?.length) {
    note.textContent =
      "No displayable country has been recorded yet, since signed-in activity is only ever recorded at country level rather than as a precise location.";
  } else if (unplaced) {
    note.textContent = `${formatCount(unplaced)} ${unplaced === 1 ? "account is" : "accounts are"} included in the totals but cannot be drawn, since markers only ever show aggregated countries, never precise locations.`;
  } else {
    note.textContent = "Drag to rotate, and note the markers show aggregated countries, never precise locations.";
  }

  return mountGlobe(main.querySelector("#globe-host"), {
    markers: payload.markers || [],
    world,
    autoRotate: true,
    height: 410,
  });
}
