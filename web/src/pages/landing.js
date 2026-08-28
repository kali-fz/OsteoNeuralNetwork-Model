/**
 * Public landing page.
 *
 * The composition and product copy deliberately follow app.py's final
 * Streamlit homepage. The runtime remains the standalone Vite application:
 * every number, contributor and country marker still comes from a same-origin
 * Worker API, and the globe keeps its existing canvas implementation.
 */

import heroMoss from "../../../assets/hero-moss.svg?url";
import globeMarkup from "../globe/markup.html?raw";
import world from "../globe/countries-110m.json";
import { mountGlobe } from "../globe/globe.js";
import { getContributors, getGlobe, getStats } from "../api.js";

const MODEL_VERSION = "0.1.0";

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
  if (value === null || value === undefined || value === "") return "—";
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
}

function statTile(value, label) {
  return `
    <div class="onnm-home-stat">
      <span class="onnm-home-stat-value">${formatCount(value)}</span>
      <span class="onnm-home-stat-label">${label}</span>
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
    : `<a class="onnm-home-cta onnm-btn onnm-btn-primary" href="/api/auth/google/start">Create a free account with Google</a>`;

  main.insertAdjacentHTML(
    "beforeend",
    `
    <div class="onnm-home-grid">
      <div class="onnm-home-mission">
        <section class="onnm-home-hero" style="--hero-img: url('${heroMoss}')">
          <p class="onnm-hero-eyebrow">Open bone X-ray research · ONNM v${MODEL_VERSION}</p>
          <h1 class="onnm-hero-title">Test the current model and help us train the next one.</h1>
          <p class="onnm-hero-subtitle">
            ONNM v${MODEL_VERSION} is our first trained model for bone X-rays. Create a
            free account to test it with DICOM files or standard image formats, then
            choose whether each radiograph can be reviewed for future training.
          </p>
          <div class="onnm-hero-points" aria-label="Account and contribution highlights">
            <span>Free Google account</span>
            <span>Optional, human-reviewed contributions</span>
          </div>
        </section>
        ${cta}
        <p class="onnm-home-tool-note">Research tool only. This is not a medical device or medical advice.</p>
      </div>

      <section class="onnm-home-community" aria-labelledby="community-heading">
        <div class="onnm-globe-heading">
          <span>Community reach</span>
          <strong id="community-heading">Country-level activity</strong>
        </div>
        <div class="onnm-globe-host" id="globe-host">${globeMarkup}</div>
        <p class="onnm-globe-note" id="globe-note" aria-live="polite">Loading community activity…</p>
      </section>
    </div>

    <section class="onnm-home-stats" id="stats" aria-label="Project totals">
      ${statTile(null, "Registered users")}
      ${statTile(null, "Reviewed shared scans")}
      ${statTile(null, "Countries represented")}
    </section>

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
        Every result must be reviewed by a qualified clinician. This prototype has no
        FDA, CE, or MHRA clearance and must never direct patient care.
      </p>
    </section>

    <section class="onnm-terms" id="terms" aria-labelledby="terms-heading">
      <div class="onnm-section-heading">
        <span>Terms of use</span>
        <strong id="terms-heading">What happens to an image you upload</strong>
      </div>

      <div class="onnm-terms-notice">
        <h3>You can withdraw a shared image — until it is approved</h3>
        <p>
          Sharing is optional and is asked separately for every image. While a
          shared image is still waiting to be reviewed, you can delete it at any
          time from <a href="/profile" data-link>your account page</a>, and it
          leaves the review queue immediately.
        </p>
        <p>
          <strong>That window closes when a reviewer approves the image.</strong>
          Approval adds it to the research training set, and once a model has
          been trained on an image there is no way to remove what the model
          learned from it. We cannot take it back out afterwards, and we will not
          claim otherwise. Approval is therefore the last point at which the
          decision is still yours — so if you are unsure, delete it before then.
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
          <strong>A shared image is stored as a 256-pixel processed copy</strong>
          — never your original file, and never its metadata. DICOM tags are
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
      : `<p class="onnm-muted">Approved contributors can choose to show their Google name and photo here from My Profile. No account is published automatically.</p>`;
  } else {
    contributorsHost.innerHTML = `<p class="onnm-muted">Contributor profiles are temporarily unavailable. Google profiles remain private by default.</p>`;
  }

  const note = main.querySelector("#globe-note");
  const payload = globe.status === "fulfilled"
    ? globe.value
    : { markers: [], unplaced: { signups: 0, contributors: 0 } };
  const unplaced =
    (payload.unplaced?.signups || 0) + (payload.unplaced?.contributors || 0);
  if (globe.status !== "fulfilled") {
    note.textContent =
      "Community activity could not be loaded. The globe is still available for orientation, but it contains no live markers.";
  } else if (!payload.markers?.length) {
    note.textContent =
      "No displayable country has been recorded yet. Signed-in activity is recorded at country level without sharing a precise location.";
  } else if (unplaced) {
    note.textContent = `${formatCount(unplaced)} ${unplaced === 1 ? "account is" : "accounts are"} included in the totals but cannot be drawn. Markers show aggregated countries, never precise locations.`;
  } else {
    note.textContent = "Drag to rotate. Markers show aggregated countries, never precise locations.";
  }

  return mountGlobe(main.querySelector("#globe-host"), {
    markers: payload.markers || [],
    world,
    autoRotate: true,
    height: 410,
  });
}
