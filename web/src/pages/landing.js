/**
 * The landing page: what ONNM is, who has contributed, and where from.
 *
 * The globe is the centrepiece and the thing most easily got wrong. It draws
 * country centroids and nothing finer -- see worker/lib/geo.js and the note in
 * globe/globe.js on what a marker is allowed to contain.
 */

import globeMarkup from "../globe/markup.html?raw";
import world from "../globe/countries-110m.json";
import { mountGlobe } from "../globe/globe.js";
import { getGlobe, getStats } from "../api.js";

function statTile(value, label, hint) {
  return `
    <div class="onnm-stat">
      <div class="onnm-stat-value">${value}</div>
      <div class="onnm-stat-label">${label}</div>
      ${hint ? `<div class="onnm-stat-hint">${hint}</div>` : ""}
    </div>`;
}

export async function renderLanding(main) {
  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-hero">
      <p class="onnm-eyebrow">Research demonstration</p>
      <h1 class="onnm-hero-title">
        A second look at a bone radiograph, and a map of what it looked at.
      </h1>
      <p class="onnm-hero-lede">
        ONNM screens a radiograph for a possible bone lesion, tells you how
        confident it is, and draws a heat map showing which region drove that
        answer. It is a research model, not a diagnosis, and it says so on every
        result it produces.
      </p>
    </section>

    <section class="onnm-stats" id="stats" aria-label="Project totals"></section>

    <section class="onnm-panel onnm-globe-panel">
      <div class="onnm-globe-copy">
        <h2>Where contributors are</h2>
        <p class="onnm-muted">
          Each dot is a country, not a person. Location is resolved to a
          two-letter country code at Cloudflare's edge and nothing finer is ever
          stored, so this map cannot be zoomed into an individual. No IP address
          is recorded and the browser's location API is never called.
        </p>
        <p class="onnm-globe-note" id="globe-note"></p>
      </div>
      <div class="onnm-globe-host" id="globe-host">${globeMarkup}</div>
    </section>

    <section class="onnm-panel">
      <h2>What you get back</h2>
      <ol class="onnm-steps">
        <li><strong>A check that it is a radiograph at all.</strong> Anything
          else is refused before the model sees it, with the reason shown.</li>
        <li><strong>A calibrated probability</strong>, not a raw score, cut at a
          threshold fitted on validation data.</li>
        <li><strong>A Grad-CAM overlay</strong> showing the region that drove
          the answer, so a wrong answer is visibly wrong.</li>
        <li><strong>An honest refusal</strong> when the model is too uncertain
          to call it, rather than a confident guess.</li>
      </ol>
    </section>`,
  );

  // Stats and globe data load in parallel; neither blocks the other, and a
  // failure in one must not blank the other.
  const [stats, globe] = await Promise.allSettled([getStats(), getGlobe()]);

  const statsHost = main.querySelector("#stats");
  if (stats.status === "fulfilled") {
    const s = stats.value;
    statsHost.innerHTML = [
      statTile(s.users ?? "—", "Registered users"),
      statTile(s.submissions ?? "—", "Scans run"),
      statTile(s.approved ?? "—", "Approved for training", "after human review"),
    ].join("");
  } else {
    statsHost.innerHTML = `<p class="onnm-muted">Totals are unavailable right now.</p>`;
  }

  if (globe.status !== "fulfilled") {
    main.querySelector("#globe-note").textContent =
      "The contributor map is unavailable right now.";
    return null;
  }

  const payload = globe.value;
  const host = main.querySelector("#globe-host");

  // Counts that are deliberately not on the map are stated rather than hidden,
  // so the picture never quietly disagrees with the totals beside it.
  const unplaced =
    (payload.unplaced?.signups || 0) + (payload.unplaced?.contributors || 0);
  const note = main.querySelector("#globe-note");
  note.textContent = payload.markers.length
    ? unplaced
      ? `${payload.totals?.countries_represented ?? payload.markers.length} countries shown. ${unplaced} ${unplaced === 1 ? "account is" : "accounts are"} in a country this map cannot place, and are counted but not drawn.`
      : `${payload.totals?.countries_represented ?? payload.markers.length} ${payload.totals?.countries_represented === 1 ? "country" : "countries"} represented.`
    : "No locations have been recorded yet.";

  return mountGlobe(host, {
    markers: payload.markers,
    world,
    autoRotate: true,
    height: 460,
  });
}
