/**
 * The scanner.
 *
 * TWO THINGS HERE ARE NOT COSMETIC
 * --------------------------------
 * 1. The threshold slider re-cuts the verdict in the browser, never by asking
 *    the model again. A threshold is a cut on an already-computed probability,
 *    so a re-run would return bit-identical numbers -- and would wake a
 *    container that is billed by the second. See withThreshold in api.js.
 *
 * 2. An out-of-distribution refusal is rendered as a refusal, with its reasons,
 *    and never as a prediction. The server does not send a prediction for a
 *    rejected upload at all, so there is nothing here that could accidentally
 *    present one.
 */

import { scan, warmup, withThreshold } from "../api.js";

const CLASS_LABELS = { normal: "Normal", benign: "Benign", malignant: "Malignant" };

function escapeHtml(text) {
  return String(text).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function probabilityBars(probs) {
  const rows = Object.entries(probs)
    .map(([name, value]) => {
      const pct = (value * 100).toFixed(1);
      return `
        <div class="onnm-bar-row">
          <div class="onnm-bar-label">${CLASS_LABELS[name] || escapeHtml(name)}</div>
          <div class="onnm-bar-track">
            <div class="onnm-bar-fill" style="width:${pct}%"></div>
          </div>
          <div class="onnm-bar-value">${pct}%</div>
        </div>`;
    })
    .join("");
  return `<div class="onnm-bars">${rows}</div>`;
}

function verdictClass(view) {
  if (view.inconclusive) return "onnm-verdict-inconclusive";
  return view.isLesion ? "onnm-verdict-lesion" : "onnm-verdict-normal";
}

function renderRejection(result) {
  const checks = (result.ood?.checks || [])
    .map(
      (c) => `
      <li class="${c.passed ? "onnm-check-pass" : "onnm-check-fail"}">
        <strong>${escapeHtml(c.name.replace(/_/g, " "))}</strong>
        <span>${escapeHtml(c.detail)}</span>
      </li>`,
    )
    .join("");

  return `
    <div class="onnm-result onnm-result-rejected">
      <h3>Not processed: this does not look like a radiograph</h3>
      <p class="onnm-muted">
        The model was not run. These checks decide what reaches it, and they
        refuse anything that does not look like a bone X-ray, so a holiday photo
        never receives a confident-sounding diagnosis.
      </p>
      <ul class="onnm-checks">${checks}</ul>
    </div>`;
}

/**
 * What the picture actually is, in the two cases the server can send.
 *
 * They are different claims and must not share a caption. Grad-CAM is an
 * attribution: it answers "which regions moved the score for class X", which is
 * why it has a class at all -- and why, with cam_class "auto", the old caption
 * could read "taken against the Benign class" underneath a Normal verdict and
 * leave a reader none the wiser.
 *
 * A lesion map is a trained, class-agnostic output: "this is where the lesion
 * is". It has no class to be taken against, so saying so would be false.
 *
 * Defaults to the Grad-CAM wording when `kind` is absent, so a browser holding
 * cached JS against a newer API still renders something true.
 */
/**
 * Notes about the uploaded IMAGE, as opposed to the finding.
 *
 * Kept above the verdict deliberately: "this is two views in one picture" changes
 * how the reader should weigh everything below it, so it is not a footnote. It
 * never alters the verdict itself -- the scan still ran and still answered.
 *
 * Defensive about the shape because an older cached frontend may be talking to a
 * newer API, or the reverse; an absent `advisories` key simply renders nothing.
 */
function advisoryNotes(advisories) {
  if (!Array.isArray(advisories) || advisories.length === 0) return "";
  return advisories
    .map((a) => `<p class="onnm-note">${escapeHtml(String(a && a.message ? a.message : ""))}</p>`)
    .join("");
}

function overlayCaption(overlay) {
  if (overlay.kind === "lesion_map") {
    return "Where the model located a lesion. This is a trained output, not an "
      + "after-the-fact estimate of what moved the answer.";
  }
  const name = CLASS_LABELS[overlay.cam_class] || overlay.cam_class || "predicted";
  return "Heat map taken against the <strong>" + escapeHtml(name)
    + "</strong> class. Warmer regions moved the answer more.";
}

function overlayAlt(overlay) {
  return overlay.kind === "lesion_map"
    ? "Predicted lesion location over the radiograph"
    : "Grad-CAM heat map over the radiograph";
}

function renderResult(result, threshold) {
  const view = withThreshold(result, threshold, {
    confidenceFloor: result.confidence_floor,
    entropyGate: result.entropy_gate,
  });
  const p = result.prediction || {};

  return `
    <div class="onnm-result">
      <div class="onnm-verdict ${verdictClass(view)}">
        <div class="onnm-verdict-label">${escapeHtml(view.label)}</div>
        <div class="onnm-verdict-confidence">${view.confidence.toFixed(1)}% confidence</div>
      </div>

      ${advisoryNotes(result.advisories)}

      ${
        view.inconclusive
          ? `<p class="onnm-note">
               A lesion call was withdrawn because the model was not confident
               enough to make it (peak probability ${(view.maxProbability * 100).toFixed(0)}%,
               against a ${(result.confidence_floor * 100).toFixed(0)}% floor).
               The gate can only ever withdraw a positive finding, never create one.
             </p>`
          : ""
      }

      <div class="onnm-result-grid">
        <div>
          <h4>Where the model looked</h4>
          ${
            result.overlay
              ? `<img class="onnm-overlay" alt="${escapeHtml(overlayAlt(result.overlay))}"
                      src="data:image/png;base64,${result.overlay.png_b64}" />
                 <p class="onnm-caption">${overlayCaption(result.overlay)}</p>`
              : `<p class="onnm-muted">No heat map was produced for this image.</p>`
          }
        </div>
        <div>
          <h4>Class probabilities</h4>
          ${probabilityBars(result.class_probabilities || {})}
          <p class="onnm-caption">
            Calibrated with temperature ${Number(p.temperature).toFixed(3)}.
            Decision threshold ${threshold.toFixed(4)}.
          </p>

          <label class="onnm-slider-label" for="threshold">
            Decision threshold
            <output id="threshold-out">${threshold.toFixed(2)}</output>
          </label>
          <input class="onnm-slider" type="range" id="threshold"
                 min="0.05" max="0.95" step="0.01" value="${threshold}" />
          <p class="onnm-caption">
            Lower this to catch more lesions and accept more false alarms.
            Moving it re-reads the same probabilities in your browser; the model
            is not run again.
          </p>
        </div>
      </div>
    </div>`;
}

export async function renderScanner(main) {
  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-panel">
      <h1>Scan a radiograph</h1>
      <p class="onnm-muted">
        DICOM (.dcm, .dicom, .ima), PNG, JPEG, BMP or TIFF, up to 50 MB.
        DICOM metadata is stripped before anything is stored.
      </p>

      <div class="onnm-uploader">
        <label class="onnm-btn" for="file">Choose a radiograph</label>
        <input id="file" type="file" class="onnm-visually-hidden"
               accept=".dcm,.dicom,.ima,.png,.jpg,.jpeg,.bmp,.tif,.tiff" />
        <span id="filename" class="onnm-muted">No file chosen</span>
      </div>

      <label class="onnm-consent">
        <input type="checkbox" id="consent" />
        <span>
          Share this image to help improve the model. Off by default. Only the
          256-pixel processed image is stored, never the original file or its
          metadata, and sharing is asked per image rather than once.
          <strong>By ticking this I confirm I have the right to share this image,
          and that it shows no visible name, hospital number or other
          identifier.</strong>
          You can delete a shared image from your account page at any time until a
          reviewer approves it. After approval, ask us and we will still delete the
          stored copy, but what a model has already learned from it cannot be
          reversed.
          <a href="/terms" data-link>Terms of use</a> ·
          <a href="/privacy" data-link>Privacy notice</a>.
        </span>
      </label>

      <p><button class="onnm-btn onnm-btn-primary" id="analyse" disabled>Analyse</button></p>

      <div id="status" class="onnm-status" role="status" aria-live="polite"></div>
      <div id="results"></div>
    </section>`,
  );

  const fileInput = main.querySelector("#file");
  const filename = main.querySelector("#filename");
  const analyse = main.querySelector("#analyse");
  const status = main.querySelector("#status");
  const results = main.querySelector("#results");
  const consent = main.querySelector("#consent");

  let current = null;

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    filename.textContent = file ? file.name : "No file chosen";
    analyse.disabled = !file;
    // Start the container now, while the visitor is still reading. This hides
    // the cold start behind their own decision-making rather than adding it to
    // the wait after they press Analyse.
    if (file) warmup();
  });

  analyse.addEventListener("click", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;

    analyse.disabled = true;
    results.innerHTML = "";
    status.textContent = "Analysing… the model may take a few seconds to wake.";

    try {
      const result = await scan(file, { shareConsent: consent.checked });
      status.textContent = "";

      if (!result.is_radiograph) {
        results.innerHTML = renderRejection(result);
        return;
      }

      current = result;
      const threshold = result.prediction?.decision_threshold ?? result.default_threshold;
      results.innerHTML = renderResult(result, threshold);
      wireSlider();

      if (result.budget?.warn) {
        status.textContent = `Note: ${Math.round((result.budget.remainingSeconds || 0) / 60)} minutes of scanning capacity remain this month.`;
      }
    } catch (error) {
      status.textContent = "";
      results.innerHTML = `<div class="onnm-banner onnm-banner-error" role="alert">${escapeHtml(error.message)}</div>`;
    } finally {
      analyse.disabled = false;
    }
  });

  function wireSlider() {
    const slider = results.querySelector("#threshold");
    if (!slider) return;
    slider.addEventListener("input", () => {
      const value = Number(slider.value);
      results.innerHTML = renderResult(current, value);
      wireSlider();
      // Restore focus so dragging with the keyboard is not interrupted.
      const next = results.querySelector("#threshold");
      next.focus();
      next.value = String(value);
    });
  }

  return null;
}
