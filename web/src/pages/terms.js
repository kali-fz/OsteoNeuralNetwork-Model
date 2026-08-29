/**
 * The Terms gate: the one thing every account passes through.
 *
 * WHY THIS PAGE EXISTS
 * --------------------
 * Until this was built, "Sign in with Google" went straight to Google. An account
 * was created and a scan could be run without anyone agreeing to anything, and
 * `users.tos_accepted_at` was written as a copy of `created_at` -- a record that a
 * row was inserted, not that a human read something.
 *
 * That gap was load-bearing in the compliance work. `compliance/DPIA.md` names
 * "the Terms" as the mitigation for four separate risks, and `compliance/ROPA.md`
 * names Art 6(1)(b), performance of a contract, as the lawful basis. Both assumed
 * this page.
 *
 * WHAT MAKES IT A GATE RATHER THAN A CHECKBOX
 * -------------------------------------------
 * Nothing here is trusted. Ticking the box posts to /api/terms/accept, and only
 * the server's answer unlocks anything: signed out it mints a signed cookie that
 * `authStart` requires before it will talk to Google, and signed in it writes the
 * acceptance onto the account row. A visitor who edits this file in their own
 * browser gets exactly as far as a visitor who does not.
 *
 * The text is adapted from the Terms that shipped with the Streamlit deployment
 * (recoverable at `git show ea1ce2e^:src/legal.py`). Every statement about where
 * data lives had to be rewritten, because that version described software running
 * on one person's machine and this one describes a hosted service.
 */

import { acceptTerms, getSession } from "../api.js";
import { navigate } from "../main.js";

/**
 * The version string recorded against each acceptance.
 *
 * A date rather than a number, so that reading a stored value tells you which text
 * was on the page without consulting a changelog. Changing the Terms means
 * changing this, and the stored version is what a future re-consent prompt would
 * compare against.
 */
export const TERMS_VERSION = "2026-08-29";

/**
 * The Terms themselves.
 *
 * Kept as data rather than inlined into the markup so the same text can be shown
 * on this page, quoted in the footer, and published elsewhere without three copies
 * drifting apart.
 */
const SECTIONS = [
  {
    heading: "1. What ONNM is, and what it is not",
    body: `ONNM is free, open-source research software. It accepts a bone radiograph
    and returns an experimental classification, a calibrated probability for each
    class, and a heat map showing which part of the image influenced the result.
    <strong>It is an unvalidated research prototype, not a medical device and not a
    clinical service.</strong> It holds no FDA, CE or MHRA clearance and has not been
    clinically validated. Access may be changed, suspended or withdrawn at any time.`,
  },
  {
    heading: "2. Never use it for a clinical decision",
    body: `Do not use ONNM's output to make, confirm or delay any diagnosis,
    treatment or referral, and never in an emergency. Every radiograph requires
    review by a qualified clinician. Model output can be incomplete, biased,
    incorrect or misleading, and the model is known to produce false positives on
    complex but normal anatomy. If you have a health concern, speak to a doctor.`,
  },
  {
    heading: "3. Only upload images you are entitled to use",
    body: `You must have the authority and any consent needed to process an image
    before you upload it. <strong>Do not upload an identifiable patient
    radiograph.</strong> De-identification cannot remove a name, hospital number or
    other identifier that is burned into the image pixels, so you must check for
    those and remove them yourself first. DICOM headers are stripped before anything
    is stored, but that protects metadata, not pixels.`,
  },
  {
    heading: "4. What is stored, and only if you ask for it",
    body: `Sharing is off by default and is asked separately for every image, because
    agreeing once says nothing about the next file you open. If you do not tick the
    box, your image is analysed and never written down. If you do, a
    <strong>256-pixel processed copy</strong> is stored for human review, never your
    original file and never its metadata. Either way a record of the scan and its
    result is kept against your account so your history can be shown.`,
  },
  {
    heading: "5. You can withdraw a shared image until it is approved",
    body: `While a shared image is waiting for review you can delete it at any time
    from your account page, and it leaves the review queue immediately.
    <strong>That window closes when a reviewer approves it.</strong> Approval adds
    the image to the research training set, and once a model has trained on an image
    there is no way to remove what the model learned from it. We cannot take it back
    out afterwards and we will not claim otherwise. If you are unsure, delete it
    before then.`,
  },
  {
    heading: "6. Who ever sees a shared image",
    body: `Three parties, and no others: you, the single reviewer account that
    assigns its label, and the model it is used to train. Shared images are not
    published, sold, or passed to anyone else. If a reviewer rejects an upload, the
    stored copy is deleted automatically within seven days.`,
  },
  {
    heading: "7. Your account",
    body: `Accounts are created through Google Sign-In. ONNM never receives your
    Google password. Your email address and Google account identifier link scans to
    your account; your name and photo are not shown publicly unless you choose to
    appear as a contributor. Keep access to your Google account secure, and do not
    attempt to reach another user's records, bypass safeguards, upload malicious
    files, or re-identify anyone.`,
  },
  {
    heading: "8. Location",
    body: `Your country is recorded once, at country level only, from the connection
    your browser makes. <strong>No IP address is stored</strong> and the map shows
    aggregated countries rather than places.`,
  },
  {
    heading: "9. Intellectual property",
    body: `The source code is licensed under the repository's licence. Model weights
    are published under CC BY-NC 4.0. Training data carries its own separate terms.
    Uploading an image does not transfer ownership of it; you grant a non-exclusive
    licence to store, process and use a de-identified copy for the research purposes
    described above, and only if you ticked the sharing box.`,
  },
  {
    heading: "10. No warranties, and the limits of liability",
    body: `To the fullest extent permitted by law the service is provided "as is" and
    "as available", with no warranty of accuracy, fitness for a particular purpose,
    availability or regulatory compliance. To the fullest extent permitted by
    applicable law, the maintainers are not liable for clinical decisions, missed or
    delayed care, false positives, false negatives, data loss, or indirect or
    consequential damages arising from use of the service. Nothing here excludes
    liability that cannot lawfully be excluded, and a blanket waiver may be
    unenforceable where you live.`,
  },
  {
    heading: "11. Changes, and how to reach us",
    body: `Mandatory law where you are located applies, and if any provision is
    unenforceable the rest continues in effect. Updated Terms apply once they have
    been presented for acceptance; a material change should require agreeing again.
    For questions, access requests or deletion requests, use the contact channel
    published in the repository's <code>SECURITY.md</code>.`,
  },
];

function renderSections() {
  return SECTIONS.map(
    (section) => `
      <section class="onnm-terms-clause">
        <h3>${section.heading}</h3>
        <p>${section.body}</p>
      </section>`,
  ).join("");
}

/**
 * @param {HTMLElement} main
 * @param {{session: object|null}} state
 */
export async function renderTerms(main, state) {
  const signedIn = Boolean(state.session?.signed_in);

  // Two audiences, one page. A signed-out visitor is on their way to Google; a
  // signed-in one already has an account that predates this gate and is being
  // asked to catch up. The difference is one sentence and where the button goes.
  const lede = signedIn
    ? `Your account was created before these Terms were published. Please read them
       and agree to carry on using the scanner. Nothing you have already done is
       affected, and your scan history is untouched.`
    : `Please read these before creating an account. You will need to agree to them
       to sign in.`;

  const action = signedIn ? "Agree and continue" : "Agree and continue to Google sign-in";

  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-panel onnm-terms-page">
      <h1>Terms of use</h1>
      <p class="onnm-hero-lede">${lede}</p>
      <p class="onnm-caption">Version ${TERMS_VERSION}</p>

      <div class="onnm-terms-notice">
        <h3>The short version</h3>
        <p>
          This is a research prototype, not a medical device. Never use it to make a
          medical decision. Nothing you upload is stored unless you tick the sharing
          box for that image, and you can delete a shared image at any time until a
          reviewer approves it, after which it is part of the training set and cannot
          be removed.
        </p>
      </div>

      <div class="onnm-terms-clauses">${renderSections()}</div>

      <div class="onnm-terms-gate">
        <label class="onnm-consent">
          <input type="checkbox" id="agree" />
          <span>
            I have read and agree to these Terms. I understand ONNM is a research
            prototype, is not a medical device, and must not be used to make a
            medical decision.
          </span>
        </label>

        <p>
          <button class="onnm-btn onnm-btn-primary" id="accept" type="button" disabled>
            ${action}
          </button>
        </p>
        <p class="onnm-status" id="terms-status" role="status" aria-live="polite"></p>
      </div>
    </section>`,
  );

  const agree = main.querySelector("#agree");
  const accept = main.querySelector("#accept");
  const status = main.querySelector("#terms-status");

  agree.addEventListener("change", () => {
    accept.disabled = !agree.checked;
    status.textContent = "";
  });

  accept.addEventListener("click", async () => {
    if (!agree.checked) return;
    accept.disabled = true;
    status.textContent = "Recording your agreement…";

    try {
      await acceptTerms(TERMS_VERSION);
    } catch (error) {
      accept.disabled = false;
      status.textContent = error.message;
      return;
    }

    if (signedIn) {
      // Re-read the session so the rest of the app sees `terms_accepted: true`
      // and stops routing this visitor back here.
      state.session = await getSession().catch(() => state.session);
      navigate("/scanner");
      return;
    }

    // A full navigation rather than a router push: this leaves the application
    // for Google, and the acceptance cookie just minted is what /api/auth/google/
    // start now requires before it will begin.
    window.location.href = "/api/auth/google/start";
  });

  return null;
}
