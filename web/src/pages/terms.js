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
 * names the Terms as the contract that gives account processing its lawful basis.
 * Both assumed this page.
 *
 * WHAT MAKES IT A GATE RATHER THAN A CHECKBOX
 * -------------------------------------------
 * Nothing here is trusted. Ticking the box posts to /api/terms/accept, and only
 * the server's answer unlocks anything: signed out it mints a signed cookie that
 * `authStart` requires before it will talk to Google, and signed in it writes the
 * acceptance onto the account row. A visitor who edits this file in their own
 * browser gets exactly as far as a visitor who does not.
 *
 * HOW THIS VERSION DIFFERS FROM 2026-08-29
 * ----------------------------------------
 * The first version carried over wording from a much earlier deployment on
 * different infrastructure, and it showed. Reviewing it against the code that
 * actually runs produced sixteen findings, recorded in
 * `compliance/TERMS_REVIEW_PACK.md`. The substantive changes here:
 *
 *   - The operator is named, and a monitored contact address is given (was: an
 *     indirection to a file in the repository).
 *   - Governing law and jurisdiction now exist, with a consumer carve-out.
 *   - A prohibited-use section, covering illegal content, automated access, model
 *     extraction and circumvention, with an explicit right to suspend or close an
 *     account for breach. None of this existed.
 *   - The erasure clause now separates the stored copy, which is always
 *     deletable, from the trained weights, which are not, and cites the
 *     derogation it relies on instead of asserting a house rule.
 *   - The liability exclusion carves out death and personal injury caused by
 *     negligence, which cannot lawfully be excluded and whose absence risked the
 *     whole clause.
 *   - "Three parties, and no others" was literally untrue and is now accurate.
 *   - A minimum age, a route to close an account, a breach-notification
 *     statement, and the right to complain to the ICO were all missing.
 *
 * The full Article 13 privacy information lives in `privacy.js`, because a
 * contract is not a privacy notice and this document should not pretend to be one.
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
export const TERMS_VERSION = "2026-08-30";

/**
 * Who the counterparty is.
 *
 * A contract needs a named party, and UK GDPR Article 13(1)(a) needs a named
 * controller. The previous version had neither: it said "the maintainers" and
 * pointed questions at a file in the repository, which is an indirection rather
 * than an identity.
 *
 * "Controller" is not a role that gets appointed; it is a description of whoever
 * decides why and how personal data is processed. That is the operator, acting
 * personally rather than for an institution, and `compliance/DPIA.md` §A records
 * the same. If ONNM ever runs under a university or company, this changes and so
 * does the whole controller/processor analysis.
 *
 * `worker/lib/terms.test.js` fails while a placeholder is present here, so an
 * unresolved value cannot reach production.
 */
export const OPERATOR = {
  name: "Khalid Faiz",
  capacity: "acting in a personal capacity as an independent researcher",
  contact: "kzfhero@gmail.com",
};

/**
 * The Terms themselves.
 *
 * Kept as data rather than inlined into the markup so the same text can be shown
 * on this page, quoted in the footer, and published elsewhere without three copies
 * drifting apart.
 */
const SECTIONS = [
  {
    heading: "1. Who we are, and what these Terms cover",
    body: `ONNM (OsteoNeuralNetwork Model) is operated by
    <strong>${OPERATOR.name}</strong>, ${OPERATOR.capacity}. In these Terms "we", "us"
    and "our" mean the operator, and "you" means the person using the service. These
    Terms are the agreement between us and govern your use of the website, the scanner
    and your account. Our <a href="/privacy" data-link>Privacy notice</a> explains what
    we do with personal data and forms part of this agreement.
    <br /><br />
    You can reach us at <a href="mailto:${OPERATOR.contact}">${OPERATOR.contact}</a>.
    That address is monitored, and it is the one to use for a question about these
    Terms, a request about your data, or a problem with the service. Security
    vulnerabilities should go through the process in the repository's
    <code>SECURITY.md</code> instead, so they are handled privately.`,
  },
  {
    heading: "2. What ONNM is, and what it is not",
    body: `ONNM is free, open-source research software. It accepts a bone radiograph
    and returns an experimental classification, a calibrated probability for each
    class, and a heat map showing which part of the image influenced the result.
    <strong>It is an unvalidated research prototype, not a medical device and not a
    clinical service.</strong> It holds no FDA, CE or MHRA clearance and has not been
    clinically validated. Access may be changed, suspended or withdrawn at any time.`,
  },
  {
    heading: "3. Never use it for a clinical decision",
    body: `Do not use ONNM's output to make, confirm or delay any diagnosis,
    treatment or referral, and never in an emergency. Every radiograph requires
    review by a qualified clinician. Model output can be incomplete, biased,
    incorrect or misleading, and the model is known to produce false positives on
    complex but normal anatomy. In testing it also missed roughly one malignancy in
    three. If you have a health concern, speak to a doctor.`,
  },
  {
    heading: "4. Who may use ONNM",
    body: `<strong>You must be 18 or over.</strong> ONNM processes medical imaging, it
    is not designed for children, and we do not offer it to them. You may use ONNM for
    research, education and personal interest. You may not build it into any clinical
    workflow, product or service, or offer its output to anyone else as if it were a
    clinical opinion.`,
  },
  {
    heading: "5. Only upload images you are entitled to use",
    body: `You must have the authority and any consent needed to process an image
    before you upload it. <strong>Do not upload an identifiable patient
    radiograph.</strong> De-identification cannot remove a name, hospital number or
    other identifier that is burned into the image pixels, so you must check for
    those and remove them yourself first. DICOM headers are stripped before anything
    is stored, but that protects metadata, not pixels.
    <br /><br />
    If you upload an image in breach of this section and that causes us a loss, a
    claim or a reasonable cost, you agree to reimburse us for it. This does not apply
    to the extent the law does not allow it, and it does not affect your rights as a
    consumer.`,
  },
  {
    heading: "6. What you must not do",
    body: `You must not: upload anything unlawful, or anything you have no right to
    upload; attempt to identify a person from an image or from anything the service
    shows you; try to reach another user's records; bypass or circumvent any limit,
    gate or safeguard, including the review process and the usage limits; access the
    service by automated means, scrape it, or send requests at a rate the interface
    does not offer; attempt to reconstruct, extract or copy the model from its
    outputs; upload malicious files or anything intended to disrupt the service; or
    present ONNM's output to anyone as a clinical opinion.
    <br /><br />
    <strong>Unlawful content is treated seriously and separately.</strong> If content
    you upload appears to be unlawful we will preserve it, report it to the
    appropriate authority, and co-operate with any lawful request, and we will not
    delete material we have been asked to preserve. We may suspend or close your
    account immediately where we reasonably believe you have breached this section.`,
  },
  {
    heading: "7. What is stored, and only if you ask for it",
    body: `Sharing is off by default and is asked separately for every image, because
    agreeing once says nothing about the next file you open. If you do not tick the
    box, your image is analysed and never written down. If you do, a
    <strong>256-pixel processed copy</strong> is stored for human review, never your
    original file and never its metadata. Either way a record of the scan and its
    result is kept against your account so your history can be shown. The
    <a href="/privacy" data-link>Privacy notice</a> sets out how long each of these is
    kept and on what lawful basis.`,
  },
  {
    heading: "8. Your choices over a shared image",
    body: `While a shared image is waiting for review you can delete it yourself at any
    time from your account page, and it leaves the review queue immediately. After a
    reviewer has approved it, that button is gone, but your rights are not: ask us and
    <strong>we will delete the stored copy, whether or not it has been
    approved</strong>, and we will confirm when it is done.
    <br /><br />
    What we cannot do is reverse what a model has already learned from an image it
    trained on. No technique exists that removes one image's contribution from a set
    of trained weights, and we will not pretend otherwise. For that, and only for
    that, we rely on the research exemption in Article 17(3)(d) UK GDPR, together with
    the safeguards described in the <a href="/privacy" data-link>Privacy notice</a>.
    If that distinction matters to you, delete the image before it is approved.`,
  },
  {
    heading: "9. Who sees a shared image",
    body: `Three parties: you, the single reviewer account that assigns its label, and
    the model it is used to train. Beyond those, the only others who hold your image
    are the infrastructure providers who store and process it on our instructions, who
    are named in the <a href="/privacy" data-link>Privacy notice</a>. Shared images are
    not published, sold, or passed to anyone else. If a reviewer rejects an upload, the
    stored copy is deleted automatically within seven days.`,
  },
  {
    heading: "10. Your account, and how it can end",
    body: `Accounts are created through Google Sign-In. ONNM never receives your
    Google password. Your email address and Google account identifier link scans to
    your account; your name and photo are not shown publicly unless you choose to
    appear as a contributor. Keep access to your Google account secure.
    <br /><br />
    <strong>You can close your account at any time</strong> by asking us at
    <a href="mailto:${OPERATOR.contact}">${OPERATOR.contact}</a>. We will delete your
    account and your scan history within 30 days, subject only to the limit on trained
    weights explained in section 8. We may suspend or close your account if you breach
    these Terms or if the law requires it, and we will tell you why unless we are
    prevented from doing so.`,
  },
  {
    heading: "11. Location",
    body: `Your country is recorded once, at country level only, from the connection
    your browser makes. <strong>No IP address is stored</strong> and the map shows
    aggregated countries rather than places.`,
  },
  {
    heading: "12. Intellectual property",
    body: `The source code is licensed under <strong>Apache-2.0</strong>. Model weights
    are published under <strong>CC BY-NC 4.0</strong>, which permits non-commercial use
    with attribution. Training data carries its own separate terms, and some of it is
    both non-commercial and no-derivatives, which restricts what may be redistributed;
    the datasets and their licences are listed in the repository's model card.
    <br /><br />
    Uploading an image does not transfer ownership of it. It stays yours. If you ticked
    the sharing box for that image, you grant us a non-exclusive, worldwide,
    royalty-free licence to store, process and use a de-identified copy for the research
    purposes described in these Terms, and for no other purpose.`,
  },
  {
    heading: "13. No warranties, and the limits of liability",
    body: `To the fullest extent permitted by law the service is provided "as is" and
    "as available", with no warranty of accuracy, fitness for a particular purpose,
    availability or regulatory compliance. To the fullest extent permitted by
    applicable law, we are not liable for clinical decisions, missed or delayed care,
    false positives, false negatives, data loss, or indirect or consequential damages
    arising from use of the service.
    <br /><br />
    <strong>Nothing in these Terms limits or excludes our liability for death or
    personal injury caused by negligence, for fraud or fraudulent misrepresentation, or
    for any other liability that cannot lawfully be limited or excluded.</strong> If you
    are a consumer, you keep all the rights the law gives you, and nothing here affects
    them.`,
  },
  {
    heading: "14. Privacy, your rights, and complaints",
    body: `Our <a href="/privacy" data-link>Privacy notice</a> explains what we collect,
    why, how long we keep it, who it reaches, and the rights you have over it, including
    access, correction, erasure, objection and portability. Write to us at
    <a href="mailto:${OPERATOR.contact}">${OPERATOR.contact}</a> to exercise any of them
    and we will respond within one month.
    <br /><br />
    If there is a personal data breach that is likely to result in a high risk to you,
    we will tell you without undue delay and explain what happened and what to do. You
    also have the right to complain to the Information Commissioner's Office at
    <a href="https://ico.org.uk/make-a-complaint/" rel="noopener noreferrer"
    target="_blank">ico.org.uk</a>, and we would ask that you raise it with us first so
    we have the chance to put it right.`,
  },
  {
    heading: "15. Changes to these Terms",
    body: `Updated Terms apply once they have been presented to you for acceptance, and
    a material change will require you to agree again before you can carry on using the
    scanner. <strong>Every version of this text is kept in the project's public git
    history</strong>, identified by the version date shown on this page, so you can
    always see exactly what you agreed to and when it changed.`,
  },
  {
    heading: "16. Governing law, and where disputes are heard",
    body: `These Terms are governed by the law of England and Wales, and the courts of
    England and Wales have exclusive jurisdiction. <strong>If you are a consumer
    resident elsewhere, you may also bring proceedings in your country of residence, and
    you keep the protection of any mandatory consumer law that applies there.</strong>
    If any provision of these Terms is found unenforceable, the rest continues in
    effect.`,
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
      <p class="onnm-caption">
        Version ${TERMS_VERSION} ·
        <a href="/privacy" data-link>Privacy notice</a>
      </p>

      <div class="onnm-terms-notice">
        <h3>The short version</h3>
        <p>
          This is a research prototype, not a medical device. Never use it to make a
          medical decision. You must be 18 or over. Nothing you upload is stored unless
          you tick the sharing box for that image, and you can delete a shared image
          yourself until a reviewer approves it. After that, ask us and we will still
          delete the stored copy, but what a model has already learned from it cannot
          be reversed.
        </p>
      </div>

      <div class="onnm-terms-clauses">${renderSections()}</div>

      <div class="onnm-terms-gate">
        <label class="onnm-consent">
          <input type="checkbox" id="agree" />
          <span>
            I am 18 or over. I have read and agree to these Terms and the Privacy
            notice. I understand ONNM is a research prototype, is not a medical device,
            and must not be used to make a medical decision. I will only upload images
            I have the right to upload.
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
