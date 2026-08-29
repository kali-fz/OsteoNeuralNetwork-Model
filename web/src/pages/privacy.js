/**
 * The Privacy notice.
 *
 * WHY THIS PAGE EXISTS
 * --------------------
 * The Terms are a contract. A contract is not a privacy notice, and until this
 * page was written the site had neither: what stood in its place was a thirty-word
 * summary in a footer disclosure, under a paragraph that told visitors outright
 * that its privacy information was incomplete.
 *
 * UK GDPR Article 13 sets out what a data subject has to be told when their data
 * is collected from them, and almost none of it was anywhere on the site: the
 * controller's identity, the lawful basis for each purpose, the recipients, the
 * transfers and their safeguards, the retention periods, the rights, the right to
 * withdraw consent, and the right to complain to the ICO. This page is that
 * information, and `compliance/ROPA.md` is the internal record it is generated
 * from, so the two should be changed together.
 *
 * WHY THE LAWFUL BASES READ AS THEY DO
 * ------------------------------------
 * The earlier drafts of the DPIA and ROPA proposed legitimate interests for the
 * shared-image corpus, with the per-image tick box described as "an interface
 * control" rather than as the legal basis. That position is defensible on paper
 * and it was wrong for this system, because the tick box already behaves in every
 * respect like consent: granular, opt-in, off by default, asked separately for
 * every single file, and withdrawable. A regulator reading the interface would
 * call it consent whatever the register said.
 *
 * So it is consent, and it is written down as consent. The objection to consent
 * was always that a model cannot be untrained, which would make withdrawal
 * impossible to honour. That objection does not survive contact with Article 7(3),
 * whose second sentence says withdrawal does not affect the lawfulness of
 * processing carried out before it. Withdrawal stops future use and deletes the
 * stored copy; the weights already trained are processing that was lawful when it
 * happened, and Article 17(3)(d) covers their retention for research.
 */

// The operator is defined once, in terms.js, because the contract that names them
// as the counterparty is where they primarily appear. Both documents must name the
// same party, and one constant is how that stays true.
import { OPERATOR } from "./terms.js";

/**
 * The version of this notice.
 *
 * Dated like the Terms version, and for the same reason: a stored value should
 * say which text a person actually read.
 */
export const PRIVACY_VERSION = "2026-08-30";

/**
 * Retention. Generated from the same facts as `compliance/ROPA.md` §E.
 *
 * Every row here is a period the code actually enforces or the operator has
 * committed to. Nothing is aspirational, because a retention schedule that
 * describes an intention rather than a behaviour is worse than none: it tells a
 * data subject something untrue about their own data.
 */
const RETENTION = [
  ["Your account and scan history", "Until you close your account. We delete within 30 days of being asked.", "On request"],
  ["A scan you did not share", "No image is stored at all. Only the verdict and the time, against your account.", "With the account"],
  ["A shared image awaiting review", "Until it is reviewed, or until you delete it yourself, whichever is first.", "You, at any time"],
  ["A shared image a reviewer rejected", "Deleted automatically within 7 days.", "Automatic"],
  ["A shared image a reviewer approved", "Kept for the life of the research. The stored copy is still deleted on request.", "On request"],
  ["Country code", "For the life of the account.", "With the account"],
  ["One-use location token", "Deleted as soon as it is used or expires.", "Automatic"],
  ["Public contributor name and photo", "Until you turn the contributor toggle off.", "You, at any time"],
  ["Platform logs", "Cloudflare's own retention period for Workers observability.", "Provider"],
];

const SECTIONS = [
  {
    heading: "1. Who is responsible for your data",
    body: `ONNM (OsteoNeuralNetwork Model) is operated by
    <strong>__OPERATOR_NAME__</strong>, __OPERATOR_CAPACITY__, who is the data
    controller for the processing described here. You can reach us at
    <a href="mailto:__OPERATOR_CONTACT__">__OPERATOR_CONTACT__</a> about anything in
    this notice, including a request to see, correct or delete your data.
    <br /><br />
    We have not appointed a Data Protection Officer. We are not required to, because
    this is a deliberately small research deployment rather than large-scale
    processing, and the reasoning is recorded in the project's Data Protection Impact
    Assessment.`,
  },
  {
    heading: "2. What this notice covers",
    body: `This notice covers the hosted service at osteoneuralnetwork.com: the
    website, the scanner, your account, and the human review of images people choose
    to share. It sits alongside the <a href="/terms" data-link>Terms of use</a>, which
    is the agreement between us.
    <br /><br />
    Some of what we process is <strong>data concerning health</strong>, which the law
    treats as a special category needing extra protection. A radiograph is health
    data, and so is the verdict the model produces from it. Section 5 explains the
    additional condition we rely on for those.`,
  },
  {
    heading: "3. What we collect",
    body: `<strong>When you create an account:</strong> your email address, your Google
    account identifier, and, if you choose to appear publicly as a contributor, your
    display name and profile photo. We never receive your Google password.
    <br /><br />
    <strong>When you run a scan:</strong> the image itself, in memory, for as long as
    the analysis takes; and a record of the result, meaning the verdict, the class
    probabilities, the threshold used, and whether the out-of-distribution gate flagged
    it. <strong>The image is not stored unless you tick the sharing box for that
    image.</strong> If you do tick it, we keep a 256-pixel processed copy, never your
    original file. Metadata is removed before anything is stored: DICOM headers are
    stripped and the image is re-encoded without EXIF.
    <br /><br />
    <strong>Where you are:</strong> a two-letter country code, once, so the homepage
    map can show which countries have contributed. <strong>We do not store your IP
    address</strong>, anywhere, ever. The database column is constrained to two
    characters, so it is physically incapable of holding anything more precise than a
    country.
    <br /><br />
    <strong>We do not use advertising or analytics cookies</strong> and we do not track
    you across other sites. The only cookies we set are the ones that keep you signed
    in and carry your agreement to the Terms through the sign-in process.`,
  },
  {
    heading: "4. Why we process it, and our lawful basis",
    body: `<strong>To give you an account and keep your scans separate from everyone
    else's.</strong> Basis: Article 6(1)(b), performance of our contract with you,
    which is the Terms of use.
    <br /><br />
    <strong>To run the scan you asked for and show you the result.</strong> Basis:
    Article 6(1)(b), performance of the contract, together with Article 9(2)(a),
    your explicit consent to us processing health data, which is what uploading an
    image for analysis is.
    <br /><br />
    <strong>To keep a shared image for human review and possible research
    training.</strong> Basis: Article 6(1)(a) and Article 9(2)(a), your explicit
    consent, given separately for each individual image by ticking the sharing box.
    This is genuinely optional. The scanner works identically if you never tick it.
    <br /><br />
    <strong>To show country-level contribution counts on the homepage.</strong> Basis:
    Article 6(1)(f), our legitimate interest in showing that the project has reach,
    balanced against a country code being close to the least identifying thing we
    could display.
    <br /><br />
    <strong>To show your name and photo as a contributor.</strong> Basis: Article
    6(1)(a), consent. Off by default, and reversible whenever you like.
    <br /><br />
    <strong>To keep the service secure and diagnose faults.</strong> Basis: Article
    6(1)(f), our legitimate interest in the service working and not being abused.`,
  },
  {
    heading: "5. Health data, research, and what consent means here",
    body: `Where we rely on your consent you can withdraw it at any time, and
    withdrawing is as easy as giving it: delete the image from your account page, or
    write to us. <strong>Withdrawal does not affect processing that already happened
    lawfully</strong>, which is the position Article 7(3) sets out.
    <br /><br />
    That matters for one specific thing. If a reviewer has already approved an image
    and a model has already trained on it, we will still delete the stored copy on
    request, but we cannot reverse what the model learned. No technique exists that
    removes a single image's contribution from a set of trained weights. For the
    continued retention of approved images in the research set, and for that limit on
    erasure, we rely on Article 9(2)(j) and Article 17(3)(d), scientific research,
    with the Article 89(1) safeguards described in section 9.`,
  },
  {
    heading: "6. Who your data reaches",
    body: `<strong>Cloudflare</strong> hosts everything: the website, the database
    where accounts and shared images are stored, and the container the model runs in.
    They process it on our instructions under their data processing addendum.
    <br /><br />
    <strong>Google</strong> provides sign-in. They tell us your email address and an
    account identifier when you sign in; we tell them nothing about what you scan.
    <br /><br />
    <strong>One reviewer.</strong> A shared image is shown to a single pinned reviewer
    account so a human can label it. That is the only person who sees it.
    <br /><br />
    We do not sell your data, we do not share it for advertising, and we do not pass it
    to anyone else except where the law requires it, or where content appears to be
    unlawful and must be reported.`,
  },
  {
    heading: "7. Where your data goes",
    body: `Cloudflare and Google both operate globally, so your data may be processed
    outside the UK. Both providers make an international data transfer agreement
    available as part of their standard terms, incorporating the UK Addendum to the
    EU Standard Contractual Clauses, and that is the safeguard we rely on. You can ask
    us for details of the arrangement for either provider.`,
  },
  {
    heading: "8. How long we keep things",
    table: RETENTION,
    body: `We do not keep anything "just in case". Where a period is enforced by code
    rather than by someone remembering, the table says so.`,
  },
  {
    heading: "9. How we protect it",
    body: `Sharing is off by default and asked for every image, and an image that
    arrives without that flag is discarded rather than stored, so a bug in the website
    cannot quietly retain something you did not offer. Images are downscaled to 256
    pixels and stripped of metadata before storage. Sessions use a signed, host-scoped,
    HTTP-only cookie. Access to the review queue is restricted to one account and needs
    a separate key that the website itself does not hold. Every database query is
    parameterised, and the database rejects contradictory review decisions at the
    storage layer rather than trusting the application.
    <br /><br />
    The deployment is deliberately capped: 500 accounts, 50 submissions per account per
    day, and 200 MB of stored images in total. Those limits bound how much data can
    exist at all, which is a privacy measure as much as a cost one.`,
  },
  {
    heading: "10. Your rights",
    body: `You have the right to <strong>be told</strong> what we hold, to
    <strong>see</strong> it, to have it <strong>corrected</strong>, to have it
    <strong>deleted</strong>, to <strong>restrict</strong> or <strong>object</strong> to
    what we do with it, to <strong>withdraw consent</strong>, and to receive the data
    you gave us in a portable form.
    <br /><br />
    Some of this you can do yourself, immediately: your account page lists every scan
    you have run and lets you delete any shared image that has not yet been reviewed,
    and the contributor toggle is reversible at any time. For anything else, write to
    <a href="mailto:__OPERATOR_CONTACT__">__OPERATOR_CONTACT__</a>. <strong>We will
    respond within one month.</strong>
    <br /><br />
    Two honest limits. Erasure of an approved image is limited as described in section
    5: we delete the stored copy but cannot untrain a model. And if someone asks about
    a patient depicted in an image that another person uploaded, we usually cannot
    identify who that is, because we deliberately hold nothing that would let us; in
    that situation Article 11 applies and we may need identifying information before we
    can act.`,
  },
  {
    heading: "11. If something goes wrong",
    body: `If there is a personal data breach that is likely to result in a high risk to
    your rights, <strong>we will tell you without undue delay</strong>, explain what
    happened, what data was involved, and what you can do about it. Where the law
    requires it we will also report it to the Information Commissioner's Office within
    72 hours. The project keeps a written incident response procedure covering exactly
    this.`,
  },
  {
    heading: "12. Children",
    body: `ONNM is for adults. <strong>You must be 18 or over to use it</strong>, and we
    do not knowingly collect data from children. If you believe a child has created an
    account, tell us and we will delete it.`,
  },
  {
    heading: "13. Changes, and how to complain",
    body: `If we change this notice we will update the version date at the top of the
    page, and every previous version is kept in the project's public source history so
    you can see what changed. Where a change is material we will ask you to agree
    again.
    <br /><br />
    If you are unhappy with how we have handled your data, please tell us first so we
    have a chance to put it right. You also have the right to complain to the
    <strong>Information Commissioner's Office</strong>, the UK's data protection
    regulator, at <a href="https://ico.org.uk/make-a-complaint/" rel="noopener
    noreferrer" target="_blank">ico.org.uk/make-a-complaint</a> or on 0303 123 1113.`,
  },
];

function renderTable(rows) {
  return `
    <div class="onnm-table-wrap">
      <table class="onnm-table">
        <thead>
          <tr><th>What</th><th>How long</th><th>Deleted by</th></tr>
        </thead>
        <tbody>
          ${rows
            .map(
              ([what, how, who]) =>
                `<tr><td>${what}</td><td>${how}</td><td>${who}</td></tr>`,
            )
            .join("")}
        </tbody>
      </table>
    </div>`;
}

function renderSections(operator) {
  // The operator's details appear in three clauses. Substituted at render time
  // rather than written into each string, so there is exactly one place to change
  // when the controller's identity is confirmed.
  const fill = (text) =>
    text
      .replaceAll("__OPERATOR_NAME__", operator.name)
      .replaceAll("__OPERATOR_CAPACITY__", operator.capacity)
      .replaceAll("__OPERATOR_CONTACT__", operator.contact);

  return SECTIONS.map(
    (section) => `
      <section class="onnm-terms-clause">
        <h3>${section.heading}</h3>
        <p>${fill(section.body)}</p>
        ${section.table ? renderTable(section.table) : ""}
      </section>`,
  ).join("");
}

/**
 * @param {HTMLElement} main
 */
export async function renderPrivacy(main) {
  main.insertAdjacentHTML(
    "beforeend",
    `
    <section class="onnm-panel onnm-terms-page">
      <h1>Privacy notice</h1>
      <p class="onnm-hero-lede">
        What we collect, why, how long we keep it, and what you can tell us to do
        about it. Written to be read rather than to be survived.
      </p>
      <p class="onnm-caption">
        Version ${PRIVACY_VERSION} ·
        <a href="/terms" data-link>Terms of use</a>
      </p>

      <div class="onnm-terms-notice">
        <h3>The short version</h3>
        <p>
          Your radiograph is analysed and then forgotten, unless you tick the sharing
          box for that specific image. We never store your IP address. We do not use
          advertising or analytics cookies, and we never sell anything. You can see
          everything we hold on your account page, delete a shared image yourself until
          a reviewer approves it, and close your account whenever you want.
        </p>
      </div>

      <div class="onnm-terms-clauses">${renderSections(OPERATOR)}</div>
    </section>`,
  );

  return null;
}
