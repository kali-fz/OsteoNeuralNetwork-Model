# GRC Implementation Requirements: OsteoNeuralNetwork-Model

## Document control

| Field | Value |
|---|---|
| Document | GRC Implementation Requirements — OsteoNeuralNetwork-Model (ONNM) |
| Version | 2.0 |
| Status | **Draft for legal review — not yet signed off** |
| Owner | kali-fz (project lead) |
| GRC contributor | Yaso-cyber |
| Last updated | 2026-08-24 |
| Applies to | Repository `OsteoNeuralNetwork-Model`, the hosted Streamlit demo, the `onnm-community` Cloudflare Worker, and the `onn-model` D1 database |
| Review cycle | Quarterly, and on any change to intended purpose, dataset, hosting, or user population |

> **This document is not legal advice.** It is an engineering-side compliance register written
> to be checked by a qualified professional. Every statutory citation, classification and date
> below must be verified with a solicitor experienced in UK/EU medical device regulation and
> data protection before it is relied on. Regulatory timelines in particular have been subject
> to amendment; confirm each one against the current published text rather than this file.

---

## 0. How to read this document

Sections 1–14 state what applies and why. **Section 15 is the operative part**: a prioritised
register of the gaps that actually exist in this project today, each with the evidence that
establishes it. If time is short, read Section 1 and Section 15.

Claims about the system are grounded in the repository as of 2026-08-24 and cite the file that
evidences them. Where this document contradicts an earlier draft, Appendix A records what
changed and why.

---

## 1. System description and scope

Accurate system description is the foundation of every classification below. A compliance
document describing the wrong device, or the wrong dataset, provides no protection and may
itself be evidence of inadequate governance.

### 1.1 What the system actually is

*   **Clinical task:** three-class triage of **primary bone tumours** on **plain 2D
    radiographs** — `normal` / `benign` / `malignant`. It is **not** an osteoporosis tool, and
    it does not process CT or MRI. (`MODEL_CARD.md`, `README.md`)
*   **Architecture:** DenseNet-121, ImageNet-pretrained, 3-class head, 256 px grayscale input,
    MONAI transform chain, temperature-scaled probabilities, Grad-CAM on `features.denseblock4`.
*   **Stated intended purpose:** research prototype for researchers studying explainable
    medical imaging. Explicitly out of scope: diagnosis, screening, paediatric use, CT/MRI,
    chest/skull/spine films, post-operative films with hardware, and "any deployment where the
    output reaches a patient or clinician as a finding". (`MODEL_CARD.md`)
*   **Measured performance, which governs the risk position:** malignant recall **0.633**
    (95% CI 0.490–0.776) — roughly **one malignant film in three is missed**. Macro ROC-AUC
    0.893. Single-distribution (BTXRD only), **no external validation**.
*   **Explainability status:** Grad-CAM localisation is measured at **pointing game 0.0936**,
    mean IoU 0.0428, over 267 annotated test films, with **no chance baseline established**.
    The project's own honest characterisation is "measurable, and roughly at chance".
    (`MODEL_CARD.md`, `TODO.md`)

### 1.2 Deployment configurations

Two configurations exist and they carry materially different obligations:

*   **Local installation.** Inference and storage on the operator's own machine, Streamlit
    bound to loopback, no network service contacted.
*   **Hosted deployment.** The public app at `osteoneuralnetwork-model-*.streamlit.app`, with
    Google OIDC sign-in and a Cloudflare Worker + D1 backend. **This configuration transfers
    personal data to third parties** and is the one that drives most of this document.

### 1.3 Data inventory (hosted deployment)

| Data | Where held | Notes |
|---|---|---|
| Email address (normalised) | Cloudflare D1 `users` | Identifier |
| PBKDF2-HMAC-SHA256 password hash | Cloudflare D1 `users` | Password accounts only |
| Google `sub`, display name, profile photo URL | Cloudflare D1 `users` | Name/photo published only on explicit opt-in |
| User UUID, account + ToS timestamps | Cloudflare D1 `users` | |
| `signup_country`, `country_captured_at` | Cloudflare D1 `users` | Two-letter ISO country only; no IP or finer location stored |
| Uploaded radiograph pixels | Streamlit Cloud (in memory, inference) | Every upload reaches Streamlit's server |
| **256 px processed image, base64** | Cloudflare D1 `submissions.image_b64` | **Stored only on explicit per-image opt-in** (`shared = 1`) |
| Verdict, class probabilities, threshold, calibration flag, OOD flag/score | Cloudflare D1 `submissions` | Stored for every submission |
| `consent_at`, `origin_country`, triage bucket, review status, admin label, reviewer | Cloudflare D1 `submissions` | Review-loop metadata |
| Capture tokens (`token_hash`, `expires_at`, `used_at`) | Cloudflare D1 `location_capture_tokens` | Purged when used or expired (`worker.js:512`) |

**Special category data.** Radiographic images processed to infer the presence of a bone
tumour are data concerning health under **UK GDPR Article 4(15)** and are special category data
under **Article 9(1)**, regardless of de-identification, because they remain linked to an
authenticated account.

### 1.4 Third-party recipients

| Party | Role | What it receives |
|---|---|---|
| **Streamlit Community Cloud** | Hosting / processor | **Every uploaded image**, because inference runs on their infrastructure |
| **Google** | Identity provider | That the user signed in; returns email, `sub`, name, photo. Receives no images |
| **Cloudflare** (Workers, D1) | Backend / processor | Accounts, submission records, opted-in images, country codes |
| **GitHub** | Source and checkpoint hosting | Public code; model weights via Releases |

All four operate internationally, so **international transfer** rules apply.

### 1.5 Technical controls already implemented

These are genuine mitigations and should be cited as such in any assessment:

*   Per-image sharing consent, **off by default**, asked separately per image; an image
    arriving without `shared = 1` is discarded rather than trusted (`worker.js:706–709`).
*   Deployment caps that evidence "test deployment, not product launch": `MAX_USERS = 500`,
    `MAX_SUBMISSIONS_PER_USER_PER_DAY = 50`, `MAX_TOTAL_BYTES_STORED = 200 MB`,
    `MAX_IMAGE_B64_BYTES ≈ 450 KB` (`worker.js:37–42`).
*   Bearer-key authentication on **every** data route; separate `API_KEY` and `ADMIN_KEY` so
    the deployed app cannot approve its own training data.
*   Admin pinned to a single account, enforced in three places (schema CHECK constraint,
    Worker header, UI gate).
*   DICOM PII stripping, UUID filenames, EXIF-free re-encode; country recorded at
    country-level only, captured via a one-use browser token so no IP is stored
    (`migrations/0004`, `0006`).
*   Human review gate before any community submission reaches training, enforced in four
    places, with a trigger making "hotdog, benign" unsayable.
*   Guarded model promotion with a version ledger (`ONN.md`, `model_versions.json`); a
    regressed run is recorded as `held` and the previous checkpoint keeps serving.
*   Published legal notices: Terms of Service, Privacy Policy, Medical Disclaimer, Cookie
    Notice (`src/legal.py`).

---

## 2. Medical device regulation (UK and EU)

### 2.1 Qualification as a medical device

Under the **Medical Devices Regulations 2002 (SI 2002/618)** as amended, and **Regulation (EU)
2017/745 (EU MDR)** for the EU market, software qualifies as a medical device where its
intended purpose includes diagnosis, prevention, monitoring, prediction, prognosis or treatment
of disease. Software intended to flag suspected primary bone tumours for human review falls
within that definition **once it is used to inform decisions about identifiable patients**.

### 2.2 The "Research Use Only" position — what it does and does not do

The project labels itself a research prototype with no MHRA, CE or FDA clearance, and states
this in the README, the model card, the app disclaimer and the Terms of Service. **This is
appropriate, necessary and worth preserving**, but its limits must be understood:

*   RUO labelling is a statement of **intended purpose**. It is effective while the intended
    purpose is genuinely research and the software is not put into clinical service.
*   It provides **no protection** if a treating clinician in fact uses the output to influence
    the care of an individual patient. Regulators assess actual use and the manner of
    presentation, not only the label.
*   The hosted public demo is the principal exposure. It is reachable by anyone with a Google
    account, including clinicians, and the software cannot verify who is using it or why.
*   **Practical consequence:** the disclaimers must remain unavoidable at the point of use, and
    the project should not market or describe the tool in terms that imply clinical utility.

### 2.3 Classification if the intended purpose ever changes

If ONNM is ever placed on the market or put into service for a clinical purpose:

*   **UK:** UK MDR 2002 classification rules apply. Diagnostic software informing clinical
    management would not qualify as Class I; a formal **conformity assessment by a UK Approved
    Body**, **UKCA marking**, and a **Quality Management System** would be required. The UK is
    reforming its medical device framework, so the applicable route and transitional
    arrangements must be confirmed with the MHRA at the time.
*   **EU:** under **EU MDR Rule 11** (software providing information used for diagnostic or
    therapeutic decisions), software of this kind is **Class IIa at minimum**, rising to IIb or
    III where decisions may cause serious deterioration or death. Given that the target
    condition includes malignancy, a higher class is plausible and must be assessed, not
    assumed.
*   **Standards that would become engaged:** ISO 13485 (QMS), ISO 14971 (risk management),
    IEC 62304 (software life cycle), IEC 82304-1 (health software), ISO/IEC 42001 (AI
    management systems), and BS 30440 (validation framework for AI in healthcare).
*   **MHRA AI Airlock** — the regulator's sandbox for AI medical devices — is worth
    investigating before any clinical route is attempted.

### 2.4 Evidence that would be required

The current evidence base would not support a conformity assessment. In particular: single
dataset, no external validation, malignant recall 0.633, no clinical evaluation, no defined
operating point, and no post-market surveillance plan. **This is not a criticism of a research
prototype — it is the reason the research label must be defended.**

---

## 3. EU AI Act (Regulation (EU) 2024/1689)

### 3.1 Territorial reach

The AI Act applies to providers established outside the EU where the system is placed on the EU
market **or where the output produced by the system is used in the EU**. A publicly accessible
hosted demo is reachable from the EU, so reach must be assessed rather than assumed absent.

### 3.2 Risk tier — how high-risk status would arise

**Correction to the common assumption:** a medical device does **not** become high-risk via
**Annex III**. Annex III lists stand-alone high-risk use cases (biometrics, education,
employment, essential services, law enforcement and so on). Medical devices become high-risk
under **Article 6(1)**, because they are products covered by the EU harmonisation legislation
listed in **Annex I** and are required to undergo third-party conformity assessment under that
legislation.

The practical consequence is significant: **the high-risk classification is triggered by, and
travels with, the medical device classification.** While ONNM is genuinely research-only and
not a device placed on the market, Article 6(1) is not engaged.

### 3.3 Research exclusion

**Article 2** excludes AI systems and models specifically developed and put into service for
the sole purpose of scientific research and development, and excludes research, testing and
development activity prior to placing on the market. Testing in real-world conditions is not
covered by that exclusion. **This exclusion is currently the project's principal AI Act
position and should be documented deliberately rather than relied on implicitly.**

### 3.4 Open-source limitation

The AI Act's free and open-source exemption **does not apply to high-risk AI systems**, nor to
prohibited practices or systems with transparency obligations. Publishing under Apache-2.0
would therefore not exempt ONNM if it became high-risk.

### 3.5 Timelines

Verify all dates against the current published text; the phased application has been subject to
amendment proposals.

| Milestone | Date |
|---|---|
| Entry into force | 1 August 2024 |
| Prohibited practices, AI literacy | 2 February 2025 |
| GPAI model obligations, governance | 2 August 2025 |
| **Annex III high-risk obligations** | 2 August 2026 |
| **Article 6(1) / Annex I embedded product obligations** | **2 August 2027** |

Because a medical device is an Annex I product, the relevant deadline would be **2027**, not
2026. Do not plan against the wrong date in either direction.

### 3.6 Obligations that would attach

Risk management system; data and data governance; technical documentation; automated
record-keeping and logging; transparency and instructions for use; **human oversight designed
into the interface**; accuracy, robustness and cybersecurity; quality management system;
conformity assessment; registration; post-market monitoring; serious incident reporting.

---

## 4. UK data protection (UK GDPR, DPA 2018, Data (Use and Access) Act 2025)

### 4.1 Controller identification

The Privacy Policy states that "the person or organization running the deployment is the data
controller/operator". For the hosted demo, **that is the project lead personally**. This should
be stated explicitly rather than left as a general formula, because it determines who carries
the statutory duties and who must be named in notices and to the ICO.

### 4.2 Lawful basis and Article 9 condition

Two things are required and are commonly conflated: an **Article 6** lawful basis and a
separate **Article 9(2)** condition for special category data.

*   **Article 6:** legitimate interests (Art 6(1)(f)) is defensible for research, supported by
    a documented Legitimate Interests Assessment. Consent (Art 6(1)(a)) is available but
    fragile: it is withdrawable at any time, and withdrawal would require removing the data
    from training sets and arguably from derived model weights — a serious operational hazard.
*   **Article 9:** the likely condition is **Art 9(2)(j)** — scientific research purposes —
    read with **DPA 2018 Schedule 1 Part 1 paragraph 4** and the **Article 89(1)** safeguards
    (data minimisation, pseudonymisation, and measures ensuring the processing does not permit
    decisions about specific individuals).
*   **Caution:** relying on the research condition requires the processing genuinely to be
    scientific research and to meet the Art 89 safeguards. A public demo open to the world is
    a weaker fit than a defined research protocol with an approved population.

### 4.3 Data (Use and Access) Act 2025

The DUAA 2025 introduces flexibilities relevant here — including provision for **broad consent
to areas of scientific research** and a clearer position on the compatibility of reusing data
for research purposes. **Commencement is phased**, so each provision relied on must be checked
for whether it is actually in force at the time of reliance, and ICO guidance updated under the
Act should be followed rather than pre-Act commentary.

### 4.4 Anonymisation versus pseudonymisation

Stripping DICOM headers, regenerating filenames and re-encoding to remove metadata produces
**pseudonymised** data, which remains fully within scope of UK GDPR. It is not anonymisation,
for three reasons the project already recognises:

1.  Records remain linked to an authenticated account.
2.  De-identification **cannot remove identifiers burned into image pixels** — the Terms place
    that duty on the user, which is a reasonable allocation but not a technical control.
3.  Medical images may be re-identifiable from their content.

The Privacy Policy already states this correctly. It must not be softened.

### 4.5 DPIA — mandatory, and currently absent

A **Data Protection Impact Assessment is required under Article 35**. This processing meets
multiple ICO screening criteria simultaneously: special category and health data, innovative
technology, use of AI, and processing that could result in physical harm. **No DPIA exists in
this repository.** This is the single most significant documentary gap.

### 4.6 Records of Processing Activities

**Article 30** requires a ROPA. The Art 30(5) small-organisation derogation does **not** apply
where the processing involves special category data. **No ROPA exists.**

### 4.7 Data subject rights — operational readiness

Rights of access, rectification, erasure, restriction, portability and objection apply.
Article 89 provides derogations for research, but they are conditional and must be
individually justified.

*   **Current gap:** per-user scan deletion in the UI is an **open, unimplemented item**
    (`TODO.md`, "Medium — app & delivery"). Erasure currently requires manual operator action
    against D1.
*   A documented, tested request-handling procedure is required, with a named contact and a
    one-month response deadline.

### 4.8 Storage limitation and retention

The Privacy Policy states records remain until the operator deletes them, and that operators
"should define a documented retention period". **No retention schedule exists.** Only capture
tokens are purged automatically. A defined retention period per data category, with an
automated or scheduled deletion mechanism, is required.

### 4.9 International transfers

Personal data reaches Streamlit, Google, Cloudflare and GitHub, all operating internationally.
Where transfers leave the UK, a transfer mechanism is required — **UK IDTA**, or the **UK
Addendum to the EU SCCs**, supported by a **Transfer Risk Assessment** — unless covered by
adequacy. **No transfer documentation exists.**

### 4.10 Processor arrangements

**Article 28** requires a written contract with each processor containing specified terms.
Standard terms of service for free tiers may not satisfy Article 28. Each provider's data
processing terms must be obtained, reviewed, and recorded.

### 4.11 Transparency

Privacy information must be concise, intelligible and accessible, and must be provided at the
point of collection. The existing Privacy Policy is unusually good — specific, honest about the
hosted configuration, and clear that images reach Streamlit. Two defects:

*   The **"This really is a radiograph" dispute button tells the user "Sends the image to a
    human reviewer", which is false when the row carries no image** (`TODO.md`, traced end to
    end 2026-08-23). Beyond the engineering defect, a false statement about what happens to a
    user's data is a transparency failure and potentially a misleading practice.
*   The ICO expects a named controller and contact details. "The maintainers through the
    repository's published contact channels" is unlikely to suffice for the hosted service.

### 4.12 Children

The service is not directed at children and paediatric use is out of scope. There is currently
**no age assurance**. Given Google sign-in is the gate, the practical control is the Terms plus
Google's own account age rules. The **Age Appropriate Design Code** should be reviewed if the
service is ever likely to attract under-18s.

### 4.13 Security of processing

**Article 32** requires appropriate technical and organisational measures. See Section 8.

### 4.14 Cookies and electronic communications (PECR)

The **Privacy and Electronic Communications Regulations 2003 (PECR)** govern storage of and
access to information on a user's device, separately from UK GDPR. Consent is required unless
the cookie is **strictly necessary** for a service the user has requested.

The project's position — no advertising, analytics, tracking pixels or cross-site profiling,
with only Streamlit's technically necessary session and WebSocket tokens — falls within the
strictly-necessary exemption, so **no cookie banner is required on the current build**
(`src/legal.py`, Cookie Notice). Two conditions attach:

*   The exemption is lost the moment any analytics, error-reporting, embedded media or
    third-party script is added. Adding one requires a consent mechanism, not merely a notice.
*   The Cookie Notice already says this and instructs the operator to update it. That
    instruction should be treated as a controlled change, not a suggestion.

---

## 5. Research ethics and clinical governance

This section was absent from the earlier draft and is a material omission for a UK project
processing health data.

*   **Research ethics approval.** Research involving NHS patients, staff, or data, or requiring
    access to identifiable health records, generally requires **HRA approval and a Research
    Ethics Committee opinion**. Work confined to a public, licensed, already-collected research
    dataset (BTXRD) may not, but **the community loop — collecting new radiographs from
    members of the public via a web app for the purpose of improving a model — is new data
    collection for research** and its status must be determined, not assumed.
*   **Institutional sponsorship.** If any contributor is acting in a university capacity, the
    institution may require sponsorship, its own ethics review, and its own DPIA. Two Master's
    students are named as collaborators in the README; their institutions' positions should be
    confirmed.
*   **Provenance and consent for uploaded images.** The Terms require the user to hold "all
    necessary authority and consent". This is the correct allocation but is unverifiable. The
    risk that a user uploads a real patient's radiograph without a lawful basis is live, and
    the controller cannot fully discharge it by contract alone.
*   **Publication.** Any publication of results derived from community submissions should state
    the consent basis and the review process.

---

## 6. Clinical safety standards (if the tool ever touches NHS care)

Also absent from the earlier draft. These apply to health IT deployed in NHS-facing contexts and
are contractual and regulatory expectations, not optional good practice:

*   **DCB0129** — Clinical Risk Management: its Application in the Manufacture of Health IT
    Systems. Requires a nominated **Clinical Safety Officer** (a registered clinician), a
    Clinical Risk Management File, a hazard log, and a Clinical Safety Case Report.
*   **DCB0160** — the equivalent standard for the deploying organisation.
*   **NHS DTAC** — Digital Technology Assessment Criteria, the baseline entry gate for NHS
    procurement, covering clinical safety, data protection, technical security,
    interoperability and usability.
*   **Current position:** none of these exist, which is consistent with a research prototype
    that is not NHS-facing. **They become blocking prerequisites the moment any NHS
    conversation starts.**

---

## 7. Intellectual property, dataset licensing and open source

### 7.1 BTXRD — the binding constraint

Training data is **BTXRD** (3,746 annotated radiographs), licensed **CC BY-NC-ND 4.0**. This is
more restrictive than commonly assumed and imposes three distinct limits:

*   **BY** — attribution required wherever the dataset or derivatives are used or described.
*   **NC** — **non-commercial only**. Any commercialisation, including a paid service, a
    commercial research contract, or a spin-out, is outside the licence and would require
    separate permission from the rights holder.
*   **ND** — **no derivatives may be redistributed**. The model card already draws the correct
    conclusion: **Grad-CAM overlays and case reports are derivative images and must remain
    local**. They must not be published in papers, README files, marketing material, issue
    threads, or the hosted app's public surfaces.

### 7.2 The unresolved question — are trained weights a derivative?

`TODO.md` records this as an open decision: "Whether to publish the weights openly. They derive
from BTXRD (CC BY-NC-ND 4.0), and whether trained weights are a 'derivative' under that licence
is unsettled."

**This is correctly identified and currently unresolved, yet weights are already published via
a GitHub Release and fetched by the app at runtime** (`src/checkpoint_fetch.py`, release
`v0.1.0`). The legal question is therefore not hypothetical — it is live. It requires a
determination, and if the answer is unfavourable, either permission from the BTXRD rights holder
or withdrawal of the public weights.

### 7.3 Repository licence — a concrete, immediately fixable defect

The README displays an **Apache-2.0 badge**, but **there is no `LICENSE` file in the
repository**. Without it, the default position is exclusive copyright: contributors and users
have no clear grant, the badge is potentially misleading, and Apache-2.0's patent grant and
`NOTICE` mechanics are not actually in force. **Add the full Apache-2.0 text as `LICENSE`, or
correct the badge.**

### 7.4 Third-party components

*   **DenseNet-121 ImageNet-pretrained weights** (torchvision) — check the licence of the
    pretrained weights, not only the library.
*   **MONAI, PyTorch, Streamlit** and the wider dependency tree — an **SBOM** should be
    produced (see Section 8), which also serves licence-compliance review.
*   **Contributor terms.** With no `LICENSE`, no `CONTRIBUTING.md` and no DCO or CLA, inbound
    contribution rights are undefined.

### 7.5 Text and data mining

The UK TDM exception (CDPA 1988 s.29A) permits copying for computational analysis **for
non-commercial research only**, and requires lawful access. It does not authorise commercial
model training on copyrighted medical datasets. Since BTXRD is separately licensed NC, the same
boundary applies from two directions.

---

## 8. Cybersecurity and resilience

### 8.1 Legal drivers

*   **UK GDPR Article 32** — appropriate technical and organisational security measures.
*   **EU Cyber Resilience Act (Regulation (EU) 2024/2847)** — obligations for products with
    digital elements placed on the EU market. **Vulnerability and incident reporting
    obligations apply from September 2026**, with the main body of obligations following in
    December 2027. Free and open-source software developed outside a commercial activity is
    treated differently; that carve-out must be assessed rather than assumed.
*   **Authorised testing only.** Any penetration testing must be conducted solely against
    infrastructure the project is authorised to test. Testing Streamlit, Google or Cloudflare
    infrastructure without written authorisation risks liability under the **Computer Misuse
    Act 1990**.

### 8.2 Open security finding — unremediated credential exposure

`TODO.md` records, under "Low — housekeeping":

> "Rotate the credentials. The Cloudflare API token and R2 token were pasted into a chat
> transcript, and `ONNM_COMMUNITY_KEY` / `ONNM_ADMIN_KEY` appeared in screenshots during setup."

**This is a known credential exposure that has not been remediated, and it is classified as
low priority.** That classification is wrong. `ONNM_ADMIN_KEY` unlocks the review queue,
approvals and export — that is, access to every shared radiograph in D1. Assessment:

*   **Rotate all four secrets now**, and record the rotation date.
*   Determine whether the exposure requires a **personal data breach assessment** under Article
    33. If unauthorised access to health data cannot be ruled out, the **72-hour ICO
    notification clock** is engaged from the point of awareness.
*   The positive finding is that a prior audit confirmed no secret value ever entered git
    history, and `.env` and `.streamlit/secrets.toml` are gitignored — that audit should be
    repeated and its result recorded.

### 8.3 Security posture — assessment

Strengths: bearer auth on every route; privilege separation between app and admin keys;
parameterised queries; request and storage caps; loopback binding by default locally;
Cloudflare platform encryption at rest; admin constrained by schema, header and UI.

**Training-data integrity deserves explicit credit as a security control.** The community loop
is an untrusted input path into a model that people may act on, which makes **data poisoning** a
genuine attack, not a theoretical one. The mitigations already built are: a three-bucket triage
applied on arrival and re-applied on user feedback (`valid_bone` / `misc` / `contradiction`);
a mandatory human review gate enforced in four places before any label reaches training; a
database trigger making contradictory bucket/label pairs unrepresentable; per-user daily
submission caps; and guarded model promotion that holds a regressed run rather than shipping it.
Together these mean a malicious uploader must defeat a human reviewer, not merely a filter.
**This should be described in the DPIA as a mitigating control.**

Gaps to address:

*   **No `SECURITY.md`** and no coordinated vulnerability disclosure policy. The CRA expects a
    documented reporting channel, and it is basic hygiene for a public repository.
*   **No SBOM.** Required in substance by the CRA and useful for licence review.
*   **No dependency vulnerability scanning** recorded (Dependabot or equivalent).
*   **No encryption at rest added by ONNM** beyond the platform default; local SQLite is
    unencrypted by design, with full-disk encryption left to the operator.
*   **Key-person and segregation risk:** a single hardcoded admin account is both the reviewer
    and the operator. There is no second pair of eyes, and no documented process if that
    account is unavailable.
*   **Google OAuth consent screen remains in testing mode** (`TODO.md`), capping listed users
    at 100 and showing an "unverified app" interstitial — relevant to both security posture and
    user trust.
*   **No logging or audit retention policy**, and Worker observability sampling is set to 10%.

---

## 9. Product liability and professional liability

*   **UK — Consumer Protection Act 1987.** Strict liability for damage caused by defective
    products. Whether standalone software is a "product" under the 1987 Act has historically
    been contested; the safer working assumption is that it may be.
*   **EU — the new Product Liability Directive (EU) 2024/2853.** This **expressly brings
    software, including AI systems, within the definition of a product**, covers defects
    arising from software updates and machine learning, and eases the burden of proof for
    claimants in technically complex cases. Member states transpose by **9 December 2026**.
    This materially increases exposure for any EU placing on the market and supersedes the
    older directive that UK commentary often still reflects.
*   **Clinical negligence.** If a clinician relies on the tool and a patient is harmed, claims
    may be brought against the clinician; the developer's exposure depends on how the tool was
    presented and what warnings were given. **The strongest protection is the accuracy and
    prominence of the limitations, especially the 0.633 malignant recall.**
*   **Enforceability of disclaimers.** The Terms already acknowledge that an absolute
    zero-liability waiver may be unenforceable. Under the **Unfair Contract Terms Act 1977**
    and the **Consumer Rights Act 2015**, liability for death or personal injury caused by
    negligence **cannot be excluded**. Given the clinical domain, the disclaimer should be
    read as risk communication, not as a shield.
*   **Insurance.** No professional indemnity or public liability cover is recorded. This should
    be considered before any use beyond personal research, and is normally a prerequisite for
    institutional collaboration.

---

## 10. Transparency, explainability and human oversight

### 10.1 The Grad-CAM problem — an honest statement is required

The earlier draft asserted that deploying Grad-CAM heatmaps mitigates liability by providing
interpretable reasoning. **On the evidence in this repository, that reasoning is unsafe and
must not be relied upon.**

Grad-CAM localisation is measured at **pointing game 0.0936, mean IoU 0.0428**, with **no chance
baseline established** — and a lesion box covering roughly a tenth of the frame would be hit
roughly a tenth of the time by accident. The project's own conclusion is that this "does not yet
support the claim 'the model looks at lesions'".

An explanation that does not actually explain, but is presented to a user as evidence, is worse
than no explanation: it invites unjustified confidence and could itself be characterised as
misleading. The correct position is the one already taken in the Medical Disclaimer — **"Grad-CAM
shows model attention and does not prove pathological localization or reasoning"** — and it must
be preserved in substance anywhere heatmaps are shown.

**Action:** establish the chance baseline (score a randomly-initialised model identically). Until
that comparison exists, do not claim localisation in any publication, README, or user-facing text.

### 10.2 Human oversight

The community loop already enforces human review before any label reaches training, in four
places. That is a genuine and well-implemented oversight control and should be documented as
such. Note that oversight of the *training loop* is not the same as oversight of the *inference
output* shown to a user, which is unsupervised by design.

### 10.3 Automated decision-making

**Article 22** restricts solely automated decisions producing legal or similarly significant
effects. The system produces a classification but does not itself make a decision about a
person, and outputs are explicitly not to be used for care decisions. **The position is
defensible while the research framing holds**, and would require reassessment if it did not.

### 10.4 AI transparency to users

The app should continue to make clear that the user is interacting with an AI system, what its
measured limitations are, and what happens to their data. The current notices do this well.

---

## 11. Equality, bias and health inequality

Absent from the earlier draft, and material for a health AI system.

*   **Equality Act 2010.** Indirect discrimination is possible where a model performs
    differently across protected characteristics. The **Public Sector Equality Duty** would
    apply to any NHS or public-body deployment.
*   **Known bias exposure:** the model is trained and evaluated on **BTXRD only**, with **no
    external validation**. Performance by age, sex, ethnicity, body habitus, scanner
    manufacturer or acquisition protocol is **unknown and unmeasured**. The model card states
    performance elsewhere "should be presumed worse".
*   **Documented failure mode:** false positives on complex normal anatomy (pelvis, hip, growth
    plates), including a normal pelvis at 59.6% and a normal femur at 69.8%. Growth plates are
    an age-linked feature, which is a demographic signal.
*   **Class imbalance:** only ~342 malignant images exist in total (~49 in test), so every
    malignant metric carries a wide interval by construction.
*   **Action:** record demographic performance as a known, unquantified gap; do not publish
    subgroup claims without subgroup data; run the stratified per-anatomy report already
    scripted (`scripts/stratified_report.py`) and record the result.

### 11.1 Accessibility of the interface

A distinct obligation from bias, and easily overlooked. Under the **Equality Act 2010** a
service provider owes a duty to make **reasonable adjustments** for disabled users, and that
duty is **anticipatory** — it does not wait for a complaint.

*   The target standard is **WCAG 2.2 Level AA**. For public-sector bodies the Public Sector
    Bodies (Websites and Mobile Applications) Accessibility Regulations 2018 make it a hard
    requirement with a mandatory accessibility statement; ONNM is not currently in that
    category, but any NHS or university deployment would bring it in.
*   **Specific risk in this app:** the primary output is **colour-coded and image-based** — a
    Grad-CAM heatmap over a greyscale radiograph. Colour alone must not be the only carrier of
    meaning, and the textual verdict, probabilities and limitations must remain available to
    screen readers independently of the overlay.
*   **Action:** run an accessibility audit against WCAG 2.2 AA alongside the planned UI
    redesign (`TODO.md`, "Redesign the UI"), which is the cheapest point at which to fix it.

---

## 12. Supply chain and third-party governance

*   Obtain and record **data processing terms** for Streamlit Community Cloud, Cloudflare and
    Google; confirm each satisfies Article 28.
*   **Free-tier risk.** The project runs deliberately at zero cost. Free tiers typically carry
    weaker contractual commitments, no SLA, and a unilateral right to change or withdraw
    service. Record this as an accepted risk, including the consequence that the service may
    become unavailable without notice.
*   **Streamlit Community Cloud receives every uploaded image.** This is the highest-exposure
    third-party relationship and is correctly disclosed in the Privacy Policy; it must also be
    reflected in the DPIA and ROPA.
*   Record sub-processor positions and international transfer routes for each provider.

---

## 13. Incident management

*   **Personal data breach:** assess against Article 33; notify the ICO within **72 hours** of
    awareness where the risk threshold is met; notify affected individuals under Article 34
    where the risk is high. **No incident response plan exists.**
*   **Clinical or safety incident:** if the tool ever becomes a device, vigilance and serious
    incident reporting to the MHRA apply, and under the AI Act serious incidents are reportable
    for high-risk systems.
*   **Security incident:** CRA vulnerability reporting from September 2026 where in scope.
*   **Immediate application:** the credential exposure in Section 8.2 should be run through this
    process as the first test of it.

---

## 14. Records, audit and evidence

Compliance is demonstrated by records. The following should exist and be retained:

| Artefact | Status |
|---|---|
| DPIA | **Drafted, unsigned** — `compliance/DPIA.md` |
| ROPA | **Drafted, unsigned** — `compliance/ROPA.md` |
| Legitimate Interests Assessment | **Missing** — required to complete the DPIA §4.1 |
| Transfer Risk Assessment + IDTA/Addendum | **Missing** |
| Retention schedule | **Drafted, values not set** — `compliance/ROPA.md` §E |
| Incident response plan + incident log | **Drafted, unsigned** — `compliance/INCIDENT_RESPONSE.md` |
| `LICENSE` file | **Exists** — canonical Apache-2.0 text |
| `SECURITY.md` + disclosure policy | **Exists** |
| `CONTRIBUTING.md` + DCO | **Exists** |
| SBOM | **Missing** |
| Model version ledger | **Exists** (`ONN.md`, `model_versions.json`) |
| Published ToS / Privacy / Disclaimer / Cookie notice | **Exists** (`src/legal.py`) |
| Test evidence and metrics with confidence intervals | **Exists** (`reports/<run>/`) |
| Secret-in-git-history audit | **Exists**, dated 2026-08-23; repeat and re-record |

---

## 15. Gap register

Prioritised. **P1 = act now**, P2 = before any expansion of use, P3 = before any clinical or
commercial route.

| # | Pri | Gap | Evidence | Action |
|---|---|---|---|---|
| 1 | **P1** | Exposed credentials not rotated; classified "Low" | `TODO.md` housekeeping | **STILL OPEN.** Rotate all four secrets; assess Art 33 breach duty; record date. Logged as incident 001 in `compliance/INCIDENT_RESPONSE.md` |
| 2 | **P1** | No DPIA despite mandatory Art 35 triggers | Repo-wide search: absent | **DRAFTED** — `compliance/DPIA.md`. Awaiting controller decisions and signature |
| 3 | **P1** | No `LICENSE` file, Apache-2.0 badge displayed | Repo root listing; `README.md` badge | **CLOSED** — canonical Apache-2.0 text added as `LICENSE` |
| 4 | **P1** | Dispute button states images go to a reviewer when none was stored | `TODO.md`, traced 2026-08-23 | **CLOSED** — `render_rejection_dispute` now takes a required `shared` argument and states what actually happened |
| 5 | **P1** | Weights published while "are weights a derivative of CC BY-NC-ND?" is unresolved | `TODO.md` open decision; release `v0.1.0` live | **STILL OPEN.** Obtain determination; seek permission or withdraw |
| 6 | **P2** | No ROPA (Art 30(5) derogation unavailable) | Repo-wide search: absent | **DRAFTED** — `compliance/ROPA.md`. Controller fields outstanding |
| 7 | **P2** | No retention schedule; no automated erasure of submissions | `src/legal.py`; only tokens purged | **PARTLY** — schedule drafted at `compliance/ROPA.md` §E; periods not set and deletion not implemented |
| 8 | **P2** | No user-facing deletion; erasure is manual | `TODO.md` open item | Implement per-user scan deletion; document DSR procedure |
| 9 | **P2** | No Art 28 processor terms recorded for Streamlit/Cloudflare/Google | Not present | Obtain, review, record |
| 10 | **P2** | No transfer mechanism or TRA documented | Not present | Complete IDTA/Addendum + TRA |
| 11 | **P2** | Controller not named; contact is "repository channels" | `src/legal.py` | Name the controller and a contact route |
| 12 | **P2** | No `SECURITY.md`, no disclosure policy, no SBOM, no dependency scanning | Repo root | **PARTLY** — `SECURITY.md` and `CONTRIBUTING.md` added. SBOM and dependency scanning still outstanding |
| 13 | **P2** | No incident response plan | Not present | **DRAFTED** — `compliance/INCIDENT_RESPONSE.md`, with gap #1 logged as incident 001 |
| 14 | **P2** | Grad-CAM presented as explanation while measured near chance | `MODEL_CARD.md`, `TODO.md` | Establish chance baseline; make no localisation claim until then |
| 15 | **P2** | Research-ethics status of community data collection undetermined | Section 5 | Determine HRA/REC and institutional position |
| 16 | **P2** | Single admin account is sole reviewer and operator | `TODO.md`, schema CHECK | Document key-person risk; plan continuity |
| 17 | **P3** | No demographic or subgroup performance data | `MODEL_CARD.md` | Run stratified report; record as known gap |
| 18 | **P3** | No external validation | `MODEL_CARD.md` | Required before any clinical claim |
| 19 | **P3** | No QMS, ISO 13485/14971/IEC 62304, no clinical evaluation | Not present | Required for any device route |
| 20 | **P3** | No DCB0129 Clinical Safety Officer or safety case | Section 6 | Required before NHS-facing use |
| 21 | **P3** | No professional indemnity insurance | Section 9 | Consider before non-personal use |
| 22 | **P3** | Google OAuth consent screen in testing mode | `TODO.md` | Decide publish vs remain capped at 100 users |
| 23 | **P2** | No accessibility audit; verdict is colour- and image-led | Section 11.1 | Audit against WCAG 2.2 AA during the planned UI redesign |
| 24 | **P3** | No DPO appointment decision recorded | Section 16 | Assess Art 37 trigger and record the decision either way |
| 25 | **P1** | **The test suite wrote to the production database.** `backend.create_user` routes to live D1 whenever a community key is configured and ignores the `path` argument, so `pytest` on a credentialled machine created real accounts. 3 `@example.com` accounts had reached production D1 | `src/backend.py:49–60`; 4 failures in `tests/test_auth_database.py` reproduced at clean HEAD; D1 count confirmed 2026-08-24 | **CLOSED** — autouse fixture in `tests/conftest.py` clears the community environment and resets the memoised client; `tests/test_backend_isolation.py` pins it; the 3 accounts were purged from D1 on 2026-08-24 after a backup, `quick_check` ok, no real account or submission affected |
| 26 | **P2** | `ONNM_COMMUNITY_URL`, `ONNM_COMMUNITY_KEY` and `ONNM_ADMIN_KEY` are set as **OS-level** environment variables on a development machine, so any script, shell or ad-hoc command talks to production by default | Confirmed 2026-08-24: present in a fresh subprocess with no `.env` loaded | Scope the variables to the app's launch rather than the user environment. The test fixture closes the test path only; every other path is still live |

---

## 16. Roles and responsibilities

| Role | Holder | Responsibility |
|---|---|---|
| Data controller (hosted) | kali-fz | All UK GDPR controller duties; ICO contact |
| Project lead / maintainer | kali-fz | Intended purpose, releases, model promotion |
| GRC | Yaso-cyber | This register, DPIA, ROPA, policy review |
| Admin / reviewer | single pinned account | Review gate, approvals, export — **key-person risk** |
| Clinical Safety Officer | **Unfilled** | Required only if NHS-facing (DCB0129) |
| Data Protection Officer | **Not appointed** | Assess whether Art 37 requires one; large-scale special category processing is a trigger |

---

## 17. Review schedule

Review quarterly, and immediately on any of: change of intended purpose; new dataset; new
hosting provider; removal of the research framing; any approach to clinical or NHS use; any
commercial interest; any personal data breach; publication of new UK or EU guidance affecting
medical device AI.

---

## Appendix A — corrections applied to the earlier draft

Recorded so the changes are visible to whoever reviews this next.

| Earlier statement | Correction | Why it matters |
|---|---|---|
| "predict or diagnose **osteoporosis**" | Triage of **primary bone tumours** (normal/benign/malignant) on plain radiographs | A compliance document describing the wrong clinical purpose supports no classification and undermines the file's credibility |
| Training on **Stanford MURA**, "prohibits commercial use" | Training on **BTXRD**, licensed **CC BY-NC-ND 4.0** | The real licence is stricter: it also forbids redistributing derivatives, which reaches Grad-CAM overlays and possibly the weights |
| High-risk **via Annex III** | High-risk arises via **Article 6(1) + Annex I** for medical devices | Annex III is the standalone list; the mechanism determines both the trigger and the deadline |
| High-risk obligations enforceable **August 2026** | **2 August 2027** for Annex I embedded products; Aug 2026 is the Annex III date | Planning against the wrong date in either direction is a real cost |
| XAI/Grad-CAM **mitigates liability** | Grad-CAM here is measured **at roughly chance**; presenting it as reasoning is unsafe | The project's own measurements contradict the claim; asserting it could increase exposure |
| Consent should be avoided in favour of legitimate interests | Retained, but split into the required **Art 6 basis** and separate **Art 9(2)(j) condition** with Art 89 safeguards | Conflating the two is the most common UK GDPR error in health research |
| Liability under **CPA 1987** only | Added **PLD (EU) 2024/2853**, which expressly covers software and AI | The new directive materially changes EU exposure and is transposed by December 2026 |
| — | Added: clinical safety (DCB0129/0160, DTAC), research ethics and HRA, equality and bias, supply chain, incident management, records, and the gap register | These were absent and are the sections a regulator or reviewer would ask for first |
| — | Added on review: PECR and cookies (§4.14), interface accessibility under the Equality Act and WCAG 2.2 AA (§11.1), and training-data poisoning resistance as a named security control (§8.3) | Cookies are a separate regime from UK GDPR; accessibility is an anticipatory duty distinct from model bias; and the anti-poisoning controls are real mitigations that were going undocumented |

## Appendix B — instruments referenced

UK Medical Devices Regulations 2002 (SI 2002/618) · Regulation (EU) 2017/745 (EU MDR) ·
Regulation (EU) 2024/1689 (EU AI Act) · UK GDPR · Data Protection Act 2018 ·
Data (Use and Access) Act 2025 · Regulation (EU) 2024/2847 (Cyber Resilience Act) ·
Directive (EU) 2024/2853 (Product Liability) · Consumer Protection Act 1987 ·
Unfair Contract Terms Act 1977 · Consumer Rights Act 2015 · Equality Act 2010 ·
Computer Misuse Act 1990 · Copyright, Designs and Patents Act 1988 s.29A ·
DCB0129 / DCB0160 · NHS DTAC · ISO 13485 · ISO 14971 · IEC 62304 · IEC 82304-1 ·
ISO/IEC 42001 · BS 30440 · ICO Age Appropriate Design Code
