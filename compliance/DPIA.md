# Data Protection Impact Assessment: ONNM hosted deployment

**Status: DRAFT for GRC review, not signed off.** Prepared by the engineering side to be
completed and signed by the controller. Sections marked **[CONTROLLER]** require a decision
that only the controller can make. This is not legal advice.

| Field | Value |
|---|---|
| Assessment | DPIA under UK GDPR Article 35 |
| Subject | ONNM hosted deployment and community submission loop |
| Version | 0.2 (draft) |
| Prepared | 2026-08-29 |
| Previous version | 0.1, 2026-08-24 |
| Controller | **Khalid Faiz**, acting in a personal capacity as an independent researcher |
| Prepared by | Engineering, for GRC review (Yaso-cyber) |
| DPO consulted | No DPO appointed, see §8 |
| Signed off by | **Not yet signed** |
| Review | On any change of purpose, dataset, hosting, or user population; otherwise annually |

Related: `ROPA.md`, `INCIDENT_RESPONSE.md`, `TERMS_REVIEW_PACK.md`,
`../grc_compliance_prompt.md`.

> **What changed in 0.2.** Version 0.1 described inference running on a third-party Python
> hosting platform, which received every uploaded image. **That platform has been removed
> from the project entirely**; inference now runs in a Cloudflare container and no image
> leaves Cloudflare's infrastructure. Four risk rows (R1, R2, R7, R13) named "the Terms" as
> their mitigation at a time when no Terms gate existed; it exists now and is enforced
> server side. R6 is closed; R3 and R8 are substantially reduced with named actions
> remaining. **R14, unlawful content, is new**: it was absent from 0.1 and should not have
> been.

---

## 1. Why a DPIA is required

Article 35(1) requires a DPIA where processing is likely to result in a high risk to
rights and freedoms. Article 35(3)(b) makes it mandatory for processing **special category
data on a large scale**. Independently of scale, this processing meets several ICO
screening criteria at once:

| Criterion | Present? | Evidence |
|---|---|---|
| Special category / health data | **Yes** | Radiographs processed to infer presence of a bone tumour |
| Innovative technology | **Yes** | Deep neural network classifier |
| Use of AI to make inferences about people | **Yes** | Three-class malignancy triage |
| Risk of physical harm | **Yes** | A missed malignancy could contribute to delayed care |
| Vulnerable data subjects | **Possible** | Patients; contributors may upload third-party images |
| Combining datasets | **Yes** | Community submissions merged into a training corpus |

**Conclusion: a DPIA is required.** Two or more criteria alone indicate high risk.

Note on scale: the deployment is capped at 500 users, 50 submissions per user per day and
200 MB of stored images (`cloudflare/src/worker.js:37-42`), so "large scale" is arguable.
**The DPIA is required regardless**, because health data plus AI plus risk of physical
harm is sufficient on the ICO screening list.

---

## 2. Description of the processing

### 2.1 Nature

Two configurations exist. **This DPIA covers the hosted deployment**; a purely local
installation transfers nothing and is out of scope except where noted.

Flow for the hosted app:

1. Visitor reads the Terms of use and the Privacy notice, both reachable without an
   account, and ticks to agree. **Without that, no account can be created**: the Worker
   refuses to begin the OAuth flow.
2. User signs in with Google (OIDC). ONNM receives email, `sub`, display name, photo URL.
3. A signed-in browser posts a short-lived, one-use token directly to the Cloudflare
   Worker, which records **only a two-letter country code**. No IP address or finer
   location is stored (`cloudflare/migrations/0006_browser_country_capture.sql`).
4. User uploads a radiograph (DICOM, PNG, JPEG, BMP, TIFF). **Inference runs in a
   Cloudflare container addressed through a Durable Object. The image does not leave
   Cloudflare's infrastructure**, and it is not written to storage at this stage.
5. An out-of-distribution gate may reject the image before inference.
6. DenseNet-121 returns three-class probabilities; Grad-CAM produces a heatmap.
7. A submission record is written to Cloudflare D1. **The 256 px processed image is stored
   only if the user ticks the sharing box**, which is off by default and asked separately
   for every image (`worker.js:706-709`).
8. Shared submissions enter a human review queue visible to one pinned reviewer account.
   Only a human-approved label can reach training.
9. A rejected image is deleted automatically within seven days by a scheduled job.

### 2.2 Scope: data processed

See `ROPA.md` for the itemised register. In summary: account identifiers (email, Google
`sub`, UUID, display name, photo URL, password hash for legacy password accounts), Terms
acceptance timestamp and version, country code, model outputs (verdict, probabilities, OOD
flags), and, **on explicit per-image consent only**, a 256 px processed radiograph.

**Special category data:** the radiographs, and the inferred verdict, are data concerning
health (Art 4(15)) and therefore Article 9 data. The data remains **pseudonymised, not
anonymous**, because it is linked to an authenticated account.

### 2.3 Context

- **Data subjects:** registered users (researchers, students, curious members of the
  public), and, potentially and unverifiably, the **patients depicted in uploaded
  radiographs**, who are not users and have no relationship with the controller.
- **Relationship:** voluntary, no service dependency, users can stop at any time.
- **Expectations:** users are told plainly, before any account exists, that inference runs
  on Cloudflare, that sharing is separate and optional, and that an approved image cannot
  be untrained. This is unusually transparent for a project of this size and is a genuine
  mitigation.
- **Children:** **the Terms now require users to be 18 or over.** Paediatric use is out of
  scope. There is no age assurance beyond that declaration and Google's own account rules.
- **Prior concerns:** none recorded from data subjects.

### 2.4 Purposes

1. Authenticate users and maintain accounts.
2. Perform the inference the user requested and display the result.
3. Show the user their own submission history.
4. With explicit consent, retain a de-identified 256 px image for human review and
   possible inclusion in a research training set, to improve model accuracy.
5. Display country-level aggregate counts on the public homepage.
6. Secure and troubleshoot the deployment.

**Benefits:** research into explainable medical imaging; a documented, honest evaluation
of a model class often deployed with weaker evidence; a corpus of confirmed
out-of-distribution negatives that improves the safety gate.

---

## 3. Consultation

| Party | Status |
|---|---|
| Data subjects | **[CONTROLLER]** Not yet consulted. Consider a short notice inviting comment, or record why consultation is disproportionate |
| DPO | None appointed, see §8 |
| Processors (Cloudflare, Google) | Both publish Article 28 terms and a UK transfer mechanism as standard. **[CONTROLLER]** confirm acceptance and file copies, see R7 |
| Information security | Reviewed internally; see `../grc_compliance_prompt.md` §8 |
| Clinical input | **[CONTROLLER]** No clinician involved. Required before any claim of clinical utility |
| GRC review | **In progress**, see `TERMS_REVIEW_PACK.md` |
| ICO prior consultation (Art 36) | Not required unless a high residual risk cannot be mitigated, see §9 |

---

## 4. Necessity and proportionality

### 4.1 Lawful basis (Article 6): **[CONTROLLER] to confirm**

Version 0.1 proposed Article 6(1)(f), legitimate interests, throughout. **That position has
been revised.** The bases now recorded in `ROPA.md`, and stated to users in the Privacy
notice §4, are:

| Processing | Article 6 | Article 9 |
|---|---|---|
| Accounts | 6(1)(b) contract, the Terms | n/a |
| Running the scan | 6(1)(b) contract | 9(2)(a) explicit consent |
| Shared image, review and research | 6(1)(a) consent | 9(2)(a) explicit consent; 9(2)(j) for retention after withdrawal |
| Country and globe | 6(1)(f) legitimate interests | n/a |
| Contributor profile | 6(1)(a) consent | n/a |
| Security and logs | 6(1)(f) legitimate interests | n/a |

**Why the change.** The sharing tick box is granular, opt-in, off by default, asked per
image and withdrawable. It is consent in every observable respect, and recording something
else in the register would mean the interface and the paperwork disagreed. The objection to
consent, that a model cannot be untrained so withdrawal cannot be honoured, is answered by
**Article 7(3)**: withdrawal does not affect the lawfulness of processing already carried
out. Withdrawal stops future use and deletes the stored copy; the trained weights rest on
9(2)(j), with Art 17(3)(d) limiting erasure to that extent and no further.

Article 6(1)(b) for accounts is now genuinely available, which it was not in 0.1: it
requires a contract, and until the Terms gate shipped there was no moment at which anyone
agreed to anything.

### 4.2 Article 9 condition: **[CONTROLLER] to confirm**

Explicit consent, 9(2)(a), for collection and review. **Article 9(2)(j)**, scientific
research, read with **DPA 2018 Schedule 1 Part 1 paragraph 4** and the **Article 89(1)**
safeguards, for continued retention of approved images in the research corpus.

Article 89(1) safeguards in place: data minimisation (256 px, DICOM PII stripped, UUID
filenames, EXIF-free re-encode, country-level location only); pseudonymisation; and
technical measures ensuring the processing does not support decisions about specific
individuals (the model's output is explicitly not for clinical use).

**Risk to this position:** a public demo open to anyone is a weaker fit for "scientific
research" than a defined protocol with an approved population. A **research protocol
document** would materially strengthen it. See risk R9.

### 4.3 Necessity

| Purpose | Data used | Could less be used? |
|---|---|---|
| Authentication | Email, `sub` | No, required to isolate one user's history from another's |
| Inference | Image pixels | No, the image is the input |
| History | Submission metadata | No |
| Research corpus | 256 px image | **Already minimised**, original resolution is never stored; only 256 px, only on consent |
| Globe display | Country code | **Already minimised**, country only, deliberately chosen over coordinates |
| Contributor display | Name, photo | **Opt-in only**, off by default, reversible |

### 4.4 Proportionality

The design shows deliberate minimisation, and this should be recorded as evidence:

- Migration `0004` documents the explicit decision **not to store coordinates**, because a
  precise coordinate plus a timestamp plus a malignant verdict is jointly identifying even
  with names stripped. The schema is *incapable* of holding finer than country.
- Migration `0006` replaced server-side geolocation with a one-use browser token so that
  **no IP address is seen, logged or stored**.
- Sharing is off by default and asked per image; an image arriving without `shared = 1` is
  discarded rather than trusted, so an app-side bug cannot silently retain it.
- **Removing the external inference host** eliminated an entire processor that previously
  received every uploaded image, including images the user never chose to share.
- Deployment caps evidence a test deployment rather than a product launch. **The 200 MB
  storage ceiling is a control placed at the resource rather than at the identity**, so it
  bounds total exposure regardless of how many accounts an abuser creates.

---

## 5. Risks to individuals

Likelihood and severity: Low / Medium / High. Overall risk is the combination.

| ID | Risk | Likelihood | Severity | Overall |
|---|---|---|---|---|
| R1 | A user uploads a real patient's radiograph without authority or consent; the patient is a data subject with no relationship to the controller | **Medium** | **High** | **High** |
| R2 | Identifiers burned into image pixels are stored and shown to a reviewer; de-identification cannot remove them | **Medium** | **High** | **High** |
| R3 | Credential compromise exposes shared radiographs and all account emails | Low | **High** | **Medium** |
| R4 | Re-identification from image content combined with account data | Low | High | **Medium** |
| R5 | A user relies on a "normal" verdict and delays seeking care. Measured malignant recall is **0.633**, roughly one in three missed | **Medium** | **High** | **High** |
| R6 | Data retained longer than necessary | Low | Medium | **Low** |
| R7 | Processor terms not confirmed; no transfer evidence filed | Medium | Medium | **Medium** |
| R8 | A data subject cannot exercise erasure or closure without operator involvement | Medium | Medium | **Medium** |
| R9 | The Article 9(2)(j) research condition is challenged because the deployment is an open public demo rather than a defined protocol | Medium | High | **Medium** |
| R10 | Free-tier provider withdraws or changes service; data becomes inaccessible or is handled unexpectedly | Medium | Medium | **Medium** |
| R11 | Grad-CAM presented as explanation is measured at roughly chance, inviting unjustified confidence | **Medium** | Medium | **Medium** |
| R12 | Malicious uploads poison the training corpus, degrading a model others may rely on | Low | Medium | **Low** |
| R13 | A child's data is processed without appropriate safeguards | Low | High | **Medium** |
| **R14** | **A user uploads unlawful content. It is stored in D1 and displayed to the reviewer, so the controller hosts it and a person views it** | **Low** | **High** | **High** |

---

## 6. Measures to reduce risk

| ID | Measure | Effect | Residual | Status |
|---|---|---|---|---|
| R1 | Terms §5 require the user to hold all necessary authority and consent, with an explicit instruction not to upload identifiable radiographs, and an indemnity for breach | Reduced | **Medium** | **In place**, but contractual only and unverifiable |
| R1 | Repeat the warranty on the per-image sharing checkbox, so the representation is made at the moment of the act rather than once at sign-up | Reduced | Low | **Proposed**, cheapest available control |
| R2 | Terms §5 and Privacy notice warn that pixel-burned identifiers cannot be removed | Reduced | Medium | **In place** |
| R2 | A reviewer action that purges immediately rather than on the 7-day timer, plus the standing instruction printed on the review console | Reduced | Low | **Proposed** |
| R3 | `ONNM_ADMIN_KEY` rotated; separate app and admin keys limit blast radius; web app deliberately holds no admin key; **git-history audit repeated across all 116 commits, confirming no credential value has ever been committed** | **Strongly reduced** | Low | **Partly closed.** See below |
| R3 | **[CONTROLLER]** Rotate the Cloudflare API token and the R2 token, or confirm they were already rotated, and record the dates. Then complete the Article 33 assessment for incident 001 and record the outcome either way | Reduced | Low | **OUTSTANDING**, `INCIDENT_RESPONSE.md` §6 |
| R4 | 256 px downscale, DICOM PII strip, UUID filenames, country-level location only | Reduced | Low | **In place** |
| R5 | Prominent, unavoidable medical disclaimer; the tick box itself requires the user to state they understand it is not a medical device; malignant recall stated in Terms §3 and first in the model card; verdict never presented as diagnosis | Reduced | **Medium** | **In place**, cannot be eliminated while the demo is public |
| R6 | Retention schedule defined in `ROPA.md` §E and published in Privacy notice §8. Rejected images purged automatically after 7 days; capture tokens purged on use or expiry | **Strongly reduced** | Low | **In place**, except account closure |
| R7 | Both providers publish Art 28 terms and a UK transfer mechanism as standard. Confirm acceptance, file copies, complete a TRA for each | Reduced | Medium | **[CONTROLLER] action** |
| R8 | Self-service withdrawal of a shared image before approval; deletion of the stored copy on request at any time; account closure within 30 days committed in Terms §10 | Reduced | Medium | **Partly in place**, closure is manual |
| R9 | Write a short research protocol defining purpose, population and duration; determine HRA/REC applicability | Not yet reduced | Medium | **OUTSTANDING** |
| R10 | Accept and document the risk; keep an export path; the version ledger keeps model artefacts reproducible | Accepted | Medium | **Accepted risk** |
| R11 | Disclaimer already states Grad-CAM does not prove localisation; establish the chance baseline before any claim | Reduced | Low | **Partly in place** |
| R12 | Three-bucket triage, mandatory human review in four places, database trigger making contradictory labels unrepresentable, per-user daily caps, guarded model promotion | **Strongly reduced** | Low | **In place** |
| R13 | **Terms §4 require users to be 18 or over**, declared in the acceptance tick box; Privacy notice §12 states the same; Google account rules apply | Reduced | Low | **In place** |
| R14 | Terms §6 prohibit unlawful content, reserve immediate suspension, and state that such content will be preserved and reported rather than deleted on the usual timer | Reduced | Medium | **In place** |
| R14 | Incident procedure for unlawful content: do not download, do not forward, preserve, report, delete only on instruction | Reduced | Medium | **In place**, `INCIDENT_RESPONSE.md` §7 |
| R14 | **[CONTROLLER]** Gate the sharing feature behind operator approval, so only approved accounts can place an image in front of a reviewer | **Strongly reduced** | Low | **Proposed**, the measure that collapses this risk rather than mitigating it |

---

## 7. Outcome

| Risk | Residual | Accepted by |
|---|---|---|
| R1 Third-party patient images | Medium | **[CONTROLLER]** |
| R2 Burned-in identifiers | Medium | **[CONTROLLER]** |
| R3 Credential compromise | Low | **[CONTROLLER]** Admin key rotated and history audit clean; two provider tokens still to confirm |
| R5 Reliance on a false-negative | Medium | **[CONTROLLER]**, inherent to a public demo |
| R6 Retention | Low | **[CONTROLLER]** |
| R7 Processor and transfer evidence | Medium until filed | **[CONTROLLER] action** |
| R8 Erasure and closure | Medium until closure is self-service | **[CONTROLLER]** |
| R14 Unlawful content | Medium, Low if sharing is approval-gated | **[CONTROLLER]** decision required |
| Others | Low to Medium | **[CONTROLLER]** |

**Recommendation:** the processing may continue. The three risks that version 0.1 said must
not be accepted have all moved: **R6 (retention) is closed**, **R7 (processor terms) is now
a filing action rather than a gap**, and **R3 (credentials) is substantially reduced**, with
two provider tokens left to confirm and one Article 33 assessment to record.

**The open decision that most changes the risk profile is R14**: whether sharing stays open
to any signed-in account or is gated behind operator approval.

---

## 8. DPO

No Data Protection Officer is appointed. **[CONTROLLER]** must record a decision under
Article 37: an appointment is required where core activities consist of **large-scale
processing of special category data**. Given the deliberate caps, "large scale" is
arguable, but the decision must be documented either way rather than left unaddressed.

## 9. Prior consultation (Article 36)

Prior consultation with the ICO is required only where a **high residual risk cannot be
mitigated**. On the analysis above, no residual risk is both high and unmitigable.
**[CONTROLLER]** should revisit this if the decision is taken to accept R1, R5 or R14 at a
high residual level rather than reduce them.

## 10. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Controller | Khalid Faiz | | |
| GRC | Yaso-cyber | | |
| DPO | n/a | | |

**This DPIA is not effective until signed.**
