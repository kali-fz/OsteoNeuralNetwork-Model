# Data Protection Impact Assessment — ONNM hosted deployment

**Status: DRAFT — not signed off.** Prepared by the engineering side to be completed and
signed by the controller. Sections marked **[CONTROLLER]** require a decision that only the
controller can make. This is not legal advice.

| Field | Value |
|---|---|
| Assessment | DPIA under UK GDPR Article 35 |
| Subject | ONNM hosted deployment and community submission loop |
| Version | 0.1 (draft) |
| Prepared | 2026-08-24 |
| Controller | kali-fz — **[CONTROLLER]** confirm whether acting personally or on behalf of an institution |
| Prepared by | Engineering, with GRC input (Yaso-cyber) |
| DPO consulted | No DPO appointed — see §8 |
| Signed off by | **Not yet signed** |
| Review | On any change of purpose, dataset, hosting, or user population; otherwise annually |

Related: `../grc_compliance_prompt.md` (compliance register), `ROPA.md`,
`INCIDENT_RESPONSE.md`.

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
200 MB of stored images (`cloudflare/src/worker.js:37–42`), so "large scale" is arguable.
**The DPIA is required regardless**, because health data plus AI plus risk of physical
harm is sufficient on the ICO screening list.

---

## 2. Description of the processing

### 2.1 Nature

Two configurations exist. **This DPIA covers the hosted deployment**; a purely local
installation transfers nothing and is out of scope except where noted.

Flow for the hosted app:

1. User signs in with Google (OIDC). ONNM receives email, `sub`, display name, photo URL.
2. A signed-in browser posts a short-lived, one-use token directly to the Cloudflare
   Worker, which records **only a two-letter country code**. No IP address or finer
   location is stored (`cloudflare/migrations/0006_browser_country_capture.sql`).
3. User uploads a radiograph (DICOM, PNG, JPEG, BMP, TIFF). **Inference runs on Streamlit
   Community Cloud's server, so the image reaches Streamlit.**
4. An out-of-distribution gate may reject the image before inference.
5. DenseNet-121 returns three-class probabilities; Grad-CAM produces a heatmap.
6. A submission record is written to Cloudflare D1. **The 256 px processed image is stored
   only if the user ticks the sharing box**, which is off by default and asked separately
   for every image (`worker.js:706–709`).
7. Shared submissions enter a human review queue. Only a human-approved label can reach
   training.

### 2.2 Scope — data processed

See `ROPA.md` for the itemised register. In summary: account identifiers (email, Google
`sub`, UUID, display name, photo URL, password hash for legacy password accounts),
consent and ToS timestamps, country code, model outputs (verdict, probabilities, OOD
flags), and — **on explicit per-image consent only** — a 256 px processed radiograph.

**Special category data:** the radiographs, and the inferred verdict, are data concerning
health (Art 4(15)) and therefore Article 9 data. The data remains **pseudonymised, not
anonymous**, because it is linked to an authenticated account.

### 2.3 Context

- **Data subjects:** registered users (researchers, students, curious members of the
  public), and — potentially and unverifiably — the **patients depicted in uploaded
  radiographs**, who are not users and have no relationship with the controller.
- **Relationship:** voluntary, no service dependency, users can stop at any time.
- **Expectations:** users are told plainly in the Privacy Policy that images reach
  Streamlit and that sharing is separate and optional. This is unusually transparent for a
  project of this size and is a genuine mitigation.
- **Children:** not directed at children; paediatric use is out of scope; **no age
  assurance beyond Google's own account rules**.
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
| DPO | None appointed — see §8 |
| Processors (Streamlit, Cloudflare, Google) | **Not yet approached** for Article 28 terms — see §6, risk R7 |
| Information security | Reviewed internally; see `../grc_compliance_prompt.md` §8 |
| Clinical input | **[CONTROLLER]** No clinician involved. Required before any claim of clinical utility |
| ICO prior consultation (Art 36) | Not required unless a high residual risk cannot be mitigated — see §9 |

---

## 4. Necessity and proportionality

### 4.1 Lawful basis (Article 6) — **[CONTROLLER] to confirm**

Proposed: **Article 6(1)(f), legitimate interests**, supported by a documented Legitimate
Interests Assessment. Rationale: the interest is scientific research into medical imaging;
processing is limited to what the user voluntarily submits; and the user retains control
over the sharing decision.

Consent (Art 6(1)(a)) is available but **fragile as the primary basis**: it is withdrawable
at any time, and withdrawal would require removing the data from training sets and
arguably from derived model weights, which is not currently technically achievable.

**Note the distinction carefully:** legitimate interests may be the Article 6 basis while
the *sharing* decision remains a genuine, granular, opt-in consent at the interface level.
The two are not in conflict — one is the legal basis, the other is a user-facing control
and a mitigation.

### 4.2 Article 9 condition — **[CONTROLLER] to confirm**

Proposed: **Article 9(2)(j)**, scientific research, read with **DPA 2018 Schedule 1 Part 1
paragraph 4** and the **Article 89(1)** safeguards.

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
| Authentication | Email, `sub` | No — required to isolate one user's history from another's |
| Inference | Image pixels | No — the image is the input |
| History | Submission metadata | No |
| Research corpus | 256 px image | **Already minimised** — original resolution is never stored; only 256 px, only on consent |
| Globe display | Country code | **Already minimised** — country only, deliberately chosen over coordinates |
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
- Deployment caps evidence a test deployment rather than a product launch.

---

## 5. Risks to individuals

Likelihood and severity: Low / Medium / High. Overall risk is the combination.

| ID | Risk | Likelihood | Severity | Overall |
|---|---|---|---|---|
| R1 | A user uploads a real patient's radiograph without authority or consent; the patient is a data subject with no relationship to the controller | **Medium** | **High** | **High** |
| R2 | Identifiers burned into image pixels are stored and shown to a reviewer; de-identification cannot remove them | **Medium** | **High** | **High** |
| R3 | Credential compromise exposes shared radiographs and all account emails — the admin key unlocks review and export | **Medium** | **High** | **High** |
| R4 | Re-identification from image content combined with account data | Low | High | **Medium** |
| R5 | A user relies on a "normal" verdict and delays seeking care. Measured malignant recall is **0.633** — roughly one in three missed | **Medium** | **High** | **High** |
| R6 | Data retained indefinitely because no retention schedule exists | **High** | Medium | **High** |
| R7 | No Article 28 contract with processors; no transfer mechanism documented | **High** | Medium | **High** |
| R8 | A data subject cannot exercise erasure because there is no user-facing deletion | **High** | Medium | **Medium** |
| R9 | The Article 9(2)(j) research condition is challenged because the deployment is an open public demo rather than a defined protocol | Medium | High | **Medium** |
| R10 | Free-tier provider withdraws or changes service; data becomes inaccessible or is handled unexpectedly | Medium | Medium | **Medium** |
| R11 | Grad-CAM presented as explanation is measured at roughly chance, inviting unjustified confidence | **Medium** | Medium | **Medium** |
| R12 | Malicious uploads poison the training corpus, degrading a model others may rely on | Low | Medium | **Low** |
| R13 | A child's data is processed without appropriate safeguards | Low | High | **Medium** |

---

## 6. Measures to reduce risk

| ID | Measure | Effect | Residual | Status |
|---|---|---|---|---|
| R1 | Terms require the user to hold all necessary authority and consent; explicit instruction not to upload identifiable radiographs to the hosted app | Reduced | **Medium** | **In place**, but contractual only and unverifiable |
| R1 | **[CONTROLLER]** Consider an interstitial confirmation at upload, and whether the hosted demo should accept uploads at all versus a fixed sample set | Reduced | Low | **Proposed** |
| R2 | Warning in Terms and Privacy Policy that pixel-burned identifiers cannot be removed | Reduced | Medium | **In place** |
| R2 | Reviewer instruction to reject and delete any image showing burned-in identifiers | Reduced | Low | **Proposed** — add to the review console |
| R3 | **Rotate the exposed credentials immediately**; separate app and admin keys already limit blast radius; hosted app deliberately holds no admin key | Reduced | Medium | **OUTSTANDING — P1** |
| R3 | Record rotation dates; repeat the git-history secret audit | Reduced | Low | **Proposed** |
| R4 | 256 px downscale, DICOM PII strip, UUID filenames, country-level location only | Reduced | Low | **In place** |
| R5 | Prominent, unavoidable medical disclaimer; malignant recall stated first in the model card; verdict never presented as diagnosis | Reduced | **Medium** | **In place** — cannot be eliminated while the demo is public |
| R6 | Define a retention schedule and implement scheduled deletion | Not yet reduced | **High** | **OUTSTANDING** |
| R7 | Obtain processor terms; complete IDTA/Addendum and a Transfer Risk Assessment | Not yet reduced | **High** | **OUTSTANDING** |
| R8 | Implement per-user scan deletion in the UI; document a DSR procedure with a named contact | Not yet reduced | Medium | **OUTSTANDING** |
| R9 | Write a short research protocol defining purpose, population and duration; determine HRA/REC applicability | Not yet reduced | Medium | **OUTSTANDING** |
| R10 | Accept and document the risk; keep an export path; the version ledger keeps model artefacts reproducible | Accepted | Medium | **Accepted risk** |
| R11 | Disclaimer already states Grad-CAM does not prove localisation; establish the chance baseline before any claim | Reduced | Low | **Partly in place** |
| R12 | Three-bucket triage, mandatory human review in four places, database trigger making contradictory labels unrepresentable, per-user daily caps, guarded model promotion | **Strongly reduced** | Low | **In place** |
| R13 | Terms exclude paediatric use; Google account rules apply; review the Age Appropriate Design Code if the audience changes | Reduced | Low | **Partly in place** |

---

## 7. Outcome

| Risk | Residual | Accepted by |
|---|---|---|
| R1 Third-party patient images | Medium | **[CONTROLLER]** |
| R2 Burned-in identifiers | Medium | **[CONTROLLER]** |
| R3 Credential compromise | **High until rotation is complete** | **Must not be accepted — remediate** |
| R5 Reliance on a false-negative | Medium | **[CONTROLLER]** — inherent to a public demo |
| R6 Retention | High until scheduled | **Must not be accepted — remediate** |
| R7 Processor and transfer documentation | High until complete | **Must not be accepted — remediate** |
| R8 Erasure | Medium until implemented | **[CONTROLLER]** |
| Others | Low–Medium | **[CONTROLLER]** |

**Recommendation:** the processing may continue **only** while remediation of R3, R6 and
R7 is actively in progress. R3 (credential rotation) should be treated as immediate.

---

## 8. DPO

No Data Protection Officer is appointed. **[CONTROLLER]** must record a decision under
Article 37: an appointment is required where core activities consist of **large-scale
processing of special category data**. Given the deliberate caps, "large scale" is
arguable, but the decision must be documented either way rather than left unaddressed.

## 9. Prior consultation (Article 36)

Prior consultation with the ICO is required only where a **high residual risk cannot be
mitigated**. On the analysis above, R3, R6 and R7 are all mitigable by ordinary means and
are being remediated, so prior consultation is **not currently indicated**.
**[CONTROLLER]** should revisit this if remediation stalls, or if the decision is taken to
accept R1 or R5 at a high residual level rather than reduce them.

## 10. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Controller | kali-fz | | |
| GRC | Yaso-cyber | | |
| DPO | n/a | | |

**This DPIA is not effective until signed.**
