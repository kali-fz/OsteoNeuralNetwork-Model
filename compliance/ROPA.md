# Record of Processing Activities — ONNM

**Status: DRAFT — not signed off.** Record maintained under **UK GDPR Article 30(1)**.
Fields marked **[CONTROLLER]** require confirmation. This is not legal advice.

The Article 30(5) derogation for organisations under 250 staff **does not apply**, because
the processing involves special category data and is not occasional.

| Field | Value |
|---|---|
| Version | 0.1 (draft) |
| Last updated | 2026-08-24 |
| Review | Annually, and on any change to purpose, data, processor, or transfer route |

Related: `DPIA.md`, `INCIDENT_RESPONSE.md`, `../grc_compliance_prompt.md`.

---

## A. Controller details

| Item | Value |
|---|---|
| Controller | kali-fz — **[CONTROLLER]** confirm personal or institutional capacity |
| Contact for data protection | **[CONTROLLER]** — a monitored address is required; "the repository's published contact channels" is not sufficient for the hosted service |
| DPO | Not appointed — decision to be recorded (see `DPIA.md` §8) |
| ICO registration | **[CONTROLLER]** — assess whether registration and the data protection fee are required |

---

## B. Processing activities

### B1 — Account management and authentication

| Field | Detail |
|---|---|
| Purpose | Authenticate users; isolate each user's submission history; enforce the Terms |
| Categories of data subject | Registered users |
| Categories of data | Email; Google `sub`; user UUID; display name; profile photo URL; account and ToS acceptance timestamps; PBKDF2-HMAC-SHA256 password hash (password accounts only) |
| Special category | No |
| Article 6 basis | 6(1)(b) performance of a contract (the Terms), or 6(1)(f) legitimate interests — **[CONTROLLER]** confirm |
| Recipients | Cloudflare (Workers, D1); Google (identity provider) |
| Transfers | Outside UK — see §C |
| Retention | **Not defined — OUTSTANDING.** Proposed: delete on account closure, or after 24 months of inactivity |
| Security | Bearer-key auth on every route; parameterised queries; platform encryption at rest; PBKDF2 with 600,000 iterations and unique salts |

### B2 — Radiograph inference (the core service)

| Field | Detail |
|---|---|
| Purpose | Perform the classification the user requested and display the result with a Grad-CAM heatmap |
| Categories of data subject | Registered users; **and the patients depicted in uploaded images**, who may not be users |
| Categories of data | Radiograph pixel data; derived 256 px model input; model verdict, class probabilities, threshold, calibration state; OOD flag and score |
| Special category | **Yes — data concerning health, Art 4(15) / Art 9(1)** |
| Article 6 basis | 6(1)(f) legitimate interests — **[CONTROLLER]** confirm |
| Article 9 condition | 9(2)(j) scientific research + DPA 2018 Sch 1 Pt 1 para 4 + Art 89(1) safeguards — **[CONTROLLER]** confirm |
| Recipients | **Streamlit Community Cloud — receives every uploaded image, because inference runs on their server**; Cloudflare (verdict metadata) |
| Transfers | Outside UK — see §C |
| Retention | Image: held in memory for the request, not written to disk by ONNM. Verdict metadata: **not defined — OUTSTANDING** |
| Security | Upload validation; OOD pre-screen; request size ceiling 1.5 MB; DICOM PII strip; EXIF-free re-encode; UUID filenames |

### B3 — Community submission and review loop

| Field | Detail |
|---|---|
| Purpose | With explicit consent, retain a de-identified 256 px image for human review and possible inclusion in a research training set |
| Categories of data subject | Consenting users; patients depicted in shared images |
| Categories of data | 256 px processed image (base64); image sha256 and byte count; `consent_at`; triage bucket and reason; review status; admin bucket, label, note; reviewer identity; user feedback and comments |
| Special category | **Yes** |
| Article 6 basis | 6(1)(f) legitimate interests, with granular opt-in consent as an interface control — **[CONTROLLER]** confirm |
| Article 9 condition | 9(2)(j) scientific research — **[CONTROLLER]** confirm |
| Recipients | Cloudflare (D1); a single pinned admin/reviewer account |
| Transfers | Outside UK — see §C |
| Retention | **Not defined — OUTSTANDING.** Proposed: retain approved research records for the life of the research; delete rejected and unreviewed rows after 12 months |
| Security | Sharing off by default and asked per image; an image without `shared = 1` is discarded rather than stored; review gate enforced in four places; database trigger rejecting contradictory bucket/label pairs; separate admin key not held by the hosted app; caps of 50 submissions per user per day and 200 MB total |

### B4 — Country capture and the public globe

| Field | Detail |
|---|---|
| Purpose | Display country-level aggregate contribution counts on the public homepage |
| Categories of data subject | Registered users |
| Categories of data | Two-letter ISO country code; `country_captured_at`; short-lived one-use capture token hash, expiry and use timestamp |
| Special category | No |
| Article 6 basis | 6(1)(f) legitimate interests — **[CONTROLLER]** confirm |
| Recipients | Cloudflare |
| Transfers | Outside UK — see §C |
| Retention | Country code: for the life of the account. Capture tokens: **purged automatically** when used or expired |
| Security | **No IP address is seen, logged or stored** — the browser posts a one-use token and Cloudflare supplies only the country. The schema is constrained by CHECK to a two-character uppercase code, so it is incapable of holding finer location |

### B5 — Public contributor profile

| Field | Detail |
|---|---|
| Purpose | Show a contributor's name and photo on the public homepage |
| Categories of data subject | Users who opt in |
| Categories of data | Display name; profile photo URL; approved contribution count |
| Special category | No |
| Article 6 basis | **6(1)(a) consent** — this is a genuine, freely given, reversible publication choice |
| Recipients | Cloudflare; **the public** |
| Retention | Until the user turns the toggle off, which removes the stored public name and photo |
| Security | Off by default; requires an approved contribution; reversible at any time |

### B6 — Security, logging and troubleshooting

| Field | Detail |
|---|---|
| Purpose | Secure the deployment, detect abuse, diagnose faults |
| Categories of data | Cloudflare Workers observability logs (10% head sampling); Streamlit platform logs |
| Article 6 basis | 6(1)(f) legitimate interests |
| Retention | Per provider default — **[CONTROLLER]** confirm and record |
| Security | Provider-managed |

---

## C. Transfers outside the UK

| Recipient | Role | Location | Mechanism | Status |
|---|---|---|---|---|
| Streamlit Community Cloud (Snowflake) | Processor — hosting, receives every image | US | IDTA or UK Addendum + TRA | **OUTSTANDING** |
| Cloudflare | Processor — Workers, D1 | Global edge; D1 primary observed in Western Europe | IDTA or UK Addendum + TRA | **OUTSTANDING** |
| Google | Identity provider | US | IDTA or UK Addendum + TRA | **OUTSTANDING** |
| GitHub (Microsoft) | Source and model checkpoint hosting | US | Public code only; assess if any personal data flows | **[CONTROLLER]** confirm |

**No Article 28 processor contract has been obtained or reviewed for any of the above.**
Free-tier standard terms may not satisfy Article 28(3).

---

## D. Data subject rights

| Right | Current position |
|---|---|
| Access | Manual — operator queries D1. **No documented procedure** |
| Rectification | Manual |
| Erasure | **Manual only.** Per-user deletion in the UI is an open, unimplemented item. Art 89 research derogations may apply but must be justified case by case |
| Restriction | Manual |
| Portability | Not implemented; applies only to 6(1)(a)/6(1)(b) processing carried out by automated means |
| Object | Must be honoured for 6(1)(f) processing unless compelling legitimate grounds are demonstrated |
| Automated decision-making (Art 22) | Not engaged — the system produces a classification, not a decision about a person, and outputs are explicitly not for care decisions |

**Response deadline: one month.** A named contact and a tested procedure are
**OUTSTANDING**.

---

## E. Retention schedule — TO BE COMPLETED

**[CONTROLLER]** must set each period. Proposed starting points:

| Data | Proposed retention | Deletion method |
|---|---|---|
| Account record | Account closure, or 24 months inactivity | Manual now; **automate** |
| Submission metadata (unshared) | 12 months | **To implement** |
| Shared image — rejected at review | 12 months | **To implement** |
| Shared image — approved into research corpus | Life of the research, then review | Documented decision |
| Capture tokens | On use or expiry | **Automated already** |
| Public profile name/photo | Until toggled off | **Automated already** |
| Logs | Provider default | Confirm |

---

## F. Outstanding items

1. Name the controller and a monitored data protection contact.
2. Confirm Article 6 bases and Article 9 conditions; write the Legitimate Interests Assessment.
3. Obtain and review Article 28 terms from Streamlit, Cloudflare and Google.
4. Complete IDTA/Addendum and a Transfer Risk Assessment for each transfer.
5. Set and implement the retention schedule in §E.
6. Implement user-facing deletion and document the DSR procedure.
7. Assess ICO registration and the data protection fee.
8. Record the Article 37 DPO decision.
