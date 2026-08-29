# Record of Processing Activities: ONNM

**Status: DRAFT for GRC review.** Record maintained under **UK GDPR Article 30(1)**.
Fields marked **[CONTROLLER]** require a decision only the controller can make. This is not
legal advice.

The Article 30(5) derogation for organisations under 250 staff **does not apply**, because
the processing involves special category data and is not occasional.

| Field | Value |
|---|---|
| Version | 0.2 (draft) |
| Last updated | 2026-08-29 |
| Previous version | 0.1, 2026-08-24 |
| Review | Annually, and on any change to purpose, data, processor, or transfer route |

Related: `DPIA.md`, `INCIDENT_RESPONSE.md`, `TERMS_REVIEW_PACK.md`,
`../grc_compliance_prompt.md`.

> **What changed in 0.2.** Version 0.1 described an architecture that no longer exists: it
> recorded a third-party Python hosting platform as a processor receiving every uploaded
> image. That platform was removed from the project entirely. **Inference now runs in a
> Cloudflare container**, and the only processors are Cloudflare and Google. The lawful
> bases in §B2 and §B3 have also changed, from legitimate interests to consent, for the
> reason given in §B3. Retention in §E is now specified rather than outstanding, and most
> of it is enforced in code.

---

## A. Controller details

| Item | Value |
|---|---|
| Controller | **Khalid Faiz**, acting in a personal capacity as an independent researcher. Named in the published Terms §1 and Privacy notice §1. This is a description of who decides the purposes and means of the processing, not an appointed role; it would change if ONNM were ever operated under a university or company |
| Contact for data protection | `kzfhero@gmail.com`, published in the Terms §1 and the Privacy notice §1 as a monitored address |
| DPO | Not appointed. Decision recorded in `DPIA.md` §8 |
| ICO registration | **[CONTROLLER]** assess whether registration and the data protection fee are required |

---

## B. Processing activities

### B1: Account management and authentication

| Field | Detail |
|---|---|
| Purpose | Authenticate users; isolate each user's submission history; enforce the Terms |
| Categories of data subject | Registered users |
| Categories of data | Email; Google `sub`; user UUID; display name; profile photo URL; account creation timestamp; Terms acceptance timestamp and version; PBKDF2-HMAC-SHA256 password hash (legacy password accounts only) |
| Special category | No |
| Article 6 basis | **6(1)(b) performance of a contract.** The contract is the Terms of use, which since 2026-08-29 every account must accept before it can be created |
| Recipients | Cloudflare (Workers, D1); Google (identity provider) |
| Transfers | Outside UK, see §C |
| Retention | Until the user closes their account. Deletion within 30 days of request, committed in Terms §10. **Closure is not yet self-service; see §F item 3** |
| Security | Bearer-key auth on every route; parameterised queries; platform encryption at rest; PBKDF2 with 600,000 iterations and unique salts; signed host-scoped HttpOnly session cookies |

**Note on acceptance evidence.** `tos_accepted_at` was meaningless in version 0.1: it was
written as a copy of `created_at`, so it recorded that a row was inserted, not that a human
agreed. Since the Terms gate shipped it records an actual acceptance, and `tos_version`
records which text was shown. No IP address or user agent is recorded against an
acceptance, because the system stores neither anywhere; this is a deliberate trade of
evidential strength for data minimisation.

### B2: Radiograph inference (the core service)

| Field | Detail |
|---|---|
| Purpose | Perform the classification the user requested and display the result with a Grad-CAM heatmap |
| Categories of data subject | Registered users; **and the patients depicted in uploaded images**, who may not be users |
| Categories of data | Radiograph pixel data; derived 256 px model input; model verdict, class probabilities, threshold, calibration state; OOD flag and score |
| Special category | **Yes, data concerning health, Art 4(15) / Art 9(1)** |
| Article 6 basis | **6(1)(b) performance of a contract.** The user asked for a scan; running it is the service |
| Article 9 condition | **9(2)(a) explicit consent.** Uploading a radiograph for analysis is an explicit, specific act directed at exactly this processing |
| Recipients | **Cloudflare only.** Inference runs in a Cloudflare container reached through a Durable Object; the image does not leave Cloudflare's infrastructure |
| Transfers | Outside UK, see §C |
| Retention | **The image is never written to storage.** It is held in memory for the request and discarded, unless the user separately ticks the sharing box, which is B3. Verdict metadata is retained with the account, per B1 |
| Security | Upload validation; OOD pre-screen; request size ceiling 1.5 MB; DICOM PII strip; EXIF-free re-encode; UUID filenames; single container instance |

### B3: Community submission and review loop

| Field | Detail |
|---|---|
| Purpose | With explicit consent, retain a de-identified 256 px image for human review and possible inclusion in a research training set |
| Categories of data subject | Consenting users; patients depicted in shared images |
| Categories of data | 256 px processed image (base64); image sha256 and byte count; `consent_at`; triage bucket and reason; review status; admin bucket, label, note; reviewer identity; user feedback and comments |
| Special category | **Yes** |
| Article 6 basis | **6(1)(a) consent** |
| Article 9 condition | **9(2)(a) explicit consent** for collection and review. **9(2)(j) scientific research**, with DPA 2018 Sch 1 Pt 1 para 4 and the Art 89(1) safeguards, for continued retention in the research corpus after consent is withdrawn |
| Recipients | Cloudflare (D1); a single pinned admin/reviewer account |
| Transfers | Outside UK, see §C |
| Retention | Pending: until reviewed or withdrawn by the user. Rejected: **deleted automatically within 7 days**, enforced by a scheduled job. Approved: life of the research, with the stored copy deleted on request |
| Security | Sharing off by default and asked per image; an image without `shared = 1` is discarded rather than stored; review gate enforced in four places; database trigger rejecting contradictory bucket/label pairs; separate admin key not held by the web app; caps of 50 submissions per user per day and 200 MB total |

**Why this is consent and not legitimate interests.** Version 0.1 recorded legitimate
interests here, describing the per-image tick box as "an interface control" rather than as
the lawful basis. That position was abandoned on review. The tick box is granular, opt-in,
off by default, asked separately for every single file, and withdrawable; a regulator
reading the interface would characterise it as consent whatever this register said, and a
register that disagrees with the interface is a liability rather than a defence.

The objection to consent was that a model cannot be untrained, so withdrawal could not be
honoured. **Article 7(3) disposes of that**: withdrawal does not affect the lawfulness of
processing carried out before it. Withdrawal stops future use and deletes the stored copy.
The already-trained weights are processing that was lawful when it happened, and their
retention rests on 9(2)(j) with Art 17(3)(d) limiting erasure to that extent and no further.
This is stated plainly to the data subject in Terms §8 and Privacy notice §5.

### B4: Country capture and the public globe

| Field | Detail |
|---|---|
| Purpose | Display country-level aggregate contribution counts on the public homepage |
| Categories of data subject | Registered users |
| Categories of data | Two-letter ISO country code; `country_captured_at`; short-lived one-use capture token hash, expiry and use timestamp |
| Special category | No |
| Article 6 basis | 6(1)(f) legitimate interests: showing the project's reach, against a country code being close to the least identifying thing that could be displayed |
| Recipients | Cloudflare |
| Transfers | Outside UK, see §C |
| Retention | Country code: for the life of the account. Capture tokens: **purged automatically** when used or expired |
| Security | **No IP address is seen, logged or stored.** The browser posts a one-use token and Cloudflare supplies only the country. The schema is constrained by CHECK to a two-character uppercase code, so it is incapable of holding finer location |

### B5: Public contributor profile

| Field | Detail |
|---|---|
| Purpose | Show a contributor's name and photo on the public homepage |
| Categories of data subject | Users who opt in |
| Categories of data | Display name; profile photo URL; approved contribution count |
| Special category | No |
| Article 6 basis | **6(1)(a) consent**: a genuine, freely given, reversible publication choice |
| Recipients | Cloudflare; **the public** |
| Retention | Until the user turns the toggle off, which removes the stored public name and photo |
| Security | Off by default; requires an approved contribution; reversible at any time |

### B6: Security, logging and troubleshooting

| Field | Detail |
|---|---|
| Purpose | Secure the deployment, detect abuse, diagnose faults |
| Categories of data | Cloudflare Workers observability logs (10% head sampling) |
| Article 6 basis | 6(1)(f) legitimate interests |
| Retention | Cloudflare's own retention period for Workers observability. **[CONTROLLER]** confirm and record the figure |
| Security | Provider-managed |

---

## C. Transfers outside the UK

| Recipient | Role | Location | Mechanism | Status |
|---|---|---|---|---|
| Cloudflare | Processor: Workers, D1, Containers, inference | Global edge; D1 primary observed in Western Europe | Cloudflare's Data Processing Addendum, incorporating the UK Addendum to the EU SCCs | **[CONTROLLER]** confirm the DPA has been accepted on the account, and file a copy |
| Google | Identity provider | US | Google's Cloud / API Data Processing Terms, incorporating the UK Addendum to the EU SCCs | **[CONTROLLER]** confirm acceptance and file a copy |
| GitHub (Microsoft) | Source and model checkpoint hosting | US | Public code only; no personal data flows | Confirmed: nothing personal is published |

Both providers publish Article 28 processor terms and an international transfer mechanism
as part of their standard terms, so the mechanism exists and does not need to be
negotiated. **What remains outstanding is confirming acceptance and retaining evidence**,
plus a Transfer Risk Assessment for each. That is a controller action, not an engineering
one.

Version 0.1 recorded a third processor receiving every uploaded image. That is no longer
true, and its removal is the single largest reduction in transfer risk this project has
made.

---

## D. Data subject rights

| Right | Current position |
|---|---|
| Information (Arts 13, 14) | **Satisfied.** A full Privacy notice is published at `/privacy` and linked from the Terms, the footer, and the acceptance tick box |
| Access | On request to the published address, within one month. **Procedure not yet written; see §F item 4** |
| Rectification | On request. Display name and photo are self-service |
| Erasure | **Self-service for a shared image until it is approved.** After approval, the stored copy is deleted on request; the trained weights are not reversible, and Art 17(3)(d) is relied on for that and nothing else. Account closure on request within 30 days, committed in Terms §10 |
| Restriction | On request, manual |
| Portability | Applies to the 6(1)(a) and 6(1)(b) processing here. **Not yet implemented; see §F item 4** |
| Object | Must be honoured for the 6(1)(f) processing in B4 and B6 unless compelling legitimate grounds are demonstrated |
| Withdraw consent | **Self-service** for B3 before approval and for B5 at any time; otherwise on request. As easy to withdraw as to give, per Art 7(3) |
| Automated decision-making (Art 22) | Not engaged: the system produces a classification, not a decision about a person, and outputs are explicitly not for care decisions |

**Response deadline: one month**, stated to data subjects in the Privacy notice §10.

**Manifestly unfounded or excessive requests.** Art 12(5) permits a reasonable fee or
refusal, with reasons and with the right to complain. **Art 11** applies where the
controller cannot identify the data subject: a patient depicted in an image uploaded by
somebody else is not identifiable to this controller by design, and Arts 15 to 20 do not
apply unless they supply identifying information. Both positions are legitimate and both
should be in the written procedure rather than improvised under a deadline.

---

## E. Retention schedule

Each period below is either enforced in code or committed to the data subject in the
published Privacy notice §8. Nothing here is aspirational.

| Data | Retention | Mechanism |
|---|---|---|
| Account record and scan history | Until account closure; deleted within 30 days of request | Committed in Terms §10. **Manual until closure is self-service** |
| Submission metadata, image not shared | With the account. No image is stored at all | Enforced: an image without `shared = 1` is discarded |
| Shared image, awaiting review | Until reviewed or withdrawn by the user | Self-service withdrawal |
| Shared image, rejected at review | **7 days** | **Automated**, scheduled job |
| Shared image, approved into the research corpus | Life of the research; stored copy deleted on request | Documented decision; Art 17(3)(d) for the weights only |
| Capture tokens | On use or expiry | **Automated** |
| Public profile name and photo | Until toggled off | **Automated**, self-service |
| Logs | Cloudflare default | **[CONTROLLER]** confirm the figure |

**[CONTROLLER]** to confirm what "life of the research" means as a period, and set a review
date rather than leaving it open-ended. A defined end, even a distant one, is materially
easier to defend than none.

---

## F. Outstanding items

1. ~~Name the controller and confirm the capacity.~~ **Done**: Khalid Faiz, personal
   capacity. Named in both published notices.
2. **Confirm the provider DPAs** have been accepted for Cloudflare and Google, file
   copies, and complete a Transfer Risk Assessment for each.
3. **Implement self-service account closure.** Currently there is no route; the 30-day
   commitment in Terms §10 is honoured manually.
4. **Write the DSR procedure**, including access and portability, with the Art 12(5) and
   Art 11 positions recorded.
5. Confirm the Cloudflare log retention figure and record it in B6 and §E.
6. Define "life of the research" and set a review date.
7. Assess ICO registration and the data protection fee.
8. Record the Article 37 DPO decision (drafted in `DPIA.md` §8, needs signature).
