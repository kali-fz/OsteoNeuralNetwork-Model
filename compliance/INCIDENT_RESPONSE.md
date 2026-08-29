# Incident Response Plan: ONNM

**Status: DRAFT for GRC review, not signed off.** Covers personal data breaches, security
incidents, model-safety incidents and unlawful content for the ONNM hosted deployment.
This is not legal advice.

| Field | Value |
|---|---|
| Version | 0.2 (draft) |
| Last updated | 2026-08-29 |
| Previous version | 0.1, 2026-08-24 |
| Owner | Khalid Faiz (controller and incident lead) |
| Review | Annually, and after every incident |

Related: `DPIA.md`, `ROPA.md`, `TERMS_REVIEW_PACK.md`, `../SECURITY.md`,
`../grc_compliance_prompt.md`.

> **What changed in 0.2.** Containment and notification steps referred to configuration on
> a third-party Python hosting platform that has been removed from the project; those
> steps are rewritten for the current Cloudflare deployment. **Section 7, unlawful content,
> is new**, and covers a scenario the plan did not previously address at all: content that
> must be preserved and reported rather than contained and deleted.

---

## 1. The clock

**A personal data breach must be assessed against UK GDPR Article 33 and, where the
threshold is met, notified to the ICO within 72 hours of becoming aware of it.**

"Aware" means having a reasonable degree of certainty that a security incident has
occurred leading to personal data being compromised. **The clock starts at awareness, not
at confirmation.** If assessment is incomplete at 72 hours, notify anyway and supply the
remainder in phases, Article 33(4) expressly permits this.

Given ONNM holds health data, the risk threshold is reached easily. **When in doubt,
notify.**

---

## 2. Roles

| Role | Holder | Responsibility |
|---|---|---|
| Incident lead | Khalid Faiz | Owns the response; decides on notification |
| Technical response | Khalid Faiz | Containment, rotation, forensics |
| GRC | Yaso-cyber | Article 33/34 assessment, records, ICO liaison |
| Reporter | Anyone | Raises the incident |

**Key-person risk:** the incident lead, technical responder and sole admin account are the
same person. **[CONTROLLER]** should record a documented deputy. This matters most for §7,
where the correct response involves not acting alone.

---

## 3. Severity

| Level | Definition | Examples |
|---|---|---|
| **S1 Critical** | Health data or credentials exposed, or unauthorised access to D1 | Admin key compromise; D1 dump leaked; one user reading another's radiographs |
| **S2 High** | Personal data affected without confirmed health-data exposure | Account emails exposed; auth bypass with no evidence of use |
| **S3 Medium** | Security weakness with no confirmed data impact | Injection fixed before exploitation; dependency CVE reachable |
| **S4 Low** | No personal data or safety impact | Rate-limit bypass; missing header |
| **SM Model safety** | Output could contribute to clinical harm | Wrong checkpoint served; calibration mismatch; verdict/label mismatch |
| **UC Unlawful content** | Material that is unlawful to hold has been uploaded | See §7. **Handled differently from every other row: preserve, do not delete** |

---

## 4. Procedure

### Step 1: Record (immediately)

Open an entry in the incident log (§8). Record the time of awareness, **this timestamp
starts the 72-hour clock**, what is known, and who is involved. Record it even if the
incident later proves to be nothing.

### Step 2: Contain

- Rotate affected Worker secrets: `npx wrangler secret put API_KEY` / `ADMIN_KEY`. Both
  live in the Cloudflare deployment only; there is no second configuration to keep in step.
- If D1 integrity is in question, take an export **to a private location outside the
  repository** before further change. `cloudflare/*.sql` is gitignored by pattern, but do
  not rely on that as the only control: the export contains every user's email, their
  password hash and every shared radiograph as base64.
- Where necessary, disable the Google OAuth client, or roll the Worker back to a previous
  version to take the service offline.
- **Do not destroy evidence.** Preserve logs before rotating or redeploying.

### Step 3: Assess

Establish: what data, whose data, how many people, whether it was health data, whether
data left the controller's systems, whether it is recoverable, and what harm could follow.

**Health data plus identifiability is a high-risk combination.** Radiographs plus account
emails will normally meet both the Article 33 and the Article 34 thresholds.

### Step 4: Notify

| Who | When | How |
|---|---|---|
| **ICO** | Within **72 hours** of awareness, unless unlikely to result in risk | ICO online reporting, or 0303 123 1113 |
| **Data subjects** (Art 34) | **Without undue delay** where the risk is high | Direct message to affected accounts; public notice if direct contact is impossible |
| **Processors / providers** | As soon as relevant | Cloudflare support; Google support |
| **MHRA** | Only if ONNM is ever a regulated device | Vigilance reporting |
| **Institutions** | If any contributor acts in a university capacity | Their own incident process may also bind |

The Privacy notice §11 promises data subjects that we will tell them without undue delay
where the risk is high, and explain what happened and what to do. That is a published
commitment, not just a statutory duty.

If a decision is taken **not** to notify, **record the reasoning**. Article 33(1) requires
the justification to be documented, and "we decided it was fine" is not a justification.

### Step 5: Recover

Fix the root cause, not the symptom. Redeploy in the correct order, **D1 migration first,
Worker second**; deploying a Worker that reads columns D1 does not have breaks the live
site. Verify the fix, and confirm the caps and auth boundary still hold.

### Step 6: Learn

Within 10 working days: write up the root cause, update this plan, add a regression test,
and add or close the relevant row in the gap register in `../grc_compliance_prompt.md`.

---

## 5. Model safety incidents (SM)

Distinct from data breaches, and easy to overlook because nothing leaks.

1. Identify the affected checkpoint by **sha256**, not by run name. The digest is what
   actually arrived.
2. Check `ONN.md` and `model_versions.json` for what should be serving.
3. Roll back by repinning `reports/PRODUCTION` through `scripts/version_model.py`. Rollback
   is the designed default: a regressed run is recorded as `held` and the previous
   checkpoint keeps serving.
4. If wrong output may have reached users, consider a notice. The tool is research-only,
   but a materially wrong verdict is still worth telling people about.
5. Record the incident in `ONN.md` alongside the version history.

---

## 6. Open incident: carried forward

| Item | Status |
|---|---|
| **Credential exposure.** The Cloudflare API token and R2 token were pasted into a chat transcript; `ONNM_COMMUNITY_KEY` and `ONNM_ADMIN_KEY` appeared in screenshots during setup | **Partly remediated** |
| Completed | `ONNM_ADMIN_KEY` rotated. **Git-history audit repeated across all 116 commits: no credential value has ever been committed.** The only sensitive-looking values in the repository are the admin email address and the D1 database id, both deliberately public |
| **Still required** | **[CONTROLLER]** rotate the Cloudflare API token and the R2 token, or confirm they were already rotated, and record the dates. Then assess against Article 33 and record the outcome either way, including a decision not to notify |
| Mitigating facts | No secret ever entered git history; `.env` is gitignored; D1 dumps are gitignored by pattern; the web app deliberately holds no admin key |

**This should be the first incident run through this plan**, and it cannot be closed until
the rotation dates and the Article 33 assessment are recorded.

---

## 7. Unlawful content (UC)

**This branch exists because the upload path accepts arbitrary images.** That is not
hypothetical: a photograph of a person against a white background was uploaded, passed the
out-of-distribution gate, and reached the review queue. A hostile user could upload
something that is unlawful to possess, tick the sharing box, and it would be stored in D1
and displayed on the reviewer's screen, because displaying it is exactly what the review
queue does.

**Everything else in this plan says contain and delete. This section says preserve and
report.** Following the ordinary procedure here would destroy evidence and could itself be
an offence.

### If you believe content may be unlawful

1. **Stop looking at it.** Close the review console. Do not open it again, do not scroll
   back, and do not view it a second time "to be sure".
2. **Do not download it, copy it, print it, screenshot it, or send it to anyone**,
   including the GRC reviewer, a colleague, or a support ticket. Describe it in words if
   you must describe it at all. Forwarding such material can be a separate offence
   regardless of intent.
3. **Do not delete it, and suspend the automatic purge.** Rejected images are deleted
   after seven days by a scheduled job; that timer must not be allowed to destroy material
   that may be evidence. Record the `submission_id` and note that a legal hold applies.
4. **Record the facts**: submission id, timestamp, the account that uploaded it, and the
   time you became aware. Nothing about the content itself beyond the minimum.
5. **Suspend the account** under Terms §6, which reserves that right expressly.
6. **Report it** to the appropriate authority, and take their instruction on what to do
   with the stored copy:
   - **Child sexual abuse material:** report to the Internet Watch Foundation
     (`report.iwf.org.uk`). Do not view, do not copy, do not send them the file unless they
     ask. In an emergency, or where a child is at immediate risk, call 999.
   - **Terrorist material:** report through GOV.UK's report terrorist content service.
   - **Anything else unlawful:** report to the police on 101, or 999 if urgent.
   - **[CONTROLLER]** confirm these routes and record a named contact before this plan is
     signed.
7. **Delete only on instruction**, and record who gave it and when.
8. **Tell the GRC reviewer that a UC incident is open**, and no more than that, so the
   record exists without the material spreading.

### Is it also a personal data breach?

Usually not. Unlawful content uploaded by a user is not, by itself, a compromise of
personal data held by the controller, so the §1 clock is not automatically running.
Assess it separately, and note that the uploader's own account data is not the concern
here. Record the assessment either way.

### The measure that would make this rare

Terms §6 prohibits unlawful content and reserves suspension, which deters and gives a
basis to act. It does not prevent an upload. **The control that would collapse this risk
is gating the sharing feature behind operator approval**, so that only accounts the
operator has approved can place an image in front of a reviewer. Scanning would stay open
to everyone. This is recorded as R14 in `DPIA.md` and is an open decision for the
controller.

---

## 8. Incident log

| ID | Date aware | Severity | Summary | Art 33 assessed | Notified | Closed |
|---|---|---|---|---|---|---|
| 001 | Setup period, 2026 | S1 (assessed) | Credential exposure via chat transcript and screenshots. Admin key rotated; git-history audit clean | **Not yet** | - | **Open** |
