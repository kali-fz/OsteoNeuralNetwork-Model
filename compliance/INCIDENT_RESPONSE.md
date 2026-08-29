# Incident Response Plan: ONNM

**Status: DRAFT, not signed off.** Covers personal data breaches, security incidents and
model-safety incidents for the ONNM hosted deployment. This is not legal advice.

| Field | Value |
|---|---|
| Version | 0.1 (draft) |
| Last updated | 2026-08-24 |
| Owner | kali-fz (controller) |
| Review | Annually, and after every incident |

Related: `DPIA.md`, `ROPA.md`, `../SECURITY.md`, `../grc_compliance_prompt.md`.

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
| Incident lead | kali-fz | Owns the response; decides on notification |
| Technical response | kali-fz | Containment, rotation, forensics |
| GRC | Yaso-cyber | Article 33/34 assessment, records, ICO liaison |
| Reporter | Anyone | Raises the incident |

**Key-person risk:** the incident lead, technical responder and sole admin account are the
same person. **[CONTROLLER]** should record a documented deputy.

---

## 3. Severity

| Level | Definition | Examples |
|---|---|---|
| **S1 Critical** | Health data or credentials exposed, or unauthorised access to D1 | Admin key compromise; D1 dump leaked; one user reading another's radiographs |
| **S2 High** | Personal data affected without confirmed health-data exposure | Account emails exposed; auth bypass with no evidence of use |
| **S3 Medium** | Security weakness with no confirmed data impact | Injection fixed before exploitation; dependency CVE reachable |
| **S4 Low** | No personal data or safety impact | Rate-limit bypass; missing header |
| **SM Model safety** | Output could contribute to clinical harm | Wrong checkpoint served; calibration mismatch; verdict/label mismatch |

---

## 4. Procedure

### Step 1: Record (immediately)

Open an entry in the incident log (§7). Record the time of awareness, **this timestamp
starts the 72-hour clock**, what is known, and who is involved. Record it even if the
incident later proves to be nothing.

### Step 2: Contain

- Rotate affected secrets: `npx wrangler secret put API_KEY` / `ADMIN_KEY`, **and update
  the matching Streamlit configuration**, or the live app breaks.
- If D1 integrity is in question, take an export **to a private location outside the
  repository** before further change, `cloudflare/*.sql` is gitignored by pattern, but do
  not rely on that as the only control.
- Where necessary, disable the Google OAuth client or take the Streamlit app offline.
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
| **Processors / providers** | As soon as relevant | Cloudflare, Streamlit, Google support |
| **MHRA** | Only if ONNM is ever a regulated device | Vigilance reporting |
| **Institutions** | If any contributor acts in a university capacity | Their own incident process may also bind |

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
| **Credential exposure.** The Cloudflare API token and R2 token were pasted into a chat transcript; `ONNM_COMMUNITY_KEY` and `ONNM_ADMIN_KEY` appeared in screenshots during setup | **OPEN, not remediated.** Classified "Low" in `TODO.md`; that classification is wrong, because the admin key unlocks review, approval and export of every shared radiograph |
| Required actions | Rotate all four secrets and update the Streamlit configuration; record rotation dates; assess against Article 33 and record the outcome either way; repeat the git-history secret audit |
| Mitigating facts | A prior audit confirmed no secret value ever entered git history; `.env` and `.streamlit/secrets.toml` are gitignored; the hosted app deliberately holds no admin key |

**This should be the first incident run through this plan.**

---

## 7. Incident log

| ID | Date aware | Severity | Summary | Art 33 assessed | Notified | Closed |
|---|---|---|---|---|---|---|
| 001 | Setup period, 2026 | S1 (assessed) | Credential exposure via chat transcript and screenshots | **Not yet** | - | **Open** |
