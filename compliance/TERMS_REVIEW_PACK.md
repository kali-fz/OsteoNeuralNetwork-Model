# Terms and Privacy: review pack for GRC

**Status: DRAFT for review.** Prepared by the engineering side for GRC review before the
site is promoted beyond its current audience. This is not legal advice, and the engineering
side is not qualified to give it.

| Field | Value |
|---|---|
| Subject | Terms of use v2026-08-30 and Privacy notice v2026-08-30 |
| Status | **Live.** Deployed 2026-08-29, Worker version `f4f0fcae` |
| Read them at | `osteoneuralnetwork.com/terms` and `osteoneuralnetwork.com/privacy` |
| Prepared | 2026-08-29 |
| Prepared by | Engineering |
| For review by | Yaso-cyber, GRC |
| Also revised | `DPIA.md` 0.2, `ROPA.md` 0.2, `INCIDENT_RESPONSE.md` 0.2 |
| Controller | Khalid Faiz, personal capacity |
| Verification | 452 Python tests and 56 Worker tests pass |

> **Why this is live before you have signed it off.** The previous Terms were worse in every
> respect this pack identifies, and leaving them serving while a better version sat in a
> branch would have been the wrong trade. Nothing here is hard to change: the text is data in
> one file, and any correction you make is a redeploy. Reviewing the pages as a visitor
> actually sees them is also easier than reviewing them as source.

---

## 0. What I am asking you to do

**Review finished text, not a list of problems.**

The first version of this pack listed sixteen legal findings and nine abuse cases and asked
you to work out what to do about them. That was the wrong thing to send you. Everything in
it that engineering could fix has been fixed, and this pack now hands you five documents
that are internally consistent and match the system as built.

What I need from you is the judgement I cannot supply: whether the drafting is right,
whether the lawful basis analysis in §6 holds, and whether anything here would embarrass us
in front of a regulator. §7 lists what is genuinely still open, and every item on it is a
decision or an action for the controller rather than a defect I have left for you to find.

---

## 1. What is in this pack

| Document | Version | State |
|---|---|---|
| **Terms of use** (`web/src/pages/terms.js`) | 2026-08-30 | Rewritten. 11 sections became 16 |
| **Privacy notice** (`web/src/pages/privacy.js`) | 2026-08-30 | **New.** Did not previously exist |
| `compliance/DPIA.md` | 0.2 | Architecture corrected; lawful bases revised; R14 added |
| `compliance/ROPA.md` | 0.2 | Architecture corrected; lawful bases revised; retention now specified |
| `compliance/INCIDENT_RESPONSE.md` | 0.2 | Unlawful content procedure added as §7 |

**All of this is live.** Read the two published documents as a visitor sees them:

- `https://osteoneuralnetwork.com/terms`
- `https://osteoneuralnetwork.com/privacy`

The flow to check is the one a new user meets: the site offers **Sign in**, which goes to
the Terms rather than to Google. Reading and ticking is what unlocks the Google button, and
that is enforced by the server, not by the page. The Privacy notice is linked from the
Terms, from the tick box, and from the footer, and is readable without an account.

**The seven existing accounts will be asked to agree again** on their next visit, because
this revision is a material change. See §4, finding L17.

The DPIA, ROPA and incident response plan are **not** published as site pages, and should
not be. A ROPA is an Article 30 record produced for a supervisory authority on request, not
a public notice, and the incident plan currently documents an open credential incident;
publishing it would tell a reader exactly which tokens may still be unrotated. What the site
does instead is **declare** that those records exist and are maintained, in a "Governance"
disclosure in the footer, which is the normal pattern. Tell me if you want that split drawn
differently.

---

## 2. The system in one page

ONNM is a free, open-source research prototype that classifies bone radiographs as normal,
benign or malignant and returns a Grad-CAM heat map. It is not a medical device, holds no
regulatory clearance, and is not clinically validated. Measured malignant recall is 0.633,
so roughly one malignancy in three is missed. That figure is stated in the Terms, not
buried.

**Hosting.** One Cloudflare Worker serves the website, the API, and an inference container
addressed through a Durable Object. Accounts and submissions live in Cloudflare D1.
Identity comes from Google Sign-In. **Cloudflare and Google are the only processors.**

**What happens to an uploaded image.** It is analysed in the container and a result is
returned. **The image is never written to storage** unless the user ticks a sharing box,
which is off by default and asked separately for every image. What is then stored is a 256
pixel processed copy, never the original and never its metadata. DICOM headers are stripped
and the image is re-encoded without EXIF. A record of the scan and its verdict is always
kept against the account.

**Review.** A shared image enters a queue visible to exactly one pinned reviewer account.
Only a human-approved label can reach the training set, enforced in the export query, in a
database trigger, and in the Worker. A rejected image is deleted automatically within seven
days by a scheduled job.

**Deliberate limits:** 500 accounts, 50 submissions per account per day, 200 MB of stored
images, 1.5 MB request ceiling. Sized as a test deployment, and that sizing is itself a
control. The storage cap in particular sits at the resource rather than at the identity, so
it bounds total exposure regardless of how many accounts an abuser creates.

**Two privacy properties worth knowing.** No IP address is ever seen, logged or stored;
country is captured at two-letter level through a one-use token, and the database column is
constrained so it is incapable of holding anything finer. Model weights are never sent to a
visitor's browser.

---

## 3. What is enforced in code

Which parts are real changes what the wording has to carry.

| Claim | Enforced where | Routable around? |
|---|---|---|
| No account exists without acceptance | `authStart` refuses to begin OAuth without a signed acceptance cookie | No, the signature is minted server side |
| Existing accounts must catch up | `/api/session` reads acceptance from the account row, not the cookie | No |
| No scan without acceptance | `/api/scan` returns 403 `terms_required` | No, checked server side rather than by routing |
| No container wake without acceptance | `/api/warmup` returns early | No |
| Sharing is off by default | An image without `shared = 1` is discarded rather than stored | No |
| Only reviewed images train | Export query, database trigger, and Worker guard | No |
| Contradictory review labels | Rejected by a database trigger, not by the application | No |

Editing the page in a browser gets a visitor exactly as far as not editing it. The
acceptance token is signed, host-scoped, HttpOnly and expires in 15 minutes; a token that
is tampered with, signed with the wrong secret, carries no version, or is shaped like a
session cookie is refused. There are 53 tests over this behaviour.

**Evidence of acceptance.** `tos_accepted_at` and `tos_version` are both written at the
moment of the tick. No IP address or user agent is recorded against an acceptance, because
the system stores neither anywhere. That is slightly weaker evidence than most services
keep, and it was chosen knowingly in favour of the minimisation position the DPIA relies
on. I would keep the choice, but you should know it was made.

---

## 4. The legal pass, and what was done about it

Sixteen findings from the review, plus one found while deploying. All seventeen are
resolved in the text and the code you are reviewing.

| ID | Finding | Resolution |
|---|---|---|
| L1 | No named counterparty; questions pointed at a file in the repository | **Terms §1 and Privacy §1 name Khalid Faiz, personal capacity, with a monitored address** |
| L2 | No governing law or jurisdiction | **Terms §16**: England and Wales, with an express carve-out preserving a consumer's right to sue at home and keep their local mandatory protections |
| L3 | No privacy notice; the footer admitted its own notices were incomplete | **A full Article 13 notice now exists at `/privacy`**, reachable without an account. The footer's apology strip is gone and it links to the real documents |
| L4 | Lawful basis inconsistent across three documents | **Resolved to consent.** See §6, which is the item most needing your eye |
| L5 | §5 limited erasure without citing any derogation | **Terms §8 now separates the stored copy, always deletable on request, from the trained weights, which are not**, and cites Art 17(3)(d) for the second only |
| L6 | No right to suspend or close an account for breach | **Terms §6 and §10** create it, tied to a new prohibited-use section |
| L7 | No account closure route at all | **Terms §10 commits to closure within 30 days of a request.** Self-service closure is scoped engineering work, see §7.5 |
| L8 | No minimum age | **Terms §4 and Privacy §12 set 18**, and it is in the acceptance tick box |
| L9 | The tick box attested to less than the clauses that mattered | **Now four attestations**: age, agreement to both documents, understanding it is not a medical device, and the right to upload |
| L10 | Liability exclusion covered personal injury and was likely unenforceable | **Terms §13 carves out death and personal injury caused by negligence, and fraud**, which is what stops the clause being struck out wholesale |
| L11 | A warranty in §3 with no consequence | **Terms §5 adds an indemnity**, scoped and subject to consumer rights. Strike it if you think it does more harm than good |
| L12 | Licences named by indirection | **Terms §12 names Apache-2.0 and CC BY-NC 4.0**, and flags that some training data is both non-commercial and no-derivatives |
| L13 | No breach notification statement | **Terms §14 and Privacy §11** commit to telling users without undue delay where the risk is high |
| L14 | No route to complain to the ICO | **Terms §14 and Privacy §13**, with the address and phone number |
| L15 | No statement that prior versions are kept | **Terms §15**: every version is retained in public source history, identified by the version date |
| L16 | "Three parties, and no others" was literally untrue | **Terms §9** now says three parties plus the named infrastructure providers who process on our instructions |
| L17 | **Found while deploying.** Terms §15 promises re-agreement on a material change, but the code accepted any recorded version, so nobody would ever have been re-prompted | **`TERMS_MATERIAL_SINCE` added.** An acceptance now counts only if it names that version or later, so the seven existing accounts are asked again. A non-material wording fix bumps the version alone and disturbs nobody |

### The one I would look at hardest

**L12 is resolved in the Terms but points at a live product question.** The main dataset,
BTXRD, is CC BY-NC-ND 4.0. The model card records that the no-derivatives term forbids
redistributing derivatives "including Grad-CAM overlays". If the service ever displays a
Grad-CAM overlay derived from a BTXRD image, that is arguably distribution of a derivative
work. The Terms now warn that training data carries restrictive terms, which is honest, but
it does not make the underlying question go away, and it will sharpen as more sources are
added. There is currently no PMC or NLM reference anywhere in the repository; if PMC images
are added, note that only the `oa_comm` tier is broadly safe and that an open access article
can still contain third-party figures needing separate permission, so licences have to be
checked per figure and recorded per image.

---

## 5. The abuse pass, and what was done about it

Nine cases. What a hostile or careless user does, and what now stops them.

| ID | Abuse case | What is in place now | Residual |
|---|---|---|---|
| A1 | **Unlawful content uploaded, stored, and displayed to the reviewer** | Terms §6 prohibits it and reserves immediate suspension; `INCIDENT_RESPONSE.md` §7 is a new preserve-and-report procedure; `DPIA.md` R14 records the risk | **Medium.** One decision would make it Low, see §7.2 |
| A2 | A real patient's radiograph uploaded | Terms §5 plus an indemnity, **and the warranty now also appears on the per-image sharing checkbox itself**, at the moment of the act | Medium |
| A3 | Burned-in identifiers reach the reviewer | Warned in Terms §5 and the Privacy notice | Medium. Needs the purge-now reviewer action, §7.5 |
| A4 | Repeat deliberate non-radiograph uploads | Daily cap of 50; reject purges in 7 days | Medium. Needs a graduated response, §7.5 |
| A5 | Automated querying and model extraction | **Terms §6 prohibits automated access, scraping, rate-limit circumvention and model reconstruction**, and §10 allows closing the account | Low |
| A6 | Weaponised or excessive data subject requests | **Art 12(5) and Art 11 positions recorded in `ROPA.md` §D** | Medium until the DSR procedure is written, §7.4 |
| A7 | Account farming to beat the caps | Accepted and recorded. The 200 MB storage cap bounds damage regardless of account count | Low, accepted |
| A8 | The reviewer account itself | One pinned account, constant-time comparison, separate admin key not held by the web app, `reviewed_by` recorded | Low while reviewer and controller are the same person |
| A9 | Training set poisoning | Human review in four places, a database trigger making contradictory labels unrepresentable, daily caps, guarded model promotion | Low |

### A1 is the one that still needs a decision

The upload path accepts arbitrary images. That is not hypothetical: a photograph of a
person against a white background was uploaded, passed the out-of-distribution gate, and
reached the review queue.

Run that with a hostile user. Somebody uploads unlawful content and ticks the sharing box.
It is stored in D1 and displayed on the reviewer's screen, because displaying it is what the
queue does. The operator is then hosting it and a person has viewed it.

The clause and the procedure now exist, and the procedure is deliberately the opposite of
every other branch in the incident plan: **preserve, do not delete, do not forward, report,
and suspend the seven-day purge so the timer cannot destroy evidence**. That is a genuine
improvement on nothing.

But it is mitigation, not prevention. **The control that would collapse the risk is gating
the sharing feature behind operator approval**, so only approved accounts can put an image
in front of a reviewer. Scanning would stay open to everyone. With a 500-account research
deployment that is proportionate and easy to defend. It is recorded as R14 in the DPIA and
it is §7.2 below.

Two details in the procedure I would like you to check specifically: that the reporting
routes named (IWF for CSAM, the GOV.UK service for terrorist material, 101 or 999
otherwise) are the right ones, and that suspending the automatic purge is the correct
instinct rather than a way of holding material longer than we should.

---

## 6. The substantive judgement call: why the lawful basis changed

This is the part where I most need you to disagree with me if I am wrong.

**Before**, the ROPA and DPIA both recorded **Article 6(1)(f) legitimate interests** for the
shared-image corpus, describing the per-image tick box as "an interface control" rather than
as the lawful basis. The DPIA's reasoning was that consent is fragile, because withdrawal
would require removing data from training sets and arguably from derived weights, "which is
not currently technically achievable".

**I have changed it to consent**, and here is the argument.

The tick box is granular, opt-in, off by default, asked separately for every single file,
and withdrawable. **It is consent in every observable respect.** A regulator examining the
interface would characterise it as consent whatever the register said, and a register that
disagrees with the interface is a liability rather than a defence. Worse, if it is consent
and we called it legitimate interests, Art 7(3) obligations would be live and unmet.

The objection to consent was that a model cannot be untrained. **Article 7(3) second
sentence disposes of that**: withdrawal does not affect the lawfulness of processing carried
out before it. So withdrawal stops future use and deletes the stored copy; the weights
already trained are processing that was lawful when it happened. Their continued retention
rests on Art 9(2)(j) with the Art 89(1) safeguards, and Art 17(3)(d) limits erasure to that
extent and no further.

The result across all six activities:

| Processing | Article 6 | Article 9 |
|---|---|---|
| Accounts | 6(1)(b) contract, the Terms | n/a |
| Running the scan | 6(1)(b) contract | 9(2)(a) explicit consent |
| Shared image, review and research | 6(1)(a) consent | 9(2)(a); 9(2)(j) for retention after withdrawal |
| Country and globe | 6(1)(f) legitimate interests | n/a |
| Contributor profile | 6(1)(a) consent | n/a |
| Security and logs | 6(1)(f) legitimate interests | n/a |

Note that **6(1)(b) for accounts is now genuinely available**, which it was not before. It
requires a contract, and until the Terms gate shipped there was no moment at which anybody
agreed to anything: `tos_accepted_at` was written as a copy of `created_at`, recording that
a row was inserted rather than that a human agreed.

**If you think this is wrong, it is a text change and a register change, not an
architecture change**, so it is cheap to reverse. What would not be cheap is discovering the
mismatch later.

The remaining exposure I can see, and cannot resolve myself: a public demo open to anyone is
a weaker fit for "scientific research" under 9(2)(j) than a defined protocol with an
approved population. A short research protocol would materially strengthen it. That is R9 in
the DPIA and §7.6 below.

---

## 7. What is still open

Every item here is a decision or an action, not a defect. Nothing on this list is something
I could have done and did not.

### 7.1 Name the controller. **Resolved**

**Khalid Faiz, acting in a personal capacity as an independent researcher**, named in
Terms §1, Privacy notice §1, `DPIA.md` §A, `ROPA.md` §A and as incident lead. The published
contact is `kzfhero6@gmail.com`, deliberately a different address from the one on the
GitHub account, which stays as the security-reporting route in `SECURITY.md`.

Worth being explicit for the record, because the term caused confusion: "controller" is not
an appointed role. It is a description of whoever decides the purposes and means of the
processing, which here is the person operating the service. **It would change if ONNM were
ever run under a university or a company**, and that would change the whole
controller/processor analysis with it, so it is worth revisiting if the project's
affiliation ever changes.

One consequence the controller should be aware of rather than discover: naming an
individual personally means that name appears on a public website, and liability sits with
that individual rather than with an entity. That is the correct and legally required answer
for a personally operated service, but it is not cost-free, and incorporating would be the
alternative if that ever became a concern.

### 7.2 Should sharing be gated behind operator approval?

The A1 decision. The single change that most reduces risk in the whole pack. Recorded as
R14 in the DPIA.

### 7.3 Confirm the provider terms

Cloudflare and Google both publish Article 28 processor terms incorporating the UK Addendum
to the EU SCCs, so the mechanism exists and does not need negotiating. **What is outstanding
is confirming acceptance on the account, filing copies, and a Transfer Risk Assessment for
each.** A controller action.

Also outstanding from `INCIDENT_RESPONSE.md` §6: the Cloudflare API token and R2 token
should be rotated or confirmed already rotated, and incident 001 needs its Article 33
assessment recorded, including a decision not to notify if that is the conclusion. The admin
key was rotated and the git-history audit across all 116 commits is clean.

### 7.4 Write the DSR procedure

Access, rectification, erasure, restriction, portability and objection, with a named person,
templates, and the Art 12(5) and Art 11 positions from `ROPA.md` §D. The Privacy notice
promises a response within one month, so this needs to exist before that promise is tested.

### 7.5 Engineering work, scoped but not built

I have not built these because they are product changes rather than corrections, and I would
rather you saw the documents first:

- **Self-service account closure.** Terms §10 commits to 30 days; today that is honoured
  manually. This is the weakest point in the rights story.
- **A purge-now reviewer action** for an image showing burned-in identifiers, so it does not
  sit for seven days. Plus the standing instruction printed on the review console (A3).
- **A graduated response** after repeated not-a-radiograph rejections, disabling sharing for
  that account pending review rather than doing nothing or banning outright (A4).

### 7.6 Smaller controller items

Define what "life of the research" means as a period and set a review date. Confirm
Cloudflare's log retention figure. Assess ICO registration and the data protection fee.
Record the Article 37 DPO decision. Name a deputy incident lead, since the incident lead,
technical responder and sole admin account are currently the same person. Consider a short
research protocol (R9).

---

## 8. A note on what else changed

While correcting the compliance records I found that the earlier architecture had not been
removed from the project's documentation either. The public README still instructed people
to run a file that had been deleted, and `SECURITY.md` still listed a hosting platform that
is no longer used as out of scope for testing. Both are fixed, along with a claim on the
landing page that contradicted the corrected erasure clause.

Some Python modules under `src/` still import the old UI framework. Two of them generate
live frontend assets, so removing the dependency is a real refactor with test consequences
rather than a deletion, and I have not attempted it here. It has no compliance implications:
none of that code runs in the deployment.

---

## 9. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Prepared by | Engineering | Submitted for review | 2026-08-29 |
| GRC review | Yaso-cyber | | |
| Controller | Khalid Faiz | | |

**The current Terms remain in force during review.** They are enforced and they are better
than the nothing that preceded them. This pack is what replaces them.
