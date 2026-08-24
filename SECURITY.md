# Security Policy

ONNM handles medical images and account data. This document says how to report a
vulnerability, what is in scope, and what must not be tested.

## Reporting a vulnerability

Report privately. **Do not open a public issue for a security defect.**

- **Preferred:** GitHub [private vulnerability reporting](https://github.com/kali-fz/OsteoNeuralNetwork-Model/security/advisories/new)
  on this repository.
- **Alternative:** email `kzfhero@gmail.com` with `ONNM SECURITY` in the subject.

Please include what you found, how to reproduce it, what an attacker could reach, and
any log or request evidence. If you believe personal data or radiographs were exposed,
say so explicitly and early — that changes the response clock (see below).

### What to expect

| Stage | Target |
|---|---|
| Acknowledgement | 3 working days |
| Initial assessment and severity | 10 working days |
| Fix or documented mitigation | Depends on severity; we will keep you updated |
| Public disclosure | Coordinated with you, after a fix is available |

This is a zero-budget research project maintained by a small team. There is **no bug
bounty** and no payment. Credit in the advisory is offered unless you prefer otherwise.

### If personal data may have been exposed

Exposure of accounts or radiographs is a potential personal data breach under UK GDPR
Article 33, which carries a **72-hour notification deadline to the ICO from the point of
awareness**. Reports that mention data exposure are triaged immediately and ahead of the
timetable above.

## Scope

### In scope

- The Cloudflare Worker at `onnm-community.kali-fz.workers.dev` and the code in
  `cloudflare/src/`.
- The application code in this repository: `app.py`, `src/`, `scripts/`.
- The Streamlit application logic as deployed, including authentication, session
  handling, upload validation, and the community submission path.
- Dependency vulnerabilities that are actually reachable from this project's code.

Findings we particularly want: authentication or authorisation bypass on any Worker
route; anything that lets one account read another's submissions; anything that reaches
the admin review or export path without the admin key; injection into D1; and any way to
make the app store an image the user did not consent to share.

### Out of scope — do not test

**Only test infrastructure this project controls.** Testing third-party infrastructure
without written authorisation may be an offence under the **Computer Misuse Act 1990**,
and it is not authorised by this policy:

- **Streamlit Community Cloud** — the hosting platform.
- **Google** — the identity provider.
- **Cloudflare's own platform**, as distinct from our Worker's application logic.
- **GitHub**.

Report platform issues to those vendors directly under their own disclosure programmes.

Also out of scope:

- Denial of service, load testing, or anything that degrades the service for others. The
  Worker enforces deliberate caps (`MAX_USERS`, per-user daily submission limits, a total
  storage ceiling); please do not exhaust them.
- Automated scanner output with no demonstrated impact.
- Missing hardening headers with no exploit path.
- Social engineering of contributors or users.
- Uploading real patient data to demonstrate anything. **Never** send identifiable
  radiographs; use synthetic or properly de-identified images.

## Known limitations, stated up front

These are design decisions, not undiscovered vulnerabilities, and are documented in
`grc_compliance_prompt.md`:

- The hosted demo is a **research prototype**, not a clinical system, and is deliberately
  capped as a test deployment.
- Images shared to the community loop are stored in D1 as base64 under Cloudflare's
  platform encryption; **ONNM adds no encryption of its own on top**.
- The local SQLite database is **not encrypted at rest**; full-disk encryption is the
  operator's responsibility.
- De-identification cannot remove identifiers **burned into image pixels**.
- A single pinned admin account holds review and export rights. This is a deliberate
  privilege restriction and a known key-person risk.

## Supported versions

Only the current `main` branch and the currently deployed Worker are supported. There are
no long-term support branches.

## Secrets

If you find a credential in this repository, its history, an issue, a screenshot, or any
published artefact, report it through the channel above and **do not use it**. Secrets are
kept in `.env` and `.streamlit/secrets.toml`, both gitignored; D1 dumps are gitignored by
pattern because `wrangler d1 export` writes them into `cloudflare/` by default and they
contain every user's email, password hash, and shared radiographs.
