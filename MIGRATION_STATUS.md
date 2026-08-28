# ONNM migration to Cloudflare — status and open issues

**Branch:** restoration prepared on `recovery/cloudflare-parity` for publication to `main`.
**Live:** https://onnm.kali-fz.workers.dev
**Updated:** 2026-08-28. Read `RESTORATION_PLAN.md` before changing anything.

This document exists so a new agent can continue without re-deriving the work. It is
honest about what is verified, what is untested, and what is broken.

---

## 1. Why this migration exists

ONNM ran as a 1,371-line Streamlit app on Streamlit Community Cloud. It is being moved
because of an unhideable Streamlit watermark on public-repo apps, visible lag from
Streamlit's rerun model, and a 0.078-CPU-core / 2.7 GB ceiling that torch + MONAI exceeds.

An Oracle Cloud A1 instance was the intended replacement. After two days Oracle returned
only "Out of host capacity". **That path is abandoned.** DuckDNS existed only because
Oracle hands out a bare IP and is no longer needed.

### The constraint that shapes everything

The owner has stated that the **$5/month Cloudflare Workers Paid subscription is the only
money available on this project, ever**. Do not propose a second paid service, a plan
upgrade, a purchased domain, or any metered add-on.

Before this migration the Cloudflare account had **no payment method**, so overage
physically could not bill. A card is now on file. That safety net is gone, which is why
the spend guards in §6 are load-bearing rather than decorative.

### Why the original plan in `System_migration.md` is wrong

`System_migration.md` has **not been updated** and still describes two inference paths that
do not exist. Both were verified dead against vendor documentation:

- **Cloudflare Workers AI (BYO ONNX)** — Workers Free allows 10 ms CPU per request and a
  3 MB gzipped script. Workers AI has no self-serve custom-model upload; the docs route
  custom models to a "Custom Requirements Form", i.e. enterprise sales.
- **Hugging Face Spaces** — now requires a paid plan to create Gradio or Docker Spaces.
  The free ZeroGPU exception needs an account older than 30 days, which this one is not.

That left Cloudflare Containers, which **cannot be deployed on the Workers Free plan**.

### Why this is one Worker and not Cloudflare Pages

The plan called for Pages + Pages Functions. **Pages cannot host this.** Cloudflare
Containers are addressed through a Durable Object, and the docs are explicit: *"You cannot
create and deploy a Durable Object within a Pages project."* Container bindings are not a
supported Pages binding either.

So the whole site is a single Worker with static assets. This is why the address is
`onnm.kali-fz.workers.dev` rather than `onnm.pages.dev`.

---

## 2. Architecture

```
Browser ──► onnm.kali-fz.workers.dev          Worker "onnm"
              ├── static assets (web/dist, built by Vite)
              ├── /api/*  → worker/index.js       (run_worker_first routes these)
              ├── D1 "onn-model"                  ← the SAME existing database
              └── ONNM_INFERENCE → Durable Object → container (torch, Grad-CAM, OOD)
```

**No database migration happened.** The Worker binds the existing D1
(`961f0440-7ff1-466e-88fe-0c2b30f3083b`, schema version 6) and reaches it through the
original handlers in `cloudflare/src/worker.js`. The old `onnm-community` Worker is still
deployed and untouched, and the Streamlit app still runs, so both remain rollback paths.

### File map

| Path | What it is |
|---|---|
| `wrangler.jsonc` | Worker config. Container, DO, D1, assets, cost guards. |
| `worker/index.js` | API router, Google OAuth, session, scan, warmup, country capture. |
| `worker/container.js` | Container DO class + runtime metering. |
| `worker/lib/session.js` | HMAC-signed `__Host-` session cookies. |
| `worker/lib/google.js` | OIDC authorization-code flow with PKCE. |
| `worker/lib/breaker.js` | Monthly container-runtime budget. |
| `worker/lib/geo.js` | Port of `src/geo.py:build_markers`. |
| `worker/lib/centroids.js` | **Generated** from `src/geo.py`. Do not hand-edit. |
| `inference/service.py` | The one scan contract. Mirrors `app.py` ordering. |
| `inference/main.py` | FastAPI shell around it. |
| `inference/Dockerfile` | CPU torch image. Build context is the **repo root**. |
| `inference/model/` | Staged weights. **Gitignored.** |
| `web/` | Vite frontend. |
| `web/src/globe/globe.js` | **Generated** from `src/components/globe.py`. |
| `web/src/styles/theme.css` | **Generated** from `src/theme.py`. |
| `web/src/styles/components.css` | Hand-written. |
| `scripts/stage_inference_model.py` | Stages the serving checkpoint into the image. |
| `scripts/extract_globe_module.py` | Regenerates `globe.js`. |
| `scripts/extract_theme_css.py` | Regenerates `theme.css`. |
| `scripts/check_inference_parity.py` | Rescued from the Oracle branch. Not yet run. |

Three files are **generated**. Edit the Python source and re-run the script; do not edit
the generated output.

---

## 3. Homepage restoration — implemented locally, deployment verification pending

**Original symptom, now resolved locally:** the hero title rendered, then roughly 480 px
of blank space, and the rest of the page was pushed far below the fold.

**Original cause — diagnosed, not guessed.** `src/theme.py` already defined classes with
the same names the first migrated markup used, but with different structural assumptions. The
generated `web/src/styles/theme.css` therefore contains:

```css
.onnm-hero {
  position: relative; overflow: hidden;
  min-height: 480px; padding: 56px 0 0; margin: 0 -1rem;
}
```

That rule expects child elements `.onnm-hero-bg` and `.onnm-hero-veil` and a CSS variable
`--hero-img`. **`--hero-img` is used but never defined in the stylesheet** — Streamlit set
it at runtime. The first migrated markup had none of that structure, so the hero became an
empty 480 px box.

**Original collision set** (classes the first migrated markup used that `theme.py` styles):

```
onnm-hero          onnm-hero-title    onnm-nav        onnm-footer
onnm-stat          onnm-stat-label    onnm-stat-value onnm-account-name
```

`.onnm-nav` was also a problem: `theme.py` styles it as a full-width flex bar, while the
first migrated markup used it as a small inline group inside `.onnm-header`.

**Applied fix.** The Vite structural classes are now namespaced so they cannot collide,
while generated `theme.css` remains intact for the components that still use it.

To see the collisions:

```bash
used=$(grep -rhoE 'onnm-[a-z0-9-]+' web/src/*.js web/src/pages/*.js | sort -u)
for c in $used; do grep -q "^\.$c[ ,{]" web/src/styles/theme.css && echo "$c"; done
```

The recovery branch restores the final Streamlit composition and wording: mission card on
the left, interactive country globe on the right, CTA, live totals, scan-benefit band,
consented contributor roll, identity treatment, and legal/research footer. Generated globe
CSS is component-scoped, structural classes are namespaced, and the globe now resizes and
renders an honest empty state if its API is unavailable. Desktop and fresh 390px local
reviews passed without horizontal overflow. Four-width deployed screenshots remain open.

---

## 4. Google sign-in — contract repaired, deployed end-to-end retest still required

**Symptom reported:** "Your account could not be opened. Please try again." after
returning from Google. That is the `account_failed` branch in `authCallback`.

**Cause.** `getUserBySubject` in `cloudflare/src/worker.js` returns the user row
*directly* (`json(row)`), not wrapped as `{ user: row }`. `worker/index.js` read
`payload.user`, got `undefined`, concluded no account existed, tried to **create** one,
hit the `UNIQUE` constraint on `email`, received a 409, and mapped that to
`account_failed` — for a user whose account was in the database the whole time.

The existing-account payload bug was fixed in `5593a8b`. The recovery branch additionally
fixes first-time registration by generating and sending the required application `user_id`,
requires `email_verified === true`, preserves the legacy email fallback, and safely recovers
the simultaneous-first-login insert race. Ten focused Node contract tests cover these paths.

Two sibling bugs were also fixed in `5593a8b`:

- **`createSubmission` requires `submission_id`** and `/api/scan` never sent one. **Every
  scan would have failed to record with a 400.** Now generated with `crypto.randomUUID()`.
- **`updateContributorProfile` requires `provider_subject`**, which it re-checks in
  constant time so that holding the API key is not by itself enough to rewrite somebody's
  public profile. It was omitted, making the profile sync a silent 400 behind its own
  `.catch()`.

**Not retested end to end on the deployed Worker.** That remains the first post-deployment
owner check; unit and dry-run success do not prove the Google console, Worker secret, and
live callback are aligned.

**Lesson for whoever continues:** every call from `worker/index.js` into
`cloudflare/src/worker.js` is an undocumented contract. Read the handler before calling it.
The remaining unverified callers are `/globe`, `/contributors`, `/health`,
`/submissions`, `/location/token` and `/location/capture`.

---

## 5. What is verified working

Measured, not assumed.

| Thing | Evidence |
|---|---|
| Worker + assets | All routes 200; deep link `/scanner` returns the shell; JS/CSS serve |
| D1 access | `/api/stats` returns **7 users, 17 submissions, 4 approved, 1 pending** |
| Globe data | `/api/globe` returns GB signup (5) + contributor (2), jittered apart |
| Geo port | JS output **byte-identical** to `src/geo.py` across Tor, undetermined, unlisted, lowercase and zero-count codes; same `unplaced` total |
| OAuth redirect | Correct `client_id`, `redirect_uri`, PKCE `S256`, `state`, `prompt=select_account`; `__Host-` cookie HttpOnly/Secure/SameSite=Lax |
| Container image | Builds clean, 2.25 GB |
| Model in container | Loads in ~10 s; **warm scans ~0.93 s** at 0.5 vCPU / 4 GiB |
| Calibration | `calibrated: true`, temperature **1.410**, threshold **0.4959** |
| OOD gate | Random noise → `is_radiograph: false` and **no `prediction` key at all** |
| Container auth | Unauthenticated `POST /infer` → **401** |
| Cost guards | `max_instances: 1`; `active: 0`, `assigned: 0`; meter at 0 s / 18000 s |
| Network isolation | Container reports `"mode": "private"`, no IPv4/IPv6 — `enableInternet=false` confirmed |

### Never verified

- Four-width visual approval on the newly deployed build.
- **Sign-in end to end.** Was broken; fixed; untested.
- **A live scan through the deployed container.** Requires a session, so it has never run.
- `scripts/check_inference_parity.py` has **not been run** against the deployed container.

---

## 6. Cost model — do not weaken these

`standard-1` = ½ vCPU, 4 GiB, 8 GB disk. At published rates: memory $0.036/h + CPU
$0.036/h + disk $0.002/h ≈ **$0.074 per hour of container runtime**. Left awake
continuously that is roughly **$53/month**.

Five guards, in order of how much they save:

1. `max_instances: 1` in `wrangler.jsonc`.
2. `sleepAfter = "90s"` in `worker/container.js`.
3. **No cron trigger.** A keep-warm heartbeat is the single most expensive thing that could
   be added here. This is the opposite of the right answer on a free host.
4. `worker/lib/breaker.js` — 5 hours/month metered in D1's `meta` table, fails closed.
   Worst case ≈ **$0.37** on top of the $5.
5. A Cloudflare billing notification. **Not yet configured** — see §9.

The meter is written on `onStop`, so a container currently awake reads 0 until it sleeps.
An unclean stop is reconciled on the next `onStart`, capped at 10 minutes.

`meta` is a key/value table; the counter key is `container_seconds_YYYY_MM`, so the budget
resets monthly with no cron job. **No schema migration was needed or made.**

---

## 7. Not yet ported from Streamlit

- Feedback widget and rejection dispute (`src/community_ui.py`)
- Downloadable HTML report (`src/report.py` — already emits a self-contained document)
- GitHub star/fork counts (`src/github_stats.py`); the consented contributor roll is ported
- Full ToS / Privacy / Cookie pages (`src/legal.py`) — only a footer summary exists
- The ROC / threshold-sweep chart (`src/components/charts.py`)
- Vitest equivalents for `test_page_guards.py`, `test_globe_fallback.py`,
  `test_location_capture.py`
- Documentation: `System_migration.md`, `README.md`, `RUN_ME.md`, `overview.md`, and the
  billing comments in `cloudflare/wrangler.toml`, which still assert the account has no
  payment method — that is now false and the comment is load-bearing for anyone reasoning
  about safety.
- `compliance/DPIA.md` and `compliance/ROPA.md` need re-review; the processor list changed.

### Deliberately unchanged

`src/onnm/` (model, inference, Grad-CAM, OOD, calibration), `reports/PRODUCTION` and
`scripts/version_model.py`, `review_app.py` (local-only admin console), the D1 schema, and
the `onnm-community` Worker.

Password sign-in stays local-only. Every hosted account is `auth_provider = 'google'` with
`password_hash IS NULL` enforced by a schema CHECK, so `System_migration.md`'s "passwords
must still work" was already untrue of the hosted deployment.

---

## 8. How to build, deploy and test

```bash
# Stage the serving checkpoint into the image build context.
# Refuses if the sha256 does not match the version marked `serving` in
# model_versions.json. calibration.json MUST land beside best.pt or the model
# silently runs at temperature 1.0 and a 0.50 threshold.
python scripts/stage_inference_model.py

npm run build                              # Vite -> web/dist
npx wrangler deploy                        # includes the 2.25 GB image push
npx wrangler deploy --containers-rollout=none   # code/frontend only, fast

# Test the container locally at the real instance limits:
docker build -f inference/Dockerfile -t onnm-inference:local .
docker run -d --name onnm-test -p 8099:8080 \
  -e INFERENCE_KEY=localtestkey --memory=4g --cpus=0.5 onnm-inference:local
curl http://localhost:8099/health
```

Regenerating the generated files:

```bash
python scripts/extract_globe_module.py   # web/src/globe/globe.js, .css, markup.html
python scripts/extract_theme_css.py      # web/src/styles/theme.css
```

Both assert every substitution, so drift in the Python fails loudly rather than emitting
something subtly wrong.

### Environment notes

- Wrangler **auto-loads `.env`** from the repo root for its own credentials. `.env` holds a
  live `CLOUDFLARE_API_TOKEN` and R2 keys. It is gitignored and was verified absent from
  every commit. It does **not** become Worker vars — confirmed by dry-run.
- Docker needs WSL2. Windows 11 Home has no Hyper-V backend.
- The repo `.venv` has the Python deps; system Python does not.

---

## 9. What the owner still has to do

1. **Rotate the Google client secret.** It was pasted into a chat transcript. Add a new
   secret in the Google console, run `npx wrangler secret put GOOGLE_CLIENT_SECRET`, then
   delete the old one.
2. **Set a Cloudflare billing notification.** Suggested threshold **$8**: high enough that
   normal use is silent, low enough to catch a stuck container.
3. **Retest sign-in** after the §4 fix.

Secrets already set on the Worker: `API_KEY` (reused from `.cloudflare-keys.txt`, not a new
one), `SESSION_SECRET`, `INFERENCE_KEY`, `GOOGLE_CLIENT_SECRET`.

Google OAuth client has three redirect URIs registered: the Streamlit one (rollback), the
unused `onnm.pages.dev` one, and the live `https://onnm.kali-fz.workers.dev/api/auth/google/callback`.

---

## 10. Suggested order of work

1. Rotate the disclosed Google client secret, deploy, then retest existing and first-time
   sign-in. If it fails, check Worker logs — every failure path logs a reason.
2. Capture and approve the deployed page at 1440, 1024, 768, and 390 pixels.
3. Run a real scan; confirm a row lands in `submissions` (the `submission_id` fix is
   untested).
4. Run `scripts/check_inference_parity.py` against the deployed container. It gates
   hosted-vs-local agreement to 1e-4 including the Grad-CAM peak **location**, and is the
   check that answers "did containerising change the diagnosis".
5. Exercise the pinned local admin review flow and exported manifests.
6. Port the remaining scanner/profile features in §7.
7. Rewrite `System_migration.md`; it currently misinforms.

Do not cut over from Streamlit until 1–5 pass.
