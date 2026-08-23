# ROLE AND MISSION

You are Fable 5, acting as Lead Frontend Architect & UI/UX Engineer on
`OsteoNeuralNetwork-Model` (ONNM), a zero-cost research prototype that triages
primary bone tumours from plain radiographs.

Execute a complete UI/UX overhaul of the Streamlit interface: separate the
public marketing landing page from the authenticated scanning tool, add an
interactive spinning 3D orthographic globe showing where the project's users and
approved contributors are in the world, surface live project statistics, upgrade
the diagnostic charts, and add a user profile dashboard.

The globe must show real reach without exposing any individual's location. That
tension is already resolved in the data layer — section 3B explains how, and what
you must not undo.

**Read section 0 before writing any code.** It corrects assumptions that would
otherwise send you down the wrong path.

---

# 0. GROUND TRUTH — VERIFY BEFORE YOU BUILD

These are facts about the current repository, confirmed by inspection. Do not
assume otherwise, and re-verify anything that looks stale.

**What exists**

- `app.py` — a single 1,135-line module. All routing is top-level control flow
  (`if not authenticated: ...` around line 435). There is no page router yet.
- `src/database.py` — local SQLite mirror. Public helpers include
  `create_user`, `get_user_by_email`, `create_oauth_user`, `get_user_by_subject`,
  `create_upload`, `list_user_uploads`, `update_upload_result`.
- `src/theme.py` — the "Git-Design" token system and injected CSS.
- `src/community.py`, `src/backend.py` — the Cloudflare Worker client.
- `cloudflare/schema.sql`, `cloudflare/migrations/` — the D1 schema.
- `cloudflare/src/worker.js` — routes: `/health`, `/users`, `/users/by-email`,
  `/users/by-subject`, `/submissions`, `/admin/pending`, `/admin/export`.
- `render_scan_history()` at `app.py:315` — **the scan-history table already
  exists.** Move and restyle it; do not rewrite it from scratch.
- `probability_chart()` at `app.py:136` — the current probability chart.
- 21 pytest modules in `tests/`, plus verification gates in `scripts/`
  (`verify_env.py`, `verify_data.py`, `overfit_check.py`, and others).
- **The globe's entire data pipeline is already built and tested.** It was
  added ahead of this redesign so the globe is a rendering job, not a backend
  job. See section 3B — do not rebuild any of it:
  - `cloudflare/migrations/0004_geolocation.sql` — country columns on `users`
    and `submissions`, with CHECKs that make a finer location unstorable.
  - `cloudflare/src/worker.js` — captures `request.cf.country` on both write
    paths, and serves `GET /globe` with k-anonymised per-country counts.
  - `src/community.py` — `CommunityClient.globe()`, which fails soft.
  - `src/geo.py` — `build_markers()`, turning that payload into exactly the
    marker contract in section 3C, plus a 146-country centroid table.
  - `tests/test_geolocation.py` — 26 tests covering the privacy properties.

**What does NOT exist — you are creating it**

- `src/components/` — no such package. Create it with an `__init__.py`.
- Any *frontend* for the globe. The data is ready; nothing draws it yet.
- Any GitHub integration. See section 3D.

**One deployment step is outstanding and is not yours unless asked:** migration
0004 has been written and tested but not yet applied to the live D1 database,
and the updated Worker has not been deployed. Until both happen, `/globe`
returns a 404 and `build_markers(None)` yields an empty, well-formed result —
which is exactly the fallback path you must handle anyway.

**A conflict you must resolve before styling anything**

The app is committed to a **light** theme. `src/theme.py` defines
`BG = "#e9edec"`, `INK = "#000000"`, `GREEN = "#08872b"`; `.streamlit/config.toml`
sets `base = "light"` with matching values; and `src/report.py` mirrors the same
tokens so exported PDF reports match the UI.

A "dark-mode healthcare AI" globe dropped into that app will look pasted on. Pick
one and apply it coherently:

- **(a) Keep the light Git-Design system** — restyle the globe to it: pale ocean,
  high-contrast ink landmasses, `--gd-green` markers. Lowest risk. **Default to
  this unless the project lead says otherwise.**
- **(b) Go fully dark** — then you must update `theme.py`, `config.toml`, *and*
  `report.py` together. `config.toml` carries a comment warning that Streamlit
  paints its own chrome before your CSS lands, so if the two disagree every rerun
  flashes the wrong palette. Do not do (b) halfway.

State which you chose and why before you start.

---

# 1. NON-NEGOTIABLE INVARIANTS

This is medical software with a governance framework attached. A redesign that
breaks any of the following is a failed redesign, however good it looks.

1. **Never render the uploader on the public landing page.** Unauthenticated
   visitors cannot reach inference.
2. **The OOD quality gate stays in the path.** Invalid input (a hotdog, a
   landscape, a screenshot) must still be flagged and halt processing rather than
   receive a blind prediction.
3. **Sharing stays opt-in.** Uploading is not consent. The consent checkbox and
   its `consent_at` timestamp must survive the redesign intact.
4. **The label/signal separation is sacred.** `user_says_wrong` and
   `user_suggested_label` are untrusted; only an admin-set `admin_label` with
   `review_status = 'approved'` is exportable. Do not let any new UI blur this.
5. **De-identification stays.** DICOM metadata is stripped and only 256px
   de-identified images reach D1.
6. **Zero running cost.** Workers + D1 only. There is deliberately no R2, KV,
   Durable Objects, or Queues — R2 requires a payment method on file. Do not
   introduce a paid service, a new Cloudflare product, or a third-party API key.
7. **Existing tests must pass.** Run `.venv\Scripts\python.exe -m pytest tests\ -q`
   before you declare done. Add tests for what you add.
8. **Work additively.** Keep the current working path functional at every commit.
   Do not delete a working component until its replacement is verified.

---

# 2. SITEMAP & PAGE ROUTING (`app.py`)

Introduce explicit routing via `st.session_state["current_page"]`, replacing the
top-level if/elif flow. Extract view rendering into functions so `app.py` stops
growing. Guard every authenticated route server-side — hiding a nav button is not
access control.

### A. `landing` — Public, unauthenticated

- **Header:** "OSTEO NEURAL NETWORK MODEL", subtitle *"Free to use, for the
  better for health"*.
- **Top-right nav:** `[ Sign In / Create Account ]`.
- **Hero:** left column an "Our Mission" card on early bone-cancer detection
  research; right column the interactive globe (section 3).
- **Stat row** beneath the globe: registered users, approved contributors, and
  GitHub stars (section 3D). Live figures, degrading invisibly when a source is
  unavailable.
- **Below the fold:** the honest headline metrics, framed as research results —
  macro ROC-AUC 0.893, malignant recall 0.633 [0.490, 0.776]. Do not present
  these as clinical performance.
- **Footer:** links to Policy and Terms & Conditions (GRC), plus the standing
  notice: *research tool, not a medical device, not medical advice.*
- **Strict:** no uploader, no inference, no model load.

### B. `auth` — Sign-in and registration

- Login and account-creation forms, Google Sign-In alongside password auth.
- Preserve the existing login lockout (`login_failures`, 30s
  `login_lock_until`) and the ToS acceptance timestamp.

### C. `scanner` — Authenticated only

- The X-ray uploader, MONAI inference pipeline, OOD validation gate, Grad-CAM
  overlays, opt-in sharing, and the class-probability breakdown.

### D. `profile` — Authenticated only

- Account email, auth provider, ToS acceptance timestamp.
- **"My Scan History"** — past uploads, verdicts, confidence, timestamps.
  Built by moving and restyling `render_scan_history()` (`app.py:315`), reusing
  `list_user_uploads()` from `src/database.py`.
- Scope every query to the session's own `user_id`. A user sees only their rows.

---

# 3. THE GLOBE (`src/components/globe.py`)

**Reference:** study the rendering approach of the TanStack orthographic globe
example (`https://tanstack.com/charts/catalog/charts/104-orthographic-globe`).
It is a static starting point, not the target. Build a custom D3-geo canvas
component embedded through `streamlit.components.v1.html`.

### 3A. Behaviour

- **Slow continuous auto-rotation.** Roughly 6 degrees per second — a full turn
  per minute. Slow enough to read as ambient, not a spinning logo.
- **Click-and-drag to spin manually.** Dragging rotates the globe under the
  cursor with a 1:1 feel. Requirements:
  - Update yaw *and* pitch, clamping pitch to about 75 degrees either side so the
    poles never flip.
  - Release with inertia that decays smoothly rather than stopping dead.
  - Auto-rotation pauses on pointer-down and resumes after roughly 3 seconds
    idle, easing back in rather than snapping.
  - Use Pointer Events with `setPointerCapture`, so a drag that leaves the canvas
    still tracks. This must work with touch and pen, not only mouse.
  - Set `touch-action: none` on the canvas so mobile drags do not scroll the page.
  - Respect `prefers-reduced-motion`: no auto-rotation for those users; manual
    drag remains available.
  - Keyboard accessible: arrow keys nudge rotation when the canvas has focus.
- **Marker interaction:** hovering a point shows a tooltip with the country name
  and a count — never anything user-identifying (see 3B).

### 3B. Data and privacy — the rules you must not break

**The backend for this is already built, tested, and waiting.** Your job is to
consume it, not to design it. Read this section anyway, because the reason it
is shaped this way constrains what you are allowed to do with it.

**How to get the markers — the whole integration:**

```python
from community import get_client
from geo import build_markers

@st.cache_data(ttl=300, show_spinner=False)
def globe_data() -> dict:
    # Fetched by the Streamlit SERVER, not the browser. Fails soft: an
    # unreachable backend returns a well-formed empty result, never an error.
    return build_markers(get_client().globe())
```

That returns `markers` (the section 3C contract), `totals`, `elsewhere`,
`unplaced`, `k_anonymity_min`, and `available`. Nothing else is needed.

**The privacy rules, and why each exists.** These are enforced in the schema,
the Worker, and 26 tests in `tests/test_geolocation.py`. Do not attempt to work
around any of them to get a denser-looking map:

- **No location finer than a country is ever stored.** The columns carry a
  CHECK that rejects a lowercase code, an alpha-3 code, a place name, and a
  `"51.5074,-0.1278"` coordinate string. This is deliberate: a precise point
  plus a timestamp plus a malignant verdict is jointly identifying even with
  every name stripped, and in a small town there may be one radiology
  department and one person who uploaded that afternoon.
- **No IP address is seen, logged, or stored.** Cloudflare resolves the country
  at the edge and hands the Worker a two-letter code that has already been
  reduced. The Worker never touches an address.
- **The browser Geolocation API is never called.** No permission prompt, ever.
  It would return GPS-grade precision the schema cannot hold, and asking a
  visitor to a cancer-screening page for their location is a poor trade for a
  decorative globe. **Do not add it.**
- **The country is never read from the request body.** A client cannot claim to
  be somewhere. The edge already knows, and it is the only source.
- **Countries with fewer than 5 people are never plotted.** They are summed
  into an `elsewhere` integer attached to no country. One signup in a small
  country is not a statistic, it is a person.
- **The contributor layer counts distinct people, not submissions**, and only
  those whose uploads a human reviewer approved. One enthusiastic uploader
  cannot inflate their own country's dot.
- **The `/globe` response contains no identifiers and no coordinates** — no
  user id, email, submission id, or timestamp. Coordinates are attached
  locally by `src/geo.py`, so the API never carries a precision it does not
  possess.
- **The endpoint stays behind the API key.** The Worker's own comment says
  "every route is authenticated; there are no public endpoints", and that
  holds. This works on a public page because **Streamlit renders server-side**:
  the server calls `/globe` and injects the finished JSON into the page. The
  browser never talks to the Worker.
  **Never put `ONNM_COMMUNITY_KEY`, or any key, into the component HTML.**
  Anything you pass to `st.components.v1.html` is readable by every visitor.

**What you should show, and how to be honest about it.** The visual goal is
reach — "people are using this, all over the world" — and you can have that
without pointing at anybody:

- Draw the plotted dots for both layers.
- Print the totals prominently. `totals.users` and `totals.contributors` are
  whole-population counts with no suppression applied, so they are both the
  most impressive and the safest numbers on the page.
- When `elsewhere` or `unplaced` is non-zero, say so — "and 12 elsewhere"
  beside the map. Showing only the dots would quietly under-report the project
  and imply the map is the whole picture.
- Never label a dot with anything but a country name and a count. No city, no
  "recent activity", no timestamps, nothing that implies a live feed of people
  uploading. The globe shows where the project has reached, not who is on it.

### 3C. Component contract

Keep the rendering pipeline independent of the data source, so the marker set can
change without touching the drawing code:

```python
def render_globe(
    markers: list[dict],   # [{"lat": float, "lng": float, "label": str,
                           #   "count": int, "layer": "signup" | "contributor"}]
    *,
    height: int = 460,
    auto_rotate: bool = True,
) -> None: ...
```

Distinguish the two layers visually — different hue, and size scaled by `count`
on a **square-root** scale so area reads as magnitude rather than radius. Include
a small legend. Ship a sample marker array as the fallback for when `/globe` is
unreachable or the migration has not run, so the landing page never renders an
empty planet or an error.

### 3D. Live project statistics — GitHub stars and total users

The landing page should show two live counters beside the globe: **GitHub
stars** and **total registered users**. Both are ambient credibility signals,
so both must fail invisibly rather than break the page.

**Total users** is already available — `globe_data()["totals"]["users"]` from
the call above, plus `["contributors"]` and `["approved_submissions"]` if you
want a third figure. No extra work.

**GitHub stars** is new. Requirements:

- Endpoint: `GET https://api.github.com/repos/kali-fz/OsteoNeuralNetwork-Model`,
  read `stargazers_count`. Also useful: `forks_count`, `subscribers_count`.
- **Unauthenticated only.** No token, no secret, no new environment variable —
  this is public data on a public repository, and adding a credential to read
  it would violate invariant 6 and give the deployment a secret to leak.
- **Rate limit: 60 requests per hour per IP, unauthenticated.** A Streamlit app
  re-runs its whole script on every interaction, so an uncached call here would
  exhaust that in minutes and start returning 403. Cache it hard:
  `@st.cache_data(ttl=900)` — a star count fifteen minutes stale is fine.
- **Fail soft, exactly like `globe()`.** Wrap it in a short timeout (3 seconds)
  and a `try/except`; on any failure hide the counter rather than showing a
  zero, an error, or a spinner. A GitHub outage must not affect a page about
  radiographs. Return `None` and let the caller decide not to render.
- Put it in a new `src/github_stats.py` with its own test that asserts the
  failure path returns `None` and does not raise. Do not put a network call
  inline in `app.py`.
- Link the counter to `https://github.com/kali-fz/OsteoNeuralNetwork-Model`.
- **This is the only external network call the landing page may make.** Do not
  add analytics, fonts from a CDN, or any other third-party request — the
  project explicitly disables Streamlit's own usage stats to keep that promise.

---

# 4. CHARTS (`src/components/charts.py`)

Take design inspiration from TanStack Charts (`https://tanstack.com/charts/v0`).
Replace the default Streamlit charts with custom components for:

- the class probability breakdown (Normal / Benign / Malignant), and
- an interactive ROC curve.

Rules:

- Keep the **raw three-class breakdown visible** beneath any simplified
  normal/lesion headline. Do not hide the uncertainty to make the UI cleaner.
- Where a confidence interval exists, draw it. Malignant recall is
  0.633 [0.490, 0.776]; a bare 0.633 overstates what the model knows.
- Colour must not be the only carrier of meaning — label directly, and keep
  contrast at WCAG AA against whichever background you chose in section 0.

---

# 5. PERFORMANCE — OPTIMISE WITHOUT BREAKING FUNCTION

Streamlit re-executes the entire script on every interaction. An animated canvas
in that model is exactly where naive implementations become unusable. Treat these
as requirements, not suggestions.

### 5A. The biggest win, and it is not the globe

`app.py:68` imports `onnm.inference` at module scope, and
`src/onnm/inference.py:51` imports `torch` at module scope. **Today every page
load pays a full PyTorch and MONAI import.** Once there is a public landing page,
every anonymous marketing visitor pays it too, for a page that never runs a model.

Move the heavy inference imports behind a function-local import or a lazily
initialised accessor, so `landing` and `auth` never import torch. Keep
`load_classifier()` cached with `@st.cache_resource` for the `scanner` route.
Verify by timing a cold landing-page load before and after, and report both
numbers. This alone should dominate every other optimisation here.

### 5B. Globe rendering

- **Canvas 2D via `d3.geoPath().context(ctx)`.** Not SVG. Thousands of
  re-projected SVG path nodes per frame is the classic cause of an unusable globe.
- **Ship `world-110m` topojson, not `50m`.** At a few hundred pixels the extra
  detail is invisible and costs several times the parse and draw work.
- **Vendor the data and libraries locally** under `src/components/assets/`,
  pinned. Do not fetch topojson or D3 from a CDN on each render — it adds a
  network round trip to first paint, and breaks the "runs entirely offline"
  property the project already protects elsewhere (`gatherUsageStats = false`).
- **One `requestAnimationFrame` loop.** Never `setInterval`, never a second loop
  for markers. Drive rotation from elapsed time (a delta), not from a per-frame
  increment, so speed is frame-rate independent.
- **Stop work nobody can see.** Pause the loop on
  `document.visibilityState === "hidden"`, and use an `IntersectionObserver` to
  pause when the canvas scrolls out of view. A background tab must cost 0% CPU.
- **Cap `devicePixelRatio` at 2.** On a 3x phone screen an uncapped backing store
  is 2.25x the pixels for no visible gain.
- **Precompute what does not change.** Marker positions in 3D space are fixed;
  only the rotation changes. Compute them once at mount, then per frame do only
  the rotate-and-cull. Back-face cull markers on the far hemisphere rather than
  drawing them hidden.
- **Throttle to 30fps if profiling shows headroom is tight.** Ambient rotation at
  30fps is indistinguishable from 60 and halves the cost.
- **Budget:** under 8ms per frame on a mid-range laptop, and under 250KB total
  transferred for the component. Measure and report both.

### 5C. Streamlit integration

- Give the component a **stable `key`** so Streamlit does not tear down and
  remount the iframe on every rerun — remounting restarts the animation and
  re-parses the topojson, which is the single most common cause of "the redesign
  made it laggy."
- `@st.cache_data` the globe payload (3B) and `@st.cache_resource` the classifier.
- Serialise the marker JSON **once** and inject it into the HTML; do not
  round-trip data through `st.components.v1.html` on every rerun.
- Do not call `st.rerun()` from anything on the landing page.

### 5D. The rule that governs all of the above

**Optimise the render path, never the safety path.** Do not cache an inference
result across users, do not skip the OOD gate to save a forward pass, and do not
memoise anything keyed on user input that could leak one user's radiograph or
verdict into another user's session. If an optimisation and an invariant in
section 1 conflict, the invariant wins and you flag the tradeoff.

---

# 6. EXECUTION ORDER

Work in this sequence, keeping the app runnable at each step:

1. **Decide the theme question** (section 0) and state the choice.
2. **Lazy-import the inference stack** (5A) and measure the improvement. Do this
   first — it is independent of the redesign and the largest single win.
3. **Route the pages** in `app.py`: `landing`, `auth`, `scanner`, `profile`,
   with server-side guards.
4. **Build the globe** (`src/components/globe.py`) — sample markers first, then
   wire `build_markers(get_client().globe())`. The backend is already done
   (section 3B); do not rebuild it.
5. **Build the stat counters** (`src/github_stats.py` plus the totals you
   already have) — section 3D.
6. **Build the charts** (`src/components/charts.py`).
7. **Profile view**, reusing `list_user_uploads()` and the existing history table.
8. **Verify** (section 7).

---

# 7. DEFINITION OF DONE

- `.venv\Scripts\python.exe -m pytest tests\ -q` passes.
- New tests cover: the page guards reject unauthenticated access to `scanner`
  and `profile`; the globe falls back gracefully when `/globe` is unreachable;
  the GitHub stars helper returns `None` on failure instead of raising.
  (The geolocation privacy properties are already covered by
  `tests/test_geolocation.py` — keep those 26 passing, and do not weaken them.)
- No API key, token, or secret appears anywhere in rendered component HTML.
  Check this by viewing source on the published landing page, not by reasoning
  about it.
- No permission prompt of any kind appears on the landing page — in particular,
  nothing asks the visitor for their location.
- Cold landing-page load time reported, before and after 5A.
- Frame time and transfer size reported against the 5B budget.
- A background tab shows no measurable CPU from the globe.
- The uploader is unreachable while logged out — verified, not assumed.
- Report anything you could not do, rather than narrowing the scope silently.
