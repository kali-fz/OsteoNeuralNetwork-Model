# ONNM — Status & Backlog

Companion to `overview.md`. Checked items are verified done, not assumed.
Last audited: 2026-08-22 (Google Sign-In hosted chooser check).

**Current state:** 311 tests green in the ROCm `.venv`, repo-wide ruff clean. Cloudflare is
**deployed and verified live**; Streamlit Cloud is **deployed and serving**. Google
Sign-In is code-complete and the hosted app now reaches Google's account chooser; **the
remaining end-to-end check is selecting an approved account and confirming the D1 row**.

---

## HANDOVER — read this first

The last action taken was checking the hosted app after the HTTP 500 fix. The app now
reaches Google's account chooser, so the missing-`httpx` failure is cleared. Pick up at
account selection and D1 persistence.

### What is already true

- Worker live at `https://onnm-community.kali-fz.workers.dev`; D1 `onnm-community`
  (id `961f0440-7ff1-466e-88fe-0c2b30f3083b`) migrated to schema_version 2.
- App live at `https://osteoneuralnetwork-model-af5ynv9qxg7u8rc5epdprr.streamlit.app`.
- D1 is **empty and clean** (`users=0, submissions=0`) — all diagnostic rows removed.
- Streamlit secrets are correct. The TOML was parsed and all five `ONNM_*` keys verified
  at top level, none absorbed into `[auth]`. Do not re-litigate this.
- Google Cloud project `onn-model`, OAuth client created, redirect URI
  `.../oauth2callback` registered, consent screen in **Testing** mode (so only listed
  test users can sign in — check this before believing an "access blocked" error).

### The two bugs found and fixed this session

1. **Account creation failed with an opaque `CommunityError`.** Cloudflare's edge bans the
   default `Python-urllib/3.x` User-Agent with a 403 and a plain-text `error code: 1010`
   body, before the Worker is reached. Fixed by `community.USER_AGENT`. **Verified fixed
   against live D1** (users went 0 → 1 through the real client path).

2. **Google sign-in returned a bare HTTP 500 `Internal server error.`** Root cause is a
   missing `httpx`: `authlib.integrations.starlette_client` imports it, Authlib does not
   declare it, and Streamlit's docs only say "install Authlib". The
   `ModuleNotFoundError` is raised in `_create_oauth_client`, which sits *outside* the
   `/auth/login` route's `try/except`, so it escapes as an unhandled 500 rather than the
   route's tidy 400. Reproduced locally in a clean venv: `Authlib` + `starlette` alone
   fails with `MISSING MODULE: httpx`; adding `httpx` makes the import succeed.
   **Fix pushed (`httpx>=0.27`, `itsdangerous>=2.1` in `requirements.txt`) but NOT yet
   confirmed on the deployed app.**

### Next action (in order)

1. **Dependency fix confirmed live.** The hosted app reaches Google's account chooser,
   which proves the former missing-`httpx` 500 is no longer occurring. If this regresses
   after a future dependency change, use Manage app → logs to confirm the install; saving
   secrets alone only reboots and does not reinstall dependencies.
2. **Have the user select an approved Google test account and complete sign-in** (they
   must — it needs their credentials).
3. **Confirm it reached D1, do not assume.** `users` must move 0 → 1:
   ```
   curl -H "Authorization: Bearer $ONNM_COMMUNITY_KEY" \
        -H "User-Agent: ONNM-Streamlit/1.0" \
        https://onnm-community.kali-fz.workers.dev/health
   ```
   The User-Agent header is not optional — see bug 1.
4. If sign-in still 500s, get the traceback from Manage app → logs. Do not guess; the
   route deliberately hides its error from the client.

### Error → cause map for the sign-in flow

| Symptom | Cause |
|---|---|
| HTTP 500 `Internal server error.` at `/auth/login` | missing `httpx` (or Authlib) |
| HTTP 400 `Authentication error` | failure *inside* `authorize_redirect` — provider config |
| `redirect_uri_mismatch` | URI in Google console differs by even one character |
| `Access blocked` / not a test user | address not on the Testing-mode allowlist |
| `invalid_client` | client id or secret pasted wrong |
| Password forms still showing | `[auth]` not read — app has not restarted, or TOML malformed |
| Signs in fine but `users` stays 0 | `ONNM_COMMUNITY_KEY` absorbed into `[auth]` by TOML ordering |

---

## Done

### Infrastructure
- [x] Python 3.12 `.venv`, ROCm 7.2.1 stack on RX 7900 XT (`torch 2.9.1+rocm7.2.1`)
- [x] Package layout `src/onnm/`, editable install, ruff + pytest config
- [x] YAML config system with deep-merge overrides and named profiles
- [x] **311 tests**, synthetic fixtures, no dataset required; the torch-free
      auth/storage/OOD/report/metrics subset runs without torch (verified on a
      clean 3.13 interpreter). Full suite re-run in the ROCm `.venv` after the
      merge and the community work — all green.
- [x] CI workflow (`.github/workflows/ci.yml`): ruff lint, torch-free fast tests,
      full suite on CPU torch — gates 1–2 stay local-only (GPU/dataset)
- [x] `streamlit` in both requirements files; `rocm-sdk init` step documented in
      `requirements-rocm.txt` with its failure symptom
- [x] Notebook lint clean (`E501`/`B905`/`I001` fixed; `E402` per-file-ignored — the
      `sys.path` bootstrap must precede imports); `ruff check .` is 0 errors repo-wide
- [x] `.env` added to `.gitignore` — it was fully committable before, and holds
      API keys and backend credentials
- [x] Diagnosed the MIOpen training-BatchNorm defect; workaround `train.miopen: false`
- [x] `verify_env.py` gate 1 now runs a real train-mode forward/backward, so that
      defect is caught in seconds rather than 40 minutes into a run
- [x] Measured VRAM/throughput across batch sizes; established 64 as the optimum

### Data
- [x] BTXRD download + verification; class counts reproduce the paper exactly
- [x] Label derivation from one-hot indicator columns
- [x] Surrogate patient grouping; **verified zero group leakage** across all split pairs
- [x] DICOM handling: modality LUT, VOI LUT, MONOCHROME1 inversion (test-pinned)
- [x] Splits 2675 / 535 / 536, stratification holds to 0.1%

### Model & training
- [x] DenseNet-121, ImageNet-pretrained, 3-class head
- [x] Focal loss with tempered inverse-frequency alpha (`alpha_beta`)
- [x] `WeightedRandomSampler` + guard refusing it alongside a weighted loss
- [x] Class-asymmetric OHEM (`HardNegativeMiningLoss`) with warmup, budget cap, normalise
- [x] Aggressive augs: `RandAffine`, `RandCoarseDropout`, `RandHistogramShift`
- [x] Optional foreground cropping + guard disabling localisation scoring under it
- [x] Cosine and `ReduceLROnPlateau` schedulers; early stopping on macro ROC-AUC
- [x] Per-epoch tracking: loss, ROC-AUC, PR-AUC, sens, spec, balanced acc, F1, OHEM count
- [x] **Thermal governor** — AMD ADL via ctypes, hotspot control temp, separate memory
      ceiling, hysteresis, bounded pause. Zero cost, no new dependency.

### Calibration & metrics
- [x] Temperature scaling — guarded fit (grid + ternary + LBFGS); plain LBFGS was
      diverging up to 25× silently
- [x] Dual-mode threshold search: sensitivity-floor and specificity-floor
- [x] ECE / NLL, bootstrap CIs, clinical error breakdown, macro F1
- [x] Conflicting-constraint reporting instead of silently picking one

### App
- [x] Streamlit UI: upload → verdict → confidence → Grad-CAM, DICOM/PNG/JPEG
- [x] **Production checkpoint pinning**: `reports/PRODUCTION` marker file names the
      default run; throwaway runs (`smoke-`/`tmp-`/`debug-`) hidden from the dropdown;
      stale pin fails loudly instead of silently falling back
- [x] Auto-loads `calibration.json`; falls back to newest non-throwaway run when unpinned
- [x] **Batch/folder upload**: multi-file uploader, per-file OOD rejection report,
      batch summary table, per-case detail view; content-hash dedupe across reruns
- [x] **Per-case HTML report export** (`src/report.py`) — verdict, probability table,
      embedded original + Grad-CAM overlay, calibration note, full disclaimer,
      print-to-PDF styling; plus overlay-PNG and JSON export buttons
- [x] **Interactive ROC threshold-sweep chart** in the sidebar (altair, reads
      `threshold_sweep.json` written by `scripts/calibrate.py --sweep`)
- [x] Loopback-only bind, telemetry off
- [x] Medical disclaimer; calibration state and warnings surfaced in sidebar
- [x] **Local accounts**: SQLite `data/users.db`, salted PBKDF2-HMAC-SHA256 (600k
      iters), ToS-acceptance timestamps, login throttling, logout clears session
- [x] **De-identified upload storage**: `data/user_uploads/{uuid}/`, UUID filenames,
      DICOM PII-strip (private tags, PN/date fields, regenerated UIDs), standard
      images re-encoded to metadata-free PNG; "My Past Scans" per-user history
- [x] **GRC legal framework** (`src/legal.py`): ToS, Privacy Policy, Medical
      Disclaimer, Cookie Notice, rendered as expandable footers + consent checkbox
- [x] **OOD gate stage 1**: pre-inference radiograph validator (`onnm.ood`) — color,
      dynamic-range, histogram-entropy, edge-density, size checks; hard "Invalid
      Image" rejection in the app with per-check reasons
- [x] **OOD gate stage 2**: predictive-entropy + max-prob uncertainty gating; lesion
      calls below the 0.65 floor or at/above 0.90 normalized entropy render as
      "Non-Diagnostic / Inconclusive", never as a finding

### Evaluation & documentation
- [x] **Calibration reliability diagrams** (`reliability_bins` + `plot_reliability_diagram`),
      auto-included in every run's HTML report alongside scalar ECE
- [x] **Stratified metrics engine** (`stratified_metrics`) + `scripts/stratified_report.py`
      for per-anatomy and per-subtype error tables (script needs the GPU box to run)
- [x] **`MODEL_CARD.md`**: intended use, training data, measured performance,
      limitations, known failure modes, ethical considerations
- [x] **TTA support**: `collect_logits(..., tta_hflip=True)` + `scripts/ablate_tta.py`
- [x] **Backbone freezing**: `train.freeze_backbone_epochs` config knob
      (`set_backbone_trainable` / `head_parameters` in `onnm.model`)
- [x] **`configs/specificity_tuning.yaml`**: 320px + `alpha_beta 0.5` + 3-epoch freeze
      profile, ready to run on the GPU box

### Portability & Colab
- [x] `configure_backend` no longer disables cuDNN on CUDA. `train.miopen: false` is a
      ROCm-only workaround, but it sets the *same* torch flag on both backends, so every
      Colab run reusing `full_run.yaml` or `overnight.yaml` would have trained with cuDNN
      off — silently, and at several times the cost
- [x] bf16 capability check + `GradScaler`. Colab's free T4 (sm_75) has no bf16 at all, and
      the loop had no scaler because bf16 never needs one. `resolve_amp_dtype` falls back to
      fp16 loudly; the scaler is enabled only for fp16, so the local bf16 path is unchanged
- [x] Effective AMP dtype recorded in the run result — a run that claims bf16 when fp16
      happened is not comparable to one that means it
- [x] `colab` profile (no `paths:` block — `verify_data.py` and `make_splits.py` take no
      `--profile`, so moving `data_root` there would desynchronise training from its gates)
- [x] `configs/ablations/ohem_only.yaml` and `augs_only.yaml` — one variable each
- [x] `notebooks/colab_train.ipynb`
- [x] Dataset staged to Drive: `BTXRD.zip` (0.84 GB, 5614 entries, CRC-verified,
      3746 images) + `splits.json` (`content_hash db908a9afdc5d085`, asserted by the
      notebook so a Colab run cannot silently use a different partition)

### Community loop & hosting
- [x] Cloudflare Worker + D1 schema, free tier only (no R2, no payment method)
- [x] Review gate enforced in three places; a schema trigger blocks approving an
      unlabelled or unshared row
- [x] Spend guards in code: 200 MB storage cap, 50 submissions/user/day, 500 accounts,
      1.5 MB body, 600 KB image
- [x] `src/community.py` client, fails soft so a dead API never blocks inference
- [x] `src/backend.py` — accounts in D1 when configured, local SQLite otherwise
- [x] Opt-in consent (default off), feedback widget, admin review queue
- [x] `scripts/export_batch.py` -> `controls_manifest` CSV; rows pinned to train split
- [x] `dataset.py` manifest loader widened past label 0 so reviewed benign/malignant
      submissions can be used (label-0-only manifests behave exactly as before)
- [x] HF Space config (`deploy/hf-space/`) — CPU torch, binds 0.0.0.0:7860
- [x] Colab notebook cell to pull an approved batch
- [x] Opt-in consent wired into `app.py` (default off, scoped per upload, not remembered)
- [x] Feedback widget under the verdict; writes only untrusted columns
- [x] Admin review queue in the sidebar, gated on `ONNM_ADMIN_KEY` — approving requires
      choosing a label, there is deliberately no "approve as-is" button
- [x] `auth.py` now imports from `backend`, so hosted accounts survive a Space restart
      while PBKDF2 hashing stays the single tested implementation
- [x] Invariant proved against the real schema: approving without a label, approving an
      unshared row, and `admin_label='hotdog'` are all rejected by the database; a row
      the user called `benign` and the reviewer called `normal` exports as `normal`
- [x] Verified end-to-end locally: app serves HTTP 200 with the community client disabled
      (degraded mode), i.e. a dead API cannot block inference
- [x] `requirements.txt` at repo root (CPU torch) — Streamlit Community Cloud reads only
      that path; installing it locally would replace the ROCm build, so it says so
- [x] `src/checkpoint_fetch.py` — boot-time checkpoint download, verifies torch zip magic
      so a CDN 404 returning an HTML page is refused rather than written to `best.pt`
- [x] Fixed a case-collision bug found by its own test: a run directory named
      `production` and the `reports/PRODUCTION` marker are the same path on Windows and
      macOS, so writing the marker opened a directory as a file. Default run is `hosted`.

### Deployed and verified live
- [x] **Cloudflare deployed**: D1 `onnm-community` created, schema applied, `API_KEY` /
      `ADMIN_KEY` set, Worker deployed, `workers.dev` subdomain `kali-fz` created
- [x] Auth boundary proved on the live API: no key → 401, wrong key → 401,
      **app key → `/admin` → 403**, admin key → `/admin` → 200
- [x] **Streamlit Community Cloud deployed** and serving
- [x] **Fixed the Cloudflare edge 1010 block.** The default `Python-urllib` User-Agent is
      banned before the Worker is reached; `curl` and browsers pass, so testing with curl
      confirms the wrong conclusion. `community.USER_AGENT` fixes it, and a non-JSON error
      body now names the gateway instead of surfacing a bare `error code: 1010`
- [x] **Account creation verified against live D1** — `users` 0 → 1 through the real
      client path, then cleaned back to 0

### Google Sign-In
- [x] `src/oauth.py` — Streamlit native OIDC; refuses an unverified Google email
- [x] Schema + migration `0002_google_oauth.sql`: `auth_provider`, `provider_subject`,
      nullable `password_hash`, CHECK constraint pairing them, partial unique index on
      subject. Applied to live D1 (schema_version 2)
- [x] `src/database.py` equivalent rebuild for local installs, guarded to run once;
      **migration test proves existing password accounts survive it**
- [x] Worker is provider-aware: rejects a hybrid account, a subject-less federated
      account, and an unknown provider — all verified against the live API
- [x] `verify_password` returns False for a non-string; `authenticate_user` treats a
      federated account exactly like an unknown one, so the login form cannot be timed
      to discover which addresses use Google
- [x] Identity keyed on `sub`, not email; an existing password account is returned, never
      silently converted
- [x] Privacy policy corrected — it claimed "ONNM has no hosted backend" and "data stays
      on the local machine", both false once hosted. Now names Streamlit, Google and
      Cloudflare and states what each receives
- [x] `.streamlit/secrets.toml` added to `.gitignore` — it was committable in a public
      repo and would hold the Google client secret and cookie secret
- [x] `httpx` + `itsdangerous` pinned. Authlib's Starlette integration imports httpx but
      does not declare it, and the resulting error escapes the login route's handler as a
      bare HTTP 500 (**fix pushed, not yet confirmed live — see handover**)

### Runs
- [x] `full-20260822-041653` — trained, calibrated, test-evaluated (**current best**,
      val macro ROC-AUC 0.8905; this is what should be pinned as PRODUCTION)
- [x] `overnight-20260822-055132` — trained only; **regressed** to 0.8629
- [ ] `abl-ohem` / `abl-augs` — **running on Colab now.** No `reports/` directory has
      appeared in `MyDrive/OSTEONEURALNETWORK/` yet, so the notebook's final
      save-back cell has not run. Colab wipes `/content` on disconnect, so if the
      session drops before that cell the results are gone — run it before closing
      the tab.

---

## To do

### Next — finish the login flow

- [x] **Confirm the `httpx` fix landed on Streamlit Cloud.** The hosted app reaches
      Google's account chooser; selecting an approved account and confirming D1
      persistence is still pending.
- [ ] **Walk the loop once, before inviting anyone else.** Sign in → upload with sharing
      ticked → flag the result wrong → review and label it in the sidebar →
      `python scripts/export_batch.py --dry-run` → confirm the row appears with *your*
      label, not the model's or the user's. Nothing below this line has been exercised
      against real data yet.
- [ ] **Decide whether to publish the OAuth consent screen.** It is in Testing mode, so
      only listed test users can sign in (cap 100). Publishing opens it to any Google
      account; with only `openid`/`email`/`profile` that needs no Google verification,
      but testers see an "unverified app" interstitial either way.
- [ ] **Rotate the credentials shared during setup.** The Cloudflare API token and the R2
      token were both pasted into a chat transcript, and `.env.backup-1787434019` still
      holds the old R2 token — delete it.

### Done — deployment (was: "all need logins I do not have")

- [x] **Deploy Cloudflare.** Every command is in `cloudflare/README.md`; ~10 minutes.
      ```
      cd cloudflare
      npx wrangler login
      npx wrangler d1 create onnm-community      # paste the id into wrangler.toml
      npx wrangler d1 execute onnm-community --remote --file=./schema.sql
      npx wrangler secret put API_KEY            # app key
      npx wrangler secret put ADMIN_KEY          # your key, kept off the Space
      npx wrangler deploy
      ```
      Do **not** add a payment method. With no card on the account Cloudflare cannot
      bill — overage fails closed instead.
- [x] **Deploy to Streamlit Community Cloud** — <https://share.streamlit.io>, sign in
      with GitHub, main file `app.py`. Full instructions in
      `deploy/streamlit-cloud/README.md`. Free, 2.7 GB RAM, redeploys on push.
      **Hugging Face Spaces is no longer viable:** Gradio and Docker Spaces now require
      PRO and Streamlit is not offered at all; only Static (client-side, no Python)
      remains free. `deploy/hf-space/` has been removed.
- [x] **Host the 28 MB checkpoint somewhere fetchable** and set `ONNM_CHECKPOINT_URL`
      (+ `ONNM_CALIBRATION_URL`). `reports/` is gitignored so a clone has no model;
      `src/checkpoint_fetch.py` downloads one at boot and pins it. A Hugging Face
      *model* repo is free even though compute Spaces are not, and makes the model
      independently usable; GitHub Releases also works.
- [?] **Point it at `full-20260822-041653`**, not the overnight regression.
      `ONNM_CHECKPOINT_RUN = "hosted"` and both URLs point at release `v0.1.0` — but
      **nobody has verified which run's weights are actually in that release**. Confirm
      it before trusting a hosted verdict: the overnight checkpoint regressed to 0.8629
      macro ROC-AUC and would be served silently if it were the one uploaded.
- [x] **Push `main`.**
- [x] **`MODEL_CARD.md` performance section** — malignant recall **0.633 [0.490–0.776]**
      now leads the section, stated as "roughly one in three malignant films is missed"
      with the explicit warning that a normal verdict is weak evidence of absence. The
      rest of the card was already accurate (not-a-medical-device, CC BY-NC-ND, unscored
      Grad-CAM, known failure modes).
- [x] **Use the corrected app description and licence when deploying.**
      Description: `Research demo — explainable bone-lesion triage on plain radiographs.
      Not a medical device.` — **not** "detects early stages of cancer".
      Licence: `cc-by-nc-4.0`, **not** `mit`: the weights derive from BTXRD (CC BY-NC-ND
      4.0), so MIT would grant a commercial right that is not yours to give.

### Blocking — do these before trusting any result

- [ ] **Gate 4: human visual review.** `notebooks/01_data_sanity.ipynb` has **0/15 cells
      executed**. Nobody has ever looked at the preprocessed images. Assertions verify
      shape; only eyes verify content. An inverted or mis-windowed film produces perfectly
      valid arrays and a converging loss curve.
- [ ] **Gate 6: overfit check.** `scripts/overfit_check.py` has never been run against the
      current pipeline. Proves gradients actually reach the backbone.
- [ ] **Score Grad-CAM localisation.** `scripts/gradcam_report.py` has never been run on a
      trained checkpoint — no `gradcam_*` folder exists in any run. Until pointing-game
      accuracy is measured, we cannot claim the model looks at lesions rather than at
      collimation edges or implants. This is a headline feature of the project and is
      currently unverified.
      ```
      scripts\gradcam_report.py --checkpoint reports\full-20260822-041653\best.pt
      ```
- [ ] **Empirically tune the OOD validator on real data.** The heuristic thresholds
      (entropy ≤ 7.5 bits, edge density ≤ 0.45, channel spread ≤ 0.08) were set on
      synthetic phantoms. Run `onnm.ood.validate_payload` across all 3,746 BTXRD films
      and confirm ~0% false rejection; measure rejection rates on a folder of ordinary
      photographs. Same for the uncertainty gate: measure how many *true* lesion calls
      the 0.65/0.90 gates withdraw on the val split before trusting the defaults.

### High — resolve the regression

- [ ] **Calibrate + evaluate the overnight checkpoint.** It has no `calibration.json` and
      no `metrics_test.json`, so the comparison against `full` is val-ROC-only.
- [ ] **Ablate the overnight regression** (0.863 vs 0.891 macro ROC-AUC). Two suspects,
      confounded because they were changed together. **Configs and notebook are ready** —
      run `notebooks/colab_train.ipynb` cells 9 and 10:
      - `configs/ablations/augs_only.yaml` — aggressive augmentation alone (OHEM off)
      - `configs/ablations/ohem_only.yaml` — OHEM alone (augmentation at `full_run` strength)
      ROC-AUC is threshold-independent, so this is a genuine loss of ranking quality — it
      cannot be recovered by tuning the threshold.
- [ ] **Consider lowering OHEM penalty or raising `warmup_epochs`.** Malignant recall fell
      0.653 → 0.469 while FPs fell 65 → 37. That pattern is a bias shift toward "normal",
      not better discrimination.

### High — the actual fix for false positives

- [ ] **Integrate the expanded normal-control dataset.** More normals move the whole ROC
      curve; loss tricks only slide along it. Integration steps:
      1. Append rows to `dataset.xlsx` with the indicator columns set (`tumor=0`)
      2. Re-run `scripts/make_splits.py`
      3. Re-run `scripts/verify_data.py` — class counts will no longer match the paper,
         which is expected; update the assertion or pass the new expected counts
      - **Caveat:** `derive_groups` reconstructs patient identity from *consecutive image
        ids* sharing metadata. External images with unrelated ids each become their own
        group. Safe for splitting, but leakage protection is lost if the same patient
        appears twice in the new data. Assign explicit group ids if the source has them.
- [ ] **Re-test the known failure case** — the normal pelvis previously flagged at 59.6%,
      and the normal femur at 69.8%. These are the concrete regression tests for this work.
      *(needs the GPU box + dataset; blocked on the retrained checkpoint above)*
- [ ] **Report test-set specificity at the 90% floor** once a checkpoint is chosen.
      Validation says it costs ~6 additional missed cancers per 49; confirm on test.
      *(needs the GPU box + dataset)*

### Medium — model quality

- [ ] **Learned OOD detection to replace the heuristics.** Max-logit / energy scores,
      Mahalanobis distance on penultimate features, or a binary radiograph-vs-photo
      screen. The current validator is statistics-only and a grayscale photograph with
      X-ray-like statistics passes stage 1 (stage 2 then has to catch it).
- [ ] **Multi-view radiograph consensus.** BTXRD has multiple views per surrogate
      patient; aggregating AP + lateral predictions is a cheap sensitivity gain and a
      false-positive filter (a lesion visible in one view only is suspect).
- [ ] **Run `configs/specificity_tuning.yaml` on the GPU box** *(code ready; needs
      hardware)* — 320px, `alpha_beta 0.5`, 3-epoch backbone freeze. Compare against
      `full-20260822-041653` on val macro ROC-AUC, then test specificity. If the
      combined profile wins, ablate the three levers separately to attribute it.
- [ ] **Run `scripts/ablate_tta.py`** on the pinned checkpoint *(code ready; needs
      hardware)* — decide whether hflip TTA earns its 2× inference cost.
- [ ] Backbone ablation: `resnet50`, `efficientnet_b0`, `densenet169` (all already wired)
- [ ] **3D CT/MRI expansion** (long horizon): MONAI transforms generalise to 3D, but
      dataset, labels, VRAM budget, and the Grad-CAM geometry all need rework. Park
      behind a design doc; do not bolt onto the 2D pipeline.

### Medium — evaluation rigour

- [ ] **Run `scripts/stratified_report.py` on the GPU box** *(code ready; needs
      hardware + dataset)* — per-anatomy tables will confirm or refute the
      complex-joint-anatomy hypothesis; per-subtype tables split osteosarcoma from
      "other mt" and the benign subtypes.
- [ ] Fix the 5 unmapped rows if `verify_data` still reports any (tumour=1 but neither
      benign nor malignant flagged — most likely malignant, the class least able to lose
      cases) *(needs the dataset on disk)*

### Medium — app & delivery

- [ ] **DICOM metadata parser enhancements**: surface anatomy/laterality/view-position
      tags in the UI and scan history; use them to route the right per-anatomy operating
      point once per-anatomy analysis exists
- [ ] Per-user scan deletion in the UI (storage + DB rows currently require the Operator)
- [ ] Native PDF export (the HTML report ships print-to-PDF CSS; a direct PDF button
      would need a renderer dependency — only add if users ask)

### Low — housekeeping

- [ ] Create `reports/PRODUCTION` on the GPU box pinning `full-20260822-041653`, and
      delete `smoke-20260822-012828` (the app now hides `smoke-*` runs regardless)

---

## Decisions still owed by a human

- **Where the 28 MB checkpoint is hosted.** `src/checkpoint_fetch.py` takes any direct
  URL, so this is now a hosting choice rather than a code one: a Hugging Face model repo
  (free, and makes the model independently usable — closest to "a model people can use"),
  or a GitHub Release asset (no second account). Avoid Git LFS: every Community Cloud
  rebuild would spend the 1 GB/month LFS bandwidth quota.
- **Whether community data should ever reach val/test.** Currently pinned to `train` by
  `export_batch.py`. That keeps every score comparable to the numbers in `overview.md`;
  the cost is that community images never measure generalisation. Changing this would
  invalidate cross-run comparisons, so it should be a deliberate decision, not a drift.
- **Who counts as a reviewer.** `is_admin` exists in the schema but the review UI keys off
  `ONNM_ADMIN_KEY` instead. Fine for one maintainer; needs revisiting if Yasmine or anyone
  else reviews.

- **Operating point.** 80% specificity → 78.3% sensitivity; 90% → 67.0%. The gap is ~6
  missed cancers per 49. Which constraint binds is a clinical policy call, not a modelling
  one. Currently `full_run.yaml` uses 80%, `overnight.yaml` uses 90%.
- **Whether malignant-vs-benign matters at the UI level.** The app collapses both into
  "Potential Bone Lesion"; the 3-class breakdown is shown underneath.
