# ONNM: Status & Backlog

Companion to `overview.md`. Checked items are verified done, not assumed.
Last audited: 2026-08-29 (custom domain, review console, retention, terms gate scoped).

**Current state:** 452 Python tests and 37 Worker tests green, repo-wide ruff clean.
The whole application is **one Cloudflare Worker live at `osteoneuralnetwork.com`**,
D1 at **schema_version 6**, with the submission review console at `/admin` and a daily
retention job. The Streamlit deployment has been removed. Model versioning is live at
**v1.0.0** (`full-20260822-041653`).

> **Read this first if you read nothing else, THE ONE RELEASE BLOCKER.**
> Everything works and the domain is bought, but **nobody ever agrees to anything**.
> The header's "Sign in with Google" goes straight to Google, so an account is created
> and a scan can be run without a Terms acceptance step ever happening.
>
> `users.tos_accepted_at` exists and is populated, but it is **meaningless**:
> `cloudflare/src/worker.js` binds `tos_accepted_at || created` and the callback never
> sends the field, so every account has `tos_accepted_at == created_at`. It records
> that a row was inserted, not that a human read anything.
>
> This is a gap the compliance work already assumes is closed: `compliance/DPIA.md`
> risk rows R1, R2, R7 and R13 all name "the Terms" as their mitigation, and
> `compliance/ROPA.md` names Art 6(1)(b) "performance of a contract (the Terms)" as
> the lawful basis. **See `### Blocking: nobody agrees to the Terms` below.**
>
> Second, unchanged: the Grad-CAM localisation claim is *measurable* (the heatmap was
> inverted; fixed 2026-08-23) but still **roughly at chance**, pointing game 0.0936
> with no chance baseline established. It does not yet support "the model looks at
> lesions". Nothing about the classifier's ROC-AUC, recall or calibration was affected.

---

## Done

### Infrastructure
- [x] Python 3.12 `.venv`, ROCm 7.2.1 stack on RX 7900 XT (`torch 2.9.1+rocm7.2.1`)
- [x] Package layout `src/onnm/`, editable install, ruff + pytest config
- [x] YAML config system with deep-merge overrides and named profiles
- [x] **374 tests**, synthetic fixtures, no dataset required; the torch-free
      auth/storage/OOD/report/metrics subset runs without torch
- [x] CI workflow (`.github/workflows/ci.yml`): ruff lint, torch-free fast tests,
      full suite on CPU torch, gates 1–2 stay local-only (GPU/dataset)
- [x] `.env` and `.streamlit/secrets.toml` gitignored; **audited 2026-08-23: neither has
      ever been committed, and none of the 9 secret values in `.env` appears anywhere in
      git history.** The only matches are `ADMIN_EMAIL` and the D1 database id, both
      deliberately public
- [x] **D1 dumps gitignored** (`cloudflare/*.sql`, except `schema.sql` and the
      migrations). `wrangler d1 export` writes there by default, so the safe habit and
      the dangerous file come from the same command. A dump carries every user's email,
      their PBKDF2 hash, and every shared radiograph
- [x] Diagnosed the MIOpen training-BatchNorm defect; workaround `train.miopen: false`
- [x] `verify_env.py` gate 1 runs a real train-mode forward/backward
- [x] Measured VRAM/throughput across batch sizes; established 64 as the optimum
- [x] **Console output in `scripts/` is ASCII.** Windows consoles are cp1252, so an em
      dash in a `print`, or in a docstring, which argparse prints for `--help`, raised
      `UnicodeEncodeError` on the target machine. Was already latent in `ablate_tta.py`
      and `stratified_report.py`

### Data
- [x] BTXRD download + verification; class counts reproduce the paper exactly
- [x] Label derivation from one-hot indicator columns
- [x] Surrogate patient grouping; **verified zero group leakage** across all split pairs
- [x] DICOM handling: modality LUT, VOI LUT, MONOCHROME1 inversion (test-pinned)
- [x] Splits 2675 / 535 / 536, stratification holds to 0.1%
- [x] **No unmapped rows.** `verify_data.py` re-run 2026-08-23: 1879/1525/342 = 3746,
      `+0` on every class. The "5 unmapped rows" item is resolved: there are none

### Model & training
- [x] DenseNet-121, ImageNet-pretrained, 3-class head
- [x] Focal loss with tempered inverse-frequency alpha (`alpha_beta`)
- [x] `WeightedRandomSampler` + guard refusing it alongside a weighted loss
- [x] Class-asymmetric OHEM (`HardNegativeMiningLoss`) with warmup, budget cap, normalise
- [x] Aggressive augs: `RandAffine`, `RandCoarseDropout`, `RandHistogramShift`
- [x] Optional foreground cropping + guard disabling localisation scoring under it
- [x] Cosine and `ReduceLROnPlateau` schedulers; early stopping on macro ROC-AUC
- [x] Per-epoch tracking: loss, ROC-AUC, PR-AUC, sens, spec, balanced acc, F1, OHEM count
- [x] **Thermal governor**, AMD ADL via ctypes, hotspot control temp, memory ceiling

### Calibration & metrics
- [x] Temperature scaling, guarded fit (grid + ternary + LBFGS)
- [x] Dual-mode threshold search: sensitivity-floor and specificity-floor
- [x] ECE / NLL, bootstrap CIs, clinical error breakdown, macro F1
- [x] Conflicting-constraint reporting instead of silently picking one

### App
- [x] Streamlit UI: upload → verdict → confidence → Grad-CAM, DICOM/PNG/JPEG
- [x] **Production checkpoint pinning** via `reports/PRODUCTION`; throwaway runs hidden;
      stale pin fails loudly
- [x] Auto-loads `calibration.json`; falls back to newest non-throwaway run when unpinned
- [x] **Batch/folder upload** with per-file OOD rejection report and batch summary
- [x] **Per-case HTML report export** (`src/report.py`), overlay-PNG and JSON exports
- [x] **Interactive ROC threshold-sweep chart** in the sidebar
- [x] Loopback-only bind, telemetry off; medical disclaimer; calibration state surfaced
- [x] **Local accounts**: SQLite, salted PBKDF2-HMAC-SHA256 (600k iters), ToS timestamps
- [x] **De-identified upload storage** with UUID filenames and DICOM PII-strip
- [x] **GRC legal framework** (`src/legal.py`)
- [x] **OOD gate stage 1**: pre-inference radiograph validator with per-check reasons
- [x] **OOD gate stage 2**: predictive-entropy + max-prob uncertainty gating
- [x] **The app names the ONN version it is serving**, matched by file digest against
      `model_versions.json`. Run names are a local convention; the digest is what
      actually arrived, so a publish that did not take effect is visible not silent

### Community loop: three-bucket triage
- [x] Cloudflare Worker + D1, free tier only (no R2, no payment method)
- [x] **Schema v3 applied to live D1**: `triage_bucket`, `triage_reason`, `admin_bucket`,
      `admin_label` widened to include `misc`, per-bucket batch counts
- [x] **Three buckets, triaged on arrival and re-triaged on user feedback**, 
      `valid_bone` / `misc` / `contradiction`. Rule mirrored in `worker.js` and
      `community.classify_bucket`, with a test asserting the two agree
- [x] **OOD rejections are recorded, not discarded.** Previously an image the gate turned
      away left no trace, so the `misc` bucket could only ever be empty and the gate could
      only be retuned by hand
- [x] **"This really is a radiograph" dispute button**, the only witness to a false
      rejection, since inference never ran on those
- [x] **Admin pinned to one account** (`kzfhero@gmail.com`) in three places: a CHECK
      constraint on `users`, an `x-onnm-admin-user` header the Worker requires, and
      `community.is_admin` gating the UI
- [x] **`review_app.py`, the dedicated review console.** Full-width, three bucket tabs,
      base64 auto-decoded, nothing preselected. Local only, never deployed
- [x] Review gate now enforced in **four** places; a second trigger
      (`bucket_and_label_must_agree`) makes "hotdog, benign" unsayable
- [x] `scripts/sync_community.py`, claim + rebuild in one command, writing the cumulative
      `configs/controls_manifest.csv` that `base.yaml` already reads, so an approval
      reaches training with no config edit. Rebuilt rather than appended, so re-running is
      idempotent
- [x] Export writes **two** manifests, lesion rows and OOD negatives, because
      `build_records` would merge a combined file straight into the 3-class set

### Versioning & release
- [x] **`ONN.md` + `model_versions.json`**, every generation registered before anything is
      promoted; promotion is a separate guarded act. A regression is recorded as `held`,
      `reports/PRODUCTION` does not move, and the previous checkpoint keeps serving
- [x] **v1.0.0 seeded and serving**, `full-20260822-041653`, macro ROC-AUC 0.8934,
      malignant recall 0.6327, sha256 `f6b0ae7e…`
- [x] **`scripts/daily_cycle.py`**, approvals in, a guarded version out, or nothing at
      all. **Skips training entirely when no new data was approved**
- [x] **`scripts/publish_model.py`**, stages a version, refuses if the on-disk bytes no
      longer match the ledger, prints the exact secrets, and `--verify` checks a published
      URL before the deployment points at it
- [x] **Checkpoint fetch keyed on configuration, not filename.** Killed three silent
      publish bugs: a changed URL ignored when the old file existed; weights and
      calibration guarded independently (new weights at the old threshold); and
      `reports/PRODUCTION` written only when absent, which made "rename the run to force
      a re-download" *cause* a worse bug
- [x] **v1.0.0 backed up** to `G:\My Drive\ONNM-model1\` with a README, sha-verified

### Deployed and verified live
- [x] Cloudflare D1 (`onn-model`, id `961f0440-…`), Worker, both keys set
- [x] Auth boundary proved: no key → 401, app key → `/admin` → 403, admin key → 200
- [x] Streamlit Community Cloud deployed and serving
- [x] Cloudflare edge 1010 block fixed (`community.USER_AGENT`)
- [x] Google Sign-In: OIDC, provider-aware Worker, identity keyed on `sub`

### Runs
- [x] `full-20260822-041653`, trained, calibrated, test-evaluated (**current best**,
      registered as v1.0.0 and pinned as PRODUCTION)
- [x] `overnight-20260822-055132`, trained only; **regressed** to 0.8629
- [ ] `abl-ohem` / `abl-augs`, never completed. No `reports/` directory appeared in
      `MyDrive/OSTEONEURALNETWORK/`, so the Colab notebook's save-back cell never ran

---

## To do

### Blocking: nobody agrees to the Terms

**Built and deployed 2026-08-29.** Migration 0007 applied to live D1 (schema
version 7) before the deploy, all seven existing accounts carry `tos_version`
NULL and will be asked on their next visit, and `/api/auth/google/start` now
bounces to `/terms` without a valid acceptance cookie (verified live, including
against a forged cookie).

**Two things still owed before this counts as closed:**

- [ ] **GRC review of the Terms wording.** *(Yaso-cyber.)* The text was drafted
      here from the deleted Streamlit version and rewritten for the hosted
      reality. It has not been reviewed by anyone qualified, and the site should
      not be promoted widely until it has.
- [ ] **Walk the gate once as a human.** Sign in on a fresh browser, confirm the
      tick is required, confirm an existing account is asked on next visit, and
      confirm a scan works afterwards. Everything up to the session boundary is
      verified automatically; the round trip through Google is not.

- [x] **Recover and rewrite the Terms text.** `src/legal.py` was deleted in commit
      `ea1ce2e` and holds 234 lines of finished Markdown: `TERMS_OF_SERVICE` (10
      numbered sections), `PRIVACY_POLICY`, `MEDICAL_DISCLAIMER`, `COOKIE_NOTICE`.
      Recover with `git show ea1ce2e^:src/legal.py`. It was written for the Streamlit
      deployment ("stored on the Operator's machine", `*.streamlit.app`) so every
      hosting fact needs rewriting for Cloudflare, but the section skeleton and the
      liability and IP language transfer directly. Its §9 already states that material
      changes require renewed consent. Combine with the seven-item list already on the
      landing page (`web/src/pages/landing.js`, the `onnm-terms-list`).
      *(Drafted here, then reviewed by the project's GRC collaborator before it is
      relied on. Not legal advice.)*
- [x] **Add the `/terms` route.** New `web/src/pages/terms.js`; register in `ROUTES`
      and `ROUTE_TITLES` in `web/src/main.js`, and **keep it out of**
      `SIGNED_IN_ROUTES` so a signed-out visitor can reach it. The page shows the
      Terms, a tick box, and a "Continue to Google sign-in" button that stays disabled
      until ticked.
- [x] **Repoint all three sign-in links.** `signInHref()` in `web/src/main.js` is the
      choke point for two of them (the header button and the signed-out route guard),
      but **`web/src/pages/landing.js` hardcodes `/api/auth/google/start` and bypasses
      it**. That third one is the easy miss and the whole gate leaks without it.
- [x] **Make the gate real on the server.** A tick enforced only in the browser is
      decoration. `POST /api/terms/accept`; signed out it mints a short-lived signed
      cookie recording the accepted version, reusing `signSession`/`verifySession`
      from `worker/lib/session.js` exactly as the OAuth `state` and PKCE verifier
      already are. `authStart()` then refuses to begin OAuth without that cookie and
      bounces to `/?auth_error=terms_required` (add the code to `AUTH_ERRORS`).
- [x] **Persist acceptance against the account.** `authCallback()` passes the accepted
      version into `createUser` so it lands on the new row. Migration
      `cloudflare/migrations/0007_terms_acceptance.sql` following the `0006` template:
      `ALTER TABLE users ADD COLUMN tos_version TEXT;` plus bumping `schema_version`
      to `'7'`. Thread `tos_version` through `createUser` in `cloudflare/src/worker.js`
      (destructure, INSERT columns, bind list), `worker/lib/account.js`, and the
      callback in `worker/index.js`. **Migration first, deploy second.**
- [x] **Re-consent the existing accounts.** *(Owner's decision: everyone, not just new
      sign-ups.)* `/api/session` gains `terms_accepted`, derived server-side from the
      user row exactly as `is_admin` already is. A signed-in visitor without acceptance
      is routed to `/terms`; ticking posts to the same endpoint, which being
      authenticated writes straight to their row. Existing rows keep `tos_version`
      NULL, which honestly means "accepted an unrecorded version".
- [x] **Refuse scans without acceptance.** `/api/scan` and `/api/warmup` return a clear
      refusal when the acting account has not accepted. Routing alone is a UI
      convention any client can skip; this mirrors how `/api/admin/*` is guarded.
- [x] **Do not break the signed-out landing page.** It keeps full function, globe
      included. `/terms` is additive. Re-check the four surfaces afterwards: landing
      layout, globe, Google sign-in end to end, image upload returning a verdict.

### Grad-CAM: the heatmap was inverted; fixed 2026-08-23

- [x] **The CAM was inverted. Root cause found and fixed.** MONAI's `CAMBase`
      defaults `postprocessing=default_normalizer`, which maps
      `(min, max) -> (1, 0)`. Its own docstring says so: "This will flip magnitudes
      (i.e., smallest will become biggest and vice versa)." `build_cam` never
      overrode it, and `compute_cam` then min-max rescaled the already-flipped array,
      which preserves the flip rather than undoing it.

      Proven rather than inferred: against an identity-postprocessing CAM on the same
      films, correlation was **exactly -1.0000** with
      `max|shipped - (1 - correct)| = 0.0`. Every pixel was the precise inverse.

      **This explains everything the previous three entries were chasing.** Grad-CAM
      ends in a ReLU, so most of a healthy map is zero; flipping it turned that zero
      region into the "hottest evidence". The 77%-of-frame-tied-at-maximum was the zero
      region. The 68.9%-of-peaks-in-the-padding-band was padding -- zero activation,
      zero CAM, therefore "hottest" once inverted. The pointing-game 0.0000 was the peak
      landing in background by construction. It was never the target layer, and it was
      never the model.

      **Fix:** `build_cam` now passes an explicit `_identity_postprocessing` to the
      MONAI factory, leaving `compute_cam`'s min-max as the single normalisation step.
      `GradCAMpp` subclasses `GradCAM` and inherits the same default, so one change
      covers both `explain.method` values.

      **Re-scored on the pinned checkpoint**, test split, all 267 annotated films,
      `class_index=MALIGNANT_INDEX` (unchanged model, layer, training and checkpoint):

      | metric | before | after |
      |---|---:|---:|
      | pointing game | 0.0000 | **0.0936** |
      | mean IoU | 0.0237 | **0.0428** |
      | mean coverage | 0.0286 | **0.0440** |

      **The saturation is gone.** 232 of the 267 films now have
      `peak_fraction <= 0.0039` (mean 0.0010) against a degeneracy threshold of 0.05 --
      compare the old median of 77% of the frame tied at the maximum.

      **No verdict moved.** Across the 24 films common to both report runs, the maximum
      difference in `malignant_probability` was **0.000e+00** and no
      `predicted_class` changed. The heatmap is display-only: it is written at
      `inference.py:643`, returned at `:696`, and read by nothing that gates a
      decision. Every ROC-AUC, recall and specificity figure in `overview.md` stands.

      **Regression tests** in `tests/test_explainability.py` build a real model and go
      through `build_cam` -- deliberately, because the whole file was synthetic numpy
      before, which is exactly why an exactly-inverted heatmap shipped unnoticed. A
      stubbed cam object never invokes MONAI and would pass either way. Verified by
      reverting the fix: all three new tests fail, then pass again once restored.

      The pre-fix overlays are kept at
      `reports/full-20260822-041653/gradcam_test_BEFORE_polarity_fix/` for comparison.

- [ ] **Decide what `cam_degenerate` should mean now.** The flag still reports `True`,
      and it is now a false alarm. `mean_peak_fraction` is 0.1320, but that is
      **35 films (13.1%) whose malignant CAM is entirely empty** -- each scoring
      `peak_fraction = 1.0` because every pixel ties at a maximum of zero -- dragging
      up a mean whose other 232 members average 0.0010.

      An empty CAM and a saturated CAM are different facts requiring different responses,
      and the mean currently conflates them. An empty malignant CAM on a film the model
      confidently calls normal is arguably the correct output, not a defect. Options:
      report the two separately, use a median, or exclude empty maps from the statistic
      and count them alongside it.

- [ ] **Interpret the corrected localisation honestly.** Pointing game 0.0936 is low. It
      is no longer *meaningless*, which is the change, but it is not yet evidence that
      the model localises. Establish the chance baseline before claiming anything: score
      a randomly-initialised model the same way, since a lesion box covering ~10% of the
      frame is hit ~10% of the time by accident. Until that comparison exists, the honest
      statement is "measurable, and roughly at chance".

      Also note `cam_threshold: 0.5` (`configs/base.yaml:312`) now selects a small
      concentrated region rather than a large diffuse one, so IoU is penalised for a
      tight prediction against a large ground-truth box. Re-examine the threshold as its
      own decision before reading the IoU delta as progress.

- [ ] **Check the UI attention floor.** `cam_floor` defaults to 0.25
      (`app.py:712`) and was implicitly chosen against a map whose mean was 0.747. The
      corrected mean is 0.253, so the default now hides roughly half the map. It hides
      the correct half, but the number was never chosen for this distribution.

- [ ] **Gate 4: human visual review.** `notebooks/01_data_sanity.ipynb` still has
      **0/7 code cells executed**. Nobody has looked at the preprocessed images.
      Assertions verify shape; only eyes verify content. *(Cannot be delegated, but a
      contact sheet can be generated for you to look at.)*

### Blocking: OOD gate is measurably mistuned

- [x] **Empirically measure the OOD validator on real data.** `scripts/ood_sweep.py`
      (new). Run over all **3,746** BTXRD films:

      | | |
      |---|---|
      | accepted | 3,665 |
      | **falsely rejected** | **81 (2.16%)** |
      | attributable to `histogram_entropy` | **81 of 81 (100%)** |

      The TODO previously expected "~0%". It is 2.16%, and every one of them is a single
      threshold. All 81 have entropy between **7.500 and 7.764** against a limit of 7.5.

- [ ] **Decide the entropy threshold.** Raising `MAX_HISTOGRAM_ENTROPY`:

      | threshold | false rejections |
      |---|---|
      | 7.5 (current) | 81 (2.16%) |
      | 7.6 | 31 (0.83%) |
      | 7.7 | 7 (0.19%) |
      | **7.8** | **0** |

      **Not changed, and still 7.5** (`src/onnm/ood.py:60`), confirmed 2026-08-23.

      **The radiograph side is now measured** (600-film random sample): p50 6.36,
      p99 7.54, **max 7.7636**. So the gap between the highest real radiograph and the
      8.0 that `onnm.ood` quotes for photographs is **0.236 bits, about 3% of the
      scale**.

      That reframes the decision. It is not "7.5 or 7.8": this single feature has almost
      no separating power at the top of its range, so *any* threshold there is fragile.
      **Recommendation: hold at 7.5** and treat the learned OOD detector as the real fix.

      The cost side is still unmeasured. `scripts/ood_sweep.py --negatives <folder>`
      already exists and works. It only needs a folder of non-radiograph photos, which
      this repository does not contain. The community `misc` bucket is meant to become
      that corpus, but `configs/ood_manifest.csv` does not exist yet: nothing has been
      approved into it.
- [ ] **Measure what the uncertainty gate withdraws.** How many *true* lesion calls the
      0.65 / 0.90 gates suppress on val, before trusting the defaults.

### Blocking: a disputed false rejection is silently discarded

The single highest-value signal in the community loop is being thrown away, and the UI
tells the user the opposite. Traced end to end 2026-08-23:

1. The OOD gate rejects an upload. `app.py:909` calls
   `record_rejection(..., shared=SHARE_CONSENT)`. Without the share tick the row is
   stored **with no image**, deliberate and correct; consent governs the pixels.
2. The **"This really is a radiograph" button is rendered unconditionally**
   (`app.py:924`), whether or not the user shared.
3. Its help text says **"Sends the image to a human reviewer."** That is **false** for an
   unshared row: there is no image, and no reviewer will ever see it.
4. Pressing it re-triages the row to `contradiction`, the bucket `schema.sql` calls
   "worth the most per row", because each one is a demonstrated failure of the gate.
5. `pendingReview` (`worker.js:708`) filters `shared = 1`, so the row never reaches the
   queue. `/health` filters it too (`worker.js:227`, `:239`), so it is not even counted.

**This is the same problem as the entropy threshold above.** Those disputes are precisely
the evidence that decision is waiting on: 2.16% of BTXRD films trip the gate, the users
who hit that are the only witnesses, and their testimony is being dropped on the floor.

- [ ] **A. Make the UI tell the truth** (`src/community_ui.py`,
      `render_rejection_dispute`). When the row carries no image, still record the dispute
     , it is a real signal and it counts, but say plainly that nothing was kept, so a
      reviewer cannot check it, and how to make it reviewable (tick share, re-upload).
      When shared, behaviour and wording unchanged. App-side only: no schema change, no
      Worker change, no redeploy hazard. **Ships with the next Streamlit deploy.**

- [ ] **B. Make the invisible visible** (`cloudflare/src/worker.js`, `health()`). Add one
      additive, read-only field counting disputed-but-unshared pending rows. Changes no
      existing number and does not touch the review gate; gives a running count of the
      false-rejection evidence currently being lost, which is what makes the entropy
      decision measurable over time.
      **Requires a Worker deploy, see the deployment hazard at the top of this file.
      Migration 0004 must be applied to live D1 first, or the deploy breaks the site.**

- [ ] **C. Decide: prompt for consent at dispute time?** *(not proposed, a human call.)*
      Asking "share this image so a reviewer can check?" at the moment of dispute would
      convert the highest-value signal into reviewable training data. It needs a new write
      path that attaches an image to an existing submission, which is consent-sensitive on
      a table holding medical images. That is a GRC decision, not an engineering one.

      Tests to accompany A and B: one asserting the new health count catches a disputed
      unshared row against the real schema (as `tests/test_geolocation.py` does), and one
      pinning that `pendingReview`'s `shared = 1` filter is **unchanged**. The fix must
      not smuggle imageless rows into the review queue.

### Done: gate 6

- [x] **Gate 6: overfit check, PASSES.** 30 images memorised in 3 steps, accuracy 1.0,
      final loss 0.0059, in about 8 seconds. **The pipeline learns**: labels line up with
      images, normalisation preserves the signal, gradients reach the backbone.

      It had never run here, for two stacked reasons, and both are fixed:

      1. `overfit_check.py` took no `--override`, so it loaded `base.yaml` with
         `train.miopen: true`.
      2. More seriously, **`overfit_batch` never called `configure_backend`**. `train()`
         does, with a comment saying it must happen before the first conv/BN call --
         but the gate 6 path skipped it, so `train.miopen: false` was read from the
         config and silently ignored. The gate then walked into
         `RuntimeError: miopenStatusUnknownError` on the exact hardware it exists to
         check, and MIOpen spent over an hour failing to JIT-compile that kernel before
         giving up, which looked like "training is slow" rather than "the workaround is
         not applied".

### High: resolve the regression

- [ ] **Calibrate + evaluate the overnight checkpoint.** It still has no
      `calibration.json` and no `metrics_test.json`, so "it regressed" rests on val ROC
      alone. Two commands, both quick.
- [ ] **Ablate the overnight regression** (0.863 vs 0.891 macro ROC-AUC). Configs ready:
      `configs/ablations/augs_only.yaml`, `configs/ablations/ohem_only.yaml`. **Hours of
      GPU each, not started, awaiting your go-ahead.**
- [ ] **Consider lowering OHEM penalty or raising `warmup_epochs`.** Malignant recall fell
      0.653 → 0.469 while FPs fell 65 → 37: a bias shift toward "normal", not better
      discrimination.

### High: the actual fix for false positives

- [ ] **Integrate an expanded normal-control dataset.** More normals move the whole ROC
      curve; loss tricks only slide along it. The community loop now feeds
      `configs/controls_manifest.csv` automatically, so approved normals arrive here.
      **Caveat:** `derive_groups` reconstructs patient identity from consecutive image
      ids; external images each become their own group, which is safe for splitting but
      loses leakage protection if the same patient appears twice.
- [ ] **Re-test the known failure cases**, the normal pelvis flagged at 59.6% and the
      normal femur at 69.8%. *(blocked on a retrained checkpoint)*
- [ ] **Report test-set specificity at the 90% floor** once a checkpoint is chosen.
- [ ] **Run `scripts/stratified_report.py` first; it has never been run.** It is the
      cheapest step here and the per-anatomy table that would confirm or refute the
      complex-joint hypothesis does not exist yet. **Expect it to be underpowered:**
      the joint columns are extremely sparse, `wrist-joint` = 0, `hip-joint` = 3,
      `knee-joint` = 37, `shoulder-joint` = 12. Only `pelvis` (216) has usable n, so
      the honest conclusion may be "cannot tell from this dataset", which is itself
      worth writing down.
- [ ] **Judge any specificity work on ROC-AUC, not on false-positive counts.** The
      OHEM run cut FPs 65 to 37 but dropped malignant recall 0.653 to 0.469 and macro
      ROC-AUC 0.891 to 0.863. It shifted the bias toward "normal" rather than learning
      better discrimination, and an argmax FP count cannot tell those two apart.
- [ ] **MURA as a source of normal controls.** 40,895 musculoskeletal radiographs
      (Stanford), normal/abnormal labels only, free for research. No tumour subtypes,
      so it is useless for the subtype work below, but it is directly the "more
      normals" intervention this section already names as the real fix. Check its
      licence terms against the manifest's `license` column before importing.

### High: predict the tumour subtype

The goal is to name the tumour, not only call it malignant. **Read the constraint
first: the dataset does not currently support the subtypes wanted.**

- [ ] **Face what BTXRD actually contains.** `data/raw/BTXRD/dataset.xlsx` is
      3,746 rows by 37 columns with one-hot indicators and **no categorical diagnosis
      column**. The malignant side has exactly two: `osteosarcoma` (297) and `other mt`
      (45). **Ewing sarcoma, chondrosarcoma, chordoma, fibrosarcoma and adamantinoma
      do not exist as labels** and are all folded into `other mt`; the config comment
      at `configs/base.yaml` already says so. Benign has seven: `osteochondroma` (754),
      `multiple osteochondromas` (263), `simple bone cyst` (206), `giant cell tumor`
      (93), `synovial osteochondroma` (51), `osteofibroma` (44), `other bt` (115).
- [ ] **Decide the achievable task.** From BTXRD alone the ceiling is osteosarcoma vs
      other-malignant with 45 in the minority class, plus the seven benign subtypes.
      That is a real deliverable and needs no new data. The named list of five rare
      malignancies needs an external dataset or re-annotation, and **no amount of
      money fixes it**, so it is a sourcing problem rather than a training one.
- [ ] **Add a subtype head.** Touch points: `configs/base.yaml` (the `labels` block
      already has correct `subtype_columns` lists), `src/onnm/dataset.py`
      (`map_labels`, and the record dict in `build_records`), `src/onnm/model.py`
      (`_replace_head`/`build_model`, currently a single head), `src/onnm/losses.py`
      (asserts `expected == cfg.model.num_classes`), `src/onnm/metrics.py` (hardcoded
      `num_classes=3` defaults and `CLASS_NAMES` indexing), `src/onnm/train.py`.
      Subtype columns currently feed `derive_groups` only, never the label.
- [ ] **Do not let a subtype head cost lesion recall.** The three-class call is what
      the product promises; a subtype guess is an extra. Gate promotion on the
      existing guarded metrics in `scripts/version_model.py`, not on subtype accuracy.

### Medium: model quality

- [ ] **Learned OOD detection to replace the heuristics.** The community loop now
      *collects* the training data for this, `configs/ood_manifest.csv` accumulates
      human-confirmed non-radiographs, but nothing consumes it yet: `onnm.ood` has no
      learned component. `onnm.ood_eval` scores the current gate so a learned one has a
      bar to beat.
      **Say this plainly to whoever is doing the reviewing:** marking an image "not a
      radiograph" and approving it changes *nothing* about the gate's behaviour today.
      There is no `paths.ood_manifest` config key, and `grep -c ood` over
      `scripts/train.py`, `src/onnm/train.py`, `dataset.py`, `losses.py` and
      `config.py` returns 0. The rows accumulate toward a future detector and improve
      the `misc_rejection` measurement in the version ledger; that is all.
      **Why it matters now:** a photograph of a person on a white background passed the
      gate and reached the review queue in August 2026. `reports/ood_sweep.json` shows
      every one of the 81 false rejections on real films came from `histogram_entropy`
      alone, and real radiographs reach a maximum of **7.7636** against the ~8.0
      expected of photographs. A 0.236-bit gap is about 3% of scale, so this feature
      has almost no separating power at the top of its range and no threshold placed
      there will be robust. Hold at 7.5 and treat a learned detector as the fix.
      **The negatives side is still unmeasured** because the repo has no folder of
      non-radiograph photos, which is exactly the gap `ood_manifest.csv` was meant to
      fill and still has not.
- [ ] **Multi-view radiograph consensus.** BTXRD has multiple views per surrogate patient
- [ ] **Run `configs/specificity_tuning.yaml`** *(hours of GPU; awaiting go-ahead)*
- [ ] **Run `scripts/ablate_tta.py`**, decide whether hflip TTA earns its 2× cost
- [ ] Backbone ablation: `resnet50`, `efficientnet_b0`, `densenet169` *(3 full runs)*
- [ ] **3D CT/MRI expansion** (long horizon), park behind a design doc

### Medium: evaluation rigour

- [ ] **Run `scripts/stratified_report.py`**, per-anatomy tables to confirm or refute the
      complex-joint-anatomy hypothesis; per-subtype tables to split osteosarcoma out

### Medium: app & delivery

- [ ] **Redesign the UI** *(yours, starting 2026-08-23)*
- [ ] **GRC review** *(yours; the Terms text drafted for the blocking section above
      needs your GRC collaborator's eyes before the site is opened to strangers)*
- [ ] **Credit the image sources on the site.** A `<details>` block added to
      `renderFooter()` in `web/src/main.js`. The existing `.onnm-legal-summaries` CSS
      styles it for free and `<details>`/`<summary>` is already the only collapsible
      pattern in the codebase, so this needs no new CSS.
      **Generate it from the manifest, do not hand-maintain it.**
      `configs/controls_manifest.csv` already carries a `license` column per row; add
      `source` and `source_url` and build the credits from those, so the page cannot
      drift from the data actually trained on.
- [ ] **Check image licences per figure, not per article.** PubMed Central's Open
      Access Subset splits three ways and only the `oa_comm` tier (CC0, CC BY) is
      broadly safe. **An open-access article can still contain figures copyrighted by
      third parties** that need separate permission, so the check is per image and the
      answer must be recorded per image.
      Note the constraint already at the top of the chain: **BTXRD is CC BY-NC-ND
      4.0**, non-commercial *and* no redistribution of derivatives "including Grad-CAM
      overlays" (`MODEL_CARD.md`). There is currently **no PMC or NLM reference
      anywhere in the repo**, so if PMC images enter the training set that provenance
      gap exists in `MODEL_CARD.md` and `README.md` independently of the UI work.
- [ ] **DICOM metadata parser enhancements**: surface anatomy/laterality/view-position
- [ ] Per-user scan deletion in the UI *(partly done 2026-08-29: a user can now
      withdraw a shared image until it is approved, from their profile page. What is
      still missing is deleting the scan record itself, and an account-level erasure
      request.)*
- [ ] Native PDF export (only if users ask)

### Low: housekeeping

- [ ] **Walk the community loop once end to end.** Sign in → upload with sharing ticked →
      review and label it in `review_app.py` → `python scripts/sync_community.py --dry-run`
      → confirm the row appears with *your* label. Six shared submissions are waiting in
      the queue
- [ ] **Rotate the credentials.** The Cloudflare API token and R2 token were pasted into a
      chat transcript, and `ONNM_COMMUNITY_KEY` / `ONNM_ADMIN_KEY` appeared in screenshots
      during setup. `.env.backup-1787434019` is a duplicate of `.env` and can be deleted
- [ ] **Decide whether to publish the OAuth consent screen.** Testing mode caps at 100
      listed users; publishing needs no Google verification for `openid`/`email`/`profile`
      but testers see an "unverified app" interstitial either way
- [ ] Delete `smoke-20260822-012828` (the app hides `smoke-*` runs regardless)
- [x] **Verified which weights are in GitHub release `v0.1.0`** (2026-08-23). Fetched and
      hashed: sha256 `f6b0ae7e...937c7c`, which **matches v1.0.0 exactly**. The live site
      has been serving `full-20260822-041653` all along, not the overnight regression.
      Re-check any future release with
      `python scripts/publish_model.py v1.0.0 --verify "<url>"` -- quote the URL, or
      PowerShell parses the angle brackets as redirection operators

---

## Decisions still owed by a human

- **~~Whether to spend the offered £10 on more data.~~ Resolved 2026-08-29: do not.**
  The owner offered £10 to expand the dataset for subtype classification. It buys
  nothing that matters here. BTXRD already ships subtype labels, so the achievable
  subtype task needs no purchase; and the five rare malignancies actually wanted
  (Ewing, chondrosarcoma, chordoma, fibrosarcoma, adamantinoma) are absent from BTXRD
  entirely, which is a **sourcing and annotation problem that money does not solve**.
  The candidate free sources are MURA for normal controls and, with per-figure licence
  checks, PMC open-access case reports. Keep the £10.
- **The OOD entropy threshold.** 7.5 costs 2.16% false rejection; 7.8 costs none. But the
  measured gap between the highest real radiograph (7.7636) and the 8.0 quoted for
  photographs is only **0.236 bits**, so no threshold in that range is robust.
  **Recommendation: hold at 7.5**; the learned OOD detector is the real fix. The
  negatives side is still unmeasured.
- **Whether to prompt for consent when a user disputes a rejection** (item C under
  "a disputed false rejection is silently discarded"). It would turn the best signal in
  the system into reviewable data, at the cost of a new consent-sensitive write path onto
  a table of medical images.
- **Whether community data should ever reach val/test.** Currently pinned to `train`,
  which keeps every score comparable to `overview.md`. Changing it invalidates cross-run
  comparison, so it should be deliberate rather than drift.
- **Whether to publish the weights openly.** They derive from BTXRD (CC BY-NC-ND 4.0), and
  whether trained weights are a "derivative" under that licence is unsettled.
- **Operating point.** 80% specificity → 78.3% sensitivity; 90% → 67.0%. ~6 missed cancers
  per 49. A clinical policy call, not a modelling one.
- **Whether malignant-vs-benign matters at the UI level.** The app collapses both into
  "Potential Bone Lesion"; the 3-class breakdown is shown underneath.

### Resolved
- ~~**Who counts as a reviewer.**~~ One hardcoded account, enforced in the schema, the
  Worker and the UI. Revisit only if a second reviewer is ever needed.
- ~~**Where the checkpoint is hosted.**~~ GitHub Release, fetched by
  `src/checkpoint_fetch.py`, now with digest verification.
