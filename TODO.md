# ONNM — Status & Backlog

Companion to `overview.md`. Checked items are verified done, not assumed.
Last audited: 2026-08-23 (community triage, versioning, publish path, Grad-CAM audit).

**Current state:** 374 tests green in the ROCm `.venv`, repo-wide ruff clean. Cloudflare
Worker + D1 **deployed, migrated to schema_version 3, and live**; Streamlit Cloud
**deployed and serving**. Model versioning is live at **v1.0.0** (`full-20260822-041653`).

> **Read this first if you read nothing else.** The Grad-CAM localisation claim — the
> headline feature of this project — is **not currently supported by measurement**, and
> the reason is a defect in the metric, not (necessarily) in the model. See
> "Grad-CAM is unmeasurable right now" under Blocking. Nothing about the classifier's
> reported ROC-AUC, recall or calibration is affected.

---

## Done

### Infrastructure
- [x] Python 3.12 `.venv`, ROCm 7.2.1 stack on RX 7900 XT (`torch 2.9.1+rocm7.2.1`)
- [x] Package layout `src/onnm/`, editable install, ruff + pytest config
- [x] YAML config system with deep-merge overrides and named profiles
- [x] **374 tests**, synthetic fixtures, no dataset required; the torch-free
      auth/storage/OOD/report/metrics subset runs without torch
- [x] CI workflow (`.github/workflows/ci.yml`): ruff lint, torch-free fast tests,
      full suite on CPU torch — gates 1–2 stay local-only (GPU/dataset)
- [x] `.env` and `.streamlit/secrets.toml` gitignored; **audited 2026-08-23: neither has
      ever been committed, and none of the 9 secret values in `.env` appears anywhere in
      git history.** The only matches are `ADMIN_EMAIL` and the D1 database id, both
      deliberately public
- [x] **D1 dumps gitignored** (`cloudflare/*.sql`, except `schema.sql` and the
      migrations). `wrangler d1 export` writes there by default, so the safe habit and
      the dangerous file come from the same command — a dump carries every user's email,
      their PBKDF2 hash, and every shared radiograph
- [x] Diagnosed the MIOpen training-BatchNorm defect; workaround `train.miopen: false`
- [x] `verify_env.py` gate 1 runs a real train-mode forward/backward
- [x] Measured VRAM/throughput across batch sizes; established 64 as the optimum
- [x] **Console output in `scripts/` is ASCII.** Windows consoles are cp1252, so an em
      dash in a `print` — or in a docstring, which argparse prints for `--help` — raised
      `UnicodeEncodeError` on the target machine. Was already latent in `ablate_tta.py`
      and `stratified_report.py`

### Data
- [x] BTXRD download + verification; class counts reproduce the paper exactly
- [x] Label derivation from one-hot indicator columns
- [x] Surrogate patient grouping; **verified zero group leakage** across all split pairs
- [x] DICOM handling: modality LUT, VOI LUT, MONOCHROME1 inversion (test-pinned)
- [x] Splits 2675 / 535 / 536, stratification holds to 0.1%
- [x] **No unmapped rows.** `verify_data.py` re-run 2026-08-23: 1879/1525/342 = 3746,
      `+0` on every class. The "5 unmapped rows" item is resolved — there are none

### Model & training
- [x] DenseNet-121, ImageNet-pretrained, 3-class head
- [x] Focal loss with tempered inverse-frequency alpha (`alpha_beta`)
- [x] `WeightedRandomSampler` + guard refusing it alongside a weighted loss
- [x] Class-asymmetric OHEM (`HardNegativeMiningLoss`) with warmup, budget cap, normalise
- [x] Aggressive augs: `RandAffine`, `RandCoarseDropout`, `RandHistogramShift`
- [x] Optional foreground cropping + guard disabling localisation scoring under it
- [x] Cosine and `ReduceLROnPlateau` schedulers; early stopping on macro ROC-AUC
- [x] Per-epoch tracking: loss, ROC-AUC, PR-AUC, sens, spec, balanced acc, F1, OHEM count
- [x] **Thermal governor** — AMD ADL via ctypes, hotspot control temp, memory ceiling

### Calibration & metrics
- [x] Temperature scaling — guarded fit (grid + ternary + LBFGS)
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

### Community loop — three-bucket triage
- [x] Cloudflare Worker + D1, free tier only (no R2, no payment method)
- [x] **Schema v3 applied to live D1**: `triage_bucket`, `triage_reason`, `admin_bucket`,
      `admin_label` widened to include `misc`, per-bucket batch counts
- [x] **Three buckets, triaged on arrival and re-triaged on user feedback** —
      `valid_bone` / `misc` / `contradiction`. Rule mirrored in `worker.js` and
      `community.classify_bucket`, with a test asserting the two agree
- [x] **OOD rejections are recorded, not discarded.** Previously an image the gate turned
      away left no trace, so the `misc` bucket could only ever be empty and the gate could
      only be retuned by hand
- [x] **"This really is a radiograph" dispute button** — the only witness to a false
      rejection, since inference never ran on those
- [x] **Admin pinned to one account** (`kzfhero@gmail.com`) in three places: a CHECK
      constraint on `users`, an `x-onnm-admin-user` header the Worker requires, and
      `community.is_admin` gating the UI
- [x] **`review_app.py` — the dedicated review console.** Full-width, three bucket tabs,
      base64 auto-decoded, nothing preselected. Local only, never deployed
- [x] Review gate now enforced in **four** places; a second trigger
      (`bucket_and_label_must_agree`) makes "hotdog, benign" unsayable
- [x] `scripts/sync_community.py` — claim + rebuild in one command, writing the cumulative
      `configs/controls_manifest.csv` that `base.yaml` already reads, so an approval
      reaches training with no config edit. Rebuilt rather than appended, so re-running is
      idempotent
- [x] Export writes **two** manifests — lesion rows and OOD negatives — because
      `build_records` would merge a combined file straight into the 3-class set

### Versioning & release
- [x] **`ONN.md` + `model_versions.json`** — every generation registered before anything is
      promoted; promotion is a separate guarded act. A regression is recorded as `held`,
      `reports/PRODUCTION` does not move, and the previous checkpoint keeps serving
- [x] **v1.0.0 seeded and serving** — `full-20260822-041653`, macro ROC-AUC 0.8934,
      malignant recall 0.6327, sha256 `f6b0ae7e…`
- [x] **`scripts/daily_cycle.py`** — approvals in, a guarded version out, or nothing at
      all. **Skips training entirely when no new data was approved**
- [x] **`scripts/publish_model.py`** — stages a version, refuses if the on-disk bytes no
      longer match the ledger, prints the exact secrets, and `--verify` checks a published
      URL before the deployment points at it
- [x] **Checkpoint fetch keyed on configuration, not filename.** Killed three silent
      publish bugs: a changed URL ignored when the old file existed; weights and
      calibration guarded independently (new weights at the old threshold); and
      `reports/PRODUCTION` written only when absent — which made "rename the run to force
      a re-download" *cause* a worse bug
- [x] **v1.0.0 backed up** to `G:\My Drive\ONNM-model1\` with a README, sha-verified

### Deployed and verified live
- [x] Cloudflare D1 (`onn-model`, id `961f0440-…`), Worker, both keys set
- [x] Auth boundary proved: no key → 401, app key → `/admin` → 403, admin key → 200
- [x] Streamlit Community Cloud deployed and serving
- [x] Cloudflare edge 1010 block fixed (`community.USER_AGENT`)
- [x] Google Sign-In: OIDC, provider-aware Worker, identity keyed on `sub`

### Runs
- [x] `full-20260822-041653` — trained, calibrated, test-evaluated (**current best**,
      registered as v1.0.0 and pinned as PRODUCTION)
- [x] `overnight-20260822-055132` — trained only; **regressed** to 0.8629
- [ ] `abl-ohem` / `abl-augs` — never completed. No `reports/` directory appeared in
      `MyDrive/OSTEONEURALNETWORK/`, so the Colab notebook's save-back cell never ran

---

## To do

### Blocking — Grad-CAM is unmeasurable right now

- [ ] **Fix the degenerate Grad-CAM, then re-score localisation.** Run 2026-08-23 on
      `full-20260822-041653`, test split, 267 annotated images:

      | metric | value |
      |---|---|
      | pointing game (argmax) | **0.0000** |
      | mean IoU | 0.0237 |
      | mean coverage | 0.0286 |
      | **median pixels tied at the CAM maximum** | **50,176 of 65,536 (77%)** |

      The zero was **not** a result about the model. `pointing_game` used `np.argmax`,
      which returns the *first* maximal element in raster order — so on a CAM where 77%
      of the frame ties at the maximum it reported the plateau's top-left corner every
      time, which is background on essentially every radiograph. **Fixed**: the peak is
      now the centroid of the maximal region, `peak_fraction` is reported, and
      `evaluate_localisation` warns loudly when the CAM is degenerate. Regression tests
      pin all of it.

      What remains is the real problem underneath: **the CAM is saturated** (mean value
      0.72–0.96 per image) and localises nothing. Prime suspect is
      `explain.target_layer: features.denseblock4`, which is DenseNet's raw block output
      — taken *before* `features.norm5` and its ReLU. A layer comparison
      (`denseblock4` / `norm5` / `features` / `transition3`) was still running when this
      was written; finish it, pick the layer that gives a non-degenerate CAM, then
      re-score. **Until then no claim about where the model looks is supported.**
      Also worth noting: 68.9% of CAM peaks land in the zero-padding band, and excluding
      padding did not change the score.

- [ ] **Gate 4: human visual review.** `notebooks/01_data_sanity.ipynb` still has
      **0/7 code cells executed**. Nobody has looked at the preprocessed images.
      Assertions verify shape; only eyes verify content. *(Cannot be delegated — but a
      contact sheet can be generated for you to look at.)*

### Blocking — OOD gate is measurably mistuned

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

      **Not changed, deliberately.** `onnm.ood` documents photographs as sitting near 8.0,
      so 7.8 leaves a narrow margin, and the cost side has never been measured. Run
      `python scripts/ood_sweep.py --negatives <folder of photos>` first — that is the
      other half of the trade, and this is a clinical-facing behaviour change.
- [ ] **Measure what the uncertainty gate withdraws.** How many *true* lesion calls the
      0.65 / 0.90 gates suppress on val, before trusting the defaults.

### Done — gate 6

- [x] **Gate 6: overfit check — PASSES.** 30 images memorised in 3 steps, accuracy 1.0,
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

### High — resolve the regression

- [ ] **Calibrate + evaluate the overnight checkpoint.** It still has no
      `calibration.json` and no `metrics_test.json`, so "it regressed" rests on val ROC
      alone. Two commands, both quick.
- [ ] **Ablate the overnight regression** (0.863 vs 0.891 macro ROC-AUC). Configs ready:
      `configs/ablations/augs_only.yaml`, `configs/ablations/ohem_only.yaml`. **Hours of
      GPU each — not started, awaiting your go-ahead.**
- [ ] **Consider lowering OHEM penalty or raising `warmup_epochs`.** Malignant recall fell
      0.653 → 0.469 while FPs fell 65 → 37: a bias shift toward "normal", not better
      discrimination.

### High — the actual fix for false positives

- [ ] **Integrate an expanded normal-control dataset.** More normals move the whole ROC
      curve; loss tricks only slide along it. The community loop now feeds
      `configs/controls_manifest.csv` automatically, so approved normals arrive here.
      **Caveat:** `derive_groups` reconstructs patient identity from consecutive image
      ids; external images each become their own group, which is safe for splitting but
      loses leakage protection if the same patient appears twice.
- [ ] **Re-test the known failure cases** — the normal pelvis flagged at 59.6% and the
      normal femur at 69.8%. *(blocked on a retrained checkpoint)*
- [ ] **Report test-set specificity at the 90% floor** once a checkpoint is chosen.

### Medium — model quality

- [ ] **Learned OOD detection to replace the heuristics.** The community loop now
      *collects* the training data for this — `configs/ood_manifest.csv` accumulates
      human-confirmed non-radiographs — but nothing consumes it yet: `onnm.ood` has no
      learned component. `onnm.ood_eval` scores the current gate so a learned one has a
      bar to beat
- [ ] **Multi-view radiograph consensus.** BTXRD has multiple views per surrogate patient
- [ ] **Run `configs/specificity_tuning.yaml`** *(hours of GPU; awaiting go-ahead)*
- [ ] **Run `scripts/ablate_tta.py`** — decide whether hflip TTA earns its 2× cost
- [ ] Backbone ablation: `resnet50`, `efficientnet_b0`, `densenet169` *(3 full runs)*
- [ ] **3D CT/MRI expansion** (long horizon) — park behind a design doc

### Medium — evaluation rigour

- [ ] **Run `scripts/stratified_report.py`** — per-anatomy tables to confirm or refute the
      complex-joint-anatomy hypothesis; per-subtype tables to split osteosarcoma out

### Medium — app & delivery

- [ ] **Redesign the UI** *(yours — starting 2026-08-23)*
- [ ] **GRC review** *(yours)*
- [ ] **DICOM metadata parser enhancements**: surface anatomy/laterality/view-position
- [ ] Per-user scan deletion in the UI
- [ ] Native PDF export (only if users ask)

### Low — housekeeping

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

- **The OOD entropy threshold.** 7.5 costs 2.16% false rejection; 7.8 costs none but
  narrows the margin against photographs. Measure the negatives side first.
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
