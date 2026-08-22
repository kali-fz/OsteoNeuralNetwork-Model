# ONNM — Status & Backlog

Companion to `overview.md`. Checked items are verified done, not assumed.
Last audited: 2026-08-22 (community feedback loop + hosting pass).

**Current state:** 277 tests green in the ROCm `.venv`, repo-wide ruff clean, app boots
(HTTP 200). Everything for the community loop is built and committed (`9810b7a`); none of
it is deployed yet. The next actions are all *yours* — they need logins I do not have.

---

## Done

### Infrastructure
- [x] Python 3.12 `.venv`, ROCm 7.2.1 stack on RX 7900 XT (`torch 2.9.1+rocm7.2.1`)
- [x] Package layout `src/onnm/`, editable install, ruff + pytest config
- [x] YAML config system with deep-merge overrides and named profiles
- [x] **277 tests**, synthetic fixtures, no dataset required; the torch-free
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

### Next — deploy the loop (all need logins I do not have)

- [ ] **Deploy Cloudflare.** Every command is in `cloudflare/README.md`; ~10 minutes.
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
- [ ] **Deploy to Streamlit Community Cloud** — <https://share.streamlit.io>, sign in
      with GitHub, main file `app.py`. Full instructions in
      `deploy/streamlit-cloud/README.md`. Free, 2.7 GB RAM, redeploys on push.
      **Hugging Face Spaces is no longer viable:** Gradio and Docker Spaces now require
      PRO and Streamlit is not offered at all; only Static (client-side, no Python)
      remains free. `deploy/hf-space/` has been removed.
- [ ] **Host the 28 MB checkpoint somewhere fetchable** and set `ONNM_CHECKPOINT_URL`
      (+ `ONNM_CALIBRATION_URL`). `reports/` is gitignored so a clone has no model;
      `src/checkpoint_fetch.py` downloads one at boot and pins it. A Hugging Face
      *model* repo is free even though compute Spaces are not, and makes the model
      independently usable; GitHub Releases also works.
- [ ] **Point it at `full-20260822-041653`**, not the overnight regression — set
      `ONNM_CHECKPOINT_RUN` accordingly, or write `reports/PRODUCTION` locally.
- [ ] **Push `main`.** Commit `9810b7a` is local only.
- [x] **`MODEL_CARD.md` performance section** — malignant recall **0.633 [0.490–0.776]**
      now leads the section, stated as "roughly one in three malignant films is missed"
      with the explicit warning that a normal verdict is weak evidence of absence. The
      rest of the card was already accurate (not-a-medical-device, CC BY-NC-ND, unscored
      Grad-CAM, known failure modes).
- [ ] **Use the corrected app description and licence when deploying.**
      Description: `Research demo — explainable bone-lesion triage on plain radiographs.
      Not a medical device.` — **not** "detects early stages of cancer".
      Licence: `cc-by-nc-4.0`, **not** `mit`: the weights derive from BTXRD (CC BY-NC-ND
      4.0), so MIT would grant a commercial right that is not yours to give.
- [ ] **Walk the loop once yourself, before inviting your friend.** Sign up → upload with
      sharing ticked → flag the result wrong → review and label it in the sidebar →
      `python scripts/export_batch.py --dry-run` → confirm the row appears with *your*
      label, not the model's or the user's.

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
