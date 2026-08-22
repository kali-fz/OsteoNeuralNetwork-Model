# ONNM — Status & Backlog

Companion to `overview.md`. Checked items are verified done, not assumed.
Last audited: 2026-08-22.

---

## Done

### Infrastructure
- [x] Python 3.12 `.venv`, ROCm 7.2.1 stack on RX 7900 XT (`torch 2.9.1+rocm7.2.1`)
- [x] Package layout `src/onnm/`, editable install, ruff + pytest config
- [x] YAML config system with deep-merge overrides and named profiles
- [x] **201 tests**, synthetic fixtures, no dataset required to run the suite
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
- [x] Auto-discovers newest checkpoint, auto-loads `calibration.json`
- [x] Loopback-only bind, telemetry off, temp files deleted before render
- [x] Medical disclaimer; calibration state and warnings surfaced in sidebar

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

### Runs
- [x] `full-20260822-041653` — trained, calibrated, test-evaluated (**current best**)
- [x] `overnight-20260822-055132` — trained only

---

## To do

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
- [ ] **Report test-set specificity at the 90% floor** once a checkpoint is chosen.
      Validation says it costs ~6 additional missed cancers per 49; confirm on test.

### Medium — model quality

- [ ] **Try `data.image_size: 320` or `384`.** The only meaningful way to use the idle VRAM
      (currently 27% at batch 64), and plausibly helps on subtle lesions. Costs epoch time;
      re-verify Grad-CAM box geometry at the new size.
- [ ] Backbone ablation: `resnet50`, `efficientnet_b0`, `densenet169` (all already wired)
- [ ] Ensemble or TTA (hflip) — cheap variance reduction on a 49-image malignant test set
- [ ] Revisit `loss.alpha_beta` (currently 1.0, weighting malignant 5.5× normal). Lowering
      to 0.5 is the most direct specificity lever and is currently untried.
- [ ] Freeze early backbone blocks for the first few epochs — 244 malignant training images
      is very little to fine-tune 7M parameters on

### Medium — evaluation rigour

- [ ] Subtype-stratified reporting (osteosarcoma vs "other mt"; benign subtypes)
- [ ] Per-anatomy error analysis — the complaint is specifically *complex joint anatomy*,
      and the metadata has anatomy columns. This would confirm or refute the hypothesis.
- [ ] Calibration reliability diagram, not just scalar ECE
- [ ] Fix the 5 unmapped rows if `verify_data` still reports any (tumour=1 but neither
      benign nor malignant flagged — most likely malignant, the class least able to lose
      cases)

### Medium — app & delivery

- [ ] Pin a "production" checkpoint rather than always taking the newest — the app
      currently auto-selects by mtime, so a bad experimental run silently becomes default
- [ ] Batch/folder upload for reviewing a series
- [ ] Export a per-case PDF/HTML report (verdict + overlay + disclaimer)
- [ ] Show the threshold sweep as an interactive ROC curve in the sidebar
- [ ] Model card documenting intended use, training data, measured limits, failure modes

### Low — housekeeping

- [ ] Add `streamlit` to `requirements-rocm.txt` / `requirements-cuda.txt` (currently only
      in the `[app]` extra of `pyproject.toml`)
- [ ] Document `rocm-sdk init` in `requirements-rocm.txt` — it is an undocumented required
      step and its absence caused a confusing failure
- [ ] Notebook lint failures (`E402`, `E501`, `B905`, `I001`) — pre-existing, unrelated to
      the pipeline
- [ ] CI workflow running gates 1–3 on push
- [ ] Prune old `reports/` runs; `smoke-20260822-012828` is a 1-epoch throwaway still
      appearing in the app's checkpoint dropdown

---

## Decisions still owed by a human

- **Operating point.** 80% specificity → 78.3% sensitivity; 90% → 67.0%. The gap is ~6
  missed cancers per 49. Which constraint binds is a clinical policy call, not a modelling
  one. Currently `full_run.yaml` uses 80%, `overnight.yaml` uses 90%.
- **Whether malignant-vs-benign matters at the UI level.** The app collapses both into
  "Potential Bone Lesion"; the 3-class breakdown is shown underneath.
