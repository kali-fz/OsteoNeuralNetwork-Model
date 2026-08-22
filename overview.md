# ONNM — Agent Overview

Terse orientation for agents working on this repo. Facts, paths, invariants.
Human-facing detail lives in `README.md`.

---

## Mission

Explainable detection of primary bone tumours on plain radiographs. Classifies each film
**normal / benign / malignant**, and produces Grad-CAM heatmaps that are *scored against
ground-truth lesion boxes* rather than merely displayed.

**Hard constraint: $0 running cost.** Open dataset, local AMD GPU, free/open-source
libraries only (MIT/Apache-2.0). No cloud compute, no paid APIs, no telemetry. Kaggle's
free tier is the only permitted fallback.

---

## Dataset — BTXRD

- 3,746 grayscale radiographs: **1,879 normal / 1,525 benign / 342 malignant (9.1%)**
- Per-lesion boxes + segmentation polygons (LabelMe 5.x JSON), 1,867 annotated
- Source: figshare `10.6084/m9.figshare.27865398` (Sci Data 2024)
- **Licence CC BY-NC-ND 4.0.** NoDerivatives covers Grad-CAM overlays — never redistribute
  derived images. `data/` and `reports/` are gitignored.

Three release quirks, all handled in code:
1. Metadata is `dataset.xlsx`, not CSV (needs `openpyxl`).
2. Diagnosis is **one-hot indicator columns**, not a categorical label.
   `tumor=0 → normal`, `malignant=1 → malignant`, `benign=1 → benign`.
3. Table says `.jpeg` for every id; disk mixes `.jpeg`/`.jpg`. Matched on filename stem.

**No patient ID exists.** 2,826/3,746 images are multiple views of one patient.
`derive_groups` reconstructs a surrogate patient id from runs of consecutive image ids
sharing centre/age/sex/anatomy/diagnosis. Splits are grouped on it — verified zero leakage.

---

## Stack

| Layer | Choice |
|---|---|
| DL | PyTorch 2.9.1+rocm7.2.1, MONAI 1.5.2, torchvision 0.24.1 |
| Model | torchvision DenseNet-121, ImageNet-pretrained, 3-class head, dropout 0.2 |
| I/O | pydicom 3.x, Pillow — DICOM + PNG/JPEG/BMP/TIFF |
| Metrics | scikit-learn; bootstrap CIs hand-rolled |
| Viz | matplotlib, seaborn, OpenCV-headless |
| UI | Streamlit >=1.42 (`app.py`); loopback locally, Streamlit Cloud when hosted |
| Env | Python **3.12 exactly** (ROCm wheels are cp312-only), `.venv` |
| GPU | AMD RX 7900 XT 20 GB, gfx1100, ROCm 7.2.1 on Windows 11 |

ROCm reports through the CUDA API: `torch.cuda.is_available()` is True, device is
`cuda:0`, `torch.version.hip` is set. That is expected, not a bug.

---

## Environment landmines

**MIOpen cannot JIT-compile its training-mode BatchNorm kernel** on these wheels —
`<type_traits>` is not shipped (`rocm-sdk init` supplies only thrust's). Training dies with
`RuntimeError: miopenStatusUnknownError`; **inference is unaffected** because it uses the
eval-mode path. Workaround is `train.miopen: false` (ATen native kernels, ~40% slower per
step, semantically identical). `scripts/verify_env.py` gate 1 checks this.

**Windows spawns DataLoader workers** — each re-imports the module and duplicates the
~2.1 GB MONAI cache. Any script building a loader needs `if __name__ == "__main__":`.
Machine has 31.8 GB RAM; `num_workers: 2` is the measured safe ceiling.

**VRAM cannot be saturated at 256 px.** Measured: batch 64 → 5.45 GB (27%), batch 128 →
10.6 GB (53%) but 6× slower per step. Filling VRAM is not a training objective; raising
`data.image_size` is the only meaningful way to use the headroom.

**`train.miopen: false` is ROCm-only and must not be honoured on CUDA.** It exists purely
for the defect above, but the flag it sets (`torch.backends.cudnn.enabled`) is the *same*
flag on both backends — so obeying it on NVIDIA disables cuDNN and costs several times the
throughput, for a bug that cannot occur there. `configure_backend` now gates the disable on
`torch.version.hip` and logs when it ignores the flag. This matters because `full_run.yaml`
and `overnight.yaml` both set `miopen: false`, and those are exactly the configs a Colab run
reuses in order to stay comparable.

**bf16 is not universal.** The project trains in bf16 locally because it shares fp32's
exponent range and so needs no `GradScaler`. Turing cards — including Colab's free **T4
(sm_75)** — have no bf16 at all. `resolve_amp_dtype` checks `torch.cuda.is_bf16_supported()`
and falls back to fp16 with a scaler, loudly. The scaler is enabled *only* for fp16, so the
local bf16 path is unchanged. Effective dtype is recorded in the run result, because a run
that says bf16 when fp16 happened is not comparable to one that means it.

---

## Layout

```
src/onnm/
  io_radiograph.py   DICOM+JPEG loading: VOI LUT, MONOCHROME1 inversion
  dataset.py         records, label derivation, grouping, transforms, loaders, sampler
  model.py           backbones, head swap, Grad-CAM layer lookup, backbone freezing
  losses.py          FocalLoss, HardNegativeMiningLoss (OHEM), build_loss
  metrics.py         clinical metrics, bootstrap CIs, operating point, reliability
                     bins, per-stratum (anatomy/subtype) breakdowns
  train.py           training loop, schedulers, evaluate()
  calibrate.py       temperature scaling, threshold search, ECE
  explainability.py  Grad-CAM, box geometry, pointing game / IoU
  thermal.py         AMD ADL GPU telemetry + duty-cycle governor
  inference.py       single-image prediction for the app; uncertainty gating;
                     production-checkpoint pinning (reports/PRODUCTION marker)
  ood.py             pre-inference radiograph validation + softmax entropy gate
  config.py, utils.py
src/                 app-layer modules (not part of the onnm package):
  auth.py            PBKDF2 password hashing, registration, session helpers
  oauth.py           Google Sign-In via Streamlit native OIDC; identity -> account
  database.py        SQLite (data/users.db): users + per-user scan history
  storage.py         de-identified upload storage under data/user_uploads/{uuid}/
  legal.py           ToS, Privacy Policy, Medical Disclaimer, Cookie Notice text
  report.py          torch-free per-case HTML report builder (print-to-PDF ready)
  ood_validator.py   shim -> onnm.ood (as inference.py is for onnm.inference)
scripts/             download, verify_data, verify_env, make_splits, train,
                     calibrate, evaluate, gradcam_report, overfit_check,
                     stratified_report (per-anatomy/subtype errors), ablate_tta
configs/             base.yaml + overrides: densenet121_3class, full_run,
                     overnight, specificity_tuning
  ablations/         ohem_only, augs_only -- separate the overnight regression
notebooks/           01_data_sanity, kaggle_train, colab_train
tests/               311 tests, synthetic fixtures, no dataset required
app.py               Streamlit UI;  .streamlit/config.toml binds loopback, telemetry off
MODEL_CARD.md        intended use, training data, measured limits, failure modes
.github/workflows/   ci.yml — ruff + torch-free fast tests + full suite on CPU torch
```

---

## App layer — auth, storage, OOD gate

**Two sign-in paths, chosen by configuration, never both at once.** `oidc_configured()`
returns true when secrets carry an `[auth]` block with a Google client; the app then shows
only "Continue with Google" and `st.login("google")` drives Streamlit's native OIDC. With
no such block — the normal state of a local checkout — the password forms render instead.
The fallback is not a second door into the same deployment; it is what lets a clone run
without a Google client of its own.

**Federated accounts store no password, and the schema enforces it.** `users` carries
`auth_provider` ('password' | 'google') and `provider_subject`, with a CHECK constraint
pairing them: a password account needs a hash and no subject, a Google account needs a
subject and a NULL hash. This is a database constraint rather than a Worker convention
because the failure it prevents — one account reachable by two different proofs of
identity — is an authentication bypass, not a data-quality problem. `verify_password`
returns False for a non-string, and `authenticate_user` treats a federated account exactly
like an unknown one (same dummy hash, same wasted work) so the login form cannot be timed
to discover which addresses use Google.

Identity keys on Google's `sub`, not email: a Workspace address can be reassigned after an
account closes, so `get_or_create_oauth_user` looks up by subject first and falls back to
email only to *return* an existing account — never to convert one. Silently upgrading a
password account to Google would let anyone who can prove control of an address take it
over.

The Streamlit app is gated behind accounts either way. `data/users.db` (SQLite,
gitignored under `data/`) stores emails, salted PBKDF2-HMAC-SHA256 hashes
(600k iterations), ToS-acceptance timestamps, and per-user scan history.
Uploads are stored de-identified under `data/user_uploads/{user_uuid}/` with
UUID filenames: DICOM headers pass through a PII-stripping pass (private tags,
PN/date fields, patient/institution identifiers removed, UIDs regenerated);
standard images are re-encoded to metadata-free PNG. De-identification is
header-level only — burned-in pixels are not scanned. Legal text lives in
`src/legal.py` and renders in expandable footers.

**OOD gate (`onnm.ood`), two stages.** The classifier is a closed-set softmax:
any input — a hotdog photo included — is forced into normal/benign/malignant.
Stage 1 rejects non-radiographs *before* inference on named heuristics
(colorfulness, dynamic range, 256-bin histogram entropy ≤ 7.5 bits, strong-
gradient fraction ≤ 0.45, minimum size); the app then shows a hard
"Invalid Image" rejection with per-check reasons. Stage 2 computes normalized
predictive entropy and max softmax probability on every prediction; the app
downgrades a lesion call to **"Non-Diagnostic / Inconclusive"** when max prob
< 0.65 or normalized entropy ≥ 0.90. Both stages are **opt-in parameters on
`RadiographClassifier.predict`** and default off, so scripted evaluation of
the curated dataset is unchanged. The gate can only withdraw a positive call —
it never issues one and never moves the calibrated threshold.

---

## Config system

YAML, deep-merged. `base.yaml` → `--override a.yaml` → `--override b.yaml` → `--profile x`.
Lists replace wholesale; dicts merge. Access via attribute or `cfg.lookup("a.b.c")`.

Profiles: `kaggle`, `colab`, `smoke`. **`colab` deliberately sets no `paths:`** —
`verify_data.py` and `make_splits.py` accept no `--profile`, so a profile that moved
`data_root` would apply to training but not to the gates that check the data. The notebook
symlinks the dataset into the default location instead.

Checkpoints **embed the config they were trained with**. `inference.py` reads that, not the
YAML on disk, so editing a config cannot desynchronise the app from a trained model.

---

## Invariants — do not break these

1. **Never fit a threshold or temperature on test.** Both are fitted on val and applied
   unchanged. `scripts/calibrate.py --split test` warns loudly.
2. **A weighted loss and `balanced_sampler` are mutually exclusive.** Both correct the
   imbalance; together they over-predict malignant. `build_sampler` raises.
3. **`data.crop_foreground` disables lesion-box localisation scoring.** It changes the
   geometry `map_box_to_model_space` models. `evaluate_localisation` raises rather than
   report meaningless numbers.
4. **MONOCHROME1 DICOM must be inverted.** Silent failure otherwise: valid arrays,
   converging loss, a model that learned an inverted world.
5. **Grad-CAM must run outside `no_grad`.** It backpropagates to the hooked layer.
6. **Temperature scaling is monotone** — it cannot change any argmax. Accuracy, recall and
   AUC are identical before and after. Only confidence moves.
7. **Accuracy is not a headline metric.** "Never malignant" scores 90.9%. Report malignant
   recall, PR-AUC, and bootstrap CIs.
8. **`RandAffine` subsumes `RandRotated`+`RandZoomd`** — enabling all three interpolates
   twice and blurs trabecular texture.
9. **The OOD/uncertainty gate defaults off in `predict()`.** Scripted evaluation of the
   curated dataset must be byte-identical with and without the app layer; only `app.py`
   passes `uncertainty_floor` / `entropy_gate` / payload validation.
10. **The uncertainty gate only withdraws lesion calls.** It may downgrade "Potential
    Bone Lesion" to "Non-Diagnostic / Inconclusive"; it never flips Normal to lesion and
    never alters the calibrated threshold, which stays fitted on validation only.
11. **Never store a plaintext password or an identified upload.** Credentials are salted
    PBKDF2 hashes; uploads are UUID-renamed and header-de-identified before they touch
    disk. `data/` stays gitignored.
12. **A federated account must never carry a password hash, and vice versa.** Enforced by
    a CHECK constraint in both `cloudflare/schema.sql` and `src/database.py`, not by the
    code that writes through them. Relaxing it makes one account reachable by two
    independent proofs of identity.
13. **Every outbound HTTP call must send an explicit User-Agent.** Cloudflare's edge bans
    the default `Python-urllib/3.x` signature with a 403 and a plain-text
    `error code: 1010` body, before the request reaches any Worker. See
    `community.USER_AGENT`.

---

## Commands

```powershell
.venv\Scripts\python.exe scripts\verify_env.py                     # gate 1
.venv\Scripts\python.exe scripts\verify_data.py                    # gate 2
.venv\Scripts\python.exe -m pytest -q                              # gate 3 (311)
.venv\Scripts\python.exe scripts\train.py --override configs\densenet121_3class.yaml --override configs\full_run.yaml --tag full
.venv\Scripts\python.exe scripts\calibrate.py --checkpoint reports\<run>\best.pt --sweep
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint reports\<run>\best.pt
.venv\Scripts\python.exe -m streamlit run app.py                   # http://localhost:8501
```

`calibrate.py` writes `calibration.json` beside the checkpoint; the app and `evaluate.py`
read it automatically. It exits non-zero when the operating point is unusable.

**Colab** (`notebooks/colab_train.ipynb`) — a second free GPU for the ablation backlog.
Everything is staged through Drive at `MyDrive/OSTEONEURALNETWORK/`: `onnm-code.zip`,
`BTXRD.zip`, `splits.json`. It unpacks to `/content` (local SSD — unzipping onto Drive is
pathologically slow for 3.7 k small files), symlinks the data into `data/raw/BTXRD`, copies
the local `splits.json` so results stay comparable, runs gates 1/2/3/6, then a 2-epoch smoke
run, then the two ablations, then copies `reports/` back to Drive. Colab wipes `/content` on
disconnect, so that last step is not optional. Install with `--no-deps`: several project
dependencies list torch, and pip would pull a CPU wheel over Colab's CUDA build.

## Community loop (hosted)

```
Google Sign-In --OIDC--> Streamlit Cloud (app.py) --HTTPS+key--> Cloudflare Worker + D1
                            OOD gate, inference                 users, submissions, feedback
                            opt-in share checkbox               review queue, batches
                            "this looks wrong" button                 |
                                                                     v
                              scripts/export_batch.py --> manifest.csv --> Colab retrain
```

Live endpoints: Worker `https://onnm-community.kali-fz.workers.dev`, D1 `onnm-community`
(id `961f0440-7ff1-466e-88fe-0c2b30f3083b`, 5 tables, schema_version 2), app
`https://osteoneuralnetwork-model-af5ynv9qxg7u8rc5epdprr.streamlit.app`.

- `cloudflare/` — Worker (`src/worker.js`), `schema.sql`, `migrations/`, `wrangler.toml`,
  deploy README. Migrations are applied by hand:
  `npx wrangler d1 execute onnm-community --remote --file=./migrations/NNNN_*.sql`.
  `0002_google_oauth.sql` rebuilds `users` (SQLite cannot relax NOT NULL in place) and
  preserves existing rows as password accounts.
- `src/community.py` — client; **fails soft** so a dead API never blocks inference.
- `src/backend.py` — accounts go to D1 when `ONNM_COMMUNITY_URL`/`_KEY` are set, local
  SQLite otherwise. `auth.py` imports from here, so hashing stays in one place.
- `src/community_ui.py` — consent checkbox, feedback widget, admin review queue.
- `scripts/export_batch.py` — approved rows to a `controls_manifest`-format CSV.

**Free tier only: Workers + D1, no R2, no payment method.** Shared images are the 256px
preprocessed PNG as base64 in D1 (~30 KB each), capped at 200 MB in the Worker. With no
card on the account, overage cannot bill — it fails closed.

**Invariant 9 — user feedback is a signal, never a label.** `user_says_wrong` and
`user_suggested_label` are untrusted; only `admin_label`, set during human review, reaches
training. Enforced three times: a schema trigger that aborts approval without a label, the
review endpoint, and the export query. Redundant on purpose — every other bug here
announces itself, this one would quietly train on a hotdog labelled "normal bone".

**Colab cannot host the app.** Runtimes are ephemeral (~90 min idle, 12 h cap), have no
persistent URL, and need the owner's browser session.

**Hosting is Streamlit Community Cloud** (`deploy/streamlit-cloud/`) — free, 2.7 GB RAM,
deploys from GitHub, redeploys on push. **Hugging Face Spaces is not an option:** Gradio
and Docker Spaces now require PRO and Streamlit is not an offered SDK at all; only Static
(client-side, no Python) is free. Netlify cannot host it either — static sites and JS
functions only. Both would require an ONNX-in-browser rewrite.

**Two hosting landmines, both already paid for — do not rediscover them.**

*Cloudflare's edge bans `Python-urllib`.* A request carrying the default urllib
User-Agent gets HTTP 403 with a plain-text `error code: 1010` body, generated at the edge
and never reaching the Worker. Every component looks correct in isolation, and `curl` and
a browser both get a clean 200, so the obvious next step — testing with curl — actively
confirms the wrong conclusion. `community.USER_AGENT` fixes it; a non-JSON error body is
now reported as coming from the gateway rather than surfaced as a bare `error code: 1010`.

*Streamlit's OIDC needs `httpx`, which nothing declares.*
`authlib.integrations.starlette_client` imports httpx, Authlib does not list it as a hard
dependency, and Streamlit's docs say only "install Authlib". The resulting
`ModuleNotFoundError` is raised inside `_create_oauth_client`, which sits *outside* the
`/auth/login` route's `try/except` (that one returns a tidy 400 "Authentication error"),
so it escapes as a bare HTTP 500 `Internal server error.` with nothing in the UI naming
the cause. `requirements.txt` pins `httpx` and `itsdangerous` explicitly.

`reports/` is gitignored, so a clone has no weights. `src/checkpoint_fetch.py` downloads
one at boot from `ONNM_CHECKPOINT_URL` and pins it. It verifies the payload starts with
torch's zip magic, because a CDN 404 typically returns HTTP 200 with an HTML page, which
would otherwise land in `best.pt` and fail much later inside `torch.load`. The default run
name is `hosted`, not `production` — that would collide with the `reports/PRODUCTION`
marker on case-insensitive filesystems.

---

## Current state

| run | epochs | val ROC-AUC | notes |
|---|---|---|---|
| `smoke-20260822-012828` | 1 | ~0.5 | throwaway; bal-acc 0.483 |
| **`full-20260822-041653`** | **26 (best 19)** | **0.8905** | **current best; calibrated + test-evaluated** |
| `overnight-20260822-055132` | 39 (best 24) | 0.8629 | aggressive augs + OHEM; **regressed** |

**`full-20260822-041653` held-out test** (n=536): macro ROC-AUC 0.893, PR-AUC 0.814,
F1 0.764, balanced accuracy 0.749. Malignant recall 0.633 [0.490–0.776]. 3 normal films
called malignant. Calibrated T=1.41, threshold 0.496 → 0.813 sensitivity / 0.848
specificity. ECE improved 0.053 → 0.017.

**The overnight run underperformed.** ROC-AUC is threshold-independent, so 0.863 vs 0.891
is genuinely worse ranking — not recoverable by tuning the threshold. False positives did
fall (65 → 37 normal films called lesion) but malignant recall fell with them (0.653 →
0.469), which is a bias shift, not better discrimination. The aggressive augmentation and
OHEM penalty are the suspects. `full-20260822-041653` remains the checkpoint to serve.

**Known open issue:** false positives on complex joint anatomy (a normal pelvis flagged at
59.6%). The durable fix is more normal controls in training, not a harsher loss — the
sensitivity/specificity curve can only be slid along, not moved, without new data. The
app-layer uncertainty gate now *suppresses presentation* of low-confidence lesion calls
(max prob < 0.65 or normalized entropy ≥ 0.90 → "Non-Diagnostic / Inconclusive"), and the
pre-inference validator rejects non-radiograph uploads outright — but neither changes the
underlying ranking quality; they are presentation-layer containment, not the data fix.

---

## Operating-point trade (validation, `full` run)

| specificity floor | threshold | sensitivity |
|---|---|---|
| 0.70 | 0.388 | 0.869 |
| 0.80 | 0.496 | 0.783 |
| 0.90 | 0.658 | 0.670 |
| 0.95 | 0.813 | 0.509 |

Moving 80% → 90% specificity costs ~6 additional missed cancers per 49. A clinical
decision, not a modelling one.
