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
| UI | Streamlit 1.62 (`app.py`), localhost-only |
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
  model.py           backbones, head swap, Grad-CAM layer lookup
  losses.py          FocalLoss, HardNegativeMiningLoss (OHEM), build_loss
  metrics.py         clinical metrics, bootstrap CIs, operating point
  train.py           training loop, schedulers, evaluate()
  calibrate.py       temperature scaling, threshold search, ECE
  explainability.py  Grad-CAM, box geometry, pointing game / IoU
  thermal.py         AMD ADL GPU telemetry + duty-cycle governor
  inference.py       single-image prediction for the app
  config.py, utils.py
scripts/             download, verify_data, verify_env, make_splits, train,
                     calibrate, evaluate, gradcam_report, overfit_check
configs/             base.yaml + overrides: densenet121_3class, full_run, overnight
  ablations/         ohem_only, augs_only -- separate the overnight regression
notebooks/           01_data_sanity, kaggle_train, colab_train
tests/               201 tests, synthetic fixtures, no dataset required
app.py               Streamlit UI;  .streamlit/config.toml binds loopback, telemetry off
```

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

---

## Commands

```powershell
.venv\Scripts\python.exe scripts\verify_env.py                     # gate 1
.venv\Scripts\python.exe scripts\verify_data.py                    # gate 2
.venv\Scripts\python.exe -m pytest -q                              # gate 3 (189)
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

**Colab cannot host the app.** Runtimes are ephemeral (~90 min idle, 12 h cap), have no
persistent URL, and need the owner's browser session. Free hosting for a public demo is
Hugging Face Spaces, which is not built yet.

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
sensitivity/specificity curve can only be slid along, not moved, without new data.

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
