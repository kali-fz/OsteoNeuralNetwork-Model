# OsteoNeuralNetwork-Model (ONNM)

Explainable detection of primary bone tumors on plain radiographs. Classifies each film as
**normal / benign / malignant**, and produces Grad-CAM heatmaps that are *scored against
ground-truth lesion boxes* rather than merely displayed.

Runs entirely on free resources: an open-access dataset, local AMD GPU via ROCm, and Kaggle's
free GPU quota as a fallback.

---

## Why the metrics look unusual

Predicting "not malignant" for every image scores **90.9% accuracy** on this dataset while
missing every cancer. Accuracy is therefore not reported as a headline number. The pipeline
optimises and reports:

| Metric | Why |
|---|---|
| **Malignant recall** | The headline. Early-stopping criterion. |
| PR-AUC | ROC-AUC is optimistic on a 9% class; the large true-negative pool flatters the false-positive rate. |
| Bootstrap 95% CIs | The test split holds **49 malignant images**. A point estimate alone is noise. |
| malignant→**normal** vs malignant→benign | A cancer called benign still gets followed up. A cancer called normal sends the patient home. Averaging those hides the difference that matters. |
| Grad-CAM pointing game | A model can hit high recall by keying on an implant or collimation edge. Recall cannot detect that; localisation can. |

---

## Dataset — BTXRD

[*A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone
Tumors*](https://www.nature.com/articles/s41597-024-04311-y) (Scientific Data, 2024),
figshare `10.6084/m9.figshare.27865398`.

- **3,746 grayscale radiographs** — 1,879 normal / 1,525 benign / **342 malignant** (9.1%)
- Per-lesion **bounding boxes and segmentation polygons** (LabelMe 5.x JSON), 1,867 annotated files
- Subtypes include osteosarcoma (297) and, folded into "other mt" (45), Ewing sarcoma
- 840 MB, single zip, no registration

**Licence: CC BY-NC-ND 4.0.** Research use is fine. NoDerivatives means derived images —
Grad-CAM overlays included — must not be redistributed. `data/` and `reports/` are gitignored.

### Three things the paper's prose does not tell you

Discovered by inspecting the actual release; all three are handled in code:

1. **The metadata ships as `dataset.xlsx`, not `dataset.csv`.** Needs `openpyxl`.
2. **The diagnosis is stored as one-hot indicator columns**, not a categorical label. The
   derivation `tumor=0 → normal`, `malignant=1 → malignant`, `benign=1 → benign` reproduces
   the published 1879/1525/342 distribution exactly, and `verify_data.py` re-checks that on
   every run.
3. **Filenames disagree with the table.** Every `image_id` ends in `.jpeg`, but the images
   directory mixes `.jpeg` and `.jpg`. Records are matched on filename stem.

### Patient grouping — read this before reporting any result

BTXRD has **no patient identifier**, but the images are not independent: **2,826 of 3,746
(75%)** fall into runs of consecutive `image_id` values sharing centre, age, sex, anatomy and
diagnosis while differing in view — multiple projections of one patient.

Splitting per image would scatter those siblings across train and test, and the resulting
score would measure memorisation of a lesion already seen from another angle.
`derive_groups()` reconstructs the grouping (3,746 images → **1,950 groups**) and
`make_splits.py` keeps every group intact.

This is a **heuristic**. It can merge two genuinely distinct patients who happen to be adjacent
and identical on every recorded field. That direction of error only withholds training data,
which is the safe way to be wrong — but state the limitation alongside any published result.

---

## Setup

### 0. Prerequisites

`git` is not on `PATH` but ships with GitHub Desktop. Add (adjust the version — it changes on
update):

```powershell
$env:PATH += ";$env:LOCALAPPDATA\GitHubDesktop\app-3.6.4\resources\app\git\cmd"
```

### 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 2. GPU — AMD Radeon (ROCm on Windows)

This machine has an **RX 7900 XT (gfx1100)**, so CUDA is unavailable. AMD ships Windows ROCm
wheels, but two prerequisites are strict:

- **AMD graphics driver ≥ 26.2.2** (Adrenalin → Settings → System → Software)
- **Python 3.12 exactly** — the wheels are cp312-only

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-rocm.txt
```

If either prerequisite fails, skip this and use the Kaggle fallback — the same code runs
unchanged with `--profile kaggle`.

### 3. Data

```powershell
.venv\Scripts\python.exe scripts\download_btxrd.py
.venv\Scripts\python.exe scripts\make_splits.py
```

---

## The six gates

Each fails faster than the one after it. Do not start training until all six are green.

| # | Command | Proves |
|---|---|---|
| 1 | `scripts\verify_env.py` | A real matmul and bf16 autocast execute on the GPU |
| 2 | `scripts\verify_data.py` | Class counts match the paper; splits are leakage-free |
| 3 | `python -m pytest tests\ -q` | 116 assertions on I/O, transforms, losses, metrics, geometry, inference |
| 4 | `notebooks/01_data_sanity.ipynb` | **Human eyes.** Images are not inverted; boxes land on lesions |
| 5 | `scripts\verify_data.py --deep` | Every image file decodes |
| 6 | `scripts\overfit_check.py` | The model memorises 32 images — the pipeline actually learns |

Gate 4 is not optional. Assertions verify shape; only looking verifies content. An inverted
radiograph produces perfectly valid float32 arrays and a converging loss curve.

Gate 6 is the highest-value check in the list. A model that cannot memorise 32 images has
misaligned labels, broken normalisation, or gradients not reaching the backbone — found in two
minutes rather than after a wasted day.

---

## Training

```powershell
.venv\Scripts\python.exe scripts\train.py --override configs\densenet121_3class.yaml
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint reports\train-<ts>\best.pt
.venv\Scripts\python.exe scripts\gradcam_report.py --checkpoint reports\train-<ts>\best.pt
```

`--profile smoke` runs one short epoch to validate the loop; `--profile kaggle` switches paths
and loader settings for the notebook fallback.

---

## The local web app

A Streamlit interface for reading one film at a time: upload, classify, and inspect the
Grad-CAM. Free, open source (Apache-2.0), and entirely local — no cloud, no API keys, no
telemetry.

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open <http://localhost:8501>. Stop it with `Ctrl+C`.

It needs a checkpoint under `reports/*/best.pt`; the sidebar lists every run it finds, newest
first. Accepts DICOM (`.dcm`/`.dicom`/`.ima`), PNG, JPEG, BMP and TIFF, and shows the
three-class breakdown alongside the binary verdict.

**Three things about it worth knowing:**

**The verdict is binary; the model is not.** The headline reads *Normal* vs *Potential Bone
Lesion*, which is `P(benign) + P(malignant)` against an adjustable threshold. Benign and
malignant are collapsed there only for legibility — the chart underneath keeps them apart,
because a cancer called benign still gets followed up while a cancer called normal sends the
patient home.

**Preprocessing is read out of the checkpoint, not out of `configs/`.** Each `best.pt` embeds
the config it was trained under, so editing a YAML months later cannot silently desynchronise
the app from the model. The backbone is also built with `pretrained=False`, since the
checkpoint overwrites every weight — which is what makes a first run work with no network.

**It binds loopback only.** Streamlit's defaults publish on `0.0.0.0` and phone home for usage
stats; `.streamlit/config.toml` turns both off, because neither belongs anywhere near a
radiograph. Uploads are held in memory and in one temporary file that is deleted before the
result renders.

The app carries a prominent disclaimer, and it is accurate: this is an unvalidated research
prototype with no regulatory clearance, and its output is not a diagnosis.

---

## Layout

```
src/onnm/
  io_radiograph.py   DICOM + JPEG loading: VOI LUT, MONOCHROME1 inversion
  dataset.py         records, label derivation, grouping, transforms, loaders
  model.py           ImageNet-pretrained backbones, head swap, CAM layer lookup
  losses.py          focal loss with inverse-frequency alpha
  metrics.py         clinical metrics, bootstrap CIs, operating point
  train.py           training loop, overfit check, evaluation
  explainability.py  Grad-CAM, box geometry, pointing game / IoU
  config.py          YAML loading with deep-merge overrides and profiles
  inference.py       single-image prediction + Grad-CAM for the web app
  utils.py           seeding, logging, device, checkpoints
scripts/             thin CLI wrappers (download, verify, split, train, evaluate, gradcam)
tests/               pytest suite; synthetic DICOM fixtures, no dataset required
configs/             base.yaml + experiment overrides
app.py               Streamlit UI (upload -> verdict -> heatmap), localhost only
.streamlit/          server defaults: loopback bind, telemetry off
notebooks/           01_data_sanity.ipynb (Gate 4), kaggle_train.ipynb (fallback)
```

---

## Design decisions worth knowing

**A custom loader instead of `LoadImaged`.** MONAI's loader returns raw stored pixel values.
For DICOM that is not enough: without the VOI LUT, subtle lytic lesions flatten into the
background, and without `PhotometricInterpretation` handling, a MONOCHROME1 film loads as a
photographic negative. That second failure is silent — valid array, plausible training curve,
a model that learned an inverted world. `tests/test_io_radiograph.py` asserts a synthetic
MONOCHROME1 file decodes identically to its MONOCHROME2 twin.

**Aspect ratio preserved.** The chain resizes the longest side to 256 then pads, rather than
squashing to a square. Lesion margin and periosteal reaction are morphological signs; a
distorted long-bone film deforms exactly the cues that carry the diagnosis.

**Horizontal flips only.** Left and right limbs are both anatomically valid, so a mirror is a
real radiograph. An upside-down film is not, and training on one teaches invariance to
information that is actually meaningful.

**bf16, not fp16.** On RDNA3, bf16 has fp32's exponent range, so it needs no `GradScaler` and
cannot silently underflow the small gradients a focal loss on a 9% class produces.

**Early stopping on malignant recall, not loss.** Validation loss is dominated by the 91% of
images that are not malignant and keeps improving long after the model stops getting better at
the only call that matters.

**Focal loss *or* a weighted sampler — never both.** Each corrects the imbalance once;
together they double-correct, driving over-prediction of malignant and destabilising training.

**`num_workers=0` on Windows.** Workers are spawned, not forked, so each duplicates the
in-memory cache. With `cache_rate=1.0` (~2.9 GB, comfortable in 32 GB) workers add memory
pressure and no throughput. The `kaggle` profile flips this — Linux has real fork semantics.

---

## Attribution

BTXRD is released under CC BY-NC-ND 4.0. Cite the dataset paper in any work using it:

> A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone
> Tumors. *Scientific Data* (2024). https://doi.org/10.1038/s41597-024-04311-y

**This is research software. It is not a medical device and must not be used for clinical
decision-making.**
