# OsteoNeuralNetwork-Model

<p align="center">
  <img src="assets/onnm-architecture.svg" alt="Animated ONNM architecture: radiograph upload, local inference, explainable result, and opt-in D1 community review loop" width="100%">
</p>

<p align="center">
  <a href="https://osteoneuralnetwork.com"><strong>osteoneuralnetwork.com</strong></a>
  · <a href="MODEL_CARD.md">Model card</a>
  · <a href="overview.md">Agent overview</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20prototype-f59e0b?style=flat-square" alt="Research prototype">
  <img src="https://img.shields.io/badge/licence-Apache--2.0-22c55e?style=flat-square" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/cost-%240%20running%20cost-22c55e?style=flat-square" alt="Zero running cost">
  <img src="https://img.shields.io/badge/medical%20device-no-dc2626?style=flat-square" alt="Not a medical device">
</p>

## Stack at a glance

| Technology | Version / choice | Users |
|---|---|---|
| Python | 3.12 | **Live, D1-backed** |
| PyTorch | 2.9.1 + ROCm 7.2.1 | Google OIDC accounts |
| MONAI | 1.5.2 | Community submissions |
| Model | DenseNet-121 · 3 classes | `normal / benign / malignant` |
| Interface | Cloudflare Worker + static SPA | [osteoneuralnetwork.com](https://osteoneuralnetwork.com) |
| Persistence | Cloudflare Worker + D1 | Live count is intentionally read from the protected D1 health tag, not hard-coded here |

## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/kali-fz"><img src="https://github.com/kali-fz.png?size=96" width="72" height="72" style="border-radius:50%" alt="kali-fz"><br><sub><b>kali-fz</b></sub></a><br><sub>Project lead</sub></td>
    <td align="center"><a href="https://github.com/umfhero"><img src="https://github.com/umfhero.png?size=96" width="72" height="72" style="border-radius:50%" alt="umfhero"><br><sub><b>umfhero</b></sub></a><br><sub>AI Assistant</sub></td>
    <td align="center"><a href="https://github.com/Yaso-cyber"><img src="https://github.com/Yaso-cyber.png?size=96" width="72" height="72" style="border-radius:50%" alt="Yaso-cyber"><br><sub><b>Yaso-cyber</b></sub></a><br><sub>GRC</sub></td>
  </tr>
</table>

## What this project is

ONNM is an explainable research prototype for triaging primary bone tumours from plain radiographs. It classifies each film as normal, benign, or malignant, then shows the evidence that drove the prediction with Grad-CAM.

The real-world problem is not simply “make a classifier.” Bone tumours are rare, radiographs are difficult to interpret consistently, and an apparently strong accuracy score can hide missed cancers. ONNM therefore focuses on the parts that make a model more useful to investigate:

- grouped patient-safe splits rather than randomly scattering multiple views across train and test;
- malignant recall, PR-AUC, confidence calibration, and bootstrap intervals rather than accuracy alone;
- an uncertainty/OOD gate that can withdraw a confident-looking call;
- Grad-CAM localisation scored against lesion boxes;
- a consent-based community loop where only de-identified 256px inputs enter D1 and only a human-approved label reaches training.

This is not a diagnostic device. It has no FDA, CE, or MHRA clearance. Every radiograph requires review by a qualified clinician.

## Architecture

The diagram above is animated on supporting SVG viewers: dashed request paths move left-to-right and the opt-in community packet travels into the review layer. The source is [assets/onnm-architecture.svg](assets/onnm-architecture.svg), so the labels stay version-controlled and readable instead of becoming an outdated screenshot.

The normal path is local to the app host:

1. A user uploads DICOM, PNG, JPEG, BMP, or TIFF.
2. The loader applies DICOM VOI/LUT rules, inverts MONOCHROME1 correctly, validates the payload, preserves aspect ratio, and produces a 256px model input.
3. DenseNet-121 returns the three-class probabilities; calibration and the uncertainty gate shape the displayed verdict.
4. Grad-CAM produces a visual explanation. The result page keeps the raw class breakdown visible beneath the simpler normal/lesion headline.
5. If the user opts in, the processed image and prediction are sent through the authenticated Worker into D1 for review. Sharing is never implied by uploading.

## How I built it

### Why I started building it

The gap I kept coming back to was in early detection tools for bone lesions, where accessible and genuinely accurate options are still thin on the ground. I wanted a zero-cost research prototype that could take a musculoskeletal X-ray and return a calibrated, explainable second opinion. Because it deals with medical data, I couldn't just hack something together and hope it held up. It had to be secure and accurate from the first design decision, which is what pushed the planning phase out well before I wrote a line of code.

### Planning before any code

Given how high the stakes are with medical data, I brought in two people to sanity-check the architecture rather than working it out alone.

On the infrastructure side, I worked with a Master's student from King's College London who is currently interning at Cloudflare on frontier AI models. Together we designed a zero-trust, edge-based backend that could handle heavy image processing without touching patient confidentiality or racking up cloud costs.

On the governance side, I brought in a second Master's student who specialises in GRC, and we drafted the legal framework and terms of service together, along with anti-misuse pipelines aimed at data poisoning, working through how to handle people who might try to dilute the model by uploading irrelevant images or mislabelling healthy scans as cancerous.

### Building it

Once the sitemap was settled, the database schemas were locked and the underlying maths had been checked, I moved into the build itself.

I set up Google Login for authentication and put a guarded Cloudflare Worker in front of D1, with every image anonymised on the way in so all DICOM patient information is stripped before it reaches the database. Object storage would have been the obvious home for the images, but R2 needs a payment method on file, so consented uploads live as de-identified 256px images inside D1 itself and the storage layer stays behind one accessor. The reasoning is written up in [cloudflare/README.md](cloudflare/README.md).

On the modelling side, I ran deep training loops on my local GPU across 3,746 clinical radiographs, split 70/15/15 and grouped by surrogate patient so that several views of the same person cannot straddle train and test. Malignant films are only 9.1% of the dataset, so I used focal loss with inverse-frequency alpha and selected the decision thresholds on validation data only, then calibrated the probabilities afterwards, which is what keeps false positives down on complex but normal joint anatomy.

Because compute is expensive and models are bad at knowing what they don't know, I also built an out-of-distribution quality gate. If someone uploads a photo of a hotdog, a landscape or anything that isn't a valid musculoskeletal X-ray, the system flags the invalid input and stops processing rather than making a blind guess, which saves compute and keeps the outputs meaningful.

To close it out, I turned the sitemap into a working web application, set up hosting for reasonable uptime, and added Grad-CAM heatmaps so a doctor or researcher can actually see what the model is reacting to rather than trusting a black box.

### How the code is organised

The project is deliberately more than a notebook wrapped in a web page. The implementation is split into small, testable boundaries:

| Layer | What it does |
|---|---|
| Radiograph I/O | Reads DICOM and common raster formats, honours VOI LUT and photometric interpretation, and strips metadata before community storage. |
| Dataset pipeline | Derives labels from BTXRD one-hot fields, reconstructs surrogate patient groups, creates leakage-resistant splits, and applies aspect-safe transforms. |
| Model | Uses ImageNet-shaped DenseNet-121 with a three-class head, calibrated probabilities, threshold selection on validation only, and Grad-CAM hooks. |
| Evaluation | Reports clinically meaningful class metrics, PR-AUC, bootstrap confidence intervals, reliability bins, malignant error paths, and lesion-localisation scores. |
| App | Provides upload, OOD checks, inference, explainability, the Terms gate, Google sign-in, and per-user session handling. |
| Community backend | A guarded Cloudflare Worker validates payloads, applies per-user and storage caps, stores consent metadata, and exposes a human review gate before export. |
| Verification | Six gates plus a focused pytest suite catch environment, data, decoding, training, calibration, privacy, and backend regressions. |

The hardest engineering constraints are practical rather than decorative: ROCm on Windows has a training-mode MIOpen failure, Windows DataLoader workers can multiply the MONAI cache, the malignant class is only 9.1% of BTXRD, and a model can learn a shortcut such as a collimation edge instead of a lesion. The code treats those as explicit failure modes with checks and documented workarounds.

Retraining is version-controlled for the same reason. Every generation is registered in [ONN.md](ONN.md) before anything is promoted, and promotion is a separate guarded step. A run that regresses is recorded as `held`, the previous checkpoint keeps serving, and a bad retrain costs a row in a table rather than the working model.

### Where it stands now

ONNM is a functional prototype built on top of proper academic collaboration and GRC planning rather than a rushed hack, and what's left is a UI redesign and the final pass on the legal side before it's ready to go further.

## Results, with the necessary caveats

The current full run uses DenseNet-121 and early stopping. On the held-out test split (`n=536`, never used for training or calibration):

| Class | Sensitivity | Specificity | PR-AUC | n |
|---|---:|---:|---:|---:|
| Normal | 0.859 | 0.805 | 0.886 | 269 |
| Benign | 0.757 | 0.852 | 0.847 | 218 |
| Malignant | 0.633 | 0.979 | 0.708 | 49 |

Macro ROC-AUC is **0.893**, macro PR-AUC **0.814**, and balanced accuracy **0.749**. The malignant recall interval is wide: **0.633 [0.490, 0.776]**. Six of 49 malignant films were called normal. Those numbers are why this repository presents the model as research software, not clinical advice.

## Run it locally

The website and API run on Wrangler, Cloudflare's local runtime:

```powershell
npm install
npm run dev
```

Open <http://localhost:8787>.

For the training and evaluation side, which is Python:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest tests/ -q
```

A checkpoint is required under `reports/*/best.pt`; the hosted deployment fetches its pinned checkpoint from the configured release URL.

For the full pipeline:

```powershell
.venv\Scripts\python.exe scripts\download_btxrd.py
.venv\Scripts\python.exe scripts\make_splits.py
.venv\Scripts\python.exe -m pytest tests\ -q
.venv\Scripts\python.exe scripts\train.py --override configs\densenet121_3class.yaml --override configs\full_run.yaml --tag full
.venv\Scripts\python.exe scripts\calibrate.py --checkpoint reports\full-<timestamp>\best.pt --sweep
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint reports\full-<timestamp>\best.pt
```

## References

| Reference | Used for |
|---|---|
| [BTXRD dataset paper](https://doi.org/10.1038/s41597-024-04311-y) | Radiograph dataset, labels, boxes, and segmentation polygons. |
| [BTXRD on figshare](https://doi.org/10.6084/m9.figshare.27865398) | Dataset distribution and release files. |
| [DenseNet](https://doi.org/10.1109/CVPR.2017.243) | Backbone architecture. |
| [Grad-CAM](https://doi.org/10.48550/arXiv.1610.02391) | Visual explanation method. |
| [PyTorch](https://pytorch.org/) | Tensor and model runtime. |
| [MONAI](https://monai.io/) | Medical-imaging transforms and data pipeline components. |
| [Cloudflare Workers](https://developers.cloudflare.com/workers/) · [D1](https://developers.cloudflare.com/d1/) · [Containers](https://developers.cloudflare.com/containers/) | The website, the authenticated API, consented review storage, and server-side inference. |
| [Vite](https://vite.dev/) | Frontend build for the standalone web application. |

## Licence and safety

The software is Apache-2.0. BTXRD is **CC BY-NC-ND 4.0**; its licence applies to the dataset and derived radiograph visualisations, including Grad-CAM overlays. Do not redistribute dataset-derived images.

**Research tool, not a medical device and not medical advice.** Do not use this output for patient-care decisions. Every radiograph requires review by a qualified clinician.
