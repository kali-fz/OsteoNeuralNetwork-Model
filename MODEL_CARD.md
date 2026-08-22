# Model Card — OsteoNeuralNetwork-Model (ONNM)

DenseNet-121 classifier for triage of primary bone tumours on plain 2D
radiographs. Last updated 2026-08-22, describing checkpoint
`full-20260822-041653` (the pinned production run).

---

## Intended use

**Research prototype only.** ONNM is an unvalidated research tool with no FDA,
CE, or MHRA clearance. It is not a medical device, must not inform patient-care
decisions, and every radiograph it touches requires review by a qualified
clinician regardless of the model's output.

- **Intended task:** flag musculoskeletal radiographs that may contain a
  primary bone tumour (benign or malignant) for prioritised human review.
- **Intended users:** researchers studying explainable medical imaging models.
- **Out of scope:** diagnosis, screening programmes, paediatric-specific use,
  CT/MRI, chest/skull/spine films, post-operative films with hardware, any
  deployment where the output reaches a patient or clinician as a finding.

## Architecture

| | |
|---|---|
| Backbone | DenseNet-121, ImageNet-pretrained (torchvision weights) |
| Head | 3-class linear head: normal / benign / malignant |
| Input | 256 px grayscale, aspect-preserved, padded; MONAI transform chain |
| Verdict | calibrated P(benign)+P(malignant) vs a validation-fitted threshold |
| Explainability | Grad-CAM on `features.denseblock4` |
| Hardware | trained on one AMD RX 7900 XT (ROCm 7.2.1, Windows) |

## Training data

- **Source:** BTXRD — 3,746 plain radiographs with tumour annotations,
  licensed **CC BY-NC-ND 4.0** (non-commercial, no redistribution of
  derivatives, including Grad-CAM overlays).
- **Splits:** 2,675 / 535 / 536 (train/val/test), grouped by a surrogate
  patient id reconstructed from consecutive-image metadata so multiple views of
  one patient never straddle a split. Zero group leakage is verified by test.
- **Class balance:** heavily skewed; only ~342 malignant images exist in total
  (~244 in training, ~49 in test). Every malignant-class metric therefore
  carries a wide confidence interval by construction.
- **Labels:** derived from the release's one-hot indicator columns; subtype
  and anatomy columns are retained for stratified reporting.

## Performance (measured, not projected)

Test-split results for `full-20260822-041653`, threshold fitted on validation
only:

- **Malignant recall (test): 0.633, 95% CI [0.490-0.776].** Roughly **one in
  three malignant films is missed**. This is the number that matters most for
  anyone reading a result, so it is stated first: a "normal" verdict from this
  model is weak evidence of absence, and must never be used to decide against
  seeking care.
- **Macro ROC-AUC (test): 0.893**
- Validation operating points (specificity-floor mode):
  - holding specificity ≥ 80% → sensitivity ≈ 78%
  - holding specificity ≥ 90% → sensitivity ≈ 67% (~6 more missed cancers
    per 49 than the 80% point)
- Calibration: temperature scaling (T ≈ 1.41) fitted on validation;
  the app refuses to present an uncalibrated probability as a policy.

Which specificity floor binds is a clinical policy decision that has **not**
been made; both are reported. Bootstrap 95% CIs accompany every headline number
in `reports/<run>/metrics_test.json` — with 49 malignant test images, the
interval is the result.

## Measured limitations

- **False positives on complex normal anatomy.** Overlapping structures
  (pelvis, hip, growth plates) have produced lesion calls on normal films
  (observed cases: a normal pelvis at 59.6%, a normal femur at 69.8%). The
  planned fix is more normal controls in training; the app-layer uncertainty
  gate only contains the symptom.
- **Grad-CAM localisation is unscored.** Pointing-game accuracy has not been
  measured (`scripts/gradcam_report.py` has never been run on a trained
  checkpoint), so the claim "the model looks at lesions" is currently
  unverified.
- **Single distribution.** Trained and evaluated on BTXRD only. No external
  validation exists; performance on other scanners, populations, or
  acquisition protocols is unknown and should be presumed worse.
- **Single view.** Each film is scored independently; no multi-view consensus.
- **2D only.** No CT/MRI/volumetric support.

## Known failure modes

- **Out-of-distribution inputs.** A closed-set softmax forces any image into
  one of three classes. Mitigation: a two-stage OOD gate (statistical
  pre-screen + predictive-entropy withdrawal of low-confidence lesion calls),
  but the pre-screen is heuristic — a grayscale photograph with X-ray-like
  statistics can pass stage 1, and the gate thresholds were set on synthetic
  phantoms, not yet tuned on real film distributions.
- **Bias shift under loss re-weighting.** An OHEM-weighted training run
  regressed macro ROC-AUC (0.891 → 0.863) while appearing to reduce false
  positives — it had shifted its bias toward "normal", not learned better
  discrimination. Threshold-independent metrics are required when judging any
  specificity intervention.
- **Confident errors.** Temperature scaling improves average calibration, not
  worst-case; individual predictions can be confidently wrong in either
  direction.

## Ethical considerations

- Uploads to the local app are de-identified (DICOM PII stripping, UUID
  filenames, EXIF-free re-encode) and stay on the local machine; see the
  Privacy Policy in `src/legal.py`.
- A missed cancer (malignant called normal) sends a patient home; this error
  class is reported separately from every other error in all evaluation
  output, and no operating point should be chosen without reading it.
- BTXRD's licence forbids redistributing derived images. Grad-CAM overlays and
  case reports must remain local.

## Maintenance

Production checkpoint pinning: `reports/PRODUCTION` names the run the app
loads by default; experimental runs never become the default silently. Gates
(env, data, tests) are documented in `overview.md`; open work is tracked in
`TODO.md`.
