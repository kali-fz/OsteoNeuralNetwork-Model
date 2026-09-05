# Model Card: OsteoNeuralNetwork-Model (ONNM)

DenseNet-121 classifier with a trained lesion-localisation head, for triage of
primary bone tumours on plain 2D radiographs. Last updated 2026-09-05,
describing **v2.0.0** / `sweep-20260904-002410-w025-20260904-150749`, the pinned
production run. It replaced v1.0.0 (`full-20260822-041653`) on 2026-09-04.

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
| Explainability | Trained lesion map (FPN-lite decoder, 64x64, ~181k params); Grad-CAM on `features.denseblock4` retained as fallback |
| Hardware | trained on one AMD RX 7900 XT (ROCm 7.2.1, Windows) |

## Training data

- **Source:** BTXRD, 3,746 plain radiographs with tumour annotations,
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

Test-split results for **v2.0.0** (`sweep-20260904-002410-w025-20260904-150749`),
threshold fitted on validation only. v1.0.0's figures are given alongside, since
those are what these replaced.

- **Malignant recall (test): 0.673, 95% CI [0.531-0.796]** (v1.0.0: 0.633).
  Roughly **one malignant film in three is still missed**. This is the number
  that matters most for anyone reading a result, so it is stated first: a
  "normal" verdict from this model is weak evidence of absence, and must never
  be used to decide against seeking care.
- **Macro ROC-AUC (test): 0.921** (v1.0.0: 0.893)
- Validation operating points (specificity-floor mode):
  - holding specificity ≥ 80% → sensitivity ≈ 88% (v1.0.0: ≈ 78%)
- **False positives did not improve, and the acceptance target is not met.** Of
  the 269 normal test films, the served threshold calls **16.4%** of them a
  lesion, against v1.0.0's 15.2%. The recall and ROC gains above were bought
  with a slight rise here, not alongside a fall. The target is under 10%. Across
  a ten-run sweep, every configuration that improved detection made this worse,
  and the model now serving has the lowest false-positive rate of any run with a
  lesion head — so this is not a tuning gap. See TODO.md: BTXRD labels no normal
  film with a named joint, so it cannot teach what a healthy joint looks like.
- Calibration: temperature scaling (T ≈ 1.41) fitted on validation;
  the app refuses to present an uncalibrated probability as a policy.

Which specificity floor binds is a clinical policy decision that has **not**
been made; both are reported. Bootstrap 95% CIs accompany every headline number
in `reports/<run>/metrics_test.json`, with 49 malignant test images, the
interval is the result.

## Measured limitations

- **False positives on complex normal anatomy.** Overlapping structures
  (pelvis, hip, growth plates) have produced lesion calls on normal films
  (observed cases: a normal pelvis at 59.6%, a normal femur at 69.8%). The
  planned fix is more normal controls in training; the app-layer uncertainty
  gate only contains the symptom.
- **Localisation supports the claim that the model looks at lesions. It did not
  before v2.0.0.** v1.0.0 explained itself with Grad-CAM, scoring a pointing
  game of **0.0936** over 267 annotated test films against a measured chance
  level of **0.0314** — barely distinguishable from dropping the peak at random,
  and so not evidence of anything. v2.0.0 replaces it with a decoder trained on
  BTXRD's lesion polygons, scoring **0.7228** on the same films against the same
  boxes, mean IoU **0.3584** (from 0.0428). The gain is the head and not a
  better backbone: Grad-CAM scored on that same v2.0.0 checkpoint still returns
  0.1348.

  **Two limits, stated plainly.** The lesion map claims a lesion somewhere on
  **19.7%** of healthy films, measured on the absolute sigmoid — that is the
  direct measure of the failure this work was aimed at, and it is not solved.
  And nothing here shows the remaining activation is specifically *joint*
  anatomy: BTXRD's anatomy label is confounded with the class, so it contains no
  normal film labelled with a named joint and the question cannot be answered
  from this dataset at all.

  Historical note: pointing-game readings of 0.0000 before 2026-08-23 were an
  artefact of an inverted heatmap (MONAI's default postprocessing maps min/max
  to 1/0), fixed then. That correction changed no prediction — the CAM was
  display-only.
- **Single distribution.** Trained and evaluated on BTXRD only. No external
  validation exists; performance on other scanners, populations, or
  acquisition protocols is unknown and should be presumed worse.
- **Single view.** Each film is scored independently; no multi-view consensus.
- **2D only.** No CT/MRI/volumetric support.

## Known failure modes

- **Out-of-distribution inputs.** A closed-set softmax forces any image into
  one of three classes. Mitigation: a two-stage OOD gate (statistical
  pre-screen + predictive-entropy withdrawal of low-confidence lesion calls),
  but the pre-screen is heuristic: a grayscale photograph with X-ray-like
  statistics can pass stage 1, and the gate thresholds were set on synthetic
  phantoms, not yet tuned on real film distributions.
- **Bias shift under loss re-weighting.** An OHEM-weighted training run
  regressed macro ROC-AUC (0.891 → 0.863) while appearing to reduce false
  positives. It had shifted its bias toward "normal" rather than learned better
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
