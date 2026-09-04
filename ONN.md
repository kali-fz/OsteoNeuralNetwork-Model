# ONN: model version ledger

**Generated from `model_versions.json` by `scripts/version_model.py render`.**
Do not edit by hand: the JSON is the source of truth and a test asserts
this file is in step with it.

Every training generation is registered here *before* anything is promoted,
and promotion is a separate, guarded act. A run that regresses is recorded as
`held`, `reports/PRODUCTION` does not move, and the previous checkpoint keeps
serving, so a bad retrain costs a row in this table and nothing else.

| level | means |
|---|---|
| major | a different model, another architecture family or task head |
| minor | a deliberate recipe change (augmentation, loss, backbone) |
| patch | the same recipe with more data, which is what the daily community loop produces |

**Serving now:** v2.0.0 (`sweep-20260904-002410-w025-20260904-150749`)

---

## Versions

| version | date | status | run | macro ROC-AUC | malignant recall | misc rejection | community rows |
|---|---|---|---|---|---|---|---|
| **v2.0.0** | 2026-09-04 16:53 UTC | serving | `sweep-20260904-002410-w025-20260904-150749` | 0.9206 | 0.6735 | - | 8 |
| **v1.0.0** | 2026-08-23 01:09 UTC | superseded | `full-20260822-041653` | 0.8934 | 0.6327 | - | - |

---

## Detail

### v2.0.0: serving

- **Registered** 2026-09-04 16:53 UTC
- **Run** `sweep-20260904-002410-w025-20260904-150749`
- **Parent** v1.0.0
- **Note** Lesion head replaces Grad-CAM as the served explanation. Pointing game 0.0936 -> 0.7228 against a chance level of 0.0314; mean IoU 0.0428 -> 0.3584. The map is a trained output, not an attribution: Grad-CAM on this same checkpoint still scores 0.1348. Classification improved rather than being traded away: macro ROC-AUC 0.8934 -> 0.9206 and malignant recall 0.6327 -> 0.6735. First version to record activation on healthy films, at 19.7 percent.
- **best.pt sha256** `2ab3aecabf56577947c12bc02ba35d59529ff557c9f330218dd56de8bd6bbd07`
- **Community data** batches = 1, class_balance = {'malignant': 4, 'normal': 4}, lesion_rows_total = 8, ood_rows_total = 0

| metric | value |
|---|---|
| balanced_accuracy | 0.7872 |
| bone_acceptance | 0.9600 |
| lesion_normal_quiet | 0.8030 |
| lesion_pointing_game | 0.7228 |
| macro_f1 | 0.8042 |
| macro_pr_auc | 0.8612 |
| macro_roc_auc | 0.9206 |
| malignant_ppv | 0.8049 |
| malignant_recall | 0.6735 |
| malignant_recall_hi | 0.7959 |
| malignant_recall_lo | 0.5306 |
| normal_specificity_at_threshold | 0.8364 |

### v1.0.0: superseded

- **Registered** 2026-08-23 01:09 UTC
- **Run** `full-20260822-041653`
- **Note** Baseline: the run every number in overview.md refers to. Trained on BTXRD alone, before the community loop existed.
- **best.pt sha256** `f6b0ae7e0f257edd58a7bff7ea4fd34aaf5ff53ddbfc09892ee42fd560937c7c`

| metric | value |
|---|---|
| balanced_accuracy | 0.7494 |
| bone_acceptance | 0.9600 |
| macro_f1 | 0.7644 |
| macro_pr_auc | 0.8136 |
| macro_roc_auc | 0.8934 |
| malignant_ppv | 0.7561 |
| malignant_recall | 0.6327 |
| malignant_recall_hi | 0.7755 |
| malignant_recall_lo | 0.4898 |
