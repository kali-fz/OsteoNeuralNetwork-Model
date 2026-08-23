# ONN — model version ledger

**Generated from `model_versions.json` by `scripts/version_model.py render`.**
Do not edit by hand: the JSON is the source of truth and a test asserts
this file is in step with it.

Every training generation is registered here *before* anything is promoted,
and promotion is a separate, guarded act. A run that regresses is recorded as
`held`, `reports/PRODUCTION` does not move, and the previous checkpoint keeps
serving — so a bad retrain costs a row in this table and nothing else.

| level | means |
|---|---|
| major | a different model — another architecture family or task head |
| minor | a deliberate recipe change (augmentation, loss, backbone) |
| patch | the same recipe, more data — what the daily community loop produces |

**Serving now:** v1.0.0 (`full-20260822-041653`)

---

## Versions

| version | date | status | run | macro ROC-AUC | malignant recall | misc rejection | community rows |
|---|---|---|---|---|---|---|---|
| **v1.0.0** | 2026-08-23 01:09 UTC | serving | `full-20260822-041653` | 0.8934 | 0.6327 | — | — |

---

## Detail

### v1.0.0 — serving

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
