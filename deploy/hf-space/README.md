---
title: ONNM Bone Lesion Triage
emoji: 🦴
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Research demo - explainable bone-lesion triage on radiographs
---

# ONNM — bone lesion triage (research demo)

**This is a research demonstration. It is not a medical device, not validated
for clinical use, and must not be used to make any care decision.**

Measured performance on held-out BTXRD test data: malignant recall
**0.633 [0.490–0.776]** — roughly one in three cancers missed. Grad-CAM
localisation has never been scored against ground-truth lesion boxes, so there
is no evidence yet that the heatmaps point at lesions rather than at
collimation edges or implants.

Do not upload identifiable patient data.

## Configuration

Set under Settings → Variables and secrets:

| name | kind | purpose |
|---|---|---|
| `ONNM_COMMUNITY_URL` | variable | Cloudflare Worker URL |
| `ONNM_COMMUNITY_KEY` | secret | app key — ordinary rows only |

`ONNM_ADMIN_KEY` must **not** be set here. Review and export run from the
maintainer's machine, so a leak of the Space's secret cannot approve training
data.

Without these the Space still runs; community features are simply disabled.
