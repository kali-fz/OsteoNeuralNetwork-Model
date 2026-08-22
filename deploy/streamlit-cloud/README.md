# Deploying ONNM to Streamlit Community Cloud

**Free, no card, deploys straight from GitHub and redeploys on every push.**

## Why not Hugging Face Spaces

Spaces changed: Gradio and Docker Spaces now require a PRO subscription, and
Streamlit is no longer offered as an SDK at all. Only **Static** Spaces remain
free, and Static means client-side only — no Python, no PyTorch. Running this
app there would mean exporting to ONNX and rewriting the frontend in JavaScript.

Streamlit Community Cloud runs `app.py` as-is, with **2.7 GB** of memory —
comfortable for CPU torch plus DenseNet-121, which needs well under half of it.

## Deploy

1. Push `main` to GitHub (the repo must be public — Community Cloud only hosts
   public apps on the free tier).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **New app**.
3. Repository `kali-fz/OsteoNeuralNetwork-Model`, branch `main`,
   main file path **`app.py`**.
4. Under **Advanced settings → Secrets**, paste:

```toml
ONNM_COMMUNITY_URL = "https://onnm-community.<subdomain>.workers.dev"
ONNM_COMMUNITY_KEY = "the API_KEY from wrangler"
ONNM_CHECKPOINT_URL = "https://<direct link to best.pt>"
ONNM_CALIBRATION_URL = "https://<direct link to calibration.json>"
ONNM_CHECKPOINT_RUN = "hosted"
```

**Never put `ONNM_ADMIN_KEY` here.** Review and export run from your machine, so
a leak of the app's key cannot approve its own training data.

5. Deploy. First build takes several minutes (torch is a large download).

## The checkpoint

`reports/` is gitignored, so a clone carries no weights and the app would start
with no model. `src/checkpoint_fetch.py` downloads one at boot from
`ONNM_CHECKPOINT_URL`, caches it to `reports/hosted/best.pt`, and writes a
`reports/PRODUCTION` marker so the app serves that run rather than picking by
modification time.

Two ways to host the 28 MB file, both free:

- **Hugging Face model repo** (recommended). Hosting *models* is still free even
  though compute Spaces are not, and it makes the model independently usable.
  Create a model repo, upload `best.pt` and `calibration.json`, then use the
  `resolve/main/` URLs:
  `https://huggingface.co/<user>/onnm-densenet121/resolve/main/best.pt`
- **GitHub Releases.** Attach both files to a release and use the asset URLs.
  Simplest, no second account.

Do not commit the checkpoint through Git LFS unless you want every Community
Cloud rebuild to spend your 1 GB/month LFS bandwidth quota.

The URL must serve the **raw** file. `checkpoint_fetch` verifies the download
starts with torch's zip magic and refuses an HTML error page, because a CDN
404 usually returns status 200 with a web page — which would otherwise surface
much later as an unpicklable-file error pointing at entirely the wrong thing.

## Space/app metadata — get these right

The repo is public and the app is a medical-imaging demo, so the description is
a claim.

- **Description:** `Research demo — explainable bone-lesion triage on plain
  radiographs. Not a medical device.`
  Not "detects early stages of cancer": measured malignant recall is
  **0.633 [0.490–0.776]**, roughly one in three cancers missed, and Grad-CAM
  localisation has never been scored.
- **Licence:** `cc-by-nc-4.0`, not `mit`. The weights derive from BTXRD, which
  is CC BY-NC-ND 4.0. MIT would tell people they may use it commercially, which
  the training data's licence does not permit you to grant.

## Limits worth knowing

- Public apps only on the free tier.
- Apps sleep after ~7 days idle and wake on visit.
- 2.7 GB memory, 1 GB storage.
- Uploaded images go to Streamlit's servers. The disclaimer must say so, and
  must warn against identifiable patient data.
