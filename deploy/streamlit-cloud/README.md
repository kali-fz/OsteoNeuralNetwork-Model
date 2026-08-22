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

# Google Sign-In — see the next section for where these come from.
[auth]
redirect_uri  = "https://<your-app>.streamlit.app/oauth2callback"
cookie_secret = "<64 random hex characters>"

[auth.google]
client_id           = "<id>.apps.googleusercontent.com"
client_secret       = "<secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

> **Order matters.** This is TOML: every bare key must come *above* the first
> `[section]` header, or it silently becomes a member of that section. Putting
> `ONNM_COMMUNITY_KEY` below `[auth]` makes it `auth.ONNM_COMMUNITY_KEY`, the
> app reads it as unset, and accounts quietly fall back to the container's
> throwaway SQLite instead of D1 — with no error anywhere to explain why.

**Never put `ONNM_ADMIN_KEY` here.** Review and export run from your machine, so
a leak of the app's key cannot approve its own training data.

5. Deploy. First build takes several minutes (torch is a large download).

## Google Sign-In

The app uses Google Sign-In rather than its own password forms. ONNM never
receives a password, so there is none to store, reset, or leak — the only things
kept are the email address and Google's `sub` (a stable account identifier).

Creating the OAuth client is the one part that cannot be scripted: it needs a
signed-in Google account and the consent screen is a human decision.

1. <https://console.cloud.google.com/> → create a project, e.g. **ONNM**.
2. **APIs & Services → OAuth consent screen**
   - User type **External**.
   - App name `OsteoNeuralNetwork-Model`, your email for both support and
     developer contact.
   - Scopes: leave the defaults (`openid`, `email`, `profile`). Do **not** add
     any others — anything beyond these is a "sensitive scope" and drags the
     project into Google's verification review for no benefit here.
   - While the app is in **Testing**, only accounts listed under **Test users**
     can sign in, up to 100. Add your own address and each tester's. This is the
     right mode for a research prototype: it keeps the app closed by default.
     Pressing **Publish app** opens it to any Google account; with only the
     three basic scopes that needs no verification, though testers will see an
     "unverified app" interstitial they must click through.
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**
   - Application type **Web application**.
   - **Authorised redirect URIs** — add both, exactly:
     - `https://<your-app>.streamlit.app/oauth2callback`
     - `http://localhost:8501/oauth2callback` (only if you want local sign-in)

     The path must be `/oauth2callback`; Streamlit serves the callback there and
     Google refuses any redirect that is not character-for-character on the list.
4. Copy the **client ID** and **client secret** into the secrets block above.
5. `cookie_secret` is any 64 random hex characters — it signs the session
   cookie, so treat it as a secret and do not reuse one from elsewhere.
   Generate one with:
   `python -c "import secrets; print(secrets.token_hex(32))"`

Changing these secrets restarts the app; sign-in works about a minute later.

### What this changes about accounts

- `users.password_hash` is `NULL` for a Google account and the schema *enforces*
  that it stays NULL, so an account can never be reachable both by password and
  by Google. That pairing is checked in the database rather than trusted to the
  Worker, because getting it wrong would be an authentication bypass.
- Identity is keyed on `sub`, not the email address, so a user who changes their
  Google address keeps the same account and the same scan history.
- An existing password account is *not* converted by signing in with the same
  address. Silently doing so would let anyone who can prove control of an email
  address take over that account.
- Uploads are unaffected: submissions still carry `user_id` and still land in
  D1 for retraining exactly as before.

### If sign-in fails

- **`redirect_uri_mismatch`** — the URI in Google's console differs from
  `[auth].redirect_uri`. They must match exactly, including `https` and no
  trailing slash.
- **"Access blocked: app not verified"** — the account is not on the test-user
  list and the app is unpublished. Add the address, or publish.
- **Sign-in works but the account does not appear in D1** — check the TOML
  ordering warning above; `ONNM_COMMUNITY_KEY` has probably been absorbed into
  the `[auth]` table.

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
