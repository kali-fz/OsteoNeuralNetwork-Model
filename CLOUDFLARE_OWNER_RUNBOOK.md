# ONNM production deployment — Cloudflare account owner runbook

This runbook is only for the owner of the Cloudflare and Google OAuth accounts.
The application code is already on GitHub `main`, and its GitHub Actions checks pass.
The live Worker is still serving the previous version because GitHub is not currently
configured to deploy it automatically.

**Repository:** `https://github.com/kali-fz/OsteoNeuralNetwork-Model`

**Worker:** `onnm`

**Live URL:** `https://onnm.kali-fz.workers.dev`

## 1. Use a clean checkout of `main`

Do not deploy from an old or dirty checkout.

```bash
git clone https://github.com/kali-fz/OsteoNeuralNetwork-Model.git
cd OsteoNeuralNetwork-Model
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
```

The expected application commit at the time this runbook was written is `8ba4c0b` or a
newer commit containing it.

## 2. Rotate the Google OAuth client secret

The previous Google client secret was disclosed in a chat transcript and must be replaced.

1. Open Google Cloud Console.
2. Go to **APIs & Services → Credentials**.
3. Open the OAuth 2.0 client used by ONNM.
4. Create/reset the client secret.
5. Confirm this authorised redirect URI is still present:

   ```text
   https://onnm.kali-fz.workers.dev/api/auth/google/callback
   ```

6. Keep the replacement secret private. Do not paste it into this repository, `.env`, a
   commit, an issue, or a chat.

Log in to the correct Cloudflare account and install the replacement secret interactively:

```bash
npm ci
npx wrangler login
npx wrangler whoami
npx wrangler secret put GOOGLE_CLIENT_SECRET
```

When Wrangler prompts for the value, paste the new Google secret there. After it is stored
successfully, revoke/delete the old secret in Google Cloud Console.

The existing Worker secrets `API_KEY`, `SESSION_SECRET`, and `INFERENCE_KEY` should remain
unchanged. Do not print, copy, or replace them during this deployment.

## 3. Validate without deploying

```bash
npm run test:auth
npm run build
npm run check -- --containers-rollout=none
```

All three commands must succeed. The final command is deliberately a dry-run; it does not
change production.

## 4. Deploy the restored website and Worker code

The inference image and model have not changed in this restoration, so deploy only the
Worker code and frontend assets:

```bash
npx wrangler deploy --containers-rollout=none
```

Do **not** run `npm run deploy` for this release. That command stages the model and rebuilds
the large inference image unnecessarily.

Wait for Wrangler to report a successful deployment and the `onnm.kali-fz.workers.dev`
address.

## 5. Required live checks

Open a private/incognito browser window so cached frontend files do not hide the release.

1. Open `https://onnm.kali-fz.workers.dev`.
2. Confirm the homepage headline is:

   ```text
   Test the current model and help us train the next one.
   ```

3. Confirm the mission card and globe appear side by side on desktop.
4. Confirm the page scrolls through totals, scan benefits, contributors, and the footer.
5. Test Google sign-in with an existing ONNM account.
6. Sign out, then test with a genuinely new Google account.
7. Confirm both users reach the signed-in page without `account_failed`.
8. Run one controlled radiograph scan and confirm its result/history state is truthful.

Do not invite a wider test group if existing login, new-account login, scan persistence, or
consent behaviour fails.

## 6. Billing safety

In the Cloudflare dashboard, create a billing notification around **$8**. The application
already limits the inference container to one instance, sleeps it after 90 seconds, and has
a five-hour monthly runtime breaker. Do not add a keep-warm cron trigger.

## 7. If the deployment is unhealthy

Keep the Streamlit deployment available as the rollback path.

In Cloudflare, open **Workers & Pages → onnm → Deployments**, select the previous healthy
deployment, and use the dashboard rollback control. Do not change the D1 schema or delete
database rows as part of a frontend rollback.

## 8. Optional automatic deployment after the manual release works

Only configure this after the manual deployment and live checks pass.

1. Open **Cloudflare → Workers & Pages → onnm → Settings → Builds**.
2. Connect the GitHub repository `kali-fz/OsteoNeuralNetwork-Model`.
3. Set the production branch to `main`.
4. Use this build command:

   ```bash
   npm ci && npm run build
   ```

5. Use this deploy command while the container image remains unchanged:

   ```bash
   npx wrangler deploy --containers-rollout=none
   ```

Future changes to `inference/Dockerfile`, the model weights, calibration, or inference
service require the separate full container release procedure. Do not silently deploy
those with this frontend-only command.

## 9. Send back this evidence

- Wrangler deployment success output, with secrets removed.
- Confirmation that the new homepage headline is live.
- Existing-account Google login result.
- New-account Google login result.
- Controlled scan result and whether its history row appeared.
- Confirmation that the billing notification is enabled.
