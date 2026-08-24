# Finish the ONNM community deployment

Give this file to the agent working on the Cloudflare account that owns the ONNM
database. The Streamlit redesign is already on `main`. The remaining production work is
to migrate the existing D1 database and deploy the matching Worker.

This one deployment fixes both unfinished home-page features:

- the globe reads approved contribution countries from `GET /globe`;
- the contributor section reads opted-in profiles from `GET /contributors`, while the
  profile toggle writes to `POST /users/profile`.

Do not create a replacement database. The existing database contains the community users,
submissions, reviews, and approved contributions that the globe needs.

## Production resources

| Resource | Expected value |
|---|---|
| Worker | `onnm-community` |
| Worker URL | `https://onnm-community.kali-fz.workers.dev` |
| D1 database name | `onn-model` |
| D1 database ID | `961f0440-7ff1-466e-88fe-0c2b30f3083b` |
| Repository schema target | `5` |
| Streamlit app | `https://osteoneuralnetwork-model-af5ynv9qxg7u8rc5epdprr.streamlit.app` |

The database name is `onn-model`, not `onnm-community`. The latter is the Worker name.

## 1. Start from the published code

Run these commands in PowerShell from the repository root:

```powershell
git status --short
git pull --ff-only origin main
cd cloudflare
```

If `git status --short` reports local changes, stop and preserve them before pulling.

## 2. Use a separate Wrangler login

Do not replace another person's default Wrangler login. Current Wrangler releases support
named authentication profiles, so create one for this project:

```powershell
npx wrangler auth create onnm
npx wrangler whoami --profile onnm
npx wrangler d1 list --profile onnm
```

Complete the browser sign-in with the Cloudflare account that owns the resources above.
Before continuing, confirm that `d1 list` contains database ID
`961f0440-7ff1-466e-88fe-0c2b30f3083b`. If Wrangler returns error 7404 or the database is
missing, the wrong Cloudflare account is active. Stop rather than creating a new database.

Do not paste an API token, Worker secret, Google client secret, or database export into
the repository or an agent conversation.

## 3. Inspect and back up D1

Read the current schema version:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT key, value FROM meta WHERE key = 'schema_version';"
```

Export a recovery copy outside the repository before changing the schema:

```powershell
$onnmBackup = Join-Path $env:TEMP "onn-model-before-v5.sql"
npx wrangler d1 export onn-model --remote --profile onnm --output=$onnmBackup
Write-Output $onnmBackup
```

Keep the exported file private because it contains account and contribution data.

## 4. Apply only the missing migrations

Migration files use `ALTER TABLE`, so do not rerun one that has already succeeded.

- If the schema version is `3`, apply 0004 and then 0005.
- If the schema version is `4`, apply only 0005.
- If the schema version is `5`, skip both migrations.
- If the value is missing or lower than `3`, stop and inspect the older migration history
  before making changes.

Commands for a version 3 database:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --file=./migrations/0004_geolocation.sql
npx wrangler d1 execute onn-model --remote --profile onnm --file=./migrations/0005_public_contributor_profiles.sql
```

For a version 4 database, run only the second command. Confirm the result before deploying
the Worker:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT key, value FROM meta WHERE key = 'schema_version';"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA table_info(users);"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA table_info(submissions);"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA quick_check;"
```

The version must be `5`. The `users` output must include `signup_country`, `display_name`,
`profile_picture_url`, and `public_contributor_profile`. The `submissions` output must
include `origin_country`, and `PRAGMA quick_check` must return `ok`.

## 5. Check secrets and deploy the Worker

```powershell
npx wrangler secret list --profile onnm
npx wrangler deploy --profile onnm
```

The secret list should contain `API_KEY` and `ADMIN_KEY`. Listing secrets does not reveal
their values. A normal deploy preserves existing secrets. If either name is missing, stop
and obtain the correct value from the owner; do not generate a replacement without also
updating the matching Streamlit configuration and local review setup.

Deploy the Worker only after D1 reaches schema version 5. The new Worker reads the new
columns, so deploying it first can turn a missing migration into production errors.

## 6. Verify the real application

Open the Streamlit app and complete this check with a Google account that has an approved
contribution:

1. Sign out and sign in again. This lets the Worker record the account's country from
   Cloudflare's country-level request metadata.
2. Return to the home page. The temporary community-service message should be gone, and
   the globe should show the account's country. The display threshold is currently one
   account.
3. Open the profile by clicking the Google name or photo in the header.
4. Enable `Show me in the public contributors section` if the account owner wants to be
   public. This setting is private by default and must not be enabled without consent.
5. Return home. The contributor card should show the Google name, photo, and approved
   contribution count. An account with no approved contribution must remain absent even
   when the toggle is enabled.
6. Open the scanner, upload a test image, and confirm that the themed uploader and the
   single account header still work.

Older approved submissions may not have their own country. The globe query falls back to
the user's country after the next sign-in, which is why the first verification step
matters.

## Failure guide

| Symptom | Meaning and next check |
|---|---|
| D1 error 7404 | The Wrangler profile is authenticated to the wrong Cloudflare account. |
| `no such column` from the Worker | D1 is not at schema version 5. Inspect `meta` and the table columns. |
| Globe and contributor endpoints return 404 | The updated Worker was not deployed to `onnm-community`. |
| Globe loads but shows no country | Sign in again, then confirm `users.signup_country` is populated and an approved contribution exists. |
| Contributor toggle is unavailable | Confirm `/contributors` is live and the Worker can read the version 5 columns. |
| Toggle works but no card appears | The account is private or has no approved contribution. This is expected privacy behaviour. |

Wrangler's profile, D1 execute, export, and deploy syntax is documented by Cloudflare:

- <https://developers.cloudflare.com/workers/wrangler/commands/general/>
- <https://developers.cloudflare.com/d1/wrangler-commands/>
- <https://developers.cloudflare.com/d1/best-practices/import-export-data/>
