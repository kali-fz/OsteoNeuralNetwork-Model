# Repair the two UK account locations

Give this file to the agent using the Cloudflare account that owns ONNM. The original
community deployment has already been completed. This run is only for deploying the
browser-country fix and replacing the two incorrect US account locations with `GB`.

Do not create a new Worker, D1 database, Wrangler profile, API key, or Google OAuth
credential. Do not change every US row. Reset only the two accounts that their owners
confirm are in the UK.

## Production resources

| Resource | Value |
|---|---|
| Worker | `onnm-community` |
| Worker URL | `https://onnm-community.kali-fz.workers.dev` |
| D1 database | `onn-model` |
| D1 database ID | `961f0440-7ff1-466e-88fe-0c2b30f3083b` |
| Required schema version | `6` |
| Wrangler profile | `onnm` |
| Streamlit app | `https://osteoneuralnetwork-model-af5ynv9qxg7u8rc5epdprr.streamlit.app` |

## 1. Pull the current code and confirm Cloudflare access

From the repository root in PowerShell:

```powershell
git status --short
```

If this reports changes that have not been preserved, stop before pulling. Otherwise run:

```powershell
git pull --ff-only origin main
Set-Location cloudflare
npx wrangler whoami --profile onnm
npx wrangler d1 list --profile onnm
```

The D1 list must contain database ID `961f0440-7ff1-466e-88fe-0c2b30f3083b`. Error 7404
or a missing database means the wrong Cloudflare account is active.

## 2. Back up D1 and identify the two UK accounts

```powershell
$onnmBackup = Join-Path $env:TEMP ("onn-model-before-country-repair-{0}.sql" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
npx wrangler d1 export onn-model --remote --profile onnm --output=$onnmBackup
Write-Output $onnmBackup

npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT key, value FROM meta WHERE key = 'schema_version';"
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT user_id, email, display_name, signup_country, country_captured_at FROM users ORDER BY created_at;"
```

Keep the backup private. It contains account and contribution data.

From the second query, identify the two accounts whose owners confirm they are in the UK.
Copy their exact `user_id` values. In the commands below, replace `<UK_USER_ID_1>` and
`<UK_USER_ID_2>` with those UUIDs. If the rows are ambiguous, stop and ask the owners;
do not guess from a name or alter all accounts currently marked `US`.

## 3. Ensure migration 0006 and the current Worker are deployed

If the schema query returned `5`, run:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --file=./migrations/0006_browser_country_capture.sql
```

If it returned `6`, do not rerun the migration. For any other version, stop and inspect
the migration history before changing data.

Verify version 6 and the capture columns, then deploy the current Worker:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT key, value FROM meta WHERE key = 'schema_version';"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA table_info(users);"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA table_info(location_capture_tokens);"
npx wrangler secret list --profile onnm
npx wrangler deploy --profile onnm
```

The schema version must be `6`. `users` must contain `country_captured_at`, and
`location_capture_tokens` must contain `token_hash`, `user_id`, `expires_at`, `used_at`,
and `used_nonce`. The secret list must still contain `API_KEY` and `ADMIN_KEY`; never
print or replace their values.

## 4. Clear only the two incorrect derived locations

These fields contain country-level metadata, not uploaded images. The approved scans and
review decisions remain untouched. Ask both owners to sign out and close the Streamlit app
before this step. Run each command after replacing both placeholders:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "DELETE FROM location_capture_tokens WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>');"
npx wrangler d1 execute onn-model --remote --profile onnm --command "UPDATE submissions SET origin_country = NULL WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>');"
npx wrangler d1 execute onn-model --remote --profile onnm --command "UPDATE users SET signup_country = NULL, country_captured_at = NULL WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>');"
```

Confirm that exactly two user rows are now waiting for capture:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT user_id, display_name, signup_country, country_captured_at FROM users WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>');"
```

Both rows must show `NULL` for `signup_country` and `country_captured_at`. If they do not,
stop before asking anyone to sign in.

## 5. Capture both accounts from the UK

Complete these steps separately for each account owner:

1. Use a normal UK internet connection. Disable any VPN, browser VPN, proxy, or remote
   browser session that exits through another country.
2. Open the Streamlit app and sign out completely.
3. Sign back in with the correct Google account.
4. Stay on the home page for several seconds, then refresh it once.
5. Sign out before repeating the process with the second Google account.

The signed-in browser sends a short-lived one-use token directly to Cloudflare. Cloudflare
records only its two-letter country code. No IP address or precise location is stored.

## 6. Prove the repair in D1

After both owners have signed in, run:

```powershell
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT user_id, display_name, signup_country, country_captured_at FROM users WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>');"
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT user_id, origin_country, COUNT(*) AS submissions FROM submissions WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>') GROUP BY user_id, origin_country;"
npx wrangler d1 execute onn-model --remote --profile onnm --command "SELECT user_id, expires_at, used_at FROM location_capture_tokens WHERE user_id IN ('<UK_USER_ID_1>', '<UK_USER_ID_2>') ORDER BY expires_at DESC;"
npx wrangler d1 execute onn-model --remote --profile onnm --command "PRAGMA quick_check;"
```

Required result:

- both user rows have `signup_country = GB` and a non-null `country_captured_at`;
- both users' historical submission rows have `origin_country = GB`;
- each account has a recent capture token with a non-null `used_at`;
- `PRAGMA quick_check` returns `ok`.

Finally, refresh the Streamlit home page. The globe may retain an old aggregate for up to
five minutes. After that, it must show one combined UK marker and fill the United Kingdom.
The United States must disappear unless a different, genuinely US account is recorded.

## If it still reports the United States

- No capture-token row: confirm the Streamlit app is running current `main` and redeploy
  `onnm-community` from the current `cloudflare` directory.
- Token exists but `used_at` is null: inspect the browser console for a failed
  `/location/capture` request and confirm the current Worker was deployed after migration
  0006.
- Token was used but the result is `US`: Cloudflare saw a US network exit. Disable the VPN,
  proxy, Brave VPN, or remote browser; repeat steps 4–6 for only that account.
- D1 says `GB` but the globe says `US`: wait five minutes for the Streamlit aggregate cache,
  then restart or refresh the app.

Do not hardcode the globe to the UK. The D1 verification above is the source of truth and
keeps future contributors in their real countries.

## Recovery

If the wrong account was reset, do not import the full backup over a live database. No
uploads or review decisions were changed; ask that account owner to sign in once from
their normal connection so Cloudflare can restore the correct country. Keep the private
backup until both intended accounts have passed the D1 checks, then follow the owner's
normal secure-backup retention process.
