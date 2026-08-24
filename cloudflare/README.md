# ONNM community API — deploy guide

Cloudflare Worker + D1 storing accounts, predictions, user feedback and the
human-reviewed training batches. **Free tier only, no payment method required.**

Everything below is a command you run — I cannot log into your Cloudflare
account, so the deploy step is yours.

---

## Why there is no R2

Object storage would be the obvious home for images, but enabling R2 generally
requires a payment method on file, and a card on file is the only way this
project can be billed. Instead each shared image is stored in D1 as a base64
PNG of the **256 px preprocessed array** — about 30 KB, which is exactly what
retraining consumes anyway.

At the 200 MB cap enforced in `src/worker.js` that is roughly 6,600 images,
using ~4% of D1's 5 GB free allowance. If you outgrow it, `src/community.py`
puts image storage behind one function pair (`encode_image_for_sharing` /
`decode_shared_image`), so moving to R2 later is a contained change.

**With no payment method on the account, Cloudflare cannot bill you.** Overage
returns errors instead. The caps in the Worker exist so you hit an
intelligible refusal from our own code long before any platform limit.

---

## One-time setup

```bash
cd cloudflare
npx wrangler login                      # opens a browser
npx wrangler d1 create onnm-community
```

`d1 create` prints a `database_id`. Paste it into `wrangler.toml`, replacing
`REPLACE_WITH_DATABASE_ID`.

Then create the tables:

```bash
npx wrangler d1 execute onnm-community --remote --file=./schema.sql
```

An existing database is brought forward with the migrations instead, in order:

```bash
npx wrangler d1 execute onnm-community --remote --file=./migrations/0002_google_oauth.sql
npx wrangler d1 execute onnm-community --remote --file=./migrations/0003_triage_buckets.sql
npx wrangler d1 execute onn-model --remote --file=./migrations/0004_geolocation.sql
npx wrangler d1 execute onn-model --remote --file=./migrations/0005_public_contributor_profiles.sql
npx wrangler d1 execute onn-model --remote --file=./migrations/0006_browser_country_capture.sql
```

`0003` rebuilds `users` and `submissions` — SQLite cannot amend a CHECK
constraint in place — and preserves every row. It is not idempotent: running it
twice fails on `CREATE TABLE users_new`, harmlessly, before touching anything.
`SELECT value FROM meta WHERE key = 'schema_version'` should read `6` after all migrations.

### Generate two keys

```bash
python -c "import secrets; print('API_KEY  ', secrets.token_urlsafe(32)); print('ADMIN_KEY', secrets.token_urlsafe(32))"

npx wrangler secret put API_KEY        # paste the first
npx wrangler secret put ADMIN_KEY      # paste the second
```

Two keys, deliberately. `API_KEY` goes into the public-facing Streamlit
Community Cloud app and can only write ordinary rows. `ADMIN_KEY` unlocks `/admin/*` — the
review queue, approvals and export — and lives only on your machine. If the
app's secret ever leaked, the leaker still could not approve their own
training data.

### Deploy

```bash
npx wrangler deploy
```

It prints a URL like `https://onnm-community.<subdomain>.workers.dev`.
Check it:

```bash
curl -H "Authorization: Bearer $API_KEY" https://onnm-community.<subdomain>.workers.dev/health
```

---

## Who can review

One account: `kzfhero@gmail.com`, user id
`c2c5a209-4aaa-4eb9-b112-b2929b6dbe12`. That id is hardcoded in three places,
and all three have to agree:

| where | what it enforces |
|---|---|
| `schema.sql` | a CHECK constraint — no other row can ever hold `is_admin = 1` |
| `src/worker.js` | `/admin/*` requires an `x-onnm-admin-user` header naming that id |
| `src/community.py` | `is_admin()`, which gates the review UI |

It is a constant rather than a setting on purpose. Review is the only path by
which any data reaches training, so "who may review" is a property of this
deployment, not a preference: an environment variable can be mistyped and a
database flag can be granted by a future endpoint, but moving this needs a code
change and a migration.

Note what the two admin checks are each for. `ADMIN_KEY` authenticates the
**caller** — it says the request comes from trusted software. The header
identifies the **account**. Anyone holding the key could of course also send
the header; the pairing is not defence against a stolen key. It is what stops
the likelier failure, where an app process that legitimately holds the key
serves the review queue to whichever ordinary user happens to be signed in.

If the review tab never unlocks for you, the account id in D1 has diverged from
the one hardcoded here — Google accounts get a fresh UUID when first created:

```bash
npx wrangler d1 execute onnm-community --remote \
  --command "SELECT user_id, email, is_admin FROM users WHERE email = 'kzfhero@gmail.com'"
```

If the id differs, change the three constants to match it rather than editing
the row: the id in the database is the one Google Sign-In will keep producing.

---

## The three buckets

Every shared submission is triaged into one of three queues on arrival, and
re-triaged if the user disputes the result. The review question is different in
each, which is why the UI is three tabs and not one list.

| bucket | what it holds | what it retrains |
|---|---|---|
| `valid_bone` | the OOD gate accepted it; the classifier ran | the lesion classifier |
| `misc` | the gate rejected it — a hotdog, a screenshot, a photo of a wall | the OOD detector, as negatives |
| `contradiction` | the system disagreed with itself | both, depending on the label |

A row is a contradiction when the gate rejected an image the user insists is a
radiograph (a false rejection nobody else can witness, since inference never
ran), or when the gate accepted one the user says is not a radiograph at all
while the classifier confidently diagnosed it. A user disputing the *grade* —
"you said malignant, I think benign" — is not a contradiction; that is a
labelling disagreement for review, and it stays in `valid_bone`.

The automatic bucket is written to `triage_bucket`. What you confirm during
review is written to `admin_bucket`, and only that one is exported. They are
separate columns for the same reason `admin_label` is separate from
`user_suggested_label`: the automatic value is the guess of the very system
being retrained, so it cannot serve as its own ground truth. That is also why
the review form preselects nothing.

Uploads the gate rejects are now **recorded** rather than discarded, which is
what makes the `misc` bucket non-empty. `onnm.ood` stage 1 is four hand-tuned
thresholds with no learned component and no negative examples; every hotdog
someone uploads is one, and this is the only way to collect them.

---

## Configure the clients

**Streamlit Community Cloud** → your app → Settings → Secrets (TOML):

```toml
ONNM_COMMUNITY_URL = "https://onnm-community.<subdomain>.workers.dev"
ONNM_COMMUNITY_KEY = "the API_KEY generated above"
```

Do **not** put `ADMIN_KEY` there.

**Your machine**, for review and export:

```bash
export ONNM_COMMUNITY_URL=https://onnm-community.<subdomain>.workers.dev
export ONNM_ADMIN_KEY=...
```

---

## The loop

```bash
python scripts/export_batch.py --dry-run              # what is ready
python scripts/export_batch.py --note "generation 2"  # claim and write it
```

Review happens in the app's Community expander in the sidebar, which opens
only for the admin account and only where `ONNM_ADMIN_KEY` is set — so run the
app from a local checkout to review. Approving requires you to state both the
bucket and the true label, and the two must agree: the database refuses a
`misc` row carrying a diagnosis, and a bone row labelled `misc`.

The export writes two manifests, because a batch retrains two different things:

```
data/community/<batch-id>/
    images/…                manifest.csv        -> paths.controls_manifest
    ood_negatives/…         ood_manifest.csv    -> OOD hardening
    batch.json              counts and paths, for the notebook
```

They are separate files rather than one file with a `bucket` column because
`build_records` would read a combined manifest and, finding a label column it
recognises, merge every row into the three-class training set.

---

## API

Every data route requires `Authorization: Bearer <key>`. The browser-facing
capture route accepts only a short-lived, one-use token minted by the app; it
does not expose stored data.

| method | path | key | purpose |
|---|---|---|---|
| GET | `/health` | app | counts, storage used, limits |
| GET | `/globe` | app | country-level signup and approved-contributor counts |
| GET | `/contributors` | app | opted-in names, photos, and approved contribution counts |
| POST | `/location/token` | app | mint a short-lived browser capture token |
| POST | `/location/capture` | one-use token | record Cloudflare's country code for that account |
| POST | `/users` | app | create account (hash only, never a password) |
| GET | `/users/by-email` | app | look up for login |
| GET | `/users/by-subject` | app | look up a Google account by stable subject |
| POST | `/users/profile` | app | refresh Google fields and change contributor-profile opt-in |
| POST | `/submissions` | app | record a prediction |
| GET | `/submissions?user_id=` | app | a user's history |
| POST | `/submissions/:id/feedback` | app | user flags a result wrong |
| GET | `/admin/pending?bucket=` | admin | review queue for one bucket, disputed first |
| POST | `/admin/review/:id` | admin | approve (with bucket **and** label) or reject |
| POST | `/admin/export` | admin | claim approved rows into a batch |

`/admin/*` additionally requires `x-onnm-admin-user:
c2c5a209-4aaa-4eb9-b112-b2929b6dbe12`. `src/community.py` sends it on every
admin call; a `curl` test needs it added by hand or it returns 403.

### Limits enforced in code

| limit | value |
|---|---|
| request body | 1.5 MB |
| single image | 600 KB base64 |
| total image storage | 200 MB, then writes are refused |
| submissions per user per day | 50 |
| accounts | 500 |

---

## The invariant

A user saying "this was wrong" is a **signal**, never a **label**.

`user_says_wrong` and `user_suggested_label` are untrusted — anyone with an
account can write them. `admin_label` is trusted and only a human review can
set it. Only `admin_label` reaches training.

This is enforced four times: a schema trigger that aborts any approval without
a label and bucket, a second trigger that refuses a bucket/label pair which
contradicts itself, a check in the review endpoint, and the export query's
`WHERE` clause — with a fifth check in `scripts/export_batch.py`, the only one
that runs on your machine rather than at the edge.

The redundancy is deliberate. Every other bug in this system announces itself;
this one would quietly train the next generation on a hotdog labelled "normal
bone" and nothing downstream would notice. `misc` being a real, exportable
label makes that failure *easier* to reach by hand — hence the second trigger,
which exists solely to make "hotdog, benign" unsayable.
