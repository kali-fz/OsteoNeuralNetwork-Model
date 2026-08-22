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

### Generate two keys

```bash
python -c "import secrets; print('API_KEY  ', secrets.token_urlsafe(32)); print('ADMIN_KEY', secrets.token_urlsafe(32))"

npx wrangler secret put API_KEY        # paste the first
npx wrangler secret put ADMIN_KEY      # paste the second
```

Two keys, deliberately. `API_KEY` goes into the public-facing Hugging Face
Space and can only write ordinary rows. `ADMIN_KEY` unlocks `/admin/*` — the
review queue, approvals and export — and lives only on your machine. If the
Space's secret ever leaked, the leaker still could not approve their own
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

## Configure the clients

**Hugging Face Space** → Settings → Variables and secrets:

| name | value | kind |
|---|---|---|
| `ONNM_COMMUNITY_URL` | the workers.dev URL | variable |
| `ONNM_COMMUNITY_KEY` | `API_KEY` | **secret** |

Do **not** put `ADMIN_KEY` in the Space.

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

Review happens in the app's Admin tab (visible only when `ONNM_ADMIN_KEY` is
set). Approving requires you to state the true label — the database refuses an
approval without one.

---

## API

Every route requires `Authorization: Bearer <key>`. There are no public
endpoints: this stores health-adjacent data and must not be readable by
strangers.

| method | path | key | purpose |
|---|---|---|---|
| GET | `/health` | app | counts, storage used, limits |
| POST | `/users` | app | create account (hash only, never a password) |
| GET | `/users/by-email` | app | look up for login |
| POST | `/submissions` | app | record a prediction |
| GET | `/submissions?user_id=` | app | a user's history |
| POST | `/submissions/:id/feedback` | app | user flags a result wrong |
| GET | `/admin/pending` | admin | review queue, disputed first |
| POST | `/admin/review/:id` | admin | approve (with label) or reject |
| POST | `/admin/export` | admin | claim approved rows into a batch |

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

This is enforced three times: a schema trigger that aborts any approval without
a label, a check in the review endpoint, and the export query's `WHERE` clause.
The redundancy is deliberate. Every other bug in this system announces itself;
this one would quietly train the next generation on a hotdog labelled "normal
bone" and nothing downstream would notice.
