#!/usr/bin/env bash
# One-command deploy of the ONNM community API.
#
#   cd cloudflare && bash deploy.sh
#
# Reads CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID from ../.env (which is
# gitignored). Idempotent: safe to re-run. It will
#
#   1. verify the token before touching anything
#   2. create the D1 database, or reuse CLOUDFLARE_D1_DATABASE_ID if set
#   3. write the id into wrangler.toml
#   4. apply schema.sql
#   5. generate and store API_KEY / ADMIN_KEY if they are not already set
#   6. deploy the Worker
#   7. call /health to prove it works
#
# It prints the two lines you must paste into Streamlit secrets, and it never
# prints ADMIN_KEY to a shared screen without saying so.

set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE="../.env"
KEYS_FILE="../.cloudflare-keys.txt"   # gitignored; your copy of the generated keys

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. credentials -------------------------------------------------------
[ -f "$ENV_FILE" ] || fail "$ENV_FILE not found"
set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a

[ -n "${CLOUDFLARE_API_TOKEN:-}" ] || fail "CLOUDFLARE_API_TOKEN is not set in $ENV_FILE"
[ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ] || fail "CLOUDFLARE_ACCOUNT_ID is not set in $ENV_FILE"
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

log "Verifying the API token"
VERIFY=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  https://api.cloudflare.com/client/v4/user/tokens/verify)
if ! echo "$VERIFY" | grep -q '"success":true'; then
  echo "$VERIFY"
  fail "the API token is not valid.

Create one at https://dash.cloudflare.com/profile/api-tokens
  -> Create Token -> 'Edit Cloudflare Workers' template
  -> confirm the permissions include BOTH:
       Account | Workers Scripts | Edit
       Account | D1             | Edit
  -> paste it into $ENV_FILE as CLOUDFLARE_API_TOKEN=\"...\"

A Cloudflare API token is normally 40 characters."
fi
echo "token OK"

# --- 2. database ----------------------------------------------------------
DB_NAME="onnm-community"
DB_ID="${CLOUDFLARE_D1_DATABASE_ID:-}"

if [ -z "$DB_ID" ]; then
  log "Creating D1 database '$DB_NAME'"
  CREATE_OUT=$(npx --yes wrangler d1 create "$DB_NAME" 2>&1 || true)
  echo "$CREATE_OUT"
  DB_ID=$(echo "$CREATE_OUT" | grep -oE '[0-9a-f-]{36}' | head -1)
fi

if [ -z "$DB_ID" ]; then
  log "Looking up '$DB_NAME' in the account"
  DB_ID=$(npx --yes wrangler d1 list --json 2>/dev/null \
    | python -c "
import json,sys
try:
    for d in json.load(sys.stdin):
        if d.get('name') == '$DB_NAME':
            print(d.get('uuid') or d.get('id') or ''); break
except Exception:
    pass" || true)
fi

[ -n "$DB_ID" ] || fail "could not determine the D1 database id. Create it with:
  npx wrangler d1 create $DB_NAME
then set CLOUDFLARE_D1_DATABASE_ID in $ENV_FILE"
echo "database id: $DB_ID"

# --- 3. wire it into wrangler.toml ---------------------------------------
log "Writing the database id into wrangler.toml"
python - "$DB_ID" <<'PY'
import re, sys, pathlib
db_id = sys.argv[1]
p = pathlib.Path("wrangler.toml")
t = p.read_text(encoding="utf-8")
new = re.sub(r'database_id\s*=\s*"[^"]*"', f'database_id = "{db_id}"', t, count=1)
p.write_text(new, encoding="utf-8")
print("wrangler.toml updated" if new != t else "wrangler.toml already correct")
PY

# --- 4. schema ------------------------------------------------------------
log "Applying schema.sql (idempotent: every statement is CREATE ... IF NOT EXISTS)"
npx --yes wrangler d1 execute "$DB_NAME" --remote --file=./schema.sql --yes

# --- 5. secrets -----------------------------------------------------------
EXISTING=$(npx --yes wrangler secret list 2>/dev/null || echo "[]")
needs_secret() { ! echo "$EXISTING" | grep -q "\"$1\""; }

if needs_secret API_KEY || needs_secret ADMIN_KEY; then
  log "Generating API_KEY and ADMIN_KEY"
  API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  ADMIN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  printf '%s' "$API_KEY"   | npx --yes wrangler secret put API_KEY
  printf '%s' "$ADMIN_KEY" | npx --yes wrangler secret put ADMIN_KEY
  {
    echo "# ONNM Cloudflare keys, generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# API_KEY   -> Streamlit secret ONNM_COMMUNITY_KEY (public app)"
    echo "# ADMIN_KEY -> your machine only; never put it in Streamlit"
    echo "ONNM_COMMUNITY_KEY=$API_KEY"
    echo "ONNM_ADMIN_KEY=$ADMIN_KEY"
  } > "$KEYS_FILE"
  echo "keys written to $KEYS_FILE (gitignored)"
else
  echo "API_KEY and ADMIN_KEY already set; leaving them alone"
fi

# --- 6. deploy ------------------------------------------------------------
log "Deploying the Worker"
DEPLOY_OUT=$(npx --yes wrangler deploy 2>&1)
echo "$DEPLOY_OUT"
URL=$(echo "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9.-]+\.workers\.dev' | head -1)
[ -n "$URL" ] || fail "deployed, but could not parse the workers.dev URL from the output"

# --- 7. prove it works ----------------------------------------------------
log "Checking $URL/health"
if [ -f "$KEYS_FILE" ]; then
  CHECK_KEY=$(grep '^ONNM_COMMUNITY_KEY=' "$KEYS_FILE" | cut -d= -f2-)
  curl -s -H "Authorization: Bearer $CHECK_KEY" "$URL/health" | python -m json.tool || true
  echo
  echo "Unauthenticated request should be refused:"
  curl -s -o /dev/null -w "  no key -> HTTP %{http_code} (expect 401)\n" "$URL/health"
fi

log "Done. Add these to Streamlit secrets"
cat <<EOF

ONNM_COMMUNITY_URL = "$URL"
ONNM_COMMUNITY_KEY = "<the ONNM_COMMUNITY_KEY line in $KEYS_FILE>"

Do NOT add ONNM_ADMIN_KEY to Streamlit. Keep it on this machine only:
  export ONNM_COMMUNITY_URL=$URL
  export ONNM_ADMIN_KEY=<the ONNM_ADMIN_KEY line in $KEYS_FILE>
EOF
