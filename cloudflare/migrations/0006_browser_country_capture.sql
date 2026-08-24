-- Migration 0006: capture country at the visitor's Cloudflare edge.
--
-- Streamlit calls the community Worker from its Python server. The original
-- geolocation path therefore recorded that server's country, not the signed-in
-- visitor's country. A browser receives a short-lived, one-use opaque token and
-- POSTs it directly to the Worker. Cloudflare supplies only an ISO country code;
-- no IP address, coordinate, city, or postcode is stored.
--
-- Existing country values are deliberately left in place until each account
-- next signs in. Its first browser capture replaces the old value and repairs
-- that account's historical submission countries. country_captured_at then
-- prevents later travel or repeat sign-ins from moving the account.
--
-- Apply with:
--   npx wrangler d1 execute onn-model --remote --profile onnm --file=./migrations/0006_browser_country_capture.sql

ALTER TABLE users ADD COLUMN country_captured_at TEXT;

CREATE TABLE IF NOT EXISTS location_capture_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at    TEXT,
    used_nonce TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_location_tokens_expiry
    ON location_capture_tokens(expires_at);

UPDATE meta SET value = '6' WHERE key = 'schema_version';
