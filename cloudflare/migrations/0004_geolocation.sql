-- Migration 0004: coarse, country-level origin for the contribution globe.
--
-- WHAT CHANGES
-- ------------
-- 1. `users` gains `signup_country`  — where an account was created.
-- 2. `submissions` gains `origin_country` — where a submission was sent from.
-- Both are nullable two-letter ISO 3166-1 alpha-2 codes, and nothing else.
--
-- WHY COUNTRY AND NOTHING FINER
-- -----------------------------
-- The landing page wants to show that this project has reach: dots on a globe
-- for where users signed up and where approved contributions came from. The
-- obvious implementation -- store a latitude and longitude per row and simply
-- decline to print the owner's name next to it -- is not anonymous, and this
-- is medical data.
--
-- A precise coordinate, a timestamp and a malignant verdict are jointly
-- identifying even with every name stripped. In a small town there may be one
-- hospital, one radiology department, and one person who uploaded a film that
-- afternoon. Coordinates would let anyone holding a database dump work that
-- out; a country code does not.
--
-- So the finest resolution this schema is capable of storing is the country.
-- That is a property of the table, not a promise made by the code that writes
-- to it, and it is the reason the CHECK constraints below are worth their
-- awkwardness: a future endpoint cannot quietly start recording a city, a
-- postcode or an IP address without a migration that says so out loud.
--
-- WHERE THE VALUE COMES FROM
-- --------------------------
-- Cloudflare resolves the country at the edge and hands it to the Worker as
-- `request.cf.country`. It is derived from the connecting IP but arrives
-- already reduced, so the Worker never has to see, log or store an address to
-- obtain it. The browser Geolocation API is deliberately not used: it would
-- prompt the visitor, it would return GPS-grade precision this schema must
-- not hold, and asking for a location on a cancer-screening page is a poor
-- trade for a decorative globe.
--
-- Cloudflare also emits 'T1' for Tor exit nodes and 'XX' where it cannot
-- determine a country. Both satisfy the CHECK and are stored as they arrive;
-- they simply have no centroid, so the aggregation step drops them. Recording
-- them honestly is better than coercing them to NULL and losing the
-- distinction between "unknown" and "never asked".
--
-- WHY THIS IS AN ALTER AND NOT A REBUILD
-- --------------------------------------
-- Unlike 0003, nothing here changes an existing constraint. Both columns are
-- new and nullable, so `ALTER TABLE ... ADD COLUMN` is sufficient, including
-- the CHECK -- SQLite accepts a CHECK on an added column and enforces it on
-- subsequent writes. A rebuild would mean copying every stored radiograph
-- through a temporary table for no benefit.
--
-- EXISTING ROWS
-- -------------
-- Every row that predates this migration keeps NULL, and NULL is never
-- backfilled or guessed. There is no record of where those users came from,
-- and inventing one would put a dot on a map that corresponds to nothing.
-- Every query added for the globe therefore has to tolerate NULL, and the
-- tests assert that it does.
--
-- Apply with:
--   npx wrangler d1 execute onnm-community --remote --file=./migrations/0004_geolocation.sql

-- ---------------------------------------------------------------------------
-- users: where the account was created.
-- ---------------------------------------------------------------------------
ALTER TABLE users ADD COLUMN signup_country TEXT
    CHECK (signup_country IS NULL
           OR (length(signup_country) = 2 AND signup_country = upper(signup_country)));

-- ---------------------------------------------------------------------------
-- submissions: where the submission was sent from.
--
-- Stored per submission rather than read from the user's row, because the two
-- genuinely differ -- someone signs up at home and contributes from a hospital
-- network, or travels. Denormalising it here also means the globe's
-- contributor layer never has to join back to `users`, so the aggregation
-- query cannot accidentally select an account column.
-- ---------------------------------------------------------------------------
ALTER TABLE submissions ADD COLUMN origin_country TEXT
    CHECK (origin_country IS NULL
           OR (length(origin_country) = 2 AND origin_country = upper(origin_country)));

-- ---------------------------------------------------------------------------
-- Indexes for the two globe layers.
--
-- Both are partial, because the rows that matter are a subset: signups that
-- recorded a country, and approved submissions. The globe is read on every
-- landing-page render (behind a cache), so this is the one new read path that
-- could plausibly matter to the free-tier read quota.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_users_country
    ON users(signup_country) WHERE signup_country IS NOT NULL;

-- (origin_country, user_id) rather than origin_country alone: the contributor
-- layer counts DISTINCT users per country, not submissions, so that one
-- enthusiastic uploader cannot inflate their own country's dot.
CREATE INDEX IF NOT EXISTS idx_submissions_geo
    ON submissions(origin_country, user_id)
    WHERE review_status = 'approved' AND origin_country IS NOT NULL;

UPDATE meta SET value = '4' WHERE key = 'schema_version';
