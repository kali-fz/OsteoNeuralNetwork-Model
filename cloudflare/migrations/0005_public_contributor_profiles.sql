-- Migration 0005: opt-in public Google profiles for the contributor roll.
-- Existing accounts remain private until their owner enables the profile
-- toggle. Approved contribution consent does not imply identity publication.

ALTER TABLE users ADD COLUMN display_name TEXT;
ALTER TABLE users ADD COLUMN profile_picture_url TEXT;
ALTER TABLE users ADD COLUMN public_contributor_profile INTEGER NOT NULL DEFAULT 0
    CHECK (public_contributor_profile IN (0, 1));

UPDATE meta SET value = '5' WHERE key = 'schema_version';
