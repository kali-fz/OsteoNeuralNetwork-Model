-- Migration 0007: record which Terms a person actually agreed to.
--
-- `users.tos_accepted_at` has existed since the first schema, and until now it
-- meant nothing. createUser binds `tos_accepted_at || created` and the OAuth
-- callback never sent the field, so every Google account carries
-- `tos_accepted_at == created_at`: a record that a row was inserted, not that a
-- human read anything. There was also no Terms page to read.
--
-- This adds the column that makes the timestamp meaningful. `tos_version` holds
-- the version string of the text that was on screen when the box was ticked, so
-- an acceptance can be traced to a specific wording rather than to a date that
-- might span several revisions.
--
-- EXISTING ROWS ARE LEFT NULL ON PURPOSE
-- --------------------------------------
-- Backfilling a version would be inventing consent. NULL reads as "this account
-- predates the gate and has agreed to nothing recorded", which is true, and the
-- application treats it as not-accepted: those accounts are asked to agree on
-- their next visit. Nobody loses their history or their submissions.
--
-- The compliance documents already assume this control exists. compliance/DPIA.md
-- names "the Terms" as the mitigation for risks R1, R2, R7 and R13, and
-- compliance/ROPA.md names Art 6(1)(b), performance of a contract, as the lawful
-- basis. Until this migration there was no contract to perform.
--
-- Additive and safe: one nullable column, no table rebuild, no data rewritten.
--
-- Apply with:
--   npx wrangler d1 execute onn-model --remote --file=./migrations/0007_terms_acceptance.sql

ALTER TABLE users ADD COLUMN tos_version TEXT;

UPDATE meta SET value = '7' WHERE key = 'schema_version';
