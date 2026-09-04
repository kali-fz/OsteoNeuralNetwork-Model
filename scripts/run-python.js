#!/usr/bin/env node
/**
 * Run a project Python script with the project's own interpreter.
 *
 * WHY THIS EXISTS
 * ---------------
 * `npm run deploy` chains stage-model -> build -> wrangler deploy, and the
 * stage-model step used to invoke bare `python`. On a machine where the project
 * dependencies live in `.venv` -- which is every machine this project documents
 * -- that resolves to some *other* interpreter, which does not have torch, and
 * the deploy fails at its first step with an ImportError that says nothing about
 * virtual environments.
 *
 * Worse than failing, it fails AFTER the developer has decided to deploy, so the
 * obvious next move is to work around it by hand and skip the staging step. That
 * step is the one that proves the weights on disk are the ones the ledger says
 * are serving, so skipping it is exactly the thing `stage_inference_model.py`
 * exists to prevent.
 *
 * Naming the venv path directly in package.json is not portable: it is
 * `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` elsewhere, and
 * npm scripts have no conditional. Hence this shim.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// package.json declares "type": "module", so there is no __dirname here.
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const candidates =
  process.platform === "win32"
    ? [join(repoRoot, ".venv", "Scripts", "python.exe")]
    : [join(repoRoot, ".venv", "bin", "python3"), join(repoRoot, ".venv", "bin", "python")];

// Falling back to PATH rather than failing: a CI image or a conda environment
// may legitimately have no .venv while still having every dependency installed.
const interpreter =
  candidates.find(existsSync) || (process.platform === "win32" ? "python" : "python3");

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("usage: node scripts/run-python.js <script.py> [args...]");
  process.exit(2);
}

const result = spawnSync(interpreter, args, { cwd: repoRoot, stdio: "inherit" });

if (result.error) {
  console.error(`could not run ${interpreter}: ${result.error.message}`);
  process.exit(1);
}
// A signal-terminated child has a null status; report it as a failure rather
// than as the success that `process.exit(null)` would produce.
process.exit(result.status === null ? 1 : result.status);
