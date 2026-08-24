# Contributing to ONNM

Thanks for looking. This project handles medical images, so a few of the rules below are
firmer than they would be on an ordinary repository. Please read the two red lines first.

## Two red lines

**1. Never commit a radiograph, a derived image, or anything derived from BTXRD.**

The training data is [BTXRD](https://doi.org/10.1038/s41597-024-04104-3), licensed
**CC BY-NC-ND 4.0**. The **ND** term forbids redistributing derivatives, and the model
card draws the conclusion that matters here: **Grad-CAM overlays, case reports and
exported figures are derivative images and must stay local.** Do not put them in commits,
issues, pull requests, the README, or a paper without resolving the licence question
first. The **NC** term also means this project and its outputs are non-commercial.

**2. Never commit a secret, a database dump, or real patient data.**

`.env` and `.streamlit/secrets.toml` are gitignored. D1 dumps are gitignored **by
pattern** (`cloudflare/*.sql`) rather than by name, because `wrangler d1 export` writes
into `cloudflare/` by default and the next dump will be called something else. A dump
contains every user's email, their PBKDF2 hash, and every shared radiograph. If you need
a test image, use a synthetic one.

## Licence of your contribution

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE), the same licence as the project — inbound equals outbound.

Sign off each commit to certify you have the right to submit it, under the
[Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "Your-commit-subject"
```

If your contribution includes third-party code, say so in the pull request and name its
licence. Do not add a dependency whose licence is incompatible with Apache-2.0, and check
the licence of pretrained **weights**, not only of the library that ships them.

## Reporting security issues

Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell; use source .venv/bin/activate on POSIX
pip install -e .
```

Python 3.12. The GPU stack (PyTorch + ROCm) is only needed for training; a torch-free
subset of the tests runs without it.

## Before you open a pull request

```bash
ruff check .
pytest                        # full suite
pytest -m "not torch"         # torch-free subset, if you have no GPU stack
```

Both must be clean. CI runs ruff, the torch-free fast tests, and the full suite on CPU
torch; the environment and dataset gates are local-only because they need a GPU and the
dataset.

## House style, learned the hard way

- **Console output in `scripts/` must be ASCII.** Windows consoles are cp1252, so an em
  dash in a `print` — or in a docstring, which argparse prints for `--help` — raises
  `UnicodeEncodeError` on the target machine. This has bitten this project twice.
- **Write files as UTF-8 explicitly.** Python's `open(..., "w")` defaults to the system
  ANSI codepage on Windows; pass `encoding="utf-8"`.
- **Match the surrounding code.** Comment density here is high and deliberate: comments
  explain *why*, especially where a decision looks arbitrary. Keep that.
- **Make wrong states unrepresentable** where you reasonably can. The codebase prefers a
  required argument, a CHECK constraint or a database trigger over a convention someone
  has to remember.

## Changes that need extra care

Some areas carry consequences beyond the code. Flag these in the pull request:

| Area | Why |
|---|---|
| `cloudflare/migrations/` | Applied to a live production database. Migrations are additive and are never rerun. Never edit a migration that has already been applied — add a new one. |
| `cloudflare/src/worker.js` | **Apply the matching migration before deploying.** Deploying a Worker that reads columns D1 does not have breaks the live site. |
| Consent, sharing, or storage paths | Consent governs the pixels. A change here is a GRC decision, not only an engineering one. |
| `src/legal.py` | Published legal notices. Changes should be reviewed against `grc_compliance_prompt.md`. |
| Anything claiming model performance | Claims must be backed by a number in `reports/<run>/`, with its confidence interval. The Grad-CAM localisation figures are currently **at roughly chance**; do not claim the model localises lesions. |
| Model promotion | Promotion is a separate guarded step via `scripts/version_model.py`. A regressed run is recorded as `held` and the previous checkpoint keeps serving. Do not move `reports/PRODUCTION` by hand. |

## Commit messages

This repository uses short, hyphenated, subject-only commit messages, for example
`Fix-full-tree-Ruff-CI` or `Document-cloudflare-handoff`. Please match that.

## Governance and compliance

`grc_compliance_prompt.md` is the project's compliance register and lists the open gaps.
If your change closes one, say which number in the pull request. If it opens a new one,
say that too — an honest gap is worth more than a quiet one.
