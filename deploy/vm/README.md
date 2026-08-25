# Self-hosted ONNM deployment

Runbook for hosting the app on a VM instead of Streamlit Community Cloud.

## Why this exists

Streamlit Community Cloud fails on four counts, all structural to that platform
rather than to the app:

| symptom | cause |
|---|---|
| Third-party branding (profile avatar, "Hosted with Streamlit", GitHub/Fork) | Injected by Community Cloud. Streamlit staff: they **"cannot be hidden when apps are deployed from a public github repository"**. |
| Sleeps after 12 h idle | Platform policy. An HTTP `GET` returns 200 *without* waking it, so cron pings do not help. |
| Extremely laggy | Documented floor of **0.078 CPU cores** (max 2). A DenseNet forward plus a Grad-CAM backward at 256 px on a thirteenth of a core is seconds. |
| Random shutdowns | 2.7 GB memory ceiling. torch + MONAI + OpenCV + matplotlib exceed it under load; the container is killed and waits for a human. |

None of the three branding elements exist off that platform. Streamlit itself is
Apache-2.0 and ships none of them.

## Design principles

1. **Provider-agnostic.** `setup.sh` knows nothing about Oracle. Oracle is the
   temporary free host; a small paid VPS is the destination at the end of the
   project. That move is a re-run of this script, not a rewrite. Everything
   Oracle-specific is confined to Part B below.
2. **No committed file is modified for the VM.** All Streamlit settings are
   applied through environment variables in `onnm.service`. `.streamlit/config.toml`
   is read by Community Cloud — which stays deployed as the rollback — and its
   own comments warn that pinning `server.address` breaks that platform.
3. **The weights never leave the server.** `src/checkpoint_fetch.py` downloads
   the checkpoint onto this machine's disk. No browser receives it.
4. **The old deployment is not deleted.** It is the rollback until Phase 4.

---

## Part A — what only you can do

1. **Oracle Cloud account.** A card is required for identity verification.
   Always Free resources cannot be charged. Do *not* upgrade to Pay-As-You-Go
   unless we have discussed it.
2. **Domain.** Buy it; we point an `A` record at the VM.
3. **Google Cloud console** → Credentials → your OAuth 2.0 Client → add
   `https://YOUR_DOMAIN/oauth2callback` to **Authorised redirect URIs**.
   **Add it alongside the existing Streamlit Cloud URI — do not replace it.**
   Both must work during cutover so the rollback stays usable.

---

## Part B — provision the VM (the only Oracle-specific part)

**Shape:** `VM.Standard.A1.Flex`, **2 OCPU / 8 GB**, Ubuntu 24.04 LTS (aarch64).

**Why 8 GB and not the full 12 GB you are entitled to.** Oracle reclaims an
Always Free instance only when CPU (95th percentile), network **and** memory are
*all* below 20% across a 7-day window, with the memory criterion applying to A1
shapes. A warm torch process resident at ~2 GB is 25% of 8 GB, but only ~17% of
12 GB. Sizing down keeps real memory use above the threshold. This is honest
sizing, not synthetic load — but it is a mitigation, not a guarantee, so
verification step 10 checks it against actual OCI metrics after a week.

**Region:** pick for A1 availability. US regions frequently return
"Out of host capacity"; EU/APAC usually provision within minutes.

**Networking:**
- Reserve a **static public IP** so rebuilding the instance does not force a DNS
  change.
- Security list: ingress on 22, 80, 443 only.

> **The trap that costs everyone an hour:** Oracle's Ubuntu images ship a
> restrictive `iptables` ruleset *in addition to* the cloud security list. A port
> opened in the OCI console alone still silently drops traffic, and it looks
> exactly like "the app failed to start". `setup.sh` configures `ufw` and saves
> the rules, which handles it.

---

## Part C — run setup

```bash
ssh ubuntu@YOUR_VM_IP
git clone https://github.com/kali-fz/OsteoNeuralNetwork-Model.git /tmp/onnm-src
sudo bash /tmp/onnm-src/deploy/vm/setup.sh
```

Idempotent — safe to re-run. It picks `requirements-arm64.txt` on aarch64 and
`requirements.txt` on x86_64 automatically. Expect 10–20 minutes; the torch wheel
is large and two ARM cores are not fast.

## Part D — fill in the secrets

Three files, none of them in git:

```bash
sudo nano /etc/onnm.env                          # from onnm.env.example
sudo -u onnm nano /opt/onnm/.streamlit/secrets.toml   # from secrets.toml.example
sudo nano /etc/caddy/Caddyfile                   # replace REPLACE_WITH_YOUR_DOMAIN
sudo chmod 0600 /opt/onnm/.streamlit/secrets.toml
sudo systemctl reload caddy
sudo systemctl start onnm
sudo journalctl -u onnm -f
```

Generate a fresh cookie secret — do not reuse the Streamlit Cloud one, or a
session minted by one deployment is accepted by the other:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

**Set `ONNM_COMMUNITY_URL` and `ONNM_COMMUNITY_KEY` together or not at all.**
`src/backend.py` switches to hosted D1 only when both are present. With one
missing it silently falls back to local SQLite, which on a fresh VM is an empty
user table — every existing account appears not to exist.

---

## Part E — verification

Run in order. **Do not change DNS until 1–7 pass.**

| # | check | command / method |
|---|---|---|
| 1 | Environment gate | `sudo -u onnm /opt/onnm/.venv/bin/python /opt/onnm/scripts/verify_env.py` |
| 2 | Test suite (464) | `cd /opt/onnm && sudo -u onnm .venv/bin/python -m pytest -q` |
| 3 | **ARM inference parity** | `scripts/check_inference_parity.py` — see below. Exits non-zero on drift, so it gates rather than informs |
| 4 | Version identity | App shows the **v1.0.0** badge, not a warning — means `find_by_sha` matched the fetched checkpoint against `model_versions.json` |
| 5 | Branding | Clean incognito window: no avatar, no "Hosted with Streamlit", no GitHub/Fork |
| 6 | Google Sign-In | Full sign-in on the domain; header shows Google photo and name; sign-out works |
| 7 | Globe + community loop | Globe renders and rotates; submit a shared test scan; confirm the row via `wrangler d1 execute onn-model --remote` |
| 8 | **Rebuild drill** | Destroy the VM, rebuild from this directory alone, repeat 1–7 |
| 9 | Stays up | `sudo reboot`, confirm unattended recovery; leave untouched 72 h and confirm still serving |
| 10 | Not reclaimed | After 7 days, OCI metrics show memory utilisation above 20% |

### Step 3 in detail

The one genuine correctness risk in this move. A different CPU architecture
dispatches to different oneDNN kernels, which could produce different
probabilities. Nothing errors; the app looks fine; only the numbers move.

Run on **both** machines, then compare:

```bash
# identical on both -- the sha256 printed MUST match, or the comparison is void
python scripts/check_inference_parity.py make-probe --out probe.png

# --device cpu on both. The migration is CPU-to-CPU, so recording the local
# baseline on the ROCm GPU would compare device AND architecture at once, and a
# failure would not say which one moved.
python scripts/check_inference_parity.py record \
    --checkpoint dist/v1.0.0/best.pt --image probe.png \
    --device cpu --out parity-local.json

# then, with both records in one place
python scripts/check_inference_parity.py compare parity-local.json parity-vm.json
```

It guards the three-class probabilities, the lesion probability, the calibrated
threshold and temperature, the uncertainty-gate inputs, and both the intensity
and the **location** of the Grad-CAM peak. A heatmap that keeps its statistics
but moves its peak is a different explanation of the same score, so `heatmap_argmax`
is compared exactly.

Tolerance is 1e-4. For scale: the same checkpoint on this machine's ROCm GPU
versus its CPU differs by about 1e-6, and the calibrated threshold sits at 0.496
— so 1e-4 cannot flip a verdict that was not already exactly on the line.

Use the synthetic probe for a quick check and a real radiograph for the final
one; real data exercises the transform chain on realistic statistics.

**Step 8 is not optional.** It is what turns "Oracle terminated my account" from
a catastrophe into an afternoon, and it must be proven before cutover, not after.

---

## Part F — operations

```bash
sudo systemctl status onnm          # is it up
sudo journalctl -u onnm -f          # live logs
sudo journalctl -u onnm --since "1 hour ago" -p err
sudo systemctl restart onnm
```

Deploying a change:

```bash
sudo -u onnm git -C /opt/onnm fetch --depth 1 origin main
sudo -u onnm git -C /opt/onnm checkout -f FETCH_HEAD
sudo -u onnm /opt/onnm/.venv/bin/pip install -r /opt/onnm/requirements-arm64.txt
sudo systemctl restart onnm
```

**Backups.** Most state is already off-box in D1 (accounts, submissions, globe,
contributor roll). What lives only here is `/opt/onnm/data/user_uploads/`. Copy
it off the VM on a schedule — it contains de-identified radiographs, so treat the
destination as confidential and keep it out of the repository.

---

## Part G — if Oracle terminates the account

Documented repeatedly on Oracle's own support forums, with no reason given and
no reinstatement. The recovery, in order:

1. Point DNS back at the Streamlit Cloud app. The site is live again in minutes
   — degraded (it sleeps, it is slow, the branding is back) but serving.
2. Provision a new host anywhere.
3. Run `setup.sh`. Restore `/etc/onnm.env`, `secrets.toml`, and the uploads backup.
4. Add the new redirect URI in the Google console if the domain changed.
5. Repoint DNS.

Nothing irreplaceable is lost at any point, which is the entire reason for the
rebuild drill.

---

## Part H — Phase 4: migrating to the paid VPS

Planned, at the end of the project. Because `setup.sh` is provider-agnostic and
selects its requirements file from `uname -m`, an x86 VPS needs no code change:

1. Provision the VPS.
2. Run `setup.sh` (it picks `requirements.txt` automatically on x86_64).
3. Restore `/etc/onnm.env`, `secrets.toml`, and `data/user_uploads/`.
4. Add the redirect URI, repoint DNS, verify with Part E.
5. **Only then** delete the Oracle instance and the Streamlit Cloud deployment.

This retires the Oracle termination, tier-change, capacity and idle-reclamation
risks in one step. Expect under an hour.
