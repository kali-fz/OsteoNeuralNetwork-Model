#!/usr/bin/env bash
#
# ONNM self-hosted deployment -- idempotent, provider-agnostic setup.
#
# Runs on any systemd Linux with apt. Deliberately knows nothing about Oracle
# Cloud: the current host is Oracle's Always Free ARM tier, the eventual host is
# a small paid x86 VPS, and the whole point of this script is that the move
# between them is a re-run rather than a rewrite. Anything provider-specific
# lives in the provisioning step (see README), never in here.
#
# Safe to run repeatedly. Every step checks before it acts.
#
#   sudo ./setup.sh
#
set -euo pipefail

APP_USER="onnm"
APP_DIR="/opt/onnm"
REPO_URL="${ONNM_REPO_URL:-https://github.com/kali-fz/OsteoNeuralNetwork-Model.git}"
REPO_REF="${ONNM_REPO_REF:-main}"
VENV="${APP_DIR}/.venv"
PY="python3.12"

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf '\n!!  %s\n' "$*" >&2; }
die()  { printf '\nXX  %s\n' "$*" >&2; exit 1; }

[[ ${EUID} -eq 0 ]] || die "run as root (sudo ./setup.sh)"

# ---------------------------------------------------------------------------
# 0. Architecture -> requirements file
# ---------------------------------------------------------------------------
# The ONLY architecture-dependent decision in the deployment, resolved here at
# runtime rather than baked in, so this script is correct on both hosts:
# Oracle ARM now, an x86 VPS at the end of the project.
ARCH="$(uname -m)"
case "${ARCH}" in
  aarch64|arm64) REQ="requirements-arm64.txt" ;;
  x86_64|amd64)  REQ="requirements.txt" ;;
  *) die "unsupported architecture: ${ARCH}" ;;
esac
log "Architecture ${ARCH} -> ${REQ}"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

# Ubuntu 24.04 ships Python 3.12 as its system python3. Older Debian/Ubuntu (a
# likely x86 VPS image) does not, and the project invariant is 3.12 exactly.
if ! command -v "${PY}" >/dev/null 2>&1; then
  warn "${PY} not found; adding deadsnakes"
  apt-get install -y -qq software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
fi

apt-get install -y -qq \
  "${PY}" "${PY}-venv" "${PY}-dev" \
  build-essential git curl ca-certificates gnupg \
  libgl1 libglib2.0-0 \
  ufw fail2ban unattended-upgrades

# libgl1 + libglib2.0-0 are for opencv-python-headless. "headless" removes the
# GUI backends, not the shared-library dependency, and the failure is an
# ImportError deep inside cv2 at first import -- long after install "succeeded".

command -v "${PY}" >/dev/null 2>&1 || die "${PY} still unavailable after install"

# ---------------------------------------------------------------------------
# 2. Caddy (reverse proxy + automatic TLS)
# ---------------------------------------------------------------------------
# Chosen over nginx specifically because of Streamlit's WebSocket at
# /_stcore/stream. Caddy's reverse_proxy upgrades WebSockets transparently; the
# equivalent nginx config needs explicit Upgrade/Connection headers and a raised
# proxy_read_timeout, and getting it subtly wrong presents as an app stuck on
# "Please wait..." with nothing useful in the logs.
if ! command -v caddy >/dev/null 2>&1; then
  log "Installing Caddy"
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -qq
  apt-get install -y -qq caddy
else
  log "Caddy already installed"
fi

# ---------------------------------------------------------------------------
# 3. Service account
# ---------------------------------------------------------------------------
if ! id -u "${APP_USER}" >/dev/null 2>&1; then
  log "Creating service user ${APP_USER}"
  useradd --system --create-home --home-dir "${APP_DIR}" --shell /usr/sbin/nologin "${APP_USER}"
else
  log "Service user ${APP_USER} exists"
fi
install -d -o "${APP_USER}" -g "${APP_USER}" -m 0755 "${APP_DIR}"

# ---------------------------------------------------------------------------
# 4. Source
# ---------------------------------------------------------------------------
if [[ -d "${APP_DIR}/.git" ]]; then
  log "Updating source to ${REPO_REF}"
  sudo -u "${APP_USER}" git -C "${APP_DIR}" fetch --depth 1 origin "${REPO_REF}"
  sudo -u "${APP_USER}" git -C "${APP_DIR}" checkout -f FETCH_HEAD
else
  log "Cloning ${REPO_URL} @ ${REPO_REF}"
  sudo -u "${APP_USER}" git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${APP_DIR}"
fi

# data/ holds user uploads and is gitignored, so a fresh clone has none of it.
# 0700 because it contains de-identified radiographs, and de-identified is not
# the same as public.
install -d -o "${APP_USER}" -g "${APP_USER}" -m 0700 "${APP_DIR}/data"
install -d -o "${APP_USER}" -g "${APP_USER}" -m 0700 "${APP_DIR}/data/user_uploads"
install -d -o "${APP_USER}" -g "${APP_USER}" -m 0755 "${APP_DIR}/reports"

# ---------------------------------------------------------------------------
# 5. Virtualenv
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "Creating virtualenv"
  sudo -u "${APP_USER}" "${PY}" -m venv "${VENV}"
fi

log "Installing Python dependencies from ${REQ}"
log "(slow on 2 ARM cores -- expect 10-20 minutes; torch alone is a large wheel)"
sudo -u "${APP_USER}" "${VENV}/bin/pip" install --upgrade pip wheel --quiet
sudo -u "${APP_USER}" "${VENV}/bin/pip" install -r "${APP_DIR}/${REQ}"

# ---------------------------------------------------------------------------
# 6. systemd + Caddy config
# ---------------------------------------------------------------------------
log "Installing systemd unit"
install -m 0644 "${APP_DIR}/deploy/vm/onnm.service" /etc/systemd/system/onnm.service

if [[ ! -f /etc/onnm.env ]]; then
  log "Seeding /etc/onnm.env from example -- YOU MUST EDIT THIS"
  install -m 0600 -o root -g root "${APP_DIR}/deploy/vm/onnm.env.example" /etc/onnm.env
  warn "/etc/onnm.env contains placeholders. Fill it in before starting onnm.service."
else
  log "/etc/onnm.env exists -- left untouched"
  chmod 0600 /etc/onnm.env
fi

# Streamlit's secrets.toml carries the Google OAuth client secret and the session
# cookie secret. It is gitignored, so it is never in the clone and must be placed
# by hand. Owned by the service user because Streamlit reads it as that user.
install -d -o "${APP_USER}" -g "${APP_USER}" -m 0700 "${APP_DIR}/.streamlit"
if [[ ! -f "${APP_DIR}/.streamlit/secrets.toml" ]]; then
  warn "${APP_DIR}/.streamlit/secrets.toml is missing -- Google Sign-In will fail closed."
  warn "Create it from deploy/vm/secrets.toml.example before cutover."
fi

if [[ ! -f /etc/caddy/Caddyfile.onnm-installed ]]; then
  log "Installing Caddyfile"
  install -m 0644 "${APP_DIR}/deploy/vm/Caddyfile" /etc/caddy/Caddyfile
  touch /etc/caddy/Caddyfile.onnm-installed
  warn "Edit /etc/caddy/Caddyfile: replace REPLACE_WITH_YOUR_DOMAIN, then systemctl reload caddy"
else
  log "Caddyfile already installed -- left untouched (edit by hand)"
fi

# ---------------------------------------------------------------------------
# 7. Firewall
# ---------------------------------------------------------------------------
# Belt and braces on purpose. Oracle's Ubuntu images ship a restrictive iptables
# ruleset IN ADDITION to the cloud security list, so a port opened in the OCI
# console alone still silently drops traffic. That failure looks exactly like
# "the app did not start", and costs an hour if you have not met it before.
log "Configuring firewall"
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
if command -v netfilter-persistent >/dev/null 2>&1; then
  netfilter-persistent save >/dev/null 2>&1 || true
fi

log "Enabling unattended security upgrades"
dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true
systemctl enable --now fail2ban >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 8. Enable (not start -- secrets may not be in place yet)
# ---------------------------------------------------------------------------
systemctl daemon-reload
systemctl enable onnm.service >/dev/null

log "Setup complete."
printf '%s\n' \
  "" \
  "Remaining manual steps (not automated -- they need secrets):" \
  "" \
  "  1. Edit /etc/onnm.env          (Worker URL/keys, checkpoint URL + sha256)" \
  "  2. Create ${APP_DIR}/.streamlit/secrets.toml from deploy/vm/secrets.toml.example" \
  "     then: chown ${APP_USER}:${APP_USER} it and chmod 0600 it" \
  "  3. Edit /etc/caddy/Caddyfile   (replace REPLACE_WITH_YOUR_DOMAIN)" \
  "  4. systemctl reload caddy" \
  "  5. systemctl start onnm && journalctl -u onnm -f" \
  "" \
  "Then run the verification sequence in deploy/vm/README.md."
