"""ONNM — local Streamlit interface for bone-lesion triage on plain radiographs.

    streamlit run app.py

Inference always runs where the app runs: the model and the Grad-CAM are never
sent anywhere to be computed. The whole stack is free and open source.

Two deployments, and the difference is worth knowing before reading the code.
Run locally, nothing leaves the machine and accounts live in a SQLite file.
Hosted, accounts live in Cloudflare D1, sign-in is delegated to Google, and a
submission record -- plus the 256px image, but only with per-image consent --
is written to the community database. ``src/backend.py`` picks between the two
on configuration alone; see ``src/legal.py`` for what each one discloses.

Performance note (section 5A of REDESIGN_BRIEF):
  ``onnm.inference`` (and therefore torch) are imported lazily inside
  ``render_scanner`` and inside ``load_classifier``.  Every page that is NOT
  the scanner pays zero torch-import cost.  Cold landing-page load drops from
  several seconds to under a second on a warm Python install.

All model work lives in ``onnm.inference``. This file is presentation only.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# Importable straight from a clone, the same trick scripts/_bootstrap.py uses.
SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Lightweight imports (no torch, no MONAI) ─────────────────────────────────
from auth import (  # noqa: E402
    AuthenticationError,
    authenticate_user,
    google_sign_in_required,
    initialize_session,
    login_session,
    logout_session,
    register_user,
)
from backend import initialize_database, using_community  # noqa: E402
from checkpoint_fetch import ensure_checkpoint, serving_checkpoint_info  # noqa: E402
from community import get_client, is_admin  # noqa: E402
from community_ui import (  # noqa: E402
    admin_can_review,
    community_status,
    record_rejection,
    record_submission,
    render_admin_review,
    render_feedback,
    render_rejection_dispute,
    render_share_consent,
)
from components.charts import render_probability_chart, render_roc_chart  # noqa: E402
from components.globe import render_globe  # noqa: E402
from database import (  # noqa: E402
    DatabaseError,
    create_upload,
    list_user_uploads,
    update_upload_result,
)
from legal import COOKIE_NOTICE, MEDICAL_DISCLAIMER, PRIVACY_POLICY, TERMS_OF_SERVICE  # noqa: E402
from oauth import (  # noqa: E402
    identity_profile,
    is_signed_in,
    oidc_configured,
    render_sign_in,
    resolve_account,
    sign_out,
)
from onnm import __version__  # noqa: E402

# onnm.inference is NOT imported here — see load_classifier and render_scanner.
# onnm.io_radiograph (MONAI) is NOT imported here — imported inside render_scanner.
from onnm.ood import (  # noqa: E402 — torch-free, safe at module level
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_ENTROPY_GATE,
    REJECTION_MESSAGE,
    validate_payload,
)
from report import build_html_report  # noqa: E402
from storage import StorageError, delete_upload, is_user_file, save_upload  # noqa: E402
from theme import (  # noqa: E402
    INK,
    LINE,
    LINE_SOFT,
    MUTED,
    WHITE,
    inject_theme,
    verdict_card,
)

COLORMAPS = ["jet", "turbo", "inferno", "magma", "viridis", "hot"]

# Colour per class, reused by charts and the verdict card so that
# "malignant" means the same red everywhere in the UI.
CLASS_COLORS = {
    "normal": "#2e8b57",
    "benign": "#e0a800",
    "malignant": "#c62828",
}

DISCLAIMER_SUMMARY = """
**Research tool — not a medical device, and not medical advice.** This unvalidated
prototype has no FDA, CE, or MHRA clearance. Never use its output for patient-care
decisions; every radiograph requires review by a qualified clinician.
"""


# ── Resources ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint: str):
    """Load one checkpoint once per server process.

    The import of RadiographClassifier (and therefore torch) is inside this
    function, so torch is never imported unless the scanner page is actually
    visited.  Cached on the checkpoint path so swapping models loads the new
    one without evicting the old resident.
    """
    from onnm.inference import RadiographClassifier
    return RadiographClassifier(checkpoint, warmup=True)


@st.cache_data(show_spinner=False)
def device_info() -> dict:
    from onnm.utils import describe_device
    return describe_device()


@st.cache_data(ttl=300, show_spinner=False)
def globe_data() -> dict:
    """Fetch aggregated country counts; fails soft on any error."""
    from geo import build_markers
    return build_markers(get_client().globe())


@st.cache_data(ttl=300, show_spinner=False)
def contributor_data() -> list[dict] | None:
    """Fetch explicitly public contributor profiles without blocking the page."""
    return get_client().contributors() if using_community() else []


@st.cache_data(show_spinner=False, max_entries=64)
def inspect_upload(payload: bytes, filename: str):
    """Validate an unchanged upload once rather than on every widget rerun."""
    return hashlib.sha256(payload).hexdigest(), validate_payload(payload, filename)



# ── Navigation ───────────────────────────────────────────────────────────────

def navigate_to(page: str) -> None:
    st.session_state["current_page"] = page
    st.query_params["page"] = page
    st.rerun()


# ── Shared helpers ────────────────────────────────────────────────────────────

def render_account_header(page_title: str, *, active_page: str) -> None:
    """One account/navigation row for every authenticated page.

    Keep identity and account actions together. Duplicating them in the sidebar
    made the redesigned scanner show two competing logout controls and left the
    visible header with no indication of which account was active.
    """
    title_col, nav_col, identity_col, signout_col = st.columns([5, 1, 1.75, 0.8])
    with title_col:
        st.markdown(
            f'<div class="onnm-header-title">{html.escape(page_title)}</div>',
            unsafe_allow_html=True,
        )
    with nav_col:
        target = "scanner" if active_page == "landing" else "landing"
        label = "Scanner" if active_page == "landing" else "← Home"
        if st.button(label, key=f"account_nav_{active_page}", use_container_width=True):
            navigate_to(target)
    with identity_col:
        name = str(st.session_state.get("user_name") or "My profile")
        picture = str(st.session_state.get("user_picture") or "")
        safe_name = html.escape(name)
        if picture:
            safe_picture = html.escape(picture, quote=True)
            avatar = f'<img class="onnm-account-avatar" src="{safe_picture}" alt="">'
        else:
            initials = html.escape("".join(part[:1] for part in name.split()[:2]).upper() or "U")
            avatar = f'<span class="onnm-account-avatar onnm-account-initials">{initials}</span>'
        st.markdown(
            f'<a class="onnm-account-link" href="?page=profile" target="_self">'
            f'{avatar}<span class="onnm-account-name">{safe_name}</span></a>',
            unsafe_allow_html=True,
        )
    with signout_col:
        if st.button("Sign Out", key="account_signout", use_container_width=True):
            _do_logout()

def _hero_bg_css() -> str:
    svg_path = Path(__file__).parent / "assets" / "hero-moss.svg"
    try:
        raw = svg_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    except OSError:
        return ""


def probability_chart(probabilities: dict[str, float]):
    """Matplotlib fallback chart — kept for any callers that reference it directly."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = list(probabilities)
    values = [100.0 * probabilities[n] for n in names]
    colors = [CLASS_COLORS.get(n, "#4c72b0") for n in names]

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": LINE})
    fig, ax = plt.subplots(figsize=(6, 1.9), dpi=140)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(WHITE)
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability (%)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=9, colors=INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=LINE_SOFT, linewidth=0.8)
    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(
            min(value + 1.5, 92), bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=9, fontweight="bold", color=INK,
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    fig.tight_layout()
    return fig




MAX_PREVIEW_PX = 1024


def preview_uint8(array: np.ndarray, max_edge: int = MAX_PREVIEW_PX) -> np.ndarray:
    """Downscale for display only. The original is kept for the report export.

    `use_container_width=True` scales in the browser, so a 2010x1490 film is
    still ~2.9 MB of PNG on the wire to render a column a few hundred pixels
    wide. Capping the long edge here cuts that by an order of magnitude with no
    visible difference at the size it is actually shown.

    Display only: `png_bytes(to_display_uint8(result.original_image))` still
    feeds the HTML report from the full-resolution array, because that is meant
    to be zoomed into.

    Deliberately uncached: the resize is a few milliseconds, while
    ``st.cache_data`` would hash the multi-megabyte input array on every rerun
    to look it up. Streamlit's media manager already dedupes identical output
    by content hash, so a re-render of an unchanged image costs no upload.
    """
    from PIL import Image

    from onnm.inference import to_display_uint8

    display = to_display_uint8(array)
    height, width = display.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return display
    scale = max_edge / longest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(display).resize(size, Image.LANCZOS))


def png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def render_legal_footer() -> None:
    st.divider()
    st.caption("Legal and privacy information")
    with st.expander("Terms of Service"):
        st.markdown(TERMS_OF_SERVICE)
    with st.expander("Privacy Policy"):
        st.markdown(PRIVACY_POLICY)
    with st.expander("Medical Disclaimer"):
        st.markdown(MEDICAL_DISCLAIMER)
    with st.expander("Cookie Notice"):
        st.markdown(COOKIE_NOTICE)


# ── Page: landing ─────────────────────────────────────────────────────────────

def render_landing() -> None:
    authenticated = st.session_state.get("authenticated", False)
    hero_img = _hero_bg_css()
    hero_style = f'style="--hero-img:url(\'{hero_img}\')"' if hero_img else ""

    if authenticated:
        render_account_header("OsteoNeuralNetwork Model", active_page="landing")
    else:
        brand_col, signin_col = st.columns([5, 1])
        with brand_col:
            st.markdown(
                '<div class="onnm-public-brand">ONNM <span>Open research prototype</span></div>',
                unsafe_allow_html=True,
            )
        with signin_col:
            label = "Sign in with Google" if (using_community() or oidc_configured()) else "Sign in"
            if st.button(label, key="home_signin", use_container_width=True):
                navigate_to("auth")

    data = globe_data()
    hero_col, globe_col = st.columns([1.08, 0.92], gap="large")
    with hero_col:
        st.markdown(
            f"""
            <section class="onnm-home-hero" {hero_style}>
              <p class="onnm-hero-eyebrow">Community research model · ONNM v{__version__}</p>
              <h1 class="onnm-hero-title">Test the model. Help make it better.</h1>
              <p class="onnm-hero-subtitle">
                We have trained the first ONNM model to examine bone X-rays for normal,
                benign, and malignant patterns. Sign up free, try it on your radiographs,
                and—with your permission—help us build a better next version.
              </p>
              <div class="onnm-hero-points">
                <span>Test v{__version__}</span>
                <span>DICOM · PNG · JPG · BMP · TIFF</span>
                <span>Sharing is always optional</span>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if authenticated:
            if st.button(
                f"Test ONNM v{__version__}  →", key="hero_scanner_btn", type="primary",
                use_container_width=True,
            ):
                navigate_to("scanner")
        else:
            cta_label = (
                "Sign up free with Google  →"
                if (using_community() or oidc_configured())
                else "Sign in to scan  →"
            )
            if st.button(
                cta_label, key="hero_google_btn", type="primary", use_container_width=True
            ):
                navigate_to("auth")
        st.caption("Research tool only — not a medical device and not medical advice.")

    with globe_col:
        st.markdown(
            '<div class="onnm-globe-heading"><span>Community reach</span>'
            '<strong>Country-level and privacy protected</strong></div>',
            unsafe_allow_html=True,
        )
        markers = data.get("markers") or []
        render_globe(markers, height=410)
        elsewhere = data.get("elsewhere") or {}
        hidden = int(elsewhere.get("signup", 0)) if isinstance(elsewhere, dict) else 0
        privacy_min = int(data.get("k_anonymity_min") or 1)
        if data.get("available") and not markers:
            st.caption(
                f"No country has reached the {privacy_min}-account display threshold yet. "
                "People are still included in the totals below."
            )
        elif not data.get("available"):
            st.caption(
                "Community map data is temporarily unavailable. The globe will populate "
                "after the community service update completes."
            )
        elif hidden:
            st.caption(
                f"{hidden:,} account(s) are included in totals but hidden from the map "
                f"until their country reaches {privacy_min}."
            )
        else:
            st.caption(
                "Drag to rotate. Markers show aggregated countries, never precise locations."
            )

    totals = data.get("totals", {})
    n_users = totals.get("users", 0)
    n_contributors = totals.get("contributors", 0)
    n_scans = totals.get("approved_submissions", 0)
    n_countries = totals.get("countries_represented", 0)

    stats_html = '<div class="onnm-stats">'
    if n_users:
        stats_html += (
            f'<div class="onnm-stat">'
            f'<span class="onnm-stat-value">{n_users:,}</span>'
            f'<span class="onnm-stat-label">Registered users</span>'
            f'</div>'
        )
    if n_scans:
        stats_html += (
            f'<div class="onnm-stat">'
            f'<span class="onnm-stat-value">{n_scans:,}</span>'
            f'<span class="onnm-stat-label">Reviewed shared scans</span>'
            f'</div>'
        )
    if n_countries:
        stats_html += (
            f'<div class="onnm-stat">'
            f'<span class="onnm-stat-value">{n_countries:,}</span>'
            f'<span class="onnm-stat-label">Countries represented</span>'
            f'</div>'
        )
    stats_html += "</div>"
    if n_users or n_contributors or n_scans or n_countries:
        st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="onnm-metric-band">
          <h2>What the scanner gives you</h2>
          <div class="onnm-metric-grid">
            <div class="onnm-metric-item">
              <span class="onnm-metric-num">3-way</span>
              <span class="onnm-metric-ci">Normal · benign · malignant</span>
              <div class="onnm-metric-desc">
                A calibrated probability breakdown, not a black-box label.
              </div>
            </div>
            <div class="onnm-metric-item">
              <span class="onnm-metric-num">Grad-CAM</span>
              <span class="onnm-metric-ci">Visual explanation</span>
              <div class="onnm-metric-desc">
                A heatmap shows which part of the image influenced the result.
              </div>
            </div>
            <div class="onnm-metric-item">
              <span class="onnm-metric-num">Private</span>
              <span class="onnm-metric-ci">Consent before sharing</span>
              <div class="onnm-metric-desc">
                Your image is not added to community research without consent.
              </div>
            </div>
          </div>
          <p style="margin:20px 0 0;font-size:13px;color:#6b6457;line-height:1.6;">
            Every result must be reviewed by a qualified clinician. This prototype has no
            FDA, CE, or MHRA clearance and must never direct patient care.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    contributors = contributor_data()
    st.markdown(
        '<div class="onnm-section-heading"><span>Contributors</span>'
        '<strong>People helping improve the research dataset</strong></div>',
        unsafe_allow_html=True,
    )
    if contributors:
        cards = []
        for contributor in contributors:
            name = str(contributor.get("name") or "Contributor")
            picture = str(contributor.get("picture") or "")
            count = max(0, int(contributor.get("approved_contributions") or 0))
            safe_name = html.escape(name)
            if picture:
                avatar = (
                    f'<img class="onnm-contributor-avatar" '
                    f'src="{html.escape(picture, quote=True)}" alt="">'
                )
            else:
                initials = html.escape(
                    "".join(part[:1] for part in name.split()[:2]).upper() or "C"
                )
                avatar = (
                    f'<span class="onnm-contributor-avatar onnm-account-initials">'
                    f'{initials}</span>'
                )
            noun = "contribution" if count == 1 else "contributions"
            cards.append(
                f'<div class="onnm-contributor-card">{avatar}'
                f'<span><strong>{safe_name}</strong>'
                f'<small>{count:,} approved {noun}</small></span></div>'
            )
        st.markdown(
            f'<div class="onnm-contributor-grid">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )
    elif contributors is None:
        st.caption(
            "Contributor profiles are temporarily unavailable while the community "
            "service is updated. Your Google profile remains private."
        )
    else:
        st.caption(
            "Approved contributors can choose to show their Google name and photo here "
            "from My Profile. No account is published automatically."
        )

    st.markdown(
        f"""
        <div class="onnm-footer">
          <span>v{__version__}</span>
          <span class="onnm-footer-disclaimer">
            Research tool — not a medical device and not medical advice. This unvalidated
            prototype has no FDA, CE, or MHRA clearance. Never use its output for
            patient-care decisions. Every radiograph requires review by a qualified clinician.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_legal_footer()


# ── Page: auth ────────────────────────────────────────────────────────────────

def render_auth() -> None:
    if st.button("← Home", key="auth_back_btn"):
        navigate_to("landing")

    st.subheader("Secure access")
    if using_community():
        st.caption(
            "Accounts are stored in Cloudflare D1. Your password is hashed in this "
            "app with PBKDF2-HMAC-SHA256 before it is sent; the plaintext never "
            "leaves the browser session."
        )
    else:
        st.caption(
            "Accounts, password hashes, and scans remain on this computer. "
            "No authentication or storage service is contacted."
        )

    login_tab, create_tab = st.tabs(["Login", "Create Account"])
    with login_tab:
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            lock_until = float(st.session_state.get("login_lock_until", 0.0))
            remaining = int(max(0, lock_until - time.monotonic()))
            if remaining:
                st.error(f"Too many failed attempts. Try again in {remaining + 1} seconds.")
            else:
                user = authenticate_user(email, password)
                if user:
                    st.session_state["login_failures"] = 0
                    login_session(st.session_state, user)
                    navigate_to("scanner")
                else:
                    failures = int(st.session_state.get("login_failures", 0)) + 1
                    st.session_state["login_failures"] = failures
                    if failures >= 5:
                        st.session_state["login_lock_until"] = time.monotonic() + 30
                        st.session_state["login_failures"] = 0
                    st.error("Invalid email or password.")

    with create_tab:
        with st.form("registration_form", clear_on_submit=True):
            reg_email = st.text_input("Email", key="register_email", autocomplete="email")
            reg_password = st.text_input(
                "Password", type="password", key="register_password",
                help="At least 12 characters, including a letter and a number.",
                autocomplete="new-password",
            )
            reg_confirm = st.text_input(
                "Confirm password", type="password",
                key="register_confirmation", autocomplete="new-password",
            )
            accepted = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy, and acknowledge "
                "this is an unvalidated research prototype."
            )
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    user = register_user(reg_email, reg_password, accepted_terms=accepted)
                except AuthenticationError as exc:
                    st.error(str(exc))
                else:
                    login_session(st.session_state, user)
                    st.success("Account created.")
                    navigate_to("scanner")

    render_legal_footer()


# ── Page: scanner ─────────────────────────────────────────────────────────────

def render_scanner() -> None:
    # Lazy imports: torch/MONAI load ONLY when the scanner page is visited.
    from onnm.inference import (
        UPLOAD_TYPES,
        InferenceResult,
        find_checkpoints,
        is_throwaway_run,
        production_checkpoint,
        render_overlay,
        to_display_uint8,
    )
    from onnm.io_radiograph import RadiographReadError

    def render_serving_version(selected, pinned) -> None:
        info = serving_checkpoint_info()
        if not info or not info.get("sha256"):
            return
        try:
            from onnm.versioning import find_by_sha, load_registry
            version = find_by_sha(load_registry(), info["sha256"])
        except Exception:
            version = None
        if version is None:
            st.warning(
                "Serving a checkpoint that is not in the version ledger. "
                f"Fetched from `{info.get('checkpoint_url', 'an unrecorded URL')}` — "
                "register it or republish a known version."
            )
            return
        st.caption(
            f"Serving **ONN {version.version}** · "
            f"fetched {info.get('fetched_at', 'unknown')}"
        )
        if selected != pinned:
            st.caption("You are previewing a different checkpoint than the one being served.")

    render_account_header("Bone lesion scanner", active_page="scanner")
    st.markdown(
        f"""
        <div class="onnm-scanner-intro">
          <span>ONNM v{__version__}</span>
          <strong>Upload a radiograph and review the result with its visual explanation.</strong>
          <p>Use the sidebar only when you need to adjust the model threshold or heatmap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="onnm-disclaimer"><strong>Research tool — not a medical device, '
        'and not medical advice.</strong> This unvalidated prototype has no FDA, CE, '
        'or MHRA clearance. Never use its output for patient-care decisions; every '
        'radiograph requires review by a qualified clinician.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _status = community_status()
        if _status is not None and is_admin(
            st.session_state.get("user_id"), st.session_state.get("user_email")
        ):
            with st.expander(f"Community · {_status.get('pending_review', 0)} awaiting review"):
                st.caption(
                    "Full-width console:\n\n"
                    "`python -m streamlit run review_app.py --server.port 8502`"
                )
                admin_user = st.session_state.get("user_id")
                admin_email = st.session_state.get("user_email")
                if admin_can_review(admin_user, admin_email):
                    render_admin_review(admin_user, admin_email)
                else:
                    st.caption("Set ONNM_ADMIN_KEY to open the review queue.")

        ensure_checkpoint()
        st.header("Model")
        checkpoints = find_checkpoints()
        if not checkpoints:
            st.error("No checkpoint found under `reports/`.")
            st.code(
                "python scripts/train.py --override configs/densenet121_3class.yaml",
                language="bash",
            )
            render_legal_footer()
            st.stop()

        visible = [p for p in checkpoints if not is_throwaway_run(p)] or checkpoints
        try:
            pinned = production_checkpoint()
        except FileNotFoundError as exc:
            pinned = None
            st.warning(str(exc))
        if pinned is not None and pinned not in visible:
            visible.insert(0, pinned)

        selected = st.selectbox(
            "Checkpoint", options=visible,
            index=visible.index(pinned) if pinned is not None else 0,
            format_func=lambda p: (
                f"{p.parent.name}  [production]" if p == pinned else p.parent.name
            ),
            help="Defaults to the run pinned in `reports/PRODUCTION`.",
        )
        if pinned is None:
            st.caption("No production pin — defaulting to the newest run.")
        render_serving_version(selected, pinned)

        try:
            with st.spinner("Loading model…"):
                classifier = load_classifier(str(selected))
        except Exception as exc:
            st.exception(exc)
            st.stop()

        info = classifier.describe()
        hardware = device_info()

        if hardware.get("cuda_available"):
            st.success(
                f"**{hardware.get('device_name', 'GPU')}**  \n"
                f"{hardware.get('backend', 'GPU')} · {hardware.get('total_memory_gb', '?')} GB · "
                f"torch {hardware['torch']}"
            )
        else:
            st.warning(
                f"Running on **CPU** (torch {hardware['torch']}). "
                "Run `python scripts/verify_env.py` to diagnose."
            )

        st.caption(
            f"`{info['architecture']}` · {info['image_size']}px · "
            f"{info['cam_method']} @ `{info['cam_layer']}`"
        )
        if "trained_epochs" in info:
            st.caption(f"trained {info['trained_epochs']} epoch(s)")
        if "malignant_recall" in info:
            st.caption(f"checkpoint val malignant recall: **{info['malignant_recall']:.3f}**")

        st.divider()
        st.header("Decision")
        if info["calibrated"]:
            constraint = (
                f"holding specificity >= {info['min_specificity']:.2f}"
                if info.get("mode") == "specificity_floor"
                else f"holding sensitivity >= {info['target_sensitivity']:.2f}"
            )
            st.info(
                f"Calibrated: T={info['temperature']:.2f}, threshold "
                f"{info['default_threshold']:.3f}, {constraint}. "
                f"On validation: sensitivity {info['val_sensitivity']:.2f}, "
                f"specificity {info['val_specificity']:.2f}."
            )
            for warning in info.get("calibration_warnings", []):
                st.warning(warning)
        else:
            st.warning(
                "**Uncalibrated.** Probabilities are raw softmax outputs. "
                "Run `python scripts/calibrate.py` to fit on the validation split."
            )

        threshold = st.slider(
            "Lesion threshold", min_value=0.05, max_value=0.95,
            value=float(round(info["default_threshold"] / 0.05) * 0.05), step=0.05,
            help="P(benign)+P(malignant) at or above this is called a potential lesion.",
        )
        cam_class = st.selectbox(
            "Explain which class",
            options=["auto", "predicted", *classifier.class_names],
            help="`auto` targets the most likely lesion class.",
        )

        with st.expander("Threshold sweep (ROC)"):
            sweep_file = selected.parent / "threshold_sweep.json"
            if sweep_file.is_file():
                try:
                    sweep_rows = json.loads(sweep_file.read_text(encoding="utf-8"))["sweep"]
                except (OSError, KeyError, ValueError) as exc:
                    sweep_rows = []
                    st.warning(f"`threshold_sweep.json` is unreadable: {exc}")
                if sweep_rows:
                    render_roc_chart(sweep_rows, threshold)
            else:
                st.caption(
                    f"No sweep saved for this run. Generate with:\n"
                    "`python scripts/calibrate.py --checkpoint "
                    f"reports/{selected.parent.name}/best.pt --sweep`"
                )

        st.divider()
        st.header("Heatmap")
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
        cam_floor = st.slider("Attention floor", 0.0, 0.9, 0.25, 0.05,
            help="Hide CAM values below this threshold.")
        colormap = st.selectbox("Colormap", COLORMAPS, index=0)
        st.divider()
        st.caption("No data leaves this machine.")

    uploads = st.file_uploader(
        "Upload one or more radiographs", type=UPLOAD_TYPES, accept_multiple_files=True,
        help="DICOM (.dcm/.dicom/.ima), PNG, JPEG, BMP or TIFF.",
    )
    SHARE_CONSENT = render_share_consent("upload")

    if not uploads:
        st.markdown(
            '<div class="onnm-empty-state"><strong>Ready when you are.</strong>'
            '<span>Upload one X-ray or a series. Each film gets a classification, '
            'confidence score, and Grad-CAM heatmap.</span></div>',
            unsafe_allow_html=True,
        )
        render_legal_footer()
        return

    if "cases" not in st.session_state:
        st.session_state["cases"] = {}
    cases: dict = st.session_state["cases"]

    rejected: list[tuple[str, object, bytes]] = []
    failed: list[tuple[str, str]] = []
    ready: list[tuple[str, dict]] = []

    for uploaded in uploads:
        payload = uploaded.getvalue()
        digest, validation = inspect_upload(payload, uploaded.name)
        if not validation.is_radiograph:
            rejected.append((uploaded.name, validation, payload))
            continue

        file_key = f"{st.session_state['user_id']}:{uploaded.name}:{digest}"

        entry = cases.get(file_key)
        if entry is not None:
            entry["share_consent"] = SHARE_CONSENT
        if entry is None:
            try:
                stored = save_upload(payload, user_id=st.session_state["user_id"],
                    original_filename=uploaded.name)
            except StorageError as exc:
                failed.append((uploaded.name, str(exc)))
                continue
            entry = {"stored": stored, "record_id": None, "share_consent": SHARE_CONSENT}
            cases[file_key] = entry

        cache_key = (str(selected), cam_class)
        if entry.get("cache_key") != cache_key:
            stored = entry["stored"]
            try:
                with st.spinner(f"Running inference on {uploaded.name}…"):
                    result = classifier.predict(
                        stored.path, with_heatmap=True, threshold=threshold,
                        cam_class=cam_class, uncertainty_floor=DEFAULT_CONFIDENCE_FLOOR,
                        entropy_gate=DEFAULT_ENTROPY_GATE,
                    )
                if entry["record_id"] and not using_community():
                    update_upload_result(entry["record_id"], st.session_state["user_id"],
                        model_verdict=result.label, confidence_score=result.confidence)
                else:
                    if using_community():
                        entry["record_id"] = stored.upload_id
                    else:
                        record = create_upload(
                            upload_id=stored.upload_id, user_id=st.session_state["user_id"],
                            filename=stored.original_filename, file_path=stored.path,
                            model_verdict=result.label, confidence_score=result.confidence,
                        )
                        entry["record_id"] = record.upload_id
                entry["recorded_verdict"] = result.label
                entry["result_base"] = result
                entry["cache_key"] = cache_key
                if entry.get("submission_id") is None:
                    entry["submission_id"] = record_submission(
                        st.session_state["user_id"], result,
                        shared=entry.get("share_consent", False),
                        preprocessed=result.preprocessed_image,
                        checkpoint=Path(str(selected)).parent.name,
                    )
            except RadiographReadError as exc:
                if not entry["record_id"]:
                    delete_upload(stored.path)
                    cases.pop(file_key, None)
                failed.append((uploaded.name, f"could not decode: {exc}"))
                continue
            except (DatabaseError, StorageError) as exc:
                if not entry["record_id"]:
                    delete_upload(stored.path)
                    cases.pop(file_key, None)
                failed.append((uploaded.name, str(exc)))
                continue
            except Exception as exc:  # noqa: BLE001
                if not entry["record_id"]:
                    delete_upload(stored.path)
                    cases.pop(file_key, None)
                failed.append((uploaded.name, f"inference failed: {exc}"))
                continue

        base_result = entry.get("result_base")
        if base_result is not None:
            entry["result"] = base_result.with_threshold(
                threshold, uncertainty_floor=DEFAULT_CONFIDENCE_FLOOR,
                entropy_gate=DEFAULT_ENTROPY_GATE,
            )
            shown = entry["result"]
            if (entry.get("record_id") and not using_community()
                    and entry.get("recorded_verdict") != shown.label):
                with contextlib.suppress(DatabaseError):
                    update_upload_result(entry["record_id"], st.session_state["user_id"],
                        model_verdict=shown.label, confidence_score=shown.confidence)
                entry["recorded_verdict"] = shown.label

        ready.append((uploaded.name, entry))

    for name, validation, payload in rejected:
        st.error(f"`{name}` — {REJECTION_MESSAGE}")
        rejection_key = (
            f"rejected:{st.session_state['user_id']}:{hashlib.sha256(payload).hexdigest()}"
        )
        if rejection_key not in st.session_state:
            st.session_state[rejection_key] = record_rejection(
                st.session_state["user_id"], payload, shared=SHARE_CONSENT, filename=name)
        with st.expander(f"Why was {name} rejected?"):
            for check in validation.failures:
                st.markdown(f"- **{check.name}** — {check.detail}")
            st.caption(
                "Pre-inference heuristics (channel structure, dynamic range, intensity entropy, "
                "edge density). If a genuine radiograph is rejected, export it as an uncropped "
                "grayscale DICOM or PNG and try again."
            )
            render_rejection_dispute(
                st.session_state[rejection_key], st.session_state["user_id"], key=rejection_key)

    for name, message in failed:
        st.error(f"`{name}` — {message}")

    if not ready:
        render_legal_footer()
        return

    if len(ready) > 1:
        st.subheader(f"Series review — {len(ready)} films")
        st.dataframe(
            [{"file": name, "verdict": entry["result"].label,
              "confidence %": round(entry["result"].confidence, 1),
              "lesion %": round(100 * entry["result"].lesion_probability, 1),
              "malignant %": round(100 * entry["result"].malignant_probability, 1)}
             for name, entry in ready],
            use_container_width=True, hide_index=True,
        )
        case_name = st.selectbox("Open case", options=[name for name, _ in ready],
            help="Detailed verdict, probabilities, and Grad-CAM for one film.")
        entry = next(e for n, e in ready if n == case_name)
    else:
        case_name, entry = ready[0]

    result: InferenceResult = entry["result"]

    if result.inconclusive:
        accent = "#b26a00"
    elif result.is_lesion:
        accent = "#c62828"
    else:
        accent = "#2e8b57"

    if result.inconclusive:
        st.warning(
            f"The model's probabilities are too uncertain to present as a finding "
            f"(max class probability {100 * result.max_probability:.1f}% below the "
            f"{100 * DEFAULT_CONFIDENCE_FLOOR:.0f}% floor, or normalized entropy "
            f"{result.predictive_entropy:.2f} at/above the {DEFAULT_ENTROPY_GATE:.2f} gate). "
            "This often indicates an out-of-domain or non-diagnostic image. "
            "The raw probabilities are shown below; obtain a qualified read regardless."
        )

    verdict_card(
        result.label,
        f"{result.confidence:.1f}% confidence · decided at a "
        f"{result.threshold:.2f} lesion threshold",
        accent,
    )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.metric("Confidence", f"{result.confidence:.1f}%")
        st.metric("Lesion probability", f"{100 * result.lesion_probability:.1f}%",
            help="P(benign) + P(malignant). This is the number the threshold is applied to.")
        st.metric("Malignant probability", f"{100 * result.malignant_probability:.1f}%",
            help="Shown separately because a cancer called benign is followed up, "
                 "while a cancer called normal sends the patient home.")
        st.caption(
            f"{result.elapsed_ms:.0f} ms on `{result.device}`"
            + (f" · T={result.temperature:.2f}" if result.calibrated else " · uncalibrated")
        )
    with right:
        st.markdown("**Three-class breakdown**")
        render_probability_chart(result.class_probabilities)
        st.caption(
            "The network is a 3-way classifier. The Normal / Potential Bone Lesion verdict "
            "above collapses benign and malignant — read this chart for the distinction "
            "that changes management."
        )

    st.divider()
    render_feedback(entry.get("submission_id"), st.session_state["user_id"],
        key=str(entry.get("record_id") or "case"))
    st.subheader("Grad-CAM")

    overlay = None
    if result.heatmap is None:
        st.warning(
            "Grad-CAM could not be computed for this image; the classification above is "
            "unaffected. Check the terminal for the underlying error."
        )
        st.image(to_display_uint8(result.preprocessed_image),
            caption="Model input (256 px, aspect-preserved and padded)", use_container_width=True)
    else:
        overlay = render_overlay(result.preprocessed_image, result.heatmap,
            alpha=alpha, colormap=colormap, threshold=cam_floor)
        col_a, col_b, col_c = st.columns(3, gap="medium")
        with col_a:
            st.image(
                preview_uint8(result.original_image),
                caption=(
                    f"As uploaded — {result.original_image.shape[1]}×"
                    f"{result.original_image.shape[0]}"
                ),
                use_container_width=True,
            )
        with col_b:
            st.image(to_display_uint8(result.preprocessed_image),
                caption=f"Model input — {result.preprocessed_image.shape[1]}px, padded",
                use_container_width=True)
        with col_c:
            st.image(overlay,
                caption=f'Grad-CAM — attention for "{result.cam_class}"',
                use_container_width=True)
        st.caption(
            "Warm regions are where the model's evidence for the selected class came from. "
            "A heatmap on an implant, collimation edge, or burned-in marker means the "
            "prediction is unreliable however confident it looks."
        )

    stem = Path(case_name).stem
    export_a, export_b, export_c = st.columns(3)
    with export_a:
        report_html = build_html_report(
            filename=case_name, verdict=result.label, confidence_pct=result.confidence,
            class_probabilities=result.class_probabilities,
            lesion_probability=result.lesion_probability, threshold=result.threshold,
            calibrated=result.calibrated, temperature=result.temperature,
            inconclusive=result.inconclusive, max_probability=result.max_probability,
            predictive_entropy=result.predictive_entropy,
            checkpoint_name=selected.parent.name, app_version=__version__,
            disclaimer=MEDICAL_DISCLAIMER,
            original_png=png_bytes(to_display_uint8(result.original_image)),
            overlay_png=png_bytes(overlay) if overlay is not None else None,
            cam_class=result.cam_class,
        )
        st.download_button("Report (HTML, print to PDF)", report_html,
            file_name=f"{stem}_onnm_report.html", mime="text/html", use_container_width=True)
    with export_b:
        if overlay is not None:
            st.download_button("Overlay (PNG)", png_bytes(overlay),
                file_name=f"{stem}_gradcam.png", mime="image/png", use_container_width=True)
    with export_c:
        st.download_button("Result (JSON)", json.dumps(result.as_dict(), indent=2),
            file_name=f"{stem}_onnm.json", mime="application/json", use_container_width=True)

    with st.expander("Decoding details"):
        st.json(result.source_meta)
        st.caption(
            "`inverted` marks a MONOCHROME1 DICOM that was flipped back to a positive image; "
            "`voi_lut_applied` marks the DICOM window/level having been honoured."
        )

    st.divider()
    st.caption(
        "BTXRD is licensed CC BY-NC-ND 4.0 — NoDerivatives covers Grad-CAM overlays, so keep "
        "downloaded images local rather than redistributing them."
    )
    render_legal_footer()


# ── Page: profile ──────────────────────────────────────────────────────────────

def render_profile() -> None:
    render_account_header("My Profile", active_page="profile")

    user_id = st.session_state.get("user_id", "")
    email = st.session_state.get("user_email", "")
    provider = "Google" if oidc_configured() else "Password"
    tos_at = st.session_state.get("accepted_terms_at", "")
    safe_email = html.escape(str(email))
    safe_provider = html.escape(provider)
    safe_tos_at = html.escape(str(tos_at or "—"))

    st.markdown(
        f"""
        <div class="onnm-profile-card">
          <h3>Account</h3>
          <div class="onnm-profile-row">
            <span class="onnm-profile-key">Email</span>
            <span class="onnm-profile-val">{safe_email}</span>
          </div>
          <div class="onnm-profile-row">
            <span class="onnm-profile-key">Auth provider</span>
            <span class="onnm-profile-val">{safe_provider}</span>
          </div>
          <div class="onnm-profile-row">
            <span class="onnm-profile-key">Terms accepted</span>
            <span class="onnm-profile-val">{safe_tos_at}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if using_community() and oidc_configured():
        st.subheader("Contributor profile")
        if contributor_data() is None:
            st.caption(
                "Contributor profiles are temporarily unavailable while the community "
                "service is updated. Your Google name and photo remain private."
            )
        else:
            st.caption(
                "Your Google name and photo are private by default. Turn this on to show "
                "them on the homepage only when you have an approved contribution."
            )
            current_public = bool(st.session_state.get("public_contributor_profile", False))
            public_profile = st.toggle(
                "Show me in the public contributors section",
                value=current_public,
                key="public_contributor_toggle",
            )
            if public_profile != current_public:
                updated = get_client().update_contributor_profile(
                    user_id,
                    str(st.session_state.get("user_subject") or ""),
                    display_name=st.session_state.get("user_name"),
                    profile_picture_url=st.session_state.get("user_picture"),
                    public_profile=public_profile,
                )
                if updated:
                    st.session_state["public_contributor_profile"] = public_profile
                    contributor_data.clear()
                    st.success("Contributor visibility updated.")
                else:
                    st.error(
                        "Contributor visibility could not be updated. The community "
                        "service may still be deploying; your profile remains private."
                    )

    if st.button("Open scanner →", key="profile_open_scanner"):
        navigate_to("scanner")

    st.subheader("My Scan History")
    _render_scan_history(user_id)
    render_legal_footer()


def _render_scan_history(user_id: str) -> None:
    """Scan history scoped to the session user — displayed on the profile page."""
    if using_community():
        submissions = get_client().list_user_submissions(user_id)
        if not submissions:
            st.caption("No saved scans yet.")
            return
        st.dataframe(
            [{"Uploaded": item.get("created_at", ""), "Verdict": item.get("model_label", ""),
              "Lesion probability": f"{100 * float(item.get('lesion_probability', 0)):.1f}%",
              "Shared": "Yes" if item.get("shared") else "No",
              "Review": item.get("review_status", "pending")}
             for item in submissions],
            hide_index=True, use_container_width=True,
        )
        return

    try:
        records = list_user_uploads(user_id)
    except DatabaseError as exc:
        st.error(str(exc))
        return
    if not records:
        st.caption("No saved scans yet.")
        return

    st.dataframe(
        [{"Uploaded": r.upload_timestamp, "Filename": r.filename,
          "Verdict": r.model_verdict, "Confidence": f"{r.confidence_score:.1f}%"}
         for r in records],
        hide_index=True, use_container_width=True,
    )
    selected_record = st.selectbox(
        "Preview a saved scan", records,
        format_func=lambda r: f"{r.upload_timestamp[:16]} · {r.filename} · {r.model_verdict}",
        key="history_selection",
    )
    if not is_user_file(user_id, selected_record.file_path):
        st.warning("The saved image is missing or outside this account's storage.")
        return
    try:
        from onnm.io_radiograph import read_radiograph
        image, _ = read_radiograph(selected_record.file_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not open the saved scan: {exc}")
        return
    st.image(
        preview_uint8(image),
        caption=(
            f"{selected_record.model_verdict} · "
            f"{selected_record.confidence_score:.1f}% confidence"
        ),
        use_container_width=True,
    )


# ── Logout helper ──────────────────────────────────────────────────────────────

def _do_logout() -> None:
    logout_session(st.session_state)
    st.query_params["page"] = "landing"
    if is_signed_in():
        sign_out()
    else:
        navigate_to("landing")


# ── Entry point ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ONNM — Bone Lesion Triage",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()

initialize_session(st.session_state)
try:
    initialize_database()
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

USING_GOOGLE = oidc_configured()
if USING_GOOGLE:
    _account = resolve_account()
    if _account is not None:
        _identity = identity_profile()
        login_session(
            st.session_state,
            _account,
            display_name=_identity.get("name"),
            picture=_identity.get("picture"),
            subject=_identity.get("subject"),
        )
        _sync_key = (
            _account.user_id,
            _identity.get("name"),
            _identity.get("picture"),
        )
        if (
            using_community()
            and st.session_state.get("profile_sync_key") != _sync_key
            and get_client().update_contributor_profile(
                _account.user_id,
                _identity.get("subject") or "",
                display_name=_identity.get("name"),
                profile_picture_url=_identity.get("picture"),
                public_profile=(
                    True if st.session_state.get("public_contributor_profile") else None
                ),
            )
        ):
            st.session_state["profile_sync_key"] = _sync_key

_query_page = st.query_params.get("page")
if _query_page in {"landing", "auth", "scanner", "profile"}:
    st.session_state["current_page"] = _query_page
page = st.session_state.get("current_page", "landing")

if page == "landing":
    render_landing()
elif page == "auth":
    # Hosted accounts live in D1 and are Google identities. Never expose the
    # password-signup fallback merely because OAuth secrets are incomplete;
    # render_sign_in() reports the deployment error without creating a second
    # authentication path. Local-only installs may keep using password login.
    if google_sign_in_required(hosted=using_community(), oidc_available=USING_GOOGLE):
        if st.session_state.get("authenticated"):
            navigate_to("scanner")
        else:
            render_sign_in()
    else:
        render_auth()
elif page == "scanner":
    if not st.session_state.get("authenticated"):
        navigate_to("auth")
    else:
        render_scanner()
elif page == "profile":
    if not st.session_state.get("authenticated"):
        navigate_to("auth")
    else:
        render_profile()
else:
    navigate_to("landing")
