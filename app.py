"""ONNM — local Streamlit interface for bone-lesion triage on plain radiographs.

    streamlit run app.py

Runs entirely on this machine: the model, the Grad-CAM, and the server itself.
Nothing is uploaded anywhere, no external API is called, and the whole stack is
free and open source. Uploaded files are de-identified and retained in private local account storage
under the research-data terms shown during account creation.

All model work lives in ``onnm.inference``. This file is presentation only, and
should stay that way -- if a computation needs to move here to make the layout
work, that is a signal the inference API is missing something.
"""

from __future__ import annotations

import hashlib
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

from auth import (  # noqa: E402
    AuthenticationError,
    authenticate_user,
    initialize_session,
    login_session,
    logout_session,
    register_user,
)
from community_ui import (  # noqa: E402
    community_status,
    record_submission,
    render_admin_review,
    render_feedback,
    render_share_consent,
)
from database import (  # noqa: E402
    DatabaseError,
    create_upload,
    initialize_database,
    list_user_uploads,
    update_upload_result,
)
from legal import COOKIE_NOTICE, MEDICAL_DISCLAIMER, PRIVACY_POLICY, TERMS_OF_SERVICE  # noqa: E402
from onnm import __version__  # noqa: E402
from onnm.inference import (  # noqa: E402
    UPLOAD_TYPES,
    InferenceResult,
    RadiographClassifier,
    find_checkpoints,
    is_throwaway_run,
    production_checkpoint,
    render_overlay,
    to_display_uint8,
)
from onnm.io_radiograph import RadiographReadError, read_radiograph  # noqa: E402
from onnm.ood import (  # noqa: E402
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_ENTROPY_GATE,
    REJECTION_MESSAGE,
    validate_payload,
)
from onnm.utils import describe_device  # noqa: E402
from report import build_html_report  # noqa: E402
from storage import StorageError, delete_upload, is_user_file, save_upload  # noqa: E402
from theme import (  # noqa: E402
    INK,
    LINE,
    LINE_SOFT,
    MONO,
    MUTED,
    WHITE,
    inject_theme,
    masthead,
    verdict_card,
)

COLORMAPS = ["jet", "turbo", "inferno", "magma", "viridis", "hot"]

# Colour per class, reused by the bar chart and the verdict card so that
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


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint: str) -> RadiographClassifier:
    """Load one checkpoint, once per session-server.

    Cached on the checkpoint path: swapping models in the sidebar loads the new
    one and keeps the old resident, which is the right trade at ~30 MB per
    DenseNet-121 and lets a reader flip between runs without a reload each time.
    """
    return RadiographClassifier(checkpoint, warmup=True)


@st.cache_data(show_spinner=False)
def device_info() -> dict:
    return describe_device()


def probability_chart(probabilities: dict[str, float]):
    """Horizontal bar chart of the 3-way head, in class order not sorted order.

    Keeping normal/benign/malignant in a fixed left-to-right order matters more
    than ranking them: a reader comparing two films should find malignant in the
    same place both times.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Chart chrome follows the Git-Design canvas -- transparent figure, hairline
    # rules, monospaced ticks. The bar colours stay clinical: they encode
    # malignancy, so they are not the design system's to recolour.
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": LINE})
    names = list(probabilities)
    values = [100.0 * probabilities[n] for n in names]
    colors = [CLASS_COLORS.get(n, "#4c72b0") for n in names]

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
            f"{value:.1f}%", va="center", fontsize=9, fontweight="bold",
            color=INK,
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    fig.tight_layout()
    return fig


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


def render_authentication() -> None:
    st.subheader("Secure local access")
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
                    st.rerun()
                else:
                    failures = int(st.session_state.get("login_failures", 0)) + 1
                    st.session_state["login_failures"] = failures
                    if failures >= 5:
                        st.session_state["login_lock_until"] = time.monotonic() + 30
                        st.session_state["login_failures"] = 0
                    st.error("Invalid email or password.")

    with create_tab:
        with st.form("registration_form", clear_on_submit=True):
            email = st.text_input("Email", key="register_email", autocomplete="email")
            password = st.text_input(
                "Password",
                type="password",
                key="register_password",
                help="At least 12 characters, including a letter and a number.",
                autocomplete="new-password",
            )
            confirmation = st.text_input(
                "Confirm password",
                type="password",
                key="register_confirmation",
                autocomplete="new-password",
            )
            accepted = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy, and acknowledge "
                "this is an unvalidated research prototype."
            )
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            if password != confirmation:
                st.error("Passwords do not match.")
            else:
                try:
                    user = register_user(
                        email,
                        password,
                        accepted_terms=accepted,
                    )
                except AuthenticationError as exc:
                    st.error(str(exc))
                else:
                    login_session(st.session_state, user)
                    st.success("Account created.")
                    st.rerun()


def render_scan_history(user_id: str) -> None:
    with st.expander("My Past Scans"):
        try:
            records = list_user_uploads(user_id)
        except DatabaseError as exc:
            st.error(str(exc))
            return
        if not records:
            st.caption("No saved scans yet.")
            return

        st.dataframe(
            [
                {
                    "Uploaded": record.upload_timestamp,
                    "Filename": record.filename,
                    "Verdict": record.model_verdict,
                    "Confidence": f"{record.confidence_score:.1f}%",
                }
                for record in records
            ],
            hide_index=True,
            use_container_width=True,
        )
        selected_record = st.selectbox(
            "Preview a saved scan",
            records,
            format_func=lambda record: (
                f"{record.upload_timestamp[:16]} · {record.filename} · {record.model_verdict}"
            ),
            key="history_selection",
        )
        if not is_user_file(user_id, selected_record.file_path):
            st.warning("The saved image is missing or outside this account's storage.")
            return
        try:
            image, _ = read_radiograph(selected_record.file_path)
        except RadiographReadError as exc:
            st.error(f"Could not open the saved scan: {exc}")
            return
        st.image(
            to_display_uint8(image),
            caption=(
                f"{selected_record.model_verdict} · "
                f"{selected_record.confidence_score:.1f}% confidence"
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ONNM — Bone Lesion Triage",
    layout="wide",
    initial_sidebar_state="expanded",
)

# All presentation rules live in theme.py so the app and the exported case
# report stay one visual system. Must follow set_page_config.
inject_theme()

masthead(
    "OsteoNeuralNetwork-Model",
    eyebrow="Explainable bone-tumour triage on plain radiographs",
    meta=[f"v{__version__}", "Local", "Offline", "Zero cost"],
)
st.error(DISCLAIMER_SUMMARY)

initialize_session(st.session_state)
try:
    initialize_database()
except DatabaseError as exc:
    st.error(str(exc))
    render_legal_footer()
    st.stop()

if not st.session_state["authenticated"]:
    render_authentication()
    render_legal_footer()
    st.stop()

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.success(f"Logged in as: **{st.session_state['user_email']}**")
    if st.button("Logout", use_container_width=True):
        logout_session(st.session_state)
        st.rerun()
    render_scan_history(st.session_state["user_id"])

    st.divider()
    # Community loop status. The review queue appears only when ONNM_ADMIN_KEY
    # is set, which it is not in the hosted Space.
    _status = community_status()
    if _status is not None:
        with st.expander(f"Community · {_status.get('pending_review', 0)} awaiting review"):
            render_admin_review()

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

    # Hide throwaway runs (smoke-*, tmp-*, debug-*) from the picker unless
    # they are all that exists. They stay on disk for debugging.
    visible = [p for p in checkpoints if not is_throwaway_run(p)] or checkpoints

    try:
        pinned = production_checkpoint()
    except FileNotFoundError as exc:
        pinned = None
        st.warning(str(exc))
    if pinned is not None and pinned not in visible:
        visible.insert(0, pinned)

    selected = st.selectbox(
        "Checkpoint",
        options=visible,
        index=visible.index(pinned) if pinned is not None else 0,
        format_func=lambda p: (
            f"{p.parent.name}  [production]" if p == pinned else p.parent.name
        ),
        help="Defaults to the run pinned in `reports/PRODUCTION`; otherwise the newest "
             "non-throwaway `reports/*/best.pt`. Preprocessing is read from inside the "
             "checkpoint, so each one runs under its own training config.",
    )
    if pinned is None:
        st.caption(
            "No production pin — defaulting to the newest run. To pin one, write its "
            "folder name into `reports/PRODUCTION`."
        )

    try:
        with st.spinner("Loading model onto the GPU…"):
            classifier = load_classifier(str(selected))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
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
            f"Running on **CPU** (torch {hardware['torch']}). Inference still works, "
            "just slower. If you expected the GPU, run `python scripts/verify_env.py`."
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
        # Name the constraint that actually bound. Reporting the sensitivity
        # target when specificity is what was held fixed would describe the
        # wrong policy, which is worse than reporting nothing.
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
            "**Uncalibrated.** Probabilities are raw softmax outputs and the "
            "threshold below is a naive 0.50, which corresponds to no clinical "
            "policy. Run `python scripts/calibrate.py --checkpoint ...` to fit "
            "both on the validation split."
        )

    threshold = st.slider(
        "Lesion threshold",
        min_value=0.05, max_value=0.95,
        value=float(round(info["default_threshold"] / 0.05) * 0.05), step=0.05,
        help="P(benign) + P(malignant) at or above this is called a potential lesion. "
             "Defaults to the calibrated operating point when one has been fitted. "
             "Lower it to catch more lesions at the cost of more false alarms.",
    )
    cam_class = st.selectbox(
        "Explain which class",
        options=["auto", "predicted", *classifier.class_names],
        help="`auto` targets the most likely lesion class — including when the verdict "
             "is Normal, which answers 'where would it have been?'.",
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
                import altair as alt
                import pandas as pd

                frame = pd.DataFrame(sweep_rows)
                frame["fpr"] = 1.0 - frame["specificity"]
                roc = (
                    alt.Chart(frame)
                    .mark_line(point=True, interpolate="step-after",
                               color=INK, strokeWidth=1.5)
                    .encode(
                        x=alt.X("fpr:Q", title="1 − specificity",
                                scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("sensitivity:Q", title="sensitivity",
                                scale=alt.Scale(domain=[0, 1])),
                        tooltip=[
                            alt.Tooltip("threshold:Q", format=".3f"),
                            alt.Tooltip("sensitivity:Q", format=".3f"),
                            alt.Tooltip("specificity:Q", format=".3f"),
                            alt.Tooltip("youden_j:Q", format=".3f"),
                        ],
                    )
                )
                # Nearest sweep row to the live slider, so moving the slider
                # shows where on the curve the app is currently operating.
                current = frame.iloc[(frame["threshold"] - threshold).abs().argmin()]
                marker = (
                    alt.Chart(current.to_frame().T)
                    .mark_point(size=140, color="#c62828", filled=True)
                    .encode(x="fpr:Q", y="sensitivity:Q")
                )
                # Match the page: transparent ground, hairline rules, mono ticks.
                chart = (roc + marker).configure_view(
                    strokeWidth=1, stroke=LINE
                ).configure_axis(
                    gridColor=LINE_SOFT, domainColor=LINE,
                    tickColor=LINE, labelColor=MUTED,
                    titleColor=INK, labelFont=MONO, titleFontWeight=500,
                ).configure(background="transparent")
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "Fitted on the validation split. The red point is the slider's "
                    "current operating point; hover the curve for the threshold behind "
                    "each trade-off."
                )
        else:
            st.caption(
                "No sweep saved for this run. Generate one with:\n"
                "`python scripts/calibrate.py --checkpoint "
                f"reports/{selected.parent.name}/best.pt --sweep`"
            )

    st.divider()
    st.header("Heatmap")

    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    cam_floor = st.slider(
        "Attention floor", 0.0, 0.9, 0.25, 0.05,
        help="Hide CAM values below this so only the concentrated attention is painted.",
    )
    colormap = st.selectbox("Colormap", COLORMAPS, index=0)

    st.divider()
    st.caption(
        "No data leaves this machine. Uploads are de-identified and retained in "
        "this account's private local research storage."
    )

# -- Upload ----------------------------------------------------------------
uploads = st.file_uploader(
    "Upload one or more radiographs",
    type=UPLOAD_TYPES,
    accept_multiple_files=True,
    help="DICOM (.dcm/.dicom/.ima), PNG, JPEG, BMP or TIFF. DICOM headers are honoured: "
         "modality LUT, VOI window, and MONOCHROME1 inversion are all applied. Select "
         "multiple files to review a series; each gets its own verdict and report.",
)

# Consent for community sharing. Default OFF, and scoped to this upload rather
# than remembered: a willingness to share one teaching example says nothing
# about the next file the user opens. Absent the community API this renders
# nothing and returns False.
SHARE_CONSENT = render_share_consent("upload")

if not uploads:
    st.info(
        "Upload an X-ray to begin — or several at once to review a series. The model "
        "classifies each film, reports a confidence score, and renders a Grad-CAM "
        "heatmap showing which region drove the call."
    )
    render_legal_footer()
    st.stop()

# Per-file store/predict cache. Keyed on content digest so a Streamlit rerun
# (every slider move) never re-saves a file or duplicates a history row.
if "cases" not in st.session_state:
    st.session_state["cases"] = {}
cases: dict = st.session_state["cases"]

rejected: list[tuple[str, object]] = []
failed: list[tuple[str, str]] = []
ready: list[tuple[str, dict]] = []

for uploaded in uploads:
    payload = uploaded.getvalue()

    # OOD gate: reject non-radiographs before they reach storage or the model.
    # A closed-set softmax forces any input into one of its three classes, so
    # an unvalidated photograph would come back as a ~50% "benign" call.
    validation = validate_payload(payload, uploaded.name)
    if not validation.is_radiograph:
        rejected.append((uploaded.name, validation))
        continue

    digest = hashlib.sha256(payload).hexdigest()
    file_key = f"{st.session_state['user_id']}:{uploaded.name}:{digest}"

    entry = cases.get(file_key)
    if entry is not None:
        entry["share_consent"] = SHARE_CONSENT
    if entry is None:
        try:
            stored = save_upload(
                payload,
                user_id=st.session_state["user_id"],
                original_filename=uploaded.name,
            )
        except StorageError as exc:
            failed.append((uploaded.name, str(exc)))
            continue
        entry = {"stored": stored, "record_id": None, "share_consent": SHARE_CONSENT}
        cases[file_key] = entry

    cache_key = (str(selected), threshold, cam_class)
    if entry.get("cache_key") != cache_key:
        stored = entry["stored"]
        try:
            with st.spinner(f"Running inference on {uploaded.name}…"):
                result = classifier.predict(
                    stored.path,
                    with_heatmap=True,
                    threshold=threshold,
                    cam_class=cam_class,
                    uncertainty_floor=DEFAULT_CONFIDENCE_FLOOR,
                    entropy_gate=DEFAULT_ENTROPY_GATE,
                )
            # Re-running inference after a threshold or heatmap change updates
            # one history record instead of duplicating files or scan entries.
            if entry["record_id"]:
                update_upload_result(
                    entry["record_id"],
                    st.session_state["user_id"],
                    model_verdict=result.label,
                    confidence_score=result.confidence,
                )
            else:
                record = create_upload(
                    upload_id=stored.upload_id,
                    user_id=st.session_state["user_id"],
                    filename=stored.original_filename,
                    file_path=stored.path,
                    model_verdict=result.label,
                    confidence_score=result.confidence,
                )
                entry["record_id"] = record.upload_id
            entry["result"] = result
            entry["cache_key"] = cache_key
            # Log to the community API. Consent was captured per file before
            # inference ran; without it no image is sent, only the verdict.
            if entry.get("submission_id") is None:
                entry["submission_id"] = record_submission(
                    st.session_state["user_id"],
                    result,
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
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            if not entry["record_id"]:
                delete_upload(stored.path)
                cases.pop(file_key, None)
            failed.append((uploaded.name, f"inference failed: {exc}"))
            continue

    ready.append((uploaded.name, entry))

for name, validation in rejected:
    st.error(f"`{name}` — {REJECTION_MESSAGE}")
    with st.expander(f"Why was {name} rejected?"):
        for check in validation.failures:
            st.markdown(f"- **{check.name}** — {check.detail}")
        st.caption(
            "These are pre-inference heuristics (channel structure, dynamic range, "
            "intensity entropy, edge density). If a genuine radiograph is rejected, "
            "export it as an uncropped grayscale DICOM or PNG and try again."
        )
for name, message in failed:
    st.error(f"`{name}` — {message}")

if not ready:
    render_legal_footer()
    st.stop()

# -- Batch summary (only when reviewing a series) ----------------------------
if len(ready) > 1:
    st.subheader(f"Series review — {len(ready)} films")
    st.dataframe(
        [
            {
                "file": name,
                "verdict": entry["result"].label,
                "confidence %": round(entry["result"].confidence, 1),
                "lesion %": round(100 * entry["result"].lesion_probability, 1),
                "malignant %": round(100 * entry["result"].malignant_probability, 1),
            }
            for name, entry in ready
        ],
        use_container_width=True,
        hide_index=True,
    )
    case_name = st.selectbox(
        "Open case",
        options=[name for name, _ in ready],
        help="Detailed verdict, probabilities, and Grad-CAM for one film of the series.",
    )
    entry = next(e for n, e in ready if n == case_name)
else:
    case_name, entry = ready[0]

result: InferenceResult = entry["result"]

# -- Verdict ---------------------------------------------------------------
# The card is white on a hairline rule now, so only the accent is needed: the
# tinted fill it used to carry is not part of the Git-Design panel geometry.
if result.inconclusive:
    accent = "#b26a00"
elif result.is_lesion:
    accent = "#c62828"
else:
    accent = "#2e8b57"

if result.inconclusive:
    st.warning(
        "The model's probabilities are too uncertain to present as a finding "
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
    st.metric(
        "Lesion probability", f"{100 * result.lesion_probability:.1f}%",
        help="P(benign) + P(malignant). This is the number the threshold is applied to.",
    )
    st.metric(
        "Malignant probability", f"{100 * result.malignant_probability:.1f}%",
        help="Shown separately because a cancer called benign is followed up, while a "
             "cancer called normal sends the patient home.",
    )
    st.caption(
        f"{result.elapsed_ms:.0f} ms on `{result.device}`"
        + (f" · T={result.temperature:.2f}" if result.calibrated else " · uncalibrated")
    )

with right:
    st.markdown("**Three-class breakdown**")
    st.pyplot(probability_chart(result.class_probabilities), use_container_width=True)
    st.caption(
        "The network is a 3-way classifier. The Normal / Potential Bone Lesion verdict "
        "above collapses benign and malignant into one bucket — read this chart for the "
        "distinction that changes management."
    )

st.divider()

# -- Imagery ---------------------------------------------------------------
# The user can dispute the verdict. This records a flag for human review; it
# cannot change a label or the model.
render_feedback(
    entry.get("submission_id"),
    st.session_state["user_id"],
    key=str(entry.get("record_id") or "case"),
)

st.subheader("Grad-CAM")

overlay = None
if result.heatmap is None:
    st.warning(
        "Grad-CAM could not be computed for this image; the classification above is "
        "unaffected. Check the terminal for the underlying error."
    )
    st.image(
        to_display_uint8(result.preprocessed_image),
        caption="Model input (256 px, aspect-preserved and padded)",
        use_container_width=True,
    )
else:
    overlay = render_overlay(
        result.preprocessed_image, result.heatmap,
        alpha=alpha, colormap=colormap, threshold=cam_floor,
    )

    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        st.image(
            to_display_uint8(result.original_image),
            caption=f"As uploaded — {result.original_image.shape[1]}×"
                    f"{result.original_image.shape[0]}",
            use_container_width=True,
        )
    with col_b:
        st.image(
            to_display_uint8(result.preprocessed_image),
            caption=f"Model input — {result.preprocessed_image.shape[1]}px, padded",
            use_container_width=True,
        )
    with col_c:
        st.image(
            overlay,
            caption=f"Grad-CAM — attention for “{result.cam_class}”",
            use_container_width=True,
        )

    st.caption(
        "Warm regions are where the model's evidence for the selected class came from. "
        "A heatmap that lands on an implant, a collimation edge, or a burned-in L/R "
        "marker means the prediction is unreliable however confident it looks — this "
        "repository scores that failure mode explicitly via `scripts/gradcam_report.py`."
    )

# -- Exports -----------------------------------------------------------------
stem = Path(case_name).stem
export_a, export_b, export_c = st.columns(3)

with export_a:
    report_html = build_html_report(
        filename=case_name,
        verdict=result.label,
        confidence_pct=result.confidence,
        class_probabilities=result.class_probabilities,
        lesion_probability=result.lesion_probability,
        threshold=result.threshold,
        calibrated=result.calibrated,
        temperature=result.temperature,
        inconclusive=result.inconclusive,
        max_probability=result.max_probability,
        predictive_entropy=result.predictive_entropy,
        checkpoint_name=selected.parent.name,
        app_version=__version__,
        disclaimer=MEDICAL_DISCLAIMER,
        original_png=png_bytes(to_display_uint8(result.original_image)),
        overlay_png=png_bytes(overlay) if overlay is not None else None,
        cam_class=result.cam_class,
    )
    st.download_button(
        "Report (HTML, print to PDF)", report_html,
        file_name=f"{stem}_onnm_report.html", mime="text/html",
        use_container_width=True,
        help="Self-contained case report: verdict, probabilities, Grad-CAM overlay, "
             "and the medical disclaimer. Open it in any browser and print to PDF.",
    )
with export_b:
    if overlay is not None:
        st.download_button(
            "Overlay (PNG)", png_bytes(overlay),
            file_name=f"{stem}_gradcam.png", mime="image/png",
            use_container_width=True,
        )
with export_c:
    st.download_button(
        "Result (JSON)",
        json.dumps(result.as_dict(), indent=2),
        file_name=f"{stem}_onnm.json", mime="application/json",
        use_container_width=True,
    )

with st.expander("Decoding details"):
    st.json(result.source_meta)
    st.caption(
        "`inverted` marks a MONOCHROME1 DICOM that was flipped back to a positive image; "
        "left unhandled it would have reached the model as a photographic negative. "
        "`voi_lut_applied` marks the DICOM window/level having been honoured."
    )

st.divider()
st.caption(
    "BTXRD is licensed CC BY-NC-ND 4.0 — NoDerivatives covers Grad-CAM overlays, so keep "
    "downloaded images local rather than redistributing them."
)
render_legal_footer()

