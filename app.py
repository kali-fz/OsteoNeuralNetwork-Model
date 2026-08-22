"""ONNM — local Streamlit interface for bone-lesion triage on plain radiographs.

    streamlit run app.py

Runs entirely on this machine: the model, the Grad-CAM, and the server itself.
Nothing is uploaded anywhere, no external API is called, and the whole stack is
free and open source. Uploaded files are held in memory and written only to a
temporary file that is deleted before the prediction returns.

All model work lives in ``onnm.inference``. This file is presentation only, and
should stay that way -- if a computation needs to move here to make the layout
work, that is a signal the inference API is missing something.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Importable straight from a clone, the same trick scripts/_bootstrap.py uses.
SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onnm import __version__  # noqa: E402
from onnm.inference import (  # noqa: E402
    UPLOAD_TYPES,
    InferenceResult,
    RadiographClassifier,
    find_checkpoints,
    render_overlay,
    to_display_uint8,
)
from onnm.io_radiograph import RadiographReadError  # noqa: E402
from onnm.utils import describe_device  # noqa: E402

COLORMAPS = ["jet", "turbo", "inferno", "magma", "viridis", "hot"]

# Colour per class, reused by the bar chart and the verdict card so that
# "malignant" means the same red everywhere in the UI.
CLASS_COLORS = {
    "normal": "#2e8b57",
    "benign": "#e0a800",
    "malignant": "#c62828",
}

DISCLAIMER = """
**Research tool — not a medical device, and not medical advice.**
This is a free, offline, open-source research prototype running a model on your own
machine. It has not been clinically validated, carries no regulatory clearance
(FDA / CE / MHRA), and its outputs are **not** a diagnosis. It must not be used to
make, support, defer, or delay any clinical decision. Every radiograph requires
interpretation by a qualified radiologist or treating clinician.
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

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#d0d0d0"})
    names = list(probabilities)
    values = [100.0 * probabilities[n] for n in names]
    colors = [CLASS_COLORS.get(n, "#4c72b0") for n in names]

    fig, ax = plt.subplots(figsize=(6, 1.9), dpi=140)
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability (%)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", visible=False)

    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(
            min(value + 1.5, 92), bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=9, fontweight="bold",
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ONNM — Bone Lesion Triage",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      .verdict {
        border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 0.6rem;
        border-left: 8px solid var(--accent); background: var(--bg);
      }
      .verdict h2 { margin: 0; font-size: 1.55rem; color: var(--accent); }
      .verdict p  { margin: 0.25rem 0 0; font-size: 0.92rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🦴 OsteoNeuralNetwork-Model")
st.caption(
    f"Explainable bone-tumour triage on plain radiographs · v{__version__} · "
    "runs locally, offline, at zero cost"
)
st.error(DISCLAIMER)

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Model")

    checkpoints = find_checkpoints()
    if not checkpoints:
        st.error("No checkpoint found under `reports/`.")
        st.code(
            "python scripts/train.py --override configs/densenet121_3class.yaml",
            language="bash",
        )
        st.stop()

    selected = st.selectbox(
        "Checkpoint",
        options=checkpoints,
        format_func=lambda p: p.parent.name,
        help="Every `reports/*/best.pt`, newest first. Preprocessing is read from "
             "inside the checkpoint, so each one runs under its own training config.",
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
        "No data leaves this machine. Uploads live in memory and in one temporary "
        "file that is deleted before the result is shown."
    )

# -- Upload ----------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a radiograph",
    type=UPLOAD_TYPES,
    help="DICOM (.dcm/.dicom/.ima), PNG, JPEG, BMP or TIFF. DICOM headers are honoured: "
         "modality LUT, VOI window, and MONOCHROME1 inversion are all applied.",
)

if uploaded is None:
    st.info(
        "Upload an X-ray to begin. The model classifies the film, reports a confidence "
        "score, and renders a Grad-CAM heatmap showing which region drove the call."
    )
    st.stop()

payload = uploaded.getvalue()
# Re-running inference is cheap after warmup, so the cache key stays simple and
# the result has exactly one source of truth rather than a locally re-derived
# verdict that could drift from the one the model actually produced.
cache_key = (uploaded.name, len(payload), hash(payload), str(selected), threshold, cam_class)

if st.session_state.get("cache_key") != cache_key:
    try:
        with st.spinner("Running inference…"):
            st.session_state["result"] = classifier.predict(
                payload,
                filename=uploaded.name,
                with_heatmap=True,
                threshold=threshold,
                cam_class=cam_class,
            )
        st.session_state["cache_key"] = cache_key
    except RadiographReadError as exc:
        st.error(f"Could not decode `{uploaded.name}`: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error("Inference failed.")
        st.exception(exc)
        st.stop()

result: InferenceResult = st.session_state["result"]

# -- Verdict ---------------------------------------------------------------
accent = "#c62828" if result.is_lesion else "#2e8b57"
background = "rgba(198,40,40,0.08)" if result.is_lesion else "rgba(46,139,87,0.08)"

st.markdown(
    f"""
    <div class="verdict" style="--accent:{accent}; --bg:{background};">
      <h2>{result.label}</h2>
      <p>{result.confidence:.1f}% confidence · decided at a {result.threshold:.2f}
         lesion threshold</p>
    </div>
    """,
    unsafe_allow_html=True,
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
st.subheader("Grad-CAM")

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

    download_left, download_right = st.columns(2)
    stem = Path(uploaded.name).stem
    with download_left:
        st.download_button(
            "Download overlay (PNG)", png_bytes(overlay),
            file_name=f"{stem}_gradcam.png", mime="image/png",
            use_container_width=True,
        )
    with download_right:
        st.download_button(
            "Download result (JSON)",
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
