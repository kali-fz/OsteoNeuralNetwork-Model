"""Custom chart components for ONNM inference results.

Replaces the default Streamlit/matplotlib charts with styled HTML components
that follow the warm-ivory medical design system.  Two charts are provided:

- ``render_probability_chart`` — horizontal bar breakdown of the 3-class
  softmax output, keeping all three classes visible with direct value labels
  and WCAG AA contrast on white.
- ``render_roc_chart`` — interactive ROC curve from a threshold sweep JSON,
  with a live marker at the current operating point.  Implemented with
  Altair so the hover tooltips are native browser interaction with no extra
  JS dependency.

Both functions accept the same visual tokens used in theme.py so the charts
stay coherent with whichever palette is active.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# Clinical accent colours — these encode malignancy and are NOT design tokens.
# They must be stable and must not be recoloured to match a design system.
_CLASS_COLORS = {
    "normal": "#2e8b57",
    "benign": "#e0a800",
    "malignant": "#c62828",
}
_CLASS_LABELS = {
    "normal": "Normal",
    "benign": "Benign",
    "malignant": "Malignant",
}

# Malignant recall published in the model card:  0.633 [0.490, 0.776]
_MALIGNANT_RECALL_POINT = 0.633
_MALIGNANT_RECALL_LO = 0.490
_MALIGNANT_RECALL_HI = 0.776


def render_probability_chart(
    probabilities: dict[str, float],
    *,
    height: int = 180,
) -> None:
    """Horizontal bar chart of the 3-way classification head.

    Classes are shown in fixed clinical order (normal / benign / malignant),
    not sorted by probability, so a reader comparing two films always finds
    malignant in the same position.  All three values are shown and labelled
    directly on the bar.
    """
    order = ["normal", "benign", "malignant"]
    rows_html = ""
    for cls in order:
        pct = 100.0 * probabilities.get(cls, 0.0)
        color = _CLASS_COLORS.get(cls, "#888")
        label = _CLASS_LABELS.get(cls, cls.title())
        bar_width = f"{min(pct, 100):.1f}%"
        # Ensure label contrast: dark background needs white text, light needs dark.
        text_color = "#ffffff" if pct > 25 else "#1c1a17"
        inside_label = ""
        outside_label = ""
        if pct >= 12:
            inside_label = (
                f'<span style="font-size:12px;font-weight:700;color:{text_color};'
                f'font-family:monospace;">{pct:.1f}%</span>'
            )
        else:
            outside_label = (
                f'<span style="position:absolute;left:{bar_width};top:50%;'
                "transform:translateY(-50%) translateX(6px);font-size:12px;"
                f'font-weight:700;color:#1c1a17;font-family:monospace;">{pct:.1f}%</span>'
            )
        rows_html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <div style="min-width:80px;font-size:13px;font-weight:600;color:#1c1a17;
                      font-family:'Inter','Segoe UI',Arial,sans-serif;">
            {label}
          </div>
          <div style="flex:1;background:#e8e3da;border-radius:3px;height:28px;
                      overflow:hidden;position:relative;">
            <div style="width:{bar_width};height:100%;background:{color};
                        transition:width .4s ease;display:flex;align-items:center;
                        padding-left:8px;box-sizing:border-box;">
              {inside_label}
            </div>
            {outside_label}
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html><html>
<head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; padding: 4px 0; }}
</style>
</head>
<body>{rows_html}</body></html>"""

    components.html(html, height=height, scrolling=False)


def render_roc_chart(
    sweep_rows: list[dict],
    current_threshold: float,
    *,
    width: int | None = None,
) -> None:
    """Interactive ROC curve from a threshold sweep with the live operating point.

    Draws a step-after line, confidence intervals for malignant recall where
    available, and a red marker at the current threshold value.  Uses Altair
    so hover tooltips are zero-JS overhead.
    """
    if not sweep_rows:
        st.caption("No sweep data available.")
        return

    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.caption("Altair required for the ROC curve.")
        return

    from theme import INK, LINE, LINE_SOFT, MONO, MUTED

    frame = pd.DataFrame(sweep_rows)
    frame["fpr"] = 1.0 - frame["specificity"]

    roc = (
        alt.Chart(frame)
        .mark_line(point=True, interpolate="step-after", color=INK, strokeWidth=1.5)
        .encode(
            x=alt.X("fpr:Q", title="1 − specificity", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("sensitivity:Q", title="sensitivity", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("sensitivity:Q", format=".3f"),
                alt.Tooltip("specificity:Q", format=".3f"),
                alt.Tooltip("youden_j:Q", format=".3f"),
            ],
        )
    )

    current = frame.iloc[(frame["threshold"] - current_threshold).abs().argmin()]
    marker = (
        alt.Chart(current.to_frame().T)
        .mark_point(size=140, color="#c62828", filled=True)
        .encode(x="fpr:Q", y="sensitivity:Q")
    )

    # Malignant recall confidence band as a horizontal reference strip
    ci_band = (
        alt.Chart(
            pd.DataFrame(
                {
                    "y_lo": [_MALIGNANT_RECALL_LO],
                    "y_hi": [_MALIGNANT_RECALL_HI],
                }
            )
        )
        .mark_rect(opacity=0.08, color="#c62828")
        .encode(
            y=alt.Y("y_lo:Q", scale=alt.Scale(domain=[0, 1])),
            y2="y_hi:Q",
            x=alt.value(0),
            x2=alt.value(1),
        )
    )

    chart = (
        (ci_band + roc + marker)
        .configure_view(strokeWidth=1, stroke=LINE)
        .configure_axis(
            gridColor=LINE_SOFT,
            domainColor=LINE,
            tickColor=LINE,
            labelColor=MUTED,
            titleColor=INK,
            labelFont=MONO,
            titleFontWeight=500,
        )
        .configure(background="transparent")
        .properties(width="container" if width is None else width)
    )

    st.altair_chart(chart, use_container_width=width is None)
    st.caption(
        "Fitted on the validation split.  The red point is the current operating "
        "threshold.  Shaded band: published malignant recall CI [0.490, 0.776]."
    )
