"""Git-Design visual system for the ONNM interface.

A single place for the palette, type scale, and component rules so that the
Streamlit app and the exported HTML case report look like one product. The
system is the editorial, code-native language described by the ``git-design``
skill: a pale gray canvas, near-black type, hairline rules, almost no rounding,
and exactly one saturated green reserved for the primary action.

Two deliberate departures from the skill, both forced by this project:

* **No webfont.** The skill asks for ``Mona Sans``. Fetching it from a CDN would
  be an outbound request on every page load, and this app promises the user that
  nothing leaves the machine. The skill's own fallback stack is used instead, so
  the layout is unchanged and the offline guarantee holds.
* **Clinical colour survives.** Green/amber/red on the class chart and the
  verdict card are not decoration -- they encode malignancy. The skill's "one
  green per viewport" rule is applied to *actions*; the diagnostic palette is
  left alone, because recolouring it to match a design system would make the
  interface lie.
"""

from __future__ import annotations

import html as _html

import streamlit as st

# Git-Design tokens. Mirrored in report.py so the exported report matches.
INK = "#000000"
BG = "#e9edec"
INSET = "#f2f5f3"
WHITE = "#ffffff"
MUTED = "#58635b"
LINE = "#b6bfb8"
LINE_SOFT = "#d2d9d4"
GREEN = "#08872b"
GREEN_HOVER = "#0d6731"
FOCUS = "#0377ff"

SANS = '"Mona Sans", "Avenir Next", "Segoe UI", Arial, sans-serif'
MONO = '"Mona Sans Mono", "SFMono-Regular", Consolas, monospace'

_CSS = """
:root {
  --gd-bg:#e9edec; --gd-inset:#f2f5f3; --gd-white:#fff; --gd-ink:#000;
  --gd-muted:#58635b; --gd-line:#b6bfb8; --gd-line-soft:#d2d9d4;
  --gd-green:#08872b; --gd-green-hover:#0d6731; --gd-focus:#0377ff;
  --gd-sans:"Mona Sans","Avenir Next","Segoe UI",Arial,sans-serif;
  --gd-mono:"Mona Sans Mono","SFMono-Regular",Consolas,monospace;
}

/* -- Canvas ------------------------------------------------------------- */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--gd-bg); color: var(--gd-ink);
}
[data-testid="stHeader"] { background: transparent; }
html, body, [data-testid="stAppViewContainer"] * { font-family: var(--gd-sans); }
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* -- Type scale --------------------------------------------------------- */
/* Display weight stays light for its size: 425-460, tight negative tracking. */
h1, h2, h3, h4 { color: var(--gd-ink); font-family: var(--gd-sans); }
h1 { font-size: clamp(40px,5vw,64px); font-weight: 425; letter-spacing:-.035em; line-height:1.02; }
h2 { font-size: clamp(28px,3vw,32px); font-weight: 440; letter-spacing:-.02em; line-height:1.2; }
h3 { font-size: 22px; font-weight: 460; letter-spacing:-.01em; }
[data-testid="stMarkdownContainer"] p { font-size: 16px; line-height: 1.5; }

/* -- Masthead ----------------------------------------------------------- */
.gd-masthead { border-bottom: 1px solid var(--gd-line); margin-bottom: 28px; }
.gd-eyebrow {
  margin: 0 0 14px; font-family: var(--gd-mono); font-size: 14px;
  font-weight: 500; line-height: 1.6; letter-spacing: .04em; color: var(--gd-ink);
}
.gd-wordmark {
  margin: 0; font-size: clamp(40px,6.2vw,84px); font-weight: 425;
  letter-spacing: -.045em; line-height: .92;
}
.gd-meta {
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  margin: 18px 0 0; padding: 14px 0 16px;
  border-top: 1px solid var(--gd-line-soft);
  font-family: var(--gd-mono); font-size: 14px; font-weight: 500;
  line-height: 1.5; color: var(--gd-muted);
}
.gd-meta span:not(.gd-signal) { color: var(--gd-ink); }
.gd-signal { width: 10px; height: 10px; background: var(--gd-green); flex: 0 0 10px; }

/* -- Verdict card ------------------------------------------------------- */
/* Square, ruled, mono metadata: the skill's panel geometry carrying the
   clinical accent colour rather than the design system's green. */
.gd-verdict {
  border: 1px solid var(--gd-line); border-left: 8px solid var(--accent);
  border-radius: 0; background: var(--gd-white);
  padding: 22px 26px; margin: 4px 0 22px;
}
.gd-verdict h2 {
  margin: 0; font-size: clamp(28px,3.4vw,40px); font-weight: 425;
  letter-spacing: -.02em; line-height: 1.1; color: var(--accent);
}
.gd-verdict p {
  margin: 10px 0 0; font-family: var(--gd-mono); font-size: 14px;
  font-weight: 500; color: var(--gd-muted);
}

/* -- Sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {
  background: var(--gd-inset); border-right: 1px solid var(--gd-line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] h2 {
  font-family: var(--gd-mono); font-size: 14px; font-weight: 500;
  letter-spacing: .06em; text-transform: uppercase; color: var(--gd-muted);
  padding-bottom: 8px; border-bottom: 1px solid var(--gd-line-soft);
}
[data-testid="stSidebar"] hr { border-top: 1px solid var(--gd-line-soft); }

/* -- Actions ------------------------------------------------------------ */
/* One green per viewport: the form submit is the primary action, every other
   button is the outlined variant. */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 0; border: 1px solid var(--gd-line); background: rgb(0 0 0 / .05);
  color: var(--gd-ink); font-weight: 550; min-height: 46px; box-shadow: none;
  transition: background-color .2s ease-in-out, border-color .2s ease-in-out;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color: var(--gd-ink); background: var(--gd-white); color: var(--gd-ink);
}
.stFormSubmitButton > button {
  min-height: 56px; border: 0; background: var(--gd-green); color: #fff;
}
.stFormSubmitButton > button:hover { background: var(--gd-green-hover); color: #fff; border: 0; }
.stButton > button:focus, .stDownloadButton > button:focus,
.stFormSubmitButton > button:focus { box-shadow: none; }
:where(a, button, [tabindex], summary, input, select):focus-visible {
  outline: 2px solid var(--gd-focus); outline-offset: 3px;
}

/* -- Inputs ------------------------------------------------------------- */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
  border-radius: 0 !important; background: var(--gd-white) !important;
  border-color: var(--gd-line) !important;
}
[data-testid="stFileUploader"] section {
  border: 1px solid var(--gd-line); border-radius: 0; background: var(--gd-inset);
  padding: 26px;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--gd-ink); background: var(--gd-white);
}
[data-testid="stWidgetLabel"] p { font-size: 15px; font-weight: 500; }
[data-testid="stSlider"] [role="slider"] { border-radius: 0; }

/* -- Tabs --------------------------------------------------------------- */
/* Active state is white, inactive is the soft inset -- per the skill. */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border: 1px solid var(--gd-line); background: var(--gd-inset);
}
.stTabs [data-baseweb="tab"] {
  min-height: 58px; border-radius: 0; border-right: 1px solid var(--gd-line);
  background: var(--gd-inset); color: var(--gd-ink); font-weight: 500; padding: 0 26px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: var(--gd-white); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
  background: transparent;
}

/* -- Panels, alerts, metrics -------------------------------------------- */
[data-testid="stExpander"] {
  border: 1px solid var(--gd-line); border-radius: 0; background: var(--gd-white);
}
[data-testid="stExpander"] summary { font-weight: 500; }
[data-testid="stAlert"] {
  border-radius: 0; border: 1px solid var(--gd-line); border-left-width: 6px;
  background: var(--gd-white); color: var(--gd-ink);
}
[data-testid="stMetric"] {
  border: 1px solid var(--gd-line-soft); border-radius: 0;
  background: var(--gd-white); padding: 16px 18px;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--gd-mono); font-size: 13px; font-weight: 500;
  letter-spacing: .04em; text-transform: uppercase; color: var(--gd-muted);
}
[data-testid="stMetricValue"] { font-weight: 425; letter-spacing: -.02em; }

/* -- Mono for operational text: labels, ticks, filenames, code ---------- */
code, kbd, pre, [data-testid="stJson"], [data-testid="stDataFrame"] {
  font-family: var(--gd-mono) !important;
}
code {
  background: var(--gd-inset); border: 1px solid var(--gd-line-soft);
  border-radius: 0; padding: 1px 5px; color: var(--gd-ink); font-size: 90%;
}
[data-testid="stCaptionContainer"] p {
  font-family: var(--gd-mono); font-size: 13px; line-height: 1.55; color: var(--gd-muted);
}
[data-testid="stDataFrame"] { border: 1px solid var(--gd-line); border-radius: 0; }
[data-testid="stImage"] img { border: 1px solid var(--gd-line-soft); border-radius: 0; }
[data-testid="stImageCaption"] {
  font-family: var(--gd-mono); font-size: 12.5px; color: var(--gd-muted);
}
hr { border-top: 1px solid var(--gd-line-soft); }

/* -- Motion ------------------------------------------------------------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important; animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
  }
}
"""


def inject_theme() -> None:
    """Apply the Git-Design system. Call once, immediately after ``set_page_config``."""
    # Concatenated rather than interpolated: the CSS is full of braces.
    st.markdown("<style>" + _CSS + "</style>", unsafe_allow_html=True)


def masthead(title: str, eyebrow: str, meta: list[str]) -> None:
    """Full-width wordmark band with a mono eyebrow, meta row, and bottom rule.

    Replaces ``st.title``. ``meta`` items render as monospaced metadata closed by
    the 10px green signal square the skill uses to end a meta row.
    """
    cells = "".join(f"<span>{_html.escape(item)}</span>" for item in meta)
    st.markdown(
        f"""
        <div class="gd-masthead">
          <p class="gd-eyebrow">{_html.escape(eyebrow)}</p>
          <h1 class="gd-wordmark">{_html.escape(title)}</h1>
          <div class="gd-meta">{cells}<span class="gd-signal"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_card(label: str, detail: str, accent: str) -> None:
    """Square, ruled verdict panel carrying the clinical accent on its left edge."""
    st.markdown(
        f"""
        <div class="gd-verdict" style="--accent:{accent};">
          <h2>{_html.escape(label)}</h2>
          <p>{_html.escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
