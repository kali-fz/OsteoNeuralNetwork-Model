"""ONNM warm-ivory visual system.

A single place for the palette, type scale, and component rules so that the
Streamlit app and the exported HTML case report look like one product.

Design direction (section 0 of REDESIGN_BRIEF):
  warm ivory / organic living-systems / doctoral medical science.
  Sterile white for functional task surfaces; warm ivory for presentation.
  No cyberpunk, no wellness vibes, no generic "AI healthcare" blue.

Two invariants that survive this redesign:
  * **Clinical colour is sacred.** Green/amber/red on the class chart and the
    verdict card encode malignancy, not branding.  They must not be recoloured.
  * **Inter is permitted (section 3D: Google Fonts are explicitly allowed).**
    Robust system-font fallbacks ensure the page remains usable when the font
    request is blocked or slow.
"""

from __future__ import annotations

import html as _html

import streamlit as st

# ── Palette ─────────────────────────────────────────────────────────────────
# Warm-ivory presentation layer (landing hero, page background).
BG    = "#f7f4ef"
INSET = "#ffffff"   # sterile white for task surfaces

# Typography and structure.
INK       = "#1c1a17"  # warm near-black
WHITE     = "#ffffff"
MUTED     = "#6b6457"  # warm brown-muted
LINE      = "#ddd8ce"  # warm hairline border
LINE_SOFT = "#e8e3da"  # softer rule for less prominent dividers

# Primary action — clinical forest green.
GREEN       = "#2e6b47"
GREEN_HOVER = "#235335"
FOCUS       = "#1a56db"

# ── Type ────────────────────────────────────────────────────────────────────
SANS = '"Inter","Segoe UI","Helvetica Neue",Arial,sans-serif'
MONO = '"JetBrains Mono","SFMono-Regular","SF Mono",Consolas,monospace'


# ── Google Fonts (permitted by section 3D; system fallbacks ensure graceful
#    degradation when the request is blocked or the app runs fully offline) ──
_GOOGLE_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@300;400;500;600;700&amp;'
    'family=JetBrains+Mono:wght@400;500&amp;'
    'display=swap" rel="stylesheet">'
)

_CSS = """
:root {
  --bg:       #f7f4ef;
  --inset:    #ffffff;
  --white:    #ffffff;
  --ink:      #1c1a17;
  --muted:    #6b6457;
  --line:     #ddd8ce;
  --line-s:   #e8e3da;
  --green:    #2e6b47;
  --green-hv: #235335;
  --focus:    #1a56db;
  --sans:     "Inter","Segoe UI","Helvetica Neue",Arial,sans-serif;
  --mono:     "JetBrains Mono","SFMono-Regular","SF Mono",Consolas,monospace;
}

/* ── Canvas ──────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg); color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
html, body, [data-testid="stAppViewContainer"] * { font-family: var(--sans); }
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* ── Type scale ──────────────────────────────────────────────────────────── */
h1, h2, h3, h4 { color: var(--ink); font-family: var(--sans); }
h1 { font-size:clamp(40px,5vw,64px); font-weight:300; letter-spacing:-.035em; line-height:1.02; }
h2 { font-size:clamp(26px,3vw,32px); font-weight:400; letter-spacing:-.02em;  line-height:1.2; }
h3 { font-size:21px; font-weight:500; letter-spacing:-.01em; }
[data-testid="stMarkdownContainer"] p { font-size:16px; line-height:1.6; color:var(--ink); }

/* ── Landing nav ─────────────────────────────────────────────────────────── */
.onnm-nav {
  display:flex; align-items:center; justify-content:space-between;
  padding:20px 0 16px; border-bottom:1px solid var(--line);
  margin-bottom:0;
}
.onnm-nav-brand {
  font-family:var(--mono); font-size:13px; font-weight:500;
  letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
}
.onnm-nav-actions { display:flex; gap:10px; }
.onnm-nav-btn {
  display:inline-block; padding:9px 22px; font-size:14px; font-weight:500;
  border:1px solid var(--line); background:var(--white); color:var(--ink);
  cursor:pointer; text-decoration:none; font-family:var(--sans);
  transition:border-color .18s, background .18s;
}
.onnm-nav-btn:hover { border-color:var(--ink); background:var(--bg); }
.onnm-nav-btn.primary {
  background:var(--green); color:#fff; border-color:var(--green);
}
.onnm-nav-btn.primary:hover { background:var(--green-hv); border-color:var(--green-hv); }

/* ── Hero ────────────────────────────────────────────────────────────────── */
.onnm-hero {
  position:relative; overflow:hidden;
  min-height:480px; padding:56px 0 0; margin:0 -1rem;
}
.onnm-hero-bg {
  position:absolute; inset:0; z-index:0;
  background-image:var(--hero-img);
  background-size:cover; background-position:center bottom;
}
.onnm-hero-veil {
  /* warm gradient veil over the left side so text is always legible */
  position:absolute; inset:0; z-index:1;
  background: linear-gradient(
    to right,
    rgba(247,244,239,.95) 0%,
    rgba(247,244,239,.72) 46%,
    rgba(247,244,239,.15) 66%,
    rgba(247,244,239,.00) 80%
  );
}
.onnm-hero-content { position:relative; z-index:2; padding:0 2rem 56px; }
.onnm-hero-eyebrow {
  font-family:var(--mono); font-size:12px; font-weight:500;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  margin:0 0 20px;
}
.onnm-hero-title {
  font-size:clamp(38px,5.5vw,72px); font-weight:300; line-height:1.01;
  letter-spacing:-.045em; color:var(--ink); margin:0 0 18px;
}
.onnm-hero-subtitle {
  font-size:clamp(16px,1.6vw,20px); color:var(--muted); font-weight:400;
  line-height:1.55; margin:0 0 36px; max-width:420px;
}
.onnm-hero-cta {
  display:inline-block; padding:14px 34px; background:var(--green); color:#fff;
  font-size:15px; font-weight:500; text-decoration:none; font-family:var(--sans);
  border:none; cursor:pointer;
  transition:background .18s;
}
.onnm-hero-cta:hover { background:var(--green-hv); }
/* Globe sits above the moss on the right — the iframe inherits the z-index
   context from its parent column, so no explicit z-index needed here. */
.onnm-globe-wrap { padding-top:20px; }

/* ── Stats row ───────────────────────────────────────────────────────────── */
.onnm-stats {
  display:flex; flex-wrap:wrap; gap:0;
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  margin:36px 0;
}
.onnm-stat {
  flex:1; min-width:140px; padding:22px 28px;
  border-right:1px solid var(--line);
}
.onnm-stat:last-child { border-right:none; }
.onnm-stat-value {
  font-size:clamp(28px,3.5vw,40px); font-weight:300; letter-spacing:-.03em;
  color:var(--ink); line-height:1; display:block; margin-bottom:6px;
}
.onnm-stat-label {
  font-family:var(--mono); font-size:12px; font-weight:500;
  letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
}
.onnm-stat a { text-decoration:none; color:inherit; }
.onnm-stat a:hover .onnm-stat-value { color:var(--green); }

/* ── Metrics ─────────────────────────────────────────────────────────────── */
.onnm-metric-band {
  background:var(--white); border:1px solid var(--line);
  padding:36px 36px 28px; margin:0 0 36px;
}
.onnm-metric-band h2 {
  font-size:14px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin:0 0 22px; font-family:var(--mono);
}
.onnm-metric-grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:20px;
}
.onnm-metric-item {}
.onnm-metric-num {
  font-size:32px; font-weight:300; letter-spacing:-.03em;
  color:var(--ink); line-height:1; display:block;
}
.onnm-metric-ci {
  font-family:var(--mono); font-size:11px; color:var(--muted);
  margin:2px 0 6px; display:block;
}
.onnm-metric-desc {
  font-size:13px; color:var(--muted); line-height:1.5;
}

/* ── Scanner nav bar (authenticated pages) ───────────────────────────────── */
.onnm-appbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 0 14px; border-bottom:1px solid var(--line);
  margin-bottom:24px;
}
.onnm-appbar-brand {
  font-family:var(--mono); font-size:13px; font-weight:500;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink);
}
.onnm-appbar-links { display:flex; gap:6px; align-items:center; }
.onnm-appbar-link {
  padding:7px 16px; font-size:13px; font-weight:500; color:var(--muted);
  background:none; border:none; cursor:pointer; font-family:var(--sans);
  transition:color .15s;
}
.onnm-appbar-link:hover { color:var(--ink); }
.onnm-appbar-link.active { color:var(--ink); border-bottom:2px solid var(--green); }
.onnm-appbar-email {
  font-family:var(--mono); font-size:12px; color:var(--muted);
  padding:7px 12px; border-left:1px solid var(--line);
}

/* ── Profile page ────────────────────────────────────────────────────────── */
.onnm-profile-card {
  background:var(--white); border:1px solid var(--line);
  padding:28px 32px; margin-bottom:24px;
}
.onnm-profile-card h3 {
  font-size:12px; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-family:var(--mono);
  margin:0 0 16px;
}
.onnm-profile-row {
  display:flex; align-items:baseline; gap:16px; padding:8px 0;
  border-bottom:1px solid var(--line-s);
}
.onnm-profile-row:last-child { border-bottom:none; }
.onnm-profile-key {
  font-family:var(--mono); font-size:12px; font-weight:500;
  color:var(--muted); min-width:160px;
}
.onnm-profile-val { font-size:15px; color:var(--ink); }

/* ── Masthead (scanner / internal pages) ─────────────────────────────────── */
.gd-masthead { border-bottom:1px solid var(--line); margin-bottom:24px; padding-bottom:20px; }
.gd-eyebrow {
  margin:0 0 10px; font-family:var(--mono); font-size:12px;
  font-weight:500; line-height:1.6; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted);
}
.gd-wordmark {
  margin:0; font-size:clamp(32px,5vw,60px); font-weight:300;
  letter-spacing:-.04em; line-height:.96;
}
.gd-meta {
  display:flex; flex-wrap:wrap; align-items:center; gap:12px;
  margin:16px 0 0; padding:12px 0 0;
  border-top:1px solid var(--line-s);
  font-family:var(--mono); font-size:13px; font-weight:500;
  line-height:1.5; color:var(--muted);
}
.gd-meta span:not(.gd-signal) { color:var(--ink); }
.gd-signal {
  width:9px; height:9px; background:var(--green); flex:0 0 9px;
  border-radius:50%;
}

/* ── Verdict card ─────────────────────────────────────────────────────────── */
.gd-verdict {
  border:1px solid var(--line); border-left:7px solid var(--accent);
  background:var(--white); padding:22px 26px; margin:4px 0 22px;
}
.gd-verdict h2 {
  margin:0; font-size:clamp(26px,3.4vw,38px); font-weight:300;
  letter-spacing:-.02em; line-height:1.1; color:var(--accent);
}
.gd-verdict p {
  margin:10px 0 0; font-family:var(--mono); font-size:13px;
  font-weight:500; color:var(--muted);
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background:var(--white); border-right:1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
[data-testid="stSidebar"] h2 {
  font-family:var(--mono); font-size:12px; font-weight:500;
  letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
  padding-bottom:8px; border-bottom:1px solid var(--line-s);
}
[data-testid="stSidebar"] hr { border-top:1px solid var(--line-s); }

/* ── Actions ─────────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius:0; border:1px solid var(--line); background:rgba(0,0,0,.04);
  color:var(--ink); font-weight:500; min-height:44px; box-shadow:none;
  transition:background .2s, border-color .2s;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color:var(--ink); background:var(--white); color:var(--ink);
}
.stFormSubmitButton > button {
  min-height:52px; border:0; background:var(--green); color:#fff;
}
.stFormSubmitButton > button:hover { background:var(--green-hv); border:0; }
.stButton > button:focus, .stDownloadButton > button:focus,
.stFormSubmitButton > button:focus { box-shadow:none; }
:where(a,button,[tabindex],summary,input,select):focus-visible {
  outline:2px solid var(--focus); outline-offset:3px;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-baseweb="input"],[data-baseweb="select"] > div,[data-baseweb="base-input"] {
  border-radius:0 !important; background:var(--white) !important;
  border-color:var(--line) !important;
}
[data-testid="stFileUploader"] section {
  border:1px solid var(--line); background:var(--bg); padding:26px;
}
[data-testid="stFileUploader"] section:hover {
  border-color:var(--ink); background:var(--white);
}
[data-testid="stWidgetLabel"] p { font-size:14px; font-weight:500; }
[data-testid="stSlider"] [role="slider"] { border-radius:0; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap:0; border:1px solid var(--line); background:var(--bg);
}
.stTabs [data-baseweb="tab"] {
  min-height:52px; border-right:1px solid var(--line);
  background:var(--bg); color:var(--ink); font-weight:500; padding:0 24px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:var(--white); }
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {
  background:transparent;
}

/* ── Panels, alerts, metrics ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border:1px solid var(--line); background:var(--white);
}
[data-testid="stExpander"] summary { font-weight:500; }
[data-testid="stAlert"] {
  border:1px solid var(--line); border-left-width:5px;
  background:var(--white); color:var(--ink);
}
[data-testid="stMetric"] {
  border:1px solid var(--line-s); background:var(--white); padding:16px 18px;
}
[data-testid="stMetricLabel"] p {
  font-family:var(--mono); font-size:12px; font-weight:500;
  letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
}
[data-testid="stMetricValue"] { font-weight:300; letter-spacing:-.02em; }

/* ── Mono operational text ───────────────────────────────────────────────── */
code,kbd,pre,[data-testid="stJson"],[data-testid="stDataFrame"] {
  font-family:var(--mono) !important;
}
code {
  background:var(--bg); border:1px solid var(--line-s);
  padding:1px 5px; color:var(--ink); font-size:90%;
}
[data-testid="stCaptionContainer"] p {
  font-family:var(--mono); font-size:12.5px; line-height:1.55; color:var(--muted);
}
[data-testid="stDataFrame"] { border:1px solid var(--line); }
[data-testid="stImage"] img { border:1px solid var(--line-s); }
[data-testid="stImageCaption"] {
  font-family:var(--mono); font-size:12px; color:var(--muted);
}
hr { border-top:1px solid var(--line-s); }

/* ── Footer ──────────────────────────────────────────────────────────────── */
.onnm-footer {
  border-top:1px solid var(--line); padding:24px 0; margin-top:48px;
  font-family:var(--mono); font-size:12px; color:var(--muted);
  display:flex; flex-wrap:wrap; gap:12px; align-items:center;
}
.onnm-footer a { color:var(--muted); text-decoration:underline; }
.onnm-footer a:hover { color:var(--ink); }
.onnm-footer-disclaimer {
  flex:1 1 100%; font-size:11.5px; line-height:1.6;
  color:var(--muted); border-top:1px solid var(--line-s); padding-top:14px; margin-top:8px;
}

/* ── Motion ──────────────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    transition-duration:.01ms !important;
  }
}

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .onnm-hero-title  { font-size: clamp(32px,8vw,52px); }
  .onnm-hero-subtitle { max-width: 100%; }
  .onnm-stats       { flex-direction: column; }
  .onnm-stat        { border-right: none; border-bottom: 1px solid var(--line); }
  .onnm-stat:last-child { border-bottom: none; }
  .onnm-metric-grid { grid-template-columns: 1fr 1fr; }
  .onnm-appbar-email { display: none; }
}
@media (max-width: 480px) {
  .onnm-metric-grid { grid-template-columns: 1fr; }
  .onnm-hero-content { padding: 0 1rem 40px; }
}
"""


def inject_theme() -> None:
    """Apply the warm-ivory design system.

    Call once, immediately after ``st.set_page_config``.  Injects the Google
    Fonts link (Inter + JetBrains Mono) with system-font fallbacks, then the
    full design-system CSS.
    """
    # Two separate calls, deliberately.  Streamlit renders markdown before the
    # raw HTML reaches the browser, and CommonMark treats a leading <link> as a
    # type-6 HTML block, which *terminates at the first blank line*.  Prepending
    # the font links to the stylesheet therefore truncated the CSS at its first
    # blank line and dumped the remainder onto the page as visible text.  A
    # string that starts with <style> is a type-1 HTML block, which runs to
    # </style> and ignores blank lines entirely.
    st.markdown(_GOOGLE_FONTS_LINK, unsafe_allow_html=True)
    st.markdown("<style>" + _CSS + "</style>", unsafe_allow_html=True)


def masthead(title: str, eyebrow: str, meta: list[str]) -> None:
    """Full-width wordmark band for authenticated/scanner pages."""
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
    """Square ruled verdict panel with clinical accent on its left edge."""
    st.markdown(
        f"""
        <div class="gd-verdict" style="--accent:{_html.escape(accent)};">
          <h2>{_html.escape(label)}</h2>
          <p>{_html.escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
