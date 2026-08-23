"""Self-contained HTML case reports for the Streamlit app.

One prediction -> one portable HTML file: verdict, three-class probabilities,
the Grad-CAM overlay, decoding/calibration provenance, and the full medical
disclaimer. Images are embedded as base64 data URIs so the file has no
external dependencies and survives being emailed or archived. The stylesheet
includes print rules, so "print to PDF" from any browser produces the PDF
version — no PDF library, no new dependency.

Torch-free by design so it is testable on any interpreter.
"""

from __future__ import annotations

import base64
import html
from datetime import UTC, datetime

__all__ = ["build_html_report"]

_CLASS_COLORS = {
    "normal": "#2e8b57",
    "benign": "#e0a800",
    "malignant": "#c62828",
}

# Warm-ivory tokens, mirrored from theme.py.  Duplicated (not imported) so
# this module stays Streamlit-free and torch-free.  The report background is
# print-safe white; warm-ivory is for presentation layers only.
_STYLE = """
  :root {
    --bg:#ffffff; --inset:#f7f4ef; --white:#ffffff; --ink:#1c1a17;
    --muted:#6b6457; --line:#ddd8ce; --line-s:#e8e3da; --green:#2e6b47;
    --sans:"Inter","Segoe UI","Helvetica Neue",Arial,sans-serif;
    --mono:"JetBrains Mono","SFMono-Regular","SF Mono",Consolas,monospace;
  }
  * { box-sizing: border-box; }
  body { font-family: var(--sans); color: var(--ink); margin: 0;
         background: var(--bg); font-size: 16px; line-height: 1.5; }
  .page { max-width: 900px; margin: 0 auto; background: var(--white);
          padding: 40px 48px; border-inline: 1px solid var(--line-s); }
  h1 { font-size: 2.1rem; font-weight: 300; letter-spacing: -.035em;
       line-height: 1.05; margin: 0 0 14px; }
  h2 { font-size: 1.2rem; font-weight: 500; letter-spacing: -.01em;
       margin: 32px 0 10px; border-bottom: 1px solid var(--line);
       padding-bottom: 8px; }
  .meta { font-family: var(--mono); font-size: 0.80rem; font-weight: 500;
          color: var(--muted); margin: 0 0 22px; padding-bottom: 14px;
          border-bottom: 1px solid var(--line); }
  .signal { display: inline-block; width: 9px; height: 9px;
            background: var(--green); border-radius: 50%;
            vertical-align: middle; margin-left: 4px; }
  .verdict { border: 1px solid var(--line); border-left: 7px solid var(--accent);
             background: var(--white); padding: 20px 24px;
             margin: 22px 0; }
  .verdict h2 { border: none; margin: 0; padding: 0; font-size: 1.7rem;
                font-weight: 300; letter-spacing: -.025em; color: var(--accent); }
  .verdict p { margin: 8px 0 0; font-family: var(--mono); font-size: 0.80rem;
               font-weight: 500; color: var(--muted); }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  td, th { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line-s); }
  th { font-family: var(--mono); font-size: 0.76rem; font-weight: 500;
       letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
  .bar { height: 10px; display: inline-block; vertical-align: middle; }
  .images { display: flex; gap: 16px; flex-wrap: wrap; }
  .images figure { margin: 0; flex: 1 1 260px; }
  .images img { width: 100%; border: 1px solid var(--line); }
  .images figcaption { font-family: var(--mono); font-size: 0.74rem;
                       color: var(--muted); margin-top: 6px; }
  code { font-family: var(--mono); background: var(--inset);
         border: 1px solid var(--line-s); padding: 1px 5px; font-size: 90%; }
  .disclaimer { background: var(--inset); border: 1px solid var(--line);
                border-left: 5px solid #c62828;
                padding: 16px 20px; font-size: 0.82rem; white-space: pre-wrap; }
  .footnote { font-family: var(--mono); color: var(--muted);
              font-size: 0.70rem; margin-top: 26px; padding-top: 14px;
              border-top: 1px solid var(--line-s); }
  @media print {
    body { background: #fff; }
    .page { max-width: none; padding: 0; border-inline: 0; }
    .images figure { page-break-inside: avoid; }
  }
"""


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _probability_rows(class_probabilities: dict[str, float]) -> str:
    rows = []
    for name, value in class_probabilities.items():
        pct = 100.0 * float(value)
        color = _CLASS_COLORS.get(name, "#4c72b0")
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td style='width:55%'><span class='bar' "
            f"style='width:{pct:.1f}%;background:{color}'></span></td>"
            f"<td>{pct:.1f}%</td></tr>"
        )
    return "".join(rows)


def build_html_report(
    *,
    filename: str,
    verdict: str,
    confidence_pct: float,
    class_probabilities: dict[str, float],
    lesion_probability: float,
    threshold: float,
    calibrated: bool,
    temperature: float,
    inconclusive: bool,
    max_probability: float,
    predictive_entropy: float,
    checkpoint_name: str,
    app_version: str,
    disclaimer: str,
    original_png: bytes | None = None,
    overlay_png: bytes | None = None,
    cam_class: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render one case as a standalone HTML document (a string).

    ``disclaimer`` is passed in rather than imported so this module has no
    dependency on the app layer; callers pass ``legal.MEDICAL_DISCLAIMER``.
    """
    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")

    if inconclusive:
        accent = "#b26a00"
    elif verdict.lower().startswith("normal"):
        accent = "#2e8b57"
    else:
        accent = "#c62828"

    calibration_note = (
        f"temperature-scaled (T={temperature:.2f}), threshold {threshold:.3f} "
        "fitted on the validation split"
        if calibrated
        else f"UNCALIBRATED — raw softmax, naive threshold {threshold:.3f}"
    )

    uncertainty_note = ""
    if inconclusive:
        uncertainty_note = (
            "<p><strong>Uncertainty gate:</strong> the model's probabilities were too "
            f"uncertain to present as a finding (max class probability "
            f"{100 * max_probability:.1f}%, normalized entropy {predictive_entropy:.2f}). "
            "This often indicates an out-of-domain or non-diagnostic image.</p>"
        )

    figures = []
    if original_png is not None:
        figures.append(
            f"<figure><img src='{_data_uri(original_png)}' alt='uploaded radiograph'/>"
            "<figcaption>De-identified uploaded image</figcaption></figure>"
        )
    if overlay_png is not None:
        target = html.escape(cam_class or "predicted class")
        figures.append(
            f"<figure><img src='{_data_uri(overlay_png)}' alt='Grad-CAM overlay'/>"
            f"<figcaption>Grad-CAM attention for \u201c{target}\u201d — warm regions "
            "drove the prediction. Attention on implants, collimation edges, or "
            "burned-in markers makes the prediction unreliable.</figcaption></figure>"
        )
    images_block = (
        f"<h2>Imagery</h2><div class='images'>{''.join(figures)}</div>" if figures else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ONNM case report — {html.escape(filename)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="page">
  <h1>OsteoNeuralNetwork-Model — Case Report</h1>
  <div class="meta">
    file: <strong>{html.escape(filename)}</strong> ·
    generated {stamp} ·
    checkpoint <code>{html.escape(checkpoint_name)}</code> ·
    onnm v{html.escape(app_version)}<span class="signal"></span>
  </div>

  <div class="verdict" style="--accent:{accent};">
    <h2>{html.escape(verdict)}</h2>
    <p>{confidence_pct:.1f}% confidence · decided at a {threshold:.2f} lesion threshold</p>
  </div>
  {uncertainty_note}

  <h2>Three-class breakdown</h2>
  <table>
    <tr><th>class</th><th>probability</th><th></th></tr>
    {_probability_rows(class_probabilities)}
  </table>
  <p style="font-size:0.85rem;color:#555">
    Lesion probability (benign + malignant): <strong>{100 * lesion_probability:.1f}%</strong>
    · {html.escape(calibration_note)}
  </p>

  {images_block}

  <h2>Medical disclaimer</h2>
  <div class="disclaimer">{html.escape(disclaimer)}</div>

  <p class="footnote">
    This report was generated by an unvalidated research prototype with no FDA, CE, or
    MHRA clearance. It is not a diagnosis. BTXRD training data is CC BY-NC-ND 4.0 —
    keep this report and its imagery local; do not redistribute.
    Print this page to PDF from your browser for an archival copy.
  </p>
</div>
</body>
</html>
"""
