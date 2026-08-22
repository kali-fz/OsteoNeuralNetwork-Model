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

_STYLE = """
  body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         color: #1c1c1c; margin: 0; background: #f5f5f5; }
  .page { max-width: 900px; margin: 0 auto; background: #fff; padding: 32px 40px; }
  h1 { font-size: 1.35rem; margin: 0 0 2px; }
  h2 { font-size: 1.05rem; margin: 26px 0 8px; border-bottom: 1px solid #e2e2e2;
       padding-bottom: 4px; }
  .meta { color: #666; font-size: 0.82rem; margin-bottom: 18px; }
  .verdict { border-left: 6px solid var(--accent); background: var(--bg);
             padding: 14px 18px; border-radius: 4px; margin: 18px 0; }
  .verdict h2 { border: none; margin: 0; font-size: 1.25rem; color: var(--accent); }
  .verdict p { margin: 4px 0 0; color: #444; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  td, th { text-align: left; padding: 6px 10px; border-bottom: 1px solid #ececec; }
  .bar { height: 10px; border-radius: 5px; display: inline-block; vertical-align: middle; }
  .images { display: flex; gap: 16px; flex-wrap: wrap; }
  .images figure { margin: 0; flex: 1 1 260px; }
  .images img { width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  .images figcaption { font-size: 0.78rem; color: #666; margin-top: 4px; }
  .disclaimer { background: #fdf3f3; border: 1px solid #e5b8b8; border-radius: 4px;
                padding: 14px 18px; font-size: 0.82rem; white-space: pre-wrap; }
  .footnote { color: #888; font-size: 0.75rem; margin-top: 22px; }
  @media print {
    body { background: #fff; }
    .page { max-width: none; padding: 0; }
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
        accent, background = "#b26a00", "rgba(224,168,0,0.10)"
    elif verdict.lower().startswith("normal"):
        accent, background = "#2e8b57", "rgba(46,139,87,0.08)"
    else:
        accent, background = "#c62828", "rgba(198,40,40,0.08)"

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
    onnm v{html.escape(app_version)}
  </div>

  <div class="verdict" style="--accent:{accent}; --bg:{background};">
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
