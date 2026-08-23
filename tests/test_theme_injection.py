"""Regression tests for the CSS injection in src/theme.py.

The redesign shipped a bug where the Google Fonts <link> tags were string-
concatenated in front of the stylesheet and handed to a single st.markdown()
call.  Streamlit renders markdown before the HTML reaches the browser, and
CommonMark classifies a string starting with <link> as a *type 6* HTML block,
which terminates at the first blank line.  The stylesheet was therefore cut off
at its first blank line (14 lines in, right after the :root token block) and
every rule after that point was rendered as visible text on the landing page.

A string that starts with <style> is a *type 1* HTML block, which runs all the
way to </style> and ignores blank lines.  So the stylesheet must always be the
first thing in its own st.markdown() payload.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_original_streamlit = sys.modules.get("streamlit")
sys.modules["streamlit"] = SimpleNamespace(markdown=lambda *_a, **_k: None)
import theme  # noqa: E402

if _original_streamlit is None:
    del sys.modules["streamlit"]
else:
    sys.modules["streamlit"] = _original_streamlit


def _captured_payloads() -> list[str]:
    with unittest.mock.patch.object(theme.st, "markdown") as markdown:
        theme.inject_theme()
    return [call.args[0] for call in markdown.call_args_list]


def test_stylesheet_is_its_own_markdown_call():
    payloads = _captured_payloads()
    style_payloads = [p for p in payloads if "<style>" in p]
    assert len(style_payloads) == 1
    assert style_payloads[0].startswith("<style>"), (
        "the stylesheet must be the first thing in its payload, or CommonMark "
        "treats it as a type-6 HTML block and truncates it at the first blank line"
    )
    assert style_payloads[0].rstrip().endswith("</style>")


def test_fonts_link_never_precedes_the_stylesheet():
    for payload in _captured_payloads():
        if "<link" in payload:
            assert "<style>" not in payload


def test_every_css_section_survives_the_first_blank_line():
    """Guard the symptom directly: rules far past the first blank line ship."""
    payloads = _captured_payloads()
    css = next(p for p in payloads if "<style>" in p)
    head, _, tail = css.partition("\n\n")
    assert tail, "the stylesheet is expected to contain blank lines"
    for selector in (".onnm-hero", ".onnm-stats", ".gd-verdict", ".onnm-appbar"):
        assert selector in css
        assert selector not in head, (
            f"{selector} sits after a blank line -- this is exactly the region "
            "that was being dropped, so the test is meaningful"
        )


def test_inject_theme_emits_the_google_fonts_link():
    payloads = _captured_payloads()
    assert any("fonts.googleapis.com" in p for p in payloads)


def test_streamlit_material_icons_keep_their_icon_font():
    """The app font must not expose icon ligatures as overlapping text."""
    css = next(p for p in _captured_payloads() if "<style>" in p)
    assert '[data-testid="stIconMaterial"]' in css
    assert 'font-family:"Material Symbols Rounded"' in css
    assert 'font-feature-settings:"liga" !important' in css
