"""Static guards for the Streamlit regressions visible in the redesign review."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
THEME_SOURCE = (ROOT / "src" / "theme.py").read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    return APP_SOURCE.split(f"def {name}", 1)[1].split(f"def {next_name}", 1)[0]


def test_landing_has_no_inert_html_buttons() -> None:
    landing = _function_source("render_landing", "render_auth")
    assert "<button" not in landing
    assert 'key="home_signin"' in landing
    assert 'key="hero_google_btn"' in landing
    assert 'key="hero_scanner_btn"' in landing


def test_account_actions_have_one_sign_out_control() -> None:
    assert APP_SOURCE.count('key="account_signout"') == 1
    assert 'st.button("Logout"' not in APP_SOURCE


def test_home_header_does_not_repeat_the_scanner_action() -> None:
    header = _function_source("render_account_header", "_hero_bg_css")
    assert 'if active_page == "landing":' in header
    assert 'st.button("Scanner"' not in header
    assert 'st.button("Back to home"' in header


def test_hosted_scanner_does_not_make_a_false_local_storage_claim() -> None:
    scanner = _function_source("render_scanner", "render_profile")
    assert "No data leaves this machine" not in scanner
    assert "only when you enable sharing below" in scanner


def test_upload_validation_is_cached_before_scanner_reruns() -> None:
    assert "def inspect_upload" in APP_SOURCE
    scanner = _function_source("render_scanner", "render_profile")
    assert "digest, validation = inspect_upload(payload, uploaded.name)" in scanner
    assert "validation = validate_payload(payload, uploaded.name)" not in scanner


def test_material_icon_ligatures_are_restored() -> None:
    assert '[data-testid="stIconMaterial"]' in THEME_SOURCE
    assert 'font-family:"Material Symbols Rounded"' in THEME_SOURCE
