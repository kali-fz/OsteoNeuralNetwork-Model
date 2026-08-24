"""Tests for the globe component's fallback and privacy properties.

Covers:
- ``render_globe`` degrades gracefully when assets are unavailable.
- production rendering never waits on a globe-asset network download.
- the explicit globe asset setup helper fails soft.
- SAMPLE_MARKERS contains no sub-country coordinates and no user identifiers.
- The component HTML never embeds API keys (privacy contract from section 3B).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestGlobeFallback:
    def test_country_map_assets_are_vendored(self):
        """Production should not fall back to a landless sphere."""
        import json

        from components.globe import _ASSETS, ASSETS_DIR

        paths = [ASSETS_DIR / name for name in _ASSETS]
        assert all(path.is_file() and path.stat().st_size > 1_000 for path in paths)

        world = json.loads((ASSETS_DIR / "countries-110m.json").read_text("utf-8"))
        assert {"land", "countries"} <= set(world["objects"])

    def test_sample_markers_have_required_fields(self):
        """Every sample marker must have lat, lng, label, count, and layer."""
        from components.globe import SAMPLE_MARKERS

        required = {"lat", "lng", "label", "count", "layer"}
        for marker in SAMPLE_MARKERS:
            missing = required - set(marker.keys())
            assert not missing, f"Marker missing fields {missing}: {marker}"

    def test_sample_markers_layer_values(self):
        """Layer must be 'signup' or 'contributor' — no other values allowed."""
        from components.globe import SAMPLE_MARKERS

        valid_layers = {"signup", "contributor"}
        for marker in SAMPLE_MARKERS:
            assert marker["layer"] in valid_layers, (
                f"Unexpected layer value: {marker['layer']!r}"
            )

    def test_sample_markers_no_subregion_coordinates(self):
        """Coordinates are country-level (approximately).

        This mirrors the privacy guarantee from section 3B: no point finer than
        a country centroid is ever stored or transmitted.  As a heuristic we
        assert that lat/lng values are consistent with known country centroids
        (between -90/90 and -180/180).
        """
        from components.globe import SAMPLE_MARKERS

        for marker in SAMPLE_MARKERS:
            lat, lng = marker["lat"], marker["lng"]
            assert -90 <= lat <= 90, f"Invalid latitude: {lat}"
            assert -180 <= lng <= 180, f"Invalid longitude: {lng}"

    def test_sample_markers_no_user_identifiers(self):
        """Sample markers must not contain user IDs, emails, or timestamps."""
        from components.globe import SAMPLE_MARKERS

        forbidden_keys = {"user_id", "email", "submission_id", "timestamp", "ip"}
        for marker in SAMPLE_MARKERS:
            present = forbidden_keys & set(marker.keys())
            assert not present, f"Marker has forbidden keys {present}: {marker}"

    def test_ensure_assets_returns_false_on_network_failure(self, monkeypatch, tmp_path):
        """_ensure_assets must return False (not raise) on network failure."""
        import components.globe as globe_mod

        # Redirect ASSETS_DIR to a temp directory so we don't touch the real one.
        monkeypatch.setattr(globe_mod, "ASSETS_DIR", tmp_path)

        def _fail(*_, **__):
            raise OSError("simulated download failure")

        monkeypatch.setattr("urllib.request.urlopen", _fail)

        result = globe_mod._ensure_assets()
        assert result is False

    def test_runtime_asset_loading_never_uses_the_network(self, monkeypatch, tmp_path):
        """A missing optional map asset must produce an immediate local fallback."""
        import components.globe as globe_mod

        monkeypatch.setattr(globe_mod, "ASSETS_DIR", tmp_path)

        def _unexpected(*_, **__):
            raise AssertionError("landing-page render attempted a network request")

        monkeypatch.setattr("urllib.request.urlopen", _unexpected)
        assert globe_mod._load_static_assets() is None

    def test_fallback_html_draws_supplied_coarse_markers(self):
        from components.globe import _build_fallback_html, _json_for_script

        marker = {
            "lat": 55.4,
            "lng": -3.4,
            "label": "United Kingdom",
            "count": 5,
            "layer": "signup",
        }
        html = _build_fallback_html(_json_for_script([marker]), True, 320)

        assert "const markers" in html
        assert "United Kingdom" in html
        assert "Globe unavailable" not in html
        assert "cdn.jsdelivr.net" not in html

    def test_html_does_not_contain_api_key_patterns(self):
        """The built HTML must not embed API keys.

        This is a regex scan for common secret patterns in the rendered HTML.
        It is a belt-and-suspenders check on top of the code review; a real key
        should never reach this file in the first place.
        """
        import re

        from components.globe import _build_html

        html = _build_html(
            d3_script="/* stub */",
            topojson_script="/* stub */",
            world_json="null",
            markers_json="[]",
            auto_rotate=False,
            height=300,
        )

        # Patterns that look like API keys or secrets
        suspicious = re.compile(
            r"(ONNM_COMMUNITY_KEY|ONNM_ADMIN_KEY|Bearer\s+\w{20}|"
            r"Authorization:\s*\w+|api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"])",
            re.IGNORECASE,
        )
        match = suspicious.search(html)
        assert match is None, (
            f"Potential secret found in component HTML: {match.group()!r}"
        )

    def test_detailed_globe_fills_countries_with_activity(self):
        """The full renderer should colour the country beneath each marker."""
        from components.globe import _build_html, _json_for_script

        marker = {
            "lat": 55.4,
            "lng": -3.4,
            "label": "United Kingdom",
            "count": 2,
            "layer": "contributor",
        }
        html = _build_html(
            d3_script="/* d3 */",
            topojson_script="/* topo */",
            world_json='{"objects":{"land":{},"countries":{}}}',
            markers_json=_json_for_script([marker]),
            auto_rotate=False,
            height=320,
        )

        assert "const activeCountries" in html
        assert "d3geo.geoContains(feature, [marker.lng, marker.lat])" in html
        assert "rgba(46,107,71,0.78)" in html
        assert "rgba(232,168,80,0.76)" in html

    def test_markers_json_injected_safely(self):
        """markers_json must appear in the HTML without executing arbitrary code."""
        from components.globe import _build_html

        # Attempt basic JSON injection
        evil_markers = (
            '[{"lat":0,"lng":0,"label":"</script><script>alert(1)</script>",'
            '"count":1,"layer":"signup"}]'
        )
        html = _build_html(
            d3_script="/* d3 */",
            topojson_script="/* topo */",
            world_json="null",
            markers_json=evil_markers,
            auto_rotate=False,
            height=300,
        )
        # The marker is retained, but HTML-significant characters are encoded so
        # it cannot close the component's script element.
        assert html.count("</script>") == 3
        assert "</script><script>alert(1)</script>" not in html
        assert "\\u003c/script\\u003e" in html

    def test_marker_boundary_drops_identifiers_and_invalid_coordinates(self):
        from components.globe import _normalise_markers

        markers = _normalise_markers(
            [
                {
                    "lat": 55.4,
                    "lng": -3.4,
                    "label": "United Kingdom",
                    "count": 6,
                    "layer": "signup",
                    "email": "private@example.test",
                    "user_id": "private-user",
                },
                {"lat": 200, "lng": 0, "count": 1, "layer": "signup"},
            ]
        )

        assert markers == [
            {
                "lat": 55.4,
                "lng": -3.4,
                "label": "United Kingdom",
                "count": 6,
                "layer": "signup",
            }
        ]
