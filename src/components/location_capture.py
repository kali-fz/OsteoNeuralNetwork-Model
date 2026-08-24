"""Invisible browser-to-Worker country capture for signed-in accounts.

The Streamlit server cannot provide the visitor's country: Cloudflare sees the
server as the network caller. This component sends a short-lived, one-use token
directly from the browser. The Worker derives the country at its edge and never
receives a client-supplied country, coordinate, or IP-address field.
"""

from __future__ import annotations

import json


def _capture_html(worker_url: str, token: str) -> str:
    capture_url = f"{worker_url.rstrip('/')}/location/capture"
    url_json = json.dumps(capture_url).replace("<", "\\u003c")
    token_json = json.dumps(token).replace("<", "\\u003c")
    return f"""<!doctype html>
<html><body style="margin:0;background:transparent">
<script>
(() => {{
  const url = {url_json};
  const token = {token_json};
  fetch(url, {{
    method: 'POST',
    headers: {{ authorization: `Bearer ${{token}}` }},
    mode: 'cors',
    credentials: 'omit',
  }}).catch(() => {{ /* Country capture fails soft; the app remains usable. */ }});
}})();
</script>
</body></html>"""


def render_location_capture(worker_url: str, token: str) -> None:
    """Run country capture without exposing a visible widget or app secret."""
    import streamlit.components.v1 as components

    components.html(_capture_html(worker_url, token), height=0, scrolling=False)
