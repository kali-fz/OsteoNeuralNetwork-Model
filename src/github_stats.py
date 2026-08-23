"""GitHub repository statistics — star and fork counts.

Fetched from the public GitHub REST API with no authentication token.
The endpoint is public, but the unauthenticated rate limit is 60 requests /
hour / IP; with ttl=900 the app only ever uses 4 requests per hour per server
process, well inside that budget.

All callers must treat None as a valid result meaning "data unavailable":
a GitHub outage must not break a page about radiographs.

``_fetch_uncached`` is the raw network call with no Streamlit dependency, so
it can be imported and tested without a running Streamlit server.
"""

from __future__ import annotations

import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_API = "https://api.github.com/repos/kali-fz/OsteoNeuralNetwork-Model"
_TIMEOUT = 3  # seconds — a star count is not worth blocking a page for


def _fetch_uncached() -> Optional[dict]:
    """Core fetch logic — no Streamlit dependency, callable from tests."""
    try:
        req = urllib.request.Request(
            _REPO_API,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "ONNM-Streamlit/1.0 (+https://github.com/kali-fz/OsteoNeuralNetwork-Model)",
            },
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            import json
            data = json.loads(resp.read())
        return {
            "stars": int(data.get("stargazers_count", 0)),
            "forks": int(data.get("forks_count", 0)),
            "watchers": int(data.get("subscribers_count", 0)),
        }
    except Exception as exc:  # noqa: BLE001 — intentional catch-all, fail soft
        logger.info("GitHub stats unavailable: %s", exc)
        return None


def fetch_github_stats() -> Optional[dict]:
    """Return a dict with ``stars``, ``forks``, ``watchers`` or None on any failure.

    Cached for 15 minutes when running inside Streamlit; falls back to a plain
    call when Streamlit is not available (e.g., during tests).  Never raises.
    """
    try:
        import streamlit as st
        return st.cache_data(ttl=900, show_spinner=False)(_fetch_uncached)()
    except ImportError:
        return _fetch_uncached()
