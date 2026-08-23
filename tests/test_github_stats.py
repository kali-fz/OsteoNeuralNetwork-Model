"""Tests for src/github_stats.py.

Covers the critical fail-soft contract: a network failure must return None,
not raise an exception.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fresh_fetch():
    """Import fetch_github_stats without the st.cache_data wrapper."""
    import urllib.request as _ur

    # Re-import with the Streamlit cache bypassed.
    import importlib
    import github_stats
    importlib.reload(github_stats)
    return github_stats.fetch_github_stats


class TestGithubStats:
    def test_failure_returns_none_not_raise(self, monkeypatch):
        """Any network error must return None, never raise."""
        import github_stats

        def _raise(*_, **__):
            raise OSError("simulated network failure")

        monkeypatch.setattr("urllib.request.urlopen", _raise)

        # Bypass st.cache_data by calling the underlying function directly.
        # We monkeypatch the module-level function rather than the wrapper.
        result = github_stats._fetch_uncached()
        assert result is None

    def test_timeout_returns_none(self, monkeypatch):
        """A timeout is treated as unavailable, not an error."""
        import urllib.error
        import github_stats

        def _timeout(*_, **__):
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _timeout)
        result = github_stats._fetch_uncached()
        assert result is None

    def test_bad_json_returns_none(self, monkeypatch):
        """A malformed API response returns None rather than raising."""
        import io
        import github_stats

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return b"not valid json at all!"

        monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: _Resp())
        result = github_stats._fetch_uncached()
        assert result is None

    def test_successful_response_has_stars_key(self, monkeypatch):
        """A valid API response returns a dict with 'stars'."""
        import io
        import json
        import github_stats

        payload = json.dumps({
            "stargazers_count": 42,
            "forks_count": 7,
            "subscribers_count": 3,
        }).encode()

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return payload

        monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: _Resp())
        result = github_stats._fetch_uncached()
        assert result is not None
        assert result["stars"] == 42
        assert result["forks"] == 7
