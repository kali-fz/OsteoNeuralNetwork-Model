"""Tests for boot-time checkpoint fetching.

This runs once, on a hosted app's first request, against a URL a human pasted
into a secrets box. The failure that matters is not "the download broke" -- that
is loud. It is a URL that returns HTTP 200 with an HTML error page, which a CDN
does routinely for a missing file. Written to best.pt, that surfaces much later
inside torch.load as an unpicklable-file error pointing at the wrong thing
entirely, on a machine nobody can attach a debugger to.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from checkpoint_fetch import MAX_CHECKPOINT_BYTES, TORCH_MAGIC, ensure_checkpoint


class _Response(io.BytesIO):
    """Minimal stand-in for urlopen's context-manager response."""

    def __init__(self, payload: bytes, declared: int | None = None) -> None:
        super().__init__(payload)
        self.headers = {"content-length": str(len(payload) if declared is None else declared)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, payload: bytes, declared: int | None = None) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _Response(payload, declared)
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    for var in ("ONNM_CHECKPOINT_URL", "ONNM_CALIBRATION_URL", "ONNM_CHECKPOINT_RUN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_does_nothing_when_unconfigured(env) -> None:
    """A local run must be completely unaffected."""
    assert ensure_checkpoint(env) is None
    assert list(env.iterdir()) == []


def test_downloads_and_pins_the_run(env, monkeypatch) -> None:
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    _serve(monkeypatch, TORCH_MAGIC + b"weights")

    path = ensure_checkpoint(env)
    assert path is not None and path.is_file()
    assert path.read_bytes().startswith(TORCH_MAGIC)
    # Pinned, so the app serves this run rather than picking by mtime.
    assert (env / "PRODUCTION").read_text().strip() == "hosted"


def test_rejects_an_html_error_page(env, monkeypatch) -> None:
    """The CDN-404 case: status 200, but the body is a web page."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/missing")
    _serve(monkeypatch, b"<!DOCTYPE html><html><body>404 Not Found</body></html>")

    assert ensure_checkpoint(env) is None
    assert not (env / "hosted" / "best.pt").exists()
    # No partial file left behind for a later boot to trip over.
    assert not list(env.glob("hosted/*.part"))


def test_refuses_an_oversized_declared_length(env, monkeypatch) -> None:
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/huge")
    _serve(monkeypatch, TORCH_MAGIC, declared=MAX_CHECKPOINT_BYTES + 1)
    assert ensure_checkpoint(env) is None


def test_network_failure_returns_none_rather_than_raising(env, monkeypatch) -> None:
    """A hosted app must start and say it has no model, not crash on boot."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")

    def boom(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ensure_checkpoint(env) is None


def test_existing_checkpoint_is_not_redownloaded(env, monkeypatch) -> None:
    """Streamlit re-executes the script on every interaction; this must not
    re-download 28 MB each time someone moves a slider."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    target = env / "hosted"
    target.mkdir()
    (target / "best.pt").write_bytes(TORCH_MAGIC + b"already here")

    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: calls.append(a) or _Response(TORCH_MAGIC + b"new"),
    )
    ensure_checkpoint(env)
    assert calls == []


def test_calibration_failure_does_not_block_the_checkpoint(env, monkeypatch) -> None:
    """Without calibration the app runs uncalibrated and says so -- that is a
    state worth surfacing, not a reason to refuse to start."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    monkeypatch.setenv("ONNM_CALIBRATION_URL", "https://example.test/bad.json")

    seen: list[str] = []

    def serve(url, *a, **k):
        seen.append(url)
        if url.endswith("bad.json"):
            raise urllib.error.URLError("gone")
        return _Response(TORCH_MAGIC + b"weights")

    monkeypatch.setattr("urllib.request.urlopen", serve)
    path = ensure_checkpoint(env)
    assert path is not None and path.is_file()
    assert not (path.parent / "calibration.json").exists()


def test_custom_run_name_is_honoured(env, monkeypatch) -> None:
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    monkeypatch.setenv("ONNM_CHECKPOINT_RUN", "full-20260822-041653")
    _serve(monkeypatch, TORCH_MAGIC + b"weights")

    path = ensure_checkpoint(env)
    assert path.parent.name == "full-20260822-041653"
    assert (env / "PRODUCTION").read_text().strip() == "full-20260822-041653"
