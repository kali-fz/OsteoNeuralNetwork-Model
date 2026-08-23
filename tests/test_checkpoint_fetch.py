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

from checkpoint_fetch import (
    MAX_CHECKPOINT_BYTES,
    TORCH_MAGIC,
    ensure_checkpoint,
    serving_checkpoint_info,
)


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
    for var in (
        "ONNM_CHECKPOINT_URL",
        "ONNM_CALIBRATION_URL",
        "ONNM_CHECKPOINT_RUN",
        "ONNM_CHECKPOINT_SHA256",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _count_calls(monkeypatch, payload_for) -> list[str]:
    """Record every URL fetched, serving whatever ``payload_for`` returns."""
    seen: list[str] = []

    def serve(url, *a, **k):
        seen.append(url)
        payload = payload_for(url)
        if isinstance(payload, Exception):
            raise payload
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", serve)
    return seen


def _pinned(root) -> str:
    """The run named by the marker, read the way production_checkpoint reads it.

    The marker carries a comment header explaining that it is generated, so a
    test that compares the whole file would break on a wording change and say
    nothing about whether the right run is pinned.
    """
    for line in (root / "PRODUCTION").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


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
    assert _pinned(env) == "hosted"


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


def test_an_unchanged_configuration_does_not_redownload(env, monkeypatch) -> None:
    """Streamlit re-executes the script on every interaction; this must not
    re-download 28 MB each time someone moves a slider."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    seen = _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"weights")

    ensure_checkpoint(env)
    assert len(seen) == 1
    for _ in range(5):
        ensure_checkpoint(env)
    assert len(seen) == 1, "a rerun must not re-fetch an unchanged checkpoint"


def test_a_changed_url_replaces_the_cached_checkpoint(env, monkeypatch) -> None:
    """The publish path. This is what silently did nothing before.

    The old guard was "does best.pt exist", so pointing the deployment at a new
    model was ignored whenever the previous file was still on disk -- and whether
    it was depended on whether the platform happened to hand you a fresh
    container, which is not something you can see from a secrets page.
    """
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v1.pt")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"first")
    ensure_checkpoint(env)
    assert (env / "hosted" / "best.pt").read_bytes().endswith(b"first")

    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v2.pt")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"second")
    ensure_checkpoint(env)
    assert (env / "hosted" / "best.pt").read_bytes().endswith(b"second")


def test_a_file_of_unknown_provenance_is_replaced(env, monkeypatch) -> None:
    """No provenance record means we cannot say the bytes are the right ones.

    In this mode the URL is the authority on what should be served, so the
    honest resolution of "something is here and I do not know where it came
    from" is to fetch what was actually asked for.
    """
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    target = env / "hosted"
    target.mkdir()
    (target / "best.pt").write_bytes(TORCH_MAGIC + b"who put this here")

    seen = _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"the real one")
    ensure_checkpoint(env)
    assert len(seen) == 1
    assert (target / "best.pt").read_bytes().endswith(b"the real one")


def test_new_weights_never_keep_the_old_calibration(env, monkeypatch) -> None:
    """calibration.json says where the model calls a lesion.

    Applying the previous run's temperature and threshold to new weights does
    not raise and does not look wrong -- it just moves the operating point to
    one that was never measured for this model.
    """
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v1.pt")
    monkeypatch.setenv("ONNM_CALIBRATION_URL", "https://example.test/v1.json")
    _count_calls(
        monkeypatch,
        lambda url: TORCH_MAGIC + b"first" if url.endswith(".pt") else b'{"temperature": 1.41}',
    )
    ensure_checkpoint(env)
    assert b"1.41" in (env / "hosted" / "calibration.json").read_bytes()

    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v2.pt")
    monkeypatch.setenv("ONNM_CALIBRATION_URL", "https://example.test/v2.json")
    _count_calls(
        monkeypatch,
        lambda url: TORCH_MAGIC + b"second" if url.endswith(".pt") else b'{"temperature": 2.02}',
    )
    ensure_checkpoint(env)
    assert b"2.02" in (env / "hosted" / "calibration.json").read_bytes()


def test_a_dropped_calibration_url_removes_the_stale_file(env, monkeypatch) -> None:
    """Detaching calibration must actually detach it, not leave it applied."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    monkeypatch.setenv("ONNM_CALIBRATION_URL", "https://example.test/c.json")
    _count_calls(
        monkeypatch,
        lambda url: TORCH_MAGIC + b"w" if url.endswith(".pt") else b"{}",
    )
    ensure_checkpoint(env)
    assert (env / "hosted" / "calibration.json").is_file()

    monkeypatch.delenv("ONNM_CALIBRATION_URL")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"w")
    ensure_checkpoint(env)
    assert not (env / "hosted" / "calibration.json").exists()


def test_the_production_marker_follows_the_run_name(env, monkeypatch) -> None:
    """The nastiest of the three, and the one the old advice walked you into.

    "Rename the run to force a fresh download" downloaded the new weights into a
    new directory -- and left PRODUCTION naming the old one, so the app served
    the previous model while every setting in the deployment said otherwise.
    """
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v1.pt")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"first")
    ensure_checkpoint(env)
    assert "hosted" in (env / "PRODUCTION").read_text()

    monkeypatch.setenv("ONNM_CHECKPOINT_RUN", "hosted-v1-0-1")
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v2.pt")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"second")
    ensure_checkpoint(env)

    assert _pinned(env) == "hosted-v1-0-1", "the marker must follow the run, not stick"


def test_a_wrong_digest_is_refused(env, monkeypatch) -> None:
    """The magic-bytes check catches a wrong file; only a digest catches a
    truncated right one, which still starts with PK and still looks fine."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    monkeypatch.setenv("ONNM_CHECKPOINT_SHA256", "0" * 64)
    _serve(monkeypatch, TORCH_MAGIC + b"weights")

    assert ensure_checkpoint(env) is None
    assert not (env / "hosted" / "best.pt").exists()


def test_a_matching_digest_is_accepted(env, monkeypatch) -> None:
    import hashlib

    payload = TORCH_MAGIC + b"weights"
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    monkeypatch.setenv("ONNM_CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest())
    _serve(monkeypatch, payload)

    assert ensure_checkpoint(env) is not None


def test_a_failed_publish_keeps_the_previous_model_serving(env, monkeypatch) -> None:
    """An old model is a worse answer than a new one. No model is no answer."""
    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/v1.pt")
    _count_calls(monkeypatch, lambda url: TORCH_MAGIC + b"first")
    ensure_checkpoint(env)

    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/typo.pt")
    _count_calls(monkeypatch, lambda url: urllib.error.URLError("no such asset"))
    path = ensure_checkpoint(env)
    assert path is not None and path.read_bytes().endswith(b"first")


def test_it_reports_what_it_actually_has(env, monkeypatch) -> None:
    """Reads provenance, not the environment -- the two differ exactly when a
    publish did not take effect, which is when the answer matters."""
    assert serving_checkpoint_info(env) is None

    monkeypatch.setenv("ONNM_CHECKPOINT_URL", "https://example.test/best.pt")
    _serve(monkeypatch, TORCH_MAGIC + b"weights")
    ensure_checkpoint(env)

    info = serving_checkpoint_info(env)
    assert info is not None
    assert info["run"] == "hosted"
    assert info["checkpoint_url"] == "https://example.test/best.pt"
    assert len(info["sha256"]) == 64


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
    assert _pinned(env) == "full-20260822-041653"
