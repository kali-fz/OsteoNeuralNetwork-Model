"""Fetch a model checkpoint at startup when the repository does not carry one.

WHY
---
``reports/`` is gitignored, so a clone has no weights. That is correct for the
training repo -- checkpoints are build output, not source, and BTXRD's licence
makes casual redistribution of derived artefacts something to do deliberately
rather than by default. But a hosted app deploys *from* that clone and therefore
starts with no model at all.

So the checkpoint is fetched from a URL at boot and cached on local disk. Set:

    ONNM_CHECKPOINT_URL       direct link to best.pt
    ONNM_CALIBRATION_URL      optional, calibration.json for the same run
    ONNM_CHECKPOINT_RUN       optional run name (default "hosted")

Unset, this module does nothing and the existing local resolution applies --
``reports/PRODUCTION`` first, newest non-throwaway run otherwise. Nothing about
a local run changes.

The download is verified before it is trusted: a wrong URL that returns an HTML
error page would otherwise be written to ``best.pt`` and fail much later inside
``torch.load`` with a confusing message.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
# NOT "production": the pin marker is reports/PRODUCTION, and Windows and macOS
# filesystems are case-insensitive, so a run directory of that name and the
# marker file collide -- writing the marker tries to open a directory as a file
# and fails with PermissionError. Linux would have let this through and it would
# have broken only for anyone running the hosted config locally.
DEFAULT_RUN = "hosted"
# A DenseNet-121 checkpoint is ~28 MB. The ceiling is generous but finite, so a
# misconfigured URL pointing at something enormous fails fast instead of
# filling the container's disk.
MAX_CHECKPOINT_BYTES = 500 * 1024 * 1024
TORCH_MAGIC = b"PK\x03\x04"  # torch.save writes a zip archive


def _download(url: str, destination: Path, *, expect_zip: bool) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            declared = int(response.headers.get("content-length") or 0)
            if declared > MAX_CHECKPOINT_BYTES:
                logger.error("refusing %s: %d bytes exceeds the cap", url, declared)
                return False
            payload = response.read(MAX_CHECKPOINT_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("could not download %s: %s", url, exc)
        return False

    if len(payload) > MAX_CHECKPOINT_BYTES:
        logger.error("refusing %s: response exceeds the cap", url)
        return False
    # A checkpoint URL that 404s through a CDN usually returns an HTML page with
    # status 200. Writing that to best.pt would surface as an unpicklable-file
    # error at load time, pointing at the wrong thing entirely.
    if expect_zip and not payload.startswith(TORCH_MAGIC):
        logger.error(
            "refusing %s: content is not a torch checkpoint (got %r...). "
            "Check the URL serves the raw file, not an HTML page.",
            url, payload[:16],
        )
        return False

    partial.write_bytes(payload)
    partial.replace(destination)  # atomic, so a killed boot leaves no half file
    logger.info("fetched %s -> %s (%.1f MB)", url, destination, len(payload) / 1024 ** 2)
    return True


def ensure_checkpoint(reports_dir: Path | None = None) -> Path | None:
    """Download the configured checkpoint if it is not already on disk.

    Returns the checkpoint path when one is available, else None. Safe to call
    on every rerun: an existing file short-circuits, so Streamlit's re-execution
    model does not re-download on every interaction.
    """
    url = os.environ.get("ONNM_CHECKPOINT_URL", "").strip()
    if not url:
        return None

    root = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"
    run = os.environ.get("ONNM_CHECKPOINT_RUN", DEFAULT_RUN).strip() or DEFAULT_RUN
    checkpoint = root / run / "best.pt"

    if not checkpoint.is_file() and not _download(url, checkpoint, expect_zip=True):
        return None

    calibration_url = os.environ.get("ONNM_CALIBRATION_URL", "").strip()
    calibration = checkpoint.parent / "calibration.json"
    if calibration_url and not calibration.is_file():
        # Non-fatal: without it the app runs uncalibrated at a naive 0.50 cut
        # and says so in the sidebar, which is a state worth surfacing rather
        # than a reason to refuse to start.
        _download(calibration_url, calibration, expect_zip=False)

    # Pin it, so the app serves this run rather than picking by mtime.
    marker = root / "PRODUCTION"
    if not marker.is_file():
        marker.write_text(f"{run}\n", encoding="utf-8")
        logger.info("pinned %s as the production run", run)

    return checkpoint
