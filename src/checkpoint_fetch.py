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
    ONNM_CHECKPOINT_SHA256    optional, the expected digest of best.pt
    ONNM_CHECKPOINT_RUN       optional run name (default "hosted")

Unset, this module does nothing and the existing local resolution applies --
``reports/PRODUCTION`` first, newest non-throwaway run otherwise. Nothing about
a local run changes.

WHAT THE CACHE IS KEYED ON, AND WHY IT IS NOT THE FILENAME
----------------------------------------------------------
This used to short-circuit on ``best.pt`` merely existing, which made publishing
a new model unreliable in a way that was invisible from the secrets page:

1. A changed ``ONNM_CHECKPOINT_URL`` was ignored whenever the old file was still
   on disk. Whether that happened depended on whether the platform gave you a
   fresh container -- so the same action worked or silently did nothing, and you
   could not tell which from anywhere in the UI.
2. Weights and calibration were guarded independently, so it was possible to
   serve new weights at the old threshold. That does not error; it just quietly
   changes where the model calls a lesion.
3. Worst: ``reports/PRODUCTION`` was only written when absent. The advice that
   followed from (1) -- "rename the run to force a fresh download" -- therefore
   *caused* a bug on a warm container: the new weights downloaded into the new
   directory, the marker still named the old one, and the app kept serving the
   old model while every setting said otherwise.

So the cache is now keyed on the **configuration that produced it**, recorded in
a ``source.json`` beside the checkpoint. Change any of the URLs (or the expected
digest) and the next boot re-fetches weights and calibration *together*; change
nothing and no bytes move, which is what Streamlit's re-execution model needs.
The run name is now cosmetic: it names the directory, and nothing depends on
remembering to change it.

An absent or unreadable ``source.json`` counts as unknown provenance and forces
a re-fetch. In this mode the URL is the authority on what should be served, so
re-downloading is the correct resolution of "a file is here and I cannot say
where it came from".

The download is verified before it is trusted: a wrong URL that returns an HTML
error page would otherwise be written to ``best.pt`` and fail much later inside
``torch.load`` with a confusing message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

#: Records which configuration produced the files beside it. See the module
#: docstring: this, not the filename, is what decides whether a re-fetch is due.
SOURCE_RECORD = "source.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(
    url: str, destination: Path, *, expect_zip: bool, expect_sha256: str = ""
) -> str | None:
    """Fetch one file. Returns its sha256 on success, None on any refusal."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            declared = int(response.headers.get("content-length") or 0)
            if declared > MAX_CHECKPOINT_BYTES:
                logger.error("refusing %s: %d bytes exceeds the cap", url, declared)
                return None
            payload = response.read(MAX_CHECKPOINT_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("could not download %s: %s", url, exc)
        return None

    if len(payload) > MAX_CHECKPOINT_BYTES:
        logger.error("refusing %s: response exceeds the cap", url)
        return None
    # A checkpoint URL that 404s through a CDN usually returns an HTML page with
    # status 200. Writing that to best.pt would surface as an unpicklable-file
    # error at load time, pointing at the wrong thing entirely.
    if expect_zip and not payload.startswith(TORCH_MAGIC):
        logger.error(
            "refusing %s: content is not a torch checkpoint (got %r...). "
            "Check the URL serves the raw file, not an HTML page.",
            url, payload[:16],
        )
        return None

    digest = _sha256(payload)
    # The magic-bytes check catches a wrong file; only a digest catches a
    # truncated right one, which still begins with PK and still looks fine.
    if expect_sha256 and digest != expect_sha256.lower():
        logger.error(
            "refusing %s: sha256 is %s but ONNM_CHECKPOINT_SHA256 says %s. "
            "The download is truncated or the URL serves something else.",
            url, digest[:16], expect_sha256.lower()[:16],
        )
        return None

    partial.write_bytes(payload)
    partial.replace(destination)  # atomic, so a killed boot leaves no half file
    logger.info("fetched %s -> %s (%.1f MB)", url, destination, len(payload) / 1024 ** 2)
    return digest


def _read_source(directory: Path) -> dict[str, Any] | None:
    """What produced the files here, or None when that cannot be established."""
    record = directory / SOURCE_RECORD
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_current(
    directory: Path, url: str, calibration_url: str, expect_sha256: str
) -> bool:
    """Whether what is on disk already matches the configuration asking for it.

    False means re-fetch. Deliberately conservative: an unreadable record, a
    missing file, or any disagreement about where the bytes came from all count
    as stale, because the cost of a needless 28 MB download is a slow boot and
    the cost of a wrong answer here is serving the previous model forever.
    """
    if not (directory / "best.pt").is_file():
        return False
    record = _read_source(directory)
    if record is None:
        return False
    if record.get("checkpoint_url") != url:
        return False
    # Compared even when empty, so *removing* a calibration URL also invalidates
    # -- otherwise the app would keep applying a threshold you had just detached.
    if (record.get("calibration_url") or "") != calibration_url:
        return False
    # An expected digest, when given, is the strongest statement available about
    # which model should be here, so it overrides a URL that merely looks right.
    return not (
        expect_sha256 and (record.get("sha256") or "").lower() != expect_sha256.lower()
    )


def _write_source(directory: Path, record: dict[str, Any]) -> None:
    record = {**record, "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (directory / SOURCE_RECORD).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def ensure_checkpoint(reports_dir: Path | None = None) -> Path | None:
    """Fetch the configured checkpoint unless the cached one already matches.

    Returns the checkpoint path when one is available, else None. Safe to call
    on every rerun: when the configuration is unchanged this reads one small
    JSON file and returns, so Streamlit's re-execution model does not move 28 MB
    every time somebody drags a slider.

    Weights and calibration are fetched as a unit. They are two files describing
    one model -- the second says where the first calls a lesion -- and a state
    where only one of them is current is not a degraded version of serving a
    model, it is serving a different model than the one that was measured.
    """
    url = os.environ.get("ONNM_CHECKPOINT_URL", "").strip()
    if not url:
        return None

    root = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"
    run = os.environ.get("ONNM_CHECKPOINT_RUN", DEFAULT_RUN).strip() or DEFAULT_RUN
    calibration_url = os.environ.get("ONNM_CALIBRATION_URL", "").strip()
    expect_sha256 = os.environ.get("ONNM_CHECKPOINT_SHA256", "").strip()

    directory = root / run
    checkpoint = directory / "best.pt"
    calibration = directory / "calibration.json"

    if not _is_current(directory, url, calibration_url, expect_sha256):
        digest = _download(url, checkpoint, expect_zip=True, expect_sha256=expect_sha256)
        if digest is None:
            # Leave whatever was already serving in place. A failed publish must
            # not take the app down: the previous model is old, not broken.
            return checkpoint if checkpoint.is_file() else None

        # The old calibration described the old weights. Keeping it would apply
        # yesterday's threshold to today's model, which does not raise and does
        # change what the app calls a lesion.
        calibration.unlink(missing_ok=True)
        calibration_sha = None
        if calibration_url:
            # Non-fatal: without it the app runs uncalibrated at a naive 0.50
            # cut and says so in the sidebar, which is a state worth surfacing
            # rather than a reason to refuse to start.
            calibration_sha = _download(calibration_url, calibration, expect_zip=False)

        _write_source(
            directory,
            {
                "checkpoint_url": url,
                "calibration_url": calibration_url or None,
                "sha256": digest,
                "calibration_sha256": calibration_sha,
                "run": run,
            },
        )

    # Written every time, not only when absent. This marker decides which run is
    # served, so leaving a stale one in place is how a correctly-downloaded new
    # model sits on disk while the app serves the previous one -- with every
    # setting in the deployment claiming otherwise.
    marker = root / "PRODUCTION"
    desired = (
        "# Written by src/checkpoint_fetch.py from ONNM_CHECKPOINT_RUN.\n"
        "# Edit the deployment's environment, not this file.\n"
        f"{run}\n"
    )
    if not marker.is_file() or marker.read_text(encoding="utf-8") != desired:
        marker.write_text(desired, encoding="utf-8")
        logger.info("pinned %s as the production run", run)

    return checkpoint if checkpoint.is_file() else None


def serving_checkpoint_info(reports_dir: Path | None = None) -> dict[str, Any] | None:
    """What is actually being served, for the app to display. None if unknown.

    Reads the provenance record rather than the environment, so it reports what
    the app *has*, not what it was *asked* for. The two differ exactly when a
    publish did not take effect, which is the moment the answer matters.
    """
    root = Path(reports_dir) if reports_dir else REPO_ROOT / "reports"
    marker = root / "PRODUCTION"
    if not marker.is_file():
        return None
    run = ""
    for line in marker.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            run = line
            break
    if not run:
        return None
    record = _read_source(root / run) or {}
    return {"run": run, **record}
