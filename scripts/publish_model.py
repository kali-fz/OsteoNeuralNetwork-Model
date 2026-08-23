"""Stage a model version for the live site, and verify it once it is up there.

    python scripts/publish_model.py              # stage whatever is serving
    python scripts/publish_model.py v1.0.1       # stage a specific version
    python scripts/publish_model.py --verify https://.../best.pt

WHAT THIS REPLACES
------------------
Publishing used to be: find the checkpoint, upload it somewhere, remember which
secrets to change, and remember to rename the run so the app would notice. The
last step was the one that bit -- and worse, it was advice that *caused* a
different bug, because the production marker was only written when absent.

``checkpoint_fetch`` now keys its cache on the configuration rather than on a
filename, so renaming is no longer needed and the marker follows the run. What
remains is genuinely manual: uploading a 28 MB file to a host, and pasting three
values. This script does everything either side of that.

WHY STAGING IS A COPY RATHER THAN A POINTER
--------------------------------------------
``reports/`` holds every run, most of them experiments, all of them named after
a timestamp. Picking the right two files out of that at upload time, in a
browser, is exactly where the wrong generation gets published. Staging copies
the pair -- weights and calibration, which are two files describing one model --
into one directory named after the version, so the upload step is "drag this
folder's contents" rather than a decision.

THE VERIFY STEP
---------------
``--verify`` fetches the URL you just published and checks the bytes against the
digest in the ledger, *before* the deployment is pointed at it. This catches the
two failures that are otherwise invisible until the app is live and wrong: a CDN
serving an HTML error page with status 200, and a truncated upload that still
begins with a valid zip header. Both look like a working URL in a browser.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from checkpoint_fetch import MAX_CHECKPOINT_BYTES, TORCH_MAGIC  # noqa: E402
from community import USER_AGENT  # noqa: E402
from onnm.utils import get_logger  # noqa: E402
from onnm.versioning import Version, load_registry, serving  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def _digest(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            out.update(block)
    return out.hexdigest()


def _resolve(argument: str | None) -> Version | None:
    versions = load_registry()
    if not versions:
        return None
    if argument is None:
        return serving(versions) or versions[-1]
    wanted = argument if argument.startswith("v") else f"v{argument}"
    return next((v for v in versions if v.version == wanted), None)


def cmd_stage(args: argparse.Namespace) -> int:
    version = _resolve(args.version)
    if version is None:
        print(
            "no such version in the ledger. `python scripts/version_model.py list`"
            " shows what is registered.",
            file=sys.stderr,
        )
        return 2

    run_dir = REPORTS / version.run
    checkpoint = run_dir / "best.pt"
    calibration = run_dir / "calibration.json"

    if not checkpoint.is_file():
        print(
            f"{checkpoint} is not on this machine. The ledger records the version, but\n"
            "reports/ is gitignored, so the weights only exist where they were trained\n"
            "or wherever you backed them up.",
            file=sys.stderr,
        )
        return 2

    digest = _digest(checkpoint)
    if version.checkpoint_sha256 and digest != version.checkpoint_sha256:
        # The ledger says this version is a particular set of bytes. Publishing
        # different ones under its number would make every recorded metric a
        # claim about a model nobody is running.
        print(
            f"REFUSING: {checkpoint} does not match {version.version}.\n"
            f"  ledger:   {version.checkpoint_sha256}\n"
            f"  on disk:  {digest}\n"
            "That run directory has been overwritten since it was registered.",
            file=sys.stderr,
        )
        return 1

    staging = REPO_ROOT / args.out / version.version
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, staging / "best.pt")
    if calibration.is_file():
        shutil.copy2(calibration, staging / "calibration.json")

    size_mb = checkpoint.stat().st_size / 1024**2
    print(f"\n{version.version}  ({version.run})  --  staged in {staging}")
    print(f"  best.pt            {size_mb:.1f} MB")
    print(
        "  calibration.json   present"
        if calibration.is_file()
        else "  calibration.json   MISSING -- the app will run uncalibrated at a naive 0.50 cut"
    )
    print(f"  sha256             {digest}")

    print("\n1. Upload BOTH files from that folder to a host that serves raw bytes.")
    print("   A GitHub Release works and is free. Copy the direct download links.")
    print("\n2. Check the links actually serve the file before touching the deployment:")
    print(f"   python scripts/publish_model.py {version.version} --verify <best.pt URL>")
    print("\n3. Then set these in Streamlit Cloud -> Settings -> Secrets:\n")
    print('   ONNM_CHECKPOINT_URL    = "<direct link to best.pt>"')
    print('   ONNM_CALIBRATION_URL   = "<direct link to calibration.json>"')
    print(f'   ONNM_CHECKPOINT_SHA256 = "{digest}"')
    print(
        "\n   ONNM_CHECKPOINT_RUN does not need changing. The app keys its cache on\n"
        "   these values, so changing the URL is enough to make it re-fetch, and the\n"
        "   production marker follows automatically."
    )
    print("\n4. Reboot the app. The sidebar will name the version it ends up serving --")
    print("   if it does not say", version.version, "the publish did not take effect.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Fetch a published URL and check it against the ledger, before going live."""
    version = _resolve(args.version)
    expected = version.checkpoint_sha256 if version else ""

    request = urllib.request.Request(args.verify)
    request.add_header("user-agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read(MAX_CHECKPOINT_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"could not fetch {args.verify}: {exc}", file=sys.stderr)
        return 1

    print(f"fetched {len(payload) / 1024**2:.1f} MB")

    if not payload.startswith(TORCH_MAGIC):
        print(
            "NOT A CHECKPOINT. The URL returned something else -- a CDN serving an\n"
            "HTML error page with status 200 is the usual cause, and it looks like a\n"
            "working link in a browser.\n"
            f"  first bytes: {payload[:24]!r}",
            file=sys.stderr,
        )
        return 1

    digest = hashlib.sha256(payload).hexdigest()
    print(f"sha256  {digest}")

    if not expected:
        print(
            "\nThe ledger has no digest for this version, so this only confirms the URL\n"
            "serves a torch checkpoint -- not which one."
        )
        return 0

    if digest != expected:
        print(
            f"\nMISMATCH. Expected {expected}\n"
            "The upload is truncated, or the link points at a different generation.\n"
            "Do not point the deployment at this URL.",
            file=sys.stderr,
        )
        return 1

    print(f"\nMatches {version.version}. Safe to publish:\n")
    print(f'   ONNM_CHECKPOINT_URL    = "{args.verify}"')
    print(f'   ONNM_CHECKPOINT_SHA256 = "{digest}"')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "version", nargs="?", default=None,
        help="which version to publish (default: whichever is serving)",
    )
    parser.add_argument("--out", default="dist", help="staging directory")
    parser.add_argument(
        "--verify", metavar="URL", default=None,
        help="fetch this URL and check it against the ledger instead of staging",
    )
    args = parser.parse_args()
    return cmd_verify(args) if args.verify else cmd_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
