"""Stage the serving checkpoint into the container build context.

WHY THIS EXISTS
---------------
The Cloudflare Container bakes the model into its image, so something has to
decide *which* model. That decision already has an owner: ``reports/PRODUCTION``,
written only by ``scripts/version_model.py`` and only when no guarded metric
regressed. This script does not make the choice -- it reads it, proves the file
on disk is the one the ledger says is serving, and copies it where the Dockerfile
can find it.

The alternative, a hardcoded path in the Dockerfile, would be a second place that
names the production run. The two would agree right up until a promotion, and
then the site would quietly keep serving the old weights while every document in
the repository said otherwise. That failure has already happened once in this
project's history -- see the docstring of ``src/checkpoint_fetch.py``, which was
rewritten to key its cache on configuration rather than filename for exactly this
reason.

WHAT IT REFUSES TO DO
---------------------
It will not stage a checkpoint whose sha256 does not match the version recorded
as ``serving`` in ``model_versions.json``. A mismatch means the working tree and
the ledger disagree about what is deployed, and the only safe response is to stop
and say so. Pass ``--allow-unregistered`` to stage anyway; the flag exists for
local experiments and prints a loud warning.

USAGE
-----
    python scripts/stage_inference_model.py            # stage the serving model
    python scripts/stage_inference_model.py --check    # verify only, copy nothing

The staged directory is gitignored: it holds a 27 MB weights file that must never
enter a public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from onnm.inference import production_checkpoint  # noqa: E402
from onnm.versioning import find_by_sha, load_registry, serving  # noqa: E402

#: Where the Dockerfile expects to find the model. Kept inside the build context
#: because Docker cannot COPY from outside it.
STAGE_DIR = REPO_ROOT / "inference" / "model"

#: Copied alongside the weights. ``calibration.json`` is not optional in
#: practice: ``Calibration.for_checkpoint`` looks for it *beside* the checkpoint,
#: and without it the model runs at temperature 1.0 and a naive 0.50 threshold,
#: which silently changes every number the app shows. ``config.json`` is a
#: fallback only -- the checkpoint embeds its own config -- but it costs 4 KB.
COMPANIONS = ("calibration.json", "config.json")


def sha256_of(path: Path) -> str:
    """Digest a file in chunks, so a 27 MB checkpoint is not held twice in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve() -> tuple[Path, str]:
    """Return the serving checkpoint and its digest, or exit with an explanation."""
    checkpoint = production_checkpoint()
    if checkpoint is None:
        raise SystemExit(
            "reports/PRODUCTION is missing, so no model is pinned for production.\n"
            "Promote one with: python scripts/version_model.py register --run <run>"
        )
    if not checkpoint.is_file():
        raise SystemExit(f"reports/PRODUCTION names a run whose checkpoint is absent: {checkpoint}")
    return checkpoint, sha256_of(checkpoint)


def verify(checkpoint: Path, digest: str, *, allow_unregistered: bool) -> None:
    """Cross-check the file on disk against the version ledger."""
    versions = load_registry()
    current = serving(versions)
    matched = find_by_sha(versions, digest)

    print(f"  checkpoint : {checkpoint.relative_to(REPO_ROOT)}")
    print(f"  sha256     : {digest}")
    print(f"  ledger says: {current.version if current else '(nothing marked serving)'}")

    if current is not None and current.checkpoint_sha256 == digest:
        print(f"  OK         : this is {current.version}, the version marked serving")
        return

    detail = (
        f"it is registered as {matched.version} ({matched.status})"
        if matched is not None
        else "it is not registered in model_versions.json at all"
    )
    message = (
        f"The checkpoint under reports/PRODUCTION does not match the serving version.\n"
        f"  on disk : {digest}\n"
        f"  ledger  : {current.checkpoint_sha256 if current else '(none)'}\n"
        f"  {detail}\n"
        f"The working tree and the ledger disagree about what is deployed."
    )
    if not allow_unregistered:
        raise SystemExit(message + "\nRe-run with --allow-unregistered to stage it anyway.")
    print("  WARNING    : " + message.replace("\n", "\n               "))


def stage(checkpoint: Path) -> None:
    """Copy the checkpoint and its companions into the build context."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Replace rather than merge. A stale calibration.json left behind from a
    # previous run would be found beside the new weights and silently applied.
    for existing in STAGE_DIR.iterdir():
        if existing.is_file():
            existing.unlink()

    shutil.copy2(checkpoint, STAGE_DIR / "best.pt")
    print(f"  staged     : best.pt ({checkpoint.stat().st_size / 1e6:.1f} MB)")

    for name in COMPANIONS:
        source = checkpoint.parent / name
        if source.is_file():
            shutil.copy2(source, STAGE_DIR / name)
            print(f"  staged     : {name}")
        elif name == "calibration.json":
            print(
                "  WARNING    : no calibration.json beside the checkpoint. The container "
                "will run uncalibrated at a 0.50 threshold, which corresponds to no "
                "clinical policy. Run scripts/calibrate.py first."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the ledger agrees with the working tree, but copy nothing",
    )
    parser.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="stage even when the checkpoint is not the registered serving version",
    )
    args = parser.parse_args()

    print("Staging the serving checkpoint for the inference container:")
    checkpoint, digest = resolve()
    verify(checkpoint, digest, allow_unregistered=args.allow_unregistered)

    if args.check:
        print("  (--check: nothing copied)")
        return 0

    stage(checkpoint)
    print(f"  into       : {STAGE_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
