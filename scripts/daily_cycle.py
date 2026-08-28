"""The daily loop: approvals in, a guarded version out -- or nothing at all.

    .venv\\Scripts\\python.exe scripts\\daily_cycle.py            # do it
    .venv\\Scripts\\python.exe scripts\\daily_cycle.py --dry-run  # decide only

WHAT IT DOES, IN ORDER
----------------------
1. Asks the community API how many approved, unclaimed rows are waiting.
2. **If there are none, it stops.** No training, no version, no ledger row.
3. Otherwise: claims them (``sync_community.py``), retrains, calibrates,
   evaluates, and registers a new version (``version_model.py register``),
   which promotes it only if nothing regressed.

STEP 2 IS THE POINT
-------------------
Retraining on an unchanged dataset does not produce a better model. It produces
a *different* one -- different initialisation order, different augmentation
draws -- scoring within noise of the last, and if the loop promoted it anyway
the served model would wander from generation to generation for no reason, and
the version history would fill with numbers that mean nothing.

Worse, it would burn the one thing this project cannot buy more of: the local
GPU. Skipping is not an optimisation, it is what makes the daily cadence
honest. A day with no approvals genuinely has no new information in it.

WHAT PROTECTS THE MODEL
-----------------------
Training writes to a fresh ``reports/<run>/`` and never touches an existing
checkpoint. Promotion is a separate, guarded decision made afterwards -- see
``onnm.versioning`` -- so a retrain that damages the model produces a `held` row
in ``ONN.md`` and leaves ``reports/PRODUCTION`` pointing exactly where it was.
There is no step in this script that can overwrite a good model with a bad one.

RUNNING IT DAILY
----------------
Windows Task Scheduler, once a day:

    schtasks /create /tn "ONNM daily" /sc daily /st 03:00 ^
      /tr "D:\\GITHUB\\ONNM\\OsteoNeuralNetwork-Model\\.venv\\Scripts\\python.exe
           D:\\GITHUB\\ONNM\\OsteoNeuralNetwork-Model\\scripts\\daily_cycle.py"

It needs ``ONNM_COMMUNITY_URL`` and ``ONNM_ADMIN_KEY`` in the environment it
runs under, which for a scheduled task means system or user variables rather
than a shell session.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from community import CommunityClient  # noqa: E402
from onnm.utils import get_logger  # noqa: E402
from onnm.versioning import load_registry, serving  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPTS = REPO_ROOT / "scripts"


def _run(command: list[str], *, what: str) -> bool:
    """Run one pipeline step, streaming its output. Returns success."""
    print(f"\n{'=' * 72}\n{what}\n{'=' * 72}")
    print("  " + " ".join(str(part) for part in command) + "\n")
    finished = subprocess.run(command, cwd=REPO_ROOT, check=False)  # noqa: S603
    if finished.returncode != 0:
        print(f"\n{what} failed (exit {finished.returncode}).", file=sys.stderr)
        return False
    return True


def _approved_waiting(client: CommunityClient, limit: int) -> int:
    """How many approved rows are unclaimed. A dry-run export, so nothing moves."""
    result = client.export_batch(limit=limit, dry_run=True)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return int(result.get("count", 0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", default="data/community",
                        help="durable community store (point at Drive when run in Colab)")
    parser.add_argument("--limit", type=int, default=100, help="max rows to claim in one cycle")
    parser.add_argument("--level", default="patch", choices=("major", "minor", "patch"),
                        help="version bump. A data-only retrain is a patch.")
    parser.add_argument(
        "--min-rows", type=int, default=1,
        help="approvals needed before training is worth a GPU-hour. The default of 1 "
             "honours the instruction literally; raise it to batch up small days.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the config's epoch count for this cycle")
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and report, but claim nothing and train nothing")
    parser.add_argument("--force", action="store_true",
                        help="train even with no new approvals. Off by default for a reason "
                             "-- see the module docstring.")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    print(f"ONNM daily cycle -- {stamp}")

    current = serving(load_registry())
    print(f"serving: {current.version + ' (' + current.run + ')' if current else 'nothing pinned'}")

    # -- 1. is there anything new? -----------------------------------------
    client = CommunityClient()
    if not client.admin_enabled:
        print(
            "ONNM_COMMUNITY_URL and ONNM_ADMIN_KEY must be set for this to run.\n"
            "For a scheduled task that means user or system environment variables, "
            "not a shell session.",
            file=sys.stderr,
        )
        return 2

    health = client.health()
    if health is None:
        print("cannot reach the community API -- nothing done.", file=sys.stderr)
        return 2

    try:
        waiting = _approved_waiting(client, args.limit)
    except RuntimeError as exc:
        print(f"could not ask what is approved: {exc}", file=sys.stderr)
        return 2

    pending = health.get("pending_review", 0)
    print(f"community: {waiting} approved and unclaimed, {pending} still awaiting your review")

    # -- 2. the skip ---------------------------------------------------------
    if waiting < args.min_rows and not args.force:
        print(
            f"\nNot enough new data ({waiting} approved, {args.min_rows} needed). "
            "Skipping training.\n"
            "The served model is unchanged and no version was created: retraining on "
            "a dataset that has not grown moves the model without improving it, and "
            "spends the one GPU this project has to do it."
        )
        if pending:
            print(
                f"\n{pending} submission(s) are waiting for you in the review console:\n"
                "  https://onnm.kali-fz.workers.dev/admin"
            )
        return 0

    if args.dry_run:
        print(f"\n--dry-run: would claim {waiting} row(s) and train. Nothing done.")
        return 0

    # -- 3. claim, train, evaluate, register --------------------------------
    tag = f"community-{stamp}"
    if not _run(
        [PYTHON, str(SCRIPTS / "sync_community.py"), "--store", args.store,
         "--note", f"daily cycle {stamp}"],
        what="Claiming approved rows and rebuilding the manifest",
    ):
        return 1

    # Splits must be recut: the manifest just gained rows, and a stale
    # splits.json would silently leave every one of them out of training.
    if not _run(
        [PYTHON, str(SCRIPTS / "make_splits.py")],
        what="Recutting splits (community rows are pinned to train)",
    ):
        return 1

    train_command = [
        PYTHON, str(SCRIPTS / "train.py"),
        "--override", "configs/densenet121_3class.yaml",
        "--override", "configs/full_run.yaml",
        "--tag", tag,
    ]
    if args.epochs is not None:
        train_command += ["--epochs", str(args.epochs)]
    if not _run(train_command, what=f"Training {tag}"):
        return 1

    run_dir = REPO_ROOT / "reports" / tag
    checkpoint = next(
        (run_dir / name for name in ("best.pt", "last.pt") if (run_dir / name).is_file()), None
    )
    if checkpoint is None:
        print(f"training produced no checkpoint under {run_dir}", file=sys.stderr)
        return 1

    if not _run(
        [PYTHON, str(SCRIPTS / "calibrate.py"), "--checkpoint", str(checkpoint), "--sweep"],
        what="Calibrating on validation",
    ):
        return 1

    if not _run(
        [PYTHON, str(SCRIPTS / "evaluate.py"), "--checkpoint", str(checkpoint),
         "--split", "test"],
        what="Evaluating on the held-out test split",
    ):
        return 1

    if not _run(
        [PYTHON, str(SCRIPTS / "version_model.py"), "register",
         "--run", tag, "--level", args.level, "--store", args.store,
         "--note", f"daily community cycle, {waiting} newly approved row(s)"],
        what="Registering the version (and promoting it only if nothing regressed)",
    ):
        return 1

    print(
        "\nDone. Read ONN.md for what changed. If the new version was held rather "
        "than promoted, the previous checkpoint is still serving and nothing needs "
        "undoing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
