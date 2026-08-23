"""One command from "approved in the review console" to "in the training set".

WHAT THIS DOES
--------------
Three steps that were previously three manual acts:

1. Claims every approved, unbatched row out of Cloudflare into a batch
   directory under a **durable store** (``--store``).
2. Rebuilds the cumulative manifests by concatenating every batch in that
   store -- ``configs/controls_manifest.csv`` for the lesion classifier, and
   ``configs/ood_manifest.csv`` for the out-of-distribution gate.
3. Reports what training will now see.

``configs/controls_manifest.csv`` is already the default value of
``paths.controls_manifest`` in ``base.yaml``, so after this runs there is
nothing to edit: the next ``make_splits.py`` and ``train.py`` pick the rows up
on their own. That is the whole point -- an automation that ends with "now go
and change a config line" has not automated the part that gets forgotten.

WHY REBUILD RATHER THAN APPEND
------------------------------
Appending is not idempotent, and this command will be run repeatedly and
sometimes twice by accident. Rebuilding from the batch directories makes the
manifest a *derived* file: run it any number of times and the result is the
same, a half-finished run is repaired by the next one, and deleting a batch
directory removes those rows rather than orphaning them.

The batch directories are the source of truth because they are what actually
holds the images. A manifest row whose PNG is missing is skipped by
``build_records`` with a warning, which is a silent shortfall in the training
set -- so the two are regenerated together, always.

WHY THE STORE IS A SEPARATE FLAG
--------------------------------
Because Colab wipes ``/content`` on disconnect, and export **claims** rows:
the Worker sets ``batch_id`` so the same example cannot enter two generations
of training. A claim is therefore irreversible from the client, and a batch
claimed onto a disk that is about to vanish is data lost for good.

    # locally -- the store is just the repo
    python scripts/sync_community.py

    # in Colab -- the store is Drive, which survives the runtime
    python scripts/sync_community.py --store /content/drive/MyDrive/OSTEONEURALNETWORK/community

The manifest records absolute paths when the store sits outside the checkout,
which ``build_records`` handles, so training works either way.

USAGE
-----
    export ONNM_COMMUNITY_URL=https://onnm-community.<sub>.workers.dev
    export ONNM_ADMIN_KEY=...

    python scripts/sync_community.py --dry-run   # what would be claimed
    python scripts/sync_community.py             # claim, then rebuild
    python scripts/sync_community.py --rebuild-only   # no network at all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from community import VALID_LABELS, CommunityClient  # noqa: E402
from export_batch import (  # noqa: E402
    MANIFEST_COLUMNS,
    OOD_MANIFEST_COLUMNS,
    REPO_ROOT,
    write_batch,
)
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_STORE = "data/community"
DEFAULT_MANIFEST = "configs/controls_manifest.csv"
DEFAULT_OOD_MANIFEST = "configs/ood_manifest.csv"


def _rebuild(store: Path, name: str, target: Path, columns: list[str]) -> list[dict]:
    """Concatenate every batch's ``name`` into ``target``. Returns the rows.

    Rows whose image has gone missing are dropped here rather than left for
    ``build_records`` to skip with a warning. The difference matters: a warning
    buried in a training log is how a batch silently shrinks, whereas a count
    printed by the command that just claimed the rows is something you notice.
    """
    rows: list[dict] = []
    dropped = 0
    for batch_manifest in sorted(store.glob(f"*/{name}")):
        with batch_manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image = Path(row["image"])
                if not image.is_absolute():
                    image = REPO_ROOT / image
                if not image.is_file():
                    dropped += 1
                    continue
                rows.append({column: row.get(column, "") for column in columns})

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    if dropped:
        logger.warning(
            "%d row(s) in %s reference images that are no longer on disk and were "
            "left out. If the store is on Drive, check it is mounted.",
            dropped, target.name,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store", default=DEFAULT_STORE,
        help="durable directory holding the batches. Point this at Drive in Colab, "
             "because export claims rows and a claim cannot be undone from here.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="cumulative lesion manifest (the default is what base.yaml reads)")
    parser.add_argument("--ood-manifest", default=DEFAULT_OOD_MANIFEST,
                        help="cumulative manifest of confirmed non-radiographs")
    parser.add_argument("--note", default=None, help="note stored with the claimed batch")
    parser.add_argument("--limit", type=int, default=100,
                        help="max rows to claim in one go (server caps at 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is ready, claim nothing, rebuild nothing")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="rebuild the manifests from the store without contacting the API")
    args = parser.parse_args()

    store = Path(args.store).expanduser()
    if not store.is_absolute():
        store = (REPO_ROOT / store).resolve()
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = (REPO_ROOT / manifest).resolve()
    ood_manifest = Path(args.ood_manifest).expanduser()
    if not ood_manifest.is_absolute():
        ood_manifest = (REPO_ROOT / ood_manifest).resolve()

    claimed = 0
    if not args.rebuild_only:
        client = CommunityClient()
        if not client.admin_enabled:
            print(
                "ONNM_COMMUNITY_URL and ONNM_ADMIN_KEY must be set.\n"
                "  export ONNM_COMMUNITY_URL=https://onnm-community.<subdomain>.workers.dev\n"
                "  export ONNM_ADMIN_KEY=...\n"
                "Or pass --rebuild-only to work from what is already in the store.",
                file=sys.stderr,
            )
            return 2

        health = client.health()
        if health is None:
            print("cannot reach the community API -- check ONNM_COMMUNITY_URL", file=sys.stderr)
            return 2
        pending = health.get("pending_review", 0)
        print(
            f"community: {health.get('submissions')} submissions, "
            f"{pending} awaiting review, {health.get('approved')} approved"
        )
        if pending:
            print(f"  ({pending} still need approving in the review console — "
                  "they are not claimed by this run)")

        result = client.export_batch(
            note=args.note, limit=args.limit, dry_run=args.dry_run
        )
        if result.get("error"):
            print(f"export failed: {result['error']}", file=sys.stderr)
            return 1

        if args.dry_run:
            rows = result.get("rows", [])
            print(f"\n{len(rows)} row(s) ready to claim (nothing written):")
            for row in rows:
                print(f"  {row['submission_id'][:8]}  [{row.get('admin_bucket'):<13}] "
                      f"-> {row.get('admin_label')}")
            if not rows:
                print("  (nothing approved and unbatched — approve some first)")
            return 0

        claimed = result.get("count", 0)
        if claimed:
            batch_dir = store / result["batch_id"]
            written = write_batch(result, batch_dir, args.note)
            if written is None:
                print(
                    "rows failed the local gate and nothing was written. They are still "
                    "claimed server-side; fix the review and re-approve rather than "
                    "re-running.",
                    file=sys.stderr,
                )
                return 1
            print(f"\nclaimed {claimed} row(s) into {batch_dir}")
        else:
            print("\nnothing new to claim.")

    # Always rebuild, even when nothing was claimed: it repairs a manifest left
    # half-written by an interrupted run, and it is how --rebuild-only works.
    lesion_rows = _rebuild(store, "manifest.csv", manifest, MANIFEST_COLUMNS)
    ood_rows = _rebuild(store, "ood_manifest.csv", ood_manifest, OOD_MANIFEST_COLUMNS)

    by_label: dict[str, int] = {}
    for row in lesion_rows:
        try:
            name = VALID_LABELS[int(row["label"])]
        except (TypeError, ValueError, IndexError):
            continue
        by_label[name] = by_label.get(name, 0) + 1

    batches = sorted(p.name for p in store.glob("*") if p.is_dir())
    print(f"\nstore: {store}  ({len(batches)} batch(es))")
    print(f"  {manifest.name}: {len(lesion_rows)} radiograph(s) {by_label or '{}'}")
    print(f"  {ood_manifest.name}: {len(ood_rows)} confirmed non-radiograph(s)")

    if lesion_rows and manifest == (REPO_ROOT / DEFAULT_MANIFEST).resolve():
        print(
            "\nThis is already `paths.controls_manifest` in base.yaml, so nothing needs "
            "editing. Re-run make_splits.py, then train:\n"
            "  python scripts/make_splits.py\n"
            "  python scripts/train.py --override configs/densenet121_3class.yaml "
            "--override configs/full_run.yaml --tag full"
        )
    elif lesion_rows:
        print(f"\nPoint the config at it:\n  paths.controls_manifest: {manifest}")

    # A machine-readable index of the whole store, so the notebook can report
    # what it is about to train on without re-reading every batch.
    (store / "store.json").write_text(
        json.dumps(
            {
                "batches": batches,
                "claimed_this_run": claimed,
                "lesion_rows": len(lesion_rows),
                "class_balance": by_label,
                "ood_rows": len(ood_rows),
                "manifest": str(manifest),
                "ood_manifest": str(ood_manifest),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
