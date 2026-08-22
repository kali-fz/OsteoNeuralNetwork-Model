"""Pull an approved community batch out of Cloudflare into a training manifest.

This closes the loop:

    user uploads -> model predicts -> user flags it wrong -> YOU review and
    assign the true label -> this script -> next generation trains on it

WHAT IT WRITES
--------------
A directory of PNGs plus a CSV in the manifest format ``dataset.py`` already
reads (``paths.controls_manifest``). No change to the training pipeline is
needed to consume it: point ``paths.controls_manifest`` at the CSV and
``build_records`` merges the rows in alongside BTXRD.

    data/community/<batch-id>/
        images/<submission_id>.png
        manifest.csv          image,image_id,label,patient_id,split,source,...

WHAT IT REFUSES TO EXPORT
-------------------------
Anything a human has not reviewed and labelled. The Worker's query already
filters on ``review_status='approved' AND admin_label IS NOT NULL AND
shared=1``, and a schema trigger blocks approval without a label. This script
checks a third time before writing a row, because the cost of a redundant
assertion is one line and the cost of missing this is a poisoned training set
that nothing downstream would flag.

USAGE
-----
    export ONNM_COMMUNITY_URL=https://onnm-community.<sub>.workers.dev
    export ONNM_ADMIN_KEY=...

    python scripts/export_batch.py --dry-run          # what would be exported
    python scripts/export_batch.py --note "gen-2"     # claim and write it
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from community import VALID_LABELS, CommunityClient, decode_shared_image  # noqa: E402
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_INDEX = {name: index for index, name in enumerate(VALID_LABELS)}

MANIFEST_COLUMNS = [
    "image", "image_id", "label", "patient_id", "anatomy", "split",
    "source", "license", "sha256", "width", "height",
]


def _write_row(row: dict, images_dir: Path) -> dict | None:
    """Validate one exported row and write its PNG. Returns a manifest record."""
    submission_id = row.get("submission_id")
    admin_label = row.get("admin_label")
    encoded = row.get("image_b64")

    # The third gate. Should be unreachable; that is the point.
    if admin_label not in LABEL_TO_INDEX:
        logger.error("refusing %s: admin_label is %r, not a real class", submission_id, admin_label)
        return None
    if not encoded:
        logger.error("refusing %s: approved but carries no image", submission_id)
        return None

    array = decode_shared_image(encoded)
    destination = images_dir / f"{submission_id}.png"
    from PIL import Image

    Image.fromarray(array).save(destination, format="PNG", optimize=True)

    return {
        "image": destination.relative_to(REPO_ROOT).as_posix(),
        "image_id": submission_id,
        "label": LABEL_TO_INDEX[admin_label],
        # Each community submission is its own patient group. Grouping exists to
        # stop multiple views of one patient straddling a split; unrelated
        # uploads share no patient, so one group each is both correct and safe.
        "patient_id": f"community-{submission_id}",
        "anatomy": "",
        # Always train. Community data must never enter val or test: those sets
        # measure generalisation against BTXRD's held-out patients, and mixing
        # in self-selected uploads would make the score incomparable to every
        # number in overview.md.
        "split": "train",
        "source": "community",
        "license": "user-submitted, consented",
        "sha256": hashlib.sha256(base64.b64decode(encoded)).hexdigest(),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="data/community", help="output root")
    parser.add_argument("--batch-id", default=None, help="name the batch (default: timestamp)")
    parser.add_argument("--note", default=None, help="free-text note stored with the batch")
    parser.add_argument("--limit", type=int, default=100,
                        help="max rows to claim (server caps at 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be exported, claim nothing")
    parser.add_argument(
        "--merge-into",
        default=None,
        help="append to this existing manifest CSV instead of writing a standalone one "
             "(e.g. configs/controls_manifest.csv)",
    )
    args = parser.parse_args()

    client = CommunityClient()
    if not client.admin_enabled:
        print(
            "ONNM_COMMUNITY_URL and ONNM_ADMIN_KEY must be set.\n"
            "  export ONNM_COMMUNITY_URL=https://onnm-community.<subdomain>.workers.dev\n"
            "  export ONNM_ADMIN_KEY=...",
            file=sys.stderr,
        )
        return 2

    health = client.health()
    if health is None:
        print("cannot reach the community API -- check ONNM_COMMUNITY_URL", file=sys.stderr)
        return 2
    logger.info(
        "community: %s submissions, %s pending review, %s approved, %.1f%% of storage cap",
        health.get("submissions"), health.get("pending_review"),
        health.get("approved"), 100 * float(health.get("capacity_used", 0.0)),
    )

    result = client.export_batch(
        batch_id=args.batch_id, note=args.note, limit=args.limit, dry_run=args.dry_run
    )
    if result.get("error"):
        print(f"export failed: {result['error']}", file=sys.stderr)
        return 1

    rows = result.get("rows", [])
    count = result.get("count", 0)

    if args.dry_run:
        print(f"\n{count} row(s) ready to export (nothing claimed):\n")
        for row in rows:
            flag = " [user disputed]" if row.get("user_says_wrong") else ""
            print(
                f"  {row['submission_id'][:8]}  model said {row.get('model_label'):<10}"
                f"-> reviewer says {row.get('admin_label'):<10}{flag}"
            )
        if not rows:
            print("  (nothing approved and unbatched -- review some submissions first)")
        return 0

    if count == 0:
        print("nothing to export: no approved, unbatched submissions.")
        return 0

    batch_id = result["batch_id"]
    batch_dir = REPO_ROOT / args.out / batch_id
    images_dir = batch_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = [r for r in (_write_row(row, images_dir) for row in rows) if r is not None]
    if not records:
        print("every row failed validation -- nothing written", file=sys.stderr)
        return 1

    manifest = batch_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    if args.merge_into:
        target = REPO_ROOT / args.merge_into
        exists = target.is_file() and target.stat().st_size > 0
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(records)
        logger.info("appended %d rows to %s", len(records), target)

    by_label: dict[str, int] = {}
    for record in records:
        name = VALID_LABELS[record["label"]]
        by_label[name] = by_label.get(name, 0) + 1

    print(f"\nbatch {batch_id}: {len(records)} image(s) -> {batch_dir.relative_to(REPO_ROOT)}")
    print(f"  class balance: {by_label}")
    print("\nTo train on it, point the config at the manifest:")
    print(f"  paths.controls_manifest: {manifest.relative_to(REPO_ROOT).as_posix()}")
    print("Then re-run make_splits.py and train. Community rows are pinned to the")
    print("train split, so val and test stay pure BTXRD and remain comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
