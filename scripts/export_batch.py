"""Pull an approved community batch out of Cloudflare into a training manifest.

This closes the loop:

    user uploads -> model predicts -> user flags it wrong -> YOU review and
    assign the true label -> this script -> next generation trains on it

WHAT IT WRITES
--------------
Two manifests, because an approved batch retrains two different things and they
cannot share a file.

    data/community/<batch-id>/
        images/<submission_id>.png         bone radiographs
        manifest.csv                       image,image_id,label,patient_id,split,...
        ood_negatives/<submission_id>.png  everything that is not a radiograph
        ood_manifest.csv                   image,image_id,bucket,is_radiograph,...
        batch.json                         counts and paths, for the notebook

``manifest.csv`` is in the format ``dataset.py`` already reads
(``paths.controls_manifest``), so no change to the training pipeline is needed
to consume it: point ``paths.controls_manifest`` at the CSV and ``build_records``
merges the rows in alongside BTXRD.

``ood_manifest.csv`` is a separate format for a separate purpose: hardening the
out-of-distribution gate, which today is hand-written heuristics with no
learned component and no negative examples at all. Every hotdog someone uploaded
is one. These rows carry no lesion class and deliberately have no column for
one -- a file that *could* express "hotdog, benign" is a file that eventually
will.

WHY TWO FILES RATHER THAN A `bucket` COLUMN
-------------------------------------------
Because ``build_records`` would read a combined file and, finding a label column
it recognises, merge every row into the three-class training set. The separation
has to exist at the level the training pipeline can see. A misc row appearing in
``manifest.csv`` is precisely the hotdog-labelled-normal failure this whole
design is built around, and the strongest available guarantee against it is that
there is no code path which writes one there.

WHAT IT REFUSES TO EXPORT
-------------------------
Anything a human has not reviewed, bucketed and labelled. The Worker's query
already filters on ``review_status='approved' AND admin_label IS NOT NULL AND
admin_bucket IS NOT NULL AND shared=1``, and schema triggers block approval
without a label and refuse a bucket/label pair that contradicts itself. This
script checks again before writing each row, because the cost of a redundant
assertion is one line and the cost of missing this is a poisoned training set
that nothing downstream would flag.

USAGE
-----
    export ONNM_COMMUNITY_URL=https://onnm-community.<sub>.workers.dev
    export ONNM_ADMIN_KEY=...

    python scripts/export_batch.py --dry-run          # what would be exported
    python scripts/export_batch.py --note "gen-2"     # claim and write it

The admin key is necessary but not sufficient: the Worker also requires the
request to name the one account permitted to review, which the client sends
automatically. A key alone cannot export.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from community import (  # noqa: E402
    BUCKET_MISC,
    BUCKET_VALID_BONE,
    MISC_LABEL,
    VALID_LABELS,
    CommunityClient,
    decode_shared_image,
)
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_INDEX = {name: index for index, name in enumerate(VALID_LABELS)}

MANIFEST_COLUMNS = [
    "image", "image_id", "label", "patient_id", "anatomy", "split",
    "source", "license", "sha256", "width", "height",
]

#: The OOD manifest. Note the absence of a lesion-class column: these rows have
#: no diagnosis, and a schema that cannot express one cannot be made to carry a
#: wrong one. `is_radiograph` is always 0 here -- it is written out explicitly so
#: the notebook trains a binary target rather than inferring one from a filename.
OOD_MANIFEST_COLUMNS = [
    "image", "image_id", "is_radiograph", "bucket", "triage_bucket",
    "model_label", "ood_flagged", "ood_score", "source", "license",
    "sha256", "width", "height",
]


def manifest_path(destination: Path) -> str:
    """How an image should be written into a manifest's ``image`` column.

    Repo-relative when the file is inside the checkout, absolute otherwise.
    ``build_records`` resolves a relative path against the repo root and passes
    an absolute one through, so both work -- but a batch stored outside the
    repo (on Drive, so a Colab runtime being wiped does not lose it) has no
    repo-relative form at all, and ``Path.relative_to`` raises rather than
    inventing one.
    """
    try:
        return destination.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return destination.resolve().as_posix()


def _decode_and_save(row: dict, directory: Path) -> tuple:
    """Write one row's PNG. Returns ``(array, path)``."""
    from PIL import Image

    array = decode_shared_image(row["image_b64"])
    destination = directory / f"{row['submission_id']}.png"
    Image.fromarray(array).save(destination, format="PNG", optimize=True)
    return array, destination


def _check_row(row: dict) -> str | None:
    """Refuse anything that must not reach a training set. Returns a reason.

    The last of four gates -- schema trigger, review endpoint, export query,
    here -- and the only one that runs on this machine. The others are all
    remote, which is exactly why this one exists: a mistake in the Worker is a
    deploy away from being live, and this file is what stands between that and
    the manifest the next training run reads.

    ``None`` means the row is safe to write.
    """
    submission_id = row.get("submission_id")
    label, bucket = row.get("admin_label"), row.get("admin_bucket")

    if not row.get("image_b64"):
        return f"{submission_id}: approved but carries no image"
    if bucket not in (BUCKET_VALID_BONE, BUCKET_MISC, "contradiction"):
        return f"{submission_id}: admin_bucket is {bucket!r}, not a real bucket"
    if label not in LABEL_TO_INDEX and label != MISC_LABEL:
        return f"{submission_id}: admin_label is {label!r}, not a real class"
    # The pairing, restated. A 'misc' label in a bone bucket would be written to
    # the lesion manifest with no valid class index; a clinical label in a misc
    # bucket is the hotdog-called-benign case wearing a bucket assignment.
    if bucket == BUCKET_MISC and label != MISC_LABEL:
        return f"{submission_id}: bucket 'misc' with clinical label {label!r}"
    if bucket == BUCKET_VALID_BONE and label == MISC_LABEL:
        return f"{submission_id}: bucket 'valid_bone' labelled 'misc'"
    return None


def _lesion_record(row: dict, images_dir: Path) -> dict:
    """One row of ``manifest.csv``: a reviewed radiograph with a clinical class."""
    submission_id = row["submission_id"]
    array, destination = _decode_and_save(row, images_dir)
    encoded = row["image_b64"]
    digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()

    return {
        "image": manifest_path(destination),
        "image_id": submission_id,
        "label": LABEL_TO_INDEX[row["admin_label"]],
        # The group is the IMAGE, not the submission.
        #
        # This was the submission id, on the reasoning that unrelated uploads
        # share no patient, so one group each is correct and safe. That reasoning
        # has a hole, and it is the common case rather than an exotic one: one
        # person uploading one file twice produces two submissions, two ids and
        # therefore two groups -- so make_splits.py, which keeps groups intact
        # precisely to stop this, is free to put one copy in train and the other
        # in val. The validation score then partly measures memorisation of an
        # image the model was trained on, and reports it as generalisation.
        #
        # Keying on the content hash closes that: byte-identical images are one
        # group by construction, whoever sent them and however often. It cannot
        # group two *different* views of one patient, but nothing in an anonymous
        # upload could, and over-grouping is the safe direction to be wrong in.
        "patient_id": f"community-{digest[:16]}",
        "anatomy": "",
        # Always train. Community data must never enter val or test: those sets
        # measure generalisation against BTXRD's held-out patients, and mixing
        # in self-selected uploads would make the score incomparable to every
        # number in overview.md.
        "split": "train",
        "source": "community",
        "license": "user-submitted, consented",
        "sha256": digest,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def _ood_record(row: dict, negatives_dir: Path) -> dict:
    """One row of ``ood_manifest.csv``: a confirmed non-radiograph.

    These are the negatives the gate has never had. ``onnm.ood`` stage 1 is four
    hand-tuned thresholds chosen by looking at BTXRD and a handful of photographs;
    a confirmed misuse example is worth more than another round of tuning,
    because it is a real thing a real user actually uploaded.
    """
    submission_id = row["submission_id"]
    array, destination = _decode_and_save(row, negatives_dir)

    return {
        "image": manifest_path(destination),
        "image_id": submission_id,
        # Always 0. The manifest exists to teach the gate what a radiograph is
        # not; a positive row would come from BTXRD, not from here.
        "is_radiograph": 0,
        "bucket": row["admin_bucket"],
        # Kept alongside the confirmed bucket because the disagreements are the
        # useful part: a row the gate called valid and a human called misc is a
        # false negative, and the count of those is the metric to improve.
        "triage_bucket": row.get("triage_bucket") or "",
        "model_label": row.get("model_label") or "",
        "ood_flagged": int(bool(row.get("ood_flagged"))),
        "ood_score": row.get("ood_score") if row.get("ood_score") is not None else "",
        "source": "community",
        "license": "user-submitted, consented",
        "sha256": hashlib.sha256(base64.b64decode(row["image_b64"])).hexdigest(),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def write_batch(result: dict, batch_dir: Path, note: str | None = None) -> dict | None:
    """Write one claimed batch to disk. Returns its summary, or None on refusal.

    Shared with ``scripts/sync_community.py``, which needs the same bytes in a
    different directory. Everything that decides *what* a row is stays here, so
    the two entry points cannot drift into disagreeing about which manifest a
    label belongs in.
    """
    rows = result.get("rows", [])
    batch_id = result["batch_id"]

    # Refuse first, write second. A batch that is half-written and then found to
    # contain a bad row leaves images on disk that a later run could pick up, so
    # every row is checked before any PNG exists.
    refusals = [reason for reason in (_check_row(row) for row in rows) if reason]
    for reason in refusals:
        logger.error("refusing %s", reason)
    if refusals:
        return None

    # Split by what each row retrains. The label decides, not the bucket: a
    # contradiction row labelled 'benign' is a bone film the gate wrongly
    # rejected, and it belongs in the lesion manifest with the rest.
    lesion_rows = [row for row in rows if row["admin_label"] != MISC_LABEL]
    ood_rows = [row for row in rows if row["admin_label"] == MISC_LABEL]

    batch_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    manifest = batch_dir / "manifest.csv"
    if lesion_rows:
        images_dir = batch_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        records = [_lesion_record(row, images_dir) for row in lesion_rows]
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(records)

    ood_records: list[dict] = []
    ood_manifest = batch_dir / "ood_manifest.csv"
    if ood_rows:
        negatives_dir = batch_dir / "ood_negatives"
        negatives_dir.mkdir(parents=True, exist_ok=True)
        ood_records = [_ood_record(row, negatives_dir) for row in ood_rows]
        with ood_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OOD_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(ood_records)

    by_label: dict[str, int] = {}
    for record in records:
        name = VALID_LABELS[record["label"]]
        by_label[name] = by_label.get(name, 0) + 1
    by_bucket: dict[str, int] = {}
    for row in rows:
        by_bucket[row["admin_bucket"]] = by_bucket.get(row["admin_bucket"], 0) + 1

    # A machine-readable summary, so the notebook does not have to guess which
    # files a batch produced or parse console output.
    summary = {
        "batch_id": batch_id,
        "note": note,
        "rows": len(rows),
        "buckets": by_bucket,
        "lesion": {
            "count": len(records),
            "manifest": manifest_path(manifest) if records else None,
            "class_balance": by_label,
            "split": "train",
        },
        "ood": {
            "count": len(ood_records),
            "manifest": manifest_path(ood_manifest) if ood_records else None,
            "target": "is_radiograph",
        },
        "records": records,
    }
    (batch_dir / "batch.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:  # noqa: PLR0911 - each early return is a distinct refusal
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
             "(e.g. configs/controls_manifest.csv). For a cumulative manifest that is "
             "rebuilt rather than appended to, use scripts/sync_community.py.",
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
                f"  {row['submission_id'][:8]}  [{row.get('admin_bucket'):<13}] "
                f"model said {row.get('model_label'):<10}"
                f"-> reviewer says {row.get('admin_label'):<10}{flag}"
            )
        if rows:
            print(f"\n  {result.get('lesion_rows', 0)} for the lesion classifier, "
                  f"{result.get('ood_rows', 0)} for the OOD detector")
        else:
            print("  (nothing approved and unbatched -- review some submissions first)")
        return 0

    if count == 0:
        print("nothing to export: no approved, unbatched submissions.")
        return 0

    batch_id = result["batch_id"]
    batch_dir = REPO_ROOT / args.out / batch_id
    summary = write_batch(result, batch_dir, args.note)
    if summary is None:
        print(
            "row(s) failed the local gate and nothing was written.\n"
            "The rows are still claimed into this batch server-side; fix the review "
            "and re-approve rather than re-running.",
            file=sys.stderr,
        )
        return 1

    records = summary["records"]
    ood_records_count = summary["ood"]["count"]
    manifest = batch_dir / "manifest.csv"
    ood_manifest = batch_dir / "ood_manifest.csv"
    by_label = summary["lesion"]["class_balance"]
    by_bucket = summary["buckets"]

    if args.merge_into and records:
        target = REPO_ROOT / args.merge_into
        exists = target.is_file() and target.stat().st_size > 0
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(records)
        logger.info("appended %d rows to %s", len(records), target)

    print(f"\nbatch {batch_id}: {summary['rows']} row(s) -> {batch_dir}")
    print(f"  buckets: {by_bucket}")
    if records:
        print(f"  lesion classifier: {len(records)} image(s), class balance {by_label}")
    if ood_records_count:
        print(f"  OOD detector: {ood_records_count} confirmed non-radiograph(s)")

    if records:
        print("\nTo retrain the lesion classifier, point the config at the manifest:")
        print(f"  paths.controls_manifest: {manifest_path(manifest)}")
        print("Then re-run make_splits.py and train. Community rows are pinned to the")
        print("train split, so val and test stay pure BTXRD and remain comparable.")
    if ood_records_count:
        print("\nTo harden the OOD gate, feed the negatives manifest to the notebook:")
        print(f"  {manifest_path(ood_manifest)}")
        print("These rows have no lesion class and must never enter the 3-class set.")
    print(
        "\nOr let scripts/sync_community.py do all of that in one step -- it claims, "
        "writes and rebuilds the cumulative manifest base.yaml already reads."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
