"""Gate 2: prove the dataset on disk is what we think it is.

    python scripts/verify_data.py --dump-schema   # FIRST RUN: inspect the CSV
    python scripts/verify_data.py                 # validate records + splits
    python scripts/verify_data.py --deep          # also decode every image

The schema dump exists because the BTXRD paper documents the class counts but
not the CSV's column names or value spellings. Nothing in this project
hard-codes them: `configs/base.yaml` lists candidates, and this script shows
which ones are real so the label map can be written from evidence rather than
from a guess.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (path side effect)
from onnm import CLASS_NAMES
from onnm.config import load_config
from onnm.dataset import build_records, derive_groups, map_labels, read_table, resolve_column
from onnm.io_radiograph import RadiographReadError, read_radiograph
from onnm.utils import format_counts, get_logger, load_json

logger = get_logger("verify_data")

# Published in the BTXRD paper; a mismatch means the label map is wrong.
PUBLISHED_COUNTS = {"normal": 1879, "benign": 1525, "malignant": 342}
PUBLISHED_TOTAL = 3746


def dump_schema(df: pd.DataFrame, table_path: Path, max_unique: int = 25) -> None:
    print("=" * 74)
    print(f"SCHEMA: {table_path}")
    print("=" * 74)
    print(f"rows: {len(df)}   columns: {len(df.columns)}\n")

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_null = int(series.isna().sum())
        print(f"  {col!r}  dtype={series.dtype}  unique={n_unique}  null={n_null}")
        if n_unique <= max_unique:
            for value, count in series.value_counts(dropna=False).items():
                print(f"      {str(value)[:44]:<46} {count:>6}")
        else:
            sample = ", ".join(str(v)[:22] for v in series.dropna().unique()[:5])
            print(f"      e.g. {sample} ...")
        print()

    print("-" * 74)
    print("First 5 rows")
    print("-" * 74)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head())

    print("\n" + "-" * 74)
    print("ACTION REQUIRED")
    print("-" * 74)
    print("Compare the columns above against configs/base.yaml:")
    print("  labels.tumor_column / labels.class_columns  -> the one-hot indicator columns")
    print("  labels.id_column                            -> the image filename column")
    print("  columns.anatomy / columns.demographic       -> inputs to surrogate grouping")
    print(f"\nTarget distribution from the paper: {PUBLISHED_COUNTS} (total {PUBLISHED_TOTAL})")
    print("Re-run without --dump-schema once the config matches.")


def check_distribution(records: list[dict]) -> bool:
    labels = [r["label"] for r in records]
    print("\n" + "=" * 74)
    print("CLASS DISTRIBUTION")
    print("=" * 74)
    print(format_counts(labels, CLASS_NAMES))

    counts = Counter(labels)
    ok = True
    print(f"\n  {'class':<12}{'found':>8}{'published':>12}{'delta':>8}")
    print("  " + "-" * 40)
    for idx, name in enumerate(CLASS_NAMES):
        found = counts.get(idx, 0)
        expected = PUBLISHED_COUNTS.get(name, 0)
        delta = found - expected
        flag = "" if delta == 0 else "  <-- MISMATCH"
        if delta != 0:
            ok = False
        print(f"  {name:<12}{found:>8}{expected:>12}{delta:>+8}{flag}")

    if not ok:
        print("\n  Counts differ from the published figures. Most likely the label map in")
        print("  configs/base.yaml does not cover every raw value. Re-run --dump-schema")
        print("  and reconcile before training -- a mis-mapped malignant subtype silently")
        print("  deletes cases from the rarest and most important class.")
    else:
        print("\n  Matches the published distribution exactly.")

    malignant = counts.get(2, 0)
    if malignant:
        print(f"\n  Malignant share: {100 * malignant / len(labels):.1f}%")
        print(f"  A 15% test split leaves ~{round(0.15 * malignant)} malignant images, so every")
        print("  reported metric needs a bootstrap confidence interval to be meaningful.")
    return ok


def check_splits(cfg, records: list[dict]) -> bool:
    splits_path = cfg.resolve_path("paths.splits_file")
    print("\n" + "=" * 74)
    print("SPLITS")
    print("=" * 74)
    if not splits_path.is_file():
        print(f"  {splits_path} does not exist yet.")
        print("  -> python scripts/make_splits.py")
        return True

    splits = load_json(splits_path)
    by_id = {r["image_id"]: r for r in records}
    ok = True

    for name in ("train", "val", "test"):
        ids = splits.get(name, [])
        subset = [by_id[i] for i in ids if i in by_id]
        missing = len(ids) - len(subset)
        counts = Counter(r["label"] for r in subset)
        dist = ", ".join(f"{n}={counts.get(i, 0)}" for i, n in enumerate(CLASS_NAMES))
        print(f"  {name:<6} n={len(subset):<6} {dist}")
        if missing:
            print(f"         {missing} ids in splits.json have no matching record (stale split)")
            ok = False
        if counts.get(2, 0) == 0:
            print(f"         no malignant cases in {name} -- unusable")
            ok = False

    # The check that matters: the same patient must never appear on both sides
    # of the split, or the test score measures memorisation, not generalisation.
    print("\n  Leakage check (patient/group overlap between splits):")
    groups = {
        name: {by_id[i]["patient_id"] for i in splits.get(name, []) if i in by_id}
        for name in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            ok = False
            sample = list(overlap)[:5]
            print(f"    FAIL {a}/{b}: {len(overlap)} shared groups, e.g. {sample}")
        else:
            print(f"    ok   {a}/{b}: no shared groups")

    if splits.get("grouped") is False:
        print("\n  NOTE: splits.json records that no patient column was available, so these")
        print("  splits are per-image. Multiple views of one patient may straddle the")
        print("  split. State this limitation alongside any reported result.")
    return ok


def deep_check(records: list[dict], limit: int | None) -> bool:
    subset = records if limit is None else records[:limit]
    print("\n" + "=" * 74)
    print(f"DEEP DECODE ({len(subset)} files)")
    print("=" * 74)

    failures: list[tuple[str, str]] = []
    shapes: Counter = Counter()
    for i, record in enumerate(subset, 1):
        try:
            arr, meta = read_radiograph(record["image"])
            shapes[arr.shape] += 1
            if meta.get("inverted"):
                logger.info("%s: MONOCHROME1 inverted", Path(record["image"]).name)
        except RadiographReadError as exc:
            failures.append((record["image"], str(exc)))
        if i % 250 == 0:
            print(f"  {i}/{len(subset)} ...")

    print(f"\n  decoded ok : {len(subset) - len(failures)}")
    print(f"  failed     : {len(failures)}")
    print("\n  Most common shapes:")
    for shape, count in shapes.most_common(8):
        print(f"    {str(shape):<20} {count}")
    if len(shapes) > 1:
        print("\n  Shapes vary, as expected for radiographs. The transform chain resizes the")
        print("  longest side and pads, preserving aspect ratio.")

    for path, reason in failures[:20]:
        print(f"    FAIL {path}: {reason}")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dump-schema", action="store_true", help="inspect the CSV and stop")
    parser.add_argument("--deep", action="store_true", help="decode every image file")
    parser.add_argument("--deep-limit", type=int, default=None, help="cap --deep to N files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg.resolve_path("paths.data_root")
    table_path = data_root / cfg.paths.table_name

    if not table_path.is_file():
        print(f"ERROR: {table_path} not found.")
        print("  -> python scripts/download_btxrd.py")
        return 1

    df = read_table(cfg)

    if args.dump_schema:
        dump_schema(df, table_path)
        return 0

    print("=" * 74)
    print("COLUMN RESOLUTION")
    print("=" * 74)
    checks = [("id", cfg.labels.id_column), ("tumor flag", cfg.labels.tumor_column)]
    checks += [(f"class:{e['class']}", e["column"]) for e in cfg.labels.class_columns]
    for label, candidate in checks:
        hit = resolve_column(df, candidate)
        print(f"  {label:<18}{f'-> {hit!r}' if hit else '-> NOT FOUND'}")

    _, class_to_idx = map_labels(df, cfg)
    print(f"  class mapping     -> {class_to_idx}")

    groups = derive_groups(df, cfg)
    sizes = groups.value_counts()
    multi = int(sizes[sizes > 1].sum())
    print("\n  Surrogate patient grouping")
    print(f"    strategy        : {cfg.split.group_strategy}")
    print(f"    groups          : {groups.nunique()} over {len(df)} images")
    print(f"    in multi-image  : {multi} ({100 * multi / len(df):.1f}%)")
    print(f"    largest group   : {int(sizes.max())} images")
    print("    BTXRD has no patient id; groups are reconstructed from runs of")
    print("    consecutive ids sharing centre, age, sex, anatomy and diagnosis.")

    records = build_records(cfg)
    dist_ok = check_distribution(records)
    splits_ok = check_splits(cfg, records)
    deep_ok = deep_check(records, args.deep_limit) if args.deep else True

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, ok in [
        ("distribution", dist_ok),
        ("splits", splits_ok),
        ("deep decode", deep_ok if args.deep else None),
    ]:
        print(f"  {name:<16}{'SKIPPED' if ok is None else ('PASS' if ok else 'FAIL')}")

    if dist_ok and splits_ok and deep_ok:
        print("\nData verified. Next: python scripts/make_splits.py (if not done), then pytest")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
