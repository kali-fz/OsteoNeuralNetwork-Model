"""Create reproducible, leakage-free train/val/test splits.

    python scripts/make_splits.py

Writes ``data/interim/splits.json`` with a content hash so the identical split
can be reproduced locally and on Kaggle. Two properties matter:

**Stratification** -- with only 342 malignant images, an unstratified split can
easily hand one fold half the malignant cases it should have, and the resulting
metric swing gets mistaken for a modelling result.

**Grouping** -- if a patient identifier exists, every image from one patient goes
to exactly one split. Two views of the same lesion sitting either side of the
split boundary lets the model memorise rather than generalise, and inflates test
scores in a way no amount of downstream care can undo. When BTXRD offers no such
column the script says so explicitly and records ``grouped: false`` in the output,
so the limitation travels with the split instead of being forgotten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import _bootstrap  # noqa: F401  (path side effect)
from onnm import CLASS_NAMES
from onnm.config import load_config
from onnm.dataset import build_records
from onnm.utils import get_logger, save_json

logger = get_logger("make_splits")


def _grouped_holdout(
    ids: np.ndarray, y: np.ndarray, groups: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split off ``fraction`` of the data, keeping groups intact and classes stratified.

    Implemented via StratifiedGroupKFold rather than a single shuffle split
    because it is the only scikit-learn splitter that honours *both* constraints
    at once. n_splits is chosen so one fold approximates the requested fraction.
    """
    n_splits = max(2, min(20, round(1.0 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(splitter.split(ids, y, groups))
    return rest_idx, held_idx


def make_splits(records: list[dict], cfg, seed: int) -> dict:
    # External controls have already been assigned to train/val by their
    # provenance-aware ingester; splitting them again would silently violate
    # that assignment. Only BTXRD records participate in this split operation.
    split_records = [r for r in records if "_split" not in r]
    controls = [r for r in records if "_split" in r]
    ids = np.array([r["image_id"] for r in split_records])
    y = np.array([r["label"] for r in split_records])
    groups = np.array([r["patient_id"] for r in split_records])

    grouped = len(set(groups)) < len(ids)
    if grouped:
        logger.info(
            "grouping by patient/study: %d groups across %d images", len(set(groups)), len(ids)
        )
    else:
        logger.warning(
            "every image has a unique group id -- no real patient grouping is available. "
            "Splitting per image; the same patient may appear in more than one split."
        )

    test_frac = float(cfg.split.test)
    val_frac = float(cfg.split.val)

    if grouped:
        rest_idx, test_idx = _grouped_holdout(ids, y, groups, test_frac, seed)
        # val_frac is a fraction of the whole, so rescale against what remains.
        val_of_rest = val_frac / (1.0 - test_frac)
        sub_rest, sub_val = _grouped_holdout(
            ids[rest_idx], y[rest_idx], groups[rest_idx], val_of_rest, seed + 1
        )
        train_idx, val_idx = rest_idx[sub_rest], rest_idx[sub_val]
    else:
        stratify = y if bool(cfg.split.stratify) else None
        rest_idx, test_idx = train_test_split(
            np.arange(len(ids)), test_size=test_frac, random_state=seed, stratify=stratify
        )
        val_of_rest = val_frac / (1.0 - test_frac)
        train_idx, val_idx = train_test_split(
            rest_idx,
            test_size=val_of_rest,
            random_state=seed + 1,
            stratify=y[rest_idx] if stratify is not None else None,
        )

    splits = {
        "train": sorted(ids[train_idx].tolist()),
        "val": sorted(ids[val_idx].tolist()),
        "test": sorted(ids[test_idx].tolist()),
        "grouped": bool(grouped),
        "seed": seed,
        "fractions": {"train": float(cfg.split.train), "val": val_frac, "test": test_frac},
    }
    for control in controls:
        target = str(control["_split"])
        if target in {"train", "val", "test"}:
            splits[target].append(control["image_id"])
            splits[target].sort()

    payload = json.dumps(
        {k: splits[k] for k in ("train", "val", "test")}, sort_keys=True
    ).encode()
    splits["content_hash"] = hashlib.sha256(payload).hexdigest()[:16]
    return splits


def report(splits: dict, records: list[dict]) -> bool:
    by_id = {r["image_id"]: r for r in records}
    total = sum(len(splits[name]) for name in ("train", "val", "test"))

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    header = f"  {'split':<8}{'n':>7}{'pct':>7}" + "".join(f"{n:>11}" for n in CLASS_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    ok = True
    for name in ("train", "val", "test"):
        subset = [by_id[i] for i in splits[name]]
        counts = Counter(r["label"] for r in subset)
        row = f"  {name:<8}{len(subset):>7}{100 * len(subset) / total:>6.1f}%"
        for idx in range(len(CLASS_NAMES)):
            n = counts.get(idx, 0)
            share = 100 * n / max(len(subset), 1)
            row += f"{n:>6} ({share:>3.0f}%)"
        print(row)
        if counts.get(2, 0) == 0:
            print(f"    ERROR: {name} has no malignant cases")
            ok = False

    print("\n  Leakage check:")
    groups = {
        name: {by_id[i]["patient_id"] for i in splits[name]} for name in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            ok = False
            print(f"    FAIL {a}/{b}: {len(overlap)} shared groups")
        else:
            print(f"    ok   {a}/{b}: disjoint")

    ids_all = [i for name in ("train", "val", "test") for i in splits[name]]
    if len(ids_all) != len(set(ids_all)):
        print("    FAIL: an image_id appears in more than one split")
        ok = False

    if not splits["grouped"]:
        print("\n  WARNING: no patient identifier was available, so these splits are")
        print("  per-image. Disclose this in the README and in any reported result.")

    print(f"\n  content_hash: {splits['content_hash']}")
    print("  Reproduce elsewhere by copying splits.json, or re-run with the same seed.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.seed)
    out_path = cfg.resolve_path("paths.splits_file")

    if out_path.is_file() and not args.force:
        print(f"{out_path} already exists. Re-running would invalidate every result")
        print("computed against the old split. Pass --force if that is intended.")
        return 0

    records = build_records(cfg)
    splits = make_splits(records, cfg, seed)
    ok = report(splits, records)

    if not ok:
        print("\nSplits are unusable -- not written.")
        return 1

    save_json(splits, out_path)
    print(f"\nWrote {out_path}")
    print("Next: pytest tests/ -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
