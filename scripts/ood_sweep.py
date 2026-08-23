"""Measure the out-of-distribution gate against real data, not phantoms.

    python scripts/ood_sweep.py                          # all of BTXRD
    python scripts/ood_sweep.py --limit 500              # a quick read
    python scripts/ood_sweep.py --negatives path/to/photos

WHY
---
`onnm.ood` stage 1 is four hand-tuned thresholds -- colorfulness, dynamic range,
histogram entropy, edge density -- chosen by looking at a handful of synthetic
phantoms. Two numbers decide whether they are set correctly, and neither had
ever been measured:

    false rejection   how often it turns away a genuine radiograph. Every one of
                      these is a user told their X-ray is "not a radiograph".
    true rejection    how often it catches something that is not one. Every miss
                      here reaches a closed-set softmax and receives a
                      clinical-sounding verdict.

They trade against each other, so this reports both and never combines them.

WHAT IT PRINTS
--------------
The rate, and then a breakdown by which named check fired. That second part is
what makes the result actionable: "4% false rejection" says a threshold is
wrong, and "3.8 of those 4 points are histogram_entropy" says which one.

Torch-free -- numpy, PIL and pydicom only -- so it runs in seconds per hundred
images and needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from onnm.config import load_config  # noqa: E402
from onnm.ood import validate_payload  # noqa: E402
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".ima"}


def _sweep(paths: list[Path], expect_radiograph: bool) -> dict:
    """Run the gate over every file. Returns counts and per-check failures."""
    fired: Counter[str] = Counter()
    wrong: list[str] = []
    unreadable = 0
    accepted = 0

    for index, path in enumerate(paths, 1):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
            unreadable += 1
            continue

        report = validate_payload(payload, path.name)
        if report.is_radiograph:
            accepted += 1
        else:
            # Count every failing check, not just the first: a photograph that
            # trips three checks says something different about the thresholds
            # than one that scrapes past two and fails the third by a hair.
            for check in report.failures:
                fired[check.name] += 1

        if report.is_radiograph != expect_radiograph:
            wrong.append(path.name)

        if index % 500 == 0:
            print(f"  ... {index}/{len(paths)}")

    scored = len(paths) - unreadable
    return {
        "scored": scored,
        "unreadable": unreadable,
        "accepted": accepted,
        "rejected": scored - accepted,
        "misclassified": len(wrong),
        "rate": (len(wrong) / scored) if scored else float("nan"),
        "checks_fired": dict(fired.most_common()),
        "examples": wrong[:20],
    }


def _report(title: str, result: dict, label: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"  scored            {result['scored']}")
    if result["unreadable"]:
        print(f"  unreadable        {result['unreadable']}")
    print(f"  accepted          {result['accepted']}")
    print(f"  rejected          {result['rejected']}")
    print(f"  {label:<17} {result['rate']:.4f}  ({result['misclassified']}/{result['scored']})")
    if result["checks_fired"]:
        print("\n  which check fired (a rejection can trip several):")
        for name, count in result["checks_fired"].items():
            share = count / result["scored"] if result["scored"] else 0
            print(f"    {name:<24} {count:>6}  ({share:.1%} of all scored)")
    if result["examples"]:
        print(f"\n  first few: {', '.join(result['examples'][:8])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N radiographs (for a quick read)")
    parser.add_argument(
        "--negatives", default=None,
        help="a folder of things that are NOT radiographs -- photographs, "
             "screenshots. Measures the other half of the trade.",
    )
    parser.add_argument("--out", default=None, help="write the full result as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    images_dir = cfg.resolve_path("paths.data_root") / cfg.paths.images_dirname
    if not images_dir.is_dir():
        print(f"no images at {images_dir}", file=sys.stderr)
        return 2

    radiographs = sorted(p for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        radiographs = radiographs[: args.limit]

    print(f"scoring {len(radiographs)} radiographs from {images_dir}")
    positives = _sweep(radiographs, expect_radiograph=True)
    _report("BTXRD radiographs -- these should all be ACCEPTED", positives,
            "false rejection")

    summary = {"radiographs": positives}

    if args.negatives:
        folder = Path(args.negatives).expanduser()
        if not folder.is_dir():
            print(f"\nno such folder: {folder}", file=sys.stderr)
            return 2
        others = sorted(p for p in folder.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"\nscoring {len(others)} non-radiographs from {folder}")
        negatives = _sweep(others, expect_radiograph=False)
        _report("Non-radiographs -- these should all be REJECTED", negatives,
                "false acceptance")
        summary["negatives"] = negatives
    else:
        print(
            "\nNo --negatives folder given, so only half the trade is measured. A gate "
            "that rejects nothing scores a perfect false-rejection rate; point this at "
            "a folder of ordinary photographs to see what that costs."
        )

    if args.out:
        target = Path(args.out)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
