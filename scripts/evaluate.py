"""Evaluate a checkpoint on a held-out split, with bootstrap confidence intervals.

    python scripts/evaluate.py --checkpoint reports/train-.../best.pt

Reports the operating point chosen on **validation** and applied unchanged to
test. Selecting a threshold on the test split and then reporting the resulting
numbers is the most common way an otherwise honest pipeline produces an inflated
result, so the two runs are kept explicitly separate here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (path side effect)
from onnm.config import load_config
from onnm.train import evaluate
from onnm.utils import get_logger, save_json

logger = get_logger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    cfg = load_config(args.config, overrides=args.override, profile=args.profile)

    # The operating point belongs to validation; test only consumes it.
    val_result = evaluate(cfg, checkpoint, split="val")
    operating_point = val_result["operating_point"]
    print("\nOperating point selected on VALIDATION:")
    for key, value in operating_point.items():
        print(f"  {key:<14}{value:.4f}")

    if args.split != "val":
        print("\n" + "#" * 66)
        print(f"# {args.split.upper()} SPLIT")
        print("#" * 66)
        result = evaluate(cfg, checkpoint, split=args.split)
    else:
        result = val_result

    output = checkpoint.parent / f"metrics_{args.split}.json"
    save_json(
        {
            "split": args.split,
            "metrics": {
                k: v for k, v in result["metrics"].items() if not k.startswith("_")
            },
            "confidence_intervals": result["confidence_intervals"],
            "operating_point_from_val": operating_point,
        },
        output,
    )
    print(f"\nWrote {output}")

    recall_ci = result["confidence_intervals"]["malignant_recall"]
    print(
        f"\nHeadline: malignant recall = {recall_ci['point']:.3f} "
        f"(95% CI {recall_ci['lo']:.3f}-{recall_ci['hi']:.3f})"
    )
    print("With ~49 malignant test images the interval is wide by construction;")
    print("report it alongside the point estimate, never the point estimate alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
