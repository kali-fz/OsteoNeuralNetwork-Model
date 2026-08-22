"""Train the classifier.

    python scripts/train.py
    python scripts/train.py --profile smoke          # 1 epoch, small batch
    python scripts/train.py --override configs/densenet121_3class.yaml

Run the gates first -- verify_env, verify_data, make_splits, pytest, and
overfit_check -- because every one of them fails faster than a training run does.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (path side effect)
from onnm.config import load_config
from onnm.train import train
from onnm.utils import add_file_log, describe_device, get_logger, run_dir, save_json

logger = get_logger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--tag", default="train")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override, profile=args.profile)
    if args.epochs is not None:
        cfg._data["train"]["epochs"] = args.epochs

    output_dir = run_dir(cfg.resolve_path("paths.reports_dir"), args.tag)
    add_file_log(output_dir / "train.log")

    logger.info("output directory: %s", output_dir)
    for key, value in describe_device().items():
        logger.info("  %-16s%s", key, value)

    if not describe_device().get("cuda_available"):
        logger.warning(
            "No GPU detected. A full run on CPU will take many hours. Consider "
            "notebooks/kaggle_train.ipynb, or --profile smoke to validate the loop."
        )

    save_json(cfg.to_dict(), output_dir / "config.json")
    result = train(cfg, output_dir)
    save_json(
        {k: v for k, v in result.items() if k != "history"},
        output_dir / "summary.json",
    )

    print(f"\nBest {cfg.train.early_stopping_metric} = {result['best_score']:.4f} "
          f"at epoch {result['best_epoch']}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    print(f"\nNext: python scripts/evaluate.py --checkpoint {output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    # Required on Windows: DataLoader workers are spawned, not forked, and each
    # one re-imports this module.
    sys.exit(main())
