"""Gate 6: prove the pipeline can learn before spending a day training.

    python scripts/overfit_check.py

Trains on a handful of images with augmentation off. The model must reach ~100%
training accuracy. If it cannot memorise 32 images, the problem is not the task
being hard -- it is labels misaligned with images, normalisation destroying the
signal, or gradients never reaching the backbone. This check finds that in two
minutes instead of after a full training run that "just isn't learning".
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (path side effect)
from onnm.config import load_config
from onnm.train import overfit_batch
from onnm.utils import describe_device, get_logger

logger = get_logger("overfit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--target", type=float, default=0.95)
    args = parser.parse_args()

    cfg = load_config(args.config, profile=args.profile)

    print("=" * 66)
    print("OVERFIT CHECK")
    print("=" * 66)
    for key, value in describe_device().items():
        print(f"  {key:<16}{value}")
    print(f"  {'model':<16}{cfg.model.name} (pretrained={cfg.model.pretrained})")
    print(f"  {'samples':<16}{args.samples}")
    print(f"  {'max steps':<16}{args.steps}")
    print(f"  {'target acc':<16}{args.target}")
    print()

    result = overfit_batch(
        cfg, n_samples=args.samples, steps=args.steps, target_accuracy=args.target
    )

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    for key, value in result.items():
        print(f"  {key:<18}{value}")

    if result["passed"]:
        print(f"\nPASS -- memorised {result['n_samples']} images in "
              f"{result['steps_run']} steps. The pipeline learns.")
        print("Next: python scripts/train.py")
        return 0

    print("\nFAIL -- the model could not memorise a tiny batch. Check, in order:")
    print("  1. Are labels aligned with images?  -> notebooks/01_data_sanity.ipynb")
    print("  2. Does normalisation destroy the signal?  -> plot a transformed sample")
    print("  3. Are gradients reaching the backbone?  -> check requires_grad")
    print("  4. Is the learning rate sane?  -> try --steps 500")
    return 1


if __name__ == "__main__":
    sys.exit(main())
