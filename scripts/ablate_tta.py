"""A/B one checkpoint with and without horizontal-flip test-time augmentation.

    python scripts/ablate_tta.py --checkpoint reports/full-.../best.pt

TTA-hflip averages each film's logits with its mirror image before the softmax.
On limb radiographs the flip is anatomically legitimate (a left femur is a
mirrored right femur), so this is a near-free variance reduction — worth
measuring on a test split whose malignant class holds only ~49 images.

Reports macro ROC-AUC and malignant ROC/PR-AUC for both arms on the chosen
split. Ranking metrics only: TTA changes the scores, so any threshold-dependent
number would need the calibration refitted (run scripts/calibrate.py through
``collect_logits(..., tta_hflip=True)`` before adopting TTA for deployment).

Needs the GPU stack and the BTXRD dataset on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (path side effect)
from onnm.calibrate import collect_logits
from onnm.config import load_config
from onnm.dataset import build_dataloader
from onnm.metrics import auc_scores
from onnm.model import build_model
from onnm.utils import get_device, get_logger

logger = get_logger("ablate_tta")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    cfg = load_config(args.config, overrides=args.override, profile=args.profile)
    device = get_device()

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    loader = build_dataloader(cfg, args.split, shuffle=False)

    rows = []
    for tta in (False, True):
        logits, labels = collect_logits(model, loader, device, tta_hflip=tta)
        probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
        scores = auc_scores(labels, probabilities)
        rows.append((tta, scores))

    print(f"\nTTA ablation on {args.split} ({checkpoint.parent.name})")
    print(f"  {'arm':<14}{'macro ROC-AUC':>14}{'mal ROC-AUC':>13}{'mal PR-AUC':>12}")
    for tta, scores in rows:
        print(
            f"  {'hflip TTA' if tta else 'baseline':<14}"
            f"{scores['roc_auc_macro']:>14.4f}"
            f"{scores['roc_auc']['malignant']:>13.4f}"
            f"{scores['pr_auc']['malignant']:>12.4f}"
        )

    delta = rows[1][1]["roc_auc_macro"] - rows[0][1]["roc_auc_macro"]
    print(f"\n  macro ROC-AUC delta: {delta:+.4f}")
    print(
        "  On ~49 malignant test images a delta inside ±0.01 is noise; check the val\n"
        "  split agrees before believing either direction."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
