"""Fit temperature scaling and the decision threshold on the validation split.

    python scripts/calibrate.py --checkpoint reports/train-.../best.pt

Writes ``calibration.json`` next to the checkpoint. Both the Streamlit app and
``scripts/evaluate.py`` pick it up automatically from there, so this is the one
step between "the model is trained" and "the numbers it reports mean something".

Fitted on **validation**, never on test. The threshold is a free parameter; tune
it on the split you report and the reported specificity is fiction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (path side effect)
from onnm import CLASS_NAMES
from onnm.calibrate import (
    CALIBRATION_FILENAME,
    calibrate,
    collect_logits,
    format_report,
)
from onnm.config import load_config
from onnm.dataset import build_dataloader
from onnm.model import build_model
from onnm.utils import get_device, get_logger, save_json

logger = get_logger("calibrate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Leave this alone. --split test exists only for a deliberate, "
             "disclosed sanity check; a threshold fitted on test invalidates "
             "every number reported on test.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Also write the full threshold sweep, for plotting an ROC curve.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    if args.split == "test":
        print(
            "\n!! Fitting on TEST. Any metric you report from this split is now\n"
            "!! optimistically biased. Use this only as a disclosed diagnostic.\n"
        )

    cfg = load_config(args.config, overrides=args.override, profile=args.profile)
    device = get_device()

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    # shuffle=False so logits line up with labels in a stable order, which makes
    # a saved sweep reproducible run to run.
    loader = build_dataloader(cfg, args.split, shuffle=False)
    logger.info("collecting %s logits on %s", args.split, device)
    logits, labels = collect_logits(model, loader, device)

    normal_index = list(cfg.labels.classes).index("normal")
    result = calibrate(logits, labels, cfg, normal_index=normal_index)
    result.fitted_on = args.split

    print()
    print(format_report(result))

    # A model that has not learned the task produces a confident-looking
    # threshold that means nothing, so say so next to the number.
    recall = state.get("malignant_recall")
    if isinstance(recall, (int, float)) and recall < 0.5:
        print(
            f"\n  !! This checkpoint's val malignant recall is {recall:.3f}. Calibration\n"
            "  !! is a monotone rescaling -- it cannot fix a ranking this weak. Train\n"
            "  !! to convergence first; the threshold above will not survive it."
        )

    output = checkpoint.parent / CALIBRATION_FILENAME
    result.save(output)
    print(f"\nWrote {output}")
    print("The app and scripts/evaluate.py read this automatically.")

    if args.sweep:
        from onnm.calibrate import sweep_thresholds

        probabilities = torch.softmax(
            torch.as_tensor(logits) / result.temperature, dim=1
        ).numpy()
        sweep_path = checkpoint.parent / "threshold_sweep.json"
        save_json(
            {
                "classes": list(CLASS_NAMES),
                "temperature": result.temperature,
                "split": args.split,
                "sweep": sweep_thresholds(labels, probabilities, normal_index),
            },
            sweep_path,
        )
        print(f"Wrote {sweep_path}")

    # Non-zero exit on an unusable operating point, so a CI run or a shell
    # `&&` chain stops here rather than proceeding with a bad threshold.
    return 2 if result.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
