"""Per-anatomy and per-subtype error breakdown for one checkpoint.

    python scripts/stratified_report.py --checkpoint reports/full-.../best.pt

Answers two specific questions the aggregate metrics average away:

* which anatomy regions produce the false positives -- the standing complaint
  is complex joint anatomy (pelvis, hips, growth plates), and BTXRD's metadata
  has one-hot anatomy columns, so the hypothesis is directly testable;
* which tumour subtypes the model misses -- osteosarcoma vs the other
  malignancies vs each benign subtype.

Uses the calibrated operating point beside the checkpoint (calibration.json)
when present, so the strata are scored exactly as the app would score them.
Writes ``stratified_<split>.json`` next to the checkpoint.

Needs the GPU stack and the BTXRD dataset on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401  (path side effect)
from onnm.calibrate import Calibration, collect_logits, lesion_scores
from onnm.config import load_config
from onnm.dataset import build_dataloader, build_records, read_table, resolve_column
from onnm.metrics import stratified_metrics
from onnm.model import build_model
from onnm.utils import get_device, get_logger, save_json

logger = get_logger("stratified_report")


def _indicator_map(df, id_column: str, columns: list[str]) -> dict[str, str]:
    """image_id stem -> name of the first indicator column set on that row."""
    resolved = [(c, resolve_column(df, c)) for c in columns]
    resolved = [(name, col) for name, col in resolved if col is not None]

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        stem = Path(str(row[id_column]).strip()).stem
        for name, col in resolved:
            try:
                flag = float(row[col])
            except (TypeError, ValueError):
                continue
            if flag == 1.0:
                out[stem] = name
                break
    return out


def _format_table(title: str, report: dict[str, dict[str, Any]]) -> str:
    lines = [
        "",
        "=" * 78,
        title,
        "=" * 78,
        f"  {'stratum':<24}{'n':>5}{'sens':>8}{'spec':>8}{'FP':>5}{'missed':>8}"
        f"{'mal.rec':>9}",
    ]
    # False-positive-heavy strata first: that is the question being asked.
    ranked = sorted(
        report.items(),
        key=lambda kv: -(kv[1]["false_positive_rate"]
                         if kv[1]["false_positive_rate"] == kv[1]["false_positive_rate"]
                         else -1.0),
    )
    for stratum, row in ranked:
        flag = "  (low n!)" if row["low_support"] else ""
        lines.append(
            f"  {stratum:<24}{row['n']:>5}"
            f"{row['sensitivity']:>8.3f}{row['specificity']:>8.3f}"
            f"{row['false_positives']:>5}{row['missed_lesions']:>8}"
            f"{row['malignant_recall']:>9.3f}{flag}"
        )
    return "\n".join(lines)


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
    device = get_device()

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    # shuffle=False keeps loader order identical to record order, which is what
    # lets per-sample metadata line up with per-sample predictions.
    records = build_records(cfg, split=args.split)
    loader = build_dataloader(cfg, args.split, records=records, shuffle=False)
    logits, labels = collect_logits(model, loader, device)
    if len(labels) != len(records):
        print(f"ERROR: {len(records)} records but {len(labels)} predictions; aborting")
        return 1

    calibration = Calibration.for_checkpoint(checkpoint) or Calibration()
    probabilities = calibration.apply(torch.as_tensor(logits)).numpy()
    normal_index = list(cfg.labels.classes).index("normal")

    # Decide exactly as the app decides: calibrated lesion score vs threshold,
    # argmax between benign/malignant only to pick which lesion class.
    scores = lesion_scores(probabilities, normal_index)
    lesion_pred = scores >= calibration.lesion_threshold
    argmax_pred = probabilities.argmax(axis=1)
    y_pred = np.where(
        lesion_pred,
        np.where(argmax_pred == normal_index, probabilities[:, 1:].argmax(axis=1) + 1,
                 argmax_pred),
        normal_index,
    )

    df = read_table(cfg)
    id_col = resolve_column(df, cfg.labels.id_column, required=True)
    anatomy_of = _indicator_map(df, id_col, list(cfg.columns.anatomy))
    subtype_of = _indicator_map(
        df, id_col,
        list(cfg.labels.subtype_columns.benign) + list(cfg.labels.subtype_columns.malignant),
    )

    anatomy = [anatomy_of.get(r["image_id"], "unknown") for r in records]
    subtype = [
        subtype_of.get(r["image_id"], "none recorded" if r["label"] == 0 else "unlabelled")
        for r in records
    ]

    anatomy_report = stratified_metrics(labels, y_pred, anatomy, normal_index)
    subtype_report = stratified_metrics(labels, y_pred, subtype, normal_index)

    print(_format_table(f"PER-ANATOMY ({args.split}, threshold "
                        f"{calibration.lesion_threshold:.3f})", anatomy_report))
    print(_format_table(f"PER-SUBTYPE ({args.split})", subtype_report))
    print(
        "\nRead false positives per anatomy against its normal count, and treat any\n"
        "stratum marked (low n!) as an anecdote, not a finding."
    )

    output = checkpoint.parent / f"stratified_{args.split}.json"
    save_json(
        {
            "split": args.split,
            "threshold": calibration.lesion_threshold,
            "temperature": calibration.temperature,
            "per_anatomy": anatomy_report,
            "per_subtype": subtype_report,
        },
        output,
    )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
