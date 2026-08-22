"""Generate Grad-CAM overlays and score them against ground-truth lesion boxes.

    python scripts/gradcam_report.py --checkpoint reports/train-.../best.pt

Produces two things: a folder of overlay PNGs, and the numbers that say whether
those overlays mean anything -- pointing-game accuracy and CAM-vs-box IoU.

A model can post high malignant recall while keying on an implant, a collimation
edge, or a burned-in laterality marker that happens to correlate with the
scanner used for sicker patients. Recall cannot distinguish that from real
lesion detection. Pointing-game accuracy can, which is why it is reported next
to it rather than left as a picture for someone to eyeball.

LICENCE: overlays are derived images under BTXRD's CC BY-NC-ND terms. Keep them
local; do not redistribute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (path side effect)
from onnm import CLASS_NAMES, MALIGNANT_INDEX
from onnm.config import load_config
from onnm.dataset import build_records, build_transforms
from onnm.explainability import (
    annotation_path_for,
    boxes_to_mask,
    build_cam,
    cam_iou,
    compute_cam,
    coverage,
    evaluate_localisation,
    load_annotation,
    map_box_to_model_space,
    overlay_cam,
    pointing_game,
)
from onnm.model import build_model
from onnm.utils import ensure_dir, get_device, get_logger, save_json

logger = get_logger("gradcam")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-overlays", type=int, default=24)
    parser.add_argument("--max-scored", type=int, default=None,
                        help="cap how many annotated images are scored")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    cfg = load_config(args.config, overrides=args.override)
    device = get_device()
    size = int(cfg.data.image_size)

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    records = build_records(cfg, split=args.split)

    # --- Quantitative: score every annotated image ------------------------
    print("Scoring Grad-CAM against ground-truth lesion boxes...")
    scores = evaluate_localisation(
        model, cfg, records, device,
        class_index=MALIGNANT_INDEX, max_samples=args.max_scored,
    )

    print("\n" + "=" * 66)
    print("LOCALISATION")
    print("=" * 66)
    for key, value in scores.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  {key:<26}{formatted}")
    print("\n  Pointing game = fraction of images whose hottest CAM pixel falls")
    print("  inside the annotated lesion. Chance level depends on lesion size;")
    print("  compare against a randomly-initialised model to calibrate it.")

    # --- Qualitative: overlays for human review ---------------------------
    out_dir = ensure_dir(checkpoint.parent / f"gradcam_{args.split}")
    transform = build_transforms(cfg, "test", keep_meta=True)
    cam = build_cam(model, cfg)

    annotated = [
        r for r in records
        if annotation_path_for(r["image_id"], cfg).is_file()
        and r["label"] == MALIGNANT_INDEX
    ][: args.n_overlays]

    from PIL import Image

    manifest = []
    for record in annotated:
        annotation = load_annotation(annotation_path_for(record["image_id"], cfg))
        if not annotation["boxes"] or not annotation["height"]:
            continue

        mapped = [
            map_box_to_model_space(b, annotation["height"], annotation["width"], size)
            for b in annotation["boxes"]
        ]
        gt_mask = boxes_to_mask(mapped, size)

        sample = transform(record)
        image = sample["image"].unsqueeze(0).to(device)
        heatmap = compute_cam(cam, image, MALIGNANT_INDEX)

        with torch.no_grad():
            probs = torch.softmax(model(image).float(), dim=1)[0].cpu().numpy()

        overlay = overlay_cam(sample["image"][0].numpy(), heatmap, boxes=mapped)
        name = f"{record['image_id']}_p{probs[MALIGNANT_INDEX]:.2f}.png"
        Image.fromarray(overlay).save(out_dir / name)

        manifest.append(
            {
                "image_id": record["image_id"],
                "true_class": CLASS_NAMES[record["label"]],
                "predicted_class": CLASS_NAMES[int(probs.argmax())],
                "malignant_probability": float(probs[MALIGNANT_INDEX]),
                "pointing_game_hit": pointing_game(heatmap, gt_mask),
                "cam_iou": cam_iou(heatmap, gt_mask, float(cfg.explain.cam_threshold)),
                "coverage": coverage(heatmap, gt_mask, float(cfg.explain.cam_threshold)),
                "overlay": name,
            }
        )

    save_json({"localisation": scores, "overlays": manifest},
              out_dir / "gradcam_report.json")

    print(f"\nWrote {len(manifest)} overlays to {out_dir}")
    print("Green boxes are ground truth; the heatmap is the model's attention.")
    print("Review them: a high pointing-game score with visually wrong overlays")
    print("usually means the lesion boxes are large enough to be hit by accident.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
