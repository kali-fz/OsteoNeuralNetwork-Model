"""Score a checkpoint's explanation against ground-truth lesion boxes.

    python scripts/gradcam_report.py --checkpoint reports/train-.../best.pt
    python scripts/gradcam_report.py --checkpoint ... --chance-baseline

Produces two things: a folder of overlay PNGs, and the numbers that say whether
those overlays mean anything -- pointing-game accuracy and map-vs-box IoU.

A model can post high malignant recall while keying on an implant, a collimation
edge, or a burned-in laterality marker that happens to correlate with the
scanner used for sicker patients. Recall cannot distinguish that from real
lesion detection. Pointing-game accuracy can, which is why it is reported next
to it rather than left as a picture for someone to eyeball.

WHICH MAP GETS SCORED
---------------------
Whichever one the website would show, decided the same way ``onnm.inference``
decides it: a checkpoint carrying a lesion head is scored on the head's own map,
anything else on Grad-CAM. The file name is left alone because
``overnight_sweep.py``, ``daily_cycle.py`` and the version ledger all read
``gradcam_<split>/gradcam_report.json`` by that path; ``localisation.map_source``
in the JSON says which instrument produced the numbers.

On a lesion-head checkpoint Grad-CAM is scored as well, into
``localisation_gradcam``. That answers a question the head's own score cannot:
whether sharing a backbone with the lesion decoder moved the classifier's
attention too, or only added a second output.

READ THE CHANCE LEVEL FIRST
---------------------------
``chance_pointing_game`` is in every report and costs nothing: it is the mean
share of the frame the lesion boxes occupy, which is exactly how often a peak
dropped at random lands inside one. A pointing game of 0.09 against a chance
level of 0.10 is not a weak result, it is no result. ``--chance-baseline``
additionally scores this same architecture with random weights, which catches
what geometry cannot -- an untrained network still prefers the middle of a film,
and so do lesions.

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
    compute_lesion_map,
    coverage,
    evaluate_lesion_localisation,
    evaluate_localisation,
    evaluate_normal_activation,
    has_lesion_head,
    load_annotation,
    map_box_to_model_space,
    overlay_cam,
    pointing_game,
)
from onnm.model import build_model_for_checkpoint
from onnm.utils import ensure_dir, get_device, get_logger, save_json

logger = get_logger("gradcam")


def _print_scores(title: str, scores: dict) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)
    for key, value in scores.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  {key:<26}{formatted}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-overlays", type=int, default=24)
    parser.add_argument("--max-scored", type=int, default=None,
                        help="cap how many annotated images are scored")
    parser.add_argument(
        "--lesion-threshold", type=float, default=None,
        help=(
            "probability cut for the lesion map's IoU and coverage. Defaults to "
            "explain.cam_threshold. A CLI flag rather than a config key so that "
            "re-scoring a checkpoint never edits YAML the serving container reads."
        ),
    )
    parser.add_argument(
        "--chance-baseline", action="store_true",
        help=(
            "also score this architecture with RANDOM weights, which is the "
            "baseline every pointing-game number has to beat. One run of this "
            "covers a whole sweep: the weights are random, so the result depends "
            "on the architecture and the films, not on which checkpoint you "
            "pointed it at."
        ),
    )
    parser.add_argument(
        "--skip-cam-comparison", action="store_true",
        help=(
            "on a lesion-head checkpoint, do not additionally score Grad-CAM. "
            "Saves a backward pass per film; costs the answer to whether the "
            "shared backbone moved the classifier's attention."
        ),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    cfg = load_config(args.config, overrides=args.override)
    device = get_device()
    size = int(cfg.data.image_size)

    # Load first, then build what the checkpoint says it is. Building from the
    # YAML on disk and hoping it matches is what breaks the moment a checkpoint
    # carries a lesion head -- see model.build_model_for_checkpoint.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model_for_checkpoint(state, cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    records = build_records(cfg, split=args.split)

    # The website decides which map to show by asking the model whether it has a
    # decoder (onnm.inference.RadiographClassifier). Asking the same question
    # here, through the same helper, is what keeps the scored map and the served
    # map the same object rather than two things that happen to agree today.
    lesion = has_lesion_head(model)
    payload: dict = {}

    # --- Quantitative: score every annotated image ------------------------
    if lesion:
        print("Scoring the LESION HEAD's map against ground-truth lesion boxes...")
        scores = evaluate_lesion_localisation(
            model, cfg, records, device,
            max_samples=args.max_scored, threshold=args.lesion_threshold,
        )
    else:
        print("Scoring Grad-CAM against ground-truth lesion boxes...")
        scores = evaluate_localisation(
            model, cfg, records, device,
            class_index=MALIGNANT_INDEX, max_samples=args.max_scored,
        )
    payload["localisation"] = scores
    _print_scores(f"LOCALISATION ({scores['map_source']})", scores)
    print("\n  Pointing game = fraction of images whose hottest pixel falls inside")
    print("  the annotated lesion. Read it against chance_pointing_game directly")
    print("  above it, which is how often a peak dropped at random would land")
    print(f"  there on these same {scores['n_scored']} films.")

    if lesion:
        # The complaint that started this work, finally as a number. Scored on
        # the films _score_maps skips by design -- the normals -- so it adds a
        # population rather than re-reading the same one. See
        # explainability.evaluate_normal_activation for why this is lesion-head
        # only and what it deliberately does not claim.
        print("\nScoring the lesion map on NORMAL films (no lesion present)...")
        normals = evaluate_normal_activation(
            model, cfg, records, device,
            max_samples=args.max_scored, threshold=args.lesion_threshold,
        )
        payload["normal_activation"] = normals
        print(f"  films scored              : {normals['n_scored']}")
        print(
            f"  flagged somewhere         : {normals['n_flagged']} "
            f"({normals['flagged_fraction']:.1%})   <- lower is better"
        )
        print(f"  mean peak activation      : {normals['mean_max_activation']:.3f}")
        print(f"  median peak activation    : {normals['median_max_activation']:.3f}")
        print(f"  mean share of frame hot   : {normals['mean_positive_fraction']:.3%}")
        print("  Activation on healthy bone, not proof of joint attention: BTXRD")
        print("  labels no normal film with a specific joint, so the anatomy")
        print("  question stays open until external lower-limb normals are added.")

    if lesion and not args.skip_cam_comparison:
        # Not the headline number, and deliberately reported second: it says
        # whether the auxiliary loss moved the classifier, which is the claim the
        # lesion head is really making. A head that localises well while its
        # backbone's Grad-CAM is unchanged has added a picture, not understanding.
        print("\nAlso scoring Grad-CAM on the same checkpoint, for comparison...")
        cam_scores = evaluate_localisation(
            model, cfg, records, device,
            class_index=MALIGNANT_INDEX, max_samples=args.max_scored,
        )
        payload["localisation_gradcam"] = cam_scores
        _print_scores("LOCALISATION (gradcam, same checkpoint)", cam_scores)

    if args.chance_baseline:
        # Same architecture, same films, same scoring path -- weights never
        # loaded. build_model_for_checkpoint already forces pretrained=False, so
        # this is a genuinely untrained network rather than an ImageNet one.
        print("\nScoring a RANDOMLY-INITIALISED model of the same architecture...")
        untrained = build_model_for_checkpoint(state, cfg).to(device)
        untrained.eval()
        if lesion:
            payload["chance_model"] = evaluate_lesion_localisation(
                untrained, cfg, records, device,
                max_samples=args.max_scored, threshold=args.lesion_threshold,
            )
        else:
            payload["chance_model"] = evaluate_localisation(
                untrained, cfg, records, device,
                class_index=MALIGNANT_INDEX, max_samples=args.max_scored,
            )
        _print_scores("CHANCE BASELINE (random weights)", payload["chance_model"])
        del untrained

    # --- Qualitative: overlays for human review ---------------------------
    out_dir = ensure_dir(checkpoint.parent / f"gradcam_{args.split}")
    transform = build_transforms(cfg, "test", keep_meta=True)
    cam = None if lesion else build_cam(model, cfg)
    # Taken from the summary rather than re-read from the config, so --lesion-threshold
    # cannot cut the per-image rows at one value and the headline at another.
    threshold = float(scores["cam_threshold"])

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
        # The overlay must show the map that was scored, or the pictures and the
        # numbers describe different things and a human review of them proves
        # nothing about the report they sit next to.
        if lesion:
            heatmap = compute_lesion_map(model, image, size=size)
        else:
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
                "cam_iou": cam_iou(heatmap, gt_mask, threshold),
                "coverage": coverage(heatmap, gt_mask, threshold),
                "overlay": name,
            }
        )

    payload["overlays"] = manifest
    save_json(payload, out_dir / "gradcam_report.json")

    print(f"\nWrote {len(manifest)} overlays to {out_dir}")
    print(f"Green boxes are ground truth; the heat map is the {scores['map_source']}.")
    print("Review them: a high pointing-game score with visually wrong overlays")
    print("usually means the lesion boxes are large enough to be hit by accident.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
