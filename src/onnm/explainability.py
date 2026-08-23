"""Grad-CAM, and the ground truth that turns it into a number.

A heatmap that looks convincing is not evidence. BTXRD ships a bounding box and
a segmentation polygon for every annotated lesion, so the model's attention can
be *scored* rather than admired:

* **pointing game** -- does the CAM's peak land inside the true lesion box?
* **CAM-vs-box IoU** -- how well does the thresholded CAM cover the lesion?

That distinction matters clinically. A model can reach high malignant recall
while keying on an implant, a collimation edge, or a burned-in laterality
marker that happens to correlate with the scanner that imaged the sick patients.
Recall alone cannot tell those cases apart; pointing-game accuracy can.

The geometric mapping in :func:`map_box_to_model_space` must mirror
``onnm.dataset.build_transforms`` exactly. If the two ever disagree, every
localisation number silently becomes meaningless, so ``tests/test_explainability.py``
pins them together, and :func:`evaluate_localisation` refuses to run when
``data.crop_foreground`` has introduced a geometry stage the mapping does not
model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import get_logger

logger = get_logger(__name__)

Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


# ---------------------------------------------------------------------------
# Annotations (LabelMe 5.x format)
# ---------------------------------------------------------------------------
def load_annotation(path: str | Path) -> dict[str, Any]:
    """Parse one BTXRD annotation file.

    Each tumour image carries a ``rectangle`` (bounding box) and usually a
    ``polygon`` (segmentation mask) per lesion, in *original* image coordinates.
    Polygons are converted to their enclosing box so both shape types yield a
    comparable region.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("%s: unreadable annotation (%s)", path.name, exc)
        return {"boxes": [], "labels": [], "height": 0, "width": 0}

    boxes: list[Box] = []
    labels: list[str] = []

    for shape in data.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float64)
        if points.size == 0:
            continue
        shape_type = shape.get("shape_type")
        if shape_type not in ("rectangle", "polygon"):
            continue
        # A rectangle stores two opposite corners, which are not guaranteed to be
        # top-left then bottom-right; min/max handles both, and also converts a
        # polygon to its enclosing box.
        boxes.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
        labels.append(str(shape.get("label", "")))

    return {
        "boxes": boxes,
        "labels": labels,
        "height": int(data.get("imageHeight", 0)),
        "width": int(data.get("imageWidth", 0)),
        "image_path": data.get("imagePath", ""),
    }


def annotation_path_for(image_id: str, cfg) -> Path:
    data_root = cfg.resolve_path("paths.data_root")
    return data_root / cfg.paths.annotations_dirname / f"{Path(image_id).stem}.json"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def map_box_to_model_space(
    box: Box, orig_h: int, orig_w: int, size: int
) -> Box:
    """Map a box from original image coordinates into the 256x256 model input.

    Mirrors the two geometric stages of the transform chain:

    1. ``Resized(size_mode="longest")`` scales by ``size / max(H, W)``.
    2. ``ResizeWithPadOrCropd`` pads symmetrically, floor-biased to the leading
       edge -- which is what MONAI's ``SpatialPad`` does with ``method="symmetric"``.
    """
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"invalid original dimensions: {orig_h}x{orig_w}")

    scale = size / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2

    x_min, y_min, x_max, y_max = box
    return (
        float(np.clip(x_min * scale + pad_left, 0, size)),
        float(np.clip(y_min * scale + pad_top, 0, size)),
        float(np.clip(x_max * scale + pad_left, 0, size)),
        float(np.clip(y_max * scale + pad_top, 0, size)),
    )


def boxes_to_mask(boxes: list[Box], size: int) -> np.ndarray:
    """Rasterise boxes (already in model space) to a boolean mask."""
    mask = np.zeros((size, size), dtype=bool)
    for x_min, y_min, x_max, y_max in boxes:
        c0, c1 = int(np.floor(x_min)), int(np.ceil(x_max))
        r0, r1 = int(np.floor(y_min)), int(np.ceil(y_max))
        mask[max(r0, 0) : min(r1, size), max(c0, 0) : min(c1, size)] = True
    return mask


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def build_cam(model: torch.nn.Module, cfg):
    """Construct a MONAI Grad-CAM bound to the configured target layer."""
    from monai.visualize import GradCAM, GradCAMpp

    from .model import get_cam_layer

    layer = get_cam_layer(model, cfg)
    method = str(cfg.explain.get("method", "gradcam")).lower()
    factory = GradCAMpp if method == "gradcampp" else GradCAM
    logger.info("Grad-CAM: method=%s target_layer=%s", method, layer)
    return factory(nn_module=model, target_layers=layer)


def compute_cam(cam, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
    """Return a ``(H, W)`` CAM in [0, 1] for one image tensor ``(1, C, H, W)``."""
    result = cam(x=image, class_idx=class_index)
    array = result.detach().cpu().numpy()
    while array.ndim > 2:
        array = array[0]

    lo, hi = float(array.min()), float(array.max())
    return (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
#: Above this share of the frame sitting at the CAM maximum, "the hottest pixel"
#: has no meaning and the pointing game is not measuring localisation.
#:
#: 0.05 is generous: a well-behaved Grad-CAM on a 256x256 input peaks over a few
#: hundred pixels at most, so this fires only on genuinely flat maps.
DEGENERATE_PEAK_FRACTION = 0.05


def peak_fraction(cam: np.ndarray, tolerance: float = 1e-6) -> float:
    """Share of the frame sitting at the CAM's maximum.

    A localising CAM has one peak and this is near zero. A saturated one --
    which is what a badly chosen target layer produces -- can have most of the
    frame tied at the maximum, and every downstream "where does it look"
    question then has no answer.

    Reported alongside the scores rather than hidden, because a degenerate CAM
    and a model that looks in the wrong place produce identical-looking numbers
    and require completely different fixes.
    """
    if cam.size == 0:
        return float("nan")
    return float((cam >= cam.max() - tolerance).sum() / cam.size)


def pointing_game(cam: np.ndarray, gt_mask: np.ndarray) -> bool:
    """True when the CAM's hottest point falls inside the lesion.

    The peak is the **centroid of the maximal region**, not ``argmax``.

    That distinction is not pedantry. ``np.argmax`` returns the first maximal
    element in raster order, so on a CAM where thousands of pixels tie at the
    maximum it reports the top-left corner of the plateau every single time --
    which is background on essentially every radiograph. Scoring the pinned
    checkpoint that way produced a pointing-game accuracy of exactly 0.0000
    across 267 images, a number that looked like a devastating result about the
    model and was in fact a statement about tie-breaking.

    The centroid is well defined whether the maximum is one pixel or half the
    frame, and reduces to ``argmax`` when the peak is unique. It does not rescue
    a degenerate CAM -- nothing can -- but it fails honestly instead of
    reporting a confident zero. Use :func:`peak_fraction` to tell the two apart.
    """
    if not gt_mask.any():
        return False
    hottest = cam >= cam.max() - 1e-6
    rows, cols = np.nonzero(hottest)
    if len(rows) == 0:
        return False
    peak = (int(round(rows.mean())), int(round(cols.mean())))
    return bool(gt_mask[peak])


def cam_iou(cam: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between the thresholded CAM and the lesion mask."""
    if not gt_mask.any():
        return float("nan")
    predicted = cam >= threshold
    union = np.logical_or(predicted, gt_mask).sum()
    return float(np.logical_and(predicted, gt_mask).sum() / union) if union else 0.0


def coverage(cam: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of CAM mass falling inside the lesion.

    Complements IoU: a CAM that correctly highlights the lesion but also lights
    up half the film scores poor IoU yet still concentrates most of its mass in
    the right place, and the two numbers together say which failure it is.
    """
    total = cam.sum()
    return float(cam[gt_mask].sum() / total) if total > 0 else float("nan")


def evaluate_localisation(
    model: torch.nn.Module,
    cfg,
    records: list[dict],
    device: torch.device,
    class_index: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Score Grad-CAM against ground-truth boxes over annotated records.

    Only records with an annotation file are scored -- normal images have no
    lesion, so localisation is undefined for them.
    """
    from .dataset import build_transforms

    # `map_box_to_model_space` reproduces exactly two geometric stages: resize
    # the longest side, then pad symmetrically. Foreground cropping inserts a
    # third, with an offset that is different for every image, so the mapping
    # silently stops describing the transform chain. Every pointing-game hit and
    # IoU below would still be a number, and all of them would be wrong -- so
    # this refuses rather than reports.
    if bool(cfg.data.get("crop_foreground", False)):
        raise ValueError(
            "data.crop_foreground is enabled, which changes the geometry between "
            "the original image and the model input. Ground-truth lesion-box "
            "scoring cannot be mapped through it, so these metrics would be "
            "meaningless. Either evaluate localisation with crop_foreground: "
            "false, or extend map_box_to_model_space to carry the per-image crop "
            "offset first."
        )

    size = int(cfg.data.image_size)
    threshold = float(cfg.explain.get("cam_threshold", 0.5))
    transform = build_transforms(cfg, "test", keep_meta=True)
    cam = build_cam(model, cfg)
    model.eval()

    hits: list[bool] = []
    ious: list[float] = []
    coverages: list[float] = []
    peaks: list[float] = []
    n_skipped = 0

    for record in records[:max_samples] if max_samples else records:
        annotation = load_annotation(annotation_path_for(record["image_id"], cfg))
        if not annotation["boxes"] or not annotation["height"]:
            n_skipped += 1
            continue

        mapped = [
            map_box_to_model_space(b, annotation["height"], annotation["width"], size)
            for b in annotation["boxes"]
        ]
        gt_mask = boxes_to_mask(mapped, size)

        sample = transform(record)
        image = sample["image"].unsqueeze(0).to(device)
        heatmap = compute_cam(cam, image, class_index)

        hits.append(pointing_game(heatmap, gt_mask))
        ious.append(cam_iou(heatmap, gt_mask, threshold))
        coverages.append(coverage(heatmap, gt_mask, threshold))
        peaks.append(peak_fraction(heatmap))

    mean_peak = float(np.mean(peaks)) if peaks else float("nan")
    degenerate = bool(mean_peak >= DEGENERATE_PEAK_FRACTION)
    if degenerate:
        # Loud, because every score in this dict is then a number about the CAM's
        # shape rather than about where the model looks, and they do not read as
        # obviously wrong -- they read as a bad model.
        logger.warning(
            "Grad-CAM is degenerate: on average %.1f%% of each map sits at its "
            "maximum. Localisation scores below describe a saturated heatmap, not "
            "the model's attention. Check explain.target_layer -- for DenseNet, "
            "features.denseblock4 is the raw block output, before the final norm.",
            100 * mean_peak,
        )

    return {
        "n_scored": len(hits),
        "n_skipped": n_skipped,
        "pointing_game_accuracy": float(np.mean(hits)) if hits else float("nan"),
        "mean_iou": float(np.nanmean(ious)) if ious else float("nan"),
        "mean_coverage": float(np.nanmean(coverages)) if coverages else float("nan"),
        "mean_peak_fraction": mean_peak,
        "cam_degenerate": degenerate,
        "cam_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def overlay_cam(
    image: np.ndarray, cam: np.ndarray, alpha: float = 0.4, boxes: list[Box] | None = None
) -> np.ndarray:
    """Blend a CAM over a grayscale image, optionally drawing ground-truth boxes.

    Returns a uint8 RGB array. Note the CC BY-NC-ND licence on BTXRD: overlays
    are derived images and must not be redistributed.
    """
    import matplotlib.cm as cm

    grey = image.astype(np.float64)
    lo, hi = grey.min(), grey.max()
    grey = (grey - lo) / (hi - lo) if hi > lo else np.zeros_like(grey)
    rgb = np.stack([grey] * 3, axis=-1)

    heat = cm.jet(np.clip(cam, 0, 1))[..., :3]
    blended = (1 - alpha) * rgb + alpha * heat

    if boxes:
        for x_min, y_min, x_max, y_max in boxes:
            c0, c1 = int(x_min), int(min(x_max, blended.shape[1] - 1))
            r0, r1 = int(y_min), int(min(y_max, blended.shape[0] - 1))
            for row in (r0, r1):
                if 0 <= row < blended.shape[0]:
                    blended[row, c0 : c1 + 1] = [0.0, 1.0, 0.0]
            for col in (c0, c1):
                if 0 <= col < blended.shape[1]:
                    blended[r0 : r1 + 1, col] = [0.0, 1.0, 0.0]

    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
