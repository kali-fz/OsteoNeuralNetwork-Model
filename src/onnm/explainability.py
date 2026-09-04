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


def load_polygons(path: str | Path) -> dict[str, Any]:
    """Parse one annotation file keeping polygons as polygons.

    :func:`load_annotation` reduces every shape to its enclosing box, which is
    right for *scoring* -- the published localisation numbers are box-based and
    have to stay comparable -- and wrong for *supervision*. A box around an
    irregular lesion is roughly 40% normal bone (measured: polygon area averages
    0.606 of box area across 374 annotations), so training a segmentation head on
    boxes teaches it to outline rectangles.

    Note that ``load_annotation`` emits a box for the rectangle shape *and*
    another for the polygon's extent, which is harmless there because
    ``boxes_to_mask`` unions them. Here that would double-count, so rectangles
    are only used when a lesion has no polygon at all.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("%s: unreadable annotation (%s)", path.name, exc)
        return {"polygons": [], "height": 0, "width": 0}

    polygons: list[np.ndarray] = []
    rectangles: list[np.ndarray] = []
    for shape in data.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float64)
        if points.size == 0:
            continue
        if shape.get("shape_type") == "polygon":
            polygons.append(points)
        elif shape.get("shape_type") == "rectangle":
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()
            rectangles.append(
                np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float64,
                )
            )

    return {
        "polygons": polygons or rectangles,
        "height": int(data.get("imageHeight", 0)),
        "width": int(data.get("imageWidth", 0)),
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
def _identity_postprocessing(acti_map):
    """Return the CAM untouched. **Do not delete this as redundant.**

    MONAI's ``CAMBase`` defaults ``postprocessing=default_normalizer``, which is
    not the innocent rescale the name suggests. It maps ``(min, max) -> (1, 0)``,
    and its own docstring says so: "This will flip magnitudes (i.e., smallest
    will become biggest and vice versa)."

    That inversion shipped in this project until 2026-08-23 and made the heatmap
    exactly wrong -- correlation -1.0000 against the correct map, every pixel.
    Because Grad-CAM ends in a ReLU, most of a normal map is zero; flipping it
    turned that zero region into the "hottest" evidence, so the overlay painted
    the background and the zero-padding band red and left the lesion cold. It
    also produced the "degenerate/saturated CAM" that was chased for a while as
    a target-layer problem, which it never was.

    ``compute_cam`` already scales min->0 and max->1, so this is the only
    normalisation the pipeline needs. Passing it explicitly is what keeps
    MONAI's default from silently reintroducing the flip.
    """
    return acti_map


def build_cam(model: torch.nn.Module, cfg):
    """Construct a MONAI Grad-CAM bound to the configured target layer.

    ``postprocessing`` is passed explicitly -- see :func:`_identity_postprocessing`
    for why the default is wrong. ``GradCAMpp`` subclasses ``GradCAM`` and
    inherits the same default, so both methods need it.
    """
    from monai.visualize import GradCAM, GradCAMpp

    from .model import get_cam_layer

    layer = get_cam_layer(model, cfg)
    method = str(cfg.explain.get("method", "gradcam")).lower()
    factory = GradCAMpp if method == "gradcampp" else GradCAM
    logger.info("Grad-CAM: method=%s target_layer=%s", method, layer)
    return factory(
        nn_module=model, target_layers=layer, postprocessing=_identity_postprocessing
    )


def compute_cam(cam, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
    """Return a ``(H, W)`` CAM in [0, 1] for one image tensor ``(1, C, H, W)``.

    The sole normalisation step: the raw map arrives unscaled (see
    :func:`_identity_postprocessing`) and leaves with its minimum at 0 and its
    maximum at 1, **orientation preserved** -- high means "this drove the
    prediction".

    A map with no positive evidence at all is uniformly zero after Grad-CAM's
    ReLU. It returns all zeros rather than all ones, which is an honest empty
    heatmap instead of a uniformly hot one.
    """
    result = cam(x=image, class_idx=class_index)
    array = result.detach().cpu().numpy()
    while array.ndim > 2:
        array = array[0]

    lo, hi = float(array.min()), float(array.max())
    return (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)


# ---------------------------------------------------------------------------
# The lesion head's own map
# ---------------------------------------------------------------------------
def has_lesion_head(model: torch.nn.Module) -> bool:
    """True when this model carries a lesion decoder.

    The same test ``RadiographClassifier`` uses to decide which map to serve
    (``inference.py``), kept in one place so the scorer and the website can never
    disagree about which explanation a given checkpoint produces.
    """
    return hasattr(model, "seg_head")


def upsample_map(array: np.ndarray, size: int) -> np.ndarray:
    """Bring a decoder output up to the ``size x size`` model-input grid.

    The head predicts at 64x64 to stay cheap on half a vCPU, while the
    ground-truth mask is rasterised at the model input size -- so the two must be
    brought onto one grid before any of them can be compared.

    Upsampling the prediction is the right direction rather than downsampling the
    truth: it is what ``onnm.inference._resize_map`` does before the overlay is
    drawn, so the map being scored here is pixel-for-pixel the map a visitor sees.
    Bilinear with ``align_corners=False`` is the same convention as that
    function's ``cv2.INTER_LINEAR`` (both use half-pixel centres);
    ``tests/test_lesion_localisation.py`` pins the two together rather than
    trusting the claim.
    """
    if array.shape == (size, size):
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def compute_lesion_map(
    model: torch.nn.Module, image: torch.Tensor, size: int | None = None
) -> np.ndarray:
    """Return the lesion head's probability map for one image tensor ``(1, C, H, W)``.

    The counterpart of :func:`compute_cam`, and deliberately NOT rescaled the way
    that function rescales a CAM. A Grad-CAM is an unbounded attribution whose
    absolute magnitude means nothing, so min-max scaling it is the only way to
    read it; a sigmoid is already a probability per pixel, and stretching it to
    [0, 1] would turn "0.02 everywhere, nothing here" into a full-range heatmap
    claiming a lesion on a clean film. The map is returned as the model states it.

    ``return_mask`` is restored to whatever it was rather than forced to False,
    so calling this inside a loop cannot silently change the behaviour of
    whatever set it -- MONAI's Grad-CAM indexes ``logits[:, class_idx]`` and
    would fail on a tuple.
    """
    if not has_lesion_head(model):
        raise ValueError(
            "this checkpoint has no lesion head, so it produces no lesion map. "
            "Score it with evaluate_localisation (Grad-CAM) instead."
        )

    had_mask_flag = getattr(model, "return_mask", False)
    model.return_mask = True
    try:
        with torch.no_grad():
            _, mask_logits = model(image)
    finally:
        model.return_mask = had_mask_flag

    array = torch.sigmoid(mask_logits.float())[0, 0].detach().cpu().numpy()
    return upsample_map(array, size) if size else array


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


def _require_scorable_geometry(cfg) -> None:
    """Refuse to score when the transform chain has a geometry stage we cannot map.

    ``map_box_to_model_space`` reproduces exactly two geometric stages: resize the
    longest side, then pad symmetrically. Foreground cropping inserts a third,
    with an offset that is different for every image, so the mapping silently
    stops describing the transform chain. Every pointing-game hit and IoU would
    still be a number, and all of them would be wrong -- so this refuses rather
    than reports.

    Checked before the model is touched, so a misconfiguration costs a clear
    error rather than a wasted pass over the split.
    """
    if bool(cfg.data.get("crop_foreground", False)):
        raise ValueError(
            "data.crop_foreground is enabled, which changes the geometry between "
            "the original image and the model input. Ground-truth lesion-box "
            "scoring cannot be mapped through it, so these metrics would be "
            "meaningless. Either evaluate localisation with crop_foreground: "
            "false, or extend map_box_to_model_space to carry the per-image crop "
            "offset first."
        )


def _score_maps(
    map_for,
    cfg,
    records: list[dict],
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Score any per-image heat map against the ground-truth lesion boxes.

    The shared engine behind :func:`evaluate_localisation` and
    :func:`evaluate_lesion_localisation`. It exists so that a Grad-CAM and a
    lesion map are scored by *identical* code over an identical population --
    same records, same box rasterisation, same threshold, same metric
    definitions. Two scoring loops that were meant to match and drifted would
    make every comparison between the two explanations worthless, and the drift
    would not be visible in either number on its own.

    ``map_for`` takes the model-input tensor ``(1, C, H, W)`` and returns a
    ``(size, size)`` map. Only records carrying an annotation file are scored:
    a normal film has no lesion, so localisation is undefined for it.
    """
    from .dataset import build_transforms

    size = int(cfg.data.image_size)
    threshold = float(cfg.explain.get("cam_threshold", 0.5))
    transform = build_transforms(cfg, "test", keep_meta=True)

    hits: list[bool] = []
    ious: list[float] = []
    coverages: list[float] = []
    peaks: list[float] = []
    maxima: list[float] = []
    positives: list[float] = []
    lesion_fractions: list[float] = []
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
        heatmap = map_for(image)

        hits.append(pointing_game(heatmap, gt_mask))
        ious.append(cam_iou(heatmap, gt_mask, threshold))
        coverages.append(coverage(heatmap, gt_mask, threshold))
        peaks.append(peak_fraction(heatmap))
        maxima.append(float(heatmap.max()))
        positives.append(float((heatmap >= threshold).mean()))
        # The share of the frame the lesion occupies IS the pointing game's
        # chance level: a peak dropped uniformly at random lands inside a box
        # covering 10% of the film 10% of the time. Accumulated per image and
        # averaged, so the baseline describes this exact population rather than
        # a remembered figure from another split.
        lesion_fractions.append(float(gt_mask.mean()))

    def _mean(values: list[float]) -> float:
        return float(np.nanmean(values)) if values else float("nan")

    return {
        "n_scored": len(hits),
        "n_skipped": n_skipped,
        "pointing_game_accuracy": float(np.mean(hits)) if hits else float("nan"),
        "mean_iou": _mean(ious),
        "mean_coverage": _mean(coverages),
        "mean_peak_fraction": _mean(peaks),
        "mean_max_value": _mean(maxima),
        "mean_positive_fraction": _mean(positives),
        "n_below_threshold": int(sum(1 for m in maxima if m < threshold)),
        "chance_pointing_game": _mean(lesion_fractions),
        "mean_lesion_fraction": _mean(lesion_fractions),
        "cam_threshold": threshold,
    }


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

    ``chance_pointing_game`` is reported alongside the score and should be read
    first. Pointing game is not a percentage out of 100: a lesion box covering a
    tenth of the film is hit a tenth of the time by a peak dropped at random, so
    the number only means something next to that baseline.
    """
    _require_scorable_geometry(cfg)

    cam = build_cam(model, cfg)
    model.eval()
    scores = _score_maps(
        lambda image: compute_cam(cam, image, class_index),
        cfg, records, device, max_samples=max_samples,
    )

    mean_peak = scores["mean_peak_fraction"]
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

    # The sigmoid diagnostics are dropped rather than reported as nulls:
    # `compute_cam` rescales every map to [0, 1], so `mean_max_value` is 1.0 by
    # construction and `n_below_threshold` counts nothing. Both are real
    # measurements of a probability map and noise on a rescaled one.
    for key in ("mean_max_value", "mean_positive_fraction", "n_below_threshold"):
        scores.pop(key, None)

    scores["map_source"] = "gradcam"
    scores["cam_degenerate"] = degenerate
    return scores


def evaluate_lesion_localisation(
    model: torch.nn.Module,
    cfg,
    records: list[dict],
    device: torch.device,
    max_samples: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score the LESION HEAD's own map against the same ground-truth boxes.

    WHY THIS IS SEPARATE FROM THE GRAD-CAM SCORE
    --------------------------------------------
    Until this existed, ``scripts/gradcam_report.py`` scored Grad-CAM on every
    checkpoint, lesion head or not -- so a sweep run carrying the new head was
    measured by the instrument it was built to replace, and a flat pointing-game
    column would have read as "the head does not work" when it in fact measured
    nothing about the head at all.

    WHAT IS AND IS NOT COMPARABLE WITH THE GRAD-CAM NUMBERS
    -------------------------------------------------------
    ``pointing_game_accuracy`` is directly comparable: it is threshold-free, and
    both maps are scored over the same films against the same boxes by
    :func:`_score_maps`.

    ``mean_iou`` and ``mean_coverage`` are NOT, and the reason is the threshold.
    ``compute_cam`` min-max rescales a CAM so that every film has a pixel at 1.0
    and 0.5 means "half as hot as this image's hottest point" -- a per-image
    relative cut. A sigmoid is an absolute probability, so 0.5 means "the model
    puts even odds on lesion here", and a film the model considers clean can sit
    entirely below it. That is a real property worth measuring, not a defect,
    which is why ``n_below_threshold`` is reported next to it.

    ``cam_degenerate`` is deliberately absent. It asks whether the frame is tied
    at the maximum, which is a saturation failure specific to a rescaled CAM; a
    sigmoid varies continuously and would essentially never trip it. The
    equivalent questions for a probability map are "does it predict nothing
    anywhere" (``n_below_threshold``, ``mean_max_value``) and "does it predict
    everywhere" (``mean_positive_fraction``), and those are reported instead.
    """
    _require_scorable_geometry(cfg)

    if not has_lesion_head(model):
        raise ValueError(
            "this checkpoint carries no lesion head, so there is no lesion map to "
            "score. Use evaluate_localisation for a Grad-CAM checkpoint."
        )

    size = int(cfg.data.image_size)
    if threshold is not None:
        # Kept as an override rather than a new config key, so no YAML that the
        # serving container reads has to change to re-score a checkpoint.
        cfg = _with_cam_threshold(cfg, threshold)

    model.eval()
    scores = _score_maps(
        lambda image: compute_lesion_map(model, image, size=size),
        cfg, records, device, max_samples=max_samples,
    )
    scores["map_source"] = "lesion_head"

    if scores["n_scored"] and scores["n_below_threshold"] == scores["n_scored"]:
        logger.warning(
            "the lesion head never exceeds %.2f on any of the %d scored films "
            "(mean maximum %.3f). mean_iou is 0 by construction and says nothing "
            "about localisation; read pointing_game_accuracy, which is "
            "threshold-free, and consider a lower threshold for the IoU.",
            scores["cam_threshold"], scores["n_scored"], scores["mean_max_value"],
        )
    return scores


def _with_cam_threshold(cfg, threshold: float):
    """Return a copy of ``cfg`` with ``explain.cam_threshold`` replaced.

    A copy, not a mutation: ``cfg`` is shared with the caller and with anything
    else holding a reference to it, and a scoring run has no business changing
    the configuration a report is about to record.
    """
    from .config import Config

    data = cfg.to_dict()
    data.setdefault("explain", {})["cam_threshold"] = float(threshold)
    return Config(data)


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
