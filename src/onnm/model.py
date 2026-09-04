"""Model construction: ImageNet-pretrained backbones with a 3-class head.

Pretraining is not a nice-to-have here. The malignant class has 342 examples,
roughly 240 of which land in the training split. A DenseNet-121 trained from
random initialisation on that will memorise rather than generalise, and the gap
against a pretrained start is far larger than any architecture choice on offer.
So ``pretrained: true`` is the default and switching it off is an ablation, not
a convenience.

torchvision backbones are preferred over ``timm`` because torchvision ships in
AMD's ROCm wheel set -- one fewer dependency that has to resolve correctly on a
non-CUDA Windows box.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .utils import get_logger

logger = get_logger(__name__)

# Backbone name -> (torchvision factory attribute, default weights enum, head attribute)
TORCHVISION_BACKBONES: dict[str, tuple[str, str, str]] = {
    "densenet121": ("densenet121", "DenseNet121_Weights", "classifier"),
    "densenet169": ("densenet169", "DenseNet169_Weights", "classifier"),
    "resnet50": ("resnet50", "ResNet50_Weights", "fc"),
    "resnet34": ("resnet34", "ResNet34_Weights", "fc"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "classifier"),
}

# Where Grad-CAM should hook for each backbone: the last convolutional block
# before global pooling, which is where spatial evidence is still intact.
DEFAULT_CAM_LAYERS: dict[str, str] = {
    "densenet121": "features.denseblock4",
    "densenet169": "features.denseblock4",
    "resnet50": "layer4",
    "resnet34": "layer4",
    "efficientnet_b0": "features.8",
}


def _replace_head(model: nn.Module, attr: str, num_classes: int, dropout: float) -> nn.Module:
    """Swap the ImageNet 1000-way head for a ``num_classes`` head."""
    head = getattr(model, attr)

    if isinstance(head, nn.Linear):
        in_features = head.in_features
    elif isinstance(head, nn.Sequential):
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError(f"no Linear layer found in head {attr!r}")
        in_features = linears[-1].in_features
    else:
        raise TypeError(f"unsupported head type {type(head).__name__} for attribute {attr!r}")

    new_head: nn.Module = (
        nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        if dropout > 0
        else nn.Linear(in_features, num_classes)
    )
    setattr(model, attr, new_head)
    return model


def build_model(cfg, num_classes: int | None = None) -> nn.Module:
    """Build the configured classifier.

    Args:
        cfg: Project config; reads ``model.name``, ``model.pretrained``,
            ``model.num_classes`` and ``model.dropout``.
        num_classes: Overrides ``model.num_classes`` when given.
    """
    name = str(cfg.model.name).lower()
    num_classes = int(num_classes if num_classes is not None else cfg.model.num_classes)
    pretrained = bool(cfg.model.pretrained)
    dropout = float(cfg.model.get("dropout", 0.0))

    if name in TORCHVISION_BACKBONES:
        import torchvision.models as tvm

        factory_name, weights_name, head_attr = TORCHVISION_BACKBONES[name]
        weights = None
        if pretrained:
            try:
                weights = getattr(tvm, weights_name).DEFAULT
            except AttributeError:
                logger.warning("no pretrained weights available for %s", name)

        # The lesion head is opt-in, and the default False matters: with it off
        # this function is bit-identical to what it was, so every existing
        # checkpoint keeps loading and stage_inference_model.py's digest check
        # keeps passing on the model currently serving.
        if bool(cfg.model.get("lesion_head", False)):
            from .lesion_head import build_densenet_with_lesion_head

            model = build_densenet_with_lesion_head(
                name,
                weights,
                num_classes,
                dropout,
                decoder_width=int(cfg.model.get("decoder_channels", 64)),
            )
            logger.info(
                "built %s + lesion head (pretrained=%s, num_classes=%d, dropout=%.2f)",
                name, weights is not None, num_classes, dropout,
            )
            return model

        model = getattr(tvm, factory_name)(weights=weights)
        model = _replace_head(model, head_attr, num_classes, dropout)
        logger.info(
            "built %s (pretrained=%s, num_classes=%d, dropout=%.2f)",
            name, weights is not None, num_classes, dropout,
        )
        return model

    if name.startswith("monai_"):
        # MONAI's own nets carry no ImageNet weights. Kept as an ablation
        # baseline, not a recommended default at this dataset size.
        from monai.networks.nets import DenseNet121

        if pretrained:
            logger.warning(
                "MONAI nets have no ImageNet weights; %s will train from scratch, "
                "which 342 malignant cases cannot support. Prefer densenet121.", name,
            )
        return DenseNet121(
            spatial_dims=2,
            in_channels=int(cfg.data.in_channels),
            out_channels=num_classes,
            dropout_prob=dropout,
        )

    if name.startswith("vit_") or name.startswith("timm_"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"{name} needs timm: pip install 'onnm[vit]'"
            ) from exc

        return timm.create_model(
            name.removeprefix("timm_"),
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=int(cfg.data.in_channels),
            drop_rate=dropout,
        )

    raise ValueError(
        f"unknown model {name!r}; expected one of {sorted(TORCHVISION_BACKBONES)}, "
        "a monai_* or a timm_*/vit_* name"
    )


def build_model_for_checkpoint(state: dict[str, Any], cfg, num_classes: int | None = None):
    """Build the architecture a checkpoint actually contains, not the one on disk.

    WHY THIS EXISTS
    ---------------
    ``RadiographClassifier`` reads the config embedded in the checkpoint, but it
    is the only thing that does. Six other entry points build from whatever YAML
    they were passed and then load a state dict into it:

        evaluate.py, calibrate.py, gradcam_report.py, stratified_report.py,
        ablate_tta.py, and onnm.train.evaluate

    That worked only because every checkpoint so far happened to be a 3-class
    densenet121, which is what ``configs/base.yaml`` says anyway. The moment a
    checkpoint carries a lesion head, those six build a bare DenseNet and die on
    a key mismatch -- and ``daily_cycle.py`` is worse than that, because it
    passes ``--override`` to train.py (:179) but calls calibrate.py (:196) and
    evaluate.py (:204) with none at all, so the community loop would stop.

    WHAT IT TAKES FROM WHERE
    ------------------------
    Only the checkpoint's ``model`` block is honoured. Everything else -- data
    root, splits file, reports directory -- stays as the caller loaded it,
    because a checkpoint records where *its* data lived when it was trained, and
    that is not necessarily where yours is now.
    """
    from .config import Config

    data = cfg.to_dict()
    embedded = state.get("config")
    if isinstance(embedded, dict) and isinstance(embedded.get("model"), dict):
        data["model"] = copy.deepcopy(embedded["model"])
    else:
        logger.warning(
            "checkpoint has no embedded model config; building from the supplied "
            "config instead. If the architectures disagree, load_state_dict will "
            "say so rather than failing quietly."
        )

    # The state dict about to be loaded replaces every parameter, so fetching
    # ImageNet weights would buy nothing and would make this need the internet.
    data.setdefault("model", {})["pretrained"] = False
    return build_model(Config(data), num_classes=num_classes)


def get_cam_layer(model: nn.Module, cfg) -> str:
    """Resolve the Grad-CAM target layer, validating it exists on the model."""
    configured = str(cfg.explain.get("target_layer", "") or "")
    name = configured or DEFAULT_CAM_LAYERS.get(str(cfg.model.name).lower(), "")
    if not name:
        raise ValueError(
            f"no Grad-CAM layer known for model {cfg.model.name!r}; set explain.target_layer"
        )

    available = dict(model.named_modules())
    if name not in available:
        candidates = [n for n in available if n.count(".") <= 2 and n]
        raise ValueError(
            f"layer {name!r} not found on {type(model).__name__}. "
            f"Candidates include: {candidates[:15]}"
        )
    return name


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def head_parameters(model: nn.Module, cfg) -> list[nn.Parameter]:
    """The head parameters, for freeze/unfreeze bookkeeping.

    Plural "heads" when a lesion decoder is attached. The decoder is randomly
    initialised exactly like the classifier, so it belongs on the head side of
    the freeze boundary; grouping it with the pretrained backbone would freeze a
    head that has learnt nothing yet. That failure would also be quiet --
    ``set_backbone_trainable`` warns and continues rather than raising -- so the
    run record would describe a freeze that never happened.
    """
    name = str(cfg.model.name).lower()
    entry = TORCHVISION_BACKBONES.get(name)
    if entry is not None:
        parameters = list(getattr(model, entry[2]).parameters())
        seg_head = getattr(model, "seg_head", None)
        if seg_head is not None:
            parameters += list(seg_head.parameters())
        return parameters
    if hasattr(model, "get_classifier"):  # timm convention
        return list(model.get_classifier().parameters())
    if hasattr(model, "class_layers"):  # MONAI DenseNet convention
        return list(model.class_layers.parameters())
    return []


def set_backbone_trainable(model: nn.Module, cfg, trainable: bool) -> dict[str, int]:
    """Toggle ``requires_grad`` on every parameter *outside* the classification head.

    Freezing the pretrained features for the first few epochs protects them
    while the randomly initialised head finds its footing — with ~244 malignant
    training images there is very little signal to re-earn a wrecked feature,
    so not wrecking it in the first place is the cheaper strategy.

    Deliberately leaves BatchNorm layers in train mode while frozen: their
    running statistics still adapt to radiographs, which is the part of
    domain shift a frozen conv cannot absorb. Returns the parameter counts so
    the caller can log what actually happened.
    """
    head = {id(p) for p in head_parameters(model, cfg)}
    if not head:
        logger.warning(
            "cannot identify the classification head for %s; freeze request ignored",
            cfg.model.name,
        )
        return count_parameters(model)
    for parameter in model.parameters():
        if id(parameter) not in head:
            parameter.requires_grad = trainable
    return count_parameters(model)


def model_summary(model: nn.Module, cfg) -> str:
    counts = count_parameters(model)
    size = int(cfg.data.image_size)
    channels = int(cfg.data.in_channels)

    with torch.no_grad():
        dummy = torch.zeros(2, channels, size, size)
        try:
            out_shape: Any = tuple(model(dummy).shape)
        except Exception as exc:  # noqa: BLE001
            out_shape = f"<forward failed: {exc}>"

    return "\n".join(
        [
            f"model        : {cfg.model.name}",
            f"pretrained   : {cfg.model.pretrained}",
            f"input        : (B, {channels}, {size}, {size})",
            f"output       : {out_shape}",
            f"parameters   : {counts['total']:,} total / {counts['trainable']:,} trainable",
        ]
    )
