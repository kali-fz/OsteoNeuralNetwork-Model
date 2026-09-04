"""The lesion head: a second output that says *where*, not just *what*.

WHY THIS EXISTS
---------------
Until now the "Where the model looked" panel was Grad-CAM, which is not an
output of the model at all -- it is a question asked of the model afterwards.
It reads ``features.denseblock4``, an 8x8 grid at a 256px input, so the whole
heatmap holds 64 values each covering a 32x32 block. On an ankle film the entire
joint is about two of those blocks, which is why
``reports/full-20260822-041653/gradcam_test/gradcam_report.json`` records a
pointing-game accuracy of 0.0936 and ``cam_degenerate: true``.

Worse, nothing had ever *taught* the model where lesions are. BTXRD ships a
polygon for every one of its 1867 tumour images and they were used only to score
the heatmap, never to train. A classifier with no localisation signal is free to
key on whatever correlates with the label, and on a bone film that is joint
texture.

This module adds a head trained on those polygons. It is not a prettier picture:
because both heads share one backbone, the features have to satisfy "name the
class" AND "outline the lesion" at once, and they cannot do the second by
staring at the ankle.

WHY IT SUBCLASSES DENSENET RATHER THAN WRAPPING IT
---------------------------------------------------
This is the load-bearing decision in the module, and getting it wrong takes the
live site down rather than merely scoring badly.

A wrapper -- ``MultiTaskNet(backbone=densenet)`` -- renames every module:
``features.denseblock4`` becomes ``backbone.features.denseblock4``. That exact
name is ``explain.target_layer`` in configs/base.yaml, resolved by
``onnm.model.get_cam_layer``, which RAISES when it is missing.
``RadiographClassifier.__init__`` calls ``build_cam`` unconditionally, and
``inference/main.py`` constructs the service AT IMPORT. So the exception arrives
before uvicorn binds a port, and with ``max_instances: 1`` in wrangler.jsonc the
site's only inference container crash-loops with /api/scan dead for everyone.

Subclassing keeps ``.features``, ``.classifier`` and every ``state_dict`` key
byte-identical to a plain DenseNet, so Grad-CAM, ``get_cam_layer``,
``head_parameters`` and ``stage_inference_model.py``'s sha256 check all keep
working untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch import nn
from torchvision.models.densenet import DenseNet

from .utils import get_logger

logger = get_logger(__name__)

#: torchvision's DenseNet hyperparameters, restated here because they live
#: inside the factory functions (``tvm.densenet121`` and friends) and there is no
#: public way to read them back off a constructed model. Only backbones whose
#: Grad-CAM layer is ``features.denseblock4`` are listed; anything else needs its
#: own skip-connection map before it could carry this head.
DENSENET_ARCH: dict[str, tuple[int, tuple[int, int, int, int], int]] = {
    "densenet121": (32, (6, 12, 24, 16), 64),
    "densenet169": (32, (6, 12, 32, 32), 64),
}


def stage_channels(model: DenseNet) -> tuple[int, int, int, int]:
    """Channel width of each of the four denseblock outputs.

    Read off the normalisation layer that consumes each block rather than
    recomputed from growth rate and block config. Both give the same answer
    today; only this one keeps giving the right answer if torchvision ever
    changes how a block is assembled.
    """
    features = model.features
    return (
        features.transition1.norm.num_features,
        features.transition2.norm.num_features,
        features.transition3.norm.num_features,
        features.norm5.num_features,
    )


class LesionHead(nn.Module):
    """An FPN-lite decoder predicting one lesion map from four DenseNet stages.

    Deliberately tiny. This runs in a Cloudflare ``standard-1`` container: half a
    vCPU, 4 GiB, ``OMP_NUM_THREADS=1`` (see inference/Dockerfile). A U-Net
    decoder at full resolution is the textbook choice and is also the thing that
    would make every scan slow enough for a user to notice.

    Four 1x1 convolutions bring each stage to a common width; each is added to a
    bilinear upsample of the stage below it; one 3x3 convolution reads out a
    single channel. Roughly 180K parameters against the backbone's 7M, and at a
    256px input it produces a 64x64 map -- 64 times the spatial detail of the 8x8
    grid Grad-CAM reads, which is the entire reason the old heatmap could not
    localise anything smaller than a joint.

    Returns LOGITS. The loss needs them for ``binary_cross_entropy_with_logits``,
    and keeping the sigmoid at the point of use is what stops it being applied
    twice by accident.
    """

    def __init__(self, channels: tuple[int, int, int, int], width: int = 64) -> None:
        super().__init__()
        # Shallowest first, matching the order `forward` receives them in.
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, kernel_size=1) for c in channels)
        self.predict = nn.Conv2d(width, 1, kernel_size=3, padding=1)

    def forward(self, stages: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Start at the deepest stage and walk back out, adding each lateral
        # projection to an upsample of the level below it. Interpolating to the
        # lateral's own shape rather than by a factor of 2 keeps this correct for
        # input sizes that do not halve cleanly at every stage.
        merged = self.lateral[-1](stages[-1])
        for index in range(len(stages) - 2, -1, -1):
            lateral = self.lateral[index](stages[index])
            merged = lateral + F.interpolate(
                merged, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.predict(merged)


class DenseNetWithLesionHead(DenseNet):
    """DenseNet plus a lesion head, with a tensor-returning forward by default.

    ``forward`` returns a plain Tensor unless ``return_mask`` is set. That
    default is not timidity -- ten call sites assume ``model(x) -> Tensor``: the
    training loop (train.py:162, :200), ``overfit_batch`` (train.py:477),
    ``model_summary`` (model.py:212), ``collect_logits`` and its hflip-TTA branch
    (calibrate.py:405, :407), ``_warmup`` and ``predict`` (inference.py:481,
    :626), ``gradcam_report.py:123``, and **MONAI's own GradCAM**, which indexes
    ``logits[:, class_idx]`` and would fail on a tuple.

    With the default in place none of them change. The training loop opts in for
    the one place that actually wants both outputs.
    """

    def __init__(self, *args, decoder_width: int = 64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # A plain attribute, not a buffer: it is a call-site choice rather than
        # model state, and it must never travel inside a checkpoint.
        self.return_mask = False
        self.seg_head = LesionHead(stage_channels(self), width=decoder_width)

    def forward(self, x: torch.Tensor):  # noqa: D102 - see the class docstring
        features = self.features

        out = features.pool0(features.relu0(features.norm0(features.conv0(x))))

        # The stages are run one at a time rather than as `self.features(x)` so
        # the decoder can take skip connections. Each denseblock is still CALLED
        # as a module, which is what MONAI's Grad-CAM hooks fire on, so the
        # explainability path behaves exactly as on a plain DenseNet.
        stage1 = features.denseblock1(out)
        stage2 = features.denseblock2(features.transition1(stage1))
        stage3 = features.denseblock3(features.transition2(stage2))
        stage4 = features.denseblock4(features.transition3(stage3))

        # From here this is torchvision's own forward, unchanged. `norm5` returns
        # a fresh tensor, so the in-place ReLU cannot touch `stage4`, which the
        # decoder still needs.
        pooled = F.adaptive_avg_pool2d(F.relu(features.norm5(stage4), inplace=True), (1, 1))
        logits = self.classifier(torch.flatten(pooled, 1))

        if not self.return_mask:
            return logits
        return logits, self.seg_head((stage1, stage2, stage3, stage4))


def build_densenet_with_lesion_head(
    name: str, weights, num_classes: int, dropout: float, decoder_width: int = 64
) -> DenseNetWithLesionHead:
    """Construct the subclass and transplant ImageNet weights into it.

    The ImageNet checkpoint cannot supply the decoder, and that is the ONLY thing
    it is allowed not to supply. Any other missing or unexpected key means this
    subclass has drifted from torchvision's DenseNet -- which would train quite
    happily, converge to something mediocre, and read as a bad model rather than
    as a bug. So the key sets are checked rather than trusted.
    """
    if name not in DENSENET_ARCH:
        raise ValueError(
            f"model.lesion_head is enabled but {name!r} has no skip-connection map. "
            f"Supported: {sorted(DENSENET_ARCH)}. Add an entry to DENSENET_ARCH and "
            "confirm the Grad-CAM target layer before extending this."
        )

    growth_rate, block_config, num_init_features = DENSENET_ARCH[name]
    # Built at ImageNet's 1000 classes so the pretrained classifier loads
    # cleanly; the head is replaced immediately afterwards.
    model = DenseNetWithLesionHead(
        growth_rate, block_config, num_init_features, decoder_width=decoder_width
    )

    if weights is not None:
        reference = getattr(tvm, name)(weights=weights)
        missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
        stray = [key for key in missing if not key.startswith("seg_head.")]
        if stray or unexpected:
            raise RuntimeError(
                f"the lesion-head subclass does not match torchvision's {name}: "
                f"missing={stray[:5]} unexpected={list(unexpected)[:5]}. The backbone "
                "would be partly random and every metric would be wrong."
            )
        logger.info("%s: ImageNet weights loaded, decoder initialised fresh", name)

    from .model import _replace_head

    _replace_head(model, "classifier", num_classes, dropout)
    return model
