"""Compatibility shim: ``src/inference.py`` -> :mod:`onnm.inference`.

The implementation lives in ``src/onnm/inference.py`` so that it sits inside the
installed package alongside every other module it depends on, and imports as
``onnm.inference`` like the rest of the project. ``src/`` itself is the package
*root*, not a package, so a module placed here is not shipped by
``pip install -e .`` and would only be importable via a ``sys.path`` hack.

This file exists so that ``from inference import RadiographClassifier`` also
works for anyone following the original file layout. Prefer the real path.
"""

from __future__ import annotations

from onnm.inference import (  # noqa: F401
    INCONCLUSIVE_LABEL,
    LESION_LABEL,
    NORMAL_LABEL,
    SUPPORTED_SUFFIXES,
    UPLOAD_TYPES,
    InferenceResult,
    RadiographClassifier,
    default_checkpoint,
    find_checkpoints,
    predict_file,
    render_overlay,
    to_display_uint8,
)

__all__ = [
    "INCONCLUSIVE_LABEL",
    "LESION_LABEL",
    "NORMAL_LABEL",
    "SUPPORTED_SUFFIXES",
    "UPLOAD_TYPES",
    "InferenceResult",
    "RadiographClassifier",
    "default_checkpoint",
    "find_checkpoints",
    "predict_file",
    "render_overlay",
    "to_display_uint8",
]
