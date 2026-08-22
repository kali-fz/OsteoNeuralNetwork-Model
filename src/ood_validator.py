"""Compatibility shim: ``src/ood_validator.py`` -> :mod:`onnm.ood`.

The implementation lives inside the installed package (``src/onnm/ood.py``) so
it ships with ``pip install -e .`` and imports as ``onnm.ood`` like the rest of
the project. This file exists so that ``from ood_validator import ...`` also
works for anyone following the flat file layout. Prefer the real path.
"""

from __future__ import annotations

from onnm.ood import (  # noqa: F401
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_ENTROPY_GATE,
    INCONCLUSIVE_LABEL,
    REJECTION_MESSAGE,
    NonRadiographError,
    ValidationCheck,
    ValidationReport,
    ensure_radiograph,
    predictive_entropy,
    should_defer,
    validate_image,
    validate_payload,
)

__all__ = [
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_ENTROPY_GATE",
    "INCONCLUSIVE_LABEL",
    "REJECTION_MESSAGE",
    "NonRadiographError",
    "ValidationCheck",
    "ValidationReport",
    "ensure_radiograph",
    "predictive_entropy",
    "should_defer",
    "validate_image",
    "validate_payload",
]
