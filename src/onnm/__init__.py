"""OsteoNeuralNetwork-Model: explainable bone-tumor detection on plain radiographs."""

__version__ = "0.1.0"

CLASS_NAMES: tuple[str, ...] = ("normal", "benign", "malignant")
MALIGNANT_INDEX: int = 2

__all__ = ["CLASS_NAMES", "MALIGNANT_INDEX", "__version__"]
