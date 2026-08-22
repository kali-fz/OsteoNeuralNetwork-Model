"""Shared helpers: seeding, logging, device selection, checkpoints."""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "onnm"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the project logger, configured once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_log(path: str | Path, name: str = LOGGER_NAME) -> None:
    """Tee the project logger to a file as well as the console."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    get_logger(name).addHandler(handler)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    ``deterministic=True`` also pins cuDNN/MIOpen algorithm selection. It is off
    by default because it costs real throughput and, on ROCm, some conv
    algorithms have no deterministic implementation at all -- enabling it
    globally would turn a training run into a crash rather than a slow run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # allows data-only tooling to run without torch
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(prefer_gpu: bool = True):
    """Return the best available torch device.

    ROCm reports itself through the CUDA API, so an AMD RX 7900 XT shows up as
    ``cuda:0`` with ``torch.version.hip`` set. That is expected, not a bug.
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def describe_device() -> dict[str, Any]:
    """Collect a human-readable summary of the compute environment."""
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["backend"] = "ROCm/HIP" if torch.version.hip else "CUDA"
    return info


def amp_dtype_from_str(name: str):
    """Map a config string to a torch dtype, defaulting to bf16.

    bf16 is the default on RDNA3: it has the same exponent range as fp32, so it
    needs no GradScaler and cannot silently underflow gradients the way fp16
    does.
    """
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


# ---------------------------------------------------------------------------
# JSON / filesystem
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, cls=_NumpyEncoder, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_dir(reports_root: str | Path, tag: str = "run") -> Path:
    """Create a fresh timestamped directory for one experiment's outputs."""
    return ensure_dir(Path(reports_root) / f"{tag}-{timestamp()}")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def format_counts(labels: Iterable[int], class_names: Sequence[str]) -> str:
    """Render a label distribution as an aligned, scannable table."""
    labels = list(labels)
    total = len(labels) or 1
    lines = [f"{'class':<12}{'n':>7}{'pct':>9}"]
    lines.append("-" * 28)
    for idx, name in enumerate(class_names):
        n = sum(1 for label in labels if label == idx)
        lines.append(f"{name:<12}{n:>7}{100 * n / total:>8.1f}%")
    lines.append("-" * 28)
    lines.append(f"{'total':<12}{len(labels):>7}")
    return "\n".join(lines)
