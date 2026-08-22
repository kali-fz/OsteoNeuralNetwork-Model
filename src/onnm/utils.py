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


def configure_backend(use_miopen: bool = True) -> dict[str, Any]:
    """Select the convolution/normalisation backend, working around a ROCm defect.

    MIOpen JIT-compiles some kernels at first use. On the ROCm 7.2.1 Windows
    wheels this fails for ``MIOpenBatchNormFwdTrainSpatialHIP`` -- the compiler
    cannot find the C++ standard header ``type_traits``, which the wheels simply
    do not ship (``rocm-sdk init`` expands the devel tree but supplies only
    thrust's ``type_traits``, not libc++'s). The result:

        MIOpen(HIP): Error [BuildHip] HIPRTC_ERROR_COMPILATION
        fatal error: 'type_traits' file not found
        RuntimeError: miopenStatusUnknownError

    Only the *training-mode* BatchNorm kernel is affected. Inference uses the
    eval-mode path and works, which is why a model can serve predictions
    perfectly on a machine that cannot train one -- a genuinely confusing
    failure to hit for the first time forty minutes into a run.

    Setting ``use_miopen=False`` routes convolution and normalisation through
    ATen's native implementations instead. Measured on a 7900 XT with
    DenseNet-121 at 256px, batch 32: 284 ms/step versus 205 ms/step for the
    MIOpen path with BatchNorm frozen. That is ~16 minutes rather than ~11 for a
    40-epoch run -- a cheap price for keeping standard training semantics, and
    the reason this is preferred over freezing BatchNorm to dodge the kernel.

    Returns a dict describing what was selected, for logging and the run record.
    """
    import torch

    info: dict[str, Any] = {
        "miopen_requested": bool(use_miopen),
        "backend": "MIOpen" if use_miopen else "ATen native",
    }
    is_rocm = getattr(torch.version, "hip", None) is not None
    info["is_rocm"] = is_rocm

    if not use_miopen and is_rocm:
        # On ROCm the cudnn flags are the MIOpen flags.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    elif not use_miopen:
        # The flag exists solely to dodge the ROCm-Windows defect described
        # above, which cannot occur on an NVIDIA build. Honouring it there would
        # disable cuDNN and cost several times the throughput -- silently, and
        # on precisely the runs most likely to be reproduced elsewhere, since
        # configs/full_run.yaml and configs/overnight.yaml both set miopen:
        # false. So it is ignored rather than obeyed, and said out loud.
        info["backend"] = "cuDNN"
        info["miopen_ignored"] = True
        get_logger(__name__).info(
            "train.miopen=false is a ROCm-only workaround and does not apply to this "
            "CUDA build; cuDNN stays enabled"
        )
    info["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
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
