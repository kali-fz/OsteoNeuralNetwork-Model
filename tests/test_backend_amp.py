"""Tests for backend selection and mixed-precision dtype resolution.

Both of these decide something at startup that is then invisible for the rest of
a run. A wrong choice does not raise -- it produces a slower run, or a run whose
gradients quietly underflow, and either way the loss curve still looks like a
loss curve. The failures are only catchable here, before the hours are spent.

The specific hazard both tests defend against is portability. This project is
developed on ROCm/Windows and trained on CUDA/Linux (Colab), and two settings
that are correct on the first are actively harmful on the second.
"""

from __future__ import annotations

import pytest
import torch

from onnm.train import resolve_amp_dtype
from onnm.utils import configure_backend


@pytest.fixture(autouse=True)
def _restore_cudnn():
    """configure_backend mutates global torch state; put it back."""
    enabled, benchmark = torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.benchmark = benchmark


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_miopen_false_disables_cudnn_on_rocm(monkeypatch) -> None:
    """The original purpose: dodge the ROCm-Windows BatchNorm kernel defect."""
    monkeypatch.setattr(torch.version, "hip", "7.2.1", raising=False)
    info = configure_backend(use_miopen=False)
    assert info["is_rocm"] is True
    assert info["cudnn_enabled"] is False
    assert torch.backends.cudnn.benchmark is False


def test_miopen_false_is_ignored_on_cuda(monkeypatch) -> None:
    """The regression this guards.

    configs/full_run.yaml and configs/overnight.yaml both carry
    `miopen: false`, and those are exactly the configs a Colab run reuses in
    order to be comparable. Obeying the flag there would disable cuDNN and cost
    several times the throughput, to work around a bug that cannot occur on an
    NVIDIA build -- and would do it silently.
    """
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    info = configure_backend(use_miopen=False)
    assert info["is_rocm"] is False
    assert info["miopen_ignored"] is True
    assert info["cudnn_enabled"] is True
    assert info["backend"] == "cuDNN"


def test_miopen_true_leaves_cudnn_alone(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.1", raising=False)
    info = configure_backend(use_miopen=True)
    assert info["cudnn_enabled"] is True
    assert "miopen_ignored" not in info


# ---------------------------------------------------------------------------
# AMP dtype resolution
# ---------------------------------------------------------------------------
def _cuda(cfg, monkeypatch, *, bf16: bool, dtype: str = "bfloat16", amp: bool = True):
    """Pretend to be a CUDA device with a given bf16 capability."""
    cfg._data["train"]["amp"] = amp
    cfg._data["train"]["amp_dtype"] = dtype
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
    return torch.device("cuda")


def test_bfloat16_survives_on_a_card_that_supports_it(cfg, monkeypatch) -> None:
    device = _cuda(cfg, monkeypatch, bf16=True)
    assert resolve_amp_dtype(cfg, device) is torch.bfloat16


def test_bfloat16_falls_back_to_fp16_on_turing(cfg, monkeypatch, caplog) -> None:
    """Colab's free tier is a T4 (sm_75), which has no bf16 at all.

    Silently proceeding would either raise deep in autocast or emulate at a
    large cost, an hour into a run. The fallback must also be audible -- fp16
    and bf16 are not interchangeable, and a later comparison of two runs needs
    to know which one happened.
    """
    device = _cuda(cfg, monkeypatch, bf16=False)
    with caplog.at_level("WARNING", logger="onnm.train"):
        assert resolve_amp_dtype(cfg, device) is torch.float16
    assert "bfloat16" in caplog.text


def test_explicit_fp16_is_not_second_guessed(cfg, monkeypatch) -> None:
    device = _cuda(cfg, monkeypatch, bf16=True, dtype="float16")
    assert resolve_amp_dtype(cfg, device) is torch.float16


def test_amp_disabled_returns_none(cfg, monkeypatch) -> None:
    device = _cuda(cfg, monkeypatch, bf16=True, amp=False)
    assert resolve_amp_dtype(cfg, device) is None


def test_cpu_never_autocasts(cfg) -> None:
    cfg._data["train"]["amp"] = True
    assert resolve_amp_dtype(cfg, torch.device("cpu")) is None


# ---------------------------------------------------------------------------
# GradScaler pairing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.float16, True), (torch.bfloat16, False), (None, False)],
)
def test_scaler_is_enabled_only_for_fp16(dtype, expected) -> None:
    """bf16 shares fp32's exponent range and must not be scaled.

    Enabling a scaler for it would add a failure mode -- inf checks, skipped
    steps -- to a path that cannot underflow in the first place. The local
    7900 XT trains in bf16, so this is what keeps that path untouched.
    """
    scaler = torch.amp.GradScaler("cpu", enabled=(dtype is torch.float16))
    assert scaler.is_enabled() is expected


def test_clipping_happens_after_unscaling() -> None:
    """Clip threshold must be applied to true gradients, not scaled ones.

    With a scale of ~65536, a real gradient norm of 0.5 reads as 32768. Clipping
    to 1.0 before unscaling would therefore shrink every gradient by ~30000x and
    training would flatline while still logging a falling loss.
    """
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=65536.0)

    scaler.scale(param * 0.5).backward()
    # Before unscaling, the gradient is off by the scale factor. Clipping here
    # is the bug this test pins.
    assert param.grad.item() == pytest.approx(0.5 * 65536.0, rel=1e-3)

    scaler.unscale_(optimizer)
    assert param.grad.item() == pytest.approx(0.5, rel=1e-3)
    total = torch.nn.utils.clip_grad_norm_([param], 1.0)
    assert float(total) == pytest.approx(0.5, rel=1e-3)   # under the clip, untouched
