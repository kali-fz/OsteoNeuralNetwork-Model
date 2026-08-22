"""Tests for hard-negative mining and the thermal governor.

Both of these run unattended overnight, which changes what is worth testing.
A governor that fails to throttle wastes a card; one that throttles when it
should not wastes a night. A miner that amplifies the wrong samples trades the
error being complained about for the error that actually matters clinically.
None of those announce themselves in a training curve.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from onnm.losses import FocalLoss, HardNegativeMiningLoss, build_loss
from onnm.thermal import GpuTelemetry, ThermalGovernor


def _loss(**kwargs) -> HardNegativeMiningLoss:
    base = FocalLoss(gamma=2.0, reduction="none")
    return HardNegativeMiningLoss(base, normal_index=0, **kwargs)


# ---------------------------------------------------------------------------
# What gets mined
# ---------------------------------------------------------------------------
@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """6 samples: 2 normals called normal, 2 normals called lesion, 2 malignant.

    Indices 2 and 3 are the hard negatives -- truly normal, confidently wrong.
    """
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],   # normal, correct
            [4.0, 0.0, 0.0],   # normal, correct
            [0.0, 0.0, 4.0],   # normal -> called malignant   HARD NEGATIVE
            [0.0, 4.0, 0.0],   # normal -> called benign      HARD NEGATIVE
            [0.0, 0.0, 4.0],   # malignant, correct
            [4.0, 0.0, 0.0],   # malignant -> called normal (a missed cancer)
        ]
    )
    return logits, torch.tensor([0, 0, 0, 0, 2, 2])


def test_mines_only_confidently_wrong_normals(batch) -> None:
    logits, targets = batch
    loss = _loss(warmup_epochs=0, penalty=4.0)
    loss.set_epoch(0)
    loss(logits, targets)
    assert loss.n_mined == 2
    assert loss.n_normal == 4


def test_missed_cancer_is_not_mined(batch) -> None:
    """The last sample is a malignant film called normal -- high loss, not a negative.

    Classic OHEM ranks by loss and would select it. This must not: amplifying
    it here would be indistinguishable from raising the malignant class weight,
    which alpha already does, and the whole point of this term is to be
    class-asymmetric.
    """
    logits, targets = batch
    loss = _loss(warmup_epochs=0)
    loss.set_epoch(0)
    loss(logits, targets)
    # Only the two normals qualify; if the missed cancer were being mined the
    # count would be 3.
    assert loss.n_mined == 2


def test_penalty_actually_increases_the_loss(batch) -> None:
    logits, targets = batch
    baseline = _loss(warmup_epochs=0, penalty=1.0)
    magnified = _loss(warmup_epochs=0, penalty=8.0)
    baseline.set_epoch(0)
    magnified.set_epoch(0)
    assert float(magnified(logits, targets)) > float(baseline(logits, targets))


def test_gradient_concentrates_on_hard_negatives(batch) -> None:
    """The mechanism, checked directly rather than through the scalar loss.

    A larger penalty must move gradient mass toward the mined samples. If it
    only raised the total loss without changing where the gradient points, the
    term would be an elaborate learning-rate multiplier.
    """
    logits, targets = batch

    def hard_share(penalty: float) -> float:
        x = logits.clone().requires_grad_(True)
        loss = _loss(warmup_epochs=0, penalty=penalty)
        loss.set_epoch(0)
        loss(x, targets).backward()
        magnitude = x.grad.abs().sum(dim=1)
        return float(magnitude[2:4].sum() / magnitude.sum())

    assert hard_share(8.0) > hard_share(1.0)


def test_warmup_suppresses_mining(batch) -> None:
    """Before warmup the term must be a no-op, not merely small."""
    logits, targets = batch
    loss = _loss(warmup_epochs=5, penalty=8.0)

    loss.set_epoch(0)
    early = float(loss(logits, targets))
    assert loss.n_mined == 0
    assert not loss.active

    loss.set_epoch(5)
    late = float(loss(logits, targets))
    assert loss.n_mined == 2
    assert late != pytest.approx(early)


def test_max_fraction_caps_the_amplified_share() -> None:
    """A bad epoch must not amplify the whole batch and spike the effective LR."""
    logits = torch.zeros(20, 3)
    logits[:, 2] = 5.0                      # every sample called malignant
    targets = torch.zeros(20, dtype=torch.long)   # every sample truly normal

    loss = _loss(warmup_epochs=0, max_fraction=0.25)
    loss.set_epoch(0)
    loss(logits, targets)
    assert loss.n_mined == 5                # 25% of 20, not all 20


def test_normalize_keeps_the_loss_scale_stable() -> None:
    """Mean weight stays 1, so the LR schedule keeps meaning the same thing."""
    logits = torch.randn(32, 3)
    targets = torch.zeros(32, dtype=torch.long)

    normalized = _loss(warmup_epochs=0, penalty=6.0, normalize=True)
    raw = _loss(warmup_epochs=0, penalty=6.0, normalize=False)
    normalized.set_epoch(0)
    raw.set_epoch(0)
    assert float(normalized(logits, targets)) < float(raw(logits, targets))


def test_counters_reset_each_epoch(batch) -> None:
    logits, targets = batch
    loss = _loss(warmup_epochs=0)
    loss.set_epoch(0)
    loss(logits, targets)
    loss(logits, targets)
    assert loss.n_mined == 4          # accumulated across two batches
    loss.set_epoch(1)
    assert loss.n_mined == 0


def test_rejects_a_reduced_base_loss() -> None:
    """A mean-reduced base loss silently disables per-sample weighting."""
    with pytest.raises(ValueError, match="reduction='none'"):
        HardNegativeMiningLoss(nn.CrossEntropyLoss())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"penalty": 0.5}, "penalty must be"),
        ({"confidence": 1.5}, "confidence must be"),
        ({"max_fraction": 0.0}, "max_fraction must be"),
    ],
)
def test_invalid_settings_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _loss(**kwargs)


def test_build_loss_returns_plain_loss_when_mining_is_off(cfg) -> None:
    assert not isinstance(build_loss(cfg, alpha=None), HardNegativeMiningLoss)


def test_build_loss_switches_base_to_unreduced(cfg) -> None:
    """The wrapper needs per-sample values; build_loss must arrange that."""
    cfg._data["loss"]["ohem"] = {"enabled": True}
    loss = build_loss(cfg, alpha=None)
    assert isinstance(loss, HardNegativeMiningLoss)
    assert loss.base_loss.reduction == "none"


# ---------------------------------------------------------------------------
# Thermal governor
# ---------------------------------------------------------------------------
class _FakeMonitor:
    """A monitor with a scripted temperature sequence."""

    def __init__(self, temperatures: list[float], memory: float = 70.0) -> None:
        self.temperatures = list(temperatures)
        self.memory = memory
        self.available = True
        self.reads = 0

    def read(self) -> GpuTelemetry | None:
        self.reads += 1
        value = self.temperatures[min(self.reads - 1, len(self.temperatures) - 1)]
        return GpuTelemetry(hotspot_c=value, memory_c=self.memory)

    def close(self) -> None:
        self.available = False


def test_control_temperature_ignores_memory() -> None:
    """VRAM idles near 68 C; folding it into the die limit would throttle forever."""
    reading = GpuTelemetry(hotspot_c=52.0, memory_c=88.0)
    assert reading.control_c == 52.0
    assert reading.hottest == 88.0


def test_cool_gpu_is_never_paused() -> None:
    monitor = _FakeMonitor([50.0] * 10)
    governor = ThermalGovernor(high_c=75, resume_c=70, check_every=1, monitor=monitor)
    for _ in range(5):
        governor.step()
    assert governor.stats.throttle_events == 0
    assert governor.stats.paused_seconds == 0.0
    assert governor.stats.peak_c == 50.0


def test_hot_gpu_pauses_until_it_cools() -> None:
    """Crosses the limit, then cools below resume_c on the third poll."""
    monitor = _FakeMonitor([80.0, 78.0, 74.0, 69.0, 60.0])
    governor = ThermalGovernor(
        high_c=75, resume_c=70, check_every=1, poll_seconds=0.01, monitor=monitor
    )
    governor.step()
    assert governor.stats.throttle_events == 1
    assert governor.stats.paused_seconds > 0
    assert governor.stats.peak_c == 80.0


def test_pause_is_bounded_so_a_run_cannot_stall_forever() -> None:
    """A stuck-hot card must resume and log, not silently consume the night."""
    monitor = _FakeMonitor([90.0] * 200)
    governor = ThermalGovernor(
        high_c=75, resume_c=70, check_every=1, poll_seconds=0.01,
        max_pause_seconds=0.05, monitor=monitor,
    )
    governor.step()
    assert governor.stats.paused_seconds < 1.0


def test_hysteresis_is_required() -> None:
    """Equal thresholds make the governor chatter instead of cooling."""
    with pytest.raises(ValueError, match="must be below"):
        ThermalGovernor(high_c=75, resume_c=75)
    with pytest.raises(ValueError, match="must be below"):
        ThermalGovernor(high_c=75, resume_c=80)


def test_sampling_is_throttled_to_check_every() -> None:
    """An ADL call per step would be wasted work; die temp moves slowly."""
    monitor = _FakeMonitor([50.0] * 100)
    governor = ThermalGovernor(high_c=75, resume_c=70, check_every=10, monitor=monitor)
    for _ in range(100):
        governor.step()
    assert monitor.reads == 10


def test_unavailable_monitor_degrades_to_a_no_op() -> None:
    """No sensors must mean 'run without throttling', never 'crash at 3am'."""
    monitor = _FakeMonitor([50.0])
    monitor.available = False
    governor = ThermalGovernor(high_c=75, resume_c=70, check_every=1, monitor=monitor)
    for _ in range(5):
        governor.step()
    assert governor.stats.throttle_events == 0
    assert monitor.reads == 0


def test_memory_over_its_own_ceiling_also_pauses() -> None:
    monitor = _FakeMonitor([50.0] * 10, memory=99.0)
    governor = ThermalGovernor(
        high_c=75, resume_c=70, check_every=1, poll_seconds=0.01,
        max_pause_seconds=0.05, memory_limit_c=95.0, monitor=monitor,
    )
    governor.step()
    assert governor.stats.throttle_events == 1
