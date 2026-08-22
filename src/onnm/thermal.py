"""GPU temperature monitoring and duty-cycle throttling for unattended runs.

Why this is hand-rolled
-----------------------
There is no ``rocm-smi``, ``amd-smi`` or ``pynvml`` in the ROCm Windows wheel
set, and torch exposes no thermal API. What *is* present, installed by the
Adrenalin driver, is ``atiadlxx.dll`` -- AMD's Display Library. Its
``ADL2_New_QueryPMLogData_Get`` entry point returns a live sensor block, so the
whole monitor is a ctypes call against a DLL that is already on the machine. No
new dependency, no admin rights, no cost.

The older Overdrive5/6/N temperature entry points all return ``-8``
(ADL_ERR_NOT_SUPPORTED) on RDNA3, so PMLog is not merely preferred here, it is
the only one that works.

Which temperature this is
-------------------------
PMLog reports **hotspot** (junction) temperature, not edge. Hotspot is the
hottest measured point on the die and runs materially warmer than the "GPU
temperature" Adrenalin shows -- typically 10-20 C higher under load. That makes
a 75 C hotspot limit *more* conservative than a 75 C edge limit, which is the
right direction to be wrong for an unattended overnight run, but it does mean
the governor will engage earlier than the number suggests. Raise
``thermal.high_c`` if you want it to correspond to an edge reading instead.

How throttling works
--------------------
This cannot change clocks or power limits -- that needs driver-level control
this process does not have, and fighting the driver's own management would be a
bad idea anyway. What it does is reduce the *duty cycle*: when hotspot crosses
``high_c`` the training loop is paused in short sleeps until it falls back to
``resume_c``. The GPU idles, cools, and work resumes at full speed. Hysteresis
between the two thresholds stops it oscillating on and off at the boundary.
"""

from __future__ import annotations

import contextlib
import ctypes
import time
from ctypes import CFUNCTYPE, Structure, byref, c_char, c_int, c_void_p
from dataclasses import dataclass, field
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)

# PMLog sensor indices from ADL's SDK headers.
SENSOR_CLOCK_GFX = 1
SENSOR_TEMP_EDGE = 7
SENSOR_TEMP_HOTSPOT = 8
SENSOR_TEMP_MEM = 9
SENSOR_POWER = 11
SENSOR_FAN_RPM = 14

ADL_OK = 0
_MAX_SENSORS = 256


class _AdapterInfo(Structure):
    _fields_ = [
        ("iSize", c_int), ("iAdapterIndex", c_int), ("strUDID", c_char * 256),
        ("iBusNumber", c_int), ("iDeviceNumber", c_int), ("iFunctionNumber", c_int),
        ("iVendorID", c_int), ("strAdapterName", c_char * 256),
        ("strDisplayName", c_char * 256), ("iPresent", c_int), ("iExist", c_int),
        ("strDriverPath", c_char * 256), ("strDriverPathExt", c_char * 256),
        ("strPNPString", c_char * 256), ("iOSDisplayIndex", c_int),
    ]


class _SensorData(Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class _PMLogDataOutput(Structure):
    _fields_ = [("size", c_int), ("sensors", _SensorData * _MAX_SENSORS)]


_MALLOC = CFUNCTYPE(c_void_p, c_int)
_ALLOCATIONS: list[Any] = []


@_MALLOC
def _adl_malloc(size: int) -> int:
    # ADL calls this to allocate buffers it fills in. The references must
    # outlive the call, so they are parked in a module-level list rather than
    # being garbage collected the moment this returns.
    buffer = ctypes.create_string_buffer(size)
    _ALLOCATIONS.append(buffer)
    return ctypes.cast(buffer, c_void_p).value or 0


@dataclass
class GpuTelemetry:
    """One sensor reading. ``None`` for anything the adapter does not report."""

    hotspot_c: float | None = None
    memory_c: float | None = None
    edge_c: float | None = None
    fan_rpm: int | None = None
    clock_mhz: int | None = None
    power_w: float | None = None

    @property
    def control_c(self) -> float | None:
        """The die temperature the governor throttles on: hotspot, else edge.

        Deliberately **not** the maximum across all sensors. GDDR6 memory on a
        7900 XT idles around 68 C and is specified to roughly 95-105 C, so a
        75 C limit applied to the hottest sensor would trip before training even
        started and hold the duty cycle near zero all night. Memory has its own,
        much higher limit -- see :attr:`memory_c` and ``thermal.memory_limit_c``.
        """
        return self.hotspot_c if self.hotspot_c is not None else self.edge_c

    @property
    def hottest(self) -> float | None:
        """Highest of all temperature sensors. For reporting, not for control."""
        readings = [t for t in (self.hotspot_c, self.memory_c, self.edge_c) if t is not None]
        return max(readings) if readings else None

    def summary(self) -> str:
        parts = []
        if self.hotspot_c is not None:
            parts.append(f"hotspot {self.hotspot_c:.0f}C")
        if self.memory_c is not None:
            parts.append(f"mem {self.memory_c:.0f}C")
        if self.fan_rpm is not None:
            parts.append(f"fan {self.fan_rpm}rpm")
        if self.clock_mhz is not None:
            parts.append(f"{self.clock_mhz}MHz")
        return ", ".join(parts) if parts else "no sensors"


class AmdGpuMonitor:
    """Reads AMD GPU sensors through ADL. Never raises after construction.

    ``available`` is False when the DLL is missing, the adapter does not answer,
    or the driver is too old. Callers must treat that as "no thermal data" and
    decide for themselves whether to proceed -- this class does not make that
    policy decision.
    """

    def __init__(self) -> None:
        self.available = False
        self._adl: Any = None
        self._context = c_void_p()
        self._adapter: int | None = None
        self._failures = 0

        try:
            self._adl = ctypes.CDLL("atiadlxx.dll")
        except OSError as exc:
            logger.warning("atiadlxx.dll not loadable (%s); no GPU thermal data", exc)
            return

        try:
            if self._adl.ADL2_Main_Control_Create(_adl_malloc, 1, byref(self._context)) != ADL_OK:
                logger.warning("ADL2_Main_Control_Create failed; no GPU thermal data")
                return
            self._adapter = self._find_adapter()
            if self._adapter is None:
                logger.warning("no ADL adapter answered a PMLog query; no GPU thermal data")
                return
            self.available = True
        except (AttributeError, OSError) as exc:
            logger.warning("ADL unavailable (%s); no GPU thermal data", exc)

    def _find_adapter(self) -> int | None:
        """Pick the first adapter that actually returns a temperature.

        A single card enumerates once per display output, and not every entry
        answers PMLog, so the choice is made by trying rather than by index.
        """
        count = c_int(0)
        if self._adl.ADL2_Adapter_NumberOfAdapters_Get(self._context, byref(count)) != ADL_OK:
            return None
        if count.value <= 0:
            return None

        buffer = (_AdapterInfo * count.value)()
        if self._adl.ADL2_Adapter_AdapterInfo_Get(
            self._context, buffer, ctypes.sizeof(buffer)
        ) != ADL_OK:
            return None

        seen: set[int] = set()
        for info in buffer:
            index = int(info.iAdapterIndex)
            if not info.iExist or index in seen:
                continue
            seen.add(index)
            reading = self._read_adapter(index)
            if reading is not None and reading.control_c is not None:
                logger.info(
                    "GPU thermal monitor: %s (adapter %d), %s",
                    info.strAdapterName.decode(errors="ignore").strip(), index,
                    reading.summary(),
                )
                return index
        return None

    def _read_adapter(self, adapter: int) -> GpuTelemetry | None:
        output = _PMLogDataOutput()
        try:
            if self._adl.ADL2_New_QueryPMLogData_Get(
                self._context, adapter, byref(output)
            ) != ADL_OK:
                return None
        except (AttributeError, OSError):
            return None

        def sensor(index: int) -> int | None:
            entry = output.sensors[index]
            return int(entry.value) if entry.supported else None

        hotspot, memory, edge = sensor(SENSOR_TEMP_HOTSPOT), sensor(SENSOR_TEMP_MEM), sensor(
            SENSOR_TEMP_EDGE
        )
        power = sensor(SENSOR_POWER)
        return GpuTelemetry(
            hotspot_c=float(hotspot) if hotspot is not None else None,
            memory_c=float(memory) if memory is not None else None,
            edge_c=float(edge) if edge is not None else None,
            fan_rpm=sensor(SENSOR_FAN_RPM),
            clock_mhz=sensor(SENSOR_CLOCK_GFX),
            power_w=float(power) if power is not None else None,
        )

    def read(self) -> GpuTelemetry | None:
        """Current sensors, or None. Repeated failures disable the monitor."""
        if not self.available or self._adapter is None:
            return None
        reading = self._read_adapter(self._adapter)
        if reading is None:
            self._failures += 1
            if self._failures >= 5:
                # A driver reset or a sleeping adapter. Stop polling rather than
                # spending the rest of the night on failing DLL calls.
                logger.warning("ADL stopped responding after 5 attempts; thermal monitor off")
                self.available = False
            return None
        self._failures = 0
        return reading

    def close(self) -> None:
        if self._adl is not None and self._context:
            with contextlib.suppress(AttributeError, OSError):
                self._adl.ADL2_Main_Control_Destroy(self._context)
        self.available = False


@dataclass
class ThermalStats:
    """What the governor did, for the run record."""

    peak_c: float = 0.0
    peak_memory_c: float = 0.0
    throttle_events: int = 0
    paused_seconds: float = 0.0
    samples: int = 0
    monitored: bool = False
    last: GpuTelemetry | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "monitored": self.monitored,
            "peak_hotspot_c": round(self.peak_c, 1),
            "peak_memory_c": round(self.peak_memory_c, 1),
            "throttle_events": self.throttle_events,
            "paused_seconds": round(self.paused_seconds, 1),
            "samples": self.samples,
        }


class ThermalGovernor:
    """Pauses training while the GPU is above ``high_c``, resumes below ``resume_c``.

    Call :meth:`step` once per training step. Sampling is every ``check_every``
    steps because an ADL query costs about a millisecond and die temperature
    does not move meaningfully between two 200 ms steps.

    The two thresholds must differ. With a single threshold the governor
    switches on and off every sample once the GPU settles at the limit, which
    produces a stuttering duty cycle instead of a clean cool-down.
    """

    def __init__(
        self,
        high_c: float = 75.0,
        resume_c: float = 70.0,
        check_every: int = 20,
        poll_seconds: float = 2.0,
        max_pause_seconds: float = 300.0,
        memory_limit_c: float = 95.0,
        monitor: AmdGpuMonitor | None = None,
    ) -> None:
        if resume_c >= high_c:
            raise ValueError(
                f"resume_c ({resume_c}) must be below high_c ({high_c}); equal thresholds "
                "make the governor oscillate instead of cooling"
            )
        self.high_c = float(high_c)
        self.resume_c = float(resume_c)
        self.memory_limit_c = float(memory_limit_c)
        self.check_every = max(1, int(check_every))
        self.poll_seconds = float(poll_seconds)
        self.max_pause_seconds = float(max_pause_seconds)

        self.monitor = monitor if monitor is not None else AmdGpuMonitor()
        self.stats = ThermalStats(monitored=self.monitor.available)
        self._calls = 0

        if self.monitor.available:
            logger.info(
                "thermal governor armed: pause above %.0fC die hotspot (resume below "
                "%.0fC), memory ceiling %.0fC, sampled every %d steps",
                self.high_c, self.resume_c, self.memory_limit_c, self.check_every,
            )
        else:
            logger.warning(
                "thermal governor DISABLED: no GPU temperature available. Training will "
                "run at full duty cycle. The card's own driver-level protection still "
                "applies, but nothing in this process will slow it down."
            )

    def step(self) -> None:
        """Sample if due, and block while the GPU is too hot."""
        self._calls += 1
        if not self.monitor.available or self._calls % self.check_every:
            return

        reading = self.monitor.read()
        if reading is None:
            return
        self.stats.samples += 1
        self.stats.last = reading

        temperature = reading.control_c
        if temperature is None:
            return
        self.stats.peak_c = max(self.stats.peak_c, temperature)
        if reading.memory_c is not None:
            self.stats.peak_memory_c = max(self.stats.peak_memory_c, reading.memory_c)

        # Memory has its own ceiling, well above the die limit. Crossing it is
        # rare and worth a distinct message, because the remedy is different:
        # die heat is a cooler/airflow problem, memory heat is usually a
        # backplate or case-airflow one.
        memory_hot = (
            reading.memory_c is not None and reading.memory_c >= self.memory_limit_c
        )
        if temperature < self.high_c and not memory_hot:
            return
        if memory_hot:
            logger.warning(
                "memory at %.0fC, at or above the %.0fC limit", reading.memory_c,
                self.memory_limit_c,
            )

        self.stats.throttle_events += 1
        logger.warning(
            "thermal pause: %s exceeds %.0fC -- holding until it drops below %.0fC",
            reading.summary(), self.high_c, self.resume_c,
        )

        started = time.monotonic()
        while time.monotonic() - started < self.max_pause_seconds:
            time.sleep(self.poll_seconds)
            current = self.monitor.read()
            if current is None:
                break
            self.stats.samples += 1
            self.stats.last = current
            now = current.control_c
            if now is None:
                break
            self.stats.peak_c = max(self.stats.peak_c, now)
            if now <= self.resume_c:
                break
        else:
            # Cap reached. Continuing is the lesser evil: the driver enforces its
            # own hard limits, and blocking forever would silently kill the run.
            logger.error(
                "still above %.0fC after %.0fs of pausing. Resuming anyway -- check "
                "case airflow and fan curve. The driver's own thermal limits still apply.",
                self.resume_c, self.max_pause_seconds,
            )

        paused = time.monotonic() - started
        self.stats.paused_seconds += paused
        logger.info(
            "resuming after %.0fs (%s)",
            paused, self.stats.last.summary() if self.stats.last else "no reading",
        )

    def snapshot(self) -> str:
        reading = self.monitor.read() if self.monitor.available else None
        return reading.summary() if reading else "thermal monitoring unavailable"

    def close(self) -> None:
        self.monitor.close()


def build_governor(cfg) -> ThermalGovernor | None:
    """Construct the governor from config, or None when it is switched off."""
    thermal = cfg.get("thermal", None)
    if thermal is None or not bool(thermal.get("enabled", True)):
        logger.info("thermal governor disabled by config")
        return None
    return ThermalGovernor(
        high_c=float(thermal.get("high_c", 75.0)),
        resume_c=float(thermal.get("resume_c", 70.0)),
        check_every=int(thermal.get("check_every_steps", 20)),
        poll_seconds=float(thermal.get("poll_seconds", 2.0)),
        max_pause_seconds=float(thermal.get("max_pause_seconds", 300.0)),
        memory_limit_c=float(thermal.get("memory_limit_c", 95.0)),
    )


__all__ = [
    "AmdGpuMonitor",
    "GpuTelemetry",
    "ThermalGovernor",
    "ThermalStats",
    "build_governor",
]
