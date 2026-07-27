"""Unit-explicit ORCHES paper equations and conventional GPU roofline timing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..errors import ConfigurationError
from .operators import OperatorCost


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be finite and positive")


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class PaperGpuRates:
    """Eq. (1)/(4) GPU rates in MAC/s and tensor elements/s."""

    compute_mac_per_s: float
    memory_elements_per_s: float

    def __post_init__(self) -> None:
        _positive("gpu.compute_mac_per_s", self.compute_mac_per_s)
        _positive("gpu.memory_elements_per_s", self.memory_elements_per_s)

    @classmethod
    def from_byte_bandwidth(
        cls,
        *,
        compute_mac_per_s: float,
        memory_bytes_per_s: float,
        bytes_per_element: int,
    ) -> PaperGpuRates:
        _positive("gpu.memory_bytes_per_s", memory_bytes_per_s)
        _positive("bytes_per_element", float(bytes_per_element))
        return cls(
            compute_mac_per_s=compute_mac_per_s,
            memory_elements_per_s=memory_bytes_per_s / bytes_per_element,
        )


@dataclass(frozen=True)
class PaperPimRates:
    """Eq. (2)/(3) PIM rates with separate internal and host-I/O bandwidth."""

    compute_mac_per_s: float
    internal_memory_elements_per_s: float
    host_io_elements_per_s: float

    def __post_init__(self) -> None:
        _positive("pim.compute_mac_per_s", self.compute_mac_per_s)
        _positive(
            "pim.internal_memory_elements_per_s",
            self.internal_memory_elements_per_s,
        )
        _positive("pim.host_io_elements_per_s", self.host_io_elements_per_s)

    @classmethod
    def from_byte_bandwidths(
        cls,
        *,
        compute_mac_per_s: float,
        internal_memory_bytes_per_s: float,
        host_io_bytes_per_s: float,
        bytes_per_element: int,
    ) -> PaperPimRates:
        _positive("pim.internal_memory_bytes_per_s", internal_memory_bytes_per_s)
        _positive("pim.host_io_bytes_per_s", host_io_bytes_per_s)
        _positive("bytes_per_element", float(bytes_per_element))
        return cls(
            compute_mac_per_s=compute_mac_per_s,
            internal_memory_elements_per_s=(
                internal_memory_bytes_per_s / bytes_per_element
            ),
            host_io_elements_per_s=host_io_bytes_per_s / bytes_per_element,
        )


@dataclass(frozen=True)
class PaperTimingBreakdown:
    """Additive timing components used by the paper's assignment equations."""

    compute_s: float = 0.0
    memory_s: float = 0.0
    internal_memory_s: float = 0.0
    host_io_s: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("compute_s", self.compute_s),
            ("memory_s", self.memory_s),
            ("internal_memory_s", self.internal_memory_s),
            ("host_io_s", self.host_io_s),
        ):
            _nonnegative(name, value)

    @property
    def total_s(self) -> float:
        return self.compute_s + self.memory_s + self.internal_memory_s + self.host_io_s


@dataclass(frozen=True)
class CoprocessedLinearTiming:
    """Eq. (3)-(4) simultaneous GPU/PIM completion times for one alpha."""

    alpha: float
    gpu: PaperTimingBreakdown
    pim: PaperTimingBreakdown

    def __post_init__(self) -> None:
        if not isfinite(self.alpha) or self.alpha < 0 or self.alpha > 1:
            raise ConfigurationError("alpha must be finite and in [0, 1]")

    @property
    def critical_path_s(self) -> float:
        return max(self.gpu.total_s, self.pim.total_s)


def _linear_shape(branch_width: int, hidden_size: int) -> tuple[float, float]:
    if isinstance(branch_width, bool) or not isinstance(branch_width, int):
        raise ConfigurationError("branch_width must be a positive integer")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise ConfigurationError("hidden_size must be a positive integer")
    if branch_width <= 0 or hidden_size <= 0:
        raise ConfigurationError("branch_width and hidden_size must be positive")
    return float(branch_width), float(hidden_size)


def paper_gpu_linear_time(
    branch_width: int,
    hidden_size: int,
    rates: PaperGpuRates,
) -> PaperTimingBreakdown:
    """Implement the paper's GPU linear model exactly, without a factor of two."""

    width, hidden = _linear_shape(branch_width, hidden_size)
    return PaperTimingBreakdown(
        compute_s=width * hidden * hidden / rates.compute_mac_per_s,
        memory_s=(2 * width * hidden + hidden * hidden)
        / rates.memory_elements_per_s,
    )


def paper_pim_linear_time(
    branch_width: int,
    hidden_size: int,
    rates: PaperPimRates,
) -> PaperTimingBreakdown:
    """Implement the paper's PIM linear model with separate bandwidth domains."""

    width, hidden = _linear_shape(branch_width, hidden_size)
    return PaperTimingBreakdown(
        compute_s=width * hidden * hidden / rates.compute_mac_per_s,
        internal_memory_s=hidden * hidden / rates.internal_memory_elements_per_s,
        host_io_s=2 * width * hidden / rates.host_io_elements_per_s,
    )


def paper_coprocessed_linear_time(
    branch_width: int,
    hidden_size: int,
    alpha: float,
    gpu_rates: PaperGpuRates,
    pim_rates: PaperPimRates,
) -> CoprocessedLinearTiming:
    """Implement the paper's alpha-dependent GPU and PIM linear equations."""

    if not isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ConfigurationError("alpha must be finite and in [0, 1]")
    width, hidden = _linear_shape(branch_width, hidden_size)
    pim = PaperTimingBreakdown(
        compute_s=(
            width * hidden * hidden * (1 - alpha) / pim_rates.compute_mac_per_s
        ),
        internal_memory_s=(
            hidden * hidden / pim_rates.internal_memory_elements_per_s
        ),
        host_io_s=(
            width * hidden * (2 - alpha) / pim_rates.host_io_elements_per_s
        ),
    )
    gpu = PaperTimingBreakdown(
        compute_s=width * hidden * hidden * alpha / gpu_rates.compute_mac_per_s,
        memory_s=(
            width * hidden * (1 + alpha) + hidden * hidden * alpha
        )
        / gpu_rates.memory_elements_per_s,
    )
    return CoprocessedLinearTiming(alpha=alpha, gpu=gpu, pim=pim)


@dataclass(frozen=True)
class GpuRooflineRates:
    """Conventional calibrated GPU rates in FLOP/s and bytes/s."""

    compute_flop_per_s: float
    memory_bytes_per_s: float
    launch_overhead_s: float = 0.0
    synchronization_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        _positive("roofline.compute_flop_per_s", self.compute_flop_per_s)
        _positive("roofline.memory_bytes_per_s", self.memory_bytes_per_s)
        _nonnegative("roofline.launch_overhead_s", self.launch_overhead_s)
        _nonnegative(
            "roofline.synchronization_overhead_s",
            self.synchronization_overhead_s,
        )

    def with_bandwidth_scale(self, scale: float) -> GpuRooflineRates:
        if not isfinite(scale) or scale <= 0 or scale > 1:
            raise ConfigurationError("bandwidth scale must be in (0, 1]")
        return GpuRooflineRates(
            compute_flop_per_s=self.compute_flop_per_s,
            memory_bytes_per_s=self.memory_bytes_per_s * scale,
            launch_overhead_s=self.launch_overhead_s,
            synchronization_overhead_s=self.synchronization_overhead_s,
        )


@dataclass(frozen=True)
class RooflineEstimate:
    """Overlapped GPU compute/memory timing plus explicit fixed overheads."""

    compute_s: float
    memory_s: float
    launch_overhead_s: float
    synchronization_overhead_s: float

    @property
    def total_s(self) -> float:
        return (
            max(self.compute_s, self.memory_s)
            + self.launch_overhead_s
            + self.synchronization_overhead_s
        )

    @property
    def bottleneck(self) -> str:
        return "compute" if self.compute_s >= self.memory_s else "memory"


def gpu_roofline_time(
    operator: OperatorCost,
    rates: GpuRooflineRates,
) -> RooflineEstimate:
    """Time one operator using Duplex-style overlap and conventional units."""

    return RooflineEstimate(
        compute_s=operator.flops / rates.compute_flop_per_s,
        memory_s=operator.total_bytes / rates.memory_bytes_per_s,
        launch_overhead_s=rates.launch_overhead_s,
        synchronization_overhead_s=rates.synchronization_overhead_s,
    )
