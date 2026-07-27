"""Technique 1A width tiers and analytical linear-layer co-processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigurationError
from ..models.timing import (
    CoprocessedLinearTiming,
    PaperGpuRates,
    PaperPimRates,
    paper_coprocessed_linear_time,
    paper_gpu_linear_time,
    paper_pim_linear_time,
)


class WorkloadTier(str, Enum):
    """Paper's small, medium, and large branch-width regions."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class AttentionPlacement(str, Enum):
    """Device selected for one attention KV class."""

    GPU = "gpu"
    PIM = "pim"


class LinearPlacement(str, Enum):
    """Selected linear execution mode."""

    GPU = "gpu"
    PIM = "pim"
    COPROCESSED = "coprocessed"


@dataclass(frozen=True)
class TierThresholds:
    """Calibration-derived inclusive lower bounds for medium and large tiers."""

    medium_min_width: int
    large_min_width: int

    def __post_init__(self) -> None:
        for name, value in (
            ("medium_min_width", self.medium_min_width),
            ("large_min_width", self.large_min_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.medium_min_width >= self.large_min_width:
            raise ConfigurationError(
                "medium_min_width must be smaller than large_min_width"
            )

    def classify(self, branch_width: int) -> WorkloadTier:
        if isinstance(branch_width, bool) or not isinstance(branch_width, int):
            raise ConfigurationError("branch_width must be a positive integer")
        if branch_width <= 0:
            raise ConfigurationError("branch_width must be a positive integer")
        if branch_width >= self.large_min_width:
            return WorkloadTier.LARGE
        if branch_width >= self.medium_min_width:
            return WorkloadTier.MEDIUM
        return WorkloadTier.SMALL


@dataclass(frozen=True)
class OfflineAssignment:
    """Explainable T1A placement for one width/hidden-size point."""

    tier: WorkloadTier
    linear: LinearPlacement
    linear_alpha_gpu: float
    shared_attention: AttentionPlacement
    unique_attention: AttentionPlacement
    primary_linear_time_s: float
    coprocessed_time_s: float
    chosen_linear_time_s: float
    gpu_only_time_s: float
    pim_only_time_s: float
    balance: CoprocessedLinearTiming
    coprocessed_host_io_elements: float
    coprocessed_barrier_stall_s: float
    reason: str


def solve_linear_balance_alpha(
    branch_width: int,
    hidden_size: int,
    gpu_rates: PaperGpuRates,
    pim_rates: PaperPimRates,
) -> float:
    """Solve the linear Eq. (3)-(4) equality and clamp it to `[0, 1]`."""

    if branch_width <= 0 or hidden_size <= 0:
        raise ConfigurationError("branch_width and hidden_size must be positive")
    width = float(branch_width)
    hidden = float(hidden_size)

    pim_intercept = (
        width * hidden * hidden / pim_rates.compute_mac_per_s
        + 2 * width * hidden / pim_rates.host_io_elements_per_s
        + hidden * hidden / pim_rates.internal_memory_elements_per_s
    )
    pim_slope = -(
        width * hidden * hidden / pim_rates.compute_mac_per_s
        + width * hidden / pim_rates.host_io_elements_per_s
    )
    gpu_intercept = width * hidden / gpu_rates.memory_elements_per_s
    gpu_slope = (
        width * hidden * hidden / gpu_rates.compute_mac_per_s
        + (width * hidden + hidden * hidden) / gpu_rates.memory_elements_per_s
    )
    denominator = gpu_slope - pim_slope
    if denominator <= 0:
        raise ConfigurationError("linear balance equation has no positive slope gap")
    alpha = (pim_intercept - gpu_intercept) / denominator
    return min(1.0, max(0.0, alpha))


@dataclass(frozen=True)
class OfflineScheduler:
    """Technique 1A scheduler parameterized by frozen thresholds and rates."""

    thresholds: TierThresholds
    gpu_rates: PaperGpuRates
    pim_rates: PaperPimRates

    def assign(self, branch_width: int, hidden_size: int) -> OfflineAssignment:
        tier = self.thresholds.classify(branch_width)
        gpu_only = paper_gpu_linear_time(branch_width, hidden_size, self.gpu_rates)
        pim_only = paper_pim_linear_time(branch_width, hidden_size, self.pim_rates)
        alpha = solve_linear_balance_alpha(
            branch_width,
            hidden_size,
            self.gpu_rates,
            self.pim_rates,
        )
        balance = paper_coprocessed_linear_time(
            branch_width,
            hidden_size,
            alpha,
            self.gpu_rates,
            self.pim_rates,
        )

        if tier is WorkloadTier.LARGE:
            primary_mode = LinearPlacement.GPU
            primary_time = gpu_only.total_s
            shared = AttentionPlacement.GPU
        else:
            primary_mode = LinearPlacement.PIM
            primary_time = pim_only.total_s
            shared = (
                AttentionPlacement.PIM
                if tier is WorkloadTier.SMALL
                else AttentionPlacement.GPU
            )

        paper_coprocessing_condition = (
            pim_only.total_s >= balance.critical_path_s
        )
        if balance.critical_path_s < primary_time and paper_coprocessing_condition:
            linear = LinearPlacement.COPROCESSED
            chosen_time = balance.critical_path_s
            reason = (
                "balanced co-processing is faster than the tier's primary "
                f"{primary_mode.value} linear execution"
            )
        else:
            linear = primary_mode
            chosen_time = primary_time
            alpha = 1.0 if primary_mode is LinearPlacement.GPU else 0.0
            reason = (
                "co-processing does not improve the tier's primary "
                f"{primary_mode.value} linear execution or fails the paper's "
                "PIM-to-co-processing inequality"
            )

        return OfflineAssignment(
            tier=tier,
            linear=linear,
            linear_alpha_gpu=alpha,
            shared_attention=shared,
            unique_attention=AttentionPlacement.PIM,
            primary_linear_time_s=primary_time,
            coprocessed_time_s=balance.critical_path_s,
            chosen_linear_time_s=chosen_time,
            gpu_only_time_s=gpu_only.total_s,
            pim_only_time_s=pim_only.total_s,
            balance=balance,
            coprocessed_host_io_elements=(
                branch_width * hidden_size * (2 - balance.alpha)
            ),
            coprocessed_barrier_stall_s=abs(
                balance.gpu.total_s - balance.pim.total_s
            ),
            reason=reason,
        )
