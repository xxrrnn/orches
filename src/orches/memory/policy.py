"""Explicit trigger policies for ORCHES dynamic memory reorganization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..errors import ConfigurationError
from .allocator import CompactionResult, PimMemoryAllocator


@dataclass(frozen=True)
class CompactionPolicy:
    """Interval and/or beta trigger with no hidden evaluation defaults."""

    interval_verifications: int | None = None
    beta_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.interval_verifications is None and self.beta_threshold is None:
            raise ConfigurationError("at least one compaction trigger is required")
        if self.interval_verifications is not None and (
            isinstance(self.interval_verifications, bool)
            or not isinstance(self.interval_verifications, int)
            or self.interval_verifications <= 0
        ):
            raise ConfigurationError(
                "interval_verifications must be a positive integer"
            )
        if self.beta_threshold is not None and (
            not isfinite(self.beta_threshold)
            or self.beta_threshold <= 0
            or self.beta_threshold > 1
        ):
            raise ConfigurationError("beta_threshold must be in (0, 1]")


@dataclass(frozen=True)
class CompactionDecision:
    """One post-verification trigger observation and optional reorganization."""

    verification_index: int
    observed_beta: float
    trigger_reasons: tuple[str, ...]
    result: CompactionResult | None

    @property
    def triggered(self) -> bool:
        return self.result is not None


class CompactionController:
    """State machine that observes each PRM verification exactly once."""

    def __init__(self, policy: CompactionPolicy) -> None:
        self.policy = policy
        self._last_verification_index = 0

    def observe_verification(
        self,
        verification_index: int,
        allocator: PimMemoryAllocator,
    ) -> CompactionDecision:
        if (
            isinstance(verification_index, bool)
            or not isinstance(verification_index, int)
            or verification_index != self._last_verification_index + 1
        ):
            raise ConfigurationError(
                "verification_index must be contiguous and start at one"
            )
        self._last_verification_index = verification_index
        beta = allocator.stats().beta
        reasons: list[str] = []
        interval = self.policy.interval_verifications
        if interval is not None and verification_index % interval == 0:
            reasons.append("fixed_interval")
        threshold = self.policy.beta_threshold
        if threshold is not None and beta >= threshold:
            reasons.append("beta_threshold")

        result = allocator.compact() if reasons and beta > 0 else None
        return CompactionDecision(
            verification_index=verification_index,
            observed_beta=beta,
            trigger_reasons=tuple(reasons),
            result=result,
        )
