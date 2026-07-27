"""Technique 2A speculation and rollback state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..errors import ConfigurationError


class SpeculationState(str, Enum):
    """Lifecycle of one predicted next-step generation."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class SpeculationResolution:
    """Control-flow and cost outcome after the large PRM resolves a branch."""

    predicted_candidate_id: str
    actual_candidate_id: str
    final_candidate_id: str
    prediction_correct: bool
    committed_speculative_tokens: int
    discarded_speculative_tokens: int
    discarded_kv_bytes: int
    saved_latency_s: float
    rollback_latency_s: float
    regeneration_latency_s: float
    t1_disabled_during_speculation: bool


@dataclass
class SpeculationSession:
    """A single-use session that cannot commit or charge speculative work twice."""

    predicted_candidate_id: str
    speculative_tokens: int
    speculative_duration_s: float
    kv_bytes_per_token: int
    state: SpeculationState = SpeculationState.ACTIVE

    def __post_init__(self) -> None:
        if not self.predicted_candidate_id.strip():
            raise ConfigurationError("predicted_candidate_id must not be empty")
        for name, value in (
            ("speculative_tokens", self.speculative_tokens),
            ("kv_bytes_per_token", self.kv_bytes_per_token),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"{name} must be a non-negative integer")
        if not isfinite(self.speculative_duration_s) or self.speculative_duration_s < 0:
            raise ConfigurationError(
                "speculative_duration_s must be finite and non-negative"
            )

    def resolve(
        self,
        actual_candidate_id: str,
        *,
        rollback_fixed_latency_s: float,
        regeneration_latency_per_token_s: float,
    ) -> SpeculationResolution:
        """Commit a correct prediction or discard it and restart the actual branch."""

        if self.state is not SpeculationState.ACTIVE:
            raise ConfigurationError("a speculation session may be resolved only once")
        if not actual_candidate_id.strip():
            raise ConfigurationError("actual_candidate_id must not be empty")
        for name, value in (
            ("rollback_fixed_latency_s", rollback_fixed_latency_s),
            ("regeneration_latency_per_token_s", regeneration_latency_per_token_s),
        ):
            if not isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be finite and non-negative")

        correct = actual_candidate_id == self.predicted_candidate_id
        if correct:
            self.state = SpeculationState.COMMITTED
            return SpeculationResolution(
                predicted_candidate_id=self.predicted_candidate_id,
                actual_candidate_id=actual_candidate_id,
                final_candidate_id=actual_candidate_id,
                prediction_correct=True,
                committed_speculative_tokens=self.speculative_tokens,
                discarded_speculative_tokens=0,
                discarded_kv_bytes=0,
                saved_latency_s=self.speculative_duration_s,
                rollback_latency_s=0.0,
                regeneration_latency_s=0.0,
                t1_disabled_during_speculation=True,
            )

        self.state = SpeculationState.ROLLED_BACK
        regeneration = self.speculative_tokens * regeneration_latency_per_token_s
        return SpeculationResolution(
            predicted_candidate_id=self.predicted_candidate_id,
            actual_candidate_id=actual_candidate_id,
            final_candidate_id=actual_candidate_id,
            prediction_correct=False,
            committed_speculative_tokens=0,
            discarded_speculative_tokens=self.speculative_tokens,
            discarded_kv_bytes=self.speculative_tokens * self.kv_bytes_per_token,
            saved_latency_s=0.0,
            rollback_latency_s=rollback_fixed_latency_s,
            regeneration_latency_s=regeneration,
            t1_disabled_during_speculation=True,
        )
