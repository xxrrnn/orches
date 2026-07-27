"""Technique 2B token-triggered pipelined PRM verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..errors import ConfigurationError
from ..sim.event import Event, EventTimeline, Resource


@dataclass(frozen=True)
class TokenBatch:
    """Generated tokens that become available together for PRM prefilling."""

    batch_id: str
    ready_time_s: float
    token_count: int

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ConfigurationError("token batch ID must not be empty")
        if not isfinite(self.ready_time_s) or self.ready_time_s < 0:
            raise ConfigurationError("token batch ready time must be non-negative")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int):
            raise ConfigurationError("token batch count must be a positive integer")
        if self.token_count <= 0:
            raise ConfigurationError("token batch count must be a positive integer")


@dataclass(frozen=True)
class VerificationChunk:
    """One early or final PRM chunk scheduled on the GPU."""

    chunk_id: str
    batch_ids: tuple[str, ...]
    token_count: int
    event: Event
    early: bool


@dataclass(frozen=True)
class PipelinedVerificationResult:
    """Technique 2B schedule and latency comparison with serial verification."""

    chunks: tuple[VerificationChunk, ...]
    total_tokens: int
    early_verified_tokens: int
    generation_complete_s: float
    completion_s: float
    serial_completion_s: float

    @property
    def saved_latency_s(self) -> float:
        return max(0.0, self.serial_completion_s - self.completion_s)


@dataclass(frozen=True)
class VerificationPipeline:
    """Thresholded non-preemptive verifier with explicit calibrated costs."""

    min_prefill_tokens: int
    time_per_token_s: float
    fixed_chunk_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_prefill_tokens, bool)
            or not isinstance(self.min_prefill_tokens, int)
            or self.min_prefill_tokens <= 0
        ):
            raise ConfigurationError("min_prefill_tokens must be a positive integer")
        if not isfinite(self.time_per_token_s) or self.time_per_token_s <= 0:
            raise ConfigurationError("time_per_token_s must be finite and positive")
        if (
            not isfinite(self.fixed_chunk_overhead_s)
            or self.fixed_chunk_overhead_s < 0
        ):
            raise ConfigurationError(
                "fixed_chunk_overhead_s must be finite and non-negative"
            )

    def schedule(
        self,
        batches: tuple[TokenBatch, ...],
        *,
        generation_complete_s: float,
        timeline: EventTimeline | None = None,
        event_prefix: str = "verification",
    ) -> PipelinedVerificationResult:
        if not batches:
            raise ConfigurationError("at least one token batch is required")
        if not isfinite(generation_complete_s) or generation_complete_s < 0:
            raise ConfigurationError("generation_complete_s must be non-negative")
        if not event_prefix.strip():
            raise ConfigurationError("event_prefix must not be empty")
        ids = [batch.batch_id for batch in batches]
        if len(set(ids)) != len(ids):
            raise ConfigurationError("token batch IDs must be unique")
        if generation_complete_s < max(batch.ready_time_s for batch in batches):
            raise ConfigurationError(
                "generation_complete_s must not precede a token batch"
            )

        active_timeline = timeline if timeline is not None else EventTimeline()
        ordered = sorted(
            batches,
            key=lambda batch: (batch.ready_time_s, batch.batch_id),
        )
        pending: list[TokenBatch] = []
        chunks: list[VerificationChunk] = []

        def emit(*, early: bool, ready_time_s: float) -> None:
            token_count = sum(batch.token_count for batch in pending)
            chunk_id = f"{event_prefix}-{len(chunks):04d}"
            duration = (
                token_count * self.time_per_token_s + self.fixed_chunk_overhead_s
            )
            event = active_timeline.schedule(
                chunk_id,
                Resource.GPU,
                duration,
                ready_time_s=ready_time_s,
            )
            chunks.append(
                VerificationChunk(
                    chunk_id=chunk_id,
                    batch_ids=tuple(batch.batch_id for batch in pending),
                    token_count=token_count,
                    event=event,
                    early=early,
                )
            )
            pending.clear()

        for batch in ordered:
            pending.append(batch)
            if sum(item.token_count for item in pending) >= self.min_prefill_tokens:
                emit(early=True, ready_time_s=batch.ready_time_s)
        if pending:
            emit(early=False, ready_time_s=generation_complete_s)

        total_tokens = sum(batch.token_count for batch in batches)
        serial_duration = (
            total_tokens * self.time_per_token_s
            + self.fixed_chunk_overhead_s
        )
        return PipelinedVerificationResult(
            chunks=tuple(chunks),
            total_tokens=total_tokens,
            early_verified_tokens=sum(
                chunk.token_count for chunk in chunks if chunk.early
            ),
            generation_complete_s=generation_complete_s,
            completion_s=max(chunk.event.end_s for chunk in chunks),
            serial_completion_s=generation_complete_s + serial_duration,
        )
