"""Request-level replay that composes ORCHES Techniques 1, 2, and 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

from .errors import ConfigurationError
from .memory import (
    AddressCache,
    AddressLookup,
    CompactionController,
    CompactionDecision,
    CompactionTrace,
    MemoryKind,
    MemoryStats,
    PimMemoryAllocator,
    SharedKvBuffer,
    build_compaction_trace,
)
from .predictor import (
    CandidateScorePath,
    PipelinedVerificationResult,
    PredictionDecision,
    ScoreAggregation,
    SpeculationResolution,
    SpeculationSession,
    TokenBatch,
    VerificationPipeline,
    predict_candidate,
)
from .scheduler import GenerationPlan, OfflineScheduler, plan_generation_step
from .sim import Event, EventTimeline, Resource
from .workload import StepTrace, TtcRequestTrace


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def _positive_float(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be finite and positive")


def _nonnegative_float(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ReplayConfig:
    """Explicit assumptions and calibrated rates required by request replay."""

    hidden_size: int
    kv_bytes_per_token: int
    operator_repetitions: int
    large_prm_time_per_token_s: float
    rollback_fixed_latency_s: float
    regeneration_latency_per_token_s: float
    compaction_internal_bytes_per_s: float
    compaction_transaction_bytes: int
    score_aggregation: ScoreAggregation
    history_alignment: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("kv_bytes_per_token", self.kv_bytes_per_token),
            ("operator_repetitions", self.operator_repetitions),
            ("compaction_transaction_bytes", self.compaction_transaction_bytes),
        ):
            _positive_integer(name, value)
        _positive_float(
            "large_prm_time_per_token_s",
            self.large_prm_time_per_token_s,
        )
        _nonnegative_float(
            "rollback_fixed_latency_s",
            self.rollback_fixed_latency_s,
        )
        _nonnegative_float(
            "regeneration_latency_per_token_s",
            self.regeneration_latency_per_token_s,
        )
        _positive_float(
            "compaction_internal_bytes_per_s",
            self.compaction_internal_bytes_per_s,
        )
        if not isinstance(self.score_aggregation, ScoreAggregation):
            raise ConfigurationError("score_aggregation must be a ScoreAggregation")
        if not isinstance(self.history_alignment, bool):
            raise ConfigurationError("history_alignment must be a boolean")


@dataclass(frozen=True)
class GenerationExecution:
    """T1 execution intervals for one full or post-speculation step."""

    plan: GenerationPlan
    ready_time_s: float
    gpu_event: Event | None
    pim_event: Event | None
    prior_speculative_event: Event | None
    t1_work_fraction: float
    speculative_work_fraction: float
    completion_s: float


@dataclass(frozen=True)
class AddressLookupExecution:
    """Controller event associated with one candidate mapping lookup."""

    candidate_id: str
    lookup: AddressLookup
    event: Event


@dataclass(frozen=True)
class CompactionExecution:
    """T3 decision, generated DRAM traffic, and its PIM interval."""

    decision: CompactionDecision
    trace: CompactionTrace
    event: Event


@dataclass(frozen=True)
class StepReplayResult:
    """Complete decisions and accounting for one reasoning step."""

    step_index: int
    generation: GenerationExecution
    address_lookups: tuple[AddressLookupExecution, ...]
    verification: PipelinedVerificationResult
    prediction: PredictionDecision
    large_prm_event: Event
    speculation: SpeculationResolution | None
    rollback_event: Event | None
    memory_before_prune: MemoryStats
    memory_after_step: MemoryStats
    compaction_decision: CompactionDecision | None
    compaction: CompactionExecution | None


@dataclass(frozen=True)
class OrchesReplayResult:
    """Request-level timeline and aggregate M3 observability."""

    request_id: str
    steps: tuple[StepReplayResult, ...]
    timeline: EventTimeline
    final_memory: MemoryStats
    prediction_correct: int
    prediction_total: int
    compaction_read_bytes: int
    compaction_write_bytes: int

    @property
    def prediction_accuracy(self) -> float:
        if self.prediction_total == 0:
            return 0.0
        return self.prediction_correct / self.prediction_total


class OrchesRequestReplayer:
    """Single-use orchestrator for one replayable TTC request trace."""

    def __init__(
        self,
        *,
        offline_scheduler: OfflineScheduler,
        verification_pipeline: VerificationPipeline,
        allocator: PimMemoryAllocator,
        address_cache: AddressCache,
        shared_kv_buffer: SharedKvBuffer,
        compaction_controller: CompactionController | None,
        config: ReplayConfig,
    ) -> None:
        if not allocator.weights_sealed:
            raise ConfigurationError("replay requires sealed model weights")
        if any(block.kind is not MemoryKind.WEIGHT for block in allocator.blocks):
            raise ConfigurationError("replay allocator must not contain pre-existing KV")
        if allocator.alignment_bytes % config.compaction_transaction_bytes:
            raise ConfigurationError(
                "allocator alignment must be divisible by compaction transaction size"
            )
        self.offline_scheduler = offline_scheduler
        self.verification_pipeline = verification_pipeline
        self.allocator = allocator
        self.address_cache = address_cache
        self.shared_kv_buffer = shared_kv_buffer
        self.compaction_controller = compaction_controller
        self.config = config
        self._used = False

    def run(self, trace: TtcRequestTrace) -> OrchesReplayResult:
        """Replay one trace and return all decisions without hiding assumptions."""

        if self._used:
            raise ConfigurationError("an OrchesRequestReplayer instance is single-use")
        self._used = True
        timeline = EventTimeline()
        self._write_back_kv(
            block_id=f"{trace.request_id}:prompt",
            owner_id=trace.request_id,
            kind=MemoryKind.SHARED_KV,
            token_count=trace.prompt_tokens + trace.image_tokens,
        )

        first_plan = self._plan(trace.steps[0])
        current_generation = self._schedule_generation(
            timeline,
            first_plan,
            event_prefix=f"{trace.request_id}:step-0:generation",
            ready_time_s=0.0,
            t1_work_fraction=1.0,
            speculative_work_fraction=0.0,
            prior_speculative_event=None,
        )
        history_small: list[float] = []
        history_large: list[float] = []
        step_results: list[StepReplayResult] = []
        correct_predictions = 0
        prediction_total = 0

        for step in trace.steps:
            candidate_blocks = self._allocate_candidates(step)
            lookups, batches, mapping_complete_s = self._map_candidate_tokens(
                trace.request_id,
                step,
                current_generation,
                candidate_blocks,
                timeline,
            )
            verification = self.verification_pipeline.schedule(
                batches,
                generation_complete_s=max(
                    current_generation.completion_s,
                    mapping_complete_s,
                ),
                timeline=timeline,
                event_prefix=(
                    f"{trace.request_id}:step-{step.step_index}:small-prm"
                ),
            )
            prediction = predict_candidate(
                tuple(
                    CandidateScorePath(
                        candidate.candidate_id,
                        current_small_prm_score=candidate.small_prm_score,
                        historical_small_prm_scores=tuple(history_small),
                        historical_large_prm_scores=tuple(history_large),
                    )
                    for candidate in step.candidates
                ),
                history_alignment=self.config.history_alignment,
                aggregation=self.config.score_aggregation,
            )
            total_candidate_tokens = sum(
                candidate.generated_tokens for candidate in step.candidates
            )
            dependencies = tuple(
                [lookup.event.event_id for lookup in lookups]
                + [chunk.event.event_id for chunk in verification.chunks]
            )
            large_prm_event = timeline.schedule(
                f"{trace.request_id}:step-{step.step_index}:large-prm",
                Resource.GPU,
                total_candidate_tokens
                * self.config.large_prm_time_per_token_s,
                ready_time_s=current_generation.completion_s,
                dependencies=dependencies,
            )

            next_step = (
                trace.steps[step.step_index + 1]
                if step.step_index + 1 < len(trace.steps)
                else None
            )
            next_plan = self._plan(next_step) if next_step is not None else None
            speculative_event, speculative_fraction, speculation = (
                self._speculate_next_step(
                    trace.request_id,
                    step,
                    next_step,
                    next_plan,
                    prediction,
                    verification,
                    large_prm_event,
                    timeline,
                )
                if next_step is not None and next_plan is not None
                else (None, 0.0, None)
            )
            if speculation is not None:
                prediction_total += 1
                correct_predictions += int(speculation.prediction_correct)

            memory_before_prune = self.allocator.stats()
            self.allocator.promote_to_shared(
                candidate_blocks[step.selected_candidate_id]
            )
            for candidate_id in step.pruned_candidate_ids:
                block_id = candidate_blocks[candidate_id]
                self.allocator.prune(block_id)
                self.address_cache.invalidate(block_id)
            if speculation is not None and not speculation.prediction_correct:
                self._materialize_discarded_speculation(
                    trace.request_id,
                    step.step_index,
                    speculation,
                )

            compaction_decision, compaction = self._compact_after_verification(
                trace.request_id,
                step,
                large_prm_event,
                timeline,
            )
            rollback_event = self._schedule_rollback(
                trace.request_id,
                step,
                speculation,
                large_prm_event,
                timeline,
            )
            memory_after_step = self.allocator.stats()
            step_results.append(
                StepReplayResult(
                    step_index=step.step_index,
                    generation=current_generation,
                    address_lookups=lookups,
                    verification=verification,
                    prediction=prediction,
                    large_prm_event=large_prm_event,
                    speculation=speculation,
                    rollback_event=rollback_event,
                    memory_before_prune=memory_before_prune,
                    memory_after_step=memory_after_step,
                    compaction_decision=compaction_decision,
                    compaction=compaction,
                )
            )
            history_small.append(step.selected_candidate.small_prm_score)
            history_large.append(step.selected_candidate.large_prm_score)

            if next_step is not None and next_plan is not None:
                correct = speculation is not None and speculation.prediction_correct
                ready_time = (
                    rollback_event.end_s
                    if rollback_event is not None
                    else large_prm_event.end_s
                )
                if compaction is not None:
                    ready_time = max(ready_time, compaction.event.end_s)
                current_generation = self._schedule_generation(
                    timeline,
                    next_plan,
                    event_prefix=(
                        f"{trace.request_id}:step-{next_step.step_index}:generation"
                    ),
                    ready_time_s=ready_time,
                    t1_work_fraction=(1.0 - speculative_fraction if correct else 1.0),
                    speculative_work_fraction=(
                        speculative_fraction if correct else 0.0
                    ),
                    prior_speculative_event=(speculative_event if correct else None),
                )

        timeline.validate()
        return OrchesReplayResult(
            request_id=trace.request_id,
            steps=tuple(step_results),
            timeline=timeline,
            final_memory=self.allocator.stats(),
            prediction_correct=correct_predictions,
            prediction_total=prediction_total,
            compaction_read_bytes=sum(
                result.compaction.trace.read_bytes
                for result in step_results
                if result.compaction is not None
            ),
            compaction_write_bytes=sum(
                result.compaction.trace.write_bytes
                for result in step_results
                if result.compaction is not None
            ),
        )

    def _plan(self, step: StepTrace) -> GenerationPlan:
        return plan_generation_step(
            self.offline_scheduler,
            branch_width=len(step.candidates),
            shared_kv_tokens=step.shared_kv_tokens,
            unique_kv_tokens=tuple(
                candidate.unique_kv_tokens for candidate in step.candidates
            ),
            hidden_size=self.config.hidden_size,
            repetitions=self.config.operator_repetitions,
        )

    def _schedule_generation(
        self,
        timeline: EventTimeline,
        plan: GenerationPlan,
        *,
        event_prefix: str,
        ready_time_s: float,
        t1_work_fraction: float,
        speculative_work_fraction: float,
        prior_speculative_event: Event | None,
    ) -> GenerationExecution:
        for name, value in (
            ("t1_work_fraction", t1_work_fraction),
            ("speculative_work_fraction", speculative_work_fraction),
        ):
            if not isfinite(value) or value < 0 or value > 1:
                raise ConfigurationError(f"{name} must be in [0, 1]")
        pim_event = (
            timeline.schedule(
                f"{event_prefix}:pim",
                Resource.PIM,
                plan.pim_time_s * t1_work_fraction,
                ready_time_s=ready_time_s,
            )
            if t1_work_fraction > 0 and plan.pim_time_s > 0
            else None
        )
        gpu_event = (
            timeline.schedule(
                f"{event_prefix}:gpu",
                Resource.GPU,
                plan.gpu_time_s * t1_work_fraction,
                ready_time_s=ready_time_s,
            )
            if t1_work_fraction > 0 and plan.gpu_time_s > 0
            else None
        )
        completion = max(
            ready_time_s,
            *(event.end_s for event in (pim_event, gpu_event) if event is not None),
            *(
                (prior_speculative_event.end_s,)
                if prior_speculative_event is not None
                else ()
            ),
        )
        return GenerationExecution(
            plan=plan,
            ready_time_s=ready_time_s,
            gpu_event=gpu_event,
            pim_event=pim_event,
            prior_speculative_event=prior_speculative_event,
            t1_work_fraction=t1_work_fraction,
            speculative_work_fraction=speculative_work_fraction,
            completion_s=completion,
        )

    def _allocate_candidates(self, step: StepTrace) -> dict[str, str]:
        block_ids: dict[str, str] = {}
        for candidate in step.candidates:
            block_id = candidate.candidate_id
            self._write_back_kv(
                block_id=block_id,
                owner_id=candidate.candidate_id,
                kind=MemoryKind.UNIQUE_KV,
                token_count=candidate.unique_kv_tokens,
            )
            block_ids[candidate.candidate_id] = block_id
        return block_ids

    def _write_back_kv(
        self,
        *,
        block_id: str,
        owner_id: str,
        kind: MemoryKind,
        token_count: int,
    ) -> None:
        self.shared_kv_buffer.stage(
            block_id,
            owner_id=owner_id,
            kind=kind,
            size_bytes=token_count * self.config.kv_bytes_per_token,
        )
        self.shared_kv_buffer.write_back_next(self.allocator)

    def _map_candidate_tokens(
        self,
        request_id: str,
        step: StepTrace,
        generation: GenerationExecution,
        candidate_blocks: dict[str, str],
        timeline: EventTimeline,
    ) -> tuple[tuple[AddressLookupExecution, ...], tuple[TokenBatch, ...], float]:
        maximum_timestamp_us = max(
            candidate.token_timestamps_us[-1] for candidate in step.candidates
        )
        generation_span_s = generation.completion_s - generation.ready_time_s
        lookups: list[AddressLookupExecution] = []
        batches: list[TokenBatch] = []
        for candidate in step.candidates:
            lookup = self.address_cache.resolve(
                self.allocator,
                candidate_blocks[candidate.candidate_id],
            )
            first_ready = generation.ready_time_s + generation_span_s * (
                candidate.token_timestamps_us[0] / maximum_timestamp_us
            )
            lookup_event = timeline.schedule(
                f"{request_id}:step-{step.step_index}:addr:{candidate.candidate_id}",
                Resource.CONTROLLER,
                lookup.latency_s,
                ready_time_s=first_ready,
            )
            lookups.append(
                AddressLookupExecution(
                    candidate_id=candidate.candidate_id,
                    lookup=lookup,
                    event=lookup_event,
                )
            )
            for token_index, timestamp_us in enumerate(
                candidate.token_timestamps_us
            ):
                token_ready = generation.ready_time_s + generation_span_s * (
                    timestamp_us / maximum_timestamp_us
                )
                batches.append(
                    TokenBatch(
                        batch_id=(
                            f"{candidate.candidate_id}:token-{token_index:04d}"
                        ),
                        ready_time_s=max(token_ready, lookup_event.end_s),
                        token_count=1,
                    )
                )
        mapping_complete = max(lookup.event.end_s for lookup in lookups)
        return tuple(lookups), tuple(batches), mapping_complete

    def _speculate_next_step(
        self,
        request_id: str,
        step: StepTrace,
        next_step: StepTrace,
        next_plan: GenerationPlan,
        prediction: PredictionDecision,
        verification: PipelinedVerificationResult,
        large_prm_event: Event,
        timeline: EventTimeline,
    ) -> tuple[Event | None, float, SpeculationResolution]:
        speculation_ready_s = max(
            verification.completion_s,
            timeline.available_time_s(Resource.PIM),
        )
        available_window_s = max(0.0, large_prm_event.end_s - speculation_ready_s)
        duration_s = min(next_plan.pim_only_time_s, available_window_s)
        speculative_fraction = (
            duration_s / next_plan.pim_only_time_s
            if next_plan.pim_only_time_s > 0
            else 0.0
        )
        next_tokens = sum(
            candidate.generated_tokens for candidate in next_step.candidates
        )
        speculative_tokens = floor(next_tokens * speculative_fraction)
        event = (
            timeline.schedule(
                f"{request_id}:step-{step.step_index}:speculative-pim",
                Resource.PIM,
                duration_s,
                ready_time_s=verification.completion_s,
                speculative=True,
            )
            if duration_s > 0
            else None
        )
        session = SpeculationSession(
            prediction.candidate_id,
            speculative_tokens,
            duration_s,
            self.config.kv_bytes_per_token,
        )
        resolution = session.resolve(
            step.selected_candidate_id,
            rollback_fixed_latency_s=self.config.rollback_fixed_latency_s,
            regeneration_latency_per_token_s=(
                self.config.regeneration_latency_per_token_s
            ),
        )
        return event, speculative_fraction, resolution

    def _materialize_discarded_speculation(
        self,
        request_id: str,
        step_index: int,
        speculation: SpeculationResolution,
    ) -> None:
        if speculation.discarded_speculative_tokens == 0:
            return
        block_id = f"{request_id}:step-{step_index}:discarded-speculation"
        self._write_back_kv(
            block_id=block_id,
            owner_id=speculation.predicted_candidate_id,
            kind=MemoryKind.UNIQUE_KV,
            token_count=speculation.discarded_speculative_tokens,
        )
        self.allocator.prune(block_id)

    def _compact_after_verification(
        self,
        request_id: str,
        step: StepTrace,
        large_prm_event: Event,
        timeline: EventTimeline,
    ) -> tuple[CompactionDecision | None, CompactionExecution | None]:
        if self.compaction_controller is None:
            return None, None
        decision = self.compaction_controller.observe_verification(
            step.step_index + 1,
            self.allocator,
        )
        if decision.result is None:
            return decision, None
        trace = build_compaction_trace(
            decision.result,
            transaction_bytes=self.config.compaction_transaction_bytes,
        )
        event = timeline.schedule(
            f"{request_id}:step-{step.step_index}:compaction",
            Resource.PIM,
            trace.total_bytes / self.config.compaction_internal_bytes_per_s,
            ready_time_s=large_prm_event.end_s,
        )
        return decision, CompactionExecution(
            decision=decision,
            trace=trace,
            event=event,
        )

    def _schedule_rollback(
        self,
        request_id: str,
        step: StepTrace,
        speculation: SpeculationResolution | None,
        large_prm_event: Event,
        timeline: EventTimeline,
    ) -> Event | None:
        if speculation is None or speculation.prediction_correct:
            return None
        return timeline.schedule(
            f"{request_id}:step-{step.step_index}:rollback",
            Resource.CONTROLLER,
            speculation.rollback_latency_s,
            ready_time_s=large_prm_event.end_s,
        )
