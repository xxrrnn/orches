"""Schema-v2 request replay over the ORCHES T1/T2/T3 functional model."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .errors import ConfigurationError
from .memory import MemoryKind
from .predictor import (
    CandidateScorePath,
    PipelinedVerificationResult,
    PredictionDecision,
    SpeculationResolution,
    SpeculationSession,
    TokenBatch,
    predict_candidate,
)
from .replay import (
    AddressLookupExecution,
    GenerationExecution,
    OrchesReplayResult,
    OrchesRequestReplayer,
    SelectionEventExecution,
    StepReplayResult,
    VerifierCallExecution,
)
from .scheduler import GenerationPlan, plan_generation_step
from .sim import Event, EventTimeline, Resource
from .workload import (
    CandidateTraceV2,
    KvBlockTraceV2,
    SelectionEventTraceV2,
    StepTraceV2,
    TtcRequestTraceV2,
    VerifierCallTraceV2,
    VerifierKind,
)


@dataclass(frozen=True)
class V2KvIndex:
    """Derived immutable indexes over one validated logical KV tree."""

    blocks: dict[str, KvBlockTraceV2]
    candidates: dict[str, CandidateTraceV2]
    root_block_id: str

    @classmethod
    def from_trace(cls, trace: TtcRequestTraceV2) -> V2KvIndex:
        blocks = {block.block_id: block for block in trace.kv_blocks}
        candidates = {
            candidate.candidate_id: candidate
            for step in trace.steps
            for candidate in step.candidates
        }
        root = next(
            block for block in trace.kv_blocks if block.parent_block_id is None
        )
        return cls(blocks, candidates, root.block_id)

    def lineage(self, block_id: str) -> tuple[str, ...]:
        result: list[str] = []
        current: str | None = block_id
        while current is not None:
            result.append(current)
            current = self.blocks[current].parent_block_id
        result.reverse()
        return tuple(result)

    def parent_terminal_block(self, candidate: CandidateTraceV2) -> str:
        if candidate.parent_candidate_id is None:
            return self.root_block_id
        return self.candidates[candidate.parent_candidate_id].terminal_kv_block_id

    def common_parent_kv_tokens(
        self,
        candidates: tuple[CandidateTraceV2, ...],
    ) -> int:
        common: set[str] | None = None
        for candidate in candidates:
            lineage = set(self.lineage(self.parent_terminal_block(candidate)))
            common = lineage if common is None else common & lineage
        assert common is not None
        return sum(self.blocks[block_id].token_count for block_id in common)


class OrchesV2RequestReplayer(OrchesRequestReplayer):
    """Replay exact-token traces while preserving multi-beam KV lifetimes."""

    def run(self, trace: TtcRequestTraceV2) -> OrchesReplayResult:
        if self._used:
            raise ConfigurationError(
                "an OrchesRequestReplayer instance is single-use"
            )
        self._used = True
        timeline = EventTimeline()
        kv_index = V2KvIndex.from_trace(trace)
        root = kv_index.blocks[kv_index.root_block_id]
        self._write_back_kv(
            block_id=root.block_id,
            owner_id=trace.request_id,
            kind=MemoryKind.SHARED_KV,
            token_count=root.token_count,
        )

        current_plans = self._plan_step(trace.steps[0], kv_index)
        current_generation = self._schedule_generation_phases(
            timeline,
            current_plans,
            event_prefix=f"{trace.request_id}:step-0:generation",
            ready_time_s=0.0,
            t1_work_fraction=1.0,
            speculative_work_fraction=0.0,
            prior_speculative_event=None,
        )
        history: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        step_results: list[StepReplayResult] = []
        correct_predictions = 0
        prediction_total = 0

        for step in trace.steps:
            terminal_blocks = self._allocate_candidate_extensions(step, kv_index)
            lookups, batches, mapping_complete_s = self._map_candidate_tokens_v2(
                trace.request_id,
                step,
                current_generation,
                terminal_blocks,
                timeline,
            )
            driver_calls = self._selection_driver_calls(step)
            driver_ids = {call.call_id for call in driver_calls}
            small_calls = self._small_scalar_calls(step, driver_ids)
            verification = self._schedule_small_verification(
                trace.request_id,
                step,
                batches,
                small_calls,
                current_generation,
                mapping_complete_s,
                timeline,
            )
            small_scores = self._scores(small_calls)
            final_scores = self._final_scalar_scores(driver_calls)
            prediction = self._predict(
                trace,
                step,
                small_scores,
                history,
            )
            memory_before_prune = self.allocator.stats()
            verifier_executions, selection_executions = (
                self._schedule_verifier_and_selection_events(
                    trace.request_id,
                    step,
                    driver_calls,
                    lookups,
                    verification,
                    current_generation,
                    mapping_complete_s,
                    timeline,
                )
            )
            large_prm_event = verifier_executions[-1].event

            next_step = (
                trace.steps[step.step_index + 1]
                if step.step_index + 1 < len(trace.steps)
                else None
            )
            next_plans = (
                self._plan_step(next_step, kv_index) if next_step is not None else ()
            )
            speculative_event, speculative_fraction, speculation = (
                self._speculate_next_step_v2(
                    trace.request_id,
                    trace,
                    step,
                    next_step,
                    next_plans,
                    prediction,
                    verification,
                    large_prm_event,
                    timeline,
                )
                if next_step is not None and next_plans and prediction is not None
                else (None, 0.0, None)
            )
            if speculation is not None:
                prediction_total += 1
                correct_predictions += int(speculation.prediction_correct)

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
                    speculative_event=speculative_event,
                    speculative_plan=(next_plans[0] if next_plans else None),
                    speculative_work_fraction=speculative_fraction,
                    rollback_event=rollback_event,
                    memory_before_prune=memory_before_prune,
                    memory_after_step=memory_after_step,
                    compaction_decision=compaction_decision,
                    compaction=compaction,
                    verifier_calls=verifier_executions,
                    selection_events=selection_executions,
                    speculative_plans=next_plans if speculative_event is not None else (),
                )
            )
            self._update_history(
                step,
                small_scores,
                final_scores,
                history,
            )

            if next_step is not None and next_plans:
                correct = speculation is not None and speculation.prediction_correct
                ready_time = (
                    rollback_event.end_s
                    if rollback_event is not None
                    else large_prm_event.end_s
                )
                if selection_executions:
                    ready_time = max(ready_time, selection_executions[-1].event.end_s)
                if compaction is not None:
                    ready_time = max(ready_time, compaction.event.end_s)
                current_generation = self._schedule_generation_phases(
                    timeline,
                    next_plans,
                    event_prefix=(
                        f"{trace.request_id}:step-{next_step.step_index}:generation"
                    ),
                    ready_time_s=ready_time,
                    t1_work_fraction=(
                        1.0 - speculative_fraction if correct else 1.0
                    ),
                    speculative_work_fraction=(
                        speculative_fraction if correct else 0.0
                    ),
                    prior_speculative_event=(
                        speculative_event if correct else None
                    ),
                )

        timeline.validate()
        return OrchesReplayResult(
            request_id=trace.request_id,
            steps=tuple(step_results),
            timeline=timeline,
            final_memory=self.allocator.stats(),
            address_cache=self.address_cache.stats(),
            shared_kv_buffer=self.shared_kv_buffer.stats(),
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

    def _plan_step(
        self,
        step: StepTraceV2,
        kv_index: V2KvIndex,
    ) -> tuple[GenerationPlan, ...]:
        plans: list[GenerationPlan] = []
        max_generated = max(candidate.generated_tokens for candidate in step.candidates)
        for token_round in range(max_generated):
            active = tuple(
                candidate
                for candidate in step.candidates
                if candidate.generated_tokens > token_round
            )
            shared_tokens = kv_index.common_parent_kv_tokens(active)
            unique_tokens = tuple(
                candidate.input_tokens.model_token_count
                + token_round
                - shared_tokens
                for candidate in active
            )
            plans.append(
                plan_generation_step(
                    self.offline_scheduler,
                    branch_width=len(active),
                    shared_kv_tokens=shared_tokens,
                    unique_kv_tokens=unique_tokens,
                    hidden_size=self.config.hidden_size,
                    repetitions=self.config.operator_repetitions,
                )
            )
        return tuple(plans)

    def _schedule_generation_phases(
        self,
        timeline: EventTimeline,
        plans: tuple[GenerationPlan, ...],
        *,
        event_prefix: str,
        ready_time_s: float,
        t1_work_fraction: float,
        speculative_work_fraction: float,
        prior_speculative_event: Event | None,
    ) -> GenerationExecution:
        executions: list[GenerationExecution] = []
        phase_ready = ready_time_s
        for phase_index, plan in enumerate(plans):
            execution = self._schedule_generation(
                timeline,
                plan,
                event_prefix=f"{event_prefix}:token-{phase_index:04d}",
                ready_time_s=phase_ready,
                t1_work_fraction=t1_work_fraction,
                speculative_work_fraction=speculative_work_fraction,
                prior_speculative_event=(
                    prior_speculative_event if phase_index == 0 else None
                ),
            )
            executions.append(execution)
            phase_ready = execution.completion_s
        gpu_events = tuple(
            execution.gpu_event
            for execution in executions
            if execution.gpu_event is not None
        )
        pim_events = tuple(
            execution.pim_event
            for execution in executions
            if execution.pim_event is not None
        )
        return GenerationExecution(
            plan=plans[0],
            ready_time_s=ready_time_s,
            gpu_event=(gpu_events[-1] if gpu_events else None),
            pim_event=(pim_events[-1] if pim_events else None),
            prior_speculative_event=prior_speculative_event,
            t1_work_fraction=t1_work_fraction,
            speculative_work_fraction=speculative_work_fraction,
            completion_s=executions[-1].completion_s,
            phase_plans=plans,
            phase_gpu_events=gpu_events,
            phase_pim_events=pim_events,
        )

    def _allocate_candidate_extensions(
        self,
        step: StepTraceV2,
        kv_index: V2KvIndex,
    ) -> dict[str, str]:
        owned_blocks = {
            block.owner_candidate_id: block
            for block in kv_index.blocks.values()
            if block.owner_candidate_id is not None
        }
        terminals: dict[str, str] = {}
        for candidate in step.candidates:
            block = owned_blocks.get(candidate.candidate_id)
            if block is not None:
                self._write_back_kv(
                    block_id=block.block_id,
                    owner_id=candidate.candidate_id,
                    kind=MemoryKind.UNIQUE_KV,
                    token_count=block.token_count,
                )
            terminals[candidate.candidate_id] = candidate.terminal_kv_block_id
        return terminals

    def _map_candidate_tokens_v2(
        self,
        request_id: str,
        step: StepTraceV2,
        generation: GenerationExecution,
        terminal_blocks: dict[str, str],
        timeline: EventTimeline,
    ) -> tuple[tuple[AddressLookupExecution, ...], tuple[TokenBatch, ...], float]:
        ordered_indices = sorted(
            ready_index
            for candidate in step.candidates
            for ready_index in candidate.token_ready_indices
        )
        ordinal = {
            ready_index: index + 1 for index, ready_index in enumerate(ordered_indices)
        }
        generation_span_s = generation.completion_s - generation.ready_time_s
        lookups: list[AddressLookupExecution] = []
        batches: list[TokenBatch] = []
        for candidate in step.candidates:
            lookup = self.address_cache.resolve(
                self.allocator,
                terminal_blocks[candidate.candidate_id],
            )
            first_ready = generation.ready_time_s + generation_span_s * (
                ordinal[candidate.token_ready_indices[0]] / len(ordered_indices)
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
            for token_index, ready_index in enumerate(candidate.token_ready_indices):
                token_ready = generation.ready_time_s + generation_span_s * (
                    ordinal[ready_index] / len(ordered_indices)
                )
                batches.append(
                    TokenBatch(
                        batch_id=f"{candidate.candidate_id}:token-{token_index:04d}",
                        ready_time_s=max(token_ready, lookup_event.end_s),
                        token_count=1,
                    )
                )
        mapping_complete = max(lookup.event.end_s for lookup in lookups)
        return tuple(lookups), tuple(batches), mapping_complete

    @staticmethod
    def _selection_driver_calls(
        step: StepTraceV2,
    ) -> tuple[VerifierCallTraceV2, ...]:
        by_id = {call.call_id: call for call in step.verifier_calls}
        return tuple(
            by_id[call_id]
            for event in step.selection_events
            for call_id in event.verifier_call_ids
        )

    @staticmethod
    def _small_scalar_calls(
        step: StepTraceV2,
        driver_ids: set[str],
    ) -> tuple[VerifierCallTraceV2, ...]:
        candidate_ids = {candidate.candidate_id for candidate in step.candidates}
        calls_by_stage: dict[str, list[VerifierCallTraceV2]] = {}
        for call in step.verifier_calls:
            if (
                call.kind is VerifierKind.SCALAR_PRM
                and call.call_id not in driver_ids
            ):
                calls_by_stage.setdefault(call.stage, []).append(call)

        complete_stages: list[tuple[VerifierCallTraceV2, ...]] = []
        for calls in calls_by_stage.values():
            covered = tuple(
                candidate_id
                for call in calls
                for candidate_id in call.candidate_ids
            )
            if len(covered) == len(candidate_ids) and set(covered) == candidate_ids:
                complete_stages.append(tuple(calls))
        return complete_stages[-1] if complete_stages else ()

    def _schedule_small_verification(
        self,
        request_id: str,
        step: StepTraceV2,
        batches: tuple[TokenBatch, ...],
        small_calls: tuple[VerifierCallTraceV2, ...],
        generation: GenerationExecution,
        mapping_complete_s: float,
        timeline: EventTimeline,
    ) -> PipelinedVerificationResult:
        generation_complete = max(generation.completion_s, mapping_complete_s)
        if not small_calls:
            return PipelinedVerificationResult(
                chunks=(),
                total_tokens=0,
                early_verified_tokens=0,
                generation_complete_s=generation_complete,
                completion_s=generation_complete,
                serial_completion_s=generation_complete,
            )
        return self.verification_pipeline.schedule(
            batches,
            generation_complete_s=generation_complete,
            timeline=timeline,
            event_prefix=f"{request_id}:step-{step.step_index}:small-prm",
        )

    @staticmethod
    def _scores(
        calls: tuple[VerifierCallTraceV2, ...],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for call in calls:
            scores.update(zip(call.candidate_ids, call.scores))
        return scores

    @staticmethod
    def _final_scalar_scores(
        calls: tuple[VerifierCallTraceV2, ...],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for call in calls:
            if call.kind is VerifierKind.SCALAR_PRM:
                scores.update(zip(call.candidate_ids, call.scores))
        return scores

    def _predict(
        self,
        trace: TtcRequestTraceV2,
        step: StepTraceV2,
        small_scores: dict[str, float],
        history: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    ) -> PredictionDecision | None:
        if trace.beam_size != 1 or set(small_scores) != {
            candidate.candidate_id for candidate in step.candidates
        }:
            return None
        paths: list[CandidateScorePath] = []
        for candidate in step.candidates:
            parent_history = history.get(candidate.parent_candidate_id or "", ((), ()))
            paths.append(
                CandidateScorePath(
                    candidate.candidate_id,
                    current_small_prm_score=small_scores[candidate.candidate_id],
                    historical_small_prm_scores=parent_history[0],
                    historical_large_prm_scores=parent_history[1],
                )
            )
        return predict_candidate(
            tuple(paths),
            history_alignment=self.config.history_alignment,
            aggregation=self.config.score_aggregation,
        )

    def _schedule_verifier_and_selection_events(
        self,
        request_id: str,
        step: StepTraceV2,
        calls: tuple[VerifierCallTraceV2, ...],
        lookups: tuple[AddressLookupExecution, ...],
        verification: PipelinedVerificationResult,
        generation: GenerationExecution,
        mapping_complete_s: float,
        timeline: EventTimeline,
    ) -> tuple[
        tuple[VerifierCallExecution, ...],
        tuple[SelectionEventExecution, ...],
    ]:
        base_dependencies = tuple(
            [lookup.event.event_id for lookup in lookups]
            + [chunk.event.event_id for chunk in verification.chunks]
        )
        calls_by_id = {call.call_id: call for call in calls}
        verifier_executions: list[VerifierCallExecution] = []
        selection_executions: list[SelectionEventExecution] = []
        previous_selection_event: Event | None = None
        ready_time = max(
            generation.completion_s,
            mapping_complete_s,
            verification.completion_s,
        )
        for selection in step.selection_events:
            selection_calls: list[VerifierCallExecution] = []
            previous_call_event: Event | None = None
            for call_id in selection.verifier_call_ids:
                call = calls_by_id[call_id]
                dependencies = base_dependencies
                if previous_selection_event is not None:
                    dependencies = (
                        *dependencies,
                        previous_selection_event.event_id,
                    )
                if previous_call_event is not None:
                    dependencies = (*dependencies, previous_call_event.event_id)
                duration = (
                    call.model_input_tokens + len(call.generated_token_ids)
                ) * self.config.large_prm_time_per_token_s
                call_event = timeline.schedule(
                    f"{request_id}:step-{step.step_index}:verifier:{call.call_id}",
                    Resource.GPU,
                    duration,
                    ready_time_s=ready_time,
                    dependencies=dependencies,
                )
                execution = VerifierCallExecution(call.call_id, call_event)
                verifier_executions.append(execution)
                selection_calls.append(execution)
                previous_call_event = call_event

            selection_event = timeline.schedule(
                f"{request_id}:step-{step.step_index}:select-{selection.event_index}",
                Resource.CONTROLLER,
                0.0,
                ready_time_s=max(call.event.end_s for call in selection_calls),
                dependencies=tuple(
                    call.event.event_id for call in selection_calls
                ),
            )
            freed = self._release_unretained_blocks(selection)
            for block_id in selection.retained_kv_block_ids:
                if self.allocator.has_block(block_id):
                    self.allocator.promote_to_shared(block_id)
            selection_executions.append(
                SelectionEventExecution(
                    event_index=selection.event_index,
                    event=selection_event,
                    selected_candidate_ids=selection.selected_candidate_ids,
                    pruned_candidate_ids=selection.pruned_candidate_ids,
                    freed_kv_block_ids=freed,
                    retained_kv_block_ids=selection.retained_kv_block_ids,
                )
            )
            previous_selection_event = selection_event
        return tuple(verifier_executions), tuple(selection_executions)

    def _release_unretained_blocks(
        self,
        selection: SelectionEventTraceV2,
    ) -> tuple[str, ...]:
        retained = set(selection.retained_kv_block_ids)
        allocated = {
            block.block_id
            for block in self.allocator.blocks
            if block.kind is not MemoryKind.WEIGHT
        }
        to_free = allocated - retained
        freed: list[str] = []
        for candidate_id in selection.pruned_candidate_ids:
            owned = sorted(
                block.block_id
                for block in self.allocator.blocks
                if block.owner_id == candidate_id and block.block_id in to_free
            )
            for block_id in owned:
                self.allocator.prune(block_id)
                self.address_cache.invalidate(block_id)
                to_free.remove(block_id)
                freed.append(block_id)
        for block_id in sorted(to_free):
            self.allocator.prune(block_id)
            self.address_cache.invalidate(block_id)
            freed.append(block_id)
        return tuple(freed)

    def _speculate_next_step_v2(
        self,
        request_id: str,
        trace: TtcRequestTraceV2,
        step: StepTraceV2,
        next_step: StepTraceV2,
        next_plans: tuple[GenerationPlan, ...],
        prediction: PredictionDecision,
        verification: PipelinedVerificationResult,
        large_prm_event: Event,
        timeline: EventTimeline,
    ) -> tuple[Event | None, float, SpeculationResolution]:
        if trace.beam_size != 1:
            raise ConfigurationError(
                "T2A speculation is defined only for beam_size=1"
            )
        pim_only_time = sum(plan.pim_only_time_s for plan in next_plans)
        speculation_ready_s = max(
            verification.completion_s,
            timeline.available_time_s(Resource.PIM),
        )
        available_window_s = max(0.0, large_prm_event.end_s - speculation_ready_s)
        duration_s = min(pim_only_time, available_window_s)
        speculative_fraction = duration_s / pim_only_time if pim_only_time > 0 else 0.0
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
            step.selected_candidate_ids[0],
            rollback_fixed_latency_s=self.config.rollback_fixed_latency_s,
            regeneration_latency_per_token_s=(
                self.config.regeneration_latency_per_token_s
            ),
        )
        return event, speculative_fraction, resolution

    @staticmethod
    def _update_history(
        step: StepTraceV2,
        small_scores: dict[str, float],
        final_scores: dict[str, float],
        history: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    ) -> None:
        if not small_scores or not final_scores:
            return
        by_id = {candidate.candidate_id: candidate for candidate in step.candidates}
        for candidate_id in step.selected_candidate_ids:
            if candidate_id not in small_scores or candidate_id not in final_scores:
                continue
            candidate = by_id[candidate_id]
            parent_history = history.get(candidate.parent_candidate_id or "", ((), ()))
            history[candidate_id] = (
                (*parent_history[0], small_scores[candidate_id]),
                (*parent_history[1], final_scores[candidate_id]),
            )
