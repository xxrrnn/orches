"""Explicit synthetic-only migration from the development v1 trace schema."""

from __future__ import annotations

from ..errors import WorkloadTraceError
from .schema import ModelRef, SourceKind, TtcRequestTrace
from .schema_v2 import (
    TRACE_SCHEMA_VERSION_V2,
    CandidateTraceV2,
    KvBlockTraceV2,
    SelectionEventTraceV2,
    SelectionKind,
    StepTraceV2,
    TokenTensorTrace,
    TraceProvenanceV2,
    TtcRequestTraceV2,
    VerifierCallTraceV2,
    VerifierKind,
)


MIGRATION_REVISION = "orches-synthetic-v1-to-v2"


def _token_tensor(token_ids: tuple[int, ...], image_tokens: int) -> TokenTensorTrace:
    return TokenTensorTrace(
        token_ids=token_ids,
        attention_mask=(1,) * len(token_ids),
        model_token_count=len(token_ids) + image_tokens,
    )


def _lineage(blocks: dict[str, KvBlockTraceV2], block_id: str) -> tuple[str, ...]:
    lineage: list[str] = []
    current: str | None = block_id
    while current is not None:
        lineage.append(current)
        current = blocks[current].parent_block_id
    lineage.reverse()
    return tuple(lineage)


def migrate_synthetic_v1_to_v2(trace: TtcRequestTrace) -> TtcRequestTraceV2:
    """Create deterministic placeholder IDs without claiming collected evidence."""

    if trace.provenance.source_kind is not SourceKind.SYNTHETIC:
        raise WorkloadTraceError("only synthetic schema-v1 traces may be migrated")

    root_token_ids = tuple(range(trace.prompt_tokens))
    root_input = _token_tensor(root_token_ids, trace.image_tokens)
    root_block = KvBlockTraceV2(
        block_id=f"{trace.request_id}:root-kv",
        parent_block_id=None,
        owner_candidate_id=None,
        token_count=root_input.model_token_count,
    )
    blocks: list[KvBlockTraceV2] = [root_block]
    block_by_id = {root_block.block_id: root_block}
    migrated_steps: list[StepTraceV2] = []
    parent_output_ids = root_token_ids
    parent_block_id = root_block.block_id
    parent_kv_tokens = root_block.token_count
    next_synthetic_token_id = 1_000_000
    next_ready_index = 0

    for step in trace.steps:
        ready_events = sorted(
            (
                timestamp,
                candidate.candidate_id,
                token_index,
            )
            for candidate in step.candidates
            for token_index, timestamp in enumerate(candidate.token_timestamps_us)
        )
        ready_by_candidate: dict[str, list[int]] = {
            candidate.candidate_id: [] for candidate in step.candidates
        }
        for offset, (_, candidate_id, _) in enumerate(ready_events):
            ready_by_candidate[candidate_id].append(next_ready_index + offset)
        next_ready_index += len(ready_events)

        candidates: list[CandidateTraceV2] = []
        for candidate in step.candidates:
            generated_ids = tuple(
                range(
                    next_synthetic_token_id,
                    next_synthetic_token_id + candidate.generated_tokens,
                )
            )
            next_synthetic_token_id += candidate.generated_tokens
            block_id = f"{candidate.candidate_id}:kv"
            migrated = CandidateTraceV2(
                candidate_id=candidate.candidate_id,
                parent_candidate_id=candidate.parent_candidate_id,
                input_tokens=_token_tensor(parent_output_ids, trace.image_tokens),
                generated_token_ids=generated_ids,
                token_ready_indices=tuple(ready_by_candidate[candidate.candidate_id]),
                materialized_output_tokens=candidate.unique_kv_tokens,
                reused_kv_tokens=parent_kv_tokens,
                terminal_kv_block_id=block_id,
                finish_reason="synthetic-v1-migration",
            )
            candidates.append(migrated)
            block = KvBlockTraceV2(
                block_id=block_id,
                parent_block_id=parent_block_id,
                owner_candidate_id=candidate.candidate_id,
                token_count=candidate.unique_kv_tokens,
            )
            blocks.append(block)
            block_by_id[block_id] = block

        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        small_call_id = f"{trace.request_id}:s{step.step_index}:small"
        final_call_id = f"{trace.request_id}:s{step.step_index}:final"
        selected = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == step.selected_candidate_id
        )
        retained_blocks = _lineage(block_by_id, selected.terminal_kv_block_id)
        migrated_steps.append(
            StepTraceV2(
                step_index=step.step_index,
                candidates=tuple(candidates),
                verifier_calls=(
                    VerifierCallTraceV2(
                        call_id=small_call_id,
                        kind=VerifierKind.SCALAR_PRM,
                        stage="synthetic-small",
                        candidate_ids=candidate_ids,
                        input_tensors=tuple(
                            _token_tensor(
                                candidate.full_output_token_ids,
                                trace.image_tokens,
                            )
                            for candidate in candidates
                        ),
                        generated_token_ids=(),
                        scores=tuple(
                            candidate.small_prm_score for candidate in step.candidates
                        ),
                        winner_candidate_id=None,
                    ),
                    VerifierCallTraceV2(
                        call_id=final_call_id,
                        kind=VerifierKind.SCALAR_PRM,
                        stage="synthetic-final",
                        candidate_ids=candidate_ids,
                        input_tensors=tuple(
                            _token_tensor(
                                candidate.full_output_token_ids,
                                trace.image_tokens,
                            )
                            for candidate in candidates
                        ),
                        generated_token_ids=(),
                        scores=tuple(
                            candidate.large_prm_score for candidate in step.candidates
                        ),
                        winner_candidate_id=None,
                    ),
                ),
                selection_events=(
                    SelectionEventTraceV2(
                        event_index=0,
                        kind=SelectionKind.TOP_K,
                        verifier_call_ids=(final_call_id,),
                        considered_candidate_ids=candidate_ids,
                        selected_candidate_ids=(selected.candidate_id,),
                        pruned_candidate_ids=step.pruned_candidate_ids,
                        live_candidate_ids_after=(selected.candidate_id,),
                        retained_kv_block_ids=retained_blocks,
                    ),
                ),
                selected_candidate_ids=(selected.candidate_id,),
                pruned_candidate_ids=step.pruned_candidate_ids,
            )
        )
        parent_output_ids = selected.full_output_token_ids
        parent_block_id = selected.terminal_kv_block_id
        parent_kv_tokens += selected.materialized_output_tokens

    return TtcRequestTraceV2(
        trace_schema_version=TRACE_SCHEMA_VERSION_V2,
        request_id=trace.request_id,
        dataset=trace.dataset,
        dataset_id=trace.dataset_id,
        modality=trace.modality,
        difficulty=trace.difficulty,
        question_length_bucket=trace.question_length_bucket,
        image_tokens=trace.image_tokens,
        search_width=trace.search_width,
        beam_size=1,
        seed=trace.seed,
        sampling=trace.sampling,
        provenance=TraceProvenanceV2(
            source_kind=SourceKind.SYNTHETIC,
            pipeline=trace.provenance.pipeline,
            pipeline_revision=MIGRATION_REVISION,
            dataset_revision=trace.provenance.dataset_revision,
            tokenizer=trace.provenance.tokenizer,
            policy_model=trace.provenance.policy_model,
            verifier_models=(
                trace.provenance.small_prm_model,
                trace.provenance.large_prm_model,
            ),
            inference_engine=ModelRef("orches-synthetic-migration", MIGRATION_REVISION),
        ),
        root_input=root_input,
        kv_blocks=tuple(blocks),
        steps=tuple(migrated_steps),
    )
