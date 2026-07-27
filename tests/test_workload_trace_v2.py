from __future__ import annotations

import json

import pytest

from orches.cli import main
from orches.errors import WorkloadTraceError
from orches.workload import (
    CandidateTraceV2,
    KvBlockTraceV2,
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SelectionEventTraceV2,
    SelectionKind,
    SourceKind,
    StepTraceV2,
    SyntheticTraceConfig,
    TokenTensorTrace,
    TraceProvenanceV2,
    TtcRequestTraceV2,
    VerifierCallTraceV2,
    VerifierKind,
    generate_synthetic_traces,
    migrate_synthetic_v1_to_v2,
    read_jsonl,
    write_jsonl,
)
from orches.workload.io import parse_request


def _tokens(*token_ids: int) -> TokenTensorTrace:
    return TokenTensorTrace(
        token_ids=token_ids,
        attention_mask=(1,) * len(token_ids),
        model_token_count=len(token_ids),
    )


def _provenance() -> TraceProvenanceV2:
    return TraceProvenanceV2(
        source_kind=SourceKind.COLLECTED,
        pipeline="test.compute-optimal-tts",
        pipeline_revision="pipeline-commit",
        dataset_revision="dataset-commit",
        tokenizer=ModelRef("test-tokenizer", "tokenizer-commit"),
        policy_model=ModelRef("test-policy", "policy-commit"),
        verifier_models=(ModelRef("test-prm", "prm-commit"),),
        inference_engine=ModelRef("test-engine", "engine-commit"),
    )


def _candidate(
    candidate_id: str,
    parent_candidate_id: str | None,
    input_ids: tuple[int, ...],
    generated_ids: tuple[int, ...],
    ready_start: int,
    materialized_output_tokens: int,
    reused_kv_tokens: int,
    model_input_tokens: int | None = None,
) -> CandidateTraceV2:
    return CandidateTraceV2(
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        input_tokens=TokenTensorTrace(
            token_ids=input_ids,
            attention_mask=(1,) * len(input_ids),
            model_token_count=(
                len(input_ids) if model_input_tokens is None else model_input_tokens
            ),
        ),
        generated_token_ids=generated_ids,
        token_ready_indices=tuple(
            range(ready_start, ready_start + len(generated_ids))
        ),
        materialized_output_tokens=materialized_output_tokens,
        reused_kv_tokens=reused_kv_tokens,
        terminal_kv_block_id=f"{candidate_id}-kv",
        finish_reason="stop",
    )


def _scalar_call(
    call_id: str,
    stage: str,
    candidate_ids: tuple[str, ...],
    scores: tuple[float, ...],
) -> VerifierCallTraceV2:
    return VerifierCallTraceV2(
        call_id=call_id,
        kind=VerifierKind.SCALAR_PRM,
        stage=stage,
        candidate_ids=candidate_ids,
        input_tokens=_tokens(500, 501, 502),
        generated_token_ids=(),
        scores=scores,
        winner_candidate_id=None,
    )


def _two_step_beam_trace() -> TtcRequestTraceV2:
    root_input = _tokens(10, 11, 12)
    blocks = [KvBlockTraceV2("root-kv", None, None, 3)]

    step0_candidates: list[CandidateTraceV2] = []
    materialized = (1, 2, 1, 2)
    generated = ((100, 101), (110, 111), (120, 121), (130, 131))
    for index, generated_ids in enumerate(generated):
        candidate = _candidate(
            f"c{index}",
            None,
            root_input.valid_token_ids,
            generated_ids,
            index * 2,
            materialized[index],
            3,
        )
        step0_candidates.append(candidate)
        blocks.append(
            KvBlockTraceV2(
                candidate.terminal_kv_block_id,
                "root-kv",
                candidate.candidate_id,
                materialized[index],
            )
        )

    step0_ids = tuple(candidate.candidate_id for candidate in step0_candidates)
    step0 = StepTraceV2(
        step_index=0,
        candidates=tuple(step0_candidates),
        verifier_calls=(
            _scalar_call("s0-small", "layer_10", step0_ids, (0.4, 0.8, 0.2, 0.9)),
            _scalar_call("s0-final", "final", step0_ids, (0.1, 0.8, 0.3, 0.9)),
        ),
        selection_events=(
            SelectionEventTraceV2(
                event_index=0,
                kind=SelectionKind.TOP_K,
                verifier_call_ids=("s0-final",),
                considered_candidate_ids=step0_ids,
                selected_candidate_ids=("c3", "c1"),
                pruned_candidate_ids=("c0", "c2"),
                live_candidate_ids_after=("c3", "c1"),
                retained_kv_block_ids=("root-kv", "c3-kv", "c1-kv"),
            ),
        ),
        selected_candidate_ids=("c3", "c1"),
        pruned_candidate_ids=("c0", "c2"),
    )

    parent_candidates = (step0_candidates[3], step0_candidates[1])
    step1_candidates: list[CandidateTraceV2] = []
    ready_index = 8
    for parent in parent_candidates:
        for child_index in range(4):
            candidate_id = f"d{len(step1_candidates)}"
            input_ids = parent.full_output_token_ids + (700,)
            generated_ids = (800 + child_index, 900 + child_index)
            child = _candidate(
                candidate_id,
                parent.candidate_id,
                input_ids,
                generated_ids,
                ready_index,
                materialized_output_tokens=1,
                reused_kv_tokens=5,
            )
            ready_index += len(generated_ids)
            step1_candidates.append(child)
            blocks.append(
                KvBlockTraceV2(
                    child.terminal_kv_block_id,
                    parent.terminal_kv_block_id,
                    child.candidate_id,
                    2,
                )
            )

    step1_ids = tuple(candidate.candidate_id for candidate in step1_candidates)
    step1 = StepTraceV2(
        step_index=1,
        candidates=tuple(step1_candidates),
        verifier_calls=(
            _scalar_call(
                "s1-final",
                "final",
                step1_ids,
                (0.2, 0.95, 0.1, 0.4, 0.3, 0.5, 0.99, 0.6),
            ),
        ),
        selection_events=(
            SelectionEventTraceV2(
                event_index=0,
                kind=SelectionKind.TOP_K,
                verifier_call_ids=("s1-final",),
                considered_candidate_ids=step1_ids,
                selected_candidate_ids=("d6", "d1"),
                pruned_candidate_ids=("d0", "d2", "d3", "d4", "d5", "d7"),
                live_candidate_ids_after=("d6", "d1"),
                retained_kv_block_ids=(
                    "root-kv",
                    "c1-kv",
                    "c3-kv",
                    "d6-kv",
                    "d1-kv",
                ),
            ),
        ),
        selected_candidate_ids=("d6", "d1"),
        pruned_candidate_ids=("d0", "d2", "d3", "d4", "d5", "d7"),
    )

    return TtcRequestTraceV2(
        trace_schema_version=2,
        request_id="beam-request",
        dataset="MATH-500",
        dataset_id="test-item",
        modality=Modality.TEXT,
        difficulty="Level 3",
        question_length_bucket=None,
        image_tokens=0,
        search_width=4,
        beam_size=2,
        seed=0,
        sampling=SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=64),
        provenance=_provenance(),
        root_input=root_input,
        kv_blocks=tuple(blocks),
        steps=(step0, step1),
    )


def _pairwise_trace() -> TtcRequestTraceV2:
    root_input = TokenTensorTrace((20, 21, 22), (1, 1, 1), 579)
    blocks = [KvBlockTraceV2("root-kv", None, None, 579)]
    candidates: list[CandidateTraceV2] = []
    for index in range(4):
        candidate = _candidate(
            f"c{index}",
            None,
            root_input.valid_token_ids,
            (100 + index, 200 + index),
            index * 2,
            materialized_output_tokens=1,
            reused_kv_tokens=579,
            model_input_tokens=579,
        )
        candidates.append(candidate)
        blocks.append(
            KvBlockTraceV2(
                candidate.terminal_kv_block_id,
                "root-kv",
                candidate.candidate_id,
                1,
            )
        )

    def judge(
        call_id: str, first: str, second: str, winner: str
    ) -> VerifierCallTraceV2:
        return VerifierCallTraceV2(
            call_id=call_id,
            kind=VerifierKind.PAIRWISE_JUDGE,
            stage="REASONING",
            candidate_ids=(first, second),
            input_tokens=TokenTensorTrace(
                (600, 601, 602, 603),
                (1, 1, 1, 1),
                580,
            ),
            generated_token_ids=(42,),
            scores=(),
            winner_candidate_id=winner,
        )

    step = StepTraceV2(
        step_index=0,
        candidates=tuple(candidates),
        verifier_calls=(
            judge("judge-0", "c0", "c1", "c1"),
            judge("judge-1", "c2", "c3", "c2"),
            judge("judge-2", "c1", "c2", "c2"),
        ),
        selection_events=(
            SelectionEventTraceV2(
                0,
                SelectionKind.PAIRWISE,
                ("judge-0",),
                ("c0", "c1"),
                ("c1",),
                ("c0",),
                ("c1", "c2", "c3"),
                ("root-kv", "c1-kv", "c2-kv", "c3-kv"),
            ),
            SelectionEventTraceV2(
                1,
                SelectionKind.PAIRWISE,
                ("judge-1",),
                ("c2", "c3"),
                ("c2",),
                ("c3",),
                ("c1", "c2"),
                ("root-kv", "c1-kv", "c2-kv"),
            ),
            SelectionEventTraceV2(
                2,
                SelectionKind.PAIRWISE,
                ("judge-2",),
                ("c1", "c2"),
                ("c2",),
                ("c1",),
                ("c2",),
                ("root-kv", "c2-kv"),
            ),
        ),
        selected_candidate_ids=("c2",),
        pruned_candidate_ids=("c0", "c3", "c1"),
    )
    return TtcRequestTraceV2(
        trace_schema_version=2,
        request_id="pairwise-request",
        dataset="MathVista",
        dataset_id="test-image",
        modality=Modality.VISION,
        difficulty="test",
        question_length_bucket=QuestionLengthBucket.MEDIUM,
        image_tokens=576,
        search_width=4,
        beam_size=1,
        seed=0,
        sampling=SamplingConfig(temperature=0.6, top_p=0.9, max_new_tokens=64),
        provenance=_provenance(),
        root_input=root_input,
        kv_blocks=tuple(blocks),
        steps=(step,),
    )


def _beam_three_trace() -> TtcRequestTraceV2:
    source = _two_step_beam_trace()
    candidates = source.steps[0].candidates
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    step = StepTraceV2(
        step_index=0,
        candidates=candidates,
        verifier_calls=source.steps[0].verifier_calls,
        selection_events=(
            SelectionEventTraceV2(
                event_index=0,
                kind=SelectionKind.TOP_K,
                verifier_call_ids=("s0-final",),
                considered_candidate_ids=candidate_ids,
                selected_candidate_ids=("c3", "c1", "c2"),
                pruned_candidate_ids=("c0",),
                live_candidate_ids_after=("c3", "c1", "c2"),
                retained_kv_block_ids=(
                    "root-kv",
                    "c3-kv",
                    "c1-kv",
                    "c2-kv",
                ),
            ),
        ),
        selected_candidate_ids=("c3", "c1", "c2"),
        pruned_candidate_ids=("c0",),
    )
    return TtcRequestTraceV2(
        trace_schema_version=2,
        request_id="beam-three-request",
        dataset=source.dataset,
        dataset_id=source.dataset_id,
        modality=source.modality,
        difficulty=source.difficulty,
        question_length_bucket=source.question_length_bucket,
        image_tokens=source.image_tokens,
        search_width=source.search_width,
        beam_size=3,
        seed=source.seed,
        sampling=source.sampling,
        provenance=source.provenance,
        root_input=source.root_input,
        kv_blocks=source.kv_blocks[:5],
        steps=(step,),
    )


def test_v2_round_trip_preserves_exact_tokens_and_multiple_beams(tmp_path) -> None:
    trace = _two_step_beam_trace()
    path = tmp_path / "trace-v2.jsonl"

    write_jsonl(path, [trace])
    loaded = read_jsonl(path)

    assert loaded == [trace]
    assert isinstance(loaded[0], TtcRequestTraceV2)
    assert trace.steps[0].selected_candidate_ids == ("c3", "c1")
    assert {candidate.parent_candidate_id for candidate in trace.steps[1].candidates} == {
        "c3",
        "c1",
    }
    assert trace.steps[0].candidates[0].generated_tokens == 2
    assert trace.steps[0].candidates[0].materialized_output_tokens == 1
    assert trace.generated_tokens == 24
    assert trace.selected_path_tokens == 8


def test_v2_cli_summary_reports_schema_and_beam(tmp_path, capsys) -> None:
    path = tmp_path / "trace-v2.jsonl"
    write_jsonl(path, [_two_step_beam_trace()])

    status = main(["validate-trace", str(path), "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert status == 0
    assert summary["schema_versions"] == [2]
    assert summary["beam_sizes"] == [2]
    assert summary["source_kinds"] == ["collected"]


def test_v2_rejects_changed_parent_token_prefix() -> None:
    raw = _two_step_beam_trace().to_dict()
    raw["steps"][1]["candidates"][0]["input_tokens"]["token_ids"][0] = 999

    with pytest.raises(WorkloadTraceError, match="preserve parent token IDs"):
        parse_request(raw)


def test_v2_rejects_incorrect_kv_extension_count() -> None:
    raw = _two_step_beam_trace().to_dict()
    raw["kv_blocks"][1]["token_count"] += 1

    with pytest.raises(WorkloadTraceError, match="terminal KV extension"):
        parse_request(raw)


def test_v2_rejects_retaining_a_pruned_kv_lineage() -> None:
    raw = _two_step_beam_trace().to_dict()
    retained = raw["steps"][0]["selection_events"][0]["retained_kv_block_ids"]
    retained.append("c0-kv")

    with pytest.raises(WorkloadTraceError, match="retained_kv_block_ids"):
        parse_request(raw)


def test_v2_rejects_non_global_token_ready_order() -> None:
    raw = _two_step_beam_trace().to_dict()
    raw["steps"][1]["candidates"][0]["token_ready_indices"][0] = 0

    with pytest.raises(WorkloadTraceError, match="token-ready order"):
        parse_request(raw)


def test_v2_pairwise_tournament_preserves_elimination_order() -> None:
    trace = _pairwise_trace()

    assert trace.steps[0].selected_candidate_ids == ("c2",)
    assert [
        event.pruned_candidate_ids for event in trace.steps[0].selection_events
    ] == [("c0",), ("c3",), ("c1",)]
    assert trace.root_input.model_token_count == 579
    assert trace.image_tokens == 576


def test_v2_pairwise_selection_must_follow_judge_winner() -> None:
    raw = _pairwise_trace().to_dict()
    raw["steps"][0]["verifier_calls"][0]["winner_candidate_id"] = "c0"

    with pytest.raises(WorkloadTraceError, match="recorded judge winner"):
        parse_request(raw)


def test_v2_beam_greater_than_two_retains_actual_three_lineages() -> None:
    trace = _beam_three_trace()
    event = trace.steps[0].selection_events[0]

    assert trace.beam_size == 3
    assert event.live_candidate_ids_after == ("c3", "c1", "c2")
    assert set(event.retained_kv_block_ids) == {
        "root-kv",
        "c3-kv",
        "c1-kv",
        "c2-kv",
    }


def test_v2_preserves_selected_beam_order() -> None:
    raw = _two_step_beam_trace().to_dict()
    raw["steps"][0]["selected_candidate_ids"] = ["c1", "c3"]

    with pytest.raises(WorkloadTraceError, match="selected candidate order"):
        parse_request(raw)


def test_v2_rejects_character_count_as_an_unknown_size_field() -> None:
    raw = _two_step_beam_trace().to_dict()
    raw["steps"][0]["candidates"][0]["output_character_count"] = 123

    with pytest.raises(WorkloadTraceError, match="unknown fields"):
        parse_request(raw)


def test_v1_collected_trace_is_rejected() -> None:
    collected = generate_synthetic_traces(SyntheticTraceConfig())[0].to_dict()
    collected["provenance"]["source_kind"] = "collected"

    with pytest.raises(WorkloadTraceError, match="schema v1 is synthetic-only"):
        parse_request(collected)


def test_v1_synthetic_migration_is_deterministic_and_ineligible() -> None:
    source = generate_synthetic_traces(
        SyntheticTraceConfig(step_count=2, search_width=3, seed=17)
    )[0]

    first = migrate_synthetic_v1_to_v2(source)
    second = migrate_synthetic_v1_to_v2(source)

    assert first == second
    assert first.trace_schema_version == 2
    assert first.provenance.source_kind is SourceKind.SYNTHETIC
    assert first.provenance.pipeline_revision == "orches-synthetic-v1-to-v2"
    assert first.steps[1].candidates[0].parent_candidate_id in (
        first.steps[0].selected_candidate_ids
    )


def test_trace_file_must_not_mix_schema_versions(tmp_path) -> None:
    v1 = generate_synthetic_traces(SyntheticTraceConfig())[0]
    v2 = _two_step_beam_trace()

    with pytest.raises(WorkloadTraceError, match="must not mix schema versions"):
        write_jsonl(tmp_path / "mixed.jsonl", [v1, v2])
