from __future__ import annotations

from copy import deepcopy

import pytest

from orches.errors import WorkloadTraceError
from orches.workload import (
    Modality,
    QuestionLengthBucket,
    SyntheticTraceConfig,
    generate_synthetic_traces,
    read_jsonl,
    trace_sha256,
    write_jsonl,
)
from orches.workload.io import parse_request


def _trace_dict() -> dict[str, object]:
    return generate_synthetic_traces(
        SyntheticTraceConfig(
            request_count=1,
            step_count=3,
            search_width=4,
            seed=11,
        )
    )[0].to_dict()


def test_jsonl_round_trip_is_byte_stable(tmp_path) -> None:
    traces = generate_synthetic_traces(
        SyntheticTraceConfig(request_count=3, step_count=2, search_width=2, seed=9)
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_jsonl(first, traces)
    loaded = read_jsonl(first)
    write_jsonl(second, loaded)

    assert loaded == traces
    assert first.read_bytes() == second.read_bytes()
    assert trace_sha256(first) == trace_sha256(second)


def test_same_seed_reproduces_identical_control_flow() -> None:
    config = SyntheticTraceConfig(request_count=2, seed=91)

    assert generate_synthetic_traces(config) == generate_synthetic_traces(config)


def test_different_seed_changes_trace() -> None:
    first = generate_synthetic_traces(SyntheticTraceConfig(seed=1))
    second = generate_synthetic_traces(SyntheticTraceConfig(seed=2))

    assert first != second


def test_selected_candidate_must_exist() -> None:
    raw = _trace_dict()
    raw["steps"][0]["selected_candidate_id"] = "missing"

    with pytest.raises(WorkloadTraceError, match="does not exist"):
        parse_request(raw)


def test_next_step_parent_must_be_previous_selection() -> None:
    raw = _trace_dict()
    raw["steps"][1]["candidates"][0]["parent_candidate_id"] = "wrong-parent"

    with pytest.raises(WorkloadTraceError, match="has parent"):
        parse_request(raw)


def test_next_step_shared_kv_must_follow_selected_path() -> None:
    raw = _trace_dict()
    raw["steps"][1]["shared_kv_tokens"] += 1

    with pytest.raises(WorkloadTraceError, match="shared_kv_tokens"):
        parse_request(raw)


def test_token_timeline_must_match_generated_token_count() -> None:
    raw = _trace_dict()
    raw["steps"][0]["candidates"][0]["token_timestamps_us"].pop()

    with pytest.raises(WorkloadTraceError, match="length must equal"):
        parse_request(raw)


def test_pruned_candidates_must_be_complete() -> None:
    raw = _trace_dict()
    raw["steps"][0]["pruned_candidate_ids"].pop()

    with pytest.raises(WorkloadTraceError, match="every non-selected"):
        parse_request(raw)


def test_unknown_trace_field_is_rejected() -> None:
    raw = _trace_dict()
    raw["prompt_text"] = "must not be stored"

    with pytest.raises(WorkloadTraceError, match="unknown fields"):
        parse_request(raw)


def test_schema_contains_no_prompt_or_answer_body() -> None:
    raw = _trace_dict()

    assert "prompt" not in raw
    assert "answer" not in raw
    assert "completion" not in raw


def test_parse_does_not_mutate_input() -> None:
    raw = _trace_dict()
    original = deepcopy(raw)

    parse_request(raw)

    assert raw == original


def test_vision_trace_requires_and_preserves_visual_metadata() -> None:
    traces = generate_synthetic_traces(
        SyntheticTraceConfig(
            modality=Modality.VISION,
            image_tokens=576,
            question_length_bucket=QuestionLengthBucket.MEDIUM,
        )
    )

    assert traces[0].image_tokens == 576
    assert traces[0].question_length_bucket is QuestionLengthBucket.MEDIUM
    assert traces[0].steps[0].shared_kv_tokens == traces[0].prompt_tokens + 576
