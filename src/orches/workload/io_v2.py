"""Strict object parser for the paper-facing schema-v2 trace contract."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

from ..errors import WorkloadTraceError
from .schema import (
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
)
from .schema_v2 import (
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


RawMapping: TypeAlias = Mapping[str, Any]


def _mapping(value: Any, path: str) -> RawMapping:
    if not isinstance(value, Mapping):
        raise WorkloadTraceError(f"{path} must be an object")
    return value


def _keys(mapping: RawMapping, path: str, required: set[str]) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise WorkloadTraceError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise WorkloadTraceError(f"{path} has unknown fields: {', '.join(unknown)}")


def _string(mapping: RawMapping, key: str, path: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise WorkloadTraceError(f"{path}.{key} must be a string")
    return value


def _optional_string(mapping: RawMapping, key: str, path: str) -> str | None:
    value = mapping[key]
    if value is not None and not isinstance(value, str):
        raise WorkloadTraceError(f"{path}.{key} must be a string or null")
    return value


def _integer(mapping: RawMapping, key: str, path: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadTraceError(f"{path}.{key} must be an integer")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadTraceError(f"{path} must be a number")
    return float(value)


def _enum(enum_type, raw: Any, path: str):
    if not isinstance(raw, str):
        raise WorkloadTraceError(f"{path} must be a string")
    try:
        return enum_type(raw)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise WorkloadTraceError(f"{path} must be one of: {allowed}") from error


def _array(raw: Any, path: str) -> list[Any]:
    if not isinstance(raw, list):
        raise WorkloadTraceError(f"{path} must be an array")
    return raw


def _integer_array(raw: Any, path: str) -> tuple[int, ...]:
    values = _array(raw, path)
    parsed: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkloadTraceError(f"{path}[{index}] must be an integer")
        parsed.append(value)
    return tuple(parsed)


def _string_array(raw: Any, path: str) -> tuple[str, ...]:
    values = _array(raw, path)
    if any(not isinstance(value, str) for value in values):
        raise WorkloadTraceError(f"{path} must be a string array")
    return tuple(values)


def _model_ref(raw_value: Any, path: str) -> ModelRef:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"name", "revision"})
    return ModelRef(name=_string(raw, "name", path), revision=_string(raw, "revision", path))


def _sampling(raw_value: Any, path: str) -> SamplingConfig:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"temperature", "top_p", "max_new_tokens"})
    return SamplingConfig(
        temperature=_number(raw["temperature"], f"{path}.temperature"),
        top_p=_number(raw["top_p"], f"{path}.top_p"),
        max_new_tokens=_integer(raw, "max_new_tokens", path),
    )


def _provenance(raw_value: Any, path: str) -> TraceProvenanceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "source_kind",
            "pipeline",
            "pipeline_revision",
            "dataset_revision",
            "tokenizer",
            "policy_model",
            "verifier_models",
            "inference_engine",
        },
    )
    verifier_models = _array(raw["verifier_models"], f"{path}.verifier_models")
    return TraceProvenanceV2(
        source_kind=_enum(SourceKind, raw["source_kind"], f"{path}.source_kind"),
        pipeline=_string(raw, "pipeline", path),
        pipeline_revision=_string(raw, "pipeline_revision", path),
        dataset_revision=_string(raw, "dataset_revision", path),
        tokenizer=_model_ref(raw["tokenizer"], f"{path}.tokenizer"),
        policy_model=_model_ref(raw["policy_model"], f"{path}.policy_model"),
        verifier_models=tuple(
            _model_ref(model, f"{path}.verifier_models[{index}]")
            for index, model in enumerate(verifier_models)
        ),
        inference_engine=_model_ref(
            raw["inference_engine"], f"{path}.inference_engine"
        ),
    )


def _token_tensor(raw_value: Any, path: str) -> TokenTensorTrace:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"token_ids", "attention_mask", "model_token_count"})
    return TokenTensorTrace(
        token_ids=_integer_array(raw["token_ids"], f"{path}.token_ids"),
        attention_mask=_integer_array(raw["attention_mask"], f"{path}.attention_mask"),
        model_token_count=_integer(raw, "model_token_count", path),
    )


def _kv_block(raw_value: Any, path: str) -> KvBlockTraceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {"block_id", "parent_block_id", "owner_candidate_id", "token_count"},
    )
    return KvBlockTraceV2(
        block_id=_string(raw, "block_id", path),
        parent_block_id=_optional_string(raw, "parent_block_id", path),
        owner_candidate_id=_optional_string(raw, "owner_candidate_id", path),
        token_count=_integer(raw, "token_count", path),
    )


def _candidate(raw_value: Any, path: str) -> CandidateTraceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "candidate_id",
            "parent_candidate_id",
            "input_tokens",
            "generated_token_ids",
            "token_ready_indices",
            "materialized_output_tokens",
            "reused_kv_tokens",
            "terminal_kv_block_id",
            "finish_reason",
        },
    )
    return CandidateTraceV2(
        candidate_id=_string(raw, "candidate_id", path),
        parent_candidate_id=_optional_string(raw, "parent_candidate_id", path),
        input_tokens=_token_tensor(raw["input_tokens"], f"{path}.input_tokens"),
        generated_token_ids=_integer_array(
            raw["generated_token_ids"], f"{path}.generated_token_ids"
        ),
        token_ready_indices=_integer_array(
            raw["token_ready_indices"], f"{path}.token_ready_indices"
        ),
        materialized_output_tokens=_integer(
            raw, "materialized_output_tokens", path
        ),
        reused_kv_tokens=_integer(raw, "reused_kv_tokens", path),
        terminal_kv_block_id=_string(raw, "terminal_kv_block_id", path),
        finish_reason=_string(raw, "finish_reason", path),
    )


def _verifier_call(raw_value: Any, path: str) -> VerifierCallTraceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "call_id",
            "kind",
            "stage",
            "candidate_ids",
            "input_tensors",
            "generated_token_ids",
            "scores",
            "winner_candidate_id",
        },
    )
    scores_raw = _array(raw["scores"], f"{path}.scores")
    input_tensors_raw = _array(
        raw["input_tensors"], f"{path}.input_tensors"
    )
    return VerifierCallTraceV2(
        call_id=_string(raw, "call_id", path),
        kind=_enum(VerifierKind, raw["kind"], f"{path}.kind"),
        stage=_string(raw, "stage", path),
        candidate_ids=_string_array(raw["candidate_ids"], f"{path}.candidate_ids"),
        input_tensors=tuple(
            _token_tensor(tensor, f"{path}.input_tensors[{index}]")
            for index, tensor in enumerate(input_tensors_raw)
        ),
        generated_token_ids=_integer_array(
            raw["generated_token_ids"], f"{path}.generated_token_ids"
        ),
        scores=tuple(
            _number(score, f"{path}.scores[{index}]")
            for index, score in enumerate(scores_raw)
        ),
        winner_candidate_id=_optional_string(raw, "winner_candidate_id", path),
    )


def _selection_event(raw_value: Any, path: str) -> SelectionEventTraceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "event_index",
            "kind",
            "verifier_call_ids",
            "considered_candidate_ids",
            "selected_candidate_ids",
            "pruned_candidate_ids",
            "live_candidate_ids_after",
            "retained_kv_block_ids",
        },
    )
    return SelectionEventTraceV2(
        event_index=_integer(raw, "event_index", path),
        kind=_enum(SelectionKind, raw["kind"], f"{path}.kind"),
        verifier_call_ids=_string_array(
            raw["verifier_call_ids"], f"{path}.verifier_call_ids"
        ),
        considered_candidate_ids=_string_array(
            raw["considered_candidate_ids"], f"{path}.considered_candidate_ids"
        ),
        selected_candidate_ids=_string_array(
            raw["selected_candidate_ids"], f"{path}.selected_candidate_ids"
        ),
        pruned_candidate_ids=_string_array(
            raw["pruned_candidate_ids"], f"{path}.pruned_candidate_ids"
        ),
        live_candidate_ids_after=_string_array(
            raw["live_candidate_ids_after"], f"{path}.live_candidate_ids_after"
        ),
        retained_kv_block_ids=_string_array(
            raw["retained_kv_block_ids"], f"{path}.retained_kv_block_ids"
        ),
    )


def _step(raw_value: Any, path: str) -> StepTraceV2:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "step_index",
            "candidates",
            "verifier_calls",
            "selection_events",
            "selected_candidate_ids",
            "pruned_candidate_ids",
        },
    )
    candidates = _array(raw["candidates"], f"{path}.candidates")
    verifier_calls = _array(raw["verifier_calls"], f"{path}.verifier_calls")
    selection_events = _array(raw["selection_events"], f"{path}.selection_events")
    return StepTraceV2(
        step_index=_integer(raw, "step_index", path),
        candidates=tuple(
            _candidate(candidate, f"{path}.candidates[{index}]")
            for index, candidate in enumerate(candidates)
        ),
        verifier_calls=tuple(
            _verifier_call(call, f"{path}.verifier_calls[{index}]")
            for index, call in enumerate(verifier_calls)
        ),
        selection_events=tuple(
            _selection_event(event, f"{path}.selection_events[{index}]")
            for index, event in enumerate(selection_events)
        ),
        selected_candidate_ids=_string_array(
            raw["selected_candidate_ids"], f"{path}.selected_candidate_ids"
        ),
        pruned_candidate_ids=_string_array(
            raw["pruned_candidate_ids"], f"{path}.pruned_candidate_ids"
        ),
    )


def parse_request_v2(raw_value: Any, path: str = "request") -> TtcRequestTraceV2:
    """Parse one strict schema-v2 request and validate all cross references."""

    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "trace_schema_version",
            "request_id",
            "dataset",
            "dataset_id",
            "modality",
            "difficulty",
            "question_length_bucket",
            "image_tokens",
            "search_width",
            "beam_size",
            "seed",
            "sampling",
            "provenance",
            "root_input",
            "kv_blocks",
            "steps",
        },
    )
    bucket_raw = raw["question_length_bucket"]
    bucket = (
        None
        if bucket_raw is None
        else _enum(QuestionLengthBucket, bucket_raw, f"{path}.question_length_bucket")
    )
    kv_blocks = _array(raw["kv_blocks"], f"{path}.kv_blocks")
    steps = _array(raw["steps"], f"{path}.steps")
    return TtcRequestTraceV2(
        trace_schema_version=_integer(raw, "trace_schema_version", path),
        request_id=_string(raw, "request_id", path),
        dataset=_string(raw, "dataset", path),
        dataset_id=_string(raw, "dataset_id", path),
        modality=_enum(Modality, raw["modality"], f"{path}.modality"),
        difficulty=_string(raw, "difficulty", path),
        question_length_bucket=bucket,
        image_tokens=_integer(raw, "image_tokens", path),
        search_width=_integer(raw, "search_width", path),
        beam_size=_integer(raw, "beam_size", path),
        seed=_integer(raw, "seed", path),
        sampling=_sampling(raw["sampling"], f"{path}.sampling"),
        provenance=_provenance(raw["provenance"], f"{path}.provenance"),
        root_input=_token_tensor(raw["root_input"], f"{path}.root_input"),
        kv_blocks=tuple(
            _kv_block(block, f"{path}.kv_blocks[{index}]")
            for index, block in enumerate(kv_blocks)
        ),
        steps=tuple(
            _step(step, f"{path}.steps[{index}]")
            for index, step in enumerate(steps)
        ),
    )
