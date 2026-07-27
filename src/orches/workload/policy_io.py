"""Strict parsing and canonical JSONL I/O for generation-only policy traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from ..errors import WorkloadTraceError
from .policy_schema import (
    POLICY_TRACE_SCHEMA_VERSION,
    PolicyCandidateTrace,
    PolicyCollectionMode,
    PolicyGenerationCallTrace,
    PolicyRequestTrace,
    PolicySelectionKind,
    PolicySelectionTrace,
    PolicyStepTrace,
    PolicyTraceProvenance,
    WorkloadScope,
)
from .schema import (
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
)
from .schema_v2 import KvBlockTraceV2, TokenTensorTrace


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


def _boolean(mapping: RawMapping, key: str, path: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise WorkloadTraceError(f"{path}.{key} must be a boolean")
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
    return ModelRef(
        name=_string(raw, "name", path),
        revision=_string(raw, "revision", path),
    )


def _sampling(raw_value: Any, path: str) -> SamplingConfig:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"temperature", "top_p", "max_new_tokens"})
    return SamplingConfig(
        temperature=_number(raw["temperature"], f"{path}.temperature"),
        top_p=_number(raw["top_p"], f"{path}.top_p"),
        max_new_tokens=_integer(raw, "max_new_tokens", path),
    )


def _provenance(raw_value: Any, path: str) -> PolicyTraceProvenance:
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
            "inference_engine",
            "collector",
            "selector",
        },
    )
    selector_raw = raw["selector"]
    return PolicyTraceProvenance(
        source_kind=_enum(SourceKind, raw["source_kind"], f"{path}.source_kind"),
        pipeline=_string(raw, "pipeline", path),
        pipeline_revision=_string(raw, "pipeline_revision", path),
        dataset_revision=_string(raw, "dataset_revision", path),
        tokenizer=_model_ref(raw["tokenizer"], f"{path}.tokenizer"),
        policy_model=_model_ref(raw["policy_model"], f"{path}.policy_model"),
        inference_engine=_model_ref(
            raw["inference_engine"], f"{path}.inference_engine"
        ),
        collector=_model_ref(raw["collector"], f"{path}.collector"),
        selector=(
            None
            if selector_raw is None
            else _model_ref(selector_raw, f"{path}.selector")
        ),
    )


def _token_tensor(raw_value: Any, path: str) -> TokenTensorTrace:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"token_ids", "attention_mask", "model_token_count"})
    return TokenTensorTrace(
        token_ids=_integer_array(raw["token_ids"], f"{path}.token_ids"),
        attention_mask=_integer_array(
            raw["attention_mask"], f"{path}.attention_mask"
        ),
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


def _candidate(raw_value: Any, path: str) -> PolicyCandidateTrace:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "candidate_id",
            "generated_token_ids",
            "token_ready_indices",
            "materialized_output_tokens",
            "terminal_kv_block_id",
            "finish_reason",
        },
    )
    return PolicyCandidateTrace(
        candidate_id=_string(raw, "candidate_id", path),
        generated_token_ids=_integer_array(
            raw["generated_token_ids"], f"{path}.generated_token_ids"
        ),
        token_ready_indices=_integer_array(
            raw["token_ready_indices"], f"{path}.token_ready_indices"
        ),
        materialized_output_tokens=_integer(
            raw, "materialized_output_tokens", path
        ),
        terminal_kv_block_id=_string(raw, "terminal_kv_block_id", path),
        finish_reason=_string(raw, "finish_reason", path),
    )


def _generation_call(raw_value: Any, path: str) -> PolicyGenerationCallTrace:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "call_id",
            "parent_candidate_id",
            "input_tokens",
            "reused_kv_tokens",
            "rng_seed",
            "rng_stream_id",
            "candidates",
        },
    )
    candidates = _array(raw["candidates"], f"{path}.candidates")
    return PolicyGenerationCallTrace(
        call_id=_string(raw, "call_id", path),
        parent_candidate_id=_optional_string(raw, "parent_candidate_id", path),
        input_tokens=_token_tensor(raw["input_tokens"], f"{path}.input_tokens"),
        reused_kv_tokens=_integer(raw, "reused_kv_tokens", path),
        rng_seed=_integer(raw, "rng_seed", path),
        rng_stream_id=_string(raw, "rng_stream_id", path),
        candidates=tuple(
            _candidate(candidate, f"{path}.candidates[{index}]")
            for index, candidate in enumerate(candidates)
        ),
    )


def _selection(raw_value: Any, path: str) -> PolicySelectionTrace:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "decision_id",
            "kind",
            "selected_candidate_ids",
            "pruned_candidate_ids",
            "decision_artifact_sha256",
        },
    )
    return PolicySelectionTrace(
        decision_id=_string(raw, "decision_id", path),
        kind=_enum(PolicySelectionKind, raw["kind"], f"{path}.kind"),
        selected_candidate_ids=_string_array(
            raw["selected_candidate_ids"], f"{path}.selected_candidate_ids"
        ),
        pruned_candidate_ids=_string_array(
            raw["pruned_candidate_ids"], f"{path}.pruned_candidate_ids"
        ),
        decision_artifact_sha256=_optional_string(
            raw, "decision_artifact_sha256", path
        ),
    )


def _step(raw_value: Any, path: str) -> PolicyStepTrace:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"step_index", "generation_calls", "selection"})
    calls = _array(raw["generation_calls"], f"{path}.generation_calls")
    selection_raw = raw["selection"]
    return PolicyStepTrace(
        step_index=_integer(raw, "step_index", path),
        generation_calls=tuple(
            _generation_call(call, f"{path}.generation_calls[{index}]")
            for index, call in enumerate(calls)
        ),
        selection=(
            None
            if selection_raw is None
            else _selection(selection_raw, f"{path}.selection")
        ),
    )


def parse_policy_request(
    raw_value: Any, path: str = "request"
) -> PolicyRequestTrace:
    """Parse one strict generation-only request and validate all references."""

    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "policy_trace_schema_version",
            "workload_scope",
            "paper_eligible",
            "collection_mode",
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
            "dtype",
            "sampling",
            "provenance",
            "root_input",
            "kv_blocks",
            "steps",
        },
    )
    version = _integer(raw, "policy_trace_schema_version", path)
    if version != POLICY_TRACE_SCHEMA_VERSION:
        raise WorkloadTraceError(
            f"unsupported policy trace schema version {version!r}"
        )
    bucket_raw = raw["question_length_bucket"]
    kv_blocks = _array(raw["kv_blocks"], f"{path}.kv_blocks")
    steps = _array(raw["steps"], f"{path}.steps")
    return PolicyRequestTrace(
        policy_trace_schema_version=version,
        workload_scope=_enum(
            WorkloadScope, raw["workload_scope"], f"{path}.workload_scope"
        ),
        paper_eligible=_boolean(raw, "paper_eligible", path),
        collection_mode=_enum(
            PolicyCollectionMode,
            raw["collection_mode"],
            f"{path}.collection_mode",
        ),
        request_id=_string(raw, "request_id", path),
        dataset=_string(raw, "dataset", path),
        dataset_id=_string(raw, "dataset_id", path),
        modality=_enum(Modality, raw["modality"], f"{path}.modality"),
        difficulty=_string(raw, "difficulty", path),
        question_length_bucket=(
            None
            if bucket_raw is None
            else _enum(
                QuestionLengthBucket,
                bucket_raw,
                f"{path}.question_length_bucket",
            )
        ),
        image_tokens=_integer(raw, "image_tokens", path),
        search_width=_integer(raw, "search_width", path),
        beam_size=_integer(raw, "beam_size", path),
        seed=_integer(raw, "seed", path),
        dtype=_string(raw, "dtype", path),
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


def _validate_collection(traces: list[PolicyRequestTrace], source: str) -> None:
    if not traces:
        raise WorkloadTraceError(f"{source} is empty")
    request_ids = [trace.request_id for trace in traces]
    if len(set(request_ids)) != len(request_ids):
        raise WorkloadTraceError(f"{source} contains duplicate request IDs")
    modes = {trace.collection_mode for trace in traces}
    if len(modes) != 1:
        raise WorkloadTraceError(f"{source} mixes policy collection modes")
    provenances = {trace.provenance for trace in traces}
    if len(provenances) != 1:
        raise WorkloadTraceError(f"{source} mixes policy trace provenance")


def write_policy_jsonl(
    path: str | Path, traces: list[PolicyRequestTrace]
) -> None:
    """Atomically write canonical generation-only JSONL in request order."""

    _validate_collection(traces, "policy trace collection")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for trace in traces:
                output.write(
                    json.dumps(
                        trace.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                )
                output.write("\n")
        temporary.replace(destination)
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot write policy trace file {destination}: {error}"
        ) from error


def read_policy_jsonl(path: str | Path) -> list[PolicyRequestTrace]:
    """Read and validate a complete generation-only JSONL collection."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot read policy trace file {source}: {error}"
        ) from error
    if not lines:
        raise WorkloadTraceError(f"policy trace file {source} is empty")

    traces: list[PolicyRequestTrace] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise WorkloadTraceError(
                f"policy trace file {source} has blank line {line_number}"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkloadTraceError(
                f"invalid JSON in {source} line {line_number}: {error.msg}"
            ) from error
        traces.append(
            parse_policy_request(raw, path=f"request[line={line_number}]")
        )

    _validate_collection(traces, f"policy trace file {source}")
    return traces


def policy_trace_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact policy trace bytes."""

    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot hash policy trace file {source}: {error}"
        ) from error
