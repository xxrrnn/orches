"""Strict JSONL parsing and deterministic serialization for TTC traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from ..errors import WorkloadTraceError
from .schema import (
    TRACE_SCHEMA_VERSION,
    CandidateTrace,
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
    StepTrace,
    TraceProvenance,
    TtcRequestTrace,
)
from .schema_v2 import TRACE_SCHEMA_VERSION_V2, TtcRequestTraceV2
from .io_v2 import parse_request_v2


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


def _integer(mapping: RawMapping, key: str, path: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadTraceError(f"{path}.{key} must be an integer")
    return value


def _number(mapping: RawMapping, key: str, path: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadTraceError(f"{path}.{key} must be a number")
    return float(value)


def _enum(enum_type, raw: Any, path: str):
    if not isinstance(raw, str):
        raise WorkloadTraceError(f"{path} must be a string")
    try:
        return enum_type(raw)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise WorkloadTraceError(f"{path} must be one of: {allowed}") from error


def _model_ref(raw_value: Any, path: str) -> ModelRef:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"name", "revision"})
    return ModelRef(name=_string(raw, "name", path), revision=_string(raw, "revision", path))


def _sampling(raw_value: Any, path: str) -> SamplingConfig:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"temperature", "top_p", "max_new_tokens"})
    return SamplingConfig(
        temperature=_number(raw, "temperature", path),
        top_p=_number(raw, "top_p", path),
        max_new_tokens=_integer(raw, "max_new_tokens", path),
    )


def _provenance(raw_value: Any, path: str) -> TraceProvenance:
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
            "small_prm_model",
            "large_prm_model",
        },
    )
    return TraceProvenance(
        source_kind=_enum(SourceKind, raw["source_kind"], f"{path}.source_kind"),
        pipeline=_string(raw, "pipeline", path),
        pipeline_revision=_string(raw, "pipeline_revision", path),
        dataset_revision=_string(raw, "dataset_revision", path),
        tokenizer=_model_ref(raw["tokenizer"], f"{path}.tokenizer"),
        policy_model=_model_ref(raw["policy_model"], f"{path}.policy_model"),
        small_prm_model=_model_ref(raw["small_prm_model"], f"{path}.small_prm_model"),
        large_prm_model=_model_ref(raw["large_prm_model"], f"{path}.large_prm_model"),
    )


def _candidate(raw_value: Any, path: str) -> CandidateTrace:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "candidate_id",
            "parent_candidate_id",
            "generated_tokens",
            "token_timestamps_us",
            "small_prm_score",
            "large_prm_score",
            "unique_kv_tokens",
        },
    )
    parent = raw["parent_candidate_id"]
    if parent is not None and not isinstance(parent, str):
        raise WorkloadTraceError(f"{path}.parent_candidate_id must be string or null")
    timestamps_raw = raw["token_timestamps_us"]
    if not isinstance(timestamps_raw, list):
        raise WorkloadTraceError(f"{path}.token_timestamps_us must be an array")
    timestamps: list[float] = []
    for index, value in enumerate(timestamps_raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkloadTraceError(
                f"{path}.token_timestamps_us[{index}] must be a number"
            )
        timestamps.append(float(value))
    return CandidateTrace(
        candidate_id=_string(raw, "candidate_id", path),
        parent_candidate_id=parent,
        generated_tokens=_integer(raw, "generated_tokens", path),
        token_timestamps_us=tuple(timestamps),
        small_prm_score=_number(raw, "small_prm_score", path),
        large_prm_score=_number(raw, "large_prm_score", path),
        unique_kv_tokens=_integer(raw, "unique_kv_tokens", path),
    )


def _step(raw_value: Any, path: str) -> StepTrace:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "step_index",
            "shared_kv_tokens",
            "candidates",
            "selected_candidate_id",
            "pruned_candidate_ids",
        },
    )
    candidates_raw = raw["candidates"]
    if not isinstance(candidates_raw, list):
        raise WorkloadTraceError(f"{path}.candidates must be an array")
    pruned_raw = raw["pruned_candidate_ids"]
    if not isinstance(pruned_raw, list) or any(
        not isinstance(candidate_id, str) for candidate_id in pruned_raw
    ):
        raise WorkloadTraceError(f"{path}.pruned_candidate_ids must be a string array")
    return StepTrace(
        step_index=_integer(raw, "step_index", path),
        shared_kv_tokens=_integer(raw, "shared_kv_tokens", path),
        candidates=tuple(
            _candidate(candidate, f"{path}.candidates[{index}]")
            for index, candidate in enumerate(candidates_raw)
        ),
        selected_candidate_id=_string(raw, "selected_candidate_id", path),
        pruned_candidate_ids=tuple(pruned_raw),
    )


def _parse_request_v1(raw_value: Any, path: str) -> TtcRequestTrace:
    """Parse one strict schema-v1 synthetic request."""

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
            "prompt_tokens",
            "image_tokens",
            "search_width",
            "seed",
            "sampling",
            "provenance",
            "steps",
        },
    )
    bucket_raw = raw["question_length_bucket"]
    bucket = (
        None
        if bucket_raw is None
        else _enum(QuestionLengthBucket, bucket_raw, f"{path}.question_length_bucket")
    )
    steps_raw = raw["steps"]
    if not isinstance(steps_raw, list):
        raise WorkloadTraceError(f"{path}.steps must be an array")
    return TtcRequestTrace(
        trace_schema_version=_integer(raw, "trace_schema_version", path),
        request_id=_string(raw, "request_id", path),
        dataset=_string(raw, "dataset", path),
        dataset_id=_string(raw, "dataset_id", path),
        modality=_enum(Modality, raw["modality"], f"{path}.modality"),
        difficulty=_string(raw, "difficulty", path),
        question_length_bucket=bucket,
        prompt_tokens=_integer(raw, "prompt_tokens", path),
        image_tokens=_integer(raw, "image_tokens", path),
        search_width=_integer(raw, "search_width", path),
        seed=_integer(raw, "seed", path),
        sampling=_sampling(raw["sampling"], f"{path}.sampling"),
        provenance=_provenance(raw["provenance"], f"{path}.provenance"),
        steps=tuple(
            _step(step, f"{path}.steps[{index}]")
            for index, step in enumerate(steps_raw)
        ),
    )


AnyTtcRequestTrace: TypeAlias = TtcRequestTrace | TtcRequestTraceV2


def parse_request(raw_value: Any, path: str = "request") -> AnyTtcRequestTrace:
    """Dispatch one strict request object to its versioned parser."""

    raw = _mapping(raw_value, path)
    if "trace_schema_version" not in raw:
        raise WorkloadTraceError(f"{path} is missing fields: trace_schema_version")
    version = raw["trace_schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise WorkloadTraceError(f"{path}.trace_schema_version must be an integer")
    if version == TRACE_SCHEMA_VERSION:
        return _parse_request_v1(raw, path)
    if version == TRACE_SCHEMA_VERSION_V2:
        return parse_request_v2(raw, path)
    raise WorkloadTraceError(f"unsupported trace schema version {version!r}")


def write_jsonl(path: str | Path, traces: list[AnyTtcRequestTrace]) -> None:
    """Atomically write deterministic compact JSONL in request order."""

    if not traces:
        raise WorkloadTraceError("cannot write an empty trace collection")
    request_ids = [trace.request_id for trace in traces]
    if len(set(request_ids)) != len(request_ids):
        raise WorkloadTraceError("request IDs must be unique within a trace file")
    versions = {trace.trace_schema_version for trace in traces}
    if len(versions) != 1:
        raise WorkloadTraceError("one trace file must not mix schema versions")

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
        raise WorkloadTraceError(f"cannot write trace file {destination}: {error}") from error


def read_jsonl(path: str | Path) -> list[AnyTtcRequestTrace]:
    """Read and validate a complete JSONL trace collection."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkloadTraceError(f"cannot read trace file {source}: {error}") from error
    if not lines:
        raise WorkloadTraceError(f"trace file {source} is empty")

    traces: list[AnyTtcRequestTrace] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise WorkloadTraceError(f"trace file {source} has blank line {line_number}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkloadTraceError(
                f"invalid JSON in {source} line {line_number}: {error.msg}"
            ) from error
        traces.append(parse_request(raw, path=f"request[line={line_number}]"))

    request_ids = [trace.request_id for trace in traces]
    if len(set(request_ids)) != len(request_ids):
        raise WorkloadTraceError(f"trace file {source} contains duplicate request IDs")
    versions = {trace.trace_schema_version for trace in traces}
    if len(versions) != 1:
        raise WorkloadTraceError(f"trace file {source} mixes schema versions")
    return traces


def trace_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact serialized trace bytes."""

    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkloadTraceError(f"cannot hash trace file {source}: {error}") from error
