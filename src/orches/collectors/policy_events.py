"""Raw, content-free policy events and deterministic trace construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from ..errors import WorkloadTraceError
from ..workload.policy_schema import (
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
from ..workload.schema import (
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
)
from ..workload.schema_v2 import KvBlockTraceV2, TokenTensorTrace


POLICY_EVENT_SCHEMA_VERSION = 1


def _require_nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise WorkloadTraceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadTraceError(
            f"{name} must be a non-negative integer, got {value!r}"
        )


class CollectionStatus(str, Enum):
    """Terminal status retained even when no trace can be constructed."""

    SUCCESS = "success"
    FAILED = "failed"
    OOM = "oom"


@dataclass(frozen=True)
class PolicyRequestStartedEvent:
    """Request-level configuration frozen before the first model call."""

    event_index: int
    request_id: str
    dataset: str
    dataset_id: str
    modality: Modality
    difficulty: str
    question_length_bucket: QuestionLengthBucket | None
    image_tokens: int
    search_width: int
    beam_size: int
    seed: int
    dtype: str
    sampling: SamplingConfig
    collection_mode: PolicyCollectionMode
    provenance: PolicyTraceProvenance

    event_type = "request_started"

    def __post_init__(self) -> None:
        if self.event_index != 0:
            raise WorkloadTraceError("request_started event_index must be zero")
        for name, value in (
            ("event.request_id", self.request_id),
            ("event.dataset", self.dataset),
            ("event.dataset_id", self.dataset_id),
            ("event.difficulty", self.difficulty),
            ("event.dtype", self.dtype),
        ):
            _require_nonempty(name, value)
        _require_nonnegative_integer("event.image_tokens", self.image_tokens)
        if self.search_width <= 0 or self.beam_size <= 0:
            raise WorkloadTraceError(
                "request_started search_width and beam_size must be positive"
            )
        _require_nonnegative_integer("event.seed", self.seed)
        if self.modality is Modality.TEXT:
            if self.image_tokens != 0 or self.question_length_bucket is not None:
                raise WorkloadTraceError(
                    "text request events require no image metadata"
                )
        elif self.image_tokens == 0 or self.question_length_bucket is None:
            raise WorkloadTraceError(
                "vision request events require image metadata"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_event_schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "dataset": self.dataset,
            "dataset_id": self.dataset_id,
            "modality": self.modality.value,
            "difficulty": self.difficulty,
            "question_length_bucket": (
                None
                if self.question_length_bucket is None
                else self.question_length_bucket.value
            ),
            "image_tokens": self.image_tokens,
            "search_width": self.search_width,
            "beam_size": self.beam_size,
            "seed": self.seed,
            "dtype": self.dtype,
            "sampling": self.sampling.to_dict(),
            "collection_mode": self.collection_mode.value,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class PolicyGenerationOutput:
    """One sanitized worker output, preserving tokens but never decoded text."""

    sequence_index: int
    candidate_id: str
    generated_token_ids: tuple[int, ...]
    call_token_ready_indices: tuple[int, ...]
    materialized_output_tokens: int
    finish_reason: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer("output.sequence_index", self.sequence_index)
        _require_nonempty("output.candidate_id", self.candidate_id)
        if not self.generated_token_ids:
            raise WorkloadTraceError("output.generated_token_ids must not be empty")
        for index, token_id in enumerate(self.generated_token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkloadTraceError(
                    f"output.generated_token_ids[{index}] must be an integer"
                )
        if len(self.call_token_ready_indices) != len(self.generated_token_ids):
            raise WorkloadTraceError(
                "output call-token order must cover every generated token"
            )
        previous = -1
        for ready_index in self.call_token_ready_indices:
            _require_nonnegative_integer("output.call_token_ready_indices", ready_index)
            if ready_index <= previous:
                raise WorkloadTraceError(
                    "output.call_token_ready_indices must be strictly increasing"
                )
            previous = ready_index
        _require_nonnegative_integer(
            "output.materialized_output_tokens", self.materialized_output_tokens
        )
        if self.materialized_output_tokens > len(self.generated_token_ids):
            raise WorkloadTraceError(
                "output materialized tokens cannot exceed generated tokens"
            )
        _require_nonempty("output.finish_reason", self.finish_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "candidate_id": self.candidate_id,
            "generated_token_ids": list(self.generated_token_ids),
            "call_token_ready_indices": list(self.call_token_ready_indices),
            "materialized_output_tokens": self.materialized_output_tokens,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class PolicyGenerationEvent:
    """Exact input and outputs observed for one policy-engine request."""

    event_index: int
    request_id: str
    step_index: int
    call_id: str
    parent_candidate_id: str | None
    worker_request_id: str
    input_tokens: TokenTensorTrace
    rng_seed: int
    rng_stream_id: str
    outputs: tuple[PolicyGenerationOutput, ...]

    event_type = "generation"

    def __post_init__(self) -> None:
        _require_nonnegative_integer("generation.event_index", self.event_index)
        _require_nonnegative_integer("generation.step_index", self.step_index)
        for name, value in (
            ("generation.request_id", self.request_id),
            ("generation.call_id", self.call_id),
            ("generation.worker_request_id", self.worker_request_id),
            ("generation.rng_stream_id", self.rng_stream_id),
        ):
            _require_nonempty(name, value)
        if self.parent_candidate_id is not None:
            _require_nonempty(
                "generation.parent_candidate_id", self.parent_candidate_id
            )
        _require_nonnegative_integer("generation.rng_seed", self.rng_seed)
        if not self.outputs:
            raise WorkloadTraceError("generation.outputs must not be empty")
        sequence_indices = tuple(output.sequence_index for output in self.outputs)
        if sequence_indices != tuple(range(len(self.outputs))):
            raise WorkloadTraceError(
                "generation output sequence indices must be contiguous in return order"
            )
        candidate_ids = tuple(output.candidate_id for output in self.outputs)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise WorkloadTraceError(
                "generation candidate IDs must be unique within a call"
            )
        ready_indices = sorted(
            index
            for output in self.outputs
            for index in output.call_token_ready_indices
        )
        expected = list(range(sum(len(output.generated_token_ids) for output in self.outputs)))
        if ready_indices != expected:
            raise WorkloadTraceError(
                "generation call-token order must be contiguous across outputs"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_event_schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "step_index": self.step_index,
            "call_id": self.call_id,
            "parent_candidate_id": self.parent_candidate_id,
            "worker_request_id": self.worker_request_id,
            "input_tokens": self.input_tokens.to_dict(),
            "rng_seed": self.rng_seed,
            "rng_stream_id": self.rng_stream_id,
            "outputs": [output.to_dict() for output in self.outputs],
        }


@dataclass(frozen=True)
class PolicySelectionEvent:
    """Pipeline branch decision observed after one complete expansion step."""

    event_index: int
    request_id: str
    step_index: int
    decision_id: str
    kind: PolicySelectionKind
    selected_candidate_ids: tuple[str, ...]
    pruned_candidate_ids: tuple[str, ...]

    event_type = "selection"

    def __post_init__(self) -> None:
        _require_nonnegative_integer("selection.event_index", self.event_index)
        _require_nonnegative_integer("selection.step_index", self.step_index)
        _require_nonempty("selection.request_id", self.request_id)
        _require_nonempty("selection.decision_id", self.decision_id)
        if not self.selected_candidate_ids:
            raise WorkloadTraceError(
                "selection.selected_candidate_ids must not be empty"
            )
        selected = set(self.selected_candidate_ids)
        pruned = set(self.pruned_candidate_ids)
        if (
            len(selected) != len(self.selected_candidate_ids)
            or len(pruned) != len(self.pruned_candidate_ids)
            or selected & pruned
        ):
            raise WorkloadTraceError(
                "selection candidate IDs must be unique and disjoint"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_event_schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "step_index": self.step_index,
            "decision_id": self.decision_id,
            "kind": self.kind.value,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "pruned_candidate_ids": list(self.pruned_candidate_ids),
        }


@dataclass(frozen=True)
class PolicyRequestFinishedEvent:
    """Terminal collection state retained for success, failure, and OOM."""

    event_index: int
    request_id: str
    status: CollectionStatus
    error: str | None

    event_type = "request_finished"

    def __post_init__(self) -> None:
        _require_nonnegative_integer("finished.event_index", self.event_index)
        _require_nonempty("finished.request_id", self.request_id)
        if self.status is CollectionStatus.SUCCESS:
            if self.error is not None:
                raise WorkloadTraceError("successful collection must not have an error")
        elif self.error is None or not self.error.strip():
            raise WorkloadTraceError("failed/OOM collection requires an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_event_schema_version": POLICY_EVENT_SCHEMA_VERSION,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "status": self.status.value,
            "error": self.error,
        }


AnyPolicyEvent: TypeAlias = (
    PolicyRequestStartedEvent
    | PolicyGenerationEvent
    | PolicySelectionEvent
    | PolicyRequestFinishedEvent
)


def _validate_event_stream(events: list[AnyPolicyEvent], source: str) -> None:
    if not events:
        raise WorkloadTraceError(f"{source} is empty")
    if not isinstance(events[0], PolicyRequestStartedEvent):
        raise WorkloadTraceError(f"{source} must begin with request_started")
    if not isinstance(events[-1], PolicyRequestFinishedEvent):
        raise WorkloadTraceError(f"{source} must end with request_finished")
    if [event.event_index for event in events] != list(range(len(events))):
        raise WorkloadTraceError(f"{source} event indices must be contiguous")
    request_id = events[0].request_id
    if any(event.request_id != request_id for event in events):
        raise WorkloadTraceError(f"{source} must contain exactly one request")
    if any(
        isinstance(event, PolicyRequestFinishedEvent)
        for event in events[1:-1]
    ):
        raise WorkloadTraceError(f"{source} contains an early terminal event")


def write_policy_events(path: str | Path, events: list[AnyPolicyEvent]) -> None:
    """Atomically write one request's canonical, content-free raw event log."""

    _validate_event_stream(events, "policy event stream")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for event in events:
                output.write(
                    json.dumps(
                        event.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                )
                output.write("\n")
        temporary.replace(destination)
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot write policy event file {destination}: {error}"
        ) from error


def policy_event_sha256(path: str | Path) -> str:
    """Hash the exact raw event bytes used as opaque-selection evidence."""

    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot hash policy event file {source}: {error}"
        ) from error


def build_policy_request_from_events(
    events: list[AnyPolicyEvent], *, raw_event_sha256: str
) -> PolicyRequestTrace:
    """Convert one successful raw event stream into the policy trace contract."""

    _validate_event_stream(events, "policy event stream")
    started = events[0]
    finished = events[-1]
    assert isinstance(started, PolicyRequestStartedEvent)
    assert isinstance(finished, PolicyRequestFinishedEvent)
    if finished.status is not CollectionStatus.SUCCESS:
        raise WorkloadTraceError(
            f"cannot build a policy trace from {finished.status.value} events"
        )
    generations = [
        event for event in events if isinstance(event, PolicyGenerationEvent)
    ]
    selections = [
        event for event in events if isinstance(event, PolicySelectionEvent)
    ]
    if not generations:
        raise WorkloadTraceError("successful policy events contain no generation")

    first_call = generations[0]
    if first_call.step_index != 0 or first_call.parent_candidate_id is not None:
        raise WorkloadTraceError("first generation must expand the request root")
    root_input = first_call.input_tokens
    root_block_id = f"{started.request_id}/kv/root"
    blocks = [
        KvBlockTraceV2(
            block_id=root_block_id,
            parent_block_id=None,
            owner_candidate_id=None,
            token_count=root_input.model_token_count,
        )
    ]
    terminal_by_candidate: dict[str, tuple[str, int]] = {}
    token_ready_offset = 0
    step_calls: dict[int, list[PolicyGenerationCallTrace]] = {}

    for event in generations:
        if event.parent_candidate_id is None:
            parent_block_id = root_block_id
            reused_kv_tokens = root_input.model_token_count
        else:
            parent_context = terminal_by_candidate.get(event.parent_candidate_id)
            if parent_context is None:
                raise WorkloadTraceError(
                    f"generation references unknown parent {event.parent_candidate_id!r}"
                )
            parent_block_id, reused_kv_tokens = parent_context

        candidates: list[PolicyCandidateTrace] = []
        for output in event.outputs:
            extension_tokens = (
                event.input_tokens.model_token_count
                + output.materialized_output_tokens
                - reused_kv_tokens
            )
            if extension_tokens < 0:
                raise WorkloadTraceError(
                    f"candidate {output.candidate_id!r} has negative KV extension"
                )
            if extension_tokens == 0:
                terminal_block_id = parent_block_id
            else:
                terminal_block_id = f"{output.candidate_id}/kv"
                blocks.append(
                    KvBlockTraceV2(
                        block_id=terminal_block_id,
                        parent_block_id=parent_block_id,
                        owner_candidate_id=output.candidate_id,
                        token_count=extension_tokens,
                    )
                )
            terminal_tokens = reused_kv_tokens + extension_tokens
            terminal_by_candidate[output.candidate_id] = (
                terminal_block_id,
                terminal_tokens,
            )
            candidates.append(
                PolicyCandidateTrace(
                    candidate_id=output.candidate_id,
                    generated_token_ids=output.generated_token_ids,
                    token_ready_indices=tuple(
                        token_ready_offset + index
                        for index in output.call_token_ready_indices
                    ),
                    materialized_output_tokens=output.materialized_output_tokens,
                    terminal_kv_block_id=terminal_block_id,
                    finish_reason=output.finish_reason,
                )
            )
        token_ready_offset += sum(
            len(output.generated_token_ids) for output in event.outputs
        )
        step_calls.setdefault(event.step_index, []).append(
            PolicyGenerationCallTrace(
                call_id=event.call_id,
                parent_candidate_id=event.parent_candidate_id,
                input_tokens=event.input_tokens,
                reused_kv_tokens=reused_kv_tokens,
                rng_seed=event.rng_seed,
                rng_stream_id=event.rng_stream_id,
                candidates=tuple(candidates),
            )
        )

    selection_by_step: dict[int, PolicySelectionEvent] = {}
    for event in selections:
        if event.step_index in selection_by_step:
            raise WorkloadTraceError(
                f"policy events contain multiple selections for step {event.step_index}"
            )
        selection_by_step[event.step_index] = event

    steps: list[PolicyStepTrace] = []
    expected_step_indices = list(range(max(step_calls) + 1))
    if sorted(step_calls) != expected_step_indices:
        raise WorkloadTraceError("generation step indices must be contiguous")
    for step_index in expected_step_indices:
        selection_event = selection_by_step.get(step_index)
        selection = (
            None
            if selection_event is None
            else PolicySelectionTrace(
                decision_id=selection_event.decision_id,
                kind=selection_event.kind,
                selected_candidate_ids=selection_event.selected_candidate_ids,
                pruned_candidate_ids=selection_event.pruned_candidate_ids,
                decision_artifact_sha256=raw_event_sha256,
            )
        )
        steps.append(
            PolicyStepTrace(
                step_index=step_index,
                generation_calls=tuple(step_calls[step_index]),
                selection=selection,
            )
        )

    return PolicyRequestTrace(
        policy_trace_schema_version=POLICY_TRACE_SCHEMA_VERSION,
        workload_scope=WorkloadScope.GENERATION_ONLY,
        paper_eligible=False,
        collection_mode=started.collection_mode,
        request_id=started.request_id,
        dataset=started.dataset,
        dataset_id=started.dataset_id,
        modality=started.modality,
        difficulty=started.difficulty,
        question_length_bucket=started.question_length_bucket,
        image_tokens=started.image_tokens,
        search_width=started.search_width,
        beam_size=started.beam_size,
        seed=started.seed,
        dtype=started.dtype,
        sampling=started.sampling,
        provenance=started.provenance,
        root_input=root_input,
        kv_blocks=tuple(blocks),
        steps=tuple(steps),
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


def _array(raw: Any, path: str) -> list[Any]:
    if not isinstance(raw, list):
        raise WorkloadTraceError(f"{path} must be an array")
    return raw


def _integer_array(raw: Any, path: str) -> tuple[int, ...]:
    values = _array(raw, path)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise WorkloadTraceError(f"{path} must be an integer array")
    return tuple(values)


def _string_array(raw: Any, path: str) -> tuple[str, ...]:
    values = _array(raw, path)
    if any(not isinstance(value, str) for value in values):
        raise WorkloadTraceError(f"{path} must be a string array")
    return tuple(values)


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
    return ModelRef(_string(raw, "name", path), _string(raw, "revision", path))


def _sampling(raw_value: Any, path: str) -> SamplingConfig:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"temperature", "top_p", "max_new_tokens"})
    temperature = raw["temperature"]
    top_p = raw["top_p"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
    ):
        raise WorkloadTraceError(f"{path} temperature/top_p must be numbers")
    return SamplingConfig(
        temperature=float(temperature),
        top_p=float(top_p),
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
    selector = raw["selector"]
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
        selector=None if selector is None else _model_ref(selector, f"{path}.selector"),
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


def _parse_started(raw: RawMapping, path: str) -> PolicyRequestStartedEvent:
    _keys(
        raw,
        path,
        {
            "policy_event_schema_version",
            "event_index",
            "event_type",
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
            "collection_mode",
            "provenance",
        },
    )
    bucket = raw["question_length_bucket"]
    return PolicyRequestStartedEvent(
        event_index=_integer(raw, "event_index", path),
        request_id=_string(raw, "request_id", path),
        dataset=_string(raw, "dataset", path),
        dataset_id=_string(raw, "dataset_id", path),
        modality=_enum(Modality, raw["modality"], f"{path}.modality"),
        difficulty=_string(raw, "difficulty", path),
        question_length_bucket=(
            None
            if bucket is None
            else _enum(QuestionLengthBucket, bucket, f"{path}.question_length_bucket")
        ),
        image_tokens=_integer(raw, "image_tokens", path),
        search_width=_integer(raw, "search_width", path),
        beam_size=_integer(raw, "beam_size", path),
        seed=_integer(raw, "seed", path),
        dtype=_string(raw, "dtype", path),
        sampling=_sampling(raw["sampling"], f"{path}.sampling"),
        collection_mode=_enum(
            PolicyCollectionMode,
            raw["collection_mode"],
            f"{path}.collection_mode",
        ),
        provenance=_provenance(raw["provenance"], f"{path}.provenance"),
    )


def _parse_output(raw_value: Any, path: str) -> PolicyGenerationOutput:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "sequence_index",
            "candidate_id",
            "generated_token_ids",
            "call_token_ready_indices",
            "materialized_output_tokens",
            "finish_reason",
        },
    )
    return PolicyGenerationOutput(
        sequence_index=_integer(raw, "sequence_index", path),
        candidate_id=_string(raw, "candidate_id", path),
        generated_token_ids=_integer_array(
            raw["generated_token_ids"], f"{path}.generated_token_ids"
        ),
        call_token_ready_indices=_integer_array(
            raw["call_token_ready_indices"],
            f"{path}.call_token_ready_indices",
        ),
        materialized_output_tokens=_integer(
            raw, "materialized_output_tokens", path
        ),
        finish_reason=_string(raw, "finish_reason", path),
    )


def _parse_generation(raw: RawMapping, path: str) -> PolicyGenerationEvent:
    _keys(
        raw,
        path,
        {
            "policy_event_schema_version",
            "event_index",
            "event_type",
            "request_id",
            "step_index",
            "call_id",
            "parent_candidate_id",
            "worker_request_id",
            "input_tokens",
            "rng_seed",
            "rng_stream_id",
            "outputs",
        },
    )
    outputs = _array(raw["outputs"], f"{path}.outputs")
    return PolicyGenerationEvent(
        event_index=_integer(raw, "event_index", path),
        request_id=_string(raw, "request_id", path),
        step_index=_integer(raw, "step_index", path),
        call_id=_string(raw, "call_id", path),
        parent_candidate_id=_optional_string(raw, "parent_candidate_id", path),
        worker_request_id=_string(raw, "worker_request_id", path),
        input_tokens=_token_tensor(raw["input_tokens"], f"{path}.input_tokens"),
        rng_seed=_integer(raw, "rng_seed", path),
        rng_stream_id=_string(raw, "rng_stream_id", path),
        outputs=tuple(
            _parse_output(output, f"{path}.outputs[{index}]")
            for index, output in enumerate(outputs)
        ),
    )


def _parse_selection(raw: RawMapping, path: str) -> PolicySelectionEvent:
    _keys(
        raw,
        path,
        {
            "policy_event_schema_version",
            "event_index",
            "event_type",
            "request_id",
            "step_index",
            "decision_id",
            "kind",
            "selected_candidate_ids",
            "pruned_candidate_ids",
        },
    )
    return PolicySelectionEvent(
        event_index=_integer(raw, "event_index", path),
        request_id=_string(raw, "request_id", path),
        step_index=_integer(raw, "step_index", path),
        decision_id=_string(raw, "decision_id", path),
        kind=_enum(PolicySelectionKind, raw["kind"], f"{path}.kind"),
        selected_candidate_ids=_string_array(
            raw["selected_candidate_ids"], f"{path}.selected_candidate_ids"
        ),
        pruned_candidate_ids=_string_array(
            raw["pruned_candidate_ids"], f"{path}.pruned_candidate_ids"
        ),
    )


def _parse_finished(raw: RawMapping, path: str) -> PolicyRequestFinishedEvent:
    _keys(
        raw,
        path,
        {
            "policy_event_schema_version",
            "event_index",
            "event_type",
            "request_id",
            "status",
            "error",
        },
    )
    return PolicyRequestFinishedEvent(
        event_index=_integer(raw, "event_index", path),
        request_id=_string(raw, "request_id", path),
        status=_enum(CollectionStatus, raw["status"], f"{path}.status"),
        error=_optional_string(raw, "error", path),
    )


def parse_policy_event(raw_value: Any, path: str) -> AnyPolicyEvent:
    """Parse one strict event without admitting decoded text or unknown sizes."""

    raw = _mapping(raw_value, path)
    for field in ("policy_event_schema_version", "event_type"):
        if field not in raw:
            raise WorkloadTraceError(f"{path} is missing fields: {field}")
    version = _integer(raw, "policy_event_schema_version", path)
    if version != POLICY_EVENT_SCHEMA_VERSION:
        raise WorkloadTraceError(
            f"unsupported policy event schema version {version!r}"
        )
    event_type = _string(raw, "event_type", path)
    parsers = {
        PolicyRequestStartedEvent.event_type: _parse_started,
        PolicyGenerationEvent.event_type: _parse_generation,
        PolicySelectionEvent.event_type: _parse_selection,
        PolicyRequestFinishedEvent.event_type: _parse_finished,
    }
    parser = parsers.get(event_type)
    if parser is None:
        raise WorkloadTraceError(f"unsupported policy event type {event_type!r}")
    return parser(raw, path)


def read_policy_events(path: str | Path) -> list[AnyPolicyEvent]:
    """Read one request's complete raw policy event stream."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot read policy event file {source}: {error}"
        ) from error
    events: list[AnyPolicyEvent] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise WorkloadTraceError(
                f"policy event file {source} has blank line {line_number}"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkloadTraceError(
                f"invalid JSON in {source} line {line_number}: {error.msg}"
            ) from error
        events.append(parse_policy_event(raw, f"event[line={line_number}]"))
    _validate_event_stream(events, f"policy event file {source}")
    return events
