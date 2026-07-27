"""Adapter from enhanced compute-optimal-TTS worker responses to raw events."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import WorkloadTraceError
from ..workload.policy_schema import (
    PolicyCollectionMode,
    PolicySelectionKind,
    PolicyTraceProvenance,
)
from ..workload.schema import Modality, ModelRef, SamplingConfig, SourceKind
from ..workload.schema_v2 import TokenTensorTrace
from .policy_events import (
    CollectionStatus,
    PolicyGenerationEvent,
    PolicyGenerationOutput,
    PolicyRequestFinishedEvent,
    PolicyRequestStartedEvent,
    PolicySelectionEvent,
    build_policy_request_from_events,
    write_policy_events,
)


COMPUTE_OPTIMAL_TTS_REVISION = "0ee2578af1f8d6cac445c9c4c72780528bb94556"
COLLECTOR_CONFIG_SCHEMA_VERSION = 1
_REQUEST_PATH_PATTERN = re.compile(r"question_(\d+)/record_(\d+)\.jsonl$")


@dataclass(frozen=True)
class VllmOutputSnapshot:
    """One output sequence as exposed by a vLLM RequestOutput update."""

    sequence_index: int
    token_ids: tuple[int, ...]
    finish_reason: str | None


class VllmSnapshotAccumulator:
    """Convert cumulative vLLM updates into linear-size exact token evidence."""

    def __init__(self, expected_outputs: int) -> None:
        if (
            isinstance(expected_outputs, bool)
            or not isinstance(expected_outputs, int)
            or expected_outputs <= 0
        ):
            raise WorkloadTraceError(
                "vLLM expected output count must be a positive integer"
            )
        self._expected_outputs = expected_outputs
        self._prompt_token_ids: tuple[int, ...] | None = None
        self._token_ids: dict[int, tuple[int, ...]] = {}
        self._ready_indices: dict[int, list[int]] = {}
        self._materialized: dict[int, int] = {}
        self._finish_reasons: dict[int, str] = {}
        self._next_ready_index = 0

    def observe(
        self,
        prompt_token_ids: tuple[int, ...],
        outputs: tuple[VllmOutputSnapshot, ...],
    ) -> None:
        """Consume one cumulative engine update in stable output-index order."""

        if not prompt_token_ids:
            raise WorkloadTraceError("vLLM prompt_token_ids must not be empty")
        if self._prompt_token_ids is None:
            self._prompt_token_ids = prompt_token_ids
        elif prompt_token_ids != self._prompt_token_ids:
            raise WorkloadTraceError("vLLM prompt token IDs changed during one request")
        if tuple(output.sequence_index for output in outputs) != tuple(
            range(len(outputs))
        ):
            raise WorkloadTraceError(
                "vLLM output indices must be contiguous in return order"
            )
        for output in outputs:
            previous = self._token_ids.get(output.sequence_index, ())
            if output.token_ids[: len(previous)] != previous:
                raise WorkloadTraceError(
                    f"vLLM output {output.sequence_index} rewrote generated token IDs"
                )
            ready = self._ready_indices.setdefault(output.sequence_index, [])
            for _ in output.token_ids[len(previous) :]:
                ready.append(self._next_ready_index)
                self._next_ready_index += 1
            self._token_ids[output.sequence_index] = output.token_ids
            if output.finish_reason is not None:
                if not output.finish_reason:
                    raise WorkloadTraceError("vLLM finish reason must not be empty")
                prior_reason = self._finish_reasons.get(output.sequence_index)
                if prior_reason is not None and prior_reason != output.finish_reason:
                    raise WorkloadTraceError(
                        f"vLLM output {output.sequence_index} changed finish reason"
                    )
                if prior_reason is None:
                    # The terminal sampled token has not entered another decode round.
                    self._materialized[output.sequence_index] = max(
                        0, len(output.token_ids) - 1
                    )
                    self._finish_reasons[output.sequence_index] = output.finish_reason

    def final_payload(self, worker_request_id: str) -> dict[str, object]:
        """Return the exact response extension after every sequence finishes."""

        if not worker_request_id:
            raise WorkloadTraceError("vLLM worker request ID must not be empty")
        if self._prompt_token_ids is None or not self._token_ids:
            raise WorkloadTraceError("vLLM request produced no token snapshots")
        indices = tuple(range(self._expected_outputs))
        if tuple(sorted(self._token_ids)) != indices:
            raise WorkloadTraceError("vLLM final output indices are not contiguous")
        if set(self._finish_reasons) != set(indices):
            raise WorkloadTraceError("vLLM request ended before every output finished")
        return {
            "request_id": worker_request_id,
            "prompt_token_ids": list(self._prompt_token_ids),
            "output_token_ids": [list(self._token_ids[index]) for index in indices],
            "token_ready_indices": [
                list(self._ready_indices[index]) for index in indices
            ],
            "materialized_output_tokens": [
                self._materialized[index] for index in indices
            ],
            "finish_reason": [self._finish_reasons[index] for index in indices],
        }

    @property
    def is_complete(self) -> bool:
        """Whether every currently known output has a terminal engine update."""

        expected = set(range(self._expected_outputs))
        return set(self._token_ids) == expected and set(self._finish_reasons) == expected


class ComputeOptimalTtsAdapter:
    """Sanitize exact worker tokens into one source-independent generation event."""

    required_response_fields = frozenset(
        {
            "request_id",
            "prompt_token_ids",
            "output_token_ids",
            "token_ready_indices",
            "materialized_output_tokens",
            "finish_reason",
        }
    )

    @classmethod
    def generation_event(
        cls,
        response: Mapping[str, Any],
        *,
        event_index: int,
        request_id: str,
        step_index: int,
        call_id: str,
        parent_candidate_id: str | None,
        rng_seed: int,
        rng_stream_id: str,
    ) -> PolicyGenerationEvent:
        """Build an event while intentionally ignoring all decoded-text fields."""

        missing = sorted(cls.required_response_fields - set(response))
        if missing:
            raise WorkloadTraceError(
                "compute-optimal-TTS worker response is missing exact fields: "
                + ", ".join(missing)
            )
        prompt_token_ids = cls._integer_list(
            response["prompt_token_ids"], "response.prompt_token_ids"
        )
        output_token_ids = cls._nested_integer_lists(
            response["output_token_ids"], "response.output_token_ids"
        )
        token_ready_indices = cls._nested_integer_lists(
            response["token_ready_indices"], "response.token_ready_indices"
        )
        materialized = cls._integer_list(
            response["materialized_output_tokens"],
            "response.materialized_output_tokens",
        )
        finish_reasons = response["finish_reason"]
        if not isinstance(finish_reasons, list) or any(
            not isinstance(reason, str) for reason in finish_reasons
        ):
            raise WorkloadTraceError("response.finish_reason must be a string array")
        count = len(output_token_ids)
        if not (
            count
            == len(token_ready_indices)
            == len(materialized)
            == len(finish_reasons)
        ):
            raise WorkloadTraceError(
                "compute-optimal-TTS exact output arrays must have equal lengths"
            )
        worker_request_id = response["request_id"]
        if not isinstance(worker_request_id, str) or not worker_request_id:
            raise WorkloadTraceError("response.request_id must be a non-empty string")

        outputs = tuple(
            PolicyGenerationOutput(
                sequence_index=index,
                candidate_id=f"{call_id}/output-{index}",
                generated_token_ids=tuple(output_token_ids[index]),
                call_token_ready_indices=tuple(token_ready_indices[index]),
                materialized_output_tokens=materialized[index],
                finish_reason=finish_reasons[index],
            )
            for index in range(count)
        )
        return PolicyGenerationEvent(
            event_index=event_index,
            request_id=request_id,
            step_index=step_index,
            call_id=call_id,
            parent_candidate_id=parent_candidate_id,
            worker_request_id=worker_request_id,
            input_tokens=TokenTensorTrace(
                token_ids=tuple(prompt_token_ids),
                attention_mask=(1,) * len(prompt_token_ids),
                model_token_count=len(prompt_token_ids),
            ),
            rng_seed=rng_seed,
            rng_stream_id=rng_stream_id,
            outputs=outputs,
        )

    @staticmethod
    def _integer_list(raw: Any, path: str) -> list[int]:
        if not isinstance(raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw
        ):
            raise WorkloadTraceError(f"{path} must be an integer array")
        return raw

    @classmethod
    def _nested_integer_lists(cls, raw: Any, path: str) -> list[list[int]]:
        if not isinstance(raw, list):
            raise WorkloadTraceError(f"{path} must be an array of integer arrays")
        return [
            cls._integer_list(value, f"{path}[{index}]")
            for index, value in enumerate(raw)
        ]


@dataclass(frozen=True)
class ComputeOptimalTtsCollectorConfig:
    """Static provenance and output location supplied to upstream Ray workers."""

    event_dir: Path
    dataset: str
    dataset_revision: str
    dtype: str
    tokenizer: ModelRef
    policy_model: ModelRef
    inference_engine: ModelRef
    collector: ModelRef
    selector: ModelRef

    @classmethod
    def read(cls, path: str | Path) -> "ComputeOptimalTtsCollectorConfig":
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise WorkloadTraceError(
                f"cannot read compute-optimal-TTS collector config {source}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise WorkloadTraceError(
                f"invalid collector config JSON {source}: {error.msg}"
            ) from error
        if not isinstance(raw, Mapping):
            raise WorkloadTraceError("collector config must be an object")
        required = {
            "config_schema_version",
            "event_dir",
            "dataset",
            "dataset_revision",
            "dtype",
            "tokenizer",
            "policy_model",
            "inference_engine",
            "collector",
            "selector",
        }
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            detail = []
            if missing:
                detail.append("missing: " + ", ".join(missing))
            if unknown:
                detail.append("unknown: " + ", ".join(unknown))
            raise WorkloadTraceError("collector config fields " + "; ".join(detail))
        version = raw["config_schema_version"]
        if (
            isinstance(version, bool)
            or version != COLLECTOR_CONFIG_SCHEMA_VERSION
        ):
            raise WorkloadTraceError(
                f"unsupported collector config schema version {version!r}"
            )
        event_dir_raw = cls._string(raw, "event_dir")
        event_dir = Path(event_dir_raw)
        if not event_dir.is_absolute():
            event_dir = (source.parent / event_dir).resolve()
        return cls(
            event_dir=event_dir,
            dataset=cls._string(raw, "dataset"),
            dataset_revision=cls._string(raw, "dataset_revision"),
            dtype=cls._string(raw, "dtype"),
            tokenizer=cls._model_ref(raw["tokenizer"], "tokenizer"),
            policy_model=cls._model_ref(raw["policy_model"], "policy_model"),
            inference_engine=cls._model_ref(
                raw["inference_engine"], "inference_engine"
            ),
            collector=cls._model_ref(raw["collector"], "collector"),
            selector=cls._model_ref(raw["selector"], "selector"),
        )

    @staticmethod
    def _string(raw: Mapping[str, Any], key: str) -> str:
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise WorkloadTraceError(
                f"collector config {key} must be a non-empty string"
            )
        return value

    @staticmethod
    def _model_ref(raw: Any, path: str) -> ModelRef:
        if not isinstance(raw, Mapping) or set(raw) != {"name", "revision"}:
            raise WorkloadTraceError(
                f"collector config {path} must contain name and revision"
            )
        name = raw["name"]
        revision = raw["revision"]
        if not isinstance(name, str) or not isinstance(revision, str):
            raise WorkloadTraceError(
                f"collector config {path} name/revision must be strings"
            )
        return ModelRef(name=name, revision=revision)


class ComputeOptimalTtsSession:
    """Mutable per-request bridge shared by copied compute-optimal-TTS envs."""

    def __init__(
        self, started: PolicyRequestStartedEvent, event_path: str | Path
    ) -> None:
        self.started = started
        self.event_path = Path(event_path)
        self._events = [started]
        self._pending_candidates: dict[int, list[str]] = {}
        self._selected_steps: set[int] = set()
        self._calls_per_step: dict[int, int] = {}
        self._finished = False

    @property
    def events(
        self,
    ) -> tuple[
        PolicyRequestStartedEvent
        | PolicyGenerationEvent
        | PolicySelectionEvent
        | PolicyRequestFinishedEvent,
        ...,
    ]:
        """Events captured so far, exposed read-only for diagnostics and tests."""

        return tuple(self._events)

    def record_generation_result(
        self,
        result: Any,
        *,
        step_index: int,
        parent_candidate_id: str | None,
        rng_seed: int,
    ) -> tuple[str, ...]:
        """Record all worker outputs before upstream filters legal actions."""

        self._require_open()
        call_ordinal = self._calls_per_step.get(step_index, 0)
        self._calls_per_step[step_index] = call_ordinal + 1
        call_id = (
            f"{self.started.request_id}/step-{step_index}/call-{call_ordinal}"
        )
        response = {
            "request_id": getattr(result, "worker_request_id", None),
            "prompt_token_ids": getattr(result, "prompt_token_ids", None),
            "output_token_ids": getattr(result, "output_token_ids", None),
            "token_ready_indices": getattr(result, "token_ready_indices", None),
            "materialized_output_tokens": getattr(
                result, "materialized_output_tokens", None
            ),
            "finish_reason": getattr(result, "finish_reason", None),
        }
        event = ComputeOptimalTtsAdapter.generation_event(
            response,
            event_index=len(self._events),
            request_id=self.started.request_id,
            step_index=step_index,
            call_id=call_id,
            parent_candidate_id=parent_candidate_id,
            rng_seed=rng_seed,
            rng_stream_id=call_id,
        )
        self._events.append(event)
        pending = self._pending_candidates.setdefault(step_index, [])
        pending.extend(output.candidate_id for output in event.outputs)
        return tuple(output.candidate_id for output in event.outputs)

    def has_pending_step(self, step_index: int) -> bool:
        return bool(self._pending_candidates.get(step_index))

    def record_selection(
        self, step_index: int, selected_candidate_ids: tuple[str, ...]
    ) -> None:
        """Record the actual global beam decision for one generated step."""

        self._require_open()
        if step_index in self._selected_steps:
            raise WorkloadTraceError(
                f"compute-optimal-TTS step {step_index} was selected twice"
            )
        pending = self._pending_candidates.get(step_index, [])
        if not pending:
            raise WorkloadTraceError(
                f"compute-optimal-TTS step {step_index} has no generated candidates"
            )
        if not selected_candidate_ids or not set(selected_candidate_ids) <= set(
            pending
        ):
            raise WorkloadTraceError(
                f"compute-optimal-TTS step {step_index} selected unknown candidates"
            )
        selected = set(selected_candidate_ids)
        pruned = tuple(
            candidate_id for candidate_id in pending if candidate_id not in selected
        )
        kind = (
            PolicySelectionKind.OPAQUE
            if self.started.collection_mode is PolicyCollectionMode.OPAQUE_SELECTOR
            else PolicySelectionKind.SYNTHETIC
        )
        self._events.append(
            PolicySelectionEvent(
                event_index=len(self._events),
                request_id=self.started.request_id,
                step_index=step_index,
                decision_id=f"{self.started.request_id}/selection-{step_index}",
                kind=kind,
                selected_candidate_ids=selected_candidate_ids,
                pruned_candidate_ids=pruned,
            )
        )
        self._selected_steps.add(step_index)

    def finish_success(self) -> None:
        """Validate the complete tree before writing a successful terminal event."""

        self._require_open()
        finished = PolicyRequestFinishedEvent(
            event_index=len(self._events),
            request_id=self.started.request_id,
            status=CollectionStatus.SUCCESS,
            error=None,
        )
        complete_events = [*self._events, finished]
        try:
            build_policy_request_from_events(
                complete_events,
                raw_event_sha256="0" * 64,
            )
        except WorkloadTraceError as error:
            self.finish_exception(error)
            raise
        write_policy_events(self.event_path, complete_events)
        self._events = complete_events
        self._finished = True

    def finish_exception(self, error: BaseException) -> None:
        """Persist a failed or OOM stream without converting it to a trace."""

        if self._finished:
            return
        message = f"{type(error).__name__}: {error}"[:1000]
        status = (
            CollectionStatus.OOM
            if "out of memory" in message.lower() or "cuda oom" in message.lower()
            else CollectionStatus.FAILED
        )
        finished = PolicyRequestFinishedEvent(
            event_index=len(self._events),
            request_id=self.started.request_id,
            status=status,
            error=message,
        )
        self._events.append(finished)
        write_policy_events(self.event_path, self._events)
        self._finished = True

    def _require_open(self) -> None:
        if self._finished:
            raise WorkloadTraceError("compute-optimal-TTS session is already finished")


def create_compute_optimal_tts_session(
    problem: Mapping[str, Any],
    tree_config: Any,
    generation_config: Any,
) -> ComputeOptimalTtsSession | None:
    """Create one upstream session when ORCHES collector configuration is set."""

    config_path = os.environ.get("ORCHES_POLICY_COLLECTOR_CONFIG")
    if config_path is None:
        return None
    config = ComputeOptimalTtsCollectorConfig.read(config_path)
    beam_size = int(tree_config.beam_size)
    if beam_size != 1:
        raise WorkloadTraceError(
            "compute-optimal-TTS integration currently requires beam_size=1; "
            "upstream changes per-parent width when beam_size>1"
        )
    file_path = problem.get("file_path")
    if not isinstance(file_path, str):
        raise WorkloadTraceError(
            "compute-optimal-TTS problem is missing its assigned file_path"
        )
    match = _REQUEST_PATH_PATTERN.search(file_path)
    if match is None:
        raise WorkloadTraceError(
            f"cannot derive dataset identity from problem path {file_path!r}"
        )
    dataset_index, sample_index = match.groups()
    request_id = f"{config.dataset}/item-{dataset_index}/sample-{sample_index}"
    event_name = f"item-{dataset_index}-sample-{sample_index}.events.jsonl"
    difficulty = problem.get("level", "unreported")
    if not isinstance(difficulty, str):
        difficulty = str(difficulty)
    started = PolicyRequestStartedEvent(
        event_index=0,
        request_id=request_id,
        dataset=config.dataset,
        dataset_id=dataset_index,
        modality=Modality.TEXT,
        difficulty=difficulty,
        question_length_bucket=None,
        image_tokens=0,
        search_width=int(tree_config.tree_max_width),
        beam_size=beam_size,
        seed=int(generation_config.seed),
        dtype=config.dtype,
        sampling=SamplingConfig(
            temperature=float(generation_config.temperature),
            top_p=float(generation_config.top_p),
            max_new_tokens=int(generation_config.max_new_tokens),
        ),
        collection_mode=PolicyCollectionMode.OPAQUE_SELECTOR,
        provenance=PolicyTraceProvenance(
            source_kind=SourceKind.COLLECTED,
            pipeline="compute-optimal-tts",
            pipeline_revision=COMPUTE_OPTIMAL_TTS_REVISION,
            dataset_revision=config.dataset_revision,
            tokenizer=config.tokenizer,
            policy_model=config.policy_model,
            inference_engine=config.inference_engine,
            collector=config.collector,
            selector=config.selector,
        ),
    )
    return ComputeOptimalTtsSession(started, config.event_dir / event_name)
