"""Adapter from enhanced compute-optimal-TTS worker responses to raw events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import WorkloadTraceError
from ..workload.schema_v2 import TokenTensorTrace
from .policy_events import PolicyGenerationEvent, PolicyGenerationOutput


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
