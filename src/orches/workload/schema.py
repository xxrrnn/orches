"""Immutable schema for replayable test-time-compute request traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from ..errors import WorkloadTraceError


TRACE_SCHEMA_VERSION = 1


class Modality(str, Enum):
    """Input modality represented by one request trace."""

    TEXT = "text"
    VISION = "vision"


class QuestionLengthBucket(str, Enum):
    """Paper-facing MathVista question-length grouping."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SourceKind(str, Enum):
    """Whether a trace was collected from a real pipeline or synthesized."""

    COLLECTED = "collected"
    SYNTHETIC = "synthetic"


def _require_nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise WorkloadTraceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadTraceError(f"{name} must be a non-negative integer, got {value!r}")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadTraceError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ModelRef:
    """Immutable model or tokenizer identity used to collect a trace."""

    name: str
    revision: str

    def __post_init__(self) -> None:
        _require_nonempty("model.name", self.name)
        _require_nonempty("model.revision", self.revision)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "revision": self.revision}


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling inputs that affect candidate control flow."""

    temperature: float
    top_p: float
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not isfinite(self.temperature) or self.temperature < 0:
            raise WorkloadTraceError("sampling.temperature must be finite and non-negative")
        if not isfinite(self.top_p) or self.top_p <= 0 or self.top_p > 1:
            raise WorkloadTraceError("sampling.top_p must be finite and in (0, 1]")
        _require_positive_integer("sampling.max_new_tokens", self.max_new_tokens)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class TraceProvenance:
    """Frozen collector, dataset, tokenizer, and model revisions."""

    source_kind: SourceKind
    pipeline: str
    pipeline_revision: str
    dataset_revision: str
    tokenizer: ModelRef
    policy_model: ModelRef
    small_prm_model: ModelRef
    large_prm_model: ModelRef

    def __post_init__(self) -> None:
        _require_nonempty("provenance.pipeline", self.pipeline)
        _require_nonempty("provenance.pipeline_revision", self.pipeline_revision)
        _require_nonempty("provenance.dataset_revision", self.dataset_revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "pipeline": self.pipeline,
            "pipeline_revision": self.pipeline_revision,
            "dataset_revision": self.dataset_revision,
            "tokenizer": self.tokenizer.to_dict(),
            "policy_model": self.policy_model.to_dict(),
            "small_prm_model": self.small_prm_model.to_dict(),
            "large_prm_model": self.large_prm_model.to_dict(),
        }


@dataclass(frozen=True)
class CandidateTrace:
    """One generated candidate and its replay-relevant measurements."""

    candidate_id: str
    parent_candidate_id: str | None
    generated_tokens: int
    token_timestamps_us: tuple[float, ...]
    small_prm_score: float
    large_prm_score: float
    unique_kv_tokens: int

    def __post_init__(self) -> None:
        _require_nonempty("candidate.candidate_id", self.candidate_id)
        if self.parent_candidate_id is not None:
            _require_nonempty("candidate.parent_candidate_id", self.parent_candidate_id)
        _require_positive_integer("candidate.generated_tokens", self.generated_tokens)
        _require_positive_integer("candidate.unique_kv_tokens", self.unique_kv_tokens)
        if self.unique_kv_tokens != self.generated_tokens:
            raise WorkloadTraceError(
                "candidate.unique_kv_tokens must equal generated_tokens for schema v1"
            )
        if len(self.token_timestamps_us) != self.generated_tokens:
            raise WorkloadTraceError(
                "candidate.token_timestamps_us length must equal generated_tokens"
            )
        previous = -1.0
        for timestamp in self.token_timestamps_us:
            if not isfinite(timestamp) or timestamp < 0:
                raise WorkloadTraceError(
                    "candidate token timestamps must be finite and non-negative"
                )
            if timestamp < previous:
                raise WorkloadTraceError(
                    "candidate token timestamps must be monotonically non-decreasing"
                )
            previous = timestamp
        if not isfinite(self.small_prm_score) or not isfinite(self.large_prm_score):
            raise WorkloadTraceError("candidate PRM scores must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "generated_tokens": self.generated_tokens,
            "token_timestamps_us": list(self.token_timestamps_us),
            "small_prm_score": self.small_prm_score,
            "large_prm_score": self.large_prm_score,
            "unique_kv_tokens": self.unique_kv_tokens,
        }


@dataclass(frozen=True)
class StepTrace:
    """One TTC generation/verification step with an explicit branch decision."""

    step_index: int
    shared_kv_tokens: int
    candidates: tuple[CandidateTrace, ...]
    selected_candidate_id: str
    pruned_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("step.step_index", self.step_index)
        _require_positive_integer("step.shared_kv_tokens", self.shared_kv_tokens)
        if not self.candidates:
            raise WorkloadTraceError("step.candidates must not be empty")

        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise WorkloadTraceError("candidate IDs must be unique within a step")
        if self.selected_candidate_id not in candidate_ids:
            raise WorkloadTraceError(
                f"selected candidate {self.selected_candidate_id!r} does not exist"
            )
        if len(set(self.pruned_candidate_ids)) != len(self.pruned_candidate_ids):
            raise WorkloadTraceError("pruned candidate IDs must be unique")

        expected_pruned = set(candidate_ids) - {self.selected_candidate_id}
        if set(self.pruned_candidate_ids) != expected_pruned:
            raise WorkloadTraceError(
                "pruned_candidate_ids must contain every non-selected candidate exactly once"
            )

    @property
    def selected_candidate(self) -> CandidateTrace:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "shared_kv_tokens": self.shared_kv_tokens,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "pruned_candidate_ids": list(self.pruned_candidate_ids),
        }


@dataclass(frozen=True)
class TtcRequestTrace:
    """A complete TTC reasoning tree that can be replayed without model calls."""

    trace_schema_version: int
    request_id: str
    dataset: str
    dataset_id: str
    modality: Modality
    difficulty: str
    question_length_bucket: QuestionLengthBucket | None
    prompt_tokens: int
    image_tokens: int
    search_width: int
    seed: int
    sampling: SamplingConfig
    provenance: TraceProvenance
    steps: tuple[StepTrace, ...]

    def __post_init__(self) -> None:
        if self.trace_schema_version != TRACE_SCHEMA_VERSION:
            raise WorkloadTraceError(
                f"unsupported trace schema version {self.trace_schema_version!r}"
            )
        if self.provenance.source_kind is not SourceKind.SYNTHETIC:
            raise WorkloadTraceError(
                "schema v1 is synthetic-only; collected traces require schema v2"
            )
        for name, value in (
            ("request.request_id", self.request_id),
            ("request.dataset", self.dataset),
            ("request.dataset_id", self.dataset_id),
            ("request.difficulty", self.difficulty),
        ):
            _require_nonempty(name, value)
        _require_positive_integer("request.prompt_tokens", self.prompt_tokens)
        _require_nonnegative_integer("request.image_tokens", self.image_tokens)
        _require_positive_integer("request.search_width", self.search_width)
        _require_nonnegative_integer("request.seed", self.seed)

        if self.modality is Modality.TEXT:
            if self.image_tokens != 0:
                raise WorkloadTraceError("text requests must have image_tokens=0")
            if self.question_length_bucket is not None:
                raise WorkloadTraceError(
                    "text requests must not define question_length_bucket"
                )
        else:
            if self.image_tokens == 0:
                raise WorkloadTraceError("vision requests must have positive image_tokens")
            if self.question_length_bucket is None:
                raise WorkloadTraceError(
                    "vision requests must define question_length_bucket"
                )

        if not self.steps:
            raise WorkloadTraceError("request.steps must not be empty")
        expected_shared_tokens = self.prompt_tokens + self.image_tokens
        expected_parent: str | None = None
        all_candidate_ids: set[str] = set()

        for expected_index, step in enumerate(self.steps):
            if step.step_index != expected_index:
                raise WorkloadTraceError(
                    "step indices must be contiguous and start at zero"
                )
            if len(step.candidates) != self.search_width:
                raise WorkloadTraceError(
                    f"step {step.step_index} has {len(step.candidates)} candidates, "
                    f"expected search_width={self.search_width}"
                )
            if step.shared_kv_tokens != expected_shared_tokens:
                raise WorkloadTraceError(
                    f"step {step.step_index} shared_kv_tokens is "
                    f"{step.shared_kv_tokens}, expected {expected_shared_tokens}"
                )

            for candidate in step.candidates:
                if candidate.candidate_id in all_candidate_ids:
                    raise WorkloadTraceError(
                        f"candidate ID {candidate.candidate_id!r} is not request-unique"
                    )
                all_candidate_ids.add(candidate.candidate_id)
                if candidate.parent_candidate_id != expected_parent:
                    raise WorkloadTraceError(
                        f"candidate {candidate.candidate_id!r} has parent "
                        f"{candidate.parent_candidate_id!r}, expected {expected_parent!r}"
                    )

            selected = step.selected_candidate
            expected_shared_tokens += selected.unique_kv_tokens
            expected_parent = selected.candidate_id

    @property
    def candidate_count(self) -> int:
        return sum(len(step.candidates) for step in self.steps)

    @property
    def generated_tokens(self) -> int:
        return sum(
            candidate.generated_tokens
            for step in self.steps
            for candidate in step.candidates
        )

    @property
    def selected_path_tokens(self) -> int:
        return sum(step.selected_candidate.generated_tokens for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema_version": self.trace_schema_version,
            "request_id": self.request_id,
            "dataset": self.dataset,
            "dataset_id": self.dataset_id,
            "modality": self.modality.value,
            "difficulty": self.difficulty,
            "question_length_bucket": (
                self.question_length_bucket.value
                if self.question_length_bucket is not None
                else None
            ),
            "prompt_tokens": self.prompt_tokens,
            "image_tokens": self.image_tokens,
            "search_width": self.search_width,
            "seed": self.seed,
            "sampling": self.sampling.to_dict(),
            "provenance": self.provenance.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }
