"""Paper-facing workload schema with exact token and logical KV lineage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from ..errors import WorkloadTraceError
from .schema import (
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
)


TRACE_SCHEMA_VERSION_V2 = 2


def _require_nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise WorkloadTraceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadTraceError(f"{name} must be a non-negative integer, got {value!r}")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadTraceError(f"{name} must be a positive integer, got {value!r}")


def _require_unique(name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise WorkloadTraceError(f"{name} must not contain duplicates")


class VerifierKind(str, Enum):
    """Verifier behavior represented by one recorded model call."""

    SCALAR_PRM = "scalar_prm"
    PAIRWISE_JUDGE = "pairwise_judge"


class SelectionKind(str, Enum):
    """Branch-selection operation represented by one ordered event."""

    TOP_K = "top_k"
    PAIRWISE = "pairwise"


@dataclass(frozen=True)
class TraceProvenanceV2:
    """Frozen software and model identities for a schema-v2 collection."""

    source_kind: SourceKind
    pipeline: str
    pipeline_revision: str
    dataset_revision: str
    tokenizer: ModelRef
    policy_model: ModelRef
    verifier_models: tuple[ModelRef, ...]
    inference_engine: ModelRef

    def __post_init__(self) -> None:
        _require_nonempty("provenance.pipeline", self.pipeline)
        _require_nonempty("provenance.pipeline_revision", self.pipeline_revision)
        _require_nonempty("provenance.dataset_revision", self.dataset_revision)
        if not self.verifier_models:
            raise WorkloadTraceError("provenance.verifier_models must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "pipeline": self.pipeline,
            "pipeline_revision": self.pipeline_revision,
            "dataset_revision": self.dataset_revision,
            "tokenizer": self.tokenizer.to_dict(),
            "policy_model": self.policy_model.to_dict(),
            "verifier_models": [model.to_dict() for model in self.verifier_models],
            "inference_engine": self.inference_engine.to_dict(),
        }


@dataclass(frozen=True)
class TokenTensorTrace:
    """Exact submitted token IDs/mask plus the model-visible sequence length."""

    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    model_token_count: int

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise WorkloadTraceError("token tensor token_ids must not be empty")
        if len(self.attention_mask) != len(self.token_ids):
            raise WorkloadTraceError(
                "token tensor attention_mask length must equal token_ids length"
            )
        for index, token_id in enumerate(self.token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkloadTraceError(
                    f"token tensor token_ids[{index}] must be an integer"
                )
        for index, mask_value in enumerate(self.attention_mask):
            if isinstance(mask_value, bool) or mask_value not in (0, 1):
                raise WorkloadTraceError(
                    f"token tensor attention_mask[{index}] must be 0 or 1"
                )
        if self.valid_token_count == 0:
            raise WorkloadTraceError("token tensor must contain at least one valid token")
        _require_positive_integer("token tensor model_token_count", self.model_token_count)
        if self.model_token_count < self.valid_token_count:
            raise WorkloadTraceError(
                "token tensor model_token_count must cover every valid token ID"
            )

    @property
    def valid_token_ids(self) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id, mask_value in zip(self.token_ids, self.attention_mask)
            if mask_value == 1
        )

    @property
    def valid_token_count(self) -> int:
        return sum(self.attention_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "attention_mask": list(self.attention_mask),
            "model_token_count": self.model_token_count,
        }


@dataclass(frozen=True)
class KvBlockTraceV2:
    """One append-only logical KV extension in a candidate lineage."""

    block_id: str
    parent_block_id: str | None
    owner_candidate_id: str | None
    token_count: int

    def __post_init__(self) -> None:
        _require_nonempty("kv block.block_id", self.block_id)
        if self.parent_block_id is not None:
            _require_nonempty("kv block.parent_block_id", self.parent_block_id)
        if self.owner_candidate_id is not None:
            _require_nonempty("kv block.owner_candidate_id", self.owner_candidate_id)
        _require_positive_integer("kv block.token_count", self.token_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "parent_block_id": self.parent_block_id,
            "owner_candidate_id": self.owner_candidate_id,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class CandidateTraceV2:
    """One policy candidate with exact tokens and its terminal KV lineage."""

    candidate_id: str
    parent_candidate_id: str | None
    input_tokens: TokenTensorTrace
    generated_token_ids: tuple[int, ...]
    token_ready_indices: tuple[int, ...]
    materialized_output_tokens: int
    reused_kv_tokens: int
    terminal_kv_block_id: str
    finish_reason: str

    def __post_init__(self) -> None:
        _require_nonempty("candidate.candidate_id", self.candidate_id)
        if self.parent_candidate_id is not None:
            _require_nonempty("candidate.parent_candidate_id", self.parent_candidate_id)
        if not self.generated_token_ids:
            raise WorkloadTraceError("candidate.generated_token_ids must not be empty")
        for index, token_id in enumerate(self.generated_token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkloadTraceError(
                    f"candidate.generated_token_ids[{index}] must be an integer"
                )
        if len(self.token_ready_indices) != len(self.generated_token_ids):
            raise WorkloadTraceError(
                "candidate.token_ready_indices length must equal generated_token_ids length"
            )
        previous = -1
        for index, ready_index in enumerate(self.token_ready_indices):
            _require_nonnegative_integer(
                f"candidate.token_ready_indices[{index}]", ready_index
            )
            if ready_index <= previous:
                raise WorkloadTraceError(
                    "candidate.token_ready_indices must be strictly increasing"
                )
            previous = ready_index
        _require_nonnegative_integer(
            "candidate.materialized_output_tokens", self.materialized_output_tokens
        )
        if self.materialized_output_tokens > len(self.generated_token_ids):
            raise WorkloadTraceError(
                "candidate.materialized_output_tokens cannot exceed generated tokens"
            )
        _require_nonnegative_integer("candidate.reused_kv_tokens", self.reused_kv_tokens)
        if self.reused_kv_tokens > self.input_tokens.model_token_count:
            raise WorkloadTraceError(
                "candidate.reused_kv_tokens cannot exceed model input tokens"
            )
        _require_nonempty("candidate.terminal_kv_block_id", self.terminal_kv_block_id)
        _require_nonempty("candidate.finish_reason", self.finish_reason)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def full_output_token_ids(self) -> tuple[int, ...]:
        return self.input_tokens.valid_token_ids + self.generated_token_ids

    @property
    def materialized_kv_tokens(self) -> int:
        return self.input_tokens.model_token_count + self.materialized_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "input_tokens": self.input_tokens.to_dict(),
            "generated_token_ids": list(self.generated_token_ids),
            "token_ready_indices": list(self.token_ready_indices),
            "materialized_output_tokens": self.materialized_output_tokens,
            "reused_kv_tokens": self.reused_kv_tokens,
            "terminal_kv_block_id": self.terminal_kv_block_id,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class VerifierCallTraceV2:
    """One scalar-PRM or pairwise-judge model invocation."""

    call_id: str
    kind: VerifierKind
    stage: str
    candidate_ids: tuple[str, ...]
    input_tokens: TokenTensorTrace
    generated_token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    winner_candidate_id: str | None

    def __post_init__(self) -> None:
        _require_nonempty("verifier call.call_id", self.call_id)
        _require_nonempty("verifier call.stage", self.stage)
        if not self.candidate_ids:
            raise WorkloadTraceError("verifier call.candidate_ids must not be empty")
        _require_unique("verifier call.candidate_ids", self.candidate_ids)
        for candidate_id in self.candidate_ids:
            _require_nonempty("verifier call candidate ID", candidate_id)
        for index, token_id in enumerate(self.generated_token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkloadTraceError(
                    f"verifier call.generated_token_ids[{index}] must be an integer"
                )
        for score in self.scores:
            if not isfinite(score):
                raise WorkloadTraceError("verifier call scores must be finite")

        if self.kind is VerifierKind.SCALAR_PRM:
            if len(self.scores) != len(self.candidate_ids):
                raise WorkloadTraceError(
                    "scalar PRM scores must match verifier candidate IDs"
                )
            if self.winner_candidate_id is not None:
                raise WorkloadTraceError("scalar PRM call must not declare a winner")
            if self.generated_token_ids:
                raise WorkloadTraceError(
                    "scalar PRM call must not contain generated token IDs"
                )
        else:
            if len(self.candidate_ids) != 2:
                raise WorkloadTraceError(
                    "pairwise judge must compare exactly two candidates"
                )
            if self.scores:
                raise WorkloadTraceError("pairwise judge must not invent scalar scores")
            if self.winner_candidate_id not in self.candidate_ids:
                raise WorkloadTraceError(
                    "pairwise judge winner must be one of its candidates"
                )
            if not self.generated_token_ids:
                raise WorkloadTraceError(
                    "pairwise judge must record its generated decision tokens"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "kind": self.kind.value,
            "stage": self.stage,
            "candidate_ids": list(self.candidate_ids),
            "input_tokens": self.input_tokens.to_dict(),
            "generated_token_ids": list(self.generated_token_ids),
            "scores": list(self.scores),
            "winner_candidate_id": self.winner_candidate_id,
        }


@dataclass(frozen=True)
class SelectionEventTraceV2:
    """One ordered prune/retain decision and the KV state after it."""

    event_index: int
    kind: SelectionKind
    verifier_call_ids: tuple[str, ...]
    considered_candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    pruned_candidate_ids: tuple[str, ...]
    live_candidate_ids_after: tuple[str, ...]
    retained_kv_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("selection.event_index", self.event_index)
        for name, values in (
            ("selection.verifier_call_ids", self.verifier_call_ids),
            ("selection.considered_candidate_ids", self.considered_candidate_ids),
            ("selection.selected_candidate_ids", self.selected_candidate_ids),
            ("selection.live_candidate_ids_after", self.live_candidate_ids_after),
            ("selection.retained_kv_block_ids", self.retained_kv_block_ids),
        ):
            if not values:
                raise WorkloadTraceError(f"{name} must not be empty")
            _require_unique(name, values)
        _require_unique("selection.pruned_candidate_ids", self.pruned_candidate_ids)
        considered = set(self.considered_candidate_ids)
        selected = set(self.selected_candidate_ids)
        pruned = set(self.pruned_candidate_ids)
        if selected & pruned or selected | pruned != considered:
            raise WorkloadTraceError(
                "selection selected/pruned candidates must partition considered candidates"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "kind": self.kind.value,
            "verifier_call_ids": list(self.verifier_call_ids),
            "considered_candidate_ids": list(self.considered_candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "pruned_candidate_ids": list(self.pruned_candidate_ids),
            "live_candidate_ids_after": list(self.live_candidate_ids_after),
            "retained_kv_block_ids": list(self.retained_kv_block_ids),
        }


@dataclass(frozen=True)
class StepTraceV2:
    """One expansion step with ordered verifier and selection events."""

    step_index: int
    candidates: tuple[CandidateTraceV2, ...]
    verifier_calls: tuple[VerifierCallTraceV2, ...]
    selection_events: tuple[SelectionEventTraceV2, ...]
    selected_candidate_ids: tuple[str, ...]
    pruned_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("step.step_index", self.step_index)
        if not self.candidates:
            raise WorkloadTraceError("step.candidates must not be empty")
        if not self.verifier_calls:
            raise WorkloadTraceError("step.verifier_calls must not be empty")
        if not self.selection_events:
            raise WorkloadTraceError("step.selection_events must not be empty")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        call_ids = tuple(call.call_id for call in self.verifier_calls)
        _require_unique("step candidate IDs", candidate_ids)
        _require_unique("step verifier call IDs", call_ids)
        _require_unique("step.selected_candidate_ids", self.selected_candidate_ids)
        _require_unique("step.pruned_candidate_ids", self.pruned_candidate_ids)
        if not self.selected_candidate_ids:
            raise WorkloadTraceError("step.selected_candidate_ids must not be empty")
        if set(self.selected_candidate_ids) | set(self.pruned_candidate_ids) != set(
            candidate_ids
        ) or set(self.selected_candidate_ids) & set(self.pruned_candidate_ids):
            raise WorkloadTraceError(
                "step selected/pruned candidates must partition all candidates"
            )

    @property
    def selected_candidates(self) -> tuple[CandidateTraceV2, ...]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return tuple(by_id[candidate_id] for candidate_id in self.selected_candidate_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "verifier_calls": [call.to_dict() for call in self.verifier_calls],
            "selection_events": [event.to_dict() for event in self.selection_events],
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "pruned_candidate_ids": list(self.pruned_candidate_ids),
        }


@dataclass(frozen=True)
class TtcRequestTraceV2:
    """A complete exact-token TTC tree suitable for paper workload replay."""

    trace_schema_version: int
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
    sampling: SamplingConfig
    provenance: TraceProvenanceV2
    root_input: TokenTensorTrace
    kv_blocks: tuple[KvBlockTraceV2, ...]
    steps: tuple[StepTraceV2, ...]

    def __post_init__(self) -> None:
        self._validate_request_fields()
        block_by_id, block_totals, root_block_id = self._validate_kv_blocks()
        self._validate_control_flow(block_by_id, block_totals, root_block_id)

    def _validate_request_fields(self) -> None:
        if self.trace_schema_version != TRACE_SCHEMA_VERSION_V2:
            raise WorkloadTraceError(
                f"unsupported trace schema version {self.trace_schema_version!r}"
            )
        for name, value in (
            ("request.request_id", self.request_id),
            ("request.dataset", self.dataset),
            ("request.dataset_id", self.dataset_id),
            ("request.difficulty", self.difficulty),
        ):
            _require_nonempty(name, value)
        _require_nonnegative_integer("request.image_tokens", self.image_tokens)
        _require_positive_integer("request.search_width", self.search_width)
        _require_positive_integer("request.beam_size", self.beam_size)
        _require_nonnegative_integer("request.seed", self.seed)
        if self.modality is Modality.TEXT:
            if self.image_tokens != 0 or self.question_length_bucket is not None:
                raise WorkloadTraceError(
                    "text requests require image_tokens=0 and no length bucket"
                )
            if self.root_input.model_token_count != self.root_input.valid_token_count:
                raise WorkloadTraceError(
                    "text root model_token_count must equal valid token IDs"
                )
        elif self.image_tokens == 0 or self.question_length_bucket is None:
            raise WorkloadTraceError(
                "vision requests require image tokens and a length bucket"
            )
        if not self.steps:
            raise WorkloadTraceError("request.steps must not be empty")

    def _validate_kv_blocks(
        self,
    ) -> tuple[dict[str, KvBlockTraceV2], dict[str, int], str]:
        if not self.kv_blocks:
            raise WorkloadTraceError("request.kv_blocks must not be empty")
        block_by_id = {block.block_id: block for block in self.kv_blocks}
        if len(block_by_id) != len(self.kv_blocks):
            raise WorkloadTraceError("KV block IDs must be request-unique")
        roots = [block for block in self.kv_blocks if block.parent_block_id is None]
        if len(roots) != 1 or roots[0].owner_candidate_id is not None:
            raise WorkloadTraceError(
                "KV lineage must have one request root owned by no candidate"
            )
        root = roots[0]
        if root.token_count != self.root_input.model_token_count:
            raise WorkloadTraceError(
                "root KV token count must equal root model input token count"
            )

        totals: dict[str, int] = {}
        visiting: set[str] = set()

        def total(block_id: str) -> int:
            if block_id in totals:
                return totals[block_id]
            if block_id in visiting:
                raise WorkloadTraceError("KV block lineage must not contain cycles")
            block = block_by_id.get(block_id)
            if block is None:
                raise WorkloadTraceError(f"unknown KV block {block_id!r}")
            visiting.add(block_id)
            parent_total = (
                0 if block.parent_block_id is None else total(block.parent_block_id)
            )
            visiting.remove(block_id)
            totals[block_id] = parent_total + block.token_count
            return totals[block_id]

        for block_id in block_by_id:
            total(block_id)
        return block_by_id, totals, root.block_id

    @staticmethod
    def _lineage(block_by_id: dict[str, KvBlockTraceV2], block_id: str) -> set[str]:
        lineage: set[str] = set()
        current: str | None = block_id
        while current is not None:
            if current in lineage:
                raise WorkloadTraceError("KV block lineage must not contain cycles")
            lineage.add(current)
            current = block_by_id[current].parent_block_id
        return lineage

    def _validate_control_flow(
        self,
        block_by_id: dict[str, KvBlockTraceV2],
        block_totals: dict[str, int],
        root_block_id: str,
    ) -> None:
        all_candidates: dict[str, CandidateTraceV2] = {}
        all_call_ids: set[str] = set()
        all_token_ready_indices: list[int] = []
        previous_step_ready_index = -1
        previous_selected_ids: tuple[str, ...] = ()

        for expected_step_index, step in enumerate(self.steps):
            if step.step_index != expected_step_index:
                raise WorkloadTraceError(
                    "step indices must be contiguous and start at zero"
                )
            if len(step.selected_candidate_ids) > self.beam_size:
                raise WorkloadTraceError(
                    f"step {step.step_index} retains more candidates than beam_size"
                )

            expected_parent_ids: tuple[str | None, ...] = (
                (None,) if expected_step_index == 0 else previous_selected_ids
            )
            parent_counts = {parent_id: 0 for parent_id in expected_parent_ids}
            step_candidate_ids = {candidate.candidate_id for candidate in step.candidates}

            for candidate in step.candidates:
                if candidate.candidate_id in all_candidates:
                    raise WorkloadTraceError(
                        f"candidate ID {candidate.candidate_id!r} is not request-unique"
                    )
                if candidate.parent_candidate_id not in parent_counts:
                    raise WorkloadTraceError(
                        f"candidate {candidate.candidate_id!r} has an unselected parent"
                    )
                parent_counts[candidate.parent_candidate_id] += 1
                self._validate_candidate_lineage(
                    candidate,
                    all_candidates,
                    block_by_id,
                    block_totals,
                    root_block_id,
                )
                all_token_ready_indices.extend(candidate.token_ready_indices)
                all_candidates[candidate.candidate_id] = candidate

            if any(count != self.search_width for count in parent_counts.values()):
                raise WorkloadTraceError(
                    f"step {step.step_index} must generate search_width candidates per parent"
                )
            step_ready_indices = [
                ready_index
                for candidate in step.candidates
                for ready_index in candidate.token_ready_indices
            ]
            if min(step_ready_indices) <= previous_step_ready_index:
                raise WorkloadTraceError(
                    "token-ready order must respect generation step dependencies"
                )
            previous_step_ready_index = max(step_ready_indices)

            call_by_id = {call.call_id: call for call in step.verifier_calls}
            for call in step.verifier_calls:
                if call.call_id in all_call_ids:
                    raise WorkloadTraceError(
                        f"verifier call ID {call.call_id!r} is not request-unique"
                    )
                all_call_ids.add(call.call_id)
                if not set(call.candidate_ids) <= step_candidate_ids:
                    raise WorkloadTraceError(
                        f"verifier call {call.call_id!r} references another step"
                    )
                if self.modality is Modality.TEXT and (
                    call.input_tokens.model_token_count
                    != call.input_tokens.valid_token_count
                ):
                    raise WorkloadTraceError(
                        "text verifier model_token_count must equal valid token IDs"
                    )

            self._validate_selection_events(
                step,
                call_by_id,
                block_by_id,
                all_candidates,
            )
            previous_selected_ids = step.selected_candidate_ids

        if sorted(all_token_ready_indices) != list(range(len(all_token_ready_indices))):
            raise WorkloadTraceError(
                "token_ready_indices must form one request-global contiguous order"
            )

        candidate_ids = set(all_candidates)
        for block in self.kv_blocks:
            if block.owner_candidate_id is None:
                if block.block_id != root_block_id:
                    raise WorkloadTraceError(
                        f"non-root KV block {block.block_id!r} must have an owner"
                    )
                continue
            if block.owner_candidate_id not in candidate_ids:
                raise WorkloadTraceError(
                    f"KV block {block.block_id!r} has unknown owner candidate"
                )
            owner = all_candidates[block.owner_candidate_id]
            if owner.terminal_kv_block_id != block.block_id:
                raise WorkloadTraceError(
                    f"KV block {block.block_id!r} is not its owner's terminal extension"
                )

    def _validate_candidate_lineage(
        self,
        candidate: CandidateTraceV2,
        prior_candidates: dict[str, CandidateTraceV2],
        block_by_id: dict[str, KvBlockTraceV2],
        block_totals: dict[str, int],
        root_block_id: str,
    ) -> None:
        if candidate.parent_candidate_id is None:
            expected_prefix = self.root_input.valid_token_ids
            parent_block_id = root_block_id
        else:
            parent = prior_candidates[candidate.parent_candidate_id]
            expected_prefix = parent.full_output_token_ids
            parent_block_id = parent.terminal_kv_block_id

        input_ids = candidate.input_tokens.valid_token_ids
        if input_ids[: len(expected_prefix)] != expected_prefix:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} input does not preserve parent token IDs"
            )
        parent_kv_tokens = block_totals[parent_block_id]
        if candidate.reused_kv_tokens != parent_kv_tokens:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} reused_kv_tokens does not "
                "match parent lineage"
            )
        if self.modality is Modality.TEXT and (
            candidate.input_tokens.model_token_count
            != candidate.input_tokens.valid_token_count
        ):
            raise WorkloadTraceError(
                "text candidate model_token_count must equal valid token IDs"
            )

        terminal_id = candidate.terminal_kv_block_id
        if terminal_id not in block_by_id:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} references unknown terminal KV block"
            )
        expected_total = candidate.materialized_kv_tokens
        extension_tokens = expected_total - parent_kv_tokens
        if extension_tokens < 0:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} materialized fewer KV tokens "
                "than its parent"
            )
        if extension_tokens == 0:
            if terminal_id != parent_block_id:
                raise WorkloadTraceError(
                    f"candidate {candidate.candidate_id!r} must retain its parent KV block"
                )
            return

        terminal = block_by_id[terminal_id]
        if (
            terminal.owner_candidate_id != candidate.candidate_id
            or terminal.parent_block_id != parent_block_id
            or terminal.token_count != extension_tokens
            or block_totals[terminal_id] != expected_total
        ):
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} terminal KV extension is inconsistent"
            )

    def _validate_selection_events(
        self,
        step: StepTraceV2,
        call_by_id: dict[str, VerifierCallTraceV2],
        block_by_id: dict[str, KvBlockTraceV2],
        all_candidates: dict[str, CandidateTraceV2],
    ) -> None:
        live = {candidate.candidate_id for candidate in step.candidates}
        pruned_in_order: list[str] = []
        used_selection_call_ids: set[str] = set()
        for expected_index, event in enumerate(step.selection_events):
            if event.event_index != expected_index:
                raise WorkloadTraceError(
                    "selection event indices must be contiguous and start at zero"
                )
            if not set(event.verifier_call_ids) <= set(call_by_id):
                raise WorkloadTraceError("selection event references unknown verifier call")
            if used_selection_call_ids & set(event.verifier_call_ids):
                raise WorkloadTraceError(
                    "one verifier call must not drive multiple selection events"
                )
            used_selection_call_ids.update(event.verifier_call_ids)
            event_calls = tuple(call_by_id[call_id] for call_id in event.verifier_call_ids)
            considered = set(event.considered_candidate_ids)
            pruned = set(event.pruned_candidate_ids)
            selected = set(event.selected_candidate_ids)
            if event.kind is SelectionKind.PAIRWISE:
                if (
                    len(event_calls) != 1
                    or event_calls[0].kind is not VerifierKind.PAIRWISE_JUDGE
                ):
                    raise WorkloadTraceError(
                        "pairwise selection must reference one pairwise judge call"
                    )
                call = event_calls[0]
                if (
                    considered != set(call.candidate_ids)
                    or selected != {call.winner_candidate_id}
                    or pruned != set(call.candidate_ids) - {call.winner_candidate_id}
                ):
                    raise WorkloadTraceError(
                        "pairwise selection must follow the recorded judge winner"
                    )
            else:
                if any(call.kind is not VerifierKind.SCALAR_PRM for call in event_calls):
                    raise WorkloadTraceError(
                        "top-k selection must reference scalar PRM calls"
                    )
                scored_candidates = {
                    candidate_id
                    for call in event_calls
                    for candidate_id in call.candidate_ids
                }
                if not considered <= scored_candidates:
                    raise WorkloadTraceError(
                        "top-k selection candidates must have recorded scalar scores"
                    )
            if not considered <= live:
                raise WorkloadTraceError(
                    "selection event considers a candidate that is no longer live"
                )
            if not selected <= live - pruned:
                raise WorkloadTraceError(
                    "selection event selects a candidate that was pruned"
                )
            live -= pruned
            if set(event.live_candidate_ids_after) != live:
                raise WorkloadTraceError(
                    "selection live_candidate_ids_after does not match ordered pruning"
                )
            expected_blocks: set[str] = set()
            for candidate_id in live:
                expected_blocks |= self._lineage(
                    block_by_id,
                    all_candidates[candidate_id].terminal_kv_block_id,
                )
            if set(event.retained_kv_block_ids) != expected_blocks:
                raise WorkloadTraceError(
                    "selection retained_kv_block_ids does not match live KV lineages"
                )
            pruned_in_order.extend(event.pruned_candidate_ids)

        if live != set(step.selected_candidate_ids):
            raise WorkloadTraceError(
                "final live candidates must equal step.selected_candidate_ids"
            )
        if (
            step.selection_events[-1].live_candidate_ids_after
            != step.selected_candidate_ids
        ):
            raise WorkloadTraceError(
                "final live candidate order must equal selected candidate order"
            )
        if tuple(pruned_in_order) != step.pruned_candidate_ids:
            raise WorkloadTraceError(
                "step pruned candidates must preserve selection-event order"
            )

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
        return sum(
            candidate.generated_tokens
            for step in self.steps
            for candidate in step.selected_candidates
        )

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
            "image_tokens": self.image_tokens,
            "search_width": self.search_width,
            "beam_size": self.beam_size,
            "seed": self.seed,
            "sampling": self.sampling.to_dict(),
            "provenance": self.provenance.to_dict(),
            "root_input": self.root_input.to_dict(),
            "kv_blocks": [block.to_dict() for block in self.kv_blocks],
            "steps": [step.to_dict() for step in self.steps],
        }
