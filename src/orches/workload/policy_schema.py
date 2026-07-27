"""Exact-token generation-only trace contract for policy-model collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..errors import WorkloadTraceError
from .schema import (
    Modality,
    ModelRef,
    QuestionLengthBucket,
    SamplingConfig,
    SourceKind,
)
from .schema_v2 import KvBlockTraceV2, TokenTensorTrace


POLICY_TRACE_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise WorkloadTraceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkloadTraceError(
            f"{name} must be a non-negative integer, got {value!r}"
        )


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadTraceError(f"{name} must be a positive integer, got {value!r}")


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise WorkloadTraceError(f"{name} must be a lowercase SHA-256 digest")


class WorkloadScope(str, Enum):
    """Experiment interval represented by a workload artifact."""

    GENERATION_ONLY = "generation_only"


class PolicyCollectionMode(str, Enum):
    """How branch decisions are supplied while collecting policy generation."""

    SINGLE_STEP = "single_step"
    SYNTHETIC_SELECTOR = "synthetic_selector"
    OPAQUE_SELECTOR = "opaque_selector"


class PolicySelectionKind(str, Enum):
    """Evidence class for one branch decision without verifier internals."""

    SYNTHETIC = "synthetic"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class PolicyTraceProvenance:
    """Frozen identities required to reproduce policy-model generation."""

    source_kind: SourceKind
    pipeline: str
    pipeline_revision: str
    dataset_revision: str
    tokenizer: ModelRef
    policy_model: ModelRef
    inference_engine: ModelRef
    collector: ModelRef
    selector: ModelRef | None

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
            "inference_engine": self.inference_engine.to_dict(),
            "collector": self.collector.to_dict(),
            "selector": None if self.selector is None else self.selector.to_dict(),
        }


@dataclass(frozen=True)
class PolicyCandidateTrace:
    """One returned policy sequence and the KV it actually materialized."""

    candidate_id: str
    generated_token_ids: tuple[int, ...]
    token_ready_indices: tuple[int, ...]
    materialized_output_tokens: int
    terminal_kv_block_id: str
    finish_reason: str

    def __post_init__(self) -> None:
        _require_nonempty("candidate.candidate_id", self.candidate_id)
        if not self.generated_token_ids:
            raise WorkloadTraceError("candidate.generated_token_ids must not be empty")
        for index, token_id in enumerate(self.generated_token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise WorkloadTraceError(
                    f"candidate.generated_token_ids[{index}] must be an integer"
                )
        if len(self.token_ready_indices) != len(self.generated_token_ids):
            raise WorkloadTraceError(
                "candidate.token_ready_indices length must equal "
                "generated_token_ids length"
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
        _require_nonempty("candidate.terminal_kv_block_id", self.terminal_kv_block_id)
        _require_nonempty("candidate.finish_reason", self.finish_reason)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generated_token_ids": list(self.generated_token_ids),
            "token_ready_indices": list(self.token_ready_indices),
            "materialized_output_tokens": self.materialized_output_tokens,
            "terminal_kv_block_id": self.terminal_kv_block_id,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class PolicyGenerationCallTrace:
    """One inference-engine call for one root or selected parent."""

    call_id: str
    parent_candidate_id: str | None
    input_tokens: TokenTensorTrace
    reused_kv_tokens: int
    rng_seed: int
    rng_stream_id: str
    candidates: tuple[PolicyCandidateTrace, ...]

    def __post_init__(self) -> None:
        _require_nonempty("generation call.call_id", self.call_id)
        if self.parent_candidate_id is not None:
            _require_nonempty(
                "generation call.parent_candidate_id", self.parent_candidate_id
            )
        _require_nonnegative_integer(
            "generation call.reused_kv_tokens", self.reused_kv_tokens
        )
        _require_nonnegative_integer("generation call.rng_seed", self.rng_seed)
        _require_nonempty("generation call.rng_stream_id", self.rng_stream_id)
        if not self.candidates:
            raise WorkloadTraceError("generation call.candidates must not be empty")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise WorkloadTraceError(
                "candidate IDs must be unique within a generation call"
            )

    def full_output_token_ids(
        self, candidate: PolicyCandidateTrace
    ) -> tuple[int, ...]:
        return self.input_tokens.valid_token_ids + candidate.generated_token_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "parent_candidate_id": self.parent_candidate_id,
            "input_tokens": self.input_tokens.to_dict(),
            "reused_kv_tokens": self.reused_kv_tokens,
            "rng_seed": self.rng_seed,
            "rng_stream_id": self.rng_stream_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class PolicySelectionTrace:
    """One branch decision retained without pretending to contain PRM evidence."""

    decision_id: str
    kind: PolicySelectionKind
    selected_candidate_ids: tuple[str, ...]
    pruned_candidate_ids: tuple[str, ...]
    decision_artifact_sha256: str | None

    def __post_init__(self) -> None:
        _require_nonempty("selection.decision_id", self.decision_id)
        if not self.selected_candidate_ids:
            raise WorkloadTraceError(
                "selection.selected_candidate_ids must not be empty"
            )
        for name, values in (
            ("selection.selected_candidate_ids", self.selected_candidate_ids),
            ("selection.pruned_candidate_ids", self.pruned_candidate_ids),
        ):
            if len(set(values)) != len(values):
                raise WorkloadTraceError(f"{name} must not contain duplicates")
            for value in values:
                _require_nonempty(name, value)
        if set(self.selected_candidate_ids) & set(self.pruned_candidate_ids):
            raise WorkloadTraceError(
                "selection selected and pruned candidates must be disjoint"
            )
        if self.kind is PolicySelectionKind.OPAQUE:
            if self.decision_artifact_sha256 is None:
                raise WorkloadTraceError(
                    "opaque selection requires decision_artifact_sha256"
                )
            _require_sha256(
                "selection.decision_artifact_sha256",
                self.decision_artifact_sha256,
            )
        elif self.decision_artifact_sha256 is not None:
            _require_sha256(
                "selection.decision_artifact_sha256",
                self.decision_artifact_sha256,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "kind": self.kind.value,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "pruned_candidate_ids": list(self.pruned_candidate_ids),
            "decision_artifact_sha256": self.decision_artifact_sha256,
        }


@dataclass(frozen=True)
class PolicyStepTrace:
    """Generation calls and an optional externally supplied branch decision."""

    step_index: int
    generation_calls: tuple[PolicyGenerationCallTrace, ...]
    selection: PolicySelectionTrace | None

    def __post_init__(self) -> None:
        _require_nonnegative_integer("step.step_index", self.step_index)
        if not self.generation_calls:
            raise WorkloadTraceError("step.generation_calls must not be empty")
        call_ids = tuple(call.call_id for call in self.generation_calls)
        if len(set(call_ids)) != len(call_ids):
            raise WorkloadTraceError("generation call IDs must be unique within a step")

    @property
    def candidates(self) -> tuple[PolicyCandidateTrace, ...]:
        return tuple(
            candidate
            for call in self.generation_calls
            for candidate in call.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "generation_calls": [call.to_dict() for call in self.generation_calls],
            "selection": None if self.selection is None else self.selection.to_dict(),
        }


@dataclass(frozen=True)
class PolicyRequestTrace:
    """Generation-only request trace that cannot be mistaken for full TTC evidence."""

    policy_trace_schema_version: int
    workload_scope: WorkloadScope
    paper_eligible: bool
    collection_mode: PolicyCollectionMode
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
    provenance: PolicyTraceProvenance
    root_input: TokenTensorTrace
    kv_blocks: tuple[KvBlockTraceV2, ...]
    steps: tuple[PolicyStepTrace, ...]

    def __post_init__(self) -> None:
        self._validate_request_fields()
        block_by_id, block_totals, root_block_id = self._validate_kv_blocks()
        self._validate_control_flow(block_by_id, block_totals, root_block_id)

    @property
    def generation_evaluation_eligible(self) -> bool:
        return (
            self.provenance.source_kind is SourceKind.COLLECTED
            and self.collection_mode is PolicyCollectionMode.OPAQUE_SELECTOR
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
    def materialized_output_tokens(self) -> int:
        return sum(
            candidate.materialized_output_tokens
            for step in self.steps
            for candidate in step.candidates
        )

    def _validate_request_fields(self) -> None:
        if self.policy_trace_schema_version != POLICY_TRACE_SCHEMA_VERSION:
            raise WorkloadTraceError(
                "unsupported policy trace schema version "
                f"{self.policy_trace_schema_version!r}"
            )
        if self.workload_scope is not WorkloadScope.GENERATION_ONLY:
            raise WorkloadTraceError("policy trace scope must be generation_only")
        if self.paper_eligible is not False:
            raise WorkloadTraceError(
                "generation-only policy traces must set paper_eligible=false"
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
        _require_nonempty("request.dtype", self.dtype)
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
        if self.collection_mode is PolicyCollectionMode.SINGLE_STEP:
            if len(self.steps) != 1 or self.steps[0].selection is not None:
                raise WorkloadTraceError(
                    "single_step mode requires exactly one step without selection"
                )
            if self.provenance.selector is not None:
                raise WorkloadTraceError(
                    "single_step mode must not record a selector"
                )
            return
        if self.provenance.selector is None:
            raise WorkloadTraceError(
                f"{self.collection_mode.value} mode requires selector provenance"
            )
        expected_kind = (
            PolicySelectionKind.SYNTHETIC
            if self.collection_mode is PolicyCollectionMode.SYNTHETIC_SELECTOR
            else PolicySelectionKind.OPAQUE
        )
        if any(
            step.selection is None or step.selection.kind is not expected_kind
            for step in self.steps
        ):
            raise WorkloadTraceError(
                f"{self.collection_mode.value} mode requires one "
                f"{expected_kind.value} selection per step"
            )

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

    def _validate_control_flow(
        self,
        block_by_id: dict[str, KvBlockTraceV2],
        block_totals: dict[str, int],
        root_block_id: str,
    ) -> None:
        candidates: dict[
            str, tuple[PolicyCandidateTrace, PolicyGenerationCallTrace]
        ] = {}
        call_ids: set[str] = set()
        decision_ids: set[str] = set()
        ready_indices: list[int] = []
        previous_step_ready_index = -1
        previous_selected_ids: tuple[str, ...] = ()

        for expected_step_index, step in enumerate(self.steps):
            if step.step_index != expected_step_index:
                raise WorkloadTraceError(
                    "step indices must be contiguous and start at zero"
                )
            expected_parents: tuple[str | None, ...] = (
                (None,) if expected_step_index == 0 else previous_selected_ids
            )
            actual_parents = tuple(
                call.parent_candidate_id for call in step.generation_calls
            )
            if len(set(actual_parents)) != len(actual_parents) or set(
                actual_parents
            ) != set(expected_parents):
                raise WorkloadTraceError(
                    f"step {step.step_index} must contain one generation call "
                    "per selected parent"
                )

            step_candidate_ids: list[str] = []
            step_ready_indices: list[int] = []
            for call in step.generation_calls:
                if call.call_id in call_ids:
                    raise WorkloadTraceError(
                        f"generation call ID {call.call_id!r} is not request-unique"
                    )
                call_ids.add(call.call_id)
                if len(call.candidates) != self.search_width:
                    raise WorkloadTraceError(
                        f"generation call {call.call_id!r} must return "
                        "search_width candidates"
                    )
                parent_block_id, expected_prefix = self._parent_context(
                    call.parent_candidate_id,
                    candidates,
                    root_block_id,
                )
                input_prefix = call.input_tokens.valid_token_ids[
                    : len(expected_prefix)
                ]
                if input_prefix != expected_prefix:
                    raise WorkloadTraceError(
                        f"generation call {call.call_id!r} input does not preserve "
                        "parent token IDs"
                    )
                if call.reused_kv_tokens != block_totals[parent_block_id]:
                    raise WorkloadTraceError(
                        f"generation call {call.call_id!r} reused_kv_tokens does not "
                        "match parent lineage"
                    )
                if self.modality is Modality.TEXT and (
                    call.input_tokens.model_token_count
                    != call.input_tokens.valid_token_count
                ):
                    raise WorkloadTraceError(
                        "text generation model_token_count must equal valid token IDs"
                    )

                for candidate in call.candidates:
                    if candidate.candidate_id in candidates:
                        raise WorkloadTraceError(
                            f"candidate ID {candidate.candidate_id!r} is not "
                            "request-unique"
                        )
                    self._validate_candidate_kv(
                        candidate,
                        call,
                        parent_block_id,
                        block_by_id,
                    )
                    candidates[candidate.candidate_id] = (candidate, call)
                    step_candidate_ids.append(candidate.candidate_id)
                    step_ready_indices.extend(candidate.token_ready_indices)

            if min(step_ready_indices) <= previous_step_ready_index:
                raise WorkloadTraceError(
                    "token-ready order must respect generation step dependencies"
                )
            previous_step_ready_index = max(step_ready_indices)
            ready_indices.extend(step_ready_indices)

            if step.selection is None:
                previous_selected_ids = ()
                continue
            selection = step.selection
            if selection.decision_id in decision_ids:
                raise WorkloadTraceError(
                    f"selection decision ID {selection.decision_id!r} is not "
                    "request-unique"
                )
            decision_ids.add(selection.decision_id)
            selected = set(selection.selected_candidate_ids)
            pruned = set(selection.pruned_candidate_ids)
            if selected | pruned != set(step_candidate_ids):
                raise WorkloadTraceError(
                    "selection selected/pruned candidates must partition the step"
                )
            if len(selection.selected_candidate_ids) > self.beam_size:
                raise WorkloadTraceError(
                    f"step {step.step_index} retains more candidates than beam_size"
                )
            previous_selected_ids = selection.selected_candidate_ids

        if sorted(ready_indices) != list(range(len(ready_indices))):
            raise WorkloadTraceError(
                "token_ready_indices must form one request-global contiguous order"
            )

        candidate_ids = set(candidates)
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
            owner = candidates[block.owner_candidate_id][0]
            if owner.terminal_kv_block_id != block.block_id:
                raise WorkloadTraceError(
                    f"KV block {block.block_id!r} is not its owner's terminal extension"
                )

    def _parent_context(
        self,
        parent_candidate_id: str | None,
        candidates: dict[
            str, tuple[PolicyCandidateTrace, PolicyGenerationCallTrace]
        ],
        root_block_id: str,
    ) -> tuple[str, tuple[int, ...]]:
        if parent_candidate_id is None:
            return root_block_id, self.root_input.valid_token_ids
        parent, parent_call = candidates[parent_candidate_id]
        return (
            parent.terminal_kv_block_id,
            parent_call.full_output_token_ids(parent),
        )

    @staticmethod
    def _validate_candidate_kv(
        candidate: PolicyCandidateTrace,
        call: PolicyGenerationCallTrace,
        parent_block_id: str,
        block_by_id: dict[str, KvBlockTraceV2],
    ) -> None:
        extension_tokens = (
            call.input_tokens.model_token_count
            + candidate.materialized_output_tokens
            - call.reused_kv_tokens
        )
        if extension_tokens < 0:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} materialized fewer KV tokens "
                "than its reused parent lineage"
            )
        if extension_tokens == 0:
            if candidate.terminal_kv_block_id != parent_block_id:
                raise WorkloadTraceError(
                    f"candidate {candidate.candidate_id!r} with no KV extension "
                    "must reuse its parent terminal block"
                )
            return
        block = block_by_id.get(candidate.terminal_kv_block_id)
        if block is None:
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} references unknown "
                "terminal KV block"
            )
        if (
            block.parent_block_id != parent_block_id
            or block.owner_candidate_id != candidate.candidate_id
            or block.token_count != extension_tokens
        ):
            raise WorkloadTraceError(
                f"candidate {candidate.candidate_id!r} terminal KV extension "
                "does not match exact input and materialized output tokens"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_trace_schema_version": self.policy_trace_schema_version,
            "workload_scope": self.workload_scope.value,
            "paper_eligible": self.paper_eligible,
            "collection_mode": self.collection_mode.value,
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
            "provenance": self.provenance.to_dict(),
            "root_input": self.root_input.to_dict(),
            "kv_blocks": [block.to_dict() for block in self.kv_blocks],
            "steps": [step.to_dict() for step in self.steps],
        }
