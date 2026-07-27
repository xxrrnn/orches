"""Deterministic synthetic TTC traces for simulator development only."""

from __future__ import annotations

import random
from dataclasses import dataclass

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


SYNTHETIC_REVISION = "orches-synthetic-v1"


@dataclass(frozen=True)
class SyntheticTraceConfig:
    """Inputs that fully determine a synthetic trace collection."""

    request_count: int = 1
    step_count: int = 3
    search_width: int = 4
    prompt_tokens: int = 128
    min_generated_tokens: int = 4
    max_generated_tokens: int = 12
    seed: int = 0
    modality: Modality = Modality.TEXT
    image_tokens: int = 0
    question_length_bucket: QuestionLengthBucket | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("request_count", self.request_count),
            ("step_count", self.step_count),
            ("search_width", self.search_width),
            ("prompt_tokens", self.prompt_tokens),
            ("min_generated_tokens", self.min_generated_tokens),
            ("max_generated_tokens", self.max_generated_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkloadTraceError(f"synthetic.{name} must be a positive integer")
        if self.min_generated_tokens > self.max_generated_tokens:
            raise WorkloadTraceError(
                "synthetic.min_generated_tokens must not exceed max_generated_tokens"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise WorkloadTraceError("synthetic.seed must be a non-negative integer")
        if self.modality is Modality.TEXT:
            if self.image_tokens != 0 or self.question_length_bucket is not None:
                raise WorkloadTraceError(
                    "synthetic text traces require image_tokens=0 and no length bucket"
                )
        elif self.image_tokens <= 0 or self.question_length_bucket is None:
            raise WorkloadTraceError(
                "synthetic vision traces require image tokens and a length bucket"
            )


def _provenance() -> TraceProvenance:
    synthetic_model = ModelRef(name="synthetic-llm", revision=SYNTHETIC_REVISION)
    return TraceProvenance(
        source_kind=SourceKind.SYNTHETIC,
        pipeline="orches.synthetic",
        pipeline_revision=SYNTHETIC_REVISION,
        dataset_revision=SYNTHETIC_REVISION,
        tokenizer=ModelRef(name="synthetic-tokenizer", revision=SYNTHETIC_REVISION),
        policy_model=synthetic_model,
        small_prm_model=ModelRef(
            name="synthetic-small-prm", revision=SYNTHETIC_REVISION
        ),
        large_prm_model=ModelRef(
            name="synthetic-large-prm", revision=SYNTHETIC_REVISION
        ),
    )


def _timestamps(rng: random.Random, token_count: int) -> tuple[float, ...]:
    elapsed = 0.0
    timestamps: list[float] = []
    for _ in range(token_count):
        elapsed += rng.uniform(35.0, 65.0)
        timestamps.append(round(elapsed, 6))
    return tuple(timestamps)


def generate_synthetic_traces(config: SyntheticTraceConfig) -> list[TtcRequestTrace]:
    """Generate byte-reproducible control-flow traces from one seed."""

    traces: list[TtcRequestTrace] = []
    for request_index in range(config.request_count):
        request_seed = config.seed + request_index
        rng = random.Random(request_seed)
        shared_tokens = config.prompt_tokens + config.image_tokens
        parent_candidate_id: str | None = None
        steps: list[StepTrace] = []

        for step_index in range(config.step_count):
            candidates: list[CandidateTrace] = []
            for candidate_index in range(config.search_width):
                candidate_id = (
                    f"synthetic-{request_index:04d}-s{step_index:03d}-"
                    f"c{candidate_index:03d}"
                )
                generated_tokens = rng.randint(
                    config.min_generated_tokens, config.max_generated_tokens
                )
                small_score = rng.uniform(-1.0, 1.0)
                large_score = small_score + rng.uniform(-0.35, 0.35)
                candidates.append(
                    CandidateTrace(
                        candidate_id=candidate_id,
                        parent_candidate_id=parent_candidate_id,
                        generated_tokens=generated_tokens,
                        token_timestamps_us=_timestamps(rng, generated_tokens),
                        small_prm_score=round(small_score, 8),
                        large_prm_score=round(large_score, 8),
                        unique_kv_tokens=generated_tokens,
                    )
                )

            selected = max(
                candidates,
                key=lambda candidate: (candidate.large_prm_score, candidate.candidate_id),
            )
            steps.append(
                StepTrace(
                    step_index=step_index,
                    shared_kv_tokens=shared_tokens,
                    candidates=tuple(candidates),
                    selected_candidate_id=selected.candidate_id,
                    pruned_candidate_ids=tuple(
                        candidate.candidate_id
                        for candidate in candidates
                        if candidate.candidate_id != selected.candidate_id
                    ),
                )
            )
            shared_tokens += selected.unique_kv_tokens
            parent_candidate_id = selected.candidate_id

        traces.append(
            TtcRequestTrace(
                trace_schema_version=TRACE_SCHEMA_VERSION,
                request_id=f"synthetic-{request_index:04d}",
                dataset="synthetic",
                dataset_id=f"synthetic-item-{request_index:04d}",
                modality=config.modality,
                difficulty="synthetic",
                question_length_bucket=config.question_length_bucket,
                prompt_tokens=config.prompt_tokens,
                image_tokens=config.image_tokens,
                search_width=config.search_width,
                seed=request_seed,
                sampling=SamplingConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_new_tokens=config.max_generated_tokens,
                ),
                provenance=_provenance(),
                steps=tuple(steps),
            )
        )
    return traces
