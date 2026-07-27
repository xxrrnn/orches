from __future__ import annotations

import pytest

from orches.predictor import TokenBatch, VerificationPipeline
from orches.sim import EventTimeline, Resource


def test_below_threshold_waits_for_final_verification() -> None:
    pipeline = VerificationPipeline(min_prefill_tokens=10, time_per_token_s=0.1)
    result = pipeline.schedule(
        (TokenBatch("a", ready_time_s=1.0, token_count=4),),
        generation_complete_s=3.0,
    )

    assert result.early_verified_tokens == 0
    assert result.chunks[0].event.start_s == 3.0
    assert result.saved_latency_s == 0.0


def test_threshold_enables_partial_overlap() -> None:
    pipeline = VerificationPipeline(min_prefill_tokens=4, time_per_token_s=0.1)
    result = pipeline.schedule(
        (
            TokenBatch("a", ready_time_s=1.0, token_count=4),
            TokenBatch("b", ready_time_s=2.0, token_count=2),
        ),
        generation_complete_s=3.0,
    )

    assert result.early_verified_tokens == 4
    assert len(result.chunks) == 2
    assert result.chunks[0].event.start_s == 1.0
    assert result.chunks[1].event.start_s == 3.0
    assert result.saved_latency_s == pytest.approx(0.4)


def test_all_tokens_can_be_preverified() -> None:
    pipeline = VerificationPipeline(min_prefill_tokens=2, time_per_token_s=0.1)
    result = pipeline.schedule(
        (
            TokenBatch("a", ready_time_s=1.0, token_count=2),
            TokenBatch("b", ready_time_s=2.0, token_count=2),
        ),
        generation_complete_s=4.0,
    )

    assert result.early_verified_tokens == result.total_tokens
    assert all(chunk.early for chunk in result.chunks)
    assert result.completion_s < result.generation_complete_s


def test_existing_gpu_work_delays_but_does_not_duplicate_chunks() -> None:
    timeline = EventTimeline()
    timeline.schedule("large-prm", Resource.GPU, 2.0)
    pipeline = VerificationPipeline(min_prefill_tokens=2, time_per_token_s=0.1)
    result = pipeline.schedule(
        (
            TokenBatch("a", ready_time_s=0.5, token_count=2),
            TokenBatch("b", ready_time_s=1.0, token_count=2),
        ),
        generation_complete_s=3.0,
        timeline=timeline,
    )

    assert result.chunks[0].event.start_s == 2.0
    assert sum(chunk.token_count for chunk in result.chunks) == result.total_tokens
    timeline.validate()


def test_event_prefix_allows_multiple_steps_on_one_timeline() -> None:
    timeline = EventTimeline()
    pipeline = VerificationPipeline(min_prefill_tokens=1, time_per_token_s=0.1)

    first = pipeline.schedule(
        (TokenBatch("first", ready_time_s=0.0, token_count=1),),
        generation_complete_s=0.0,
        timeline=timeline,
        event_prefix="step-0-small-prm",
    )
    second = pipeline.schedule(
        (TokenBatch("second", ready_time_s=0.0, token_count=1),),
        generation_complete_s=0.0,
        timeline=timeline,
        event_prefix="step-1-small-prm",
    )

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id
    timeline.validate()


def test_serial_baseline_pays_one_chunk_overhead() -> None:
    pipeline = VerificationPipeline(
        min_prefill_tokens=2,
        time_per_token_s=0.1,
        fixed_chunk_overhead_s=0.05,
    )
    result = pipeline.schedule(
        (
            TokenBatch("a", ready_time_s=0.0, token_count=2),
            TokenBatch("b", ready_time_s=1.0, token_count=2),
        ),
        generation_complete_s=2.0,
    )

    assert len(result.chunks) == 2
    assert result.serial_completion_s == pytest.approx(2.45)
