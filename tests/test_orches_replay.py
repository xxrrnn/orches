from __future__ import annotations

from orches.memory import (
    AddressAccessTiming,
    AddressCache,
    CompactionController,
    CompactionPolicy,
    MemoryKind,
    PimMemoryAllocator,
    SharedKvBuffer,
)
from orches.models import PaperGpuRates, PaperPimRates
from orches.predictor import ScoreAggregation, VerificationPipeline
from orches.replay import OrchesRequestReplayer, ReplayConfig
from orches.scheduler import OfflineScheduler, TierThresholds
from orches.sim import Resource
from orches.workload import SyntheticTraceConfig, generate_synthetic_traces


def build_replay(
    *,
    t3_enabled: bool = True,
) -> tuple[OrchesRequestReplayer, PimMemoryAllocator]:
    allocator = PimMemoryAllocator(
        capacity_bytes=4 * 1024 * 1024,
        alignment_bytes=64,
    )
    allocator.allocate_weight("policy-weights", owner_id="model", size_bytes=4096)
    allocator.seal_weights()
    scheduler = OfflineScheduler(
        TierThresholds(medium_min_width=3, large_min_width=7),
        PaperGpuRates(
            compute_mac_per_s=10_000_000,
            memory_elements_per_s=8_000_000,
        ),
        PaperPimRates(
            compute_mac_per_s=5_000_000,
            internal_memory_elements_per_s=10_000_000,
            host_io_elements_per_s=5_000_000,
        ),
    )
    replayer = OrchesRequestReplayer(
        offline_scheduler=scheduler,
        verification_pipeline=VerificationPipeline(
            min_prefill_tokens=4,
            time_per_token_s=0.00001,
        ),
        allocator=allocator,
        address_cache=AddressCache(
            capacity_entries=8,
            timing=AddressAccessTiming(
                sram_access_s=0.000001,
                dram_access_s=0.000005,
            ),
        ),
        shared_kv_buffer=SharedKvBuffer(capacity_bytes=64 * 1024),
        compaction_controller=(
            CompactionController(CompactionPolicy(interval_verifications=1))
            if t3_enabled
            else None
        ),
        config=ReplayConfig(
            hidden_size=16,
            kv_bytes_per_token=64,
            operator_repetitions=1,
            large_prm_time_per_token_s=0.00005,
            rollback_fixed_latency_s=0.00002,
            regeneration_latency_per_token_s=0.0001,
            compaction_internal_bytes_per_s=1_000_000,
            compaction_transaction_bytes=32,
            score_aggregation=ScoreAggregation.MEAN,
            history_alignment=True,
        ),
    )
    return replayer, allocator


def synthetic_trace(seed: int):
    return generate_synthetic_traces(
        SyntheticTraceConfig(
            step_count=3,
            search_width=4,
            seed=seed,
        )
    )[0]


def test_correct_predictions_preserve_pim_work_and_resume_t1() -> None:
    replayer, _ = build_replay()
    result = replayer.run(synthetic_trace(0))

    assert result.prediction_total == 2
    assert result.prediction_correct == 2
    assert result.prediction_accuracy == 1.0
    for current, following in zip(result.steps, result.steps[1:]):
        assert current.speculation is not None
        assert current.speculation.prediction_correct
        assert current.speculation.t1_disabled_during_speculation
        assert following.generation.prior_speculative_event is not None
        assert following.generation.speculative_work_fraction > 0
        assert following.generation.t1_work_fraction < 1
        speculative = following.generation.prior_speculative_event
        assert speculative.resource is Resource.PIM
        assert speculative.start_s < current.large_prm_event.end_s
        assert speculative.end_s <= current.large_prm_event.end_s
    result.timeline.validate()


def test_misprediction_discards_kv_once_and_restarts_selected_branch() -> None:
    replayer, allocator = build_replay()
    trace = synthetic_trace(2)
    result = replayer.run(trace)
    first = result.steps[0]
    following = result.steps[1]

    assert first.prediction.candidate_id != trace.steps[0].selected_candidate_id
    assert first.speculation is not None
    assert not first.speculation.prediction_correct
    assert first.speculation.discarded_kv_bytes > 0
    assert first.rollback_event is not None
    assert first.rollback_event.start_s >= first.large_prm_event.end_s
    assert following.generation.prior_speculative_event is None
    assert following.generation.t1_work_fraction == 1.0
    assert not allocator.has_block("synthetic-0000:step-0:discarded-speculation")
    selected_block = allocator.block(trace.steps[0].selected_candidate_id)
    assert selected_block.kind is MemoryKind.SHARED_KV
    assert result.compaction_read_bytes == result.compaction_write_bytes
    assert result.compaction_read_bytes > 0
    assert result.final_memory.hole_bytes == 0


def test_t3_ablation_retains_larger_fragmented_footprint() -> None:
    enabled_replayer, _ = build_replay(t3_enabled=True)
    disabled_replayer, _ = build_replay(t3_enabled=False)

    enabled = enabled_replayer.run(synthetic_trace(2))
    disabled = disabled_replayer.run(synthetic_trace(2))

    assert disabled.compaction_read_bytes == 0
    assert disabled.final_memory.hole_bytes > 0
    assert (
        enabled.final_memory.allocated_footprint_bytes
        < disabled.final_memory.allocated_footprint_bytes
    )


def test_request_replay_exposes_all_step_decisions_and_resources() -> None:
    replayer, _ = build_replay()
    trace = synthetic_trace(0)
    result = replayer.run(trace)

    assert len(result.steps) == len(trace.steps)
    assert all(step.generation.plan.offline.reason for step in result.steps)
    assert all(step.generation.plan.online.reason for step in result.steps)
    assert all(step.compaction_decision is not None for step in result.steps)
    assert (
        sum(len(step.address_lookups) for step in result.steps)
        == trace.candidate_count
    )
    assert result.timeline.busy_time_s(Resource.GPU) > 0
    assert result.timeline.busy_time_s(Resource.PIM) > 0
    assert result.timeline.busy_time_s(Resource.CONTROLLER) > 0
    assert 0 <= result.timeline.utilization(Resource.GPU) <= 1
    assert 0 <= result.timeline.utilization(Resource.PIM) <= 1
