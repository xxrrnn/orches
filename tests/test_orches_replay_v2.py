from __future__ import annotations

import pytest

from orches.errors import ConfigurationError
from orches.memory import (
    AddressAccessTiming,
    AddressCache,
    MemoryKind,
    PimMemoryAllocator,
    SharedKvBuffer,
)
from orches.metrics import (
    EnergyActivity,
    generation_plan_activity,
    replay_activity,
)
from orches.models import PaperGpuRates, PaperPimRates
from orches.predictor import ScoreAggregation, VerificationPipeline
from orches.replay import ReplayConfig
from orches.replay_v2 import OrchesV2RequestReplayer
from orches.scheduler import OfflineScheduler, TierThresholds
from orches.workload.io import parse_request
from test_workload_trace_v2 import (
    _beam_three_trace,
    _pairwise_trace,
    _two_step_beam_trace,
)


def build_v2_replayer() -> tuple[OrchesV2RequestReplayer, PimMemoryAllocator]:
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
    replayer = OrchesV2RequestReplayer(
        offline_scheduler=scheduler,
        verification_pipeline=VerificationPipeline(
            min_prefill_tokens=4,
            time_per_token_s=0.00001,
        ),
        allocator=allocator,
        address_cache=AddressCache(
            capacity_entries=16,
            timing=AddressAccessTiming(
                sram_access_s=0.000001,
                dram_access_s=0.000005,
            ),
        ),
        shared_kv_buffer=SharedKvBuffer(capacity_bytes=64 * 1024),
        compaction_controller=None,
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


def test_v2_replay_uses_one_generation_phase_per_generated_token() -> None:
    raw = _beam_three_trace().to_dict()
    candidates = raw["steps"][0]["candidates"]
    candidates[0]["generated_token_ids"] = candidates[0][
        "generated_token_ids"
    ][:1]
    candidates[0]["token_ready_indices"] = [0]
    for candidate in candidates[1:]:
        candidate["token_ready_indices"] = [
            ready_index - 1 for ready_index in candidate["token_ready_indices"]
        ]
    trace = parse_request(raw)
    replayer, allocator = build_v2_replayer()

    result = replayer.run(trace)
    step = result.steps[0]

    assert len(step.generation.phase_plans) == 2
    assert [plan.fragments[0].branch_width for plan in step.generation.phase_plans] == [
        4,
        3,
    ]
    assert len(step.generation.phase_plans[0].fragments) == 1
    assert step.verification.total_tokens == trace.generated_tokens
    assert step.prediction is None
    assert {block.block_id for block in allocator.blocks} == {
        "policy-weights",
        "root-kv",
        "c1-kv",
        "c2-kv",
        "c3-kv",
    }
    assert allocator.block("c2-kv").size_bytes == 64
    assert allocator.block("c3-kv").size_bytes == 128


def test_v2_replay_preserves_multiple_parent_lineages_across_steps() -> None:
    trace = _two_step_beam_trace()
    replayer, allocator = build_v2_replayer()

    result = replayer.run(trace)

    assert len(result.steps) == 2
    assert all(step.prediction is None for step in result.steps)
    assert result.prediction_total == 0
    assert result.steps[0].selection_events[0].freed_kv_block_ids == (
        "c0-kv",
        "c2-kv",
    )
    assert {block.block_id for block in allocator.blocks} == {
        "policy-weights",
        "root-kv",
        "c1-kv",
        "c3-kv",
        "d1-kv",
        "d6-kv",
    }
    assert result.steps[1].generation.phase_plans[0].fragments[0].length_tokens == 3
    result.timeline.validate()


def test_v2_pairwise_replay_interleaves_judges_and_ordered_pruning() -> None:
    trace = _pairwise_trace()
    replayer, allocator = build_v2_replayer()

    result = replayer.run(trace)
    step = result.steps[0]

    assert [event.freed_kv_block_ids for event in step.selection_events] == [
        ("c0-kv",),
        ("c3-kv",),
        ("c1-kv",),
    ]
    assert len(step.verifier_calls) == 3
    assert step.selection_events[0].event.event_id in (
        step.verifier_calls[1].event.dependencies
    )
    assert step.selection_events[1].event.event_id in (
        step.verifier_calls[2].event.dependencies
    )
    expected_duration = (580 + 1) * 0.00005
    assert all(
        call.event.duration_s == pytest.approx(expected_duration)
        for call in step.verifier_calls
    )
    assert step.verification.total_tokens == 0
    assert {block.block_id for block in allocator.blocks} == {
        "policy-weights",
        "root-kv",
        "c2-kv",
    }
    assert allocator.block("c2-kv").kind is MemoryKind.SHARED_KV


def test_v2_replay_merges_candidate_local_small_prm_calls() -> None:
    raw = _beam_three_trace().to_dict()
    step = raw["steps"][0]
    batched_small = step["verifier_calls"][0]
    candidate_calls = []
    for index, candidate_id in enumerate(batched_small["candidate_ids"]):
        candidate_calls.append(
            {
                **batched_small,
                "call_id": f"s0-small-{candidate_id}",
                "candidate_ids": [candidate_id],
                "input_tensors": [batched_small["input_tensors"][index]],
                "scores": [batched_small["scores"][index]],
            }
        )
    step["verifier_calls"] = [*candidate_calls, step["verifier_calls"][1]]
    raw["beam_size"] = 1
    step["selection_events"][0].update(
        selected_candidate_ids=["c3"],
        pruned_candidate_ids=["c0", "c1", "c2"],
        live_candidate_ids_after=["c3"],
        retained_kv_block_ids=["root-kv", "c3-kv"],
    )
    step["selected_candidate_ids"] = ["c3"]
    step["pruned_candidate_ids"] = ["c0", "c1", "c2"]
    trace = parse_request(raw)
    replayer, _ = build_v2_replayer()

    result = replayer.run(trace)

    assert result.steps[0].verification.total_tokens == trace.generated_tokens
    assert result.steps[0].prediction is not None
    assert result.steps[0].prediction.candidate_id == "c3"


def test_v2_replayer_is_single_use() -> None:
    replayer, _ = build_v2_replayer()
    trace = _beam_three_trace()

    replayer.run(trace)

    with pytest.raises(ConfigurationError, match="single-use"):
        replayer.run(trace)


def test_v2_activity_counts_every_token_phase() -> None:
    replayer, _ = build_v2_replayer()
    result = replayer.run(_beam_three_trace())

    activity = replay_activity(
        result,
        bytes_per_element=2,
        cache_entry_bytes=24,
    )
    expected = sum(
        (
            generation_plan_activity(
                plan,
                work_fraction=1.0,
                bytes_per_element=2,
            )
            for plan in result.steps[0].generation.phase_plans
        ),
        start=EnergyActivity(),
    )

    assert activity.gpu_macs == pytest.approx(expected.gpu_macs)
    assert activity.pim_macs == pytest.approx(expected.pim_macs)
