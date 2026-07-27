from __future__ import annotations

import random

import pytest

from orches.errors import ConfigurationError
from orches.memory import (
    AddressAccessTiming,
    AddressCache,
    CompactionCommandKind,
    CompactionController,
    CompactionPolicy,
    GpuKvSyncSource,
    MemoryAllocationError,
    MemoryKind,
    PimMemoryAllocator,
    SharedKvBuffer,
    build_compaction_trace,
)


def make_allocator(*, capacity_bytes: int = 4096) -> PimMemoryAllocator:
    allocator = PimMemoryAllocator(
        capacity_bytes=capacity_bytes,
        alignment_bytes=64,
    )
    allocator.allocate_weight("policy", owner_id="model", size_bytes=100)
    allocator.seal_weights()
    return allocator


def test_first_fit_reuses_pruned_physical_hole() -> None:
    allocator = make_allocator()
    first = allocator.allocate_kv(
        "a",
        owner_id="candidate-a",
        kind=MemoryKind.SHARED_KV,
        size_bytes=100,
    )
    removed = allocator.allocate_kv(
        "b",
        owner_id="candidate-b",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=160,
    )
    allocator.allocate_kv(
        "c",
        owner_id="candidate-c",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=80,
    )

    allocator.prune(removed.block_id)
    replacement = allocator.allocate_kv(
        "d",
        owner_id="candidate-d",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=128,
    )

    assert first.start == allocator.reasoning_base
    assert replacement.start == removed.start
    assert allocator.stats().hole_bytes == 64
    allocator.validate()


def test_compaction_reclaims_footprint_and_updates_cache_generation() -> None:
    allocator = make_allocator()
    allocator.allocate_kv("a", owner_id="a", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.allocate_kv("b", owner_id="b", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    final = allocator.allocate_kv(
        "c", owner_id="c", kind=MemoryKind.UNIQUE_KV, size_bytes=64
    )
    cache = AddressCache(
        capacity_entries=2,
        timing=AddressAccessTiming(sram_access_s=1e-9, dram_access_s=50e-9),
    )
    assert not cache.resolve(allocator, final.block_id).hit
    assert cache.resolve(allocator, final.block_id).hit

    allocator.prune("b")
    result = allocator.compact()
    refreshed = cache.resolve(allocator, final.block_id)

    assert result.beta_before == pytest.approx(1 / 3)
    assert result.beta_after == 0
    assert result.reclaimed_bytes == 64
    assert result.moved_bytes == 64
    assert refreshed.hit is False
    assert refreshed.entry.start == final.start - 64
    assert refreshed.entry.generation == final.generation + 1
    assert refreshed.sram_accesses == 1
    assert refreshed.dram_accesses == 2
    assert refreshed.latency_s == pytest.approx(101e-9)
    assert cache.stats().stale_invalidations == 1


def test_address_cache_uses_deterministic_lru_replacement() -> None:
    allocator = make_allocator()
    for block_id in ("a", "b", "c"):
        allocator.allocate_kv(
            block_id,
            owner_id=block_id,
            kind=MemoryKind.UNIQUE_KV,
            size_bytes=64,
        )
    cache = AddressCache(
        capacity_entries=2,
        timing=AddressAccessTiming(sram_access_s=1e-9, dram_access_s=50e-9),
    )

    cache.resolve(allocator, "a")
    cache.resolve(allocator, "b")
    cache.resolve(allocator, "a")
    cache.resolve(allocator, "c")

    assert cache.resident_ids == ("a", "c")
    assert cache.stats().hits == 1
    assert cache.stats().misses == 3
    assert cache.stats().evictions == 1


def test_compaction_emits_balanced_read_write_transactions() -> None:
    allocator = make_allocator()
    allocator.allocate_kv("a", owner_id="a", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.allocate_kv("b", owner_id="b", kind=MemoryKind.UNIQUE_KV, size_bytes=96)
    allocator.prune("a")

    result = allocator.compact()
    trace = build_compaction_trace(result, transaction_bytes=32)

    assert trace.read_bytes == trace.write_bytes == 128
    assert trace.total_bytes == 256
    assert len(trace.commands) == 8
    assert trace.commands[0].kind is CompactionCommandKind.READ
    assert trace.commands[1].kind is CompactionCommandKind.WRITE
    assert trace.commands[0].address == result.moves[0].old_start
    assert trace.commands[1].address == result.moves[0].new_start


@pytest.mark.parametrize("interval", (3, 4, 5))
def test_paper_evaluation_intervals_trigger_compaction(interval: int) -> None:
    allocator = make_allocator()
    allocator.allocate_kv("a", owner_id="a", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.allocate_kv("b", owner_id="b", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.prune("a")
    controller = CompactionController(
        CompactionPolicy(interval_verifications=interval)
    )

    decisions = [
        controller.observe_verification(index, allocator)
        for index in range(1, interval + 1)
    ]

    assert all(not decision.triggered for decision in decisions[:-1])
    assert decisions[-1].triggered
    assert decisions[-1].trigger_reasons == ("fixed_interval",)
    assert allocator.stats().hole_bytes == 0


def test_explicit_beta_threshold_can_trigger_before_interval() -> None:
    allocator = make_allocator()
    allocator.allocate_kv("a", owner_id="a", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.allocate_kv("b", owner_id="b", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.prune("a")
    controller = CompactionController(
        CompactionPolicy(interval_verifications=5, beta_threshold=0.4)
    )

    decision = controller.observe_verification(1, allocator)

    assert decision.triggered
    assert decision.observed_beta == 0.5
    assert decision.trigger_reasons == ("beta_threshold",)


def test_without_t3_controller_pruned_holes_remain_visible() -> None:
    allocator = make_allocator()
    allocator.allocate_kv("a", owner_id="a", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.allocate_kv("b", owner_id="b", kind=MemoryKind.UNIQUE_KV, size_bytes=64)
    allocator.prune("a")

    before = allocator.stats()
    after = allocator.stats()

    assert before.hole_bytes == 64
    assert after == before


def test_random_pruning_and_compaction_preserve_live_logical_blocks() -> None:
    allocator = make_allocator(capacity_bytes=64 * 1024)
    generator = random.Random(20250727)
    for index in range(40):
        allocator.allocate_kv(
            f"kv-{index:02d}",
            owner_id=f"candidate-{index // 2:02d}",
            kind=(
                MemoryKind.SHARED_KV if index % 5 == 0 else MemoryKind.UNIQUE_KV
            ),
            size_bytes=generator.randrange(32, 513),
        )
    removed_ids = set(
        generator.sample(
            [block.block_id for block in allocator.blocks[1:]],
            15,
        )
    )
    for block_id in removed_ids:
        allocator.prune(block_id)
    before = {
        block.block_id: (block.owner_id, block.kind, block.size_bytes)
        for block in allocator.blocks
        if block.kind is not MemoryKind.WEIGHT
    }

    result = allocator.compact()
    after = {
        block.block_id: (block.owner_id, block.kind, block.size_bytes)
        for block in allocator.blocks
        if block.kind is not MemoryKind.WEIGHT
    }

    assert after == before
    assert allocator.stats().hole_bytes == 0
    assert result.reclaimed_bytes > 0
    assert (
        allocator.stats().peak_reasoning_span_bytes
        >= result.reasoning_span_before_bytes
    )
    allocator.validate()


def test_weights_are_immutable_and_capacity_errors_are_explicit() -> None:
    allocator = PimMemoryAllocator(capacity_bytes=256, alignment_bytes=64)
    allocator.allocate_weight("weights", owner_id="model", size_bytes=128)
    allocator.seal_weights()
    allocator.allocate_kv(
        "only-kv",
        owner_id="candidate",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=128,
    )

    with pytest.raises(ConfigurationError, match="sealed and immutable"):
        allocator.allocate_weight("late", owner_id="model", size_bytes=1)
    with pytest.raises(ConfigurationError, match="cannot be pruned"):
        allocator.prune("weights")
    with pytest.raises(MemoryAllocationError, match="does not fit"):
        allocator.allocate_kv(
            "overflow",
            owner_id="candidate",
            kind=MemoryKind.UNIQUE_KV,
            size_bytes=1,
        )


def test_pruned_logical_id_cannot_be_reused_and_alias_stale_cache_state() -> None:
    allocator = make_allocator()
    allocator.allocate_kv(
        "candidate",
        owner_id="candidate",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=64,
    )
    allocator.prune("candidate")

    with pytest.raises(ConfigurationError, match="already been used"):
        allocator.allocate_kv(
            "candidate",
            owner_id="candidate",
            kind=MemoryKind.UNIQUE_KV,
            size_bytes=64,
        )


def test_compaction_controller_rejects_skipped_verification_index() -> None:
    allocator = make_allocator()
    controller = CompactionController(CompactionPolicy(interval_verifications=3))
    controller.observe_verification(1, allocator)

    with pytest.raises(ConfigurationError, match="contiguous"):
        controller.observe_verification(3, allocator)


def test_shared_kv_buffer_accounts_bank_writeback_and_gpu_sync() -> None:
    allocator = make_allocator()
    buffer = SharedKvBuffer(capacity_bytes=128)
    buffer.stage(
        "shared",
        owner_id="selected",
        kind=MemoryKind.SHARED_KV,
        size_bytes=64,
    )
    buffer.stage(
        "unique",
        owner_id="selected",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=48,
    )

    writeback = buffer.write_back_next(allocator)
    synchronized = buffer.synchronize_gpu(allocator, writeback.block.block_id)

    assert writeback.segment.block_id == "shared"
    assert writeback.controller_to_bank_bytes == 64
    assert writeback.pim_host_bytes == 0
    assert synchronized.bytes_transferred == 64
    assert synchronized.source is GpuKvSyncSource.PIM_BANKS
    assert buffer.resident_ids == ("unique",)
    assert buffer.stats().used_bytes == 48
    assert buffer.stats().controller_to_bank_bytes == 64
    assert buffer.stats().gpu_sync_bytes == 64


def test_gpu_can_synchronize_directly_from_resident_controller_kv() -> None:
    allocator = make_allocator()
    buffer = SharedKvBuffer(capacity_bytes=128)
    buffer.stage(
        "resident",
        owner_id="selected",
        kind=MemoryKind.SHARED_KV,
        size_bytes=64,
    )

    synchronized = buffer.synchronize_gpu(allocator, "resident")

    assert synchronized.source is GpuKvSyncSource.CONTROLLER_BUFFER
    assert synchronized.bytes_transferred == 64
    assert buffer.resident_ids == ("resident",)


def test_shared_kv_buffer_rejects_overflow_without_losing_resident_data() -> None:
    buffer = SharedKvBuffer(capacity_bytes=64)
    buffer.stage(
        "resident",
        owner_id="candidate",
        kind=MemoryKind.UNIQUE_KV,
        size_bytes=48,
    )

    with pytest.raises(MemoryAllocationError, match="buffer capacity"):
        buffer.stage(
            "overflow",
            owner_id="candidate",
            kind=MemoryKind.UNIQUE_KV,
            size_bytes=32,
        )

    assert buffer.resident_ids == ("resident",)
    assert buffer.stats().used_bytes == 48
