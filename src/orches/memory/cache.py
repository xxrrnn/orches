"""Controller-die logical-to-physical address cache for ORCHES T3."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite

from ..errors import ConfigurationError
from .allocator import MemoryBlock, MemoryKind, PimMemoryAllocator, _positive_integer


@dataclass(frozen=True)
class AddressCacheEntry:
    """Cached beginning, length, and allocation generation for one KV segment."""

    block_id: str
    start: int
    length_bytes: int
    generation: int


@dataclass(frozen=True)
class AddressLookup:
    """Resolved controller address and whether SRAM supplied a valid mapping."""

    entry: AddressCacheEntry
    hit: bool
    sram_accesses: int
    dram_accesses: int
    latency_s: float


@dataclass(frozen=True)
class AddressAccessTiming:
    """Calibrated controller SRAM and dependent DRAM access latency."""

    sram_access_s: float
    dram_access_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("sram_access_s", self.sram_access_s),
            ("dram_access_s", self.dram_access_s),
        ):
            if not isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AddressCacheStats:
    """Cache counters suitable for an experiment manifest."""

    capacity_entries: int
    resident_entries: int
    hits: int
    misses: int
    evictions: int
    stale_invalidations: int
    total_lookup_latency_s: float


class AddressCache:
    """Shared fully-associative LRU cache with generation-based coherence.

    The paper specifies one cache shared by all banks but does not disclose its
    capacity, associativity, or replacement policy. This implementation exposes
    capacity explicitly and uses deterministic LRU as assumption ``A-T3-001``.
    """

    def __init__(
        self,
        *,
        capacity_entries: int,
        timing: AddressAccessTiming,
    ) -> None:
        _positive_integer("capacity_entries", capacity_entries)
        self.capacity_entries = capacity_entries
        self.timing = timing
        self._entries: OrderedDict[str, AddressCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stale_invalidations = 0
        self._total_lookup_latency_s = 0.0

    def resolve(self, allocator: PimMemoryAllocator, block_id: str) -> AddressLookup:
        """Resolve a live KV block and refresh stale post-compaction mappings."""

        if not block_id.strip():
            raise ConfigurationError("block_id must not be empty")
        if not allocator.has_block(block_id):
            if block_id in self._entries:
                del self._entries[block_id]
                self._stale_invalidations += 1
            raise ConfigurationError(
                f"cannot resolve pruned or unknown block {block_id!r}"
            )

        block = allocator.block(block_id)
        if block.kind is MemoryKind.WEIGHT:
            raise ConfigurationError(
                "the candidate address cache stores KV blocks only"
            )
        cached = self._entries.get(block_id)
        if cached is not None and self._matches(cached, block):
            self._hits += 1
            self._entries.move_to_end(block_id)
            latency = self.timing.sram_access_s + self.timing.dram_access_s
            self._total_lookup_latency_s += latency
            return AddressLookup(
                entry=cached,
                hit=True,
                sram_accesses=1,
                dram_accesses=1,
                latency_s=latency,
            )

        self._misses += 1
        if cached is not None:
            del self._entries[block_id]
            self._stale_invalidations += 1
        entry = AddressCacheEntry(
            block_id=block.block_id,
            start=block.start,
            length_bytes=block.size_bytes,
            generation=block.generation,
        )
        self._entries[block_id] = entry
        if len(self._entries) > self.capacity_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        latency = self.timing.sram_access_s + 2 * self.timing.dram_access_s
        self._total_lookup_latency_s += latency
        return AddressLookup(
            entry=entry,
            hit=False,
            sram_accesses=1,
            dram_accesses=2,
            latency_s=latency,
        )

    def invalidate(self, block_id: str) -> bool:
        """Invalidate a mapping eagerly when branch-pruning metadata is available."""

        return self._entries.pop(block_id, None) is not None

    @property
    def resident_ids(self) -> tuple[str, ...]:
        """Return IDs from least to most recently used."""

        return tuple(self._entries)

    def stats(self) -> AddressCacheStats:
        return AddressCacheStats(
            capacity_entries=self.capacity_entries,
            resident_entries=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            stale_invalidations=self._stale_invalidations,
            total_lookup_latency_s=self._total_lookup_latency_s,
        )

    @staticmethod
    def _matches(entry: AddressCacheEntry, block: MemoryBlock) -> bool:
        return (
            entry.start == block.start
            and entry.length_bytes == block.size_bytes
            and entry.generation == block.generation
        )
