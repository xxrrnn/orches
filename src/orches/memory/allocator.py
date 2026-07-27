"""Fragmentation-aware physical allocation for ORCHES KV segments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigurationError, SimulationError


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class MemoryKind(str, Enum):
    """Data classes represented in the PIM device address space."""

    WEIGHT = "weight"
    SHARED_KV = "shared_kv"
    UNIQUE_KV = "unique_kv"


@dataclass(frozen=True)
class MemoryBlock:
    """One logical object and its current contiguous physical allocation."""

    block_id: str
    owner_id: str
    kind: MemoryKind
    start: int
    size_bytes: int
    allocated_bytes: int
    generation: int = 0

    @property
    def end(self) -> int:
        return self.start + self.allocated_bytes


@dataclass(frozen=True)
class CompactionMove:
    """Physical address change for one live KV block."""

    block_id: str
    old_start: int
    new_start: int
    size_bytes: int
    allocated_bytes: int
    old_generation: int
    new_generation: int


@dataclass(frozen=True)
class CompactionResult:
    """Footprint and movement accounting for one memory reorganization."""

    moves: tuple[CompactionMove, ...]
    holes_before_bytes: int
    holes_after_bytes: int
    reasoning_span_before_bytes: int
    reasoning_span_after_bytes: int
    moved_bytes: int
    reclaimed_bytes: int
    beta_before: float
    beta_after: float


@dataclass(frozen=True)
class MemoryStats:
    """Snapshot used by the simulator and result manifests."""

    capacity_bytes: int
    weight_bytes: int
    reasoning_base: int | None
    reasoning_live_bytes: int
    reasoning_span_bytes: int
    hole_bytes: int
    beta: float
    live_bytes: int
    allocated_footprint_bytes: int
    peak_reasoning_live_bytes: int
    peak_reasoning_span_bytes: int
    peak_allocated_footprint_bytes: int
    free_bytes: int


class MemoryAllocationError(SimulationError):
    """Raised when a valid allocation does not fit in device memory."""


class PimMemoryAllocator:
    """Weights-first allocator with first-fit KV reuse and deterministic compaction.

    Model weights occupy an immutable prefix. Once ``seal_weights`` is called,
    shared and unique KV blocks are placed with aligned first fit. Pruning removes
    a block but preserves the high-water mark, making the resulting physical hole
    visible until it is reused or compaction reduces the reasoning footprint.
    """

    def __init__(self, *, capacity_bytes: int, alignment_bytes: int) -> None:
        _positive_integer("capacity_bytes", capacity_bytes)
        _positive_integer("alignment_bytes", alignment_bytes)
        self.capacity_bytes = capacity_bytes
        self.alignment_bytes = alignment_bytes
        self._blocks: dict[str, MemoryBlock] = {}
        self._used_block_ids: set[str] = set()
        self._weight_cursor = 0
        self._reasoning_base: int | None = None
        self._reasoning_high_watermark: int | None = None
        self._peak_reasoning_live_bytes = 0
        self._peak_reasoning_span_bytes = 0
        self._peak_allocated_footprint_bytes = 0

    @property
    def weights_sealed(self) -> bool:
        return self._reasoning_base is not None

    @property
    def reasoning_base(self) -> int | None:
        return self._reasoning_base

    @property
    def blocks(self) -> tuple[MemoryBlock, ...]:
        return tuple(sorted(self._blocks.values(), key=lambda block: block.start))

    def has_block(self, block_id: str) -> bool:
        return block_id in self._blocks

    def block(self, block_id: str) -> MemoryBlock:
        try:
            return self._blocks[block_id]
        except KeyError as error:
            raise ConfigurationError(f"unknown memory block {block_id!r}") from error

    def allocate_weight(
        self,
        block_id: str,
        *,
        owner_id: str,
        size_bytes: int,
    ) -> MemoryBlock:
        """Append one immutable model-weight block before reasoning starts."""

        if self.weights_sealed:
            raise ConfigurationError("model weights are sealed and immutable")
        self._validate_identity_and_size(block_id, owner_id, size_bytes)
        start = _align_up(self._weight_cursor, self.alignment_bytes)
        allocated = _align_up(size_bytes, self.alignment_bytes)
        if start + allocated > self.capacity_bytes:
            raise MemoryAllocationError(
                f"weight block {block_id!r} does not fit in device memory"
            )
        block = MemoryBlock(
            block_id=block_id,
            owner_id=owner_id,
            kind=MemoryKind.WEIGHT,
            start=start,
            size_bytes=size_bytes,
            allocated_bytes=allocated,
        )
        self._blocks[block_id] = block
        self._used_block_ids.add(block_id)
        self._weight_cursor = block.end
        self._record_peaks()
        return block

    def seal_weights(self) -> int:
        """Freeze model placement and return the aligned reasoning-memory base."""

        if self.weights_sealed:
            raise ConfigurationError("model weights have already been sealed")
        base = _align_up(self._weight_cursor, self.alignment_bytes)
        if base > self.capacity_bytes:
            raise MemoryAllocationError("aligned model weights exceed device capacity")
        self._reasoning_base = base
        self._reasoning_high_watermark = base
        self._record_peaks()
        return base

    def allocate_kv(
        self,
        block_id: str,
        *,
        owner_id: str,
        kind: MemoryKind,
        size_bytes: int,
    ) -> MemoryBlock:
        """Place a shared or unique KV segment using aligned first fit."""

        if not self.weights_sealed:
            raise ConfigurationError("seal model weights before allocating KV")
        if kind not in (MemoryKind.SHARED_KV, MemoryKind.UNIQUE_KV):
            raise ConfigurationError(
                "KV allocation kind must be shared_kv or unique_kv"
            )
        self._validate_identity_and_size(block_id, owner_id, size_bytes)
        allocated = _align_up(size_bytes, self.alignment_bytes)
        start = self._find_first_fit(allocated)
        if start is None:
            raise MemoryAllocationError(
                f"KV block {block_id!r} ({allocated} aligned bytes) does not fit"
            )
        block = MemoryBlock(
            block_id=block_id,
            owner_id=owner_id,
            kind=kind,
            start=start,
            size_bytes=size_bytes,
            allocated_bytes=allocated,
        )
        self._blocks[block_id] = block
        self._used_block_ids.add(block_id)
        assert self._reasoning_high_watermark is not None
        self._reasoning_high_watermark = max(
            self._reasoning_high_watermark,
            block.end,
        )
        self._record_peaks()
        return block

    def promote_to_shared(self, block_id: str) -> MemoryBlock:
        """Mark a selected candidate's KV as shared without moving its bytes."""

        block = self.block(block_id)
        if block.kind is MemoryKind.WEIGHT:
            raise ConfigurationError("model weights cannot become shared KV")
        if block.kind is MemoryKind.SHARED_KV:
            return block
        promoted = MemoryBlock(
            block_id=block.block_id,
            owner_id=block.owner_id,
            kind=MemoryKind.SHARED_KV,
            start=block.start,
            size_bytes=block.size_bytes,
            allocated_bytes=block.allocated_bytes,
            generation=block.generation,
        )
        self._blocks[block_id] = promoted
        return promoted

    def prune(self, block_id: str) -> MemoryBlock:
        """Remove a KV block while leaving its physical range as a hole."""

        block = self.block(block_id)
        if block.kind is MemoryKind.WEIGHT:
            raise ConfigurationError("model weights cannot be pruned")
        del self._blocks[block_id]
        return block

    def prune_owner(self, owner_id: str) -> tuple[MemoryBlock, ...]:
        """Prune every KV segment belonging to one discarded candidate."""

        if not owner_id.strip():
            raise ConfigurationError("owner_id must not be empty")
        removed = tuple(
            block
            for block in self.blocks
            if block.owner_id == owner_id and block.kind is not MemoryKind.WEIGHT
        )
        for block in removed:
            del self._blocks[block.block_id]
        return removed

    def compact(self) -> CompactionResult:
        """Move live KV blocks to a contiguous range after immutable weights."""

        if not self.weights_sealed:
            raise ConfigurationError("seal model weights before compaction")
        before = self.stats()
        assert self._reasoning_base is not None
        target = self._reasoning_base
        moves: list[CompactionMove] = []
        for block in self._kv_blocks():
            if block.start != target:
                moved = MemoryBlock(
                    block_id=block.block_id,
                    owner_id=block.owner_id,
                    kind=block.kind,
                    start=target,
                    size_bytes=block.size_bytes,
                    allocated_bytes=block.allocated_bytes,
                    generation=block.generation + 1,
                )
                self._blocks[block.block_id] = moved
                moves.append(
                    CompactionMove(
                        block_id=block.block_id,
                        old_start=block.start,
                        new_start=target,
                        size_bytes=block.size_bytes,
                        allocated_bytes=block.allocated_bytes,
                        old_generation=block.generation,
                        new_generation=moved.generation,
                    )
                )
            target += block.allocated_bytes
        self._reasoning_high_watermark = target
        after = self.stats()
        self.validate()
        return CompactionResult(
            moves=tuple(moves),
            holes_before_bytes=before.hole_bytes,
            holes_after_bytes=after.hole_bytes,
            reasoning_span_before_bytes=before.reasoning_span_bytes,
            reasoning_span_after_bytes=after.reasoning_span_bytes,
            moved_bytes=sum(move.size_bytes for move in moves),
            reclaimed_bytes=(
                before.reasoning_span_bytes - after.reasoning_span_bytes
            ),
            beta_before=before.beta,
            beta_after=after.beta,
        )

    def stats(self) -> MemoryStats:
        weight_bytes = sum(
            block.allocated_bytes
            for block in self._blocks.values()
            if block.kind is MemoryKind.WEIGHT
        )
        reasoning_live = sum(block.allocated_bytes for block in self._kv_blocks())
        if self._reasoning_base is None:
            reasoning_span = 0
        else:
            assert self._reasoning_high_watermark is not None
            reasoning_span = self._reasoning_high_watermark - self._reasoning_base
        holes = reasoning_span - reasoning_live
        beta = holes / reasoning_span if reasoning_span else 0.0
        live = weight_bytes + reasoning_live
        footprint = (
            self._reasoning_high_watermark
            if self._reasoning_high_watermark is not None
            else self._weight_cursor
        )
        return MemoryStats(
            capacity_bytes=self.capacity_bytes,
            weight_bytes=weight_bytes,
            reasoning_base=self._reasoning_base,
            reasoning_live_bytes=reasoning_live,
            reasoning_span_bytes=reasoning_span,
            hole_bytes=holes,
            beta=beta,
            live_bytes=live,
            allocated_footprint_bytes=footprint,
            peak_reasoning_live_bytes=self._peak_reasoning_live_bytes,
            peak_reasoning_span_bytes=self._peak_reasoning_span_bytes,
            peak_allocated_footprint_bytes=self._peak_allocated_footprint_bytes,
            free_bytes=self.capacity_bytes - live,
        )

    def validate(self) -> None:
        """Check alignment, ownership regions, bounds, and pairwise overlap."""

        ordered = self.blocks
        for block in ordered:
            if block.start % self.alignment_bytes:
                raise SimulationError(f"block {block.block_id!r} is misaligned")
            if block.allocated_bytes % self.alignment_bytes:
                raise SimulationError(
                    f"block {block.block_id!r} has a misaligned allocation"
                )
            if block.size_bytes > block.allocated_bytes:
                raise SimulationError(
                    f"block {block.block_id!r} exceeds its allocated range"
                )
            if block.start < 0 or block.end > self.capacity_bytes:
                raise SimulationError(f"block {block.block_id!r} is out of bounds")
            if self._reasoning_base is not None:
                if (
                    block.kind is MemoryKind.WEIGHT
                    and block.end > self._reasoning_base
                ):
                    raise SimulationError("a weight block overlaps reasoning memory")
                if (
                    block.kind is not MemoryKind.WEIGHT
                    and block.start < self._reasoning_base
                ):
                    raise SimulationError("a KV block overlaps immutable weights")
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end > current.start:
                raise SimulationError(
                    f"memory blocks overlap: {previous.block_id!r} and "
                    f"{current.block_id!r}"
                )

    def _validate_identity_and_size(
        self,
        block_id: str,
        owner_id: str,
        size_bytes: int,
    ) -> None:
        if not block_id.strip():
            raise ConfigurationError("block_id must not be empty")
        if block_id in self._used_block_ids:
            raise ConfigurationError(
                f"memory block ID {block_id!r} has already been used"
            )
        if not owner_id.strip():
            raise ConfigurationError("owner_id must not be empty")
        _positive_integer("size_bytes", size_bytes)

    def _kv_blocks(self) -> tuple[MemoryBlock, ...]:
        return tuple(
            block
            for block in self.blocks
            if block.kind is not MemoryKind.WEIGHT
        )

    def _find_first_fit(self, allocated_bytes: int) -> int | None:
        assert self._reasoning_base is not None
        cursor = self._reasoning_base
        for block in self._kv_blocks():
            if block.start - cursor >= allocated_bytes:
                return cursor
            cursor = max(cursor, block.end)
        if self.capacity_bytes - cursor >= allocated_bytes:
            return cursor
        return None

    def _record_peaks(self) -> None:
        stats = self.stats()
        self._peak_reasoning_live_bytes = max(
            self._peak_reasoning_live_bytes,
            stats.reasoning_live_bytes,
        )
        self._peak_reasoning_span_bytes = max(
            self._peak_reasoning_span_bytes,
            stats.reasoning_span_bytes,
        )
        self._peak_allocated_footprint_bytes = max(
            self._peak_allocated_footprint_bytes,
            stats.allocated_footprint_bytes,
        )
