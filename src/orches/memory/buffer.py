"""Controller-side shared KV buffer and transfer accounting for ORCHES T3."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigurationError
from .allocator import (
    MemoryAllocationError,
    MemoryBlock,
    MemoryKind,
    PimMemoryAllocator,
    _positive_integer,
)


@dataclass(frozen=True)
class BufferedKvSegment:
    """QKV output waiting in the controller before a PIM-bank writeback."""

    block_id: str
    owner_id: str
    kind: MemoryKind
    size_bytes: int


@dataclass(frozen=True)
class BufferWriteback:
    """One controller-to-bank transfer with explicit host-I/O accounting."""

    segment: BufferedKvSegment
    block: MemoryBlock
    controller_to_bank_bytes: int
    pim_host_bytes: int


@dataclass(frozen=True)
class SharedKvBufferStats:
    """Current occupancy and cumulative data movement."""

    capacity_bytes: int
    used_bytes: int
    resident_segments: int
    controller_to_bank_bytes: int
    gpu_sync_bytes: int


class GpuKvSyncSource(str, Enum):
    """Physical source used when the host GPU requests current KV data."""

    CONTROLLER_BUFFER = "controller_buffer"
    PIM_BANKS = "pim_banks"


@dataclass(frozen=True)
class GpuKvSynchronization:
    """One GPU synchronization transfer and its source location."""

    block_id: str
    source: GpuKvSyncSource
    bytes_transferred: int


class SharedKvBuffer:
    """FIFO controller buffer used for background KV writeback and GPU sync."""

    def __init__(self, *, capacity_bytes: int) -> None:
        _positive_integer("capacity_bytes", capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self._segments: OrderedDict[str, BufferedKvSegment] = OrderedDict()
        self._used_bytes = 0
        self._controller_to_bank_bytes = 0
        self._gpu_sync_bytes = 0

    def stage(
        self,
        block_id: str,
        *,
        owner_id: str,
        kind: MemoryKind,
        size_bytes: int,
    ) -> BufferedKvSegment:
        """Store an accumulated shared/unique KV segment in controller SRAM."""

        if not block_id.strip() or not owner_id.strip():
            raise ConfigurationError("buffer block_id and owner_id must not be empty")
        if block_id in self._segments:
            raise ConfigurationError(f"duplicate buffered segment {block_id!r}")
        if kind not in (MemoryKind.SHARED_KV, MemoryKind.UNIQUE_KV):
            raise ConfigurationError("the shared KV buffer accepts KV segments only")
        _positive_integer("size_bytes", size_bytes)
        if self._used_bytes + size_bytes > self.capacity_bytes:
            raise MemoryAllocationError(
                f"KV segment {block_id!r} exceeds controller-buffer capacity"
            )
        segment = BufferedKvSegment(block_id, owner_id, kind, size_bytes)
        self._segments[block_id] = segment
        self._used_bytes += size_bytes
        return segment

    def write_back_next(self, allocator: PimMemoryAllocator) -> BufferWriteback:
        """Write the oldest segment to PIM banks without PIM-host transfer."""

        if not self._segments:
            raise ConfigurationError("the shared KV buffer is empty")
        block_id, segment = next(iter(self._segments.items()))
        block = allocator.allocate_kv(
            block_id,
            owner_id=segment.owner_id,
            kind=segment.kind,
            size_bytes=segment.size_bytes,
        )
        del self._segments[block_id]
        self._used_bytes -= segment.size_bytes
        self._controller_to_bank_bytes += segment.size_bytes
        return BufferWriteback(
            segment=segment,
            block=block,
            controller_to_bank_bytes=segment.size_bytes,
            pim_host_bytes=0,
        )

    def synchronize_gpu(
        self,
        allocator: PimMemoryAllocator,
        block_id: str,
    ) -> GpuKvSynchronization:
        """Account for a fast buffered sync or a fetch from PIM banks."""

        buffered = self._segments.get(block_id)
        if buffered is not None:
            source = GpuKvSyncSource.CONTROLLER_BUFFER
            size_bytes = buffered.size_bytes
        else:
            block = allocator.block(block_id)
            if block.kind is MemoryKind.WEIGHT:
                raise ConfigurationError("GPU KV synchronization cannot fetch weights")
            source = GpuKvSyncSource.PIM_BANKS
            size_bytes = block.size_bytes
        self._gpu_sync_bytes += size_bytes
        return GpuKvSynchronization(
            block_id=block_id,
            source=source,
            bytes_transferred=size_bytes,
        )

    @property
    def resident_ids(self) -> tuple[str, ...]:
        return tuple(self._segments)

    def stats(self) -> SharedKvBufferStats:
        return SharedKvBufferStats(
            capacity_bytes=self.capacity_bytes,
            used_bytes=self._used_bytes,
            resident_segments=len(self._segments),
            controller_to_bank_bytes=self._controller_to_bank_bytes,
            gpu_sync_bytes=self._gpu_sync_bytes,
        )
