"""ORCHES Technique 3 fragmentation-aware memory structuring."""

from .allocator import (
    CompactionMove,
    CompactionResult,
    MemoryAllocationError,
    MemoryBlock,
    MemoryKind,
    MemoryStats,
    PimMemoryAllocator,
)
from .buffer import (
    BufferedKvSegment,
    BufferWriteback,
    GpuKvSynchronization,
    GpuKvSyncSource,
    SharedKvBuffer,
    SharedKvBufferStats,
)
from .cache import (
    AddressAccessTiming,
    AddressCache,
    AddressCacheEntry,
    AddressCacheStats,
    AddressLookup,
)
from .policy import CompactionController, CompactionDecision, CompactionPolicy
from .trace import (
    CompactionCommandKind,
    CompactionTrace,
    CompactionTraceCommand,
    build_compaction_trace,
)

__all__ = [
    "AddressCache",
    "AddressCacheEntry",
    "AddressCacheStats",
    "AddressLookup",
    "AddressAccessTiming",
    "BufferedKvSegment",
    "BufferWriteback",
    "CompactionController",
    "CompactionCommandKind",
    "CompactionDecision",
    "CompactionMove",
    "CompactionPolicy",
    "CompactionResult",
    "CompactionTrace",
    "CompactionTraceCommand",
    "GpuKvSynchronization",
    "GpuKvSyncSource",
    "MemoryAllocationError",
    "MemoryBlock",
    "MemoryKind",
    "MemoryStats",
    "PimMemoryAllocator",
    "SharedKvBuffer",
    "SharedKvBufferStats",
    "build_compaction_trace",
]
