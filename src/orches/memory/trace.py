"""Deterministic DRAM read/write trace generation for T3 compaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigurationError
from .allocator import CompactionResult, _positive_integer


class CompactionCommandKind(str, Enum):
    """Memory operation emitted while relocating a live KV block."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class CompactionTraceCommand:
    """One transaction-sized access in a relocation trace."""

    sequence: int
    block_id: str
    kind: CompactionCommandKind
    address: int
    size_bytes: int
    generation: int


@dataclass(frozen=True)
class CompactionTrace:
    """Ordered controller trace and aggregate internal-memory traffic."""

    commands: tuple[CompactionTraceCommand, ...]
    read_bytes: int
    write_bytes: int
    transaction_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


def build_compaction_trace(
    result: CompactionResult,
    *,
    transaction_bytes: int,
) -> CompactionTrace:
    """Emit read-before-write transactions for every physical block move."""

    _positive_integer("transaction_bytes", transaction_bytes)
    commands: list[CompactionTraceCommand] = []
    read_bytes = 0
    write_bytes = 0
    for move in result.moves:
        if move.old_start % transaction_bytes or move.new_start % transaction_bytes:
            raise ConfigurationError(
                "compaction addresses must align to the trace transaction size"
            )
        for offset in range(0, move.allocated_bytes, transaction_bytes):
            size = min(transaction_bytes, move.allocated_bytes - offset)
            commands.append(
                CompactionTraceCommand(
                    sequence=len(commands),
                    block_id=move.block_id,
                    kind=CompactionCommandKind.READ,
                    address=move.old_start + offset,
                    size_bytes=size,
                    generation=move.old_generation,
                )
            )
            read_bytes += size
            commands.append(
                CompactionTraceCommand(
                    sequence=len(commands),
                    block_id=move.block_id,
                    kind=CompactionCommandKind.WRITE,
                    address=move.new_start + offset,
                    size_bytes=size,
                    generation=move.new_generation,
                )
            )
            write_bytes += size
    return CompactionTrace(
        commands=tuple(commands),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        transaction_bytes=transaction_bytes,
    )
