"""Provenance metadata attached to every architecture parameter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class EvidenceStatus(str, Enum):
    """How directly a configuration value is supported by available evidence."""

    PAPER = "PAPER"
    INHERITED = "INHERITED"
    CALIBRATED = "CALIBRATED"
    ASSUMED = "ASSUMED"


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class SourcedValue(Generic[ValueT]):
    """A value together with its source and evidence strength."""

    value: ValueT
    source: str
    status: EvidenceStatus
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("A sourced value must have a non-empty source")
        if self.note is not None and not self.note.strip():
            raise ValueError("A sourced value note must be non-empty when provided")
