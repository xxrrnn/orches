"""Source-aware area accounting with an explicit overhead denominator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..errors import ConfigurationError
from ..provenance import EvidenceStatus


@dataclass(frozen=True)
class AreaComponent:
    """Area of one added or inherited hardware structure in square millimeters."""

    component_id: str
    area_mm2: float
    source: str
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if not self.component_id.strip() or not self.source.strip():
            raise ConfigurationError("area component ID and source must not be empty")
        if not isfinite(self.area_mm2) or self.area_mm2 < 0:
            raise ConfigurationError("component area must be finite and non-negative")


@dataclass(frozen=True)
class AreaReport:
    """Added area and overhead relative to one named baseline die area."""

    baseline_name: str
    baseline_area_mm2: float
    components: tuple[AreaComponent, ...]

    def __post_init__(self) -> None:
        if not self.baseline_name.strip():
            raise ConfigurationError("area baseline name must not be empty")
        if not isfinite(self.baseline_area_mm2) or self.baseline_area_mm2 <= 0:
            raise ConfigurationError(
                "area baseline denominator must be finite and positive"
            )
        ids = [component.component_id for component in self.components]
        if len(set(ids)) != len(ids):
            raise ConfigurationError("area component IDs must be unique")

    @property
    def added_area_mm2(self) -> float:
        return sum(component.area_mm2 for component in self.components)

    @property
    def total_area_mm2(self) -> float:
        return self.baseline_area_mm2 + self.added_area_mm2

    @property
    def overhead_fraction(self) -> float:
        return self.added_area_mm2 / self.baseline_area_mm2
