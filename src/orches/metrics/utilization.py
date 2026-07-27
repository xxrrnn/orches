"""Resource utilization reports derived from non-overlapping busy intervals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from ..errors import ConfigurationError
from ..sim import EventTimeline, Resource


@dataclass(frozen=True)
class ResourceUtilization:
    """Busy, idle, and normalized utilization for one named resource."""

    resource: str
    busy_s: float
    idle_s: float
    utilization: float


@dataclass(frozen=True)
class UtilizationReport:
    """All resource utilizations under one common wall-clock makespan."""

    makespan_s: float
    resources: tuple[ResourceUtilization, ...]

    def by_name(self, resource: str) -> ResourceUtilization:
        try:
            return next(item for item in self.resources if item.resource == resource)
        except StopIteration as error:
            raise ConfigurationError(
                f"utilization report has no resource {resource!r}"
            ) from error


def utilization_from_busy_times(
    makespan_s: float,
    busy_times_s: Mapping[str, float],
) -> UtilizationReport:
    """Validate externally measured busy times and compute idle intervals."""

    if not isfinite(makespan_s) or makespan_s < 0:
        raise ConfigurationError("makespan must be finite and non-negative")
    resources: list[ResourceUtilization] = []
    for name in sorted(busy_times_s):
        busy = busy_times_s[name]
        if not name.strip() or not isfinite(busy) or busy < 0:
            raise ConfigurationError(
                "utilization resource names must be non-empty and busy time "
                "non-negative"
            )
        if busy > makespan_s:
            raise ConfigurationError(
                f"resource {name!r} busy time exceeds the common makespan"
            )
        utilization = busy / makespan_s if makespan_s else 0.0
        resources.append(
            ResourceUtilization(
                resource=name,
                busy_s=busy,
                idle_s=makespan_s - busy,
                utilization=utilization,
            )
        )
    return UtilizationReport(makespan_s, tuple(resources))


def utilization_from_timeline(timeline: EventTimeline) -> UtilizationReport:
    """Report every ORCHES resource, including resources with no events."""

    timeline.validate()
    return utilization_from_busy_times(
        timeline.makespan_s,
        {
            resource.value: timeline.busy_time_s(resource)
            for resource in Resource
        },
    )
