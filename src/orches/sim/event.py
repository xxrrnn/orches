"""Deterministic multi-resource event timeline with dependency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from ..errors import ConfigurationError


class Resource(str, Enum):
    """Independent resources represented by the system simulator."""

    GPU = "gpu"
    PIM = "pim"
    CONTROLLER = "controller"
    LINK = "link"


@dataclass(frozen=True)
class Event:
    """One non-preemptive interval on exactly one resource."""

    event_id: str
    resource: Resource
    start_s: float
    duration_s: float
    dependencies: tuple[str, ...] = ()
    speculative: bool = False

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class EventTimeline:
    """Append-only scheduler that prevents resource overlap and time travel."""

    _events: list[Event] = field(default_factory=list)
    _by_id: dict[str, Event] = field(default_factory=dict)
    _resource_available_s: dict[Resource, float] = field(default_factory=dict)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def schedule(
        self,
        event_id: str,
        resource: Resource,
        duration_s: float,
        *,
        ready_time_s: float = 0.0,
        dependencies: tuple[str, ...] = (),
        speculative: bool = False,
    ) -> Event:
        if not event_id.strip():
            raise ConfigurationError("event_id must not be empty")
        if event_id in self._by_id:
            raise ConfigurationError(f"duplicate event ID {event_id!r}")
        if not isfinite(duration_s) or duration_s < 0:
            raise ConfigurationError("event duration must be finite and non-negative")
        if not isfinite(ready_time_s) or ready_time_s < 0:
            raise ConfigurationError("event ready time must be finite and non-negative")

        missing = [dependency for dependency in dependencies if dependency not in self._by_id]
        if missing:
            raise ConfigurationError(
                f"event {event_id!r} has unscheduled dependencies: {', '.join(missing)}"
            )
        dependency_end = max(
            (self._by_id[dependency].end_s for dependency in dependencies),
            default=0.0,
        )
        start = max(
            ready_time_s,
            dependency_end,
            self._resource_available_s.get(resource, 0.0),
        )
        event = Event(
            event_id=event_id,
            resource=resource,
            start_s=start,
            duration_s=duration_s,
            dependencies=dependencies,
            speculative=speculative,
        )
        self._events.append(event)
        self._by_id[event_id] = event
        self._resource_available_s[resource] = event.end_s
        return event

    def event(self, event_id: str) -> Event:
        try:
            return self._by_id[event_id]
        except KeyError as error:
            raise ConfigurationError(f"unknown event ID {event_id!r}") from error

    @property
    def makespan_s(self) -> float:
        return max((event.end_s for event in self._events), default=0.0)

    def busy_time_s(self, resource: Resource) -> float:
        return sum(
            event.duration_s for event in self._events if event.resource is resource
        )

    def utilization(self, resource: Resource) -> float:
        if self.makespan_s == 0:
            return 0.0
        return self.busy_time_s(resource) / self.makespan_s

    def validate(self) -> None:
        """Recheck dependencies and pairwise intervals for imported/reviewed timelines."""

        by_resource: dict[Resource, list[Event]] = {}
        for event in self._events:
            by_resource.setdefault(event.resource, []).append(event)
            for dependency in event.dependencies:
                if self._by_id[dependency].end_s > event.start_s:
                    raise ConfigurationError(
                        f"event {event.event_id!r} starts before dependency {dependency!r}"
                    )
        for resource, events in by_resource.items():
            ordered = sorted(events, key=lambda event: (event.start_s, event.event_id))
            for previous, current in zip(ordered, ordered[1:]):
                if previous.end_s > current.start_s:
                    raise ConfigurationError(
                        f"events overlap on {resource.value}: "
                        f"{previous.event_id!r} and {current.event_id!r}"
                    )
