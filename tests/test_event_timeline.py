from __future__ import annotations

import pytest

from orches.errors import ConfigurationError
from orches.sim import EventTimeline, Resource


def test_same_resource_serializes_but_different_resources_overlap() -> None:
    timeline = EventTimeline()
    gpu_first = timeline.schedule("gpu-1", Resource.GPU, 2.0)
    pim = timeline.schedule("pim-1", Resource.PIM, 3.0)
    gpu_second = timeline.schedule("gpu-2", Resource.GPU, 1.0)

    assert gpu_first.start_s == 0.0
    assert pim.start_s == 0.0
    assert gpu_second.start_s == 2.0
    assert timeline.makespan_s == 3.0
    assert timeline.utilization(Resource.GPU) == 1.0
    timeline.validate()


def test_dependency_delays_event_across_resources() -> None:
    timeline = EventTimeline()
    source = timeline.schedule("source", Resource.PIM, 2.5)
    consumer = timeline.schedule(
        "consumer",
        Resource.GPU,
        1.0,
        dependencies=(source.event_id,),
    )

    assert consumer.start_s == source.end_s


def test_unscheduled_dependency_is_rejected() -> None:
    timeline = EventTimeline()

    with pytest.raises(ConfigurationError, match="unscheduled dependencies"):
        timeline.schedule(
            "consumer",
            Resource.GPU,
            1.0,
            dependencies=("missing",),
        )
