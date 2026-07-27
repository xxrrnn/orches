"""Adapter from the full ORCHES request replay to the common result contract."""

from __future__ import annotations

from dataclasses import fields

from ..errors import ConfigurationError
from ..metrics import (
    EnergyActivity,
    EnergyRates,
    calculate_energy,
    replay_activity,
    utilization_from_timeline,
)
from ..replay import OrchesReplayResult
from .definitions import BaselineKind
from .results import (
    BaselineMetrics,
    BaselineRunResult,
    FairnessContract,
    RunStatus,
)


def adapt_orches_replay(
    result: OrchesReplayResult,
    *,
    contract: FairnessContract,
    source_revision: str,
    energy_rates: EnergyRates,
    bytes_per_element: int,
    cache_entry_bytes: int,
    verifier_activity: EnergyActivity,
) -> BaselineRunResult:
    """Account a full replay without inventing missing PRM activity."""

    if result.request_id not in contract.request_ids:
        raise ConfigurationError(
            "ORCHES replay request is absent from the fairness contract"
        )
    activity = replay_activity(
        result,
        bytes_per_element=bytes_per_element,
        cache_entry_bytes=cache_entry_bytes,
        verifier_activity=verifier_activity,
    )
    energy = calculate_energy(activity, energy_rates)
    utilization = utilization_from_timeline(result.timeline)
    components = {
        f"energy.{item.name}": getattr(energy, item.name)
        for item in fields(energy)
    }
    components.update(
        {
            f"activity.{item.name}": getattr(activity, item.name)
            for item in fields(activity)
        }
    )
    return BaselineRunResult(
        baseline=BaselineKind.ORCHES,
        contract=contract,
        status=RunStatus.SUCCESS,
        source_revision=source_revision,
        metrics=BaselineMetrics(
            latency_s=result.timeline.makespan_s,
            energy_j=energy.total_j,
            peak_memory_bytes=(
                result.final_memory.peak_allocated_footprint_bytes
            ),
            utilization={
                item.resource: item.utilization for item in utilization.resources
            },
        ),
        components=components,
    )
