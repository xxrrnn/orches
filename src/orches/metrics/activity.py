"""Convert ORCHES scheduler/replay decisions into energy activity counters."""

from __future__ import annotations

from math import isfinite

from ..errors import ConfigurationError
from ..replay import OrchesReplayResult
from ..scheduler import GenerationPlan, LinearPlacement
from .energy import EnergyActivity


def _validate_fraction(work_fraction: float) -> None:
    if not isfinite(work_fraction) or work_fraction < 0 or work_fraction > 1:
        raise ConfigurationError("work_fraction must be finite and in [0, 1]")


def generation_plan_activity(
    plan: GenerationPlan,
    *,
    work_fraction: float,
    bytes_per_element: int,
    force_all_pim: bool = False,
) -> EnergyActivity:
    """Apply Eq. (1)-(7) alpha decisions to MAC and element counters."""

    _validate_fraction(work_fraction)
    if (
        isinstance(bytes_per_element, bool)
        or not isinstance(bytes_per_element, int)
        or bytes_per_element <= 0
    ):
        raise ConfigurationError("bytes_per_element must be a positive integer")
    shared = plan.fragments[0]
    width = float(shared.branch_width)
    hidden = float(shared.hidden_size)
    scale = plan.repetitions * work_fraction

    if force_all_pim:
        linear_alpha = 0.0
        linear_uses_pim = True
    elif plan.offline.linear is LinearPlacement.COPROCESSED:
        linear_alpha = plan.offline.balance.alpha
        linear_uses_pim = True
    elif plan.offline.linear is LinearPlacement.GPU:
        linear_alpha = 1.0
        linear_uses_pim = False
    else:
        linear_alpha = 0.0
        linear_uses_pim = True

    linear_macs = width * hidden * hidden
    gpu_macs = linear_macs * linear_alpha
    pim_macs = linear_macs * (1 - linear_alpha)
    gpu_memory_elements = width * hidden * (1 + linear_alpha)
    gpu_memory_elements += hidden * hidden * linear_alpha
    pim_memory_elements = hidden * hidden if linear_uses_pim else 0.0
    host_link_elements = (
        width * hidden * (2 - linear_alpha) if linear_uses_pim else 0.0
    )

    for fragment in plan.fragments:
        alpha = (
            0.0
            if force_all_pim
            else plan.online.alphas_gpu[fragment.fragment_id]
        )
        fragment_width = float(fragment.branch_width)
        length = float(fragment.length_tokens)
        operations = fragment_width * length * hidden
        gpu_macs += operations * alpha
        pim_macs += operations * (1 - alpha)
        gpu_memory_elements += (
            fragment_width * hidden + fragment_width * length * alpha
        )
        pim_memory_elements += length * hidden
        host_link_elements += fragment_width * hidden
        host_link_elements += fragment_width * length * (1 - alpha)

    return EnergyActivity(
        gpu_macs=gpu_macs * scale,
        gpu_memory_bytes=gpu_memory_elements * bytes_per_element * scale,
        pim_macs=pim_macs * scale,
        pim_memory_bytes=pim_memory_elements * bytes_per_element * scale,
        host_link_bytes=host_link_elements * bytes_per_element * scale,
    )


def replay_activity(
    result: OrchesReplayResult,
    *,
    bytes_per_element: int,
    cache_entry_bytes: int,
    verifier_activity: EnergyActivity | None = None,
) -> EnergyActivity:
    """Aggregate generation, speculation, T3, and explicit verifier activity."""

    if (
        isinstance(cache_entry_bytes, bool)
        or not isinstance(cache_entry_bytes, int)
        or cache_entry_bytes <= 0
    ):
        raise ConfigurationError("cache_entry_bytes must be a positive integer")
    total = verifier_activity or EnergyActivity()
    for step in result.steps:
        generation_plans = step.generation.phase_plans or (step.generation.plan,)
        for plan in generation_plans:
            total += generation_plan_activity(
                plan,
                work_fraction=step.generation.t1_work_fraction,
                bytes_per_element=bytes_per_element,
            )
        speculative_plans = step.speculative_plans or (
            (step.speculative_plan,) if step.speculative_plan is not None else ()
        )
        for plan in speculative_plans:
            total += generation_plan_activity(
                plan,
                work_fraction=step.speculative_work_fraction,
                bytes_per_element=bytes_per_element,
                force_all_pim=True,
            )

    lookup_sram_accesses = sum(
        lookup.lookup.sram_accesses
        for step in result.steps
        for lookup in step.address_lookups
    )
    lookup_dram_accesses = sum(
        lookup.lookup.dram_accesses
        for step in result.steps
        for lookup in step.address_lookups
    )
    return total + EnergyActivity(
        pim_memory_bytes=lookup_dram_accesses * cache_entry_bytes,
        host_link_bytes=result.shared_kv_buffer.gpu_sync_bytes,
        controller_sram_bytes=lookup_sram_accesses * cache_entry_bytes,
        controller_buffer_bytes=(
            result.shared_kv_buffer.controller_to_bank_bytes
        ),
        compaction_read_bytes=result.compaction_read_bytes,
        compaction_write_bytes=result.compaction_write_bytes,
    )
