from __future__ import annotations

import pytest

from orches.errors import ConfigurationError
from orches.metrics import (
    AreaComponent,
    AreaReport,
    EnergyActivity,
    attacc_bank_level_energy_rates,
    calculate_energy,
    generation_plan_activity,
    utilization_from_busy_times,
    utilization_from_timeline,
)
from orches.models import PaperGpuRates, PaperPimRates
from orches.provenance import EvidenceStatus
from orches.sim import EventTimeline, Resource
from orches.scheduler import OfflineScheduler, TierThresholds, plan_generation_step


def test_attacc_energy_constants_preserve_units_and_provenance() -> None:
    rates = attacc_bank_level_energy_rates()

    assert rates.gpu_compute_pj_per_mac.value_pj == 0.32
    assert rates.gpu_compute_pj_per_mac.per == "mac"
    assert rates.gpu_memory_pj_per_byte.value_pj == pytest.approx(28.72)
    assert rates.pim_memory_pj_per_byte.value_pj == pytest.approx(4.4)
    assert rates.host_link_pj_per_byte.value_pj == 10.4
    assert rates.pim_compute_pj_per_mac.status is EvidenceStatus.INHERITED
    assert "c600051" in rates.pim_compute_pj_per_mac.source


def test_energy_is_activity_times_picojoules_converted_to_joules() -> None:
    rates = attacc_bank_level_energy_rates()
    activity = EnergyActivity(
        gpu_macs=100,
        gpu_memory_bytes=10,
        pim_macs=50,
        pim_memory_bytes=20,
        host_link_bytes=5,
        controller_sram_bytes=4,
        controller_buffer_bytes=6,
        compaction_read_bytes=8,
        compaction_write_bytes=8,
    )

    energy = calculate_energy(activity, rates)

    assert energy.gpu_compute_j == pytest.approx(100 * 0.32e-12)
    assert energy.gpu_memory_j == pytest.approx(10 * 28.72e-12)
    assert energy.pim_compute_j == pytest.approx(50 * 0.32e-12)
    assert energy.pim_memory_j == pytest.approx(20 * 4.4e-12)
    assert energy.compaction_j == pytest.approx(16 * 4.4e-12)
    assert energy.total_j == pytest.approx(
        energy.gpu_compute_j
        + energy.gpu_memory_j
        + energy.pim_compute_j
        + energy.pim_memory_j
        + energy.host_link_j
        + energy.controller_sram_j
        + energy.controller_buffer_j
        + energy.compaction_j
    )


def test_empty_activity_has_exactly_zero_energy_and_addition_is_componentwise() -> None:
    rates = attacc_bank_level_energy_rates()
    combined = EnergyActivity(gpu_macs=2) + EnergyActivity(pim_macs=3)

    assert calculate_energy(EnergyActivity(), rates).total_j == 0
    assert combined.gpu_macs == 2
    assert combined.pim_macs == 3


def test_area_overhead_uses_declared_denominator() -> None:
    report = AreaReport(
        baseline_name="PIM controller die without T3 buffer",
        baseline_area_mm2=100.0,
        components=(
            AreaComponent(
                "address-cache",
                2.0,
                "CACTI placeholder",
                EvidenceStatus.CALIBRATED,
            ),
            AreaComponent(
                "shared-kv-buffer",
                10.0,
                "synthesis placeholder",
                EvidenceStatus.CALIBRATED,
            ),
        ),
    )

    assert report.added_area_mm2 == 12.0
    assert report.total_area_mm2 == 112.0
    assert report.overhead_fraction == pytest.approx(0.12)


def test_timeline_utilization_uses_common_makespan_without_double_counting() -> None:
    timeline = EventTimeline()
    timeline.schedule("gpu", Resource.GPU, 2.0)
    timeline.schedule("pim", Resource.PIM, 4.0)

    report = utilization_from_timeline(timeline)

    assert report.makespan_s == 4.0
    assert report.by_name("gpu").busy_s == 2.0
    assert report.by_name("gpu").idle_s == 2.0
    assert report.by_name("gpu").utilization == 0.5
    assert report.by_name("pim").utilization == 1.0
    assert report.by_name("controller").utilization == 0.0


def test_external_busy_time_cannot_exceed_wall_clock() -> None:
    with pytest.raises(ConfigurationError, match="exceeds"):
        utilization_from_busy_times(1.0, {"gpu_compute": 1.1})


def test_generation_activity_conserves_linear_and_attention_macs() -> None:
    scheduler = OfflineScheduler(
        TierThresholds(3, 7),
        PaperGpuRates(200.0, 100.0),
        PaperPimRates(80.0, 200.0, 100.0),
    )
    plan = plan_generation_step(
        scheduler,
        branch_width=2,
        shared_kv_tokens=8,
        unique_kv_tokens=(3, 5),
        hidden_size=4,
        repetitions=2,
    )

    activity = generation_plan_activity(
        plan,
        work_fraction=0.5,
        bytes_per_element=2,
    )
    all_pim = generation_plan_activity(
        plan,
        work_fraction=0.5,
        bytes_per_element=2,
        force_all_pim=True,
    )

    expected_macs = (
        2 * 4 * 4
        + 2 * 8 * 4
        + 1 * 3 * 4
        + 1 * 5 * 4
    ) * 2 * 0.5
    assert activity.gpu_macs + activity.pim_macs == pytest.approx(expected_macs)
    assert all_pim.gpu_macs == 0
    assert all_pim.pim_macs == pytest.approx(expected_macs)
