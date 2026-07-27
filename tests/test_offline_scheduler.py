from __future__ import annotations

import pytest

from orches.models import (
    PaperGpuRates,
    PaperPimRates,
    paper_coprocessed_linear_time,
)
from orches.scheduler import (
    AttentionPlacement,
    LinearPlacement,
    OfflineScheduler,
    TierThresholds,
    WorkloadTier,
    solve_linear_balance_alpha,
)


GPU_RATES = PaperGpuRates(compute_mac_per_s=200.0, memory_elements_per_s=100.0)
PIM_RATES = PaperPimRates(
    compute_mac_per_s=80.0,
    internal_memory_elements_per_s=200.0,
    host_io_elements_per_s=100.0,
)


def test_width_tiers_match_paper_placement_rules() -> None:
    scheduler = OfflineScheduler(TierThresholds(3, 7), GPU_RATES, PIM_RATES)

    small = scheduler.assign(2, 16)
    medium = scheduler.assign(4, 16)
    large = scheduler.assign(8, 16)

    assert small.tier is WorkloadTier.SMALL
    assert small.shared_attention is AttentionPlacement.PIM
    assert medium.tier is WorkloadTier.MEDIUM
    assert medium.shared_attention is AttentionPlacement.GPU
    assert large.tier is WorkloadTier.LARGE
    assert large.shared_attention is AttentionPlacement.GPU
    assert all(
        decision.unique_attention is AttentionPlacement.PIM
        for decision in (small, medium, large)
    )


def test_balance_alpha_matches_dense_numerical_search() -> None:
    alpha = solve_linear_balance_alpha(4, 16, GPU_RATES, PIM_RATES)
    scheduler = OfflineScheduler(TierThresholds(3, 7), GPU_RATES, PIM_RATES)
    analytic = scheduler.assign(4, 16).balance

    grid = [index / 10000 for index in range(10001)]

    def imbalance(point: float) -> float:
        timing = paper_coprocessed_linear_time(
            4, 16, point, GPU_RATES, PIM_RATES
        )
        return abs(timing.gpu.total_s - timing.pim.total_s)

    best = min(grid, key=imbalance)

    assert analytic.alpha == pytest.approx(alpha)
    assert alpha == pytest.approx(best, abs=1e-4)


def test_assignment_reports_all_linear_candidates() -> None:
    scheduler = OfflineScheduler(TierThresholds(3, 7), GPU_RATES, PIM_RATES)
    decision = scheduler.assign(4, 16)

    assert decision.linear in {
        LinearPlacement.PIM,
        LinearPlacement.COPROCESSED,
    }
    assert decision.gpu_only_time_s > 0
    assert decision.pim_only_time_s > 0
    assert decision.coprocessed_time_s > 0
    assert decision.chosen_linear_time_s <= decision.primary_linear_time_s
    assert decision.coprocessed_host_io_elements == pytest.approx(
        4 * 16 * (2 - decision.balance.alpha)
    )
    assert decision.coprocessed_barrier_stall_s == pytest.approx(
        abs(decision.balance.gpu.total_s - decision.balance.pim.total_s)
    )
    assert decision.reason


def test_coprocessing_is_rejected_when_pim_only_is_already_faster() -> None:
    scheduler = OfflineScheduler(
        TierThresholds(3, 7),
        PaperGpuRates(compute_mac_per_s=100.0, memory_elements_per_s=100.0),
        PaperPimRates(
            compute_mac_per_s=1_000_000.0,
            internal_memory_elements_per_s=1_000_000.0,
            host_io_elements_per_s=1_000_000.0,
        ),
    )

    decision = scheduler.assign(8, 16)

    assert decision.coprocessed_time_s < decision.gpu_only_time_s
    assert decision.pim_only_time_s < decision.coprocessed_time_s
    assert decision.linear is LinearPlacement.GPU
