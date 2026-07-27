from __future__ import annotations

import itertools

import pytest

from orches.models import PaperGpuRates, PaperPimRates
from orches.scheduler import (
    AttentionFragment,
    attention_fragment_times,
    balance_attention_fragments,
)


GPU_RATES = PaperGpuRates(compute_mac_per_s=80.0, memory_elements_per_s=60.0)
PIM_RATES = PaperPimRates(
    compute_mac_per_s=50.0,
    internal_memory_elements_per_s=200.0,
    host_io_elements_per_s=100.0,
)
FRAGMENTS = (
    AttentionFragment("narrow", branch_width=1, length_tokens=12, hidden_size=8),
    AttentionFragment("medium", branch_width=3, length_tokens=8, hidden_size=8),
    AttentionFragment("wide", branch_width=6, length_tokens=4, hidden_size=8),
)


def test_fragment_equations_match_manual_terms() -> None:
    fragment = (AttentionFragment("f", 2, 3, 4),)
    gpu, pim = attention_fragment_times(
        fragment,
        {"f": 0.25},
        GPU_RATES,
        PIM_RATES,
    )

    assert gpu.compute_s == pytest.approx(2 * 3 * 4 * 0.25 / 80)
    assert gpu.memory_s == pytest.approx((2 * 4 + 2 * 3 * 0.25) / 60)
    assert pim.compute_s == pytest.approx(2 * 3 * 4 * 0.75 / 50)
    assert pim.internal_memory_s == pytest.approx(3 * 4 / 200)
    assert pim.host_io_s == pytest.approx((2 * 4 + 2 * 3 * 0.75) / 100)


def test_online_balance_orders_fragments_by_width() -> None:
    decision = balance_attention_fragments(FRAGMENTS, GPU_RATES, PIM_RATES)

    assert decision.ordered_fragment_ids == ("narrow", "medium", "wide")
    assert sum(0 < alpha < 1 for alpha in decision.alphas_gpu.values()) <= 1
    assert decision.critical_path_s == max(
        decision.gpu.total_s, decision.pim.total_s
    )
    assert decision.imbalance_stall_s == pytest.approx(
        abs(decision.gpu.total_s - decision.pim.total_s)
    )


def test_online_balance_matches_small_grid_optimum() -> None:
    decision = balance_attention_fragments(FRAGMENTS, GPU_RATES, PIM_RATES)
    grid = tuple(index / 20 for index in range(21))
    brute_force = float("inf")
    for values in itertools.product(grid, repeat=len(FRAGMENTS)):
        alphas = {
            fragment.fragment_id: alpha
            for fragment, alpha in zip(FRAGMENTS, values)
        }
        gpu, pim = attention_fragment_times(FRAGMENTS, alphas, GPU_RATES, PIM_RATES)
        brute_force = min(brute_force, max(gpu.total_s, pim.total_s))

    assert decision.critical_path_s <= brute_force + 0.05


def test_shared_growth_can_change_online_assignment() -> None:
    short = (
        AttentionFragment("shared", 4, 4, 8),
        AttentionFragment("unique", 1, 2, 8),
    )
    long = (
        AttentionFragment("shared", 4, 64, 8),
        AttentionFragment("unique", 1, 2, 8),
    )

    short_decision = balance_attention_fragments(short, GPU_RATES, PIM_RATES)
    long_decision = balance_attention_fragments(long, GPU_RATES, PIM_RATES)

    assert (
        short_decision.alphas_gpu != long_decision.alphas_gpu
        or short_decision.critical_path_s != long_decision.critical_path_s
    )
