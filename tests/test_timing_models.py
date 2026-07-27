from __future__ import annotations

import pytest

from orches.errors import ConfigurationError
from orches.models import (
    GpuRooflineRates,
    PaperGpuRates,
    PaperPimRates,
    gpu_roofline_time,
    linear_cost,
    paper_coprocessed_linear_time,
    paper_gpu_linear_time,
    paper_pim_linear_time,
)


GPU_RATES = PaperGpuRates(compute_mac_per_s=16.0, memory_elements_per_s=8.0)
PIM_RATES = PaperPimRates(
    compute_mac_per_s=8.0,
    internal_memory_elements_per_s=2.0,
    host_io_elements_per_s=4.0,
)


def test_paper_gpu_linear_equation_matches_hand_calculation() -> None:
    timing = paper_gpu_linear_time(branch_width=2, hidden_size=4, rates=GPU_RATES)

    assert timing.compute_s == 2.0
    assert timing.memory_s == 4.0
    assert timing.total_s == 6.0


def test_paper_pim_linear_equation_matches_hand_calculation() -> None:
    timing = paper_pim_linear_time(branch_width=2, hidden_size=4, rates=PIM_RATES)

    assert timing.compute_s == 4.0
    assert timing.host_io_s == 4.0
    assert timing.internal_memory_s == 8.0
    assert timing.total_s == 16.0


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_coprocessed_equations_are_finite_across_alpha(alpha: float) -> None:
    timing = paper_coprocessed_linear_time(
        branch_width=2,
        hidden_size=4,
        alpha=alpha,
        gpu_rates=GPU_RATES,
        pim_rates=PIM_RATES,
    )

    assert timing.gpu.total_s >= 0
    assert timing.pim.total_s >= 0
    assert timing.critical_path_s == max(timing.gpu.total_s, timing.pim.total_s)


def test_coprocessed_alpha_one_matches_base_gpu_components() -> None:
    base = paper_gpu_linear_time(2, 4, GPU_RATES)
    split = paper_coprocessed_linear_time(2, 4, 1.0, GPU_RATES, PIM_RATES)

    assert split.gpu.compute_s == base.compute_s
    assert split.gpu.memory_s == base.memory_s


def test_invalid_alpha_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="alpha"):
        paper_coprocessed_linear_time(2, 4, 1.1, GPU_RATES, PIM_RATES)


def test_roofline_uses_conventional_flops_and_max_overlap() -> None:
    operator = linear_cost(
        "linear", rows=2, input_features=4, output_features=4, bytes_per_element=2
    )
    rates = GpuRooflineRates(
        compute_flop_per_s=32.0,
        memory_bytes_per_s=16.0,
        launch_overhead_s=0.25,
        synchronization_overhead_s=0.5,
    )
    estimate = gpu_roofline_time(operator, rates)

    assert estimate.compute_s == operator.flops / 32.0
    assert estimate.memory_s == operator.total_bytes / 16.0
    assert estimate.total_s == max(estimate.compute_s, estimate.memory_s) + 0.75


def test_bandwidth_scaling_does_not_change_compute_rate() -> None:
    base = GpuRooflineRates(compute_flop_per_s=100.0, memory_bytes_per_s=200.0)
    scaled = base.with_bandwidth_scale(0.5)

    assert scaled.compute_flop_per_s == base.compute_flop_per_s
    assert scaled.memory_bytes_per_s == 100.0
