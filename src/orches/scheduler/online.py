"""Technique 1B online balancing across variable attention fragments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from ..errors import ConfigurationError
from ..models.timing import PaperGpuRates, PaperPimRates, PaperTimingBreakdown


@dataclass(frozen=True)
class AttentionFragment:
    """One KV fragment with paper variables `(W_i, L_i, D)`."""

    fragment_id: str
    branch_width: int
    length_tokens: int
    hidden_size: int

    def __post_init__(self) -> None:
        if not self.fragment_id.strip():
            raise ConfigurationError("fragment_id must not be empty")
        for name, value in (
            ("branch_width", self.branch_width),
            ("length_tokens", self.length_tokens),
            ("hidden_size", self.hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"fragment.{name} must be positive")


@dataclass(frozen=True)
class OnlineBalanceDecision:
    """T1B alpha vector and resulting resource imbalance."""

    alphas_gpu: dict[str, float]
    critical_fragment_id: str | None
    gpu: PaperTimingBreakdown
    pim: PaperTimingBreakdown
    critical_path_s: float
    imbalance_stall_s: float
    ordered_fragment_ids: tuple[str, ...]
    reason: str


def _validate_fragments(fragments: tuple[AttentionFragment, ...]) -> None:
    if not fragments:
        raise ConfigurationError("at least one attention fragment is required")
    ids = [fragment.fragment_id for fragment in fragments]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("attention fragment IDs must be unique")
    hidden_sizes = {fragment.hidden_size for fragment in fragments}
    if len(hidden_sizes) != 1:
        raise ConfigurationError("all attention fragments must share hidden_size D")


def attention_fragment_times(
    fragments: tuple[AttentionFragment, ...],
    alphas_gpu: Mapping[str, float],
    gpu_rates: PaperGpuRates,
    pim_rates: PaperPimRates,
) -> tuple[PaperTimingBreakdown, PaperTimingBreakdown]:
    """Evaluate the paper's multi-fragment GPU/PIM attention equations."""

    _validate_fragments(fragments)
    if set(alphas_gpu) != {fragment.fragment_id for fragment in fragments}:
        raise ConfigurationError(
            "alpha mapping must contain every fragment exactly once"
        )

    gpu_compute = 0.0
    gpu_memory = 0.0
    pim_compute = 0.0
    pim_internal = 0.0
    pim_io = 0.0
    for fragment in fragments:
        alpha = alphas_gpu[fragment.fragment_id]
        if not isfinite(alpha) or alpha < 0 or alpha > 1:
            raise ConfigurationError("fragment alphas must be finite and in [0, 1]")
        width = float(fragment.branch_width)
        length = float(fragment.length_tokens)
        hidden = float(fragment.hidden_size)
        operations = width * length * hidden

        pim_compute += operations * (1 - alpha) / pim_rates.compute_mac_per_s
        pim_internal += length * hidden / pim_rates.internal_memory_elements_per_s
        pim_io += (
            width * hidden + width * length * (1 - alpha)
        ) / pim_rates.host_io_elements_per_s

        gpu_compute += operations * alpha / gpu_rates.compute_mac_per_s
        gpu_memory += (
            width * hidden + width * length * alpha
        ) / gpu_rates.memory_elements_per_s

    return (
        PaperTimingBreakdown(compute_s=gpu_compute, memory_s=gpu_memory),
        PaperTimingBreakdown(
            compute_s=pim_compute,
            internal_memory_s=pim_internal,
            host_io_s=pim_io,
        ),
    )


def _decision(
    fragments: tuple[AttentionFragment, ...],
    alphas: dict[str, float],
    critical_fragment_id: str | None,
    ordered_ids: tuple[str, ...],
    gpu_rates: PaperGpuRates,
    pim_rates: PaperPimRates,
    reason: str,
) -> OnlineBalanceDecision:
    gpu, pim = attention_fragment_times(fragments, alphas, gpu_rates, pim_rates)
    return OnlineBalanceDecision(
        alphas_gpu=dict(alphas),
        critical_fragment_id=critical_fragment_id,
        gpu=gpu,
        pim=pim,
        critical_path_s=max(gpu.total_s, pim.total_s),
        imbalance_stall_s=abs(gpu.total_s - pim.total_s),
        ordered_fragment_ids=ordered_ids,
        reason=reason,
    )


def balance_attention_fragments(
    fragments: tuple[AttentionFragment, ...],
    gpu_rates: PaperGpuRates,
    pim_rates: PaperPimRates,
) -> OnlineBalanceDecision:
    """Apply T1B's width ordering and solve at most one continuous alpha."""

    _validate_fragments(fragments)
    ordered = tuple(
        sorted(fragments, key=lambda item: (item.branch_width, item.fragment_id))
    )
    ordered_ids = tuple(fragment.fragment_id for fragment in ordered)
    alphas = {fragment.fragment_id: 1.0 for fragment in fragments}
    gpu_before, pim_before = attention_fragment_times(
        fragments, alphas, gpu_rates, pim_rates
    )

    if pim_before.total_s >= gpu_before.total_s:
        return _decision(
            fragments,
            alphas,
            None,
            ordered_ids,
            gpu_rates,
            pim_rates,
            "PIM fixed/internal work is already on the critical path; "
            "keep fragments on GPU",
        )

    previous_difference = pim_before.total_s - gpu_before.total_s
    for fragment in ordered:
        alphas[fragment.fragment_id] = 0.0
        gpu_after, pim_after = attention_fragment_times(
            fragments, alphas, gpu_rates, pim_rates
        )
        difference = pim_after.total_s - gpu_after.total_s
        if difference >= 0 and previous_difference <= 0:
            width = float(fragment.branch_width)
            length = float(fragment.length_tokens)
            hidden = float(fragment.hidden_size)
            operation = width * length * hidden
            pim_alpha_slope = -(
                operation / pim_rates.compute_mac_per_s
                + width * length / pim_rates.host_io_elements_per_s
            )
            gpu_alpha_slope = (
                operation / gpu_rates.compute_mac_per_s
                + width * length / gpu_rates.memory_elements_per_s
            )
            denominator = gpu_alpha_slope - pim_alpha_slope
            if denominator <= 0:
                raise ConfigurationError("critical fragment has invalid alpha slopes")

            # The all-PIM endpoint for this fragment is the current state.
            alpha = difference / denominator
            alphas[fragment.fragment_id] = min(1.0, max(0.0, alpha))
            return _decision(
                fragments,
                alphas,
                fragment.fragment_id,
                ordered_ids,
                gpu_rates,
                pim_rates,
                "binary width-ordered assignments crossed; solved the "
                "critical fragment",
            )
        previous_difference = difference

    return _decision(
        fragments,
        alphas,
        None,
        ordered_ids,
        gpu_rates,
        pim_rates,
        "all fragments moved to PIM without crossing; retain the all-PIM endpoint",
    )
