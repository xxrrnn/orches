"""Compose T1A linear and T1B attention decisions for one generation step."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError
from ..models.timing import paper_pim_linear_time
from .offline import LinearPlacement, OfflineAssignment, OfflineScheduler
from .online import (
    AttentionFragment,
    OnlineBalanceDecision,
    attention_fragment_times,
    balance_attention_fragments,
)


@dataclass(frozen=True)
class GenerationPlan:
    """Per-resource duration of one repeated linear-plus-attention unit."""

    offline: OfflineAssignment
    online: OnlineBalanceDecision
    fragments: tuple[AttentionFragment, ...]
    repetitions: int
    gpu_time_s: float
    pim_time_s: float
    pim_only_time_s: float

    @property
    def critical_path_s(self) -> float:
        return max(self.gpu_time_s, self.pim_time_s)


def plan_generation_step(
    scheduler: OfflineScheduler,
    *,
    branch_width: int,
    shared_kv_tokens: int,
    unique_kv_tokens: tuple[int, ...],
    hidden_size: int,
    repetitions: int,
) -> GenerationPlan:
    """Plan T1 work while retaining every fragment and timing decision."""

    if len(unique_kv_tokens) != branch_width:
        raise ConfigurationError(
            "one unique-KV length is required for every branch"
        )
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length < 0
        for length in unique_kv_tokens
    ):
        raise ConfigurationError("unique-KV lengths must be non-negative integers")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ConfigurationError("repetitions must be a positive integer")

    fragments = (
        AttentionFragment(
            "shared",
            branch_width=branch_width,
            length_tokens=shared_kv_tokens,
            hidden_size=hidden_size,
        ),
        *(
            AttentionFragment(
                f"unique-{index:04d}",
                branch_width=1,
                length_tokens=length,
                hidden_size=hidden_size,
            )
            for index, length in enumerate(unique_kv_tokens)
            if length > 0
        ),
    )
    offline = scheduler.assign(branch_width, hidden_size)
    online = balance_attention_fragments(
        fragments,
        scheduler.gpu_rates,
        scheduler.pim_rates,
    )

    if offline.linear is LinearPlacement.COPROCESSED:
        linear_gpu_s = offline.balance.gpu.total_s
        linear_pim_s = offline.balance.pim.total_s
    elif offline.linear is LinearPlacement.GPU:
        linear_gpu_s = offline.gpu_only_time_s
        linear_pim_s = 0.0
    else:
        linear_gpu_s = 0.0
        linear_pim_s = offline.pim_only_time_s

    _, all_pim_attention = attention_fragment_times(
        fragments,
        {fragment.fragment_id: 0.0 for fragment in fragments},
        scheduler.gpu_rates,
        scheduler.pim_rates,
    )
    all_pim_linear = paper_pim_linear_time(
        branch_width,
        hidden_size,
        scheduler.pim_rates,
    )
    return GenerationPlan(
        offline=offline,
        online=online,
        fragments=fragments,
        repetitions=repetitions,
        gpu_time_s=(linear_gpu_s + online.gpu.total_s) * repetitions,
        pim_time_s=(linear_pim_s + online.pim.total_s) * repetitions,
        pim_only_time_s=(
            all_pim_linear.total_s + all_pim_attention.total_s
        )
        * repetitions,
    )
