"""Fairness contracts and normalized baseline comparison records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Mapping

from ..errors import ConfigurationError
from .definitions import BASELINES, BaselineKind


class RunStatus(str, Enum):
    """A run remains visible even when it cannot produce metrics."""

    SUCCESS = "success"
    OOM = "oom"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True)
class FairnessContract:
    """Inputs that must be identical before normalized results are comparable."""

    trace_sha256: str
    request_ids: tuple[str, ...]
    policy_model: str
    policy_revision: str
    small_prm_model: str
    small_prm_revision: str
    large_prm_model: str
    large_prm_revision: str
    tokenizer: str
    tokenizer_revision: str
    weight_bytes_per_element: int
    activation_bytes_per_element: int
    gpu_count: int
    soc_bandwidth_bytes_per_s: float
    pim_capacity_bytes: int
    hardware_contract_sha256: str

    def __post_init__(self) -> None:
        string_fields = (
            "trace_sha256",
            "policy_model",
            "policy_revision",
            "small_prm_model",
            "small_prm_revision",
            "large_prm_model",
            "large_prm_revision",
            "tokenizer",
            "tokenizer_revision",
            "hardware_contract_sha256",
        )
        for name in string_fields:
            if not str(getattr(self, name)).strip():
                raise ConfigurationError(f"fairness.{name} must not be empty")
        if not self.request_ids or any(not item.strip() for item in self.request_ids):
            raise ConfigurationError("fairness.request_ids must not be empty")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ConfigurationError("fairness.request_ids must be unique")
        for name in (
            "weight_bytes_per_element",
            "activation_bytes_per_element",
            "gpu_count",
            "pim_capacity_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"fairness.{name} must be positive")
        if (
            not isfinite(self.soc_bandwidth_bytes_per_s)
            or self.soc_bandwidth_bytes_per_s <= 0
        ):
            raise ConfigurationError(
                "fairness.soc_bandwidth_bytes_per_s must be finite and positive"
            )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BaselineMetrics:
    """SI-unit metrics accepted from analytical or native adapters."""

    latency_s: float
    energy_j: float | None = None
    peak_memory_bytes: int | None = None
    utilization: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.latency_s) or self.latency_s <= 0:
            raise ConfigurationError("metrics.latency_s must be finite and positive")
        if self.energy_j is not None and (
            not isfinite(self.energy_j) or self.energy_j < 0
        ):
            raise ConfigurationError(
                "metrics.energy_j must be finite and non-negative"
            )
        if self.peak_memory_bytes is not None and (
            isinstance(self.peak_memory_bytes, bool)
            or not isinstance(self.peak_memory_bytes, int)
            or self.peak_memory_bytes < 0
        ):
            raise ConfigurationError(
                "metrics.peak_memory_bytes must be a non-negative integer"
            )
        for name, value in self.utilization.items():
            if not name.strip() or not isfinite(value) or value < 0 or value > 1:
                raise ConfigurationError(
                    "metric utilization names must be non-empty and values in [0, 1]"
                )


@dataclass(frozen=True)
class BaselineRunResult:
    """One baseline outcome, including failures that must not be dropped."""

    baseline: BaselineKind
    contract: FairnessContract
    status: RunStatus
    source_revision: str
    metrics: BaselineMetrics | None = None
    error: str | None = None
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_revision.strip():
            raise ConfigurationError("source_revision must not be empty")
        if self.status is RunStatus.SUCCESS:
            if self.metrics is None or self.error is not None:
                raise ConfigurationError(
                    "successful runs require metrics and must not contain an error"
                )
        elif self.metrics is not None or not (self.error and self.error.strip()):
            raise ConfigurationError(
                "non-success runs require an error and must not contain metrics"
            )
        for name, value in self.components.items():
            if not name.strip() or not isfinite(value) or value < 0:
                raise ConfigurationError(
                    "component names must be non-empty and values non-negative"
                )


@dataclass(frozen=True)
class ComparisonRow:
    """Normalized values retain status instead of omitting failed systems."""

    baseline: BaselineKind
    label: str
    status: RunStatus
    latency_s: float | None
    speedup_vs_gpu: float | None
    energy_j: float | None
    energy_efficiency_vs_gpu: float | None
    error: str | None


def compare_baselines(
    results: tuple[BaselineRunResult, ...],
    *,
    required: tuple[BaselineKind, ...],
) -> tuple[ComparisonRow, ...]:
    """Validate fairness and normalize every required row to the GPU result."""

    if len(set(required)) != len(required):
        raise ConfigurationError("required baseline kinds must be unique")
    by_baseline: dict[BaselineKind, BaselineRunResult] = {}
    for result in results:
        if result.baseline in by_baseline:
            raise ConfigurationError(
                f"duplicate baseline result {result.baseline.value!r}"
            )
        by_baseline[result.baseline] = result
    missing = [baseline.value for baseline in required if baseline not in by_baseline]
    if missing:
        raise ConfigurationError(
            f"missing required baseline results: {', '.join(missing)}"
        )
    fingerprints = {
        by_baseline[baseline].contract.fingerprint for baseline in required
    }
    if len(fingerprints) != 1:
        raise ConfigurationError(
            "baseline results do not share one fairness-contract fingerprint"
        )
    gpu = by_baseline.get(BaselineKind.GPU)
    if gpu is None or gpu.status is not RunStatus.SUCCESS or gpu.metrics is None:
        raise ConfigurationError(
            "a successful GPU result is required for normalization"
        )

    rows: list[ComparisonRow] = []
    for baseline in required:
        result = by_baseline[baseline]
        metrics = result.metrics
        speedup = (
            gpu.metrics.latency_s / metrics.latency_s
            if metrics is not None
            else None
        )
        energy_efficiency = None
        if (
            metrics is not None
            and metrics.energy_j is not None
            and metrics.energy_j > 0
            and gpu.metrics.energy_j is not None
        ):
            energy_efficiency = gpu.metrics.energy_j / metrics.energy_j
        rows.append(
            ComparisonRow(
                baseline=baseline,
                label=BASELINES[baseline].label,
                status=result.status,
                latency_s=metrics.latency_s if metrics is not None else None,
                speedup_vs_gpu=speedup,
                energy_j=metrics.energy_j if metrics is not None else None,
                energy_efficiency_vs_gpu=energy_efficiency,
                error=result.error,
            )
        )
    return tuple(rows)
