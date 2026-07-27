"""Typed hardware contracts and their physical consistency checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose
from typing import Any, Iterable, Protocol

from .errors import ConfigurationError
from .provenance import EvidenceStatus, SourcedValue


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value!r}")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}")


def _require_nonempty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")


class HardwareConfig(Protocol):
    """Common behavior exposed by all hardware configuration types."""

    schema_version: int
    name: str
    kind: str

    def parameters(self) -> dict[str, SourcedValue[Any]]:
        """Return every sourced leaf parameter by its stable dotted name."""

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable validation summary."""


def evidence_counts(
    parameters: Iterable[SourcedValue[Any]],
) -> dict[str, int]:
    """Count parameters by evidence status using deterministic key order."""

    counts = Counter(parameter.status.value for parameter in parameters)
    return {status.value: counts[status.value] for status in EvidenceStatus}


def parameter_records(
    parameters: dict[str, SourcedValue[Any]],
) -> dict[str, dict[str, Any]]:
    """Convert sourced parameters into a stable, JSON-serializable mapping."""

    records: dict[str, dict[str, Any]] = {}
    for name, parameter in parameters.items():
        value = parameter.value
        records[name] = {
            "value": list(value) if isinstance(value, tuple) else value,
            "source": parameter.source,
            "status": parameter.status.value,
            "note": parameter.note,
        }
    return records


@dataclass(frozen=True)
class GpuHardwareConfig:
    """Host GPU parameters required by the ORCHES system model."""

    schema_version: int
    name: str
    kind: str
    memory_capacity_gb: SourcedValue[float]
    memory_bandwidth_gb_per_s: SourcedValue[float]
    cuda_cores: SourcedValue[int]
    tensor_cores: SourcedValue[int]
    max_gpu_frequency_mhz: SourcedValue[float]
    soc_bandwidth_scales: SourcedValue[tuple[float, ...]]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported hardware schema version {self.schema_version!r}"
            )
        if self.kind != "gpu":
            raise ConfigurationError(f"GPU configuration kind must be 'gpu', got {self.kind!r}")
        if not self.name.strip():
            raise ConfigurationError("GPU configuration name must not be empty")

        _require_positive("gpu.memory_capacity_gb", self.memory_capacity_gb.value)
        _require_positive(
            "gpu.memory_bandwidth_gb_per_s",
            self.memory_bandwidth_gb_per_s.value,
        )
        _require_positive_integer("gpu.cuda_cores", self.cuda_cores.value)
        _require_positive_integer("gpu.tensor_cores", self.tensor_cores.value)
        _require_positive(
            "gpu.max_gpu_frequency_mhz",
            self.max_gpu_frequency_mhz.value,
        )

        scales = self.soc_bandwidth_scales.value
        if not scales:
            raise ConfigurationError("gpu.soc_bandwidth_scales must not be empty")
        if any(scale <= 0 or scale > 1 for scale in scales):
            raise ConfigurationError(
                "gpu.soc_bandwidth_scales values must be in the interval (0, 1]"
            )
        if tuple(sorted(set(scales), reverse=True)) != scales:
            raise ConfigurationError(
                "gpu.soc_bandwidth_scales must be unique and in descending order"
            )
        if 1.0 not in scales:
            raise ConfigurationError("gpu.soc_bandwidth_scales must include 1.0")

    def parameters(self) -> dict[str, SourcedValue[Any]]:
        return {
            "gpu.memory_capacity_gb": self.memory_capacity_gb,
            "gpu.memory_bandwidth_gb_per_s": self.memory_bandwidth_gb_per_s,
            "gpu.cuda_cores": self.cuda_cores,
            "gpu.tensor_cores": self.tensor_cores,
            "gpu.max_gpu_frequency_mhz": self.max_gpu_frequency_mhz,
            "gpu.soc_bandwidth_scales": self.soc_bandwidth_scales,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "derived": {
                "bandwidth_points_gb_per_s": [
                    round(self.memory_bandwidth_gb_per_s.value * scale, 10)
                    for scale in self.soc_bandwidth_scales.value
                ],
            },
            "evidence": evidence_counts(self.parameters().values()),
            "parameters": parameter_records(self.parameters()),
        }


@dataclass(frozen=True)
class PimHardwareConfig:
    """PIM organization and controller parameters used by Ramulator2."""

    schema_version: int
    name: str
    kind: str
    memory_standard: SourcedValue[str]
    ramulator_organization_preset: SourcedValue[str]
    reported_capacity_gb: SourcedValue[float]
    simulated_capacity_gib: SourcedValue[float]
    channel_density_gibits: SourcedValue[float]
    channels: SourcedValue[int]
    pseudochannels_per_channel: SourcedValue[int]
    ranks_per_pseudochannel: SourcedValue[int]
    bankgroups_per_rank: SourcedValue[int]
    banks_per_bankgroup: SourcedValue[int]
    rows_per_bank: SourcedValue[int]
    columns_per_row: SourcedValue[int]
    transaction_bytes: SourcedValue[int]
    expected_bank_count: SourcedValue[int]
    gemv_lanes_per_bank: SourcedValue[int]
    host_io_bandwidth_gb_per_s: SourcedValue[float]
    ramulator_timing_preset: SourcedValue[str]
    controller_clock_ratio: SourcedValue[int]
    refresh_policy: SourcedValue[str]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported hardware schema version {self.schema_version!r}"
            )
        if self.kind != "pim":
            raise ConfigurationError(f"PIM configuration kind must be 'pim', got {self.kind!r}")
        if not self.name.strip():
            raise ConfigurationError("PIM configuration name must not be empty")

        for name, parameter in (
            ("pim.memory_standard", self.memory_standard),
            ("pim.ramulator_organization_preset", self.ramulator_organization_preset),
            ("pim.ramulator_timing_preset", self.ramulator_timing_preset),
            ("pim.refresh_policy", self.refresh_policy),
        ):
            _require_nonempty_string(name, parameter.value)

        for name, parameter in (
            ("pim.reported_capacity_gb", self.reported_capacity_gb),
            ("pim.simulated_capacity_gib", self.simulated_capacity_gib),
            ("pim.channel_density_gibits", self.channel_density_gibits),
            ("pim.host_io_bandwidth_gb_per_s", self.host_io_bandwidth_gb_per_s),
        ):
            _require_positive(name, parameter.value)

        for name, parameter in (
            ("pim.channels", self.channels),
            ("pim.pseudochannels_per_channel", self.pseudochannels_per_channel),
            ("pim.ranks_per_pseudochannel", self.ranks_per_pseudochannel),
            ("pim.bankgroups_per_rank", self.bankgroups_per_rank),
            ("pim.banks_per_bankgroup", self.banks_per_bankgroup),
            ("pim.rows_per_bank", self.rows_per_bank),
            ("pim.columns_per_row", self.columns_per_row),
            ("pim.transaction_bytes", self.transaction_bytes),
            ("pim.expected_bank_count", self.expected_bank_count),
            ("pim.gemv_lanes_per_bank", self.gemv_lanes_per_bank),
            ("pim.controller_clock_ratio", self.controller_clock_ratio),
        ):
            _require_positive_integer(name, parameter.value)

        if self.derived_bank_count != self.expected_bank_count.value:
            raise ConfigurationError(
                "PIM bank organization derives "
                f"{self.derived_bank_count} banks, but expected_bank_count is "
                f"{self.expected_bank_count.value}"
            )

        if not isclose(
            self.derived_capacity_gib,
            self.simulated_capacity_gib.value,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ConfigurationError(
                "PIM organization derives "
                f"{self.derived_capacity_gib:g} GiB, but simulated_capacity_gib is "
                f"{self.simulated_capacity_gib.value:g} GiB"
            )

        expected_address_space_bytes = int(self.simulated_capacity_gib.value * 2**30)
        if self.address_space_bytes != expected_address_space_bytes:
            raise ConfigurationError(
                "PIM address hierarchy derives "
                f"{self.address_space_bytes} bytes, but simulated capacity requires "
                f"{expected_address_space_bytes} bytes"
            )
        if self.address_space_bytes & (self.address_space_bytes - 1):
            raise ConfigurationError("PIM address space must be a power of two")

    @property
    def derived_bank_count(self) -> int:
        return (
            self.channels.value
            * self.pseudochannels_per_channel.value
            * self.ranks_per_pseudochannel.value
            * self.bankgroups_per_rank.value
            * self.banks_per_bankgroup.value
        )

    @property
    def derived_capacity_gib(self) -> float:
        return self.channels.value * self.channel_density_gibits.value / 8

    @property
    def total_gemv_lanes(self) -> int:
        return self.derived_bank_count * self.gemv_lanes_per_bank.value

    @property
    def address_space_bytes(self) -> int:
        return (
            self.derived_bank_count
            * self.rows_per_bank.value
            * self.columns_per_row.value
            * self.transaction_bytes.value
        )

    @property
    def address_bits(self) -> int:
        return self.address_space_bytes.bit_length() - 1

    def parameters(self) -> dict[str, SourcedValue[Any]]:
        return {
            "pim.memory_standard": self.memory_standard,
            "pim.ramulator_organization_preset": self.ramulator_organization_preset,
            "pim.reported_capacity_gb": self.reported_capacity_gb,
            "pim.simulated_capacity_gib": self.simulated_capacity_gib,
            "pim.channel_density_gibits": self.channel_density_gibits,
            "pim.channels": self.channels,
            "pim.pseudochannels_per_channel": self.pseudochannels_per_channel,
            "pim.ranks_per_pseudochannel": self.ranks_per_pseudochannel,
            "pim.bankgroups_per_rank": self.bankgroups_per_rank,
            "pim.banks_per_bankgroup": self.banks_per_bankgroup,
            "pim.rows_per_bank": self.rows_per_bank,
            "pim.columns_per_row": self.columns_per_row,
            "pim.transaction_bytes": self.transaction_bytes,
            "pim.expected_bank_count": self.expected_bank_count,
            "pim.gemv_lanes_per_bank": self.gemv_lanes_per_bank,
            "pim.host_io_bandwidth_gb_per_s": self.host_io_bandwidth_gb_per_s,
            "pim.ramulator_timing_preset": self.ramulator_timing_preset,
            "pim.controller_clock_ratio": self.controller_clock_ratio,
            "pim.refresh_policy": self.refresh_policy,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "derived": {
                "bank_count": self.derived_bank_count,
                "capacity_gib": self.derived_capacity_gib,
                "total_gemv_lanes": self.total_gemv_lanes,
                "address_space_bytes": self.address_space_bytes,
                "address_bits": self.address_bits,
            },
            "evidence": evidence_counts(self.parameters().values()),
            "parameters": parameter_records(self.parameters()),
        }
