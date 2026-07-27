"""Unit-explicit activity-based energy accounting shared by all systems."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

from ..errors import ConfigurationError
from ..provenance import EvidenceStatus


@dataclass(frozen=True)
class UnitEnergy:
    """One pJ-per-activity coefficient with source and evidence status."""

    value_pj: float
    per: str
    source: str
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if not isfinite(self.value_pj) or self.value_pj < 0:
            raise ConfigurationError("unit energy must be finite and non-negative")
        if not self.per.strip() or not self.source.strip():
            raise ConfigurationError(
                "unit-energy dimension and source must not be empty"
            )


@dataclass(frozen=True)
class EnergyRates:
    """Coefficients required by the M4 event/activity accounting model."""

    gpu_compute_pj_per_mac: UnitEnergy
    gpu_memory_pj_per_byte: UnitEnergy
    pim_compute_pj_per_mac: UnitEnergy
    pim_memory_pj_per_byte: UnitEnergy
    host_link_pj_per_byte: UnitEnergy
    controller_sram_pj_per_byte: UnitEnergy
    controller_buffer_pj_per_byte: UnitEnergy

    def __post_init__(self) -> None:
        expected_dimensions = {
            "gpu_compute_pj_per_mac": "mac",
            "gpu_memory_pj_per_byte": "byte",
            "pim_compute_pj_per_mac": "mac",
            "pim_memory_pj_per_byte": "byte",
            "host_link_pj_per_byte": "byte",
            "controller_sram_pj_per_byte": "byte",
            "controller_buffer_pj_per_byte": "byte",
        }
        for name, expected in expected_dimensions.items():
            if getattr(self, name).per != expected:
                raise ConfigurationError(
                    f"{name} must use the {expected!r} activity dimension"
                )


@dataclass(frozen=True)
class EnergyActivity:
    """Non-negative operation and byte counters for one request or experiment."""

    gpu_macs: float = 0.0
    gpu_memory_bytes: float = 0.0
    pim_macs: float = 0.0
    pim_memory_bytes: float = 0.0
    host_link_bytes: float = 0.0
    controller_sram_bytes: float = 0.0
    controller_buffer_bytes: float = 0.0
    compaction_read_bytes: float = 0.0
    compaction_write_bytes: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isfinite(value) or value < 0:
                raise ConfigurationError(
                    f"energy activity {item.name} must be finite and non-negative"
                )

    def __add__(self, other: EnergyActivity) -> EnergyActivity:
        if not isinstance(other, EnergyActivity):
            return NotImplemented
        return EnergyActivity(
            **{
                item.name: getattr(self, item.name) + getattr(other, item.name)
                for item in fields(self)
            }
        )


@dataclass(frozen=True)
class EnergyBreakdown:
    """Joules by physical domain; total is derived rather than stored."""

    gpu_compute_j: float
    gpu_memory_j: float
    pim_compute_j: float
    pim_memory_j: float
    host_link_j: float
    controller_sram_j: float
    controller_buffer_j: float
    compaction_j: float

    @property
    def total_j(self) -> float:
        return sum(getattr(self, item.name) for item in fields(self))


def calculate_energy(
    activity: EnergyActivity,
    rates: EnergyRates,
) -> EnergyBreakdown:
    """Multiply activity by pJ coefficients and return joules."""

    picojoule_to_joule = 1e-12
    compaction_bytes = (
        activity.compaction_read_bytes + activity.compaction_write_bytes
    )
    return EnergyBreakdown(
        gpu_compute_j=(
            activity.gpu_macs * rates.gpu_compute_pj_per_mac.value_pj
        )
        * picojoule_to_joule,
        gpu_memory_j=(
            activity.gpu_memory_bytes * rates.gpu_memory_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
        pim_compute_j=(
            activity.pim_macs * rates.pim_compute_pj_per_mac.value_pj
        )
        * picojoule_to_joule,
        pim_memory_j=(
            activity.pim_memory_bytes * rates.pim_memory_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
        host_link_j=(
            activity.host_link_bytes * rates.host_link_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
        controller_sram_j=(
            activity.controller_sram_bytes
            * rates.controller_sram_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
        controller_buffer_j=(
            activity.controller_buffer_bytes
            * rates.controller_buffer_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
        compaction_j=(
            compaction_bytes * rates.pim_memory_pj_per_byte.value_pj
        )
        * picojoule_to_joule,
    )


def attacc_bank_level_energy_rates() -> EnergyRates:
    """Reproduce AttAcc's bank-level constants without renaming energy as power."""

    source = (
        "AttAcc c60005143a6b492d7ef83231723386478b59a506 "
        "src/config.py ENERGY_TABLE"
    )
    inherited_mac = UnitEnergy(0.32, "mac", source, EvidenceStatus.INHERITED)
    inherited_sram = UnitEnergy(
        0.0034,
        "byte",
        source,
        EvidenceStatus.INHERITED,
    )
    return EnergyRates(
        gpu_compute_pj_per_mac=inherited_mac,
        gpu_memory_pj_per_byte=UnitEnergy(
            (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8,
            "byte",
            source,
            EvidenceStatus.INHERITED,
        ),
        pim_compute_pj_per_mac=inherited_mac,
        pim_memory_pj_per_byte=UnitEnergy(
            (0.11 + 0.44) * 8,
            "byte",
            source,
            EvidenceStatus.INHERITED,
        ),
        host_link_pj_per_byte=UnitEnergy(
            10.4,
            "byte",
            source,
            EvidenceStatus.INHERITED,
        ),
        controller_sram_pj_per_byte=inherited_sram,
        controller_buffer_pj_per_byte=inherited_sram,
    )
