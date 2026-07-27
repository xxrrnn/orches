"""Strict YAML loading for provenance-aware hardware configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeAlias

import yaml

from .errors import ConfigurationError
from .hardware import GpuHardwareConfig, HardwareConfig, PimHardwareConfig
from .provenance import EvidenceStatus, SourcedValue


RawMapping: TypeAlias = Mapping[str, Any]


def _as_mapping(value: Any, path: str) -> RawMapping:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _check_keys(
    mapping: RawMapping,
    *,
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(mapping)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ConfigurationError(f"{path} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{path} has unknown fields: {', '.join(unknown)}")


def _matches_type(value: Any, expected_type: type[Any]) -> bool:
    if expected_type is float:
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if expected_type is int:
        return not isinstance(value, bool) and isinstance(value, int)
    return isinstance(value, expected_type)


def _sourced(
    parent: RawMapping,
    key: str,
    expected_type: type[Any],
    *,
    path: str,
) -> SourcedValue[Any]:
    field_path = f"{path}.{key}"
    raw = _as_mapping(parent[key], field_path)
    _check_keys(
        raw,
        path=field_path,
        required={"value", "source", "status"},
        optional={"note"},
    )

    value = raw["value"]
    if not _matches_type(value, expected_type):
        raise ConfigurationError(
            f"{field_path}.value must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    source = raw["source"]
    if not isinstance(source, str) or not source.strip():
        raise ConfigurationError(f"{field_path}.source must be a non-empty string")

    status_raw = raw["status"]
    if not isinstance(status_raw, str):
        raise ConfigurationError(f"{field_path}.status must be a string")
    try:
        status = EvidenceStatus(status_raw)
    except ValueError as error:
        allowed = ", ".join(status.value for status in EvidenceStatus)
        raise ConfigurationError(
            f"{field_path}.status must be one of: {allowed}"
        ) from error

    note = raw.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ConfigurationError(f"{field_path}.note must be a non-empty string")

    normalized_value = float(value) if expected_type is float else value
    return SourcedValue(
        value=normalized_value,
        source=source,
        status=status,
        note=note,
    )


def _float_tuple_sourced(
    parent: RawMapping,
    key: str,
    *,
    path: str,
) -> SourcedValue[tuple[float, ...]]:
    field_path = f"{path}.{key}"
    raw = _as_mapping(parent[key], field_path)
    _check_keys(
        raw,
        path=field_path,
        required={"value", "source", "status"},
        optional={"note"},
    )

    values = raw["value"]
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ConfigurationError(f"{field_path}.value must be a list of numbers")

    copied = dict(raw)
    copied["value"] = tuple(float(value) for value in values)
    container = {key: copied}
    sourced = _sourced(container, key, tuple, path=path)
    return SourcedValue(
        value=sourced.value,
        source=sourced.source,
        status=sourced.status,
        note=sourced.note,
    )


def parse_hardware_config(raw_config: RawMapping) -> HardwareConfig:
    """Parse and validate a hardware configuration mapping."""

    root = _as_mapping(raw_config, "hardware")
    _check_keys(
        root,
        path="hardware",
        required={"schema_version", "name", "kind"},
        optional={"gpu", "pim"},
    )

    schema_version = root["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ConfigurationError("hardware.schema_version must be an integer")
    name = root["name"]
    kind = root["kind"]
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("hardware.name must be a non-empty string")
    if not isinstance(kind, str):
        raise ConfigurationError("hardware.kind must be a string")

    if kind == "gpu":
        if "pim" in root:
            raise ConfigurationError("GPU configuration must not contain a pim section")
        gpu = _as_mapping(root.get("gpu"), "hardware.gpu")
        required = {
            "memory_capacity_gb",
            "memory_bandwidth_gb_per_s",
            "cuda_cores",
            "tensor_cores",
            "max_gpu_frequency_mhz",
            "soc_bandwidth_scales",
        }
        _check_keys(gpu, path="hardware.gpu", required=required)
        return GpuHardwareConfig(
            schema_version=schema_version,
            name=name,
            kind=kind,
            memory_capacity_gb=_sourced(
                gpu, "memory_capacity_gb", float, path="hardware.gpu"
            ),
            memory_bandwidth_gb_per_s=_sourced(
                gpu, "memory_bandwidth_gb_per_s", float, path="hardware.gpu"
            ),
            cuda_cores=_sourced(gpu, "cuda_cores", int, path="hardware.gpu"),
            tensor_cores=_sourced(gpu, "tensor_cores", int, path="hardware.gpu"),
            max_gpu_frequency_mhz=_sourced(
                gpu, "max_gpu_frequency_mhz", float, path="hardware.gpu"
            ),
            soc_bandwidth_scales=_float_tuple_sourced(
                gpu, "soc_bandwidth_scales", path="hardware.gpu"
            ),
        )

    if kind == "pim":
        if "gpu" in root:
            raise ConfigurationError("PIM configuration must not contain a gpu section")
        pim = _as_mapping(root.get("pim"), "hardware.pim")
        required = {
            "memory_standard",
            "ramulator_organization_preset",
            "reported_capacity_gb",
            "simulated_capacity_gib",
            "channel_density_gibits",
            "channels",
            "pseudochannels_per_channel",
            "ranks_per_pseudochannel",
            "bankgroups_per_rank",
            "banks_per_bankgroup",
            "rows_per_bank",
            "columns_per_row",
            "transaction_bytes",
            "expected_bank_count",
            "gemv_lanes_per_bank",
            "host_io_bandwidth_gb_per_s",
            "ramulator_timing_preset",
            "controller_clock_ratio",
            "refresh_policy",
        }
        _check_keys(pim, path="hardware.pim", required=required)
        return PimHardwareConfig(
            schema_version=schema_version,
            name=name,
            kind=kind,
            memory_standard=_sourced(
                pim, "memory_standard", str, path="hardware.pim"
            ),
            ramulator_organization_preset=_sourced(
                pim, "ramulator_organization_preset", str, path="hardware.pim"
            ),
            reported_capacity_gb=_sourced(
                pim, "reported_capacity_gb", float, path="hardware.pim"
            ),
            simulated_capacity_gib=_sourced(
                pim, "simulated_capacity_gib", float, path="hardware.pim"
            ),
            channel_density_gibits=_sourced(
                pim, "channel_density_gibits", float, path="hardware.pim"
            ),
            channels=_sourced(pim, "channels", int, path="hardware.pim"),
            pseudochannels_per_channel=_sourced(
                pim, "pseudochannels_per_channel", int, path="hardware.pim"
            ),
            ranks_per_pseudochannel=_sourced(
                pim, "ranks_per_pseudochannel", int, path="hardware.pim"
            ),
            bankgroups_per_rank=_sourced(
                pim, "bankgroups_per_rank", int, path="hardware.pim"
            ),
            banks_per_bankgroup=_sourced(
                pim, "banks_per_bankgroup", int, path="hardware.pim"
            ),
            rows_per_bank=_sourced(
                pim, "rows_per_bank", int, path="hardware.pim"
            ),
            columns_per_row=_sourced(
                pim, "columns_per_row", int, path="hardware.pim"
            ),
            transaction_bytes=_sourced(
                pim, "transaction_bytes", int, path="hardware.pim"
            ),
            expected_bank_count=_sourced(
                pim, "expected_bank_count", int, path="hardware.pim"
            ),
            gemv_lanes_per_bank=_sourced(
                pim, "gemv_lanes_per_bank", int, path="hardware.pim"
            ),
            host_io_bandwidth_gb_per_s=_sourced(
                pim, "host_io_bandwidth_gb_per_s", float, path="hardware.pim"
            ),
            ramulator_timing_preset=_sourced(
                pim, "ramulator_timing_preset", str, path="hardware.pim"
            ),
            controller_clock_ratio=_sourced(
                pim, "controller_clock_ratio", int, path="hardware.pim"
            ),
            refresh_policy=_sourced(
                pim, "refresh_policy", str, path="hardware.pim"
            ),
        )

    raise ConfigurationError(f"hardware.kind must be 'gpu' or 'pim', got {kind!r}")


def load_hardware_config(path: str | Path) -> HardwareConfig:
    """Load, parse, and physically validate one hardware YAML file."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"Cannot read hardware config {config_path}: {error}") from error

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error
    if raw is None:
        raise ConfigurationError(f"Hardware config {config_path} is empty")
    return parse_hardware_config(_as_mapping(raw, "hardware"))
