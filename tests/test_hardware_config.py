from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from orches.config import parse_hardware_config
from orches.errors import ConfigurationError
from orches.hardware import GpuHardwareConfig, PimHardwareConfig
from orches import load_hardware_config


PROJECT_ROOT = Path(__file__).parents[1]
GPU_CONFIG = PROJECT_ROOT / "configs/hardware/agx_orin_32gb.yaml"
PIM_CONFIG = PROJECT_ROOT / "configs/hardware/orches_pim_32gb.yaml"


def _raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_gpu_paper_configuration() -> None:
    config = load_hardware_config(GPU_CONFIG)

    assert isinstance(config, GpuHardwareConfig)
    assert config.memory_capacity_gb.value == 32.0
    assert config.memory_bandwidth_gb_per_s.value == 204.8
    assert config.cuda_cores.value == 1792
    assert config.tensor_cores.value == 56
    assert config.max_gpu_frequency_mhz.value == 930.0
    assert config.soc_bandwidth_scales.value == (1.0, 0.75, 0.5)
    assert config.summary()["derived"]["bandwidth_points_gb_per_s"] == [
        204.8,
        153.6,
        102.4,
    ]


def test_pim_paper_configuration() -> None:
    config = load_hardware_config(PIM_CONFIG)

    assert isinstance(config, PimHardwareConfig)
    assert config.ramulator_organization_preset.value == "HBM3_8Gb_2R"
    assert config.channels.value == 32
    assert config.derived_capacity_gib == 32.0
    assert config.derived_bank_count == 2048
    assert config.total_gemv_lanes == 32768
    assert config.address_space_bytes == 32 * 2**30
    assert config.address_bits == 35


def test_pim_rejects_inconsistent_bank_count() -> None:
    raw = _raw_config(PIM_CONFIG)
    raw["pim"]["expected_bank_count"]["value"] = 1024

    with pytest.raises(ConfigurationError, match="derives 2048 banks"):
        parse_hardware_config(raw)


def test_pim_rejects_inconsistent_capacity() -> None:
    raw = _raw_config(PIM_CONFIG)
    raw["pim"]["simulated_capacity_gib"]["value"] = 30.0

    with pytest.raises(ConfigurationError, match="derives 32 GiB"):
        parse_hardware_config(raw)


def test_unknown_field_is_not_silently_ignored() -> None:
    raw = _raw_config(GPU_CONFIG)
    raw["gpu"]["undocumented_efficiency"] = {
        "value": 0.9,
        "source": "test",
        "status": "ASSUMED",
    }

    with pytest.raises(ConfigurationError, match="unknown fields"):
        parse_hardware_config(raw)


@pytest.mark.parametrize("metadata_key", ["source", "status"])
def test_every_parameter_requires_provenance(metadata_key: str) -> None:
    raw = _raw_config(GPU_CONFIG)
    del raw["gpu"]["cuda_cores"][metadata_key]

    with pytest.raises(ConfigurationError, match="missing required fields"):
        parse_hardware_config(raw)


def test_evidence_status_must_be_known() -> None:
    raw = _raw_config(GPU_CONFIG)
    raw["gpu"]["cuda_cores"]["status"] = "GUESSED"

    with pytest.raises(ConfigurationError, match="must be one of"):
        parse_hardware_config(raw)


def test_invalid_bandwidth_scale_order_is_rejected() -> None:
    raw = _raw_config(GPU_CONFIG)
    raw["gpu"]["soc_bandwidth_scales"]["value"] = [1.0, 0.5, 0.75]

    with pytest.raises(ConfigurationError, match="descending order"):
        parse_hardware_config(raw)


def test_source_config_is_not_mutated() -> None:
    raw = _raw_config(GPU_CONFIG)
    original = deepcopy(raw)

    parse_hardware_config(raw)

    assert raw == original
