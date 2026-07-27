from __future__ import annotations

from pathlib import Path

import pytest

from orches import load_hardware_config
from orches.errors import ConfigurationError
from orches.hardware import PimHardwareConfig
from orches.sim import (
    AttAccCommand,
    PimAddress,
    PimAddressGeometry,
    generate_mac_microbenchmark,
    parse_attacc_trace,
    render_attacc_config,
    run_attacc_ramulator,
    write_attacc_config,
    write_attacc_trace,
)


PROJECT_ROOT = Path(__file__).parents[1]
PIM_CONFIG = PROJECT_ROOT / "configs/hardware/orches_pim_32gb.yaml"
RAMULATOR = PROJECT_ROOT / "third_party/attacc_simulator/ramulator2/ramulator2"


@pytest.fixture(scope="module")
def pim_hardware() -> PimHardwareConfig:
    config = load_hardware_config(PIM_CONFIG)
    assert isinstance(config, PimHardwareConfig)
    return config


def test_address_geometry_matches_hardware_contract(
    pim_hardware: PimHardwareConfig,
) -> None:
    geometry = PimAddressGeometry.from_hardware(pim_hardware)

    assert geometry.address_space_bytes == 32 * 2**30
    assert geometry.address_bits == 35


@pytest.mark.parametrize(
    "address",
    [
        PimAddress(0, 0, 0, 0, 0, 0, 0, 0),
        PimAddress(31, 1, 1, 3, 3, 16383, 31, 31),
        PimAddress(17, 1, 0, 2, 3, 8192, 11, 7),
    ],
)
def test_address_mapping_is_reversible(
    pim_hardware: PimHardwareConfig,
    address: PimAddress,
) -> None:
    geometry = PimAddressGeometry.from_hardware(pim_hardware)

    assert geometry.decode(geometry.encode(address)) == address


def test_address_mapping_rejects_out_of_range_channel(
    pim_hardware: PimHardwareConfig,
) -> None:
    geometry = PimAddressGeometry.from_hardware(pim_hardware)

    with pytest.raises(ConfigurationError, match="channel=32"):
        geometry.encode(PimAddress(32, 0, 0, 0, 0, 0, 0))


def test_attacc_trace_round_trip(tmp_path, pim_hardware: PimHardwareConfig) -> None:
    commands = generate_mac_microbenchmark(pim_hardware, mac_rounds=2)
    trace_path = tmp_path / "microbench.trace"

    write_attacc_trace(trace_path, commands)

    assert parse_attacc_trace(trace_path) == commands
    assert sum(command.command is AttAccCommand.MAC_ALL_BANK for command in commands) == 64


def test_rendered_config_uses_orches_organization(
    tmp_path,
    pim_hardware: PimHardwareConfig,
) -> None:
    config = render_attacc_config(
        pim_hardware,
        trace_path=tmp_path / "trace",
        log_path=tmp_path / "log/cmd.log",
    )
    dram = config["MemorySystem"]["DRAM"]

    assert dram["org"]["preset"] == "HBM3_8Gb_2R"
    assert dram["org"]["channel"] == 32
    assert config["Frontend"]["Translation"]["max_addr"] == 32 * 2**30


@pytest.mark.integration
def test_attacc_ramulator_microbenchmark(
    tmp_path,
    pim_hardware: PimHardwareConfig,
) -> None:
    if not RAMULATOR.is_file():
        pytest.skip("run scripts/bootstrap.sh before the native integration test")
    trace_path = tmp_path / "microbench.trace"
    config_path = tmp_path / "ramulator.yaml"
    log_path = tmp_path / "log/cmd.log"
    log_path.parent.mkdir(parents=True)
    commands = generate_mac_microbenchmark(pim_hardware)
    write_attacc_trace(trace_path, commands)
    write_attacc_config(
        config_path,
        render_attacc_config(
            pim_hardware,
            trace_path=trace_path,
            log_path=log_path,
        ),
    )

    result = run_attacc_ramulator(RAMULATOR, config_path)

    assert result.memory_system_cycles > 0
    assert result.counters["total_num_pim_mac_all_bank_requests"] == 32
    assert result.counters["total_num_pim_softmax_requests"] == 32
