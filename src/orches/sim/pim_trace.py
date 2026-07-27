"""AttAcc-compatible PIM trace generation, configuration, and execution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from ..errors import ConfigurationError, SimulationError
from ..hardware import PimHardwareConfig
from .pim import PimAddress, PimAddressGeometry


class AttAccCommand(str, Enum):
    """Commands accepted by AttAcc's PIMLoadStoreTrace frontend."""

    LOAD = "LD"
    STORE = "ST"
    MAC_ALL_BANK = "PIM_MAC_AB"
    MAC_SAME_BANK = "PIM_MAC_SB"
    MAC_PER_BANK = "PIM_MAC_PB"
    WRITE_GEMV_BUFFER = "PIM_WR_GB"
    MOVE_SOFTMAX_BUFFER = "PIM_MV_SB"
    MOVE_GEMV_BUFFER = "PIM_MV_GB"
    SOFTMAX = "PIM_SFM"
    SET_MODEL = "PIM_SET_MODEL"
    SET_HEAD = "PIM_SET_HEAD"
    BARRIER = "PIM_BARRIER"


@dataclass(frozen=True)
class AttAccTraceCommand:
    """One two-column command line consumed by the AttAcc frontend."""

    command: AttAccCommand
    address: int

    def __post_init__(self) -> None:
        if isinstance(self.address, bool) or not isinstance(self.address, int):
            raise ConfigurationError("AttAcc command address must be an integer")
        if self.address < 0:
            raise ConfigurationError("AttAcc command address must be non-negative")

    def to_line(self) -> str:
        return f"{self.command.value} 0x{self.address:08x}"


@dataclass(frozen=True)
class RamulatorResult:
    """Parsed counters from one completed Ramulator2 invocation."""

    memory_system_cycles: int
    counters: dict[str, int]
    stdout: str


def generate_mac_microbenchmark(
    hardware: PimHardwareConfig,
    *,
    mac_rounds: int = 1,
) -> list[AttAccTraceCommand]:
    """Exercise every channel's bank-level GEMV, accumulation, and softmax path."""

    if isinstance(mac_rounds, bool) or not isinstance(mac_rounds, int) or mac_rounds <= 0:
        raise ConfigurationError("mac_rounds must be a positive integer")
    geometry = PimAddressGeometry.from_hardware(hardware)
    commands: list[AttAccTraceCommand] = []

    def append_all_channels(command: AttAccCommand, column: int = 0) -> None:
        for channel in range(geometry.channels):
            address = geometry.encode(
                PimAddress(
                    channel=channel,
                    pseudochannel=0,
                    rank=0,
                    bankgroup=0,
                    bank=0,
                    row=0,
                    column=column,
                )
            )
            commands.append(AttAccTraceCommand(command, address))

    append_all_channels(AttAccCommand.WRITE_GEMV_BUFFER)
    append_all_channels(AttAccCommand.BARRIER)
    for round_index in range(mac_rounds):
        append_all_channels(
            AttAccCommand.MAC_ALL_BANK,
            column=round_index % geometry.columns,
        )
        append_all_channels(AttAccCommand.BARRIER)
    append_all_channels(AttAccCommand.MOVE_SOFTMAX_BUFFER)
    append_all_channels(AttAccCommand.SOFTMAX)
    append_all_channels(AttAccCommand.BARRIER)
    return commands


def write_attacc_trace(
    path: str | Path,
    commands: list[AttAccTraceCommand],
) -> None:
    """Write a deterministic AttAcc command trace."""

    if not commands:
        raise ConfigurationError("AttAcc trace must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            "".join(f"{command.to_line()}\n" for command in commands),
            encoding="ascii",
        )
    except OSError as error:
        raise ConfigurationError(f"cannot write AttAcc trace {destination}: {error}") from error


def parse_attacc_trace(path: str | Path) -> list[AttAccTraceCommand]:
    """Parse the exact two-column command format used by AttAcc."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise ConfigurationError(f"cannot read AttAcc trace {source}: {error}") from error
    if not lines:
        raise ConfigurationError(f"AttAcc trace {source} is empty")

    commands: list[AttAccTraceCommand] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ConfigurationError(
                f"AttAcc trace {source} line {line_number} must have two fields"
            )
        try:
            command = AttAccCommand(fields[0])
        except ValueError as error:
            raise ConfigurationError(
                f"AttAcc trace {source} line {line_number} has unknown command"
            ) from error
        try:
            address = int(fields[1], 0)
        except ValueError as error:
            raise ConfigurationError(
                f"AttAcc trace {source} line {line_number} has invalid address"
            ) from error
        commands.append(AttAccTraceCommand(command, address))
    return commands


def render_attacc_config(
    hardware: PimHardwareConfig,
    *,
    trace_path: str | Path,
    log_path: str | Path,
) -> dict[str, object]:
    """Render Ramulator2 YAML fields from the validated ORCHES hardware contract."""

    return {
        "Frontend": {
            "impl": "PIMLoadStoreTrace",
            "path": str(Path(trace_path).resolve()),
            "clock_ratio": hardware.controller_clock_ratio.value,
            "Translation": {
                "impl": "NoTranslation",
                "max_addr": hardware.address_space_bytes,
            },
        },
        "MemorySystem": {
            "impl": "PIMDRAM",
            "clock_ratio": hardware.controller_clock_ratio.value,
            "DRAM": {
                "impl": hardware.memory_standard.value,
                "org": {
                    "preset": hardware.ramulator_organization_preset.value,
                    "channel": hardware.channels.value,
                },
                "timing": {"preset": hardware.ramulator_timing_preset.value},
            },
            "Controller": {
                "impl": "HBM3-PIM",
                "Scheduler": {"impl": "PIM"},
                "RefreshManager": {"impl": hardware.refresh_policy.value},
                "plugins": [
                    {
                        "ControllerPlugin": {
                            "impl": "HBM3TraceRecorder",
                            "path": str(Path(log_path).resolve()),
                        }
                    }
                ],
            },
            "AddrMapper": {"impl": "HBM3-PIM"},
        },
    }


def write_attacc_config(path: str | Path, config: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
    except OSError as error:
        raise ConfigurationError(
            f"cannot write AttAcc configuration {destination}: {error}"
        ) from error


def run_attacc_ramulator(
    executable: str | Path,
    config_path: str | Path,
) -> RamulatorResult:
    """Run the frozen AttAcc backend and parse its integer request counters."""

    executable_path = Path(executable)
    config = Path(config_path)
    if not executable_path.is_file():
        raise SimulationError(f"Ramulator2 executable does not exist: {executable_path}")
    try:
        completed = subprocess.run(
            [str(executable_path.resolve()), "-f", str(config.resolve())],
            cwd=config.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "stdout", None) or ""
        raise SimulationError(f"AttAcc Ramulator2 execution failed: {output}") from error

    counters = {
        name: int(value)
        for name, value in re.findall(
            r"^\s{2}([a-z][a-z0-9_]+):\s+(\d+)\s*$",
            completed.stdout,
            flags=re.MULTILINE,
        )
    }
    if "memory_system_cycles" not in counters:
        raise SimulationError("Ramulator2 output did not contain memory_system_cycles")
    cycles = counters.pop("memory_system_cycles")
    return RamulatorResult(
        memory_system_cycles=cycles,
        counters=counters,
        stdout=completed.stdout,
    )
