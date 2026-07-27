"""Reversible ORCHES PIM byte-address geometry."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError
from ..hardware import PimHardwareConfig


@dataclass(frozen=True)
class PimAddress:
    """One byte address decomposed using AttAcc/Ramulator2 hierarchy names."""

    channel: int
    pseudochannel: int
    rank: int
    bankgroup: int
    bank: int
    row: int
    column: int
    byte_offset: int = 0


@dataclass(frozen=True)
class PimAddressGeometry:
    """Mixed-radix mapper whose channel is the most significant dimension."""

    channels: int
    pseudochannels: int
    ranks: int
    bankgroups: int
    banks: int
    rows: int
    columns: int
    transaction_bytes: int

    def __post_init__(self) -> None:
        for name, value in self.dimensions:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"PIM address dimension {name} must be positive")

    @classmethod
    def from_hardware(cls, hardware: PimHardwareConfig) -> PimAddressGeometry:
        return cls(
            channels=hardware.channels.value,
            pseudochannels=hardware.pseudochannels_per_channel.value,
            ranks=hardware.ranks_per_pseudochannel.value,
            bankgroups=hardware.bankgroups_per_rank.value,
            banks=hardware.banks_per_bankgroup.value,
            rows=hardware.rows_per_bank.value,
            columns=hardware.columns_per_row.value,
            transaction_bytes=hardware.transaction_bytes.value,
        )

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        return (
            ("channel", self.channels),
            ("pseudochannel", self.pseudochannels),
            ("rank", self.ranks),
            ("bankgroup", self.bankgroups),
            ("bank", self.banks),
            ("row", self.rows),
            ("column", self.columns),
            ("byte_offset", self.transaction_bytes),
        )

    @property
    def address_space_bytes(self) -> int:
        size = 1
        for _, count in self.dimensions:
            size *= count
        return size

    @property
    def address_bits(self) -> int:
        if self.address_space_bytes & (self.address_space_bytes - 1):
            raise ConfigurationError("address_bits requires a power-of-two address space")
        return self.address_space_bytes.bit_length() - 1

    def encode(self, address: PimAddress) -> int:
        """Encode a checked hierarchy tuple into a byte address."""

        values = (
            address.channel,
            address.pseudochannel,
            address.rank,
            address.bankgroup,
            address.bank,
            address.row,
            address.column,
            address.byte_offset,
        )
        encoded = 0
        for (name, count), value in zip(self.dimensions, values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"PIM address {name} must be an integer")
            if value < 0 or value >= count:
                raise ConfigurationError(
                    f"PIM address {name}={value} is outside [0, {count})"
                )
            encoded = encoded * count + value
        return encoded

    def decode(self, encoded: int) -> PimAddress:
        """Decode one in-range byte address into hierarchy coordinates."""

        if isinstance(encoded, bool) or not isinstance(encoded, int):
            raise ConfigurationError("encoded PIM address must be an integer")
        if encoded < 0 or encoded >= self.address_space_bytes:
            raise ConfigurationError(
                f"encoded PIM address must be in [0, {self.address_space_bytes})"
            )

        values: dict[str, int] = {}
        remainder = encoded
        for name, count in reversed(self.dimensions):
            values[name] = remainder % count
            remainder //= count
        return PimAddress(
            channel=values["channel"],
            pseudochannel=values["pseudochannel"],
            rank=values["rank"],
            bankgroup=values["bankgroup"],
            bank=values["bank"],
            row=values["row"],
            column=values["column"],
            byte_offset=values["byte_offset"],
        )

    def channel_base(self, channel: int) -> int:
        return self.encode(
            PimAddress(
                channel=channel,
                pseudochannel=0,
                rank=0,
                bankgroup=0,
                bank=0,
                row=0,
                column=0,
            )
        )
