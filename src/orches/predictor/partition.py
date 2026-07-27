"""Layer-exact PRM partition used by the paper's Fig. 13 case study."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError


@dataclass(frozen=True)
class PrmLayerPartition:
    """Split one PRM into a small prefix and large remaining suffix."""

    total_layers: int
    small_prefix_layers: int

    def __post_init__(self) -> None:
        for name, value in (
            ("total_layers", self.total_layers),
            ("small_prefix_layers", self.small_prefix_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.small_prefix_layers >= self.total_layers:
            raise ConfigurationError(
                "small_prefix_layers must be smaller than total_layers"
            )

    @property
    def large_suffix_layers(self) -> int:
        return self.total_layers - self.small_prefix_layers

    @property
    def small_fraction(self) -> float:
        return self.small_prefix_layers / self.total_layers

    @property
    def adds_model_layers(self) -> bool:
        """The prefix and suffix partition the original PRM without duplication."""

        return False
