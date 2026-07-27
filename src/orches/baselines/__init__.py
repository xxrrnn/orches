"""Fair baseline definitions, native adapters, and comparison contracts."""

from .adapters import parse_attacc_csv, parse_duplex_csv
from .orches import adapt_orches_replay
from .definitions import (
    BASELINES,
    FIGURE_11_BASELINES,
    FIGURE_12_BASELINES,
    TABLE_6_BASELINES,
    BaselineDefinition,
    BaselineKind,
    PlacementStrategy,
    TechniqueSwitches,
)
from .results import (
    BaselineMetrics,
    BaselineRunResult,
    ComparisonRow,
    FairnessContract,
    RunStatus,
    compare_baselines,
)

__all__ = [
    "BASELINES",
    "FIGURE_11_BASELINES",
    "FIGURE_12_BASELINES",
    "TABLE_6_BASELINES",
    "BaselineDefinition",
    "BaselineKind",
    "BaselineMetrics",
    "BaselineRunResult",
    "ComparisonRow",
    "FairnessContract",
    "PlacementStrategy",
    "RunStatus",
    "TechniqueSwitches",
    "adapt_orches_replay",
    "compare_baselines",
    "parse_attacc_csv",
    "parse_duplex_csv",
]
