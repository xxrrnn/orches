"""Paper-facing baseline and ablation definitions for ORCHES evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaselineKind(str, Enum):
    """Every system required by the paper's comparisons and ablations."""

    GPU = "gpu"
    ATTACC = "attacc"
    DUPLEX = "duplex"
    ORCHES_A = "orches_a"
    ORCHES_B = "orches_b"
    ORCHES_C = "orches_c"
    ORCHES_T1_ONLY = "orches_t1_only"
    ORCHES_T2_ONLY = "orches_t2_only"
    ORCHES = "orches"


class PlacementStrategy(str, Enum):
    """Top-level execution policy; native baselines retain native scheduling."""

    GPU_ONLY = "gpu_only"
    ATTENTION_ON_PIM = "attention_on_pim"
    DUPLEX_NATIVE = "duplex_native"
    ALL_ON_PIM = "all_on_pim"
    ADAPTIVE_LINEAR = "adaptive_linear"
    DYNAMIC_COMPENSATION = "dynamic_compensation"
    FULL_ORCHES = "full_orches"


@dataclass(frozen=True)
class TechniqueSwitches:
    """Explicit ORCHES technique switches attached to each comparison point."""

    t1a: bool = False
    t1b: bool = False
    t2a: bool = False
    t2b: bool = False
    t3: bool = False


@dataclass(frozen=True)
class BaselineDefinition:
    """One immutable paper configuration and its execution provenance class."""

    kind: BaselineKind
    label: str
    placement: PlacementStrategy
    techniques: TechniqueSwitches
    native_backend: str | None
    paper_scope: str


BASELINES: dict[BaselineKind, BaselineDefinition] = {
    BaselineKind.GPU: BaselineDefinition(
        BaselineKind.GPU,
        "GPU",
        PlacementStrategy.GPU_ONLY,
        TechniqueSwitches(),
        None,
        "Sec. 5.1 standalone GPU baseline",
    ),
    BaselineKind.ATTACC: BaselineDefinition(
        BaselineKind.ATTACC,
        "AttAcc",
        PlacementStrategy.ATTENTION_ON_PIM,
        TechniqueSwitches(),
        "AttAcc simulator",
        "Fig. 11-12 prior GPU-PIM baseline [25]",
    ),
    BaselineKind.DUPLEX: BaselineDefinition(
        BaselineKind.DUPLEX,
        "Duplex",
        PlacementStrategy.DUPLEX_NATIVE,
        TechniqueSwitches(),
        "Duplex LLMSimulator",
        "Fig. 11 prior GPU-PIM baseline [40]",
    ),
    BaselineKind.ORCHES_A: BaselineDefinition(
        BaselineKind.ORCHES_A,
        "ORCHES-A",
        PlacementStrategy.ALL_ON_PIM,
        TechniqueSwitches(t3=True),
        None,
        "Sec. 5.3 all computation on PIM; T2 disabled",
    ),
    BaselineKind.ORCHES_B: BaselineDefinition(
        BaselineKind.ORCHES_B,
        "ORCHES-B",
        PlacementStrategy.ADAPTIVE_LINEAR,
        TechniqueSwitches(t1a=True, t3=True),
        None,
        "Sec. 5.3 adaptive linear assignment; T2 disabled",
    ),
    BaselineKind.ORCHES_C: BaselineDefinition(
        BaselineKind.ORCHES_C,
        "ORCHES-C",
        PlacementStrategy.DYNAMIC_COMPENSATION,
        TechniqueSwitches(t1a=True, t1b=True, t3=True),
        None,
        "Sec. 5.3 ORCHES-B plus dynamic compensation; T2 disabled",
    ),
    BaselineKind.ORCHES_T1_ONLY: BaselineDefinition(
        BaselineKind.ORCHES_T1_ONLY,
        "ORCHES T1 only",
        PlacementStrategy.DYNAMIC_COMPENSATION,
        TechniqueSwitches(t1a=True, t1b=True, t3=True),
        None,
        "Table 6 T1-only contribution",
    ),
    BaselineKind.ORCHES_T2_ONLY: BaselineDefinition(
        BaselineKind.ORCHES_T2_ONLY,
        "ORCHES T2 only",
        PlacementStrategy.ALL_ON_PIM,
        TechniqueSwitches(t2a=True, t2b=True, t3=True),
        None,
        "Table 6 T2-only contribution",
    ),
    BaselineKind.ORCHES: BaselineDefinition(
        BaselineKind.ORCHES,
        "ORCHES",
        PlacementStrategy.FULL_ORCHES,
        TechniqueSwitches(t1a=True, t1b=True, t2a=True, t2b=True, t3=True),
        None,
        "Sec. 5.2 full T1+T2+T3 system",
    ),
}


FIGURE_11_BASELINES = (
    BaselineKind.GPU,
    BaselineKind.ATTACC,
    BaselineKind.DUPLEX,
    BaselineKind.ORCHES,
)

FIGURE_12_BASELINES = (
    BaselineKind.GPU,
    BaselineKind.ATTACC,
    BaselineKind.ORCHES_A,
    BaselineKind.ORCHES_B,
    BaselineKind.ORCHES_C,
)

TABLE_6_BASELINES = (
    BaselineKind.GPU,
    BaselineKind.ORCHES_T1_ONLY,
    BaselineKind.ORCHES_T2_ONLY,
    BaselineKind.ORCHES,
)
