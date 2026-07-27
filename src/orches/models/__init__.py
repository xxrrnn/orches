"""Transformer architecture, operator DAG, and analytical timing models."""

from .operators import (
    OperatorCost,
    OperatorDag,
    OperatorKind,
    build_decoder_layer_dag,
    linear_cost,
)
from .transformer import ModelRole, TransformerConfig, load_transformer_config
from .timing import (
    CoprocessedLinearTiming,
    GpuRooflineRates,
    PaperGpuRates,
    PaperPimRates,
    PaperTimingBreakdown,
    RooflineEstimate,
    gpu_roofline_time,
    paper_coprocessed_linear_time,
    paper_gpu_linear_time,
    paper_pim_linear_time,
)

__all__ = [
    "ModelRole",
    "OperatorCost",
    "OperatorDag",
    "OperatorKind",
    "CoprocessedLinearTiming",
    "GpuRooflineRates",
    "PaperGpuRates",
    "PaperPimRates",
    "PaperTimingBreakdown",
    "RooflineEstimate",
    "TransformerConfig",
    "build_decoder_layer_dag",
    "gpu_roofline_time",
    "linear_cost",
    "load_transformer_config",
    "paper_coprocessed_linear_time",
    "paper_gpu_linear_time",
    "paper_pim_linear_time",
]
