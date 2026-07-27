"""Explicit Transformer operator costs and dependency DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigurationError
from .transformer import TransformerConfig


class OperatorKind(str, Enum):
    """Primitive operation categories needed by GPU/PIM timing backends."""

    LINEAR = "linear"
    NORMALIZATION = "normalization"
    ROPE = "rope"
    ATTENTION_SCORE = "attention_score"
    SOFTMAX = "softmax"
    ATTENTION_CONTEXT = "attention_context"


@dataclass(frozen=True)
class OperatorCost:
    """Shape-derived operation and traffic counts for one DAG node."""

    operator_id: str
    kind: OperatorKind
    macs: int
    elementwise_flops: int
    read_bytes: int
    write_bytes: int
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ConfigurationError("operator_id must not be empty")
        for name, value in (
            ("macs", self.macs),
            ("elementwise_flops", self.elementwise_flops),
            ("read_bytes", self.read_bytes),
            ("write_bytes", self.write_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"operator.{name} must be non-negative")
        if self.macs == 0 and self.elementwise_flops == 0:
            raise ConfigurationError("an operator must perform at least one operation")

    @property
    def flops(self) -> int:
        """Conventional FLOP count; one multiply-accumulate is two FLOPs."""

        return 2 * self.macs + self.elementwise_flops

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes


@dataclass(frozen=True)
class OperatorDag:
    """Topologically ordered operator nodes for one decoder-layer invocation."""

    operators: tuple[OperatorCost, ...]

    def __post_init__(self) -> None:
        if not self.operators:
            raise ConfigurationError("operator DAG must not be empty")
        seen: set[str] = set()
        for operator in self.operators:
            if operator.operator_id in seen:
                raise ConfigurationError(
                    f"duplicate operator ID {operator.operator_id!r}"
                )
            missing = set(operator.dependencies) - seen
            if missing:
                raise ConfigurationError(
                    f"operator {operator.operator_id!r} has non-topological dependencies: "
                    f"{', '.join(sorted(missing))}"
                )
            seen.add(operator.operator_id)

    @property
    def total_macs(self) -> int:
        return sum(operator.macs for operator in self.operators)

    @property
    def total_flops(self) -> int:
        return sum(operator.flops for operator in self.operators)

    @property
    def total_bytes(self) -> int:
        return sum(operator.total_bytes for operator in self.operators)


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def linear_cost(
    operator_id: str,
    *,
    rows: int,
    input_features: int,
    output_features: int,
    bytes_per_element: int,
    dependencies: tuple[str, ...] = (),
) -> OperatorCost:
    """Count a dense `rows x input` by `input x output` projection."""

    for name, value in (
        ("rows", rows),
        ("input_features", input_features),
        ("output_features", output_features),
        ("bytes_per_element", bytes_per_element),
    ):
        _positive(name, value)
    macs = rows * input_features * output_features
    read_elements = rows * input_features + input_features * output_features
    write_elements = rows * output_features
    return OperatorCost(
        operator_id=operator_id,
        kind=OperatorKind.LINEAR,
        macs=macs,
        elementwise_flops=0,
        read_bytes=read_elements * bytes_per_element,
        write_bytes=write_elements * bytes_per_element,
        dependencies=dependencies,
    )


def _elementwise_cost(
    operator_id: str,
    kind: OperatorKind,
    *,
    elements: int,
    flops_per_element: int,
    bytes_per_element: int,
    dependencies: tuple[str, ...],
) -> OperatorCost:
    return OperatorCost(
        operator_id=operator_id,
        kind=kind,
        macs=0,
        elementwise_flops=elements * flops_per_element,
        read_bytes=elements * bytes_per_element,
        write_bytes=elements * bytes_per_element,
        dependencies=dependencies,
    )


def build_decoder_layer_dag(
    model: TransformerConfig,
    *,
    branch_width: int,
    shared_context_tokens: int,
    unique_context_tokens: int,
) -> OperatorDag:
    """Expand one decode token per branch into an explicit decoder-layer DAG."""

    for name, value in (
        ("branch_width", branch_width),
        ("shared_context_tokens", shared_context_tokens),
    ):
        _positive(name, value)
    if unique_context_tokens < 0:
        raise ConfigurationError("unique_context_tokens must be non-negative")

    width = branch_width
    hidden = model.hidden_size
    kv_hidden = model.kv_hidden_size
    heads = model.num_attention_heads
    head_dim = model.head_dim
    element_bytes = model.bytes_per_element

    operators: list[OperatorCost] = []
    operators.append(
        _elementwise_cost(
            "input_norm",
            OperatorKind.NORMALIZATION,
            elements=width * hidden,
            flops_per_element=5,
            bytes_per_element=element_bytes,
            dependencies=(),
        )
    )
    for projection, output_features in (
        ("q_proj", hidden),
        ("k_proj", kv_hidden),
        ("v_proj", kv_hidden),
    ):
        operators.append(
            linear_cost(
                projection,
                rows=width,
                input_features=hidden,
                output_features=output_features,
                bytes_per_element=element_bytes,
                dependencies=("input_norm",),
            )
        )
    operators.append(
        _elementwise_cost(
            "rope",
            OperatorKind.ROPE,
            elements=width * (hidden + kv_hidden),
            flops_per_element=4,
            bytes_per_element=element_bytes,
            dependencies=("q_proj", "k_proj"),
        )
    )

    def score_fragment(
        prefix: str,
        context_tokens: int,
        *,
        kv_copies: int,
    ) -> str | None:
        if context_tokens == 0:
            return None
        score_id = f"{prefix}_score"
        score_elements = width * heads * context_tokens
        kv_elements = kv_copies * context_tokens * kv_hidden
        operators.append(
            OperatorCost(
                operator_id=score_id,
                kind=OperatorKind.ATTENTION_SCORE,
                macs=width * heads * head_dim * context_tokens,
                elementwise_flops=0,
                read_bytes=(width * hidden + kv_elements) * element_bytes,
                write_bytes=score_elements * element_bytes,
                dependencies=("rope",),
            )
        )
        return score_id

    shared_score = score_fragment("shared", shared_context_tokens, kv_copies=1)
    unique_score = score_fragment(
        "unique", unique_context_tokens, kv_copies=width
    )
    score_dependencies = tuple(
        score_id for score_id in (shared_score, unique_score) if score_id is not None
    )
    total_context_tokens = shared_context_tokens + unique_context_tokens
    softmax_elements = width * heads * total_context_tokens
    operators.append(
        _elementwise_cost(
            "softmax",
            OperatorKind.SOFTMAX,
            elements=softmax_elements,
            flops_per_element=7,
            bytes_per_element=element_bytes,
            dependencies=score_dependencies,
        )
    )

    def context_fragment(
        prefix: str,
        context_tokens: int,
        *,
        kv_copies: int,
    ) -> str | None:
        if context_tokens == 0:
            return None
        context_id = f"{prefix}_context"
        score_elements = width * heads * context_tokens
        kv_elements = kv_copies * context_tokens * kv_hidden
        operators.append(
            OperatorCost(
                operator_id=context_id,
                kind=OperatorKind.ATTENTION_CONTEXT,
                macs=width * heads * context_tokens * head_dim,
                elementwise_flops=0,
                read_bytes=(score_elements + kv_elements) * element_bytes,
                write_bytes=width * hidden * element_bytes,
                dependencies=("softmax", "v_proj"),
            )
        )
        return context_id

    shared_context = context_fragment("shared", shared_context_tokens, kv_copies=1)
    unique_context = context_fragment(
        "unique", unique_context_tokens, kv_copies=width
    )
    context_dependencies = tuple(
        context_id
        for context_id in (shared_context, unique_context)
        if context_id is not None
    )
    operators.append(
        linear_cost(
            "o_proj",
            rows=width,
            input_features=hidden,
            output_features=hidden,
            bytes_per_element=element_bytes,
            dependencies=context_dependencies,
        )
    )
    operators.append(
        _elementwise_cost(
            "post_attention_norm",
            OperatorKind.NORMALIZATION,
            elements=width * hidden,
            flops_per_element=5,
            bytes_per_element=element_bytes,
            dependencies=("o_proj",),
        )
    )
    for projection in ("gate_proj", "up_proj"):
        operators.append(
            linear_cost(
                projection,
                rows=width,
                input_features=hidden,
                output_features=model.intermediate_size,
                bytes_per_element=element_bytes,
                dependencies=("post_attention_norm",),
            )
        )
    operators.append(
        _elementwise_cost(
            "swiglu",
            OperatorKind.NORMALIZATION,
            elements=width * model.intermediate_size,
            flops_per_element=4,
            bytes_per_element=element_bytes,
            dependencies=("gate_proj", "up_proj"),
        )
    )
    operators.append(
        linear_cost(
            "down_proj",
            rows=width,
            input_features=model.intermediate_size,
            output_features=hidden,
            bytes_per_element=element_bytes,
            dependencies=("swiglu",),
        )
    )
    return OperatorDag(tuple(operators))
