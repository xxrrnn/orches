from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orches.errors import ConfigurationError
from orches.models import (
    ModelRole,
    OperatorKind,
    build_decoder_layer_dag,
    linear_cost,
    load_transformer_config,
)
from orches.models.transformer import parse_transformer_config


PROJECT_ROOT = Path(__file__).parents[1]
MODEL_ROOT = PROJECT_ROOT / "configs/models"


@pytest.mark.parametrize(
    ("relative_path", "hidden", "layers", "heads", "kv_heads"),
    [
        ("policy/llama3.2-1b.yaml", 2048, 16, 32, 8),
        ("policy/qwen2.5-1.5b.yaml", 1536, 28, 12, 2),
        ("policy/qwen2.5-3b.yaml", 2048, 36, 16, 2),
        ("prm/qwen2.5-1.5b-prm-tuned.yaml", 1536, 28, 12, 2),
        ("prm/qwen2.5-7b-prm-tuned.yaml", 3584, 28, 28, 4),
        ("prm/llama3.1-8b-prm-tuned.yaml", 4096, 32, 32, 8),
    ],
)
def test_paper_model_architecture_profiles(
    relative_path: str,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
) -> None:
    model = load_transformer_config(MODEL_ROOT / relative_path)

    assert model.hidden_size == hidden
    assert model.num_hidden_layers == layers
    assert model.num_attention_heads == heads
    assert model.num_key_value_heads == kv_heads
    assert model.kv_hidden_size == kv_heads * model.head_dim
    assert model.kv_bytes_per_token == 2 * layers * kv_heads * model.head_dim * 2
    assert model.estimated_weight_bytes > 0


def test_policy_and_prm_roles_are_explicit() -> None:
    policy = load_transformer_config(MODEL_ROOT / "policy/qwen2.5-3b.yaml")
    prm = load_transformer_config(
        MODEL_ROOT / "prm/qwen2.5-7b-prm-tuned.yaml"
    )

    assert policy.role is ModelRole.POLICY
    assert prm.role is ModelRole.PRM
    assert prm.source_status.value == "ASSUMED"


def test_invalid_head_shape_is_rejected() -> None:
    path = MODEL_ROOT / "policy/qwen2.5-1.5b.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["architecture"]["head_dim"] = 64

    with pytest.raises(ConfigurationError, match="hidden_size must equal"):
        parse_transformer_config(raw)


def test_linear_cost_matches_hand_calculation() -> None:
    cost = linear_cost(
        "linear",
        rows=2,
        input_features=3,
        output_features=5,
        bytes_per_element=2,
    )

    assert cost.macs == 30
    assert cost.flops == 60
    assert cost.read_bytes == (2 * 3 + 3 * 5) * 2
    assert cost.write_bytes == 2 * 5 * 2


def test_linear_arithmetic_intensity_increases_with_batch() -> None:
    narrow = linear_cost(
        "narrow", rows=1, input_features=64, output_features=64, bytes_per_element=2
    )
    wide = linear_cost(
        "wide", rows=8, input_features=64, output_features=64, bytes_per_element=2
    )

    assert wide.arithmetic_intensity > narrow.arithmetic_intensity


def test_decoder_dag_has_one_softmax_for_shared_and_unique_fragments() -> None:
    model = load_transformer_config(MODEL_ROOT / "policy/qwen2.5-1.5b.yaml")
    dag = build_decoder_layer_dag(
        model,
        branch_width=4,
        shared_context_tokens=128,
        unique_context_tokens=16,
    )
    operators = {operator.operator_id: operator for operator in dag.operators}

    assert sum(operator.kind is OperatorKind.SOFTMAX for operator in dag.operators) == 1
    assert operators["softmax"].dependencies == ("shared_score", "unique_score")
    assert set(operators["o_proj"].dependencies) == {
        "shared_context",
        "unique_context",
    }


def test_shared_attention_intensity_scales_but_unique_intensity_stays_constant() -> None:
    model = load_transformer_config(MODEL_ROOT / "policy/qwen2.5-1.5b.yaml")
    width_one = build_decoder_layer_dag(
        model,
        branch_width=1,
        shared_context_tokens=128,
        unique_context_tokens=16,
    )
    width_eight = build_decoder_layer_dag(
        model,
        branch_width=8,
        shared_context_tokens=128,
        unique_context_tokens=16,
    )
    one = {operator.operator_id: operator for operator in width_one.operators}
    eight = {operator.operator_id: operator for operator in width_eight.operators}

    assert eight["shared_score"].arithmetic_intensity > one[
        "shared_score"
    ].arithmetic_intensity
    assert eight["unique_score"].arithmetic_intensity == pytest.approx(
        one["unique_score"].arithmetic_intensity
    )
