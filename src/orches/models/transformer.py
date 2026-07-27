"""Validated Llama/Qwen architecture profiles used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import yaml

from ..errors import ConfigurationError
from ..provenance import EvidenceStatus


RawMapping: TypeAlias = Mapping[str, Any]


class ModelRole(str, Enum):
    """The TTC pipeline role served by a transformer checkpoint."""

    POLICY = "policy"
    PRM = "prm"
    VISION = "vision"


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture values needed for shape and traffic expansion."""

    schema_version: int
    name: str
    family: str
    role: ModelRole
    source_repository: str
    source_revision: str
    source_status: EvidenceStatus
    source_note: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    bytes_per_element: int
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"unsupported transformer schema version {self.schema_version!r}"
            )
        for field_name, value in (
            ("name", self.name),
            ("family", self.family),
            ("source_repository", self.source_repository),
            ("source_revision", self.source_revision),
            ("source_note", self.source_note),
        ):
            if not value.strip():
                raise ConfigurationError(f"model.{field_name} must not be empty")
        for field_name, value in (
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("bytes_per_element", self.bytes_per_element),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(
                    f"model.architecture.{field_name} must be a positive integer"
                )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ConfigurationError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigurationError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.bytes_per_element not in (1, 2, 4):
            raise ConfigurationError("bytes_per_element must be one of 1, 2, or 4")

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes for K and V across every decoder layer for one sequence token."""

        return (
            2
            * self.num_hidden_layers
            * self.kv_hidden_size
            * self.bytes_per_element
        )

    @property
    def decoder_weight_elements(self) -> int:
        """Approximate decoder matrix elements, excluding biases and norms."""

        attention_per_layer = (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.kv_hidden_size
        )
        swiglu_per_layer = 3 * self.hidden_size * self.intermediate_size
        return self.num_hidden_layers * (attention_per_layer + swiglu_per_layer)

    @property
    def embedding_weight_elements(self) -> int:
        copies = 1 if self.tie_word_embeddings else 2
        return copies * self.vocab_size * self.hidden_size

    @property
    def estimated_weight_bytes(self) -> int:
        return (
            self.decoder_weight_elements + self.embedding_weight_elements
        ) * self.bytes_per_element

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "family": self.family,
            "role": self.role.value,
            "source": {
                "repository": self.source_repository,
                "revision": self.source_revision,
                "status": self.source_status.value,
                "note": self.source_note,
            },
            "derived": {
                "head_dim": self.head_dim,
                "kv_hidden_size": self.kv_hidden_size,
                "attention_group_size": self.attention_group_size,
                "kv_bytes_per_token": self.kv_bytes_per_token,
                "decoder_weight_elements": self.decoder_weight_elements,
                "estimated_weight_bytes": self.estimated_weight_bytes,
            },
        }


def _mapping(value: Any, path: str) -> RawMapping:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _keys(mapping: RawMapping, path: str, required: set[str]) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ConfigurationError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{path} has unknown fields: {', '.join(unknown)}")


def _string(mapping: RawMapping, key: str, path: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigurationError(f"{path}.{key} must be a string")
    return value


def _integer(mapping: RawMapping, key: str, path: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}.{key} must be an integer")
    return value


def _boolean(mapping: RawMapping, key: str, path: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path}.{key} must be a boolean")
    return value


def parse_transformer_config(raw_value: Any) -> TransformerConfig:
    """Parse one strict transformer architecture mapping."""

    root = _mapping(raw_value, "model")
    _keys(
        root,
        "model",
        {"schema_version", "name", "family", "role", "source", "architecture"},
    )
    source = _mapping(root["source"], "model.source")
    _keys(source, "model.source", {"repository", "revision", "status", "note"})
    architecture = _mapping(root["architecture"], "model.architecture")
    architecture_fields = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "bytes_per_element",
        "tie_word_embeddings",
    }
    _keys(architecture, "model.architecture", architecture_fields)

    try:
        role = ModelRole(_string(root, "role", "model"))
    except ValueError as error:
        raise ConfigurationError("model.role must be policy, prm, or vision") from error
    try:
        status = EvidenceStatus(_string(source, "status", "model.source"))
    except ValueError as error:
        raise ConfigurationError(
            "model.source.status must be PAPER, INHERITED, CALIBRATED, or ASSUMED"
        ) from error

    return TransformerConfig(
        schema_version=_integer(root, "schema_version", "model"),
        name=_string(root, "name", "model"),
        family=_string(root, "family", "model"),
        role=role,
        source_repository=_string(source, "repository", "model.source"),
        source_revision=_string(source, "revision", "model.source"),
        source_status=status,
        source_note=_string(source, "note", "model.source"),
        hidden_size=_integer(architecture, "hidden_size", "model.architecture"),
        intermediate_size=_integer(
            architecture, "intermediate_size", "model.architecture"
        ),
        num_hidden_layers=_integer(
            architecture, "num_hidden_layers", "model.architecture"
        ),
        num_attention_heads=_integer(
            architecture, "num_attention_heads", "model.architecture"
        ),
        num_key_value_heads=_integer(
            architecture, "num_key_value_heads", "model.architecture"
        ),
        head_dim=_integer(architecture, "head_dim", "model.architecture"),
        vocab_size=_integer(architecture, "vocab_size", "model.architecture"),
        max_position_embeddings=_integer(
            architecture, "max_position_embeddings", "model.architecture"
        ),
        bytes_per_element=_integer(
            architecture, "bytes_per_element", "model.architecture"
        ),
        tie_word_embeddings=_boolean(
            architecture, "tie_word_embeddings", "model.architecture"
        ),
    )


def load_transformer_config(path: str | Path) -> TransformerConfig:
    """Read and validate one transformer YAML architecture profile."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read model config {source}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in model config {source}: {error}") from error
    if raw is None:
        raise ConfigurationError(f"model config {source} is empty")
    return parse_transformer_config(raw)
