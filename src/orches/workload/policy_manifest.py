"""Reproducibility manifest for one generation-only policy trace collection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from ..errors import WorkloadTraceError
from .policy_io import policy_trace_sha256, read_policy_jsonl
from .policy_schema import (
    PolicyCollectionMode,
    PolicyRequestTrace,
    WorkloadScope,
)
from .schema import ModelRef, SourceKind


POLICY_MANIFEST_SCHEMA_VERSION = 1
REQUIRED_MISSING_EVIDENCE = frozenset(
    {"verifier_activity", "verifier_scores", "verifier_token_tensors"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_nonempty(name: str, value: str) -> None:
    if not value.strip():
        raise WorkloadTraceError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise WorkloadTraceError(f"{name} must be a lowercase SHA-256 digest")


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkloadTraceError(f"cannot hash artifact {source}: {error}") from error


@dataclass(frozen=True)
class PolicyRuntimeEnvironment:
    """Software and accelerator identity visible to the collector process."""

    python_version: str
    platform: str
    accelerator: str
    accelerator_count: int
    driver_version: str
    cuda_version: str
    torch_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("runtime.python_version", self.python_version),
            ("runtime.platform", self.platform),
            ("runtime.accelerator", self.accelerator),
            ("runtime.driver_version", self.driver_version),
            ("runtime.cuda_version", self.cuda_version),
            ("runtime.torch_version", self.torch_version),
        ):
            _require_nonempty(name, value)
        if (
            isinstance(self.accelerator_count, bool)
            or not isinstance(self.accelerator_count, int)
            or self.accelerator_count < 0
        ):
            raise WorkloadTraceError(
                "runtime.accelerator_count must be a non-negative integer"
            )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "accelerator": self.accelerator,
            "accelerator_count": self.accelerator_count,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
        }


@dataclass(frozen=True)
class PolicyArtifactDigest:
    """Content-addressed raw artifact retained beside a sanitized trace."""

    path: str
    role: str
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("artifact.path", self.path)
        _require_nonempty("artifact.role", self.role)
        _require_sha256("artifact.sha256", self.sha256)

    @classmethod
    def from_path(cls, path: str | Path, role: str) -> "PolicyArtifactDigest":
        source = Path(path)
        return cls(path=str(source), role=role, sha256=_file_sha256(source))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True)
class PolicyTraceManifest:
    """Immutable binding from collection inputs to exact trace bytes."""

    manifest_schema_version: int
    workload_scope: WorkloadScope
    paper_eligible: bool
    generation_evaluation_eligible: bool
    missing_evidence: tuple[str, ...]
    trace_file: str
    trace_sha256: str
    request_ids: tuple[str, ...]
    collection_mode: PolicyCollectionMode
    collector: ModelRef
    source_checkout: ModelRef
    uv_lock_sha256: str
    command: tuple[str, ...]
    runtime: PolicyRuntimeEnvironment
    raw_artifacts: tuple[PolicyArtifactDigest, ...]

    def __post_init__(self) -> None:
        if self.manifest_schema_version != POLICY_MANIFEST_SCHEMA_VERSION:
            raise WorkloadTraceError(
                "unsupported policy manifest schema version "
                f"{self.manifest_schema_version!r}"
            )
        if self.workload_scope is not WorkloadScope.GENERATION_ONLY:
            raise WorkloadTraceError("policy manifest scope must be generation_only")
        if self.paper_eligible is not False:
            raise WorkloadTraceError(
                "generation-only manifests must set paper_eligible=false"
            )
        if not REQUIRED_MISSING_EVIDENCE <= set(self.missing_evidence):
            missing = sorted(REQUIRED_MISSING_EVIDENCE - set(self.missing_evidence))
            raise WorkloadTraceError(
                "policy manifest must declare missing evidence: " + ", ".join(missing)
            )
        if len(set(self.missing_evidence)) != len(self.missing_evidence):
            raise WorkloadTraceError(
                "manifest.missing_evidence must not contain duplicates"
            )
        for value in self.missing_evidence:
            _require_nonempty("manifest.missing_evidence", value)
        _require_nonempty("manifest.trace_file", self.trace_file)
        _require_sha256("manifest.trace_sha256", self.trace_sha256)
        _require_sha256("manifest.uv_lock_sha256", self.uv_lock_sha256)
        if not self.request_ids:
            raise WorkloadTraceError("manifest.request_ids must not be empty")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise WorkloadTraceError("manifest.request_ids must not contain duplicates")
        for request_id in self.request_ids:
            _require_nonempty("manifest.request_ids", request_id)
        if not self.command or any(not argument for argument in self.command):
            raise WorkloadTraceError(
                "manifest.command must contain non-empty command arguments"
            )
        artifact_paths = tuple(artifact.path for artifact in self.raw_artifacts)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise WorkloadTraceError("manifest raw artifact paths must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "workload_scope": self.workload_scope.value,
            "paper_eligible": self.paper_eligible,
            "generation_evaluation_eligible": self.generation_evaluation_eligible,
            "missing_evidence": list(self.missing_evidence),
            "trace_file": self.trace_file,
            "trace_sha256": self.trace_sha256,
            "request_ids": list(self.request_ids),
            "collection_mode": self.collection_mode.value,
            "collector": self.collector.to_dict(),
            "source_checkout": self.source_checkout.to_dict(),
            "uv_lock_sha256": self.uv_lock_sha256,
            "command": list(self.command),
            "runtime": self.runtime.to_dict(),
            "raw_artifacts": [artifact.to_dict() for artifact in self.raw_artifacts],
        }


def build_policy_manifest(
    trace_path: str | Path,
    traces: list[PolicyRequestTrace],
    *,
    uv_lock_path: str | Path,
    command: tuple[str, ...],
    runtime: PolicyRuntimeEnvironment,
    raw_artifacts: tuple[PolicyArtifactDigest, ...],
    trace_file: str | None = None,
    artifact_root: str | Path | None = None,
) -> PolicyTraceManifest:
    """Build and cross-check a manifest for already serialized trace bytes."""

    if not traces:
        raise WorkloadTraceError(
            "cannot build a manifest for an empty trace collection"
        )
    modes = {trace.collection_mode for trace in traces}
    provenances = {trace.provenance for trace in traces}
    if len(modes) != 1 or len(provenances) != 1:
        raise WorkloadTraceError(
            "one policy manifest requires one collection mode and provenance"
        )
    provenance = traces[0].provenance
    manifest = PolicyTraceManifest(
        manifest_schema_version=POLICY_MANIFEST_SCHEMA_VERSION,
        workload_scope=WorkloadScope.GENERATION_ONLY,
        paper_eligible=False,
        generation_evaluation_eligible=all(
            trace.generation_evaluation_eligible for trace in traces
        ),
        missing_evidence=tuple(sorted(REQUIRED_MISSING_EVIDENCE)),
        trace_file=str(trace_path) if trace_file is None else trace_file,
        trace_sha256=policy_trace_sha256(trace_path),
        request_ids=tuple(trace.request_id for trace in traces),
        collection_mode=traces[0].collection_mode,
        collector=provenance.collector,
        source_checkout=ModelRef(
            name=provenance.pipeline,
            revision=provenance.pipeline_revision,
        ),
        uv_lock_sha256=_file_sha256(uv_lock_path),
        command=command,
        runtime=runtime,
        raw_artifacts=raw_artifacts,
    )
    validate_policy_manifest(
        manifest,
        trace_path,
        traces=traces,
        artifact_root=artifact_root,
    )
    return manifest


def validate_policy_manifest(
    manifest: PolicyTraceManifest,
    trace_path: str | Path,
    *,
    traces: list[PolicyRequestTrace] | None = None,
    artifact_root: str | Path | None = None,
) -> None:
    """Reject a manifest whose hashes or identities do not match the trace."""

    loaded = read_policy_jsonl(trace_path) if traces is None else traces
    if policy_trace_sha256(trace_path) != manifest.trace_sha256:
        raise WorkloadTraceError(
            "policy manifest trace_sha256 does not match trace bytes"
        )
    if tuple(trace.request_id for trace in loaded) != manifest.request_ids:
        raise WorkloadTraceError("policy manifest request order does not match trace")
    if any(trace.collection_mode is not manifest.collection_mode for trace in loaded):
        raise WorkloadTraceError("policy manifest collection mode does not match trace")
    expected_eligibility = all(
        trace.generation_evaluation_eligible for trace in loaded
    )
    if manifest.generation_evaluation_eligible is not expected_eligibility:
        raise WorkloadTraceError(
            "policy manifest generation eligibility does not match trace evidence"
        )

    provenances = {trace.provenance for trace in loaded}
    if len(provenances) != 1:
        raise WorkloadTraceError("policy trace mixes provenance")
    provenance = loaded[0].provenance
    expected_source = ModelRef(provenance.pipeline, provenance.pipeline_revision)
    if manifest.collector != provenance.collector:
        raise WorkloadTraceError("policy manifest collector does not match trace")
    if manifest.source_checkout != expected_source:
        raise WorkloadTraceError("policy manifest source checkout does not match trace")
    if (
        provenance.source_kind is SourceKind.COLLECTED
        and not manifest.raw_artifacts
    ):
        raise WorkloadTraceError(
            "collected policy traces require at least one raw artifact digest"
        )

    artifact_hashes = {artifact.sha256 for artifact in manifest.raw_artifacts}
    uv_lock_hashes = {
        artifact.sha256
        for artifact in manifest.raw_artifacts
        if artifact.role == "uv_lock"
    }
    if uv_lock_hashes and uv_lock_hashes != {manifest.uv_lock_sha256}:
        raise WorkloadTraceError(
            "policy manifest UV lock artifact does not match uv_lock_sha256"
        )
    decision_hashes = {
        step.selection.decision_artifact_sha256
        for trace in loaded
        for step in trace.steps
        if step.selection is not None
        and step.selection.decision_artifact_sha256 is not None
    }
    if not decision_hashes <= artifact_hashes:
        raise WorkloadTraceError(
            "opaque selection decision hash is absent from manifest raw artifacts"
        )
    root = None if artifact_root is None else Path(artifact_root)
    for artifact in manifest.raw_artifacts:
        artifact_path = Path(artifact.path)
        if not artifact_path.is_absolute() and root is not None:
            artifact_path = root / artifact_path
        if _file_sha256(artifact_path) != artifact.sha256:
            raise WorkloadTraceError(
                f"policy manifest raw artifact hash does not match {artifact.path!r}"
            )


RawMapping: TypeAlias = Mapping[str, Any]


def _mapping(value: Any, path: str) -> RawMapping:
    if not isinstance(value, Mapping):
        raise WorkloadTraceError(f"{path} must be an object")
    return value


def _keys(mapping: RawMapping, path: str, required: set[str]) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise WorkloadTraceError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise WorkloadTraceError(f"{path} has unknown fields: {', '.join(unknown)}")


def _string(mapping: RawMapping, key: str, path: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise WorkloadTraceError(f"{path}.{key} must be a string")
    return value


def _string_array(raw: Any, path: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise WorkloadTraceError(f"{path} must be a string array")
    return tuple(raw)


def _boolean(mapping: RawMapping, key: str, path: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise WorkloadTraceError(f"{path}.{key} must be a boolean")
    return value


def _integer(mapping: RawMapping, key: str, path: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadTraceError(f"{path}.{key} must be an integer")
    return value


def _enum(enum_type, raw: Any, path: str):
    if not isinstance(raw, str):
        raise WorkloadTraceError(f"{path} must be a string")
    try:
        return enum_type(raw)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise WorkloadTraceError(f"{path} must be one of: {allowed}") from error


def _model_ref(raw_value: Any, path: str) -> ModelRef:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"name", "revision"})
    return ModelRef(_string(raw, "name", path), _string(raw, "revision", path))


def _runtime(raw_value: Any, path: str) -> PolicyRuntimeEnvironment:
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "python_version",
            "platform",
            "accelerator",
            "accelerator_count",
            "driver_version",
            "cuda_version",
            "torch_version",
        },
    )
    return PolicyRuntimeEnvironment(
        python_version=_string(raw, "python_version", path),
        platform=_string(raw, "platform", path),
        accelerator=_string(raw, "accelerator", path),
        accelerator_count=_integer(raw, "accelerator_count", path),
        driver_version=_string(raw, "driver_version", path),
        cuda_version=_string(raw, "cuda_version", path),
        torch_version=_string(raw, "torch_version", path),
    )


def _artifact(raw_value: Any, path: str) -> PolicyArtifactDigest:
    raw = _mapping(raw_value, path)
    _keys(raw, path, {"path", "role", "sha256"})
    return PolicyArtifactDigest(
        path=_string(raw, "path", path),
        role=_string(raw, "role", path),
        sha256=_string(raw, "sha256", path),
    )


def parse_policy_manifest(raw_value: Any) -> PolicyTraceManifest:
    """Parse one strict policy collection manifest object."""

    path = "manifest"
    raw = _mapping(raw_value, path)
    _keys(
        raw,
        path,
        {
            "manifest_schema_version",
            "workload_scope",
            "paper_eligible",
            "generation_evaluation_eligible",
            "missing_evidence",
            "trace_file",
            "trace_sha256",
            "request_ids",
            "collection_mode",
            "collector",
            "source_checkout",
            "uv_lock_sha256",
            "command",
            "runtime",
            "raw_artifacts",
        },
    )
    artifacts_raw = raw["raw_artifacts"]
    if not isinstance(artifacts_raw, list):
        raise WorkloadTraceError("manifest.raw_artifacts must be an array")
    return PolicyTraceManifest(
        manifest_schema_version=_integer(raw, "manifest_schema_version", path),
        workload_scope=_enum(
            WorkloadScope, raw["workload_scope"], "manifest.workload_scope"
        ),
        paper_eligible=_boolean(raw, "paper_eligible", path),
        generation_evaluation_eligible=_boolean(
            raw, "generation_evaluation_eligible", path
        ),
        missing_evidence=_string_array(
            raw["missing_evidence"], "manifest.missing_evidence"
        ),
        trace_file=_string(raw, "trace_file", path),
        trace_sha256=_string(raw, "trace_sha256", path),
        request_ids=_string_array(raw["request_ids"], "manifest.request_ids"),
        collection_mode=_enum(
            PolicyCollectionMode,
            raw["collection_mode"],
            "manifest.collection_mode",
        ),
        collector=_model_ref(raw["collector"], "manifest.collector"),
        source_checkout=_model_ref(
            raw["source_checkout"], "manifest.source_checkout"
        ),
        uv_lock_sha256=_string(raw, "uv_lock_sha256", path),
        command=_string_array(raw["command"], "manifest.command"),
        runtime=_runtime(raw["runtime"], "manifest.runtime"),
        raw_artifacts=tuple(
            _artifact(artifact, f"manifest.raw_artifacts[{index}]")
            for index, artifact in enumerate(artifacts_raw)
        ),
    )


def write_policy_manifest(path: str | Path, manifest: PolicyTraceManifest) -> None:
    """Atomically write one canonical, human-readable manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                manifest.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot write policy manifest {destination}: {error}"
        ) from error


def read_policy_manifest(path: str | Path) -> PolicyTraceManifest:
    """Read one strict policy collection manifest."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot read policy manifest {source}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise WorkloadTraceError(
            f"invalid JSON in policy manifest {source}: {error.msg}"
        ) from error
    return parse_policy_manifest(raw)
