"""Build a reproducible generation-only collection from terminal raw events."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import WorkloadTraceError
from ..workload.policy_io import policy_trace_sha256, write_policy_jsonl
from ..workload.policy_manifest import (
    REQUIRED_MISSING_EVIDENCE,
    PolicyArtifactDigest,
    build_policy_manifest,
    write_policy_manifest,
)
from ..workload.policy_schema import PolicyRequestTrace
from .host_probe import read_trace_host_probe
from .policy_events import (
    AnyPolicyEvent,
    CollectionStatus,
    PolicyGenerationEvent,
    PolicyRequestStartedEvent,
    build_policy_request_from_events,
    policy_event_sha256,
    read_policy_events,
)


POLICY_COLLECTION_BUILD_SCHEMA_VERSION = 1
POLICY_COLLECTION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicyCollectionBuildConfig:
    """All paths and collection command required to close one trace cell."""

    event_dir: Path
    trace_path: Path
    manifest_path: Path
    report_path: Path
    uv_lock_path: Path
    host_probe_path: Path
    collection_command: tuple[str, ...]

    @classmethod
    def read(cls, path: str | Path) -> "PolicyCollectionBuildConfig":
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise WorkloadTraceError(
                f"cannot read policy collection build config {source}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise WorkloadTraceError(
                f"invalid policy collection build JSON {source}: {error.msg}"
            ) from error
        if not isinstance(raw, Mapping):
            raise WorkloadTraceError("policy collection build config must be an object")
        required = {
            "build_schema_version",
            "event_dir",
            "trace_path",
            "manifest_path",
            "report_path",
            "uv_lock_path",
            "host_probe_path",
            "collection_command",
        }
        _require_keys(raw, required, "policy collection build config")
        version = raw["build_schema_version"]
        if (
            isinstance(version, bool)
            or version != POLICY_COLLECTION_BUILD_SCHEMA_VERSION
        ):
            raise WorkloadTraceError(
                f"unsupported policy collection build schema version {version!r}"
            )
        command_raw = raw["collection_command"]
        if not isinstance(command_raw, list) or any(
            not isinstance(argument, str) or not argument.strip()
            for argument in command_raw
        ):
            raise WorkloadTraceError(
                "policy collection collection_command must be a non-empty "
                "string array"
            )
        if not command_raw:
            raise WorkloadTraceError(
                "policy collection collection_command must not be empty"
            )
        base = source.parent
        return cls(
            event_dir=_resolve_path(raw, "event_dir", base),
            trace_path=_resolve_path(raw, "trace_path", base),
            manifest_path=_resolve_path(raw, "manifest_path", base),
            report_path=_resolve_path(raw, "report_path", base),
            uv_lock_path=_resolve_path(raw, "uv_lock_path", base),
            host_probe_path=_resolve_path(raw, "host_probe_path", base),
            collection_command=tuple(command_raw),
        )


@dataclass(frozen=True)
class PolicyCollectionBuildResult:
    """Written artifacts and deterministic machine-readable summary."""

    traces: tuple[PolicyRequestTrace, ...]
    report: dict[str, Any]


def build_policy_collection(
    config_path: str | Path,
) -> PolicyCollectionBuildResult:
    """Validate terminal events and write trace, manifest, and status report."""

    source_config = Path(config_path).resolve()
    config = PolicyCollectionBuildConfig.read(source_config)
    _validate_output_paths(config)
    probe = read_trace_host_probe(config.host_probe_path)
    if not probe.collection_ready:
        raise WorkloadTraceError(
            "trace host probe is not collection-ready; run it inside the GPU "
            "trace UV environment"
        )
    if not config.uv_lock_path.is_file():
        raise WorkloadTraceError(
            f"policy collection UV lock does not exist: {config.uv_lock_path}"
        )
    if not config.event_dir.is_dir():
        raise WorkloadTraceError(
            f"policy collection event directory does not exist: {config.event_dir}"
        )
    event_paths = sorted(config.event_dir.rglob("*.events.jsonl"))
    if not event_paths:
        raise WorkloadTraceError(
            f"policy collection contains no terminal event files: {config.event_dir}"
        )

    records: list[
        tuple[
            str,
            Path,
            list[AnyPolicyEvent],
            PolicyRequestStartedEvent,
            CollectionStatus,
        ]
    ] = []
    request_ids: set[str] = set()
    cell_fingerprint: tuple[Any, ...] | None = None
    for event_path in event_paths:
        events = read_policy_events(event_path)
        started = events[0]
        finished = events[-1]
        assert isinstance(started, PolicyRequestStartedEvent)
        fingerprint = _collection_cell_fingerprint(started)
        if cell_fingerprint is None:
            cell_fingerprint = fingerprint
        elif fingerprint != cell_fingerprint:
            raise WorkloadTraceError(
                "policy collection event files mix cell configuration; split "
                f"the run before {event_path}"
            )
        if started.request_id in request_ids:
            raise WorkloadTraceError(
                f"policy collection duplicates request ID {started.request_id!r}"
            )
        request_ids.add(started.request_id)
        records.append(
            (
                started.request_id,
                event_path,
                events,
                started,
                finished.status,
            )
        )
    records.sort(key=lambda record: record[0])
    first_started = records[0][3]
    if (
        first_started.dtype.lower() in {"bf16", "bfloat16"}
        and not probe.bf16_supported
    ):
        raise WorkloadTraceError(
            "policy collection declares bfloat16 but the trace host probe does "
            "not report BF16 support"
        )

    traces: list[PolicyRequestTrace] = []
    request_rows: list[dict[str, Any]] = []
    raw_artifacts: list[PolicyArtifactDigest] = []
    artifact_root = config.manifest_path.parent
    for request_id, event_path, events, _, status in records:
        digest = policy_event_sha256(event_path)
        raw_artifacts.append(
            _relative_artifact(
                event_path,
                role="policy_events",
                artifact_root=artifact_root,
            )
        )
        generations = [
            event for event in events if isinstance(event, PolicyGenerationEvent)
        ]
        row: dict[str, Any] = {
            "request_id": request_id,
            "event_file": os.path.relpath(event_path, config.report_path.parent),
            "event_sha256": digest,
            "status": status.value,
            "generation_calls": len(generations),
            "candidates": sum(len(event.outputs) for event in generations),
            "generated_tokens": sum(
                len(output.generated_token_ids)
                for event in generations
                for output in event.outputs
            ),
            "trace_buildable": False,
        }
        if status is CollectionStatus.SUCCESS:
            traces.append(
                build_policy_request_from_events(
                    events,
                    raw_event_sha256=digest,
                )
            )
            row["trace_buildable"] = True
            row["error"] = None
        else:
            row["error"] = events[-1].error
        request_rows.append(row)

    counts = {
        status.value: sum(row["status"] == status.value for row in request_rows)
        for status in CollectionStatus
    }
    if not traces:
        failure_report = {
            "report_schema_version": POLICY_COLLECTION_REPORT_SCHEMA_VERSION,
            "workload_scope": "generation_only",
            "paper_eligible": False,
            "generation_evaluation_eligible": False,
            "collection_buildable": False,
            "event_dir": os.path.relpath(
                config.event_dir, config.report_path.parent
            ),
            "trace_file": None,
            "trace_sha256": None,
            "manifest_file": None,
            "host_probe_file": os.path.relpath(
                config.host_probe_path, config.report_path.parent
            ),
            "host_warnings": list(probe.warnings),
            "event_files": len(records),
            "successful_traces": 0,
            "status_counts": counts,
            "requests": request_rows,
            "missing_evidence": sorted(REQUIRED_MISSING_EVIDENCE),
        }
        _write_report(config.report_path, failure_report)
        raise WorkloadTraceError(
            "policy collection has no successful request to serialize; "
            f"failure report written to {config.report_path}"
        )
    raw_artifacts.extend(
        (
            _relative_artifact(
                config.host_probe_path,
                role="trace_host_probe",
                artifact_root=artifact_root,
            ),
            _relative_artifact(
                source_config,
                role="collection_build_config",
                artifact_root=artifact_root,
            ),
            _relative_artifact(
                config.uv_lock_path,
                role="uv_lock",
                artifact_root=artifact_root,
            ),
        )
    )
    write_policy_jsonl(config.trace_path, traces)
    trace_file = os.path.relpath(config.trace_path, artifact_root)
    manifest = build_policy_manifest(
        config.trace_path,
        traces,
        uv_lock_path=config.uv_lock_path,
        command=config.collection_command,
        runtime=probe.runtime,
        raw_artifacts=tuple(raw_artifacts),
        trace_file=trace_file,
        artifact_root=artifact_root,
    )
    write_policy_manifest(config.manifest_path, manifest)

    report = {
        "report_schema_version": POLICY_COLLECTION_REPORT_SCHEMA_VERSION,
        "workload_scope": "generation_only",
        "paper_eligible": False,
        "generation_evaluation_eligible": manifest.generation_evaluation_eligible,
        "collection_buildable": True,
        "event_dir": os.path.relpath(config.event_dir, config.report_path.parent),
        "trace_file": os.path.relpath(config.trace_path, config.report_path.parent),
        "trace_sha256": policy_trace_sha256(config.trace_path),
        "manifest_file": os.path.relpath(
            config.manifest_path, config.report_path.parent
        ),
        "host_probe_file": os.path.relpath(
            config.host_probe_path, config.report_path.parent
        ),
        "host_warnings": list(probe.warnings),
        "event_files": len(records),
        "successful_traces": len(traces),
        "status_counts": counts,
        "requests": request_rows,
        "missing_evidence": list(manifest.missing_evidence),
    }
    _write_report(config.report_path, report)
    return PolicyCollectionBuildResult(traces=tuple(traces), report=report)


def _collection_cell_fingerprint(
    started: PolicyRequestStartedEvent,
) -> tuple[Any, ...]:
    return (
        started.dataset,
        started.modality,
        started.search_width,
        started.beam_size,
        started.seed,
        started.dtype,
        started.sampling,
        started.collection_mode,
        started.provenance,
    )


def _relative_artifact(
    path: Path, *, role: str, artifact_root: Path
) -> PolicyArtifactDigest:
    return PolicyArtifactDigest(
        path=os.path.relpath(path, artifact_root),
        role=role,
        sha256=_file_sha256(path),
    )


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkloadTraceError(f"cannot hash artifact {path}: {error}") from error


def _validate_output_paths(config: PolicyCollectionBuildConfig) -> None:
    outputs = {config.trace_path, config.manifest_path, config.report_path}
    if len(outputs) != 3:
        raise WorkloadTraceError(
            "policy collection trace, manifest, and report paths must be distinct"
        )
    for output in outputs:
        try:
            output.relative_to(config.event_dir)
        except ValueError:
            continue
        raise WorkloadTraceError(
            "policy collection outputs must not be written inside the raw event "
            f"directory: {output}"
        )


def _resolve_path(raw: Mapping[str, Any], key: str, base: Path) -> Path:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise WorkloadTraceError(
            f"policy collection build config {key} must be a non-empty string"
        )
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_keys(raw: Mapping[str, Any], required: set[str], path: str) -> None:
    actual = set(raw)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise WorkloadTraceError(f"{path} fields " + "; ".join(details))


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot write policy collection report {path}: {error}"
        ) from error
