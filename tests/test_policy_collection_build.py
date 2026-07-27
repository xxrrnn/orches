from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from orches.cli import main
from orches.collectors import (
    CollectionStatus,
    ComputeOptimalTtsAdapter,
    PolicyCollectionBuildConfig,
    PolicyRequestFinishedEvent,
    PolicyRequestStartedEvent,
    PolicySelectionEvent,
    TraceHostProbe,
    build_policy_collection,
    parse_trace_host_probe,
    read_trace_host_probe,
    write_policy_events,
    write_trace_host_probe,
)
from orches.errors import WorkloadTraceError
from orches.workload import (
    Modality,
    ModelRef,
    PolicyCollectionMode,
    PolicyRuntimeEnvironment,
    PolicySelectionKind,
    PolicyTraceProvenance,
    SamplingConfig,
    SourceKind,
    read_policy_jsonl,
    read_policy_manifest,
    validate_policy_manifest,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _provenance() -> PolicyTraceProvenance:
    return PolicyTraceProvenance(
        source_kind=SourceKind.COLLECTED,
        pipeline="compute-optimal-tts",
        pipeline_revision="0ee2578",
        dataset_revision="math-500-revision",
        tokenizer=ModelRef("policy", "tokenizer-revision"),
        policy_model=ModelRef("policy", "model-revision"),
        inference_engine=ModelRef("vllm", "engine-revision"),
        collector=ModelRef("orches", "collector-revision"),
        selector=ModelRef("prm", "selector-revision"),
    )


def _started(request_id: str, dataset_id: str) -> PolicyRequestStartedEvent:
    return PolicyRequestStartedEvent(
        event_index=0,
        request_id=request_id,
        dataset="MATH-500",
        dataset_id=dataset_id,
        modality=Modality.TEXT,
        difficulty="Level 3",
        question_length_bucket=None,
        image_tokens=0,
        search_width=2,
        beam_size=1,
        seed=0,
        dtype="bfloat16",
        sampling=SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=64),
        collection_mode=PolicyCollectionMode.OPAQUE_SELECTOR,
        provenance=_provenance(),
    )


def _successful_events(request_id: str, dataset_id: str):
    generation = ComputeOptimalTtsAdapter.generation_event(
        {
            "request_id": f"worker-{dataset_id}",
            "prompt_token_ids": [10, 11],
            "output_token_ids": [[100], [110]],
            "token_ready_indices": [[0], [1]],
            "materialized_output_tokens": [0, 0],
            "finish_reason": ["stop", "stop"],
        },
        event_index=1,
        request_id=request_id,
        step_index=0,
        call_id=f"{request_id}/step-0/call-0",
        parent_candidate_id=None,
        rng_seed=0,
        rng_stream_id=f"{request_id}/step-0/call-0",
    )
    return [
        _started(request_id, dataset_id),
        generation,
        PolicySelectionEvent(
            event_index=2,
            request_id=request_id,
            step_index=0,
            decision_id=f"{request_id}/selection-0",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=(generation.outputs[0].candidate_id,),
            pruned_candidate_ids=(generation.outputs[1].candidate_id,),
        ),
        PolicyRequestFinishedEvent(
            event_index=3,
            request_id=request_id,
            status=CollectionStatus.SUCCESS,
            error=None,
        ),
    ]


def _ready_probe(*, bf16_supported: bool = True) -> TraceHostProbe:
    return TraceHostProbe(
        probe_schema_version=1,
        runtime=PolicyRuntimeEnvironment(
            python_version="3.10.12",
            platform="Linux-x86_64",
            accelerator="NVIDIA RTX 5070 Ti",
            accelerator_count=1,
            driver_version="test-driver",
            cuda_version="12.8",
            torch_version="2.8.0",
        ),
        python_executable="/trace/.venv/bin/python",
        driver_accelerators=("NVIDIA RTX 5070 Ti",),
        bf16_supported=bf16_supported,
        collection_ready=True,
        warnings=(),
    )


def _write_build_fixture(tmp_path: Path, probe: TraceHostProbe | None = None) -> Path:
    event_dir = tmp_path / "raw/events"
    event_dir.mkdir(parents=True)
    write_policy_events(
        event_dir / "item-0-sample-0.events.jsonl",
        _successful_events("MATH-500/item-0/sample-0", "0"),
    )
    oom_started = _started("MATH-500/item-1/sample-0", "1")
    write_policy_events(
        event_dir / "item-1-sample-0.events.jsonl",
        [
            oom_started,
            PolicyRequestFinishedEvent(
                event_index=1,
                request_id=oom_started.request_id,
                status=CollectionStatus.OOM,
                error="RuntimeError: CUDA out of memory",
            ),
        ],
    )
    host_path = tmp_path / "inputs/host.json"
    write_trace_host_probe(host_path, _ready_probe() if probe is None else probe)
    lock_path = tmp_path / "inputs/uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    config_path = tmp_path / "inputs/build.json"
    config_path.write_text(
        json.dumps(
            {
                "build_schema_version": 1,
                "event_dir": "../raw/events",
                "trace_path": "../output/trace.policy.jsonl",
                "manifest_path": "../output/manifest.json",
                "report_path": "../output/report.json",
                "uv_lock_path": "uv.lock",
                "host_probe_path": "host.json",
                "collection_command": [
                    "bash",
                    "scripts/run.sh",
                    "--width",
                    "2",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_build_policy_collection_retains_success_and_oom_evidence(
    tmp_path, capsys
) -> None:
    config_path = _write_build_fixture(tmp_path)

    status = main(["build-policy-collection", str(config_path), "--json"])
    report = json.loads(capsys.readouterr().out)

    assert status == 0
    assert report["event_files"] == 2
    assert report["successful_traces"] == 1
    assert report["status_counts"] == {"failed": 0, "oom": 1, "success": 1}
    assert report["requests"][1]["status"] == "oom"
    assert report["requests"][1]["trace_buildable"] is False

    trace_path = tmp_path / "output/trace.policy.jsonl"
    manifest_path = tmp_path / "output/manifest.json"
    traces = read_policy_jsonl(trace_path)
    manifest = read_policy_manifest(manifest_path)
    assert [trace.request_id for trace in traces] == [
        "MATH-500/item-0/sample-0"
    ]
    assert {artifact.role for artifact in manifest.raw_artifacts} == {
        "policy_events",
        "trace_host_probe",
        "collection_build_config",
        "uv_lock",
    }
    assert len(manifest.raw_artifacts) == 5
    assert all(
        not Path(artifact.path).is_absolute()
        for artifact in manifest.raw_artifacts
    )
    validate_policy_manifest(
        manifest,
        trace_path,
        artifact_root=manifest_path.parent,
    )

    validate_status = main(
        [
            "validate-policy-trace",
            str(trace_path),
            "--manifest",
            str(manifest_path),
            "--json",
        ]
    )
    validation = json.loads(capsys.readouterr().out)
    assert validate_status == 0
    assert validation["manifest_valid"] is True


def test_build_policy_collection_is_byte_stable(tmp_path) -> None:
    config_path = _write_build_fixture(tmp_path)
    config = PolicyCollectionBuildConfig.read(config_path)

    build_policy_collection(config_path)
    first = (
        config.trace_path.read_bytes(),
        config.manifest_path.read_bytes(),
        config.report_path.read_bytes(),
    )
    build_policy_collection(config_path)

    assert first == (
        config.trace_path.read_bytes(),
        config.manifest_path.read_bytes(),
        config.report_path.read_bytes(),
    )


def test_build_policy_collection_rejects_mixed_cell_width(tmp_path) -> None:
    config_path = _write_build_fixture(tmp_path)
    config = PolicyCollectionBuildConfig.read(config_path)
    mixed_started = replace(
        _started("MATH-500/item-1/sample-0", "1"),
        search_width=4,
    )
    write_policy_events(
        config.event_dir / "item-1-sample-0.events.jsonl",
        [
            mixed_started,
            PolicyRequestFinishedEvent(
                event_index=1,
                request_id=mixed_started.request_id,
                status=CollectionStatus.OOM,
                error="RuntimeError: CUDA out of memory",
            ),
        ],
    )

    with pytest.raises(WorkloadTraceError, match="mix cell configuration"):
        build_policy_collection(config_path)


def test_build_policy_collection_rejects_bf16_on_unsupported_host(tmp_path) -> None:
    config_path = _write_build_fixture(
        tmp_path,
        probe=_ready_probe(bf16_supported=False),
    )

    with pytest.raises(WorkloadTraceError, match="BF16 support"):
        build_policy_collection(config_path)


def test_all_failed_collection_writes_report_but_no_empty_trace(tmp_path) -> None:
    config_path = _write_build_fixture(tmp_path)
    config = PolicyCollectionBuildConfig.read(config_path)
    failed_started = _started("MATH-500/item-0/sample-0", "0")
    write_policy_events(
        config.event_dir / "item-0-sample-0.events.jsonl",
        [
            failed_started,
            PolicyRequestFinishedEvent(
                event_index=1,
                request_id=failed_started.request_id,
                status=CollectionStatus.FAILED,
                error="RuntimeError: worker failed",
            ),
        ],
    )

    with pytest.raises(WorkloadTraceError, match="failure report written"):
        build_policy_collection(config_path)

    report = json.loads(config.report_path.read_text(encoding="utf-8"))
    assert report["collection_buildable"] is False
    assert report["status_counts"] == {"failed": 1, "oom": 1, "success": 0}
    assert not config.trace_path.exists()
    assert not config.manifest_path.exists()


def test_trace_host_probe_round_trip_and_strict_fields(tmp_path) -> None:
    path = tmp_path / "host.json"
    probe = _ready_probe()
    write_trace_host_probe(path, probe)

    assert read_trace_host_probe(path) == probe
    raw = probe.to_dict()
    raw["runtime"]["unexpected"] = "value"
    with pytest.raises(WorkloadTraceError, match="unknown"):
        parse_trace_host_probe(raw)


def test_policy_collection_example_config_matches_strict_schema() -> None:
    config = PolicyCollectionBuildConfig.read(
        PROJECT_ROOT
        / "configs/workloads/compute_optimal_tts_build.example.json"
    )

    assert config.event_dir == (
        PROJECT_ROOT / "artifacts/raw/compute-optimal-tts-pilot"
    )
    assert config.collection_command[-2:] == ("--bs", "3")
