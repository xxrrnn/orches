from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from orches.cli import main
from orches.collectors import (
    CollectionStatus,
    ComputeOptimalTtsAdapter,
    ComputeOptimalTtsCollectorConfig,
    ComputeOptimalTtsSession,
    PolicyRequestFinishedEvent,
    PolicyRequestStartedEvent,
    PolicySelectionEvent,
    VllmOutputSnapshot,
    VllmSnapshotAccumulator,
    build_policy_request_from_events,
    create_compute_optimal_tts_session,
    policy_event_sha256,
    read_policy_events,
    write_policy_events,
)
from orches.errors import WorkloadTraceError
from orches.workload import (
    Modality,
    ModelRef,
    PolicyCollectionMode,
    PolicySelectionKind,
    PolicyTraceProvenance,
    SamplingConfig,
    SourceKind,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _provenance() -> PolicyTraceProvenance:
    return PolicyTraceProvenance(
        source_kind=SourceKind.COLLECTED,
        pipeline="compute-optimal-tts",
        pipeline_revision="0ee2578",
        dataset_revision="math-500-revision",
        tokenizer=ModelRef("Qwen/Qwen2.5-1.5B-Instruct", "tokenizer-revision"),
        policy_model=ModelRef("Qwen/Qwen2.5-1.5B-Instruct", "model-revision"),
        inference_engine=ModelRef("vllm", "0.6.4.post1"),
        collector=ModelRef("orches.compute-optimal-tts", "collector-revision"),
        selector=ModelRef("opaque-upstream-prm", "selector-revision"),
    )


def _started() -> PolicyRequestStartedEvent:
    return PolicyRequestStartedEvent(
        event_index=0,
        request_id="math-0",
        dataset="MATH-500",
        dataset_id="test-0",
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


def _response(
    worker_request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[list[int]],
    token_ready_indices: list[list[int]],
    materialized_output_tokens: list[int],
) -> dict[str, object]:
    return {
        "request_id": worker_request_id,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "token_ready_indices": token_ready_indices,
        "materialized_output_tokens": materialized_output_tokens,
        "finish_reason": ["stop"] * len(output_token_ids),
        "text": ["sensitive output"] * len(output_token_ids),
        "usage": {"completion_tokens": sum(map(len, output_token_ids))},
    }


def _worker_result(
    worker_request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[list[int]],
    token_ready_indices: list[list[int]],
    materialized_output_tokens: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        worker_request_id=worker_request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        token_ready_indices=token_ready_indices,
        materialized_output_tokens=materialized_output_tokens,
        finish_reason=["stop"] * len(output_token_ids),
        text=["must never enter raw events"] * len(output_token_ids),
    )


def _write_collector_config(path: Path, event_dir: str = "events") -> None:
    path.write_text(
        json.dumps(
            {
                "config_schema_version": 1,
                "event_dir": event_dir,
                "dataset": "MATH-500",
                "dataset_revision": "math-500-revision",
                "dtype": "bfloat16",
                "tokenizer": {
                    "name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "revision": "tokenizer-revision",
                },
                "policy_model": {
                    "name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "revision": "model-revision",
                },
                "inference_engine": {
                    "name": "vllm",
                    "revision": "0.6.4.post1",
                },
                "collector": {
                    "name": "orches.compute-optimal-tts",
                    "revision": "collector-revision",
                },
                "selector": {
                    "name": "opaque-upstream-prm",
                    "revision": "selector-revision",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _successful_events():
    generation0 = ComputeOptimalTtsAdapter.generation_event(
        _response(
            "worker-0",
            [10, 11],
            [[100, 101], [110]],
            [[0, 2], [1]],
            [1, 0],
        ),
        event_index=1,
        request_id="math-0",
        step_index=0,
        call_id="math-0/step-0/call-0",
        parent_candidate_id=None,
        rng_seed=0,
        rng_stream_id="math-0/step-0/root",
    )
    selected0 = generation0.outputs[1].candidate_id
    generation1 = ComputeOptimalTtsAdapter.generation_event(
        _response(
            "worker-1",
            [10, 11, 110, 700],
            [[200], [210]],
            [[0], [1]],
            [0, 0],
        ),
        event_index=3,
        request_id="math-0",
        step_index=1,
        call_id="math-0/step-1/call-0",
        parent_candidate_id=selected0,
        rng_seed=1,
        rng_stream_id="math-0/step-1/parent-0",
    )
    return [
        _started(),
        generation0,
        PolicySelectionEvent(
            event_index=2,
            request_id="math-0",
            step_index=0,
            decision_id="math-0/select-0",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=(selected0,),
            pruned_candidate_ids=(generation0.outputs[0].candidate_id,),
        ),
        generation1,
        PolicySelectionEvent(
            event_index=4,
            request_id="math-0",
            step_index=1,
            decision_id="math-0/select-1",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=(generation1.outputs[0].candidate_id,),
            pruned_candidate_ids=(generation1.outputs[1].candidate_id,),
        ),
        PolicyRequestFinishedEvent(
            event_index=5,
            request_id="math-0",
            status=CollectionStatus.SUCCESS,
            error=None,
        ),
    ]


def test_compute_optimal_adapter_sanitizes_text_and_preserves_exact_tokens() -> None:
    event = _successful_events()[1]

    assert event.input_tokens.valid_token_ids == (10, 11)
    assert event.outputs[0].generated_token_ids == (100, 101)
    assert event.outputs[0].call_token_ready_indices == (0, 2)
    assert event.outputs[0].materialized_output_tokens == 1
    assert "text" not in event.to_dict()
    assert "usage" not in event.to_dict()


def test_vllm_snapshots_produce_linear_token_order_and_materialization() -> None:
    accumulator = VllmSnapshotAccumulator(expected_outputs=2)
    accumulator.observe(
        (10, 11),
        (
            VllmOutputSnapshot(0, (100,), None),
            VllmOutputSnapshot(1, (110,), None),
        ),
    )
    accumulator.observe(
        (10, 11),
        (
            VllmOutputSnapshot(0, (100, 101), "stop"),
            VllmOutputSnapshot(1, (110, 111), None),
        ),
    )
    accumulator.observe(
        (10, 11),
        (
            VllmOutputSnapshot(0, (100, 101), "stop"),
            VllmOutputSnapshot(1, (110, 111, 112), "length"),
        ),
    )

    payload = accumulator.final_payload("worker-0")

    assert payload["output_token_ids"] == [[100, 101], [110, 111, 112]]
    assert payload["token_ready_indices"] == [[0, 2], [1, 3, 4]]
    assert payload["materialized_output_tokens"] == [1, 2]
    assert payload["finish_reason"] == ["stop", "length"]


def test_vllm_accumulator_waits_for_every_requested_output() -> None:
    accumulator = VllmSnapshotAccumulator(expected_outputs=2)
    accumulator.observe(
        (10,),
        (VllmOutputSnapshot(0, (100,), "stop"),),
    )

    assert accumulator.is_complete is False

    accumulator.observe(
        (10,),
        (
            VllmOutputSnapshot(0, (100,), "stop"),
            VllmOutputSnapshot(1, (110,), "stop"),
        ),
    )
    assert accumulator.is_complete is True


def test_raw_events_round_trip_and_build_exact_policy_trace(tmp_path) -> None:
    events = _successful_events()
    first = tmp_path / "first.events.jsonl"
    second = tmp_path / "second.events.jsonl"

    write_policy_events(first, events)
    loaded = read_policy_events(first)
    write_policy_events(second, loaded)
    digest = policy_event_sha256(first)
    trace = build_policy_request_from_events(loaded, raw_event_sha256=digest)

    assert loaded == events
    assert first.read_bytes() == second.read_bytes()
    assert b"sensitive output" not in first.read_bytes()
    assert trace.generated_tokens == 5
    assert trace.materialized_output_tokens == 1
    assert trace.steps[0].generation_calls[0].candidates[1].terminal_kv_block_id == (
        "math-0/kv/root"
    )
    assert trace.steps[1].generation_calls[0].reused_kv_tokens == 2
    assert trace.steps[1].generation_calls[0].input_tokens.valid_token_ids == (
        10,
        11,
        110,
        700,
    )
    assert trace.steps[0].selection.decision_artifact_sha256 == digest


def test_raw_event_parser_rejects_decoded_text_field(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    write_policy_events(path, _successful_events())
    lines = path.read_text(encoding="utf-8").splitlines()
    generation = json.loads(lines[1])
    generation["outputs"][0]["decoded_text"] = "must not enter trace"
    lines[1] = json.dumps(generation)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(WorkloadTraceError, match="unknown fields"):
        read_policy_events(path)


def test_adapter_requires_worker_exact_token_fields() -> None:
    response = _response("worker", [10], [[11]], [[0]], [0])
    del response["prompt_token_ids"]

    with pytest.raises(WorkloadTraceError, match="prompt_token_ids"):
        ComputeOptimalTtsAdapter.generation_event(
            response,
            event_index=1,
            request_id="math-0",
            step_index=0,
            call_id="call-0",
            parent_candidate_id=None,
            rng_seed=0,
            rng_stream_id="stream-0",
        )


def test_adapter_rejects_noncontiguous_worker_token_order() -> None:
    response = _response("worker", [10], [[11], [12]], [[0], [2]], [0, 0])

    with pytest.raises(WorkloadTraceError, match="contiguous"):
        ComputeOptimalTtsAdapter.generation_event(
            response,
            event_index=1,
            request_id="math-0",
            step_index=0,
            call_id="call-0",
            parent_candidate_id=None,
            rng_seed=0,
            rng_stream_id="stream-0",
        )


def test_compute_optimal_session_records_search_and_validates_before_write(
    tmp_path,
) -> None:
    event_path = tmp_path / "request.events.jsonl"
    session = ComputeOptimalTtsSession(_started(), event_path)

    root_candidates = session.record_generation_result(
        _worker_result(
            "worker-0",
            [10, 11],
            [[100, 101], [110]],
            [[0, 2], [1]],
            [1, 0],
        ),
        step_index=0,
        parent_candidate_id=None,
        rng_seed=0,
    )
    session.record_selection(0, (root_candidates[1],))
    child_candidates = session.record_generation_result(
        _worker_result(
            "worker-1",
            [10, 11, 110, 700],
            [[200], [210]],
            [[0], [1]],
            [0, 0],
        ),
        step_index=1,
        parent_candidate_id=root_candidates[1],
        rng_seed=0,
    )
    session.record_selection(1, (child_candidates[0],))
    session.finish_success()

    events = read_policy_events(event_path)
    trace = build_policy_request_from_events(
        events,
        raw_event_sha256=policy_event_sha256(event_path),
    )
    assert len(events) == 6
    assert trace.generated_tokens == 5
    assert trace.materialized_output_tokens == 1
    assert trace.steps[1].generation_calls[0].reused_kv_tokens == 2
    assert b"must never enter raw events" not in event_path.read_bytes()


def test_compute_optimal_session_persists_oom_without_building_trace(
    tmp_path,
) -> None:
    event_path = tmp_path / "oom.events.jsonl"
    session = ComputeOptimalTtsSession(_started(), event_path)

    session.finish_exception(RuntimeError("CUDA out of memory"))

    events = read_policy_events(event_path)
    assert events[-1].status is CollectionStatus.OOM
    assert events[-1].error == "RuntimeError: CUDA out of memory"
    with pytest.raises(WorkloadTraceError, match="cannot build"):
        build_policy_request_from_events(
            events,
            raw_event_sha256=policy_event_sha256(event_path),
        )


def test_compute_optimal_factory_freezes_config_and_dataset_identity(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "collector.json"
    _write_collector_config(config_path)
    monkeypatch.setenv("ORCHES_POLICY_COLLECTOR_CONFIG", str(config_path))
    tree_config = SimpleNamespace(beam_size=1, tree_max_width=4)
    generation_config = SimpleNamespace(
        seed=7,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=256,
    )

    session = create_compute_optimal_tts_session(
        {
            "file_path": "/run/question_23/record_2.jsonl",
            "level": "Level 4",
        },
        tree_config,
        generation_config,
    )

    assert session is not None
    assert session.started.request_id == "MATH-500/item-23/sample-2"
    assert session.started.dataset_id == "23"
    assert session.started.search_width == 4
    assert session.started.seed == 7
    assert session.event_path == tmp_path / "events/item-23-sample-2.events.jsonl"
    parsed = ComputeOptimalTtsCollectorConfig.read(config_path)
    assert parsed.event_dir == tmp_path / "events"


def test_compute_optimal_factory_rejects_upstream_multi_beam_width(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "collector.json"
    _write_collector_config(config_path)
    monkeypatch.setenv("ORCHES_POLICY_COLLECTOR_CONFIG", str(config_path))

    with pytest.raises(WorkloadTraceError, match="beam_size=1"):
        create_compute_optimal_tts_session(
            {"file_path": "/run/question_0/record_0.jsonl"},
            SimpleNamespace(beam_size=2, tree_max_width=8),
            SimpleNamespace(
                seed=0,
                temperature=0.7,
                top_p=1.0,
                max_new_tokens=64,
            ),
        )


def test_compute_optimal_example_config_matches_strict_schema() -> None:
    config = ComputeOptimalTtsCollectorConfig.read(
        PROJECT_ROOT
        / "configs/workloads/compute_optimal_tts_policy.example.json"
    )

    assert config.dataset == "MATH-500"
    assert config.event_dir == (
        PROJECT_ROOT / "artifacts/raw/compute-optimal-tts-pilot"
    )
    assert config.policy_model.name == "Qwen/Qwen2.5-1.5B-Instruct"


def test_failed_event_stream_is_retained_but_not_trace_buildable(tmp_path) -> None:
    events = [
        _started(),
        PolicyRequestFinishedEvent(
            event_index=1,
            request_id="math-0",
            status=CollectionStatus.OOM,
            error="worker reported CUDA OOM",
        ),
    ]
    path = tmp_path / "oom.events.jsonl"
    write_policy_events(path, events)
    loaded = read_policy_events(path)

    assert loaded == events
    with pytest.raises(WorkloadTraceError, match="cannot build"):
        build_policy_request_from_events(
            loaded,
            raw_event_sha256=policy_event_sha256(path),
        )


def test_validate_policy_events_cli_reports_buildability(tmp_path, capsys) -> None:
    path = tmp_path / "events.jsonl"
    write_policy_events(path, _successful_events())

    status = main(["validate-policy-events", str(path), "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert status == 0
    assert summary["request_id"] == "math-0"
    assert summary["status"] == "success"
    assert summary["trace_buildable"] is True
    assert summary["paper_eligible"] is False
    assert summary["generation_calls"] == 2


def test_compute_optimal_patch_applies_to_frozen_upstream() -> None:
    upstream_git = PROJECT_ROOT / "third_party/compute-optimal-tts/.git"
    if not upstream_git.exists():
        pytest.skip("run scripts/bootstrap.sh before checking the upstream patch")

    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts/compute_optimal_tts_patch.sh"),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
