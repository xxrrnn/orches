from __future__ import annotations

import hashlib
import json

import pytest

from orches.cli import main
from orches.errors import WorkloadTraceError
from orches.workload import (
    KvBlockTraceV2,
    Modality,
    ModelRef,
    PolicyArtifactDigest,
    PolicyCandidateTrace,
    PolicyCollectionMode,
    PolicyGenerationCallTrace,
    PolicyRequestTrace,
    PolicyRuntimeEnvironment,
    PolicySelectionKind,
    PolicySelectionTrace,
    PolicyStepTrace,
    PolicyTraceProvenance,
    SamplingConfig,
    SourceKind,
    TokenTensorTrace,
    WorkloadScope,
    build_policy_manifest,
    parse_policy_request,
    read_policy_jsonl,
    read_policy_manifest,
    validate_policy_manifest,
    write_policy_jsonl,
    write_policy_manifest,
)


def _tokens(*token_ids: int) -> TokenTensorTrace:
    return TokenTensorTrace(
        token_ids=token_ids,
        attention_mask=(1,) * len(token_ids),
        model_token_count=len(token_ids),
    )


def _candidate(
    candidate_id: str,
    generated_token_ids: tuple[int, ...],
    ready_start: int,
    materialized_output_tokens: int,
) -> PolicyCandidateTrace:
    return PolicyCandidateTrace(
        candidate_id=candidate_id,
        generated_token_ids=generated_token_ids,
        token_ready_indices=tuple(
            range(ready_start, ready_start + len(generated_token_ids))
        ),
        materialized_output_tokens=materialized_output_tokens,
        terminal_kv_block_id=f"{candidate_id}-kv",
        finish_reason="stop",
    )


def _opaque_trace(decision_hash: str) -> PolicyRequestTrace:
    c0 = _candidate("c0", (100, 101), 0, 1)
    c1 = _candidate("c1", (110,), 2, 1)
    d0 = _candidate("d0", (200, 201), 3, 1)
    d1 = _candidate("d1", (210,), 5, 0)
    step0 = PolicyStepTrace(
        step_index=0,
        generation_calls=(
            PolicyGenerationCallTrace(
                call_id="generate-0-root",
                parent_candidate_id=None,
                input_tokens=_tokens(10, 11),
                reused_kv_tokens=2,
                rng_seed=0,
                rng_stream_id="request-0/step-0/root",
                candidates=(c0, c1),
            ),
        ),
        selection=PolicySelectionTrace(
            decision_id="select-0",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=("c1",),
            pruned_candidate_ids=("c0",),
            decision_artifact_sha256=decision_hash,
        ),
    )
    step1 = PolicyStepTrace(
        step_index=1,
        generation_calls=(
            PolicyGenerationCallTrace(
                call_id="generate-1-c1",
                parent_candidate_id="c1",
                input_tokens=_tokens(10, 11, 110, 700),
                reused_kv_tokens=3,
                rng_seed=1,
                rng_stream_id="request-0/step-1/c1",
                candidates=(d0, d1),
            ),
        ),
        selection=PolicySelectionTrace(
            decision_id="select-1",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=("d0",),
            pruned_candidate_ids=("d1",),
            decision_artifact_sha256=decision_hash,
        ),
    )
    return PolicyRequestTrace(
        policy_trace_schema_version=1,
        workload_scope=WorkloadScope.GENERATION_ONLY,
        paper_eligible=False,
        collection_mode=PolicyCollectionMode.OPAQUE_SELECTOR,
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
        provenance=PolicyTraceProvenance(
            source_kind=SourceKind.COLLECTED,
            pipeline="compute-optimal-tts",
            pipeline_revision="0ee2578",
            dataset_revision="math-500-revision",
            tokenizer=ModelRef("Qwen/Qwen2.5-1.5B-Instruct", "tokenizer-revision"),
            policy_model=ModelRef("Qwen/Qwen2.5-1.5B-Instruct", "model-revision"),
            inference_engine=ModelRef("vllm", "engine-revision"),
            collector=ModelRef("orches.compute-optimal-tts", "collector-revision"),
            selector=ModelRef("opaque-upstream-prm", "selector-revision"),
        ),
        root_input=_tokens(10, 11),
        kv_blocks=(
            KvBlockTraceV2("root-kv", None, None, 2),
            KvBlockTraceV2("c0-kv", "root-kv", "c0", 1),
            KvBlockTraceV2("c1-kv", "root-kv", "c1", 1),
            KvBlockTraceV2("d0-kv", "c1-kv", "d0", 2),
            KvBlockTraceV2("d1-kv", "c1-kv", "d1", 1),
        ),
        steps=(step0, step1),
    )


def _runtime() -> PolicyRuntimeEnvironment:
    return PolicyRuntimeEnvironment(
        python_version="3.10.12",
        platform="Linux-x86_64",
        accelerator="NVIDIA RTX 5070 Ti",
        accelerator_count=1,
        driver_version="test-driver",
        cuda_version="test-cuda",
        torch_version="test-torch",
    )


def test_policy_trace_round_trip_is_byte_stable(tmp_path) -> None:
    raw_hash = hashlib.sha256(b"opaque decisions\n").hexdigest()
    trace = _opaque_trace(raw_hash)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_policy_jsonl(first, [trace])
    loaded = read_policy_jsonl(first)
    write_policy_jsonl(second, loaded)

    assert loaded == [trace]
    assert first.read_bytes() == second.read_bytes()
    assert trace.generation_evaluation_eligible is True
    assert trace.paper_eligible is False
    assert trace.generated_tokens == 6
    assert trace.materialized_output_tokens == 3


def test_policy_trace_rejects_changed_parent_token_prefix() -> None:
    trace = _opaque_trace("a" * 64)
    raw = trace.to_dict()
    raw["steps"][1]["generation_calls"][0]["input_tokens"]["token_ids"][0] = 99

    with pytest.raises(WorkloadTraceError, match="preserve parent token IDs"):
        parse_policy_request(raw)


def test_policy_trace_rejects_incorrect_kv_extension() -> None:
    trace = _opaque_trace("a" * 64)
    raw = trace.to_dict()
    raw["kv_blocks"][3]["token_count"] = 1

    with pytest.raises(WorkloadTraceError, match="terminal KV extension"):
        parse_policy_request(raw)


def test_policy_trace_rejects_character_count_field() -> None:
    trace = _opaque_trace("a" * 64)
    raw = trace.to_dict()
    raw["steps"][0]["generation_calls"][0]["candidates"][0][
        "output_character_count"
    ] = 12

    with pytest.raises(WorkloadTraceError, match="unknown fields"):
        parse_policy_request(raw)


def test_opaque_selection_requires_raw_decision_digest() -> None:
    with pytest.raises(WorkloadTraceError, match="decision_artifact_sha256"):
        PolicySelectionTrace(
            decision_id="select",
            kind=PolicySelectionKind.OPAQUE,
            selected_candidate_ids=("c0",),
            pruned_candidate_ids=("c1",),
            decision_artifact_sha256=None,
        )


def test_policy_manifest_binds_trace_lock_runtime_and_raw_evidence(
    tmp_path,
) -> None:
    raw_path = tmp_path / "policy-events.jsonl"
    raw_path.write_bytes(b"opaque decisions\n")
    artifact = PolicyArtifactDigest.from_path(raw_path, "policy_events")
    trace_path = tmp_path / "policy-trace.jsonl"
    trace = _opaque_trace(artifact.sha256)
    write_policy_jsonl(trace_path, [trace])
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")

    manifest = build_policy_manifest(
        trace_path,
        [trace],
        uv_lock_path=lock_path,
        command=("uv", "run", "orches", "collect-policy-trace"),
        runtime=_runtime(),
        raw_artifacts=(artifact,),
        trace_file="policy-trace.jsonl",
    )
    manifest_path = tmp_path / "manifest.json"
    write_policy_manifest(manifest_path, manifest)
    loaded = read_policy_manifest(manifest_path)

    assert loaded == manifest
    assert loaded.generation_evaluation_eligible is True
    assert set(loaded.missing_evidence) == {
        "verifier_activity",
        "verifier_scores",
        "verifier_token_tensors",
    }
    validate_policy_manifest(loaded, trace_path)

    raw_path.write_bytes(b"changed decisions\n")
    with pytest.raises(WorkloadTraceError, match="raw artifact hash"):
        validate_policy_manifest(loaded, trace_path)


def test_policy_manifest_rejects_changed_trace_bytes(tmp_path) -> None:
    raw_path = tmp_path / "events.jsonl"
    raw_path.write_bytes(b"opaque decisions\n")
    artifact = PolicyArtifactDigest.from_path(raw_path, "policy_events")
    trace_path = tmp_path / "trace.jsonl"
    trace = _opaque_trace(artifact.sha256)
    write_policy_jsonl(trace_path, [trace])
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    manifest = build_policy_manifest(
        trace_path,
        [trace],
        uv_lock_path=lock_path,
        command=("collector",),
        runtime=_runtime(),
        raw_artifacts=(artifact,),
    )
    raw = trace.to_dict()
    raw["request_id"] = "changed"
    changed = parse_policy_request(raw)
    write_policy_jsonl(trace_path, [changed])

    with pytest.raises(WorkloadTraceError, match="trace_sha256"):
        validate_policy_manifest(manifest, trace_path)


def test_validate_policy_trace_cli_reports_explicit_scope(tmp_path, capsys) -> None:
    raw_path = tmp_path / "events.jsonl"
    raw_path.write_bytes(b"opaque decisions\n")
    artifact = PolicyArtifactDigest.from_path(raw_path, "policy_events")
    trace_path = tmp_path / "trace.jsonl"
    trace = _opaque_trace(artifact.sha256)
    write_policy_jsonl(trace_path, [trace])
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    manifest = build_policy_manifest(
        trace_path,
        [trace],
        uv_lock_path=lock_path,
        command=("collector",),
        runtime=_runtime(),
        raw_artifacts=(artifact,),
    )
    manifest_path = tmp_path / "manifest.json"
    write_policy_manifest(manifest_path, manifest)

    status = main(
        [
            "validate-policy-trace",
            str(trace_path),
            "--manifest",
            str(manifest_path),
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert status == 0
    assert summary["workload_scope"] == "generation_only"
    assert summary["paper_eligible"] is False
    assert summary["generation_evaluation_eligible"] is True
    assert summary["manifest_valid"] is True
    assert summary["candidates"] == 4
