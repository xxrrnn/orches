# Compute-Optimal-TTS Policy Collector Integration

## Scope

This integration targets the frozen upstream commit:

```text
0ee2578af1f8d6cac445c9c4c72780528bb94556
```

The integration is split into two ordered patches:

1. `0001-expose-exact-policy-tokens.patch` carries the seed into vLLM and
   exposes exact prompt/output token IDs, logical ready order, and output-KV
   materialization.
2. `0002-record-policy-control-flow.patch` binds worker candidates to accepted
   legal actions, carries IDs through search nodes, records the actual global
   beam decision, and closes every request with success, failed, or OOM status.

Neither patch changes PRM scoring. Together they produce a complete raw
generation event stream; they do not yet collect PRM token tensors or scores,
so the output is a generation-only artifact rather than a complete paper TTC
trace.

## Apply And Audit

From the ORCHES repository root:

```bash
bash scripts/compute_optimal_tts_patch.sh --check
bash scripts/compute_optimal_tts_patch.sh
export ORCHES_PYTHONPATH=$PWD/src
export ORCHES_POLICY_COLLECTOR_CONFIG=$PWD/configs/workloads/compute_optimal_tts_policy.json
export TRACE_TEXT_PROJECT=/absolute/path/to/frozen-trace-text-uv-project
export ORCHES_PYTHON_EXECUTABLE="$(uv run --project "$TRACE_TEXT_PROJECT" \
  --frozen python -c 'import sys; print(sys.executable)')"
```

The patched upstream launch scripts require `ORCHES_PYTHONPATH` so their vLLM
worker imports the dependency-light snapshot accumulator from this repository.
They also require the absolute interpreter obtained from a frozen UV project;
they never activate Conda, including inside tmux worker windows.
To restore the frozen checkout:

```bash
bash scripts/compute_optimal_tts_patch.sh --reverse
```

Create `compute_optimal_tts_policy.json` from the adjacent `.example.json` and
replace every `REPLACE_WITH_*` value with an immutable model, tokenizer,
dataset, PRM, or ORCHES commit. Relative `event_dir` paths resolve from the
collector config directory. Do not run a paper-facing collection with symbolic
branches such as `main` or `latest`.

The apply script rejects any upstream revision other than the frozen commit and
applies both patches atomically in order. The ORCHES test suite verifies the
combined series with `git apply --check`.

## Worker Evidence

For each vLLM request, the patched worker returns:

```text
request_id
prompt_token_ids
output_token_ids[]
token_ready_indices[]
materialized_output_tokens[]
finish_reason[]
```

`VllmSnapshotAccumulator` consumes cumulative `RequestOutput` updates and emits
only newly observed token order, so storage is linear rather than quadratic in
generated length. It waits for all requested `n` outputs before finalizing.

When an output first reports a terminal finish reason, its last sampled token
has not entered another decode forward pass. Therefore:

```text
materialized_output_tokens = max(0, generated_token_ids - 1)
```

This value is derived at the engine transition, not from decoded text. At the
next reasoning step, the exact submitted prompt captures whether that final
token and any step-template tokens are subsequently materialized.

The patch also carries the evaluation seed through `LMCallingConfig`, the HTTP
request, and vLLM `SamplingParams`. The worker response may still contain text
for upstream compatibility; `ComputeOptimalTtsAdapter` deliberately copies
only the exact fields above into ORCHES raw events.

## Search Control Flow

The control-flow patch records worker output before upstream legal-action
filters. Each accepted action keeps its generated candidate ID. The ID then
travels through `LanguageNode`; immediately after the global beam heap is
reduced, the collector records the actual selected IDs and treats every other
generated candidate as pruned. This preserves candidates discarded because of
duplicate decoded actions or non-`stop` finish reasons without deriving any
hardware quantity from their text.

The current executable integration requires:

```text
method=beam_search
num_sequence=1
tree_max_width=search_width
```

This is the ORCHES paper's text configuration. Upstream changes each parent's
generation count to `tree_max_width / beam_size` when `num_sequence > 1`, which
does not satisfy the current trace contract of `search_width` children per
retained parent. The collector rejects that mode instead of mislabeling it.

Every successful request is rebuilt through the strict policy schema before
its event file is written. Parent token prefixes, width, selected/pruned
partition, KV lineage, and token-ready order must all validate. Exceptions and
CUDA OOMs write terminal raw event files but cannot be converted to successful
traces.

## Pilot Procedure

After filling the collector config and launching the patched vLLM/PRM services,
capture the host identity using the same frozen UV interpreter:

```bash
"${ORCHES_PYTHON_EXECUTABLE}" -m orches.cli probe-trace-host \
  --output artifacts/manifests/compute-optimal-tts-pilot.host.json --json
```

`collection_ready` must be true. For a BF16 cell, `bf16_supported` must also be
true. Then run upstream with the paper-compatible settings:

```bash
cd third_party/compute-optimal-tts/src
bash scripts/run.sh \
  --LM Qwen/Qwen2.5-1.5B-Instruct \
  --RM YOUR_PRM_MODEL \
  --task MATH \
  --method beam_search \
  --width 2 \
  --num_seq 1 \
  --num_q 3 \
  --bs 3
```

The upstream service command depends on the target GPU count; use the matching
frozen `scripts/serve_gpu*.sh` after applying the patches. Validate each emitted
request from the ORCHES root:

```bash
uv run --frozen orches validate-policy-events \
  artifacts/raw/compute-optimal-tts-pilot/item-0-sample-0.events.jsonl --json
```

Repeat the same request set with the same immutable revisions and seed. Exact
event hashes should be compared before local and server traces are pooled.
After every request has a terminal event, fill
`configs/workloads/compute_optimal_tts_build.json` from its example and run:

```bash
uv run --frozen orches build-policy-collection \
  configs/workloads/compute_optimal_tts_build.json --json
```

This command writes the generation-only trace, reproducibility manifest, and
per-request validation report. It hashes successful, failed, and OOM events,
the host probe, build config, and exact trace UV lock. A cell with no successful
request writes a failure report but no empty trace or manifest.

## Remaining Work

No GPU model run has been performed in this workspace, so the repository still
has no real policy trace. Host probing and event-set conversion are implemented;
the next checkpoint must freeze the 5070-compatible trace UV lock, run the
width-2 pilot, and record deterministic repeat hashes. PRM tensor/score
instrumentation remains a separate later milestone and is required before full
paper latency or energy claims.
