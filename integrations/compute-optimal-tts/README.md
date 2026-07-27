# Compute-Optimal-TTS Policy Collector Integration

## Scope

This integration targets the frozen upstream commit:

```text
0ee2578af1f8d6cac445c9c4c72780528bb94556
```

The first patch exposes exact policy-model evidence that upstream currently
discards. It does not change PRM scoring and does not yet attach search-tree
selection hooks. Consequently, applying this patch alone does not produce a
complete event log or paper result.

## Apply And Audit

From the ORCHES repository root:

```bash
bash scripts/compute_optimal_tts_patch.sh --check
bash scripts/compute_optimal_tts_patch.sh
export ORCHES_PYTHONPATH=$PWD/src
```

The patched upstream launch scripts require `ORCHES_PYTHONPATH` so their vLLM
worker imports the dependency-light snapshot accumulator from this repository.
To restore the frozen checkout:

```bash
bash scripts/compute_optimal_tts_patch.sh --reverse
```

The apply script rejects any upstream revision other than the frozen commit.
The patch itself is verified by the ORCHES test suite with `git apply --check`.

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

## Remaining Hook

The next integration patch must connect candidate IDs to the accepted legal
actions and record the beam search's actual selected/pruned IDs. Until that hook
is implemented and a GPU run succeeds, the repository has no real policy trace.
