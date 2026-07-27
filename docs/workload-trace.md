# Versioned TTC Workload Trace Contract

## Scope

The repository supports two complete-TTC schemas and one deliberately separate
policy-collection schema:

| Contract | Purpose | Paper evaluation |
|---|---|---|
| TTC v1 | Deterministic one-parent synthetic replay used by the existing simulator | Never eligible |
| TTC v2 | Exact-token, multi-beam, scalar/pairwise verifier, and logical-KV collection contract | Structurally eligible after real collection and manifest validation |
| Policy v1 | Exact policy generation calls without verifier internals | Generation-only comparison after opaque collection; never full-paper eligible |

No real model trace has been collected yet. Schema v2 defines and validates the
artifact that the compute-optimal-TTS and LLaVA-CoT collectors must produce; it
does not manufacture model observations.

## Generation-Only Policy Contract

`policy_schema.py` records policy-model observations before PRM enrichment. One
generation call contains one parent, exact submitted token tensor, reused KV,
RNG seed/stream, and ordered candidates. Candidates retain exact returned IDs,
logical token-ready order, actual output-KV materialization, terminal KV block,
and finish reason.

Three modes keep evidence strength explicit:

| Mode | Decision source | Valid use |
|---|---|---|
| `single_step` | None | Collector/token/KV development |
| `synthetic_selector` | Deterministic test selector | Multi-step simulator development |
| `opaque_selector` | Upstream selector, bound by raw artifact SHA-256 | Generation-only comparison |

Opaque means the collector preserves selected/pruned IDs but makes no claim
about PRM scores, tensors, timing, or energy. `policy_manifest.py` binds the
canonical JSONL to `uv.lock`, source/collector revisions, command, runtime GPU
stack, and raw artifacts. `validate-policy-trace --manifest` validates that
boundary. Enrichment later expands each call into TTC-v2 candidates and adds
real verifier calls; TTC-v2 validation is not relaxed in the meantime.

## Raw Policy Events

`collectors/policy_events.py` is the boundary between a model pipeline and the
generation-only policy contract. A per-request event file contains:

```text
request_started
generation          one per root/selected-parent model call
selection           one actual selected/pruned partition per step
request_finished    success, failed, or OOM
```

Generation outputs carry exact IDs, logical ready order, and actual
materialization; decoded text is rejected. Successful streams are converted
deterministically into Policy v1, while failed/OOM streams remain raw evidence
and cannot be presented as successful traces.

The compute-optimal-TTS worker integration consumes cumulative vLLM snapshots.
It verifies append-only IDs, waits for all requested output sequences, and
records the terminal sampled token as not yet KV-materialized. The first
external patch exposes these fields and the sampling seed. The second patch
binds candidate IDs to accepted legal actions, carries them through
`LanguageNode`, and records the actual global beam selection. A successful
request is rebuilt through the strict Policy-v1 validator before write; failed
and OOM streams remain raw-only evidence. The integration is executable for
the paper's `num_sequence=1` mode, but no GPU trace has been collected yet.

Both versions exclude prompt text, generated text, reference answers, and
correctness. Correctness belongs in a separate evaluation record keyed by
`request_id` and `dataset_id`.

## Schema v1

Version 1 records prompt/image token counts, one selected candidate per step,
one scalar small/final PRM score per candidate, and synthetic token timestamps.
It enforces:

```text
candidate.unique_kv_tokens = candidate.generated_tokens
next.shared_kv_tokens = current.shared_kv_tokens
                      + selected.unique_kv_tokens
next.candidate.parent_candidate_id = current.selected_candidate_id
```

These are synthetic development rules, not hardware facts. The constructor now
rejects `source_kind=collected` for v1. The optional v1-to-v2 migration generates
deterministic placeholder token IDs and remains marked `source_kind=synthetic`;
it cannot become evaluation evidence.

## Schema v2 Request Fields

| Field | Meaning |
|---|---|
| `trace_schema_version` | Strict version; `2` |
| request/dataset/modality fields | Public workload identity and paper grouping |
| `image_tokens` | Actual model-visible visual token/feature count |
| `search_width` | Candidate count generated per retained parent |
| `beam_size` | Maximum candidate count retained after one step |
| `sampling` | Temperature, top-p, and maximum new tokens |
| `provenance` | Pipeline, dataset, tokenizer, policy, verifier, and inference-engine revisions |
| `root_input` | Exact submitted token IDs/mask and model-visible token count |
| `kv_blocks` | Request-global append-only logical KV lineage |
| `steps` | Ordered generation, verification, selection, and pruning events |

One JSONL file must contain a single schema version and request-unique IDs.
Serialization remains sorted, compact, ASCII, newline-terminated, and
byte-deterministic.

## Exact Token Accounting

`TokenTensorTrace` stores:

```text
token_ids             exact IDs submitted to the model
attention_mask        exact 0/1 validity mask
valid_token_count     sum(attention_mask), derived
model_token_count     sequence positions processed by the model
```

For text, `model_token_count` must equal the valid token-ID count. For vision,
it may be larger because image features expand into model-visible sequence
positions; `image_tokens` is recorded independently from the processor/model
shape.

Each candidate stores exact `generated_token_ids`, a request-global contiguous
`token_ready_indices` order, and `materialized_output_tokens`. The latter is
separate because the final sampled token may not have entered another forward
iteration and therefore may not yet own KV.

Character count, whitespace splitting, decoded-text length, and re-tokenized
normalized output are not schema fields and cannot drive hardware accounting.
Unknown fields are rejected.

## Logical KV Lineage

`KvBlockTraceV2` is an append-only logical extension:

```text
block_id
parent_block_id
owner_candidate_id
token_count
```

The request has exactly one ownerless root block. A candidate records its
`reused_kv_tokens` and `terminal_kv_block_id`. Validation requires:

```text
parent lineage tokens = candidate.reused_kv_tokens
terminal lineage tokens = input.model_token_count
                        + materialized_output_tokens
extension tokens = terminal lineage tokens - parent lineage tokens
```

If the extension is zero, the candidate must retain the parent terminal block.
Otherwise its terminal block must be owned by that candidate, point to the
parent terminal block, and contain exactly the extension token count. Orphan,
extra, cyclic, and multiply interpreted blocks are rejected.

## Width, Beam, and Selection

Width and beam are independent. Step zero generates `search_width` candidates
from the request root. Every later retained parent must generate exactly
`search_width` children. A step may retain up to `beam_size` candidates, and all
of them become valid parents of the next step.

Every ordered selection event records considered, selected, pruned, and
still-live candidate IDs plus the complete retained KV-block set. Validation
reconstructs the union of every live candidate's ancestry and requires an exact
match. Thus a width-4/beam-3 step can retain three non-positional winners, while
freeing only the actual losing candidate's KV.

## Verifier Calls

Schema v2 represents both paper workload families:

- `scalar_prm`: one exact input tensor and one finite score per candidate;
  a model invocation may batch several candidates or be recorded as one call
  per candidate, and separate stages can record layer-10 and final PRM outputs;
- `pairwise_judge`: exactly two candidates, exact judge input/generated token
  IDs in one combined judge tensor, no invented scalar score, and one winner.

`model_input_tokens` is derived as the sum of all `input_tensors`. This avoids
the invalid assumption that a batched scalar PRM call has one shared token
sequence: each candidate is processed with its own exact submitted tensor.

A top-k event must reference scalar PRM calls covering every considered
candidate. A pairwise event must reference exactly one pairwise call and follow
its recorded winner. Pairwise tournament events therefore preserve the precise
loser-prune order required for LLaVA-CoT KV lifetime replay.

## Schema-v2 Replay

`OrchesV2RequestReplayer` consumes the validated tree without reducing it to
v1's single winner. One generation phase is created for each output-token
round. For every active candidate, attention length is derived from its exact
model input, the intersection of active parent KV lineages, and the number of
earlier generated tokens. Candidates that finish early leave later rounds.

Physical KV allocation uses `materialized_output_tokens`. Selection-driving
verifier duration uses the sum of exact input-tensor model positions plus any
generated judge-decision tokens. Every ordered selection event then releases
all physical blocks outside `retained_kv_block_ids`; a later pairwise judge
depends on the preceding selection event.

Logical `token_ready_indices` are mapped proportionally into the simulated T1
generation interval. Absolute collection latency is not replayed. T2A is
enabled only when `beam_size=1`; retaining several beams has no unique predicted
branch under the paper's current predictor definition. T1, T2B, verifier
execution, and T3 remain active for multi-beam traces.

The current T2B adapter assigns incremental work to policy tokens becoming
ready. Exact cross-tokenizer readiness of PRM input positions is a collector
gap tracked by `A-T2-003`, not a reason to substitute character counts or RTX
5070 Ti timestamps.

## Validation Coverage

Focused tests cover byte-stable v2 round trips, width 4 with non-first winners,
two retained parents expanded in the next step, beam size 3, generated/KV count
differences, MathVista-style pairwise elimination, exact parent token prefixes,
global token-ready dependencies, retained-KV ancestry, v1 collected rejection,
deterministic synthetic migration, per-token replay, ordered physical pruning,
candidate-local PRM calls, and phase-complete activity accounting.

## Paper Mapping

Exact candidate and verifier token tensors provide Sec. 2.2 and Fig. 3 workload
sizes. Search width, retained beams, and KV lineages provide Sec. 3.1 variable
parallelism and Sec. 3.2 branch dependencies. Ordered readiness and layer-10
calls feed Technique 2; ordered pruning and KV block lifetime feed Technique 3.
Dataset/model provenance carries Sec. 5.1 evaluation identity.
