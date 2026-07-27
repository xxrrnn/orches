# TTC Workload Trace Contract

## Scope

The version 1 JSONL trace is a development contract for deterministic synthetic
requests. It can replay the repository's current one-parent, one-winner scalar
PRM examples without invoking a language model again. It is not sufficient for
paper workload collection because it does not preserve exact token IDs, actual
KV materialization, multiple selected beams, or pairwise verifier calls. One
JSON object represents one request.

The trace deliberately excludes prompt text, generated text, reference answers,
and correctness. Correctness belongs in a separate evaluation record keyed by
`request_id` and `dataset_id`. This keeps architecture timing input independent
from answer grading and avoids storing unnecessary content.

## Request Fields

| Field | Meaning |
|---|---|
| `trace_schema_version` | Strict schema version; currently `1` |
| `request_id` | Trace-file-unique replay identifier |
| `dataset`, `dataset_id`, `difficulty` | Public workload identity and paper grouping |
| `modality` | `text` or `vision` |
| `question_length_bucket` | Vision-only `short`, `medium`, or `long` grouping |
| `prompt_tokens`, `image_tokens` | Initial shared context sizes |
| `search_width` | Candidate count generated at every step |
| `seed`, `sampling` | Inputs affecting generation control flow |
| `provenance` | Pipeline, dataset, tokenizer, policy, and PRM revisions |
| `steps` | Ordered generation/verification decisions |

Version 1 traces must use `source_kind=synthetic` and are never eligible for
paper evaluation. Real compute-optimal-TTS and LLaVA-CoT collection requires
schema v2 as specified in `trace-collection-plan.md`.

## Step and Candidate Fields

Each step records its zero-based index, common `shared_kv_tokens`, all generated
candidates, the selected candidate, and every pruned candidate ID. A candidate
records a request-unique ID, its parent ID, generated/unique-KV token count,
relative per-token completion timestamps in microseconds, and both small- and
large-PRM scores.

Version 1 makes the following synthetic-only simplification:

```text
candidate.unique_kv_tokens = candidate.generated_tokens
```

The next step is derived only from the selected branch:

```text
next.shared_kv_tokens = current.shared_kv_tokens
                      + selected.unique_kv_tokens

next.candidate.parent_candidate_id = current.selected_candidate_id
```

These equations are v1 validation rules, not hardware facts. A real collector
must count the exact `input_ids`/masks submitted to the model and the generated
token IDs returned by the engine. It must record which sequence positions
actually materialized KV and retain/free those KV blocks according to the
ordered top-k or pairwise selection events. Character length, whitespace token
counts, re-tokenized output text, and a fixed single-winner assumption are not
valid substitutes.

For `beam_size > 1`, schema v2 keeps every selected candidate live, records its
parent and KV-block lineage, and maps each next expansion to the actual retained
parent or parents. Shared KV is computed from common token ancestry; it is not
assumed to be `prompt_tokens + sum(generated_tokens)`.

## Validation Invariants

The parser rejects unknown or missing fields, unsupported versions, duplicate
IDs, invalid modality metadata, non-contiguous steps, candidate counts that do
not equal `search_width`, missing selected branches, incomplete prune sets,
invalid parent links, broken shared-KV growth, non-finite PRM scores, and token
timelines that are the wrong length or are not monotonic.

JSONL serialization uses sorted keys, compact separators, ASCII output, and one
newline per request. Given the same schema objects, serialized bytes and the
reported SHA-256 are deterministic.

## Paper Mapping

The v1 request/step/candidate structure is a synthetic approximation of Sec.
2.2 and Fig. 3's TTC generation and verification tree. Schema v2 will provide
the exact token, verifier, selection, and KV-lineage inputs required by Sec. 3.1
variable parallelism, Sec. 3.2 branch dependency, Technique 1 online
compensation, Technique 2 prediction, Technique 3 fragmentation, and Sec. 5.1
paper evaluation.
