# TTC Workload Trace Contract

## Scope

The version 1 JSONL trace records the control flow and tensor-relevant sizes of
one completed TTC reasoning request. It is sufficient to replay generation,
verification, branch selection, pruning, and KV growth without invoking a
language model again. One JSON object represents one request.

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

Collected traces must identify the frozen compute-optimal-TTS or LLaVA-o1
revision. Synthetic traces use `source_kind=synthetic` and are never eligible
for paper evaluation.

## Step and Candidate Fields

Each step records its zero-based index, common `shared_kv_tokens`, all generated
candidates, the selected candidate, and every pruned candidate ID. A candidate
records a request-unique ID, its parent ID, generated/unique-KV token count,
relative per-token completion timestamps in microseconds, and both small- and
large-PRM scores.

Version 1 treats each candidate's newly generated tokens as its unique KV
fragment:

```text
candidate.unique_kv_tokens = candidate.generated_tokens
```

The next step is derived only from the selected branch:

```text
next.shared_kv_tokens = current.shared_kv_tokens
                      + selected.unique_kv_tokens

next.candidate.parent_candidate_id = current.selected_candidate_id
```

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

The request/step/candidate structure represents Sec. 2.2 and Fig. 3's TTC
generation and verification tree. Search width and branch-dependent KV growth
provide the inputs for Sec. 3.1 variable parallelism, Sec. 3.2 branch
dependency, Technique 1 online compensation, Technique 2 prediction, and
Technique 3 fragmentation. Dataset/model fields carry Sec. 5.1 evaluation
identity.
