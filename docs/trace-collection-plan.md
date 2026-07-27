# Real Workload Trace Collection Plan

## Purpose

This document separates three inputs that were previously conflated:

1. **workload trace**: model-dependent reasoning control flow, candidate token
   counts, verifier decisions, and KV growth;
2. **hardware timing profile**: analytical or calibrated GPU/PIM rates used to
   replay that control flow;
3. **validation measurements**: optional measurements used to quantify the
   error of the analytical hardware model.

A workload trace collected on an RTX 5070 Ti can be replayed against the AGX
Orin analytical model because absolute 5070 Ti latency is not used as Orin
latency. Model checkpoint, tokenizer, precision, seed, sampling, and search
control flow must still match the target experiment.

## Why AGX Orin Measurement Is Not a Mandatory Trace Input

ORCHES Sec. 5.1 states that its simulator extends AttAcc, that GPU and PIM
modeling were validated by the prior work, and that the same unit latency and
unit energy are retained. The public baselines follow this approach:

- AttAcc `src/devices.py` computes GPU layer time as the maximum of compute and
  memory time. Compute uses peak FLOP/s, a fixed 0.8 compute-utilization factor,
  and thread-block occupancy. Memory uses peak bandwidth, a fixed 0.85
  bandwidth-utilization factor, and modeled traffic. ACT/NORM and NVLink are
  exceptional A100-derived fits; the target workload is not profiled.
- Duplex `src/hardware/linear_impl.cpp` computes GPU operation time as
  `max(FLOPs / peak_FLOP_per_s, bytes / bandwidth)`. Its attention
  implementations use the same analytical rule and its optional Ramulator path
  replaces ideal memory time with simulated cycles. It does not require a
  workload run on the modeled GPU.

Neither public baseline requires a measured latency trace for every model and
GPU. Therefore ORCHES reproduction will support two explicit evidence paths:

| Profile | GPU rates | Valid claim |
|---|---|---|
| `paper_method` | AttAcc-style analytical model, paper bandwidth, inherited utilization factors, explicit Orin peak/assumed fields | Reproduction using the method described in Sec. 5.1 |
| `orin_calibrated` | Same operator model with rates fitted from AGX Orin microbenchmarks | Independent validation and calibrated reproduction |

`paper_method` is sufficient to run the paper reproduction. It is not evidence
that every modeled Orin kernel has been measured. `orin_calibrated` is stronger
validation, not a prerequisite for collecting model control flow.

An RTX 5070 Ti measurement must never be substituted for AGX Orin compute,
bandwidth, launch, energy, or thermal parameters.

## Trace Semantics

The replay trace is hardware-independent and contains:

- dataset/request identity and public difficulty metadata;
- exact model, tokenizer, dataset, and collector revisions;
- seed, temperature, top-p, maximum tokens, and search width;
- prompt/model-input and image token counts taken from model tensors, never
  decoded character counts;
- every candidate generated at every reasoning step;
- candidate token IDs or a stable hash, generated-token count, and parent;
- the verifier inputs, outputs, decisions, and selected branch;
- per-candidate KV lineage, actual retained KV after every selection, and
  pruned candidate IDs;
- logical token-ready order used by Technique 2B.

Raw text and images stay in access-controlled raw artifacts. The simulator
JSONL stores public IDs, counts, scores/decisions, and content hashes.

### Token and KV Counting Invariants

Character count is never a hardware quantity. The collector must not use
`len(decoded_text)`, whitespace splitting, or re-tokenization of normalized
output to derive generation, verifier, or KV sizes. It records counts from the
exact tensors used by the model:

```text
model_input_tokens       = valid entries in the submitted input_ids/attention mask
generated_token_ids      = token IDs returned beyond that exact input boundary
verifier_input_tokens    = valid token IDs submitted to the PRM/judge
kv_materialized_tokens   = sequence positions for which the model produced K/V
next_prefix_tokens       = exact model input length at the next expansion
```

Special tokens, chat-template tokens, PRM step tags, visual placeholders, and
stage markers count whenever they are present in the submitted model inputs.
Image token/feature length is obtained from the multimodal processor/model
input shapes and masks; image dimensions or prompt characters are not proxies.

`generated_token_ids` and `kv_materialized_tokens` are separate because the
final sampled token may not have been fed through another decode iteration.
The collector resolves this using the engine output/KV metadata and verifies it
against the exact next-step input. It does not assume that they are always
equal.

For width greater than two, KV retention follows the actual selection events:

1. each candidate owns a KV extension whose parent is explicit;
2. each scalar-PRM top-k decision or pairwise judge decision records the
   retained and pruned candidate IDs in order;
3. a pruned candidate's unique KV becomes dead at that decision point;
4. every selected candidate remains live if beam size is greater than one;
5. the next expansion references its actual selected parent or parents;
6. shared KV is derived from token-ID ancestry/longest common prefix, not from
   `prompt_tokens + sum(generated_tokens)`.

The paper's main text setup uses `num_sequence=1`, so one branch normally
survives each text step. Schema v2 uses `selected_candidate_ids` and
explicit parent/KV-block references so width and beam size are not conflated.
For LLaVA-CoT width 4, the trace retains the complete pairwise elimination
order; memory replay can free each losing candidate at the recorded judge
decision instead of pretending all candidates were pruned simultaneously.

Absolute token wall time is diagnostic only. Replay scales logical token-ready
positions to the simulated generation event, so a 5070 Ti does not leak its
latency into the Orin result. Schema v2 records a request-global contiguous
`token_ready_indices` order. Schema v1 retains synthetic monotonic
`token_timestamps_us` values only for compatibility.

## Implemented Schema and Remaining Collector Gap

Schema v2 is implemented in `src/orches/workload/schema_v2.py` with strict JSON
parsing in `io_v2.py`. It closes three structural gaps in synthetic schema v1:

1. Technique 2's small PRM is the first 10 layers of the original PRM. A real
   text trace must contain both the layer-10 early-exit score and the final
   score for each candidate.
2. LLaVA-CoT uses pairwise judging, not one scalar PRM call over all candidates.
   A vision trace must preserve each tournament comparison, judge input/output
   token counts, winner, and final selected candidate.
3. KV growth and retention must come from exact token-ID lineage and actual
   top-k/tournament choices. Schema v1 assumes one selected candidate and
   `unique_kv_tokens == generated_tokens`, which is too restrictive for general
   beam selection and may miss template/special-token effects.

The implementation provides `scalar_prm` and `pairwise_judge` calls, ordered
selection events, selected-candidate lists, per-candidate scalar input tensors,
exact input/generated token IDs,
separate output-KV materialization, and append-only KV lineage blocks. It does
not invent a numeric vision score when the source pipeline only produced a
pairwise choice. Tests cover width 4, beam size 3, two retained parents, and
pairwise elimination order.

Schema-v2 replay is also implemented: generation is expanded by actual output
token round, parent-lineage intersections drive shared/unique attention, exact
materialized positions drive physical KV bytes, and verifier/selection events
drive ordered pruning. The remaining gap is collection, not representation or
functional replay: no compute-optimal-TTS or LLaVA-CoT run has emitted a real
schema-v2 trace yet. Synthetic v1 migration is explicitly ineligible because
v1 never contained real token IDs.

## Text Pipeline Audit

Frozen source:

```text
third_party/compute-optimal-tts
commit 0ee2578af1f8d6cac445c9c4c72780528bb94556
```

The paper-compatible upstream configuration is:

```text
method          beam_search
num_sequence    1
tree width      2 through 8
temperature     0.7
top_p           1.0
max_new_tokens  2048
tree_max_depth  40
step prompt     enabled
dataset         MATH-500
```

`num_sequence=1` is important: every step expands `width` candidates under the
single selected parent, which matches the current ORCHES request schema. The
upstream code already exposes candidate text, token count, final PRM history,
selected path, and MATH Level 1-5 metadata. Its saved result does not retain the
complete rejected tree or token-ready events.

The collector patch will be maintained outside `third_party/` and applied from
the ORCHES repository. It will add events at these boundaries:

| Boundary | Collected data |
|---|---|
| `CoTEnv.update_legal_actions` | exact input token IDs/count, parent, all candidate IDs, generated token IDs/counts, finish reasons, generation order |
| vLLM `generate_stream` | per-candidate logical token-ready progress |
| `SearchTree._expand_leaf_node` | full candidate PRM histories and final scores |
| PRM forward path | layer-10 early-exit score and final-layer score |
| beam selection | ordered selected/pruned IDs, retained KV blocks, and next-parent mapping |
| evaluator | dataset ID, Level 1-5, answer-correctness side record |

The first paper-oriented checkpoint combination is:

```text
policy: Qwen/Qwen2.5-1.5B-Instruct
PRM:    public 1.5B PRM proxy recorded in the manifest
width:  2 and 4
levels: one request from Levels 1, 3, and 5
seed:   0
dtype:  BF16 or FP16, recorded exactly
```

The exact `*-PRM-Tuned` checkpoints named by ORCHES are not publicly identified.
Public Qwen/RLHFlow/Skywork checkpoints remain proxies under `A-MODEL-001`; the
trace cannot be described as checkpoint-exact until those revisions are known.

## Vision Pipeline Audit

The frozen `third_party/LLaVA-o1` repository is only a November 2024 redirect
to `PKU-YuanGroup/LLaVA-CoT`; it contains no executable collector. The closest
public paper-era implementation is the LLaVA-CoT inference commit:

```text
8983878f7b7fa3fe72232fa656be0b2ac41c6338
2025-01-22, "inference"
```

That implementation uses `Xkev/Llama-3.2V-11B-cot`, BF16, temperature 0.6,
top-p 0.9, and maximum 2048 new tokens. Stage beam search performs four ordered
stages:

```text
SUMMARY -> CAPTION -> REASONING -> CONCLUSION
```

At each stage it generates candidates and runs a random pairwise tournament in
which the same 11B model acts as judge. Upstream hard-codes 10 candidates; the
ORCHES paper explicitly evaluates widths 2 and 4, so the adapter must expose
the width while retaining all other generation/judge behavior. The random
tournament pairing order and every judge decision are part of the trace.

For MathVista, image token count is taken from the processor output, not
estimated from image dimensions or decoded text. Submitted `input_ids`, masks,
cross-attention/vision shapes, and exact generated token IDs are recorded before
sanitization. Short/medium/long buckets will be deterministic
tertiles of tokenized question length over the frozen evaluation split because
the paper does not publish thresholds. The exact thresholds and tie rule are
stored in the dataset manifest under `A-VISION-002`.

The vision collector records, per stage:

- shared multimodal prefix token count;
- exact submitted model-input tokens and all generated candidate token
  IDs/counts;
- candidate parent and stage end marker;
- every pairwise judge prompt length and generated length;
- tournament pairing order, winner, and loser-prune point;
- final selected candidate, all pruned candidates, and retained KV lineage.

## RTX 5070 Ti Pilot

The current execution sandbox cannot access NVML, so the card and memory size
must be confirmed on the actual host with `nvidia-smi`. If it is the common
16-GiB configuration, use it as follows:

| Workload | 5070 Ti use | Paper eligibility |
|---|---|---|
| 1B/1.5B policy + 1.5B PRM, width 2/4 | Collector development and small real text traces in BF16/FP16 | Eligible after exact manifest and schema checks |
| 3B policy + 1.5B PRM | Attempt after the smallest pair; reduce concurrency, not precision | Eligible only if unquantized and no OOM |
| 1B-3B policy + 7B/8B PRM | Likely too tight with two resident engines | Move to a 40/48/80-GiB server GPU |
| LLaVA-CoT 11B BF16 | Weight plus runtime memory exceeds a practical 16-GiB budget | Run on server; 4-bit is smoke-only |

Quantized or CPU-offloaded runs may test collector correctness, but their branch
decisions are not mixed into the main FP16/BF16 trace set.

Pilot sequence:

1. probe driver, CUDA capability, free memory, BF16 support, and installed
   compiler;
2. collect three MATH requests at width 2 with the smallest model pair;
3. repeat exactly and require identical sanitized trace hashes;
4. expand to width 4 and one request from each MATH difficulty level;
5. compare trace invariants and answer side records, ignoring 5070 wall time;
6. run a quantized LLaVA-CoT one-image smoke test only if the server is not yet
   available.

The upstream text environment pins vLLM 0.6.4.post1 and torch 2.5.1. Those
versions predate this GPU generation and must not silently define the 5070 Ti
environment. We will keep two uv locks:

- `trace-text-paper`: frozen upstream stack for A100/H100 server reproduction;
- `trace-text-local`: a 5070-compatible stack selected by a host probe, with
  its changed engine versions recorded in provenance.

The model, tokenizer, seed, and sampling contract remains common. Engine-version
differences require a deterministic hash comparison before local and server
traces may be pooled.

## Server Collection Matrix

Text main matrix:

```text
datasets: MATH-500, then LiveCodeBench
policies: Llama3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B
PRMs:     Qwen2.5-1.5B, Qwen2.5-7B, Llama3.1-8B proxies
widths:   2, 3, 4, 5, 6, 7, 8
seed:     0 main; seeds 1 and 2 as sampling sensitivity
dtype:    FP16/BF16, fixed per complete comparison set
```

Vision main matrix:

```text
dataset:  MathVista frozen evaluation split
model:    Xkev/Llama-3.2V-11B-cot
mode:     stage beam
widths:   2, 4
buckets:  short, medium, long by frozen token-count tertiles
seed:     0 main; additional seeds reported separately
dtype:    BF16
```

Use one 48/80-GiB GPU per text policy/PRM pair where possible. Server parallelism
is across independent experiment cells; requests within one cell retain the
pipeline's ordering and fixed seed behavior. Start with 10 requests, validate,
then 50 requests, then the full split. Failed/OOM requests remain manifest rows.

## Reproducible Environment Layout

The implementation will create independent uv projects because the text and
vision stacks have incompatible historical transformer/runtime requirements:

```text
environments/
  trace-text-paper/{pyproject.toml,uv.lock}
  trace-text-local/{pyproject.toml,uv.lock}
  trace-vision/{pyproject.toml,uv.lock}
```

No collector will run `pip install` at runtime. CUDA wheel indexes, Python
version, platform markers, and build options are part of each lock/manifest.
Large model and dataset snapshots are external artifacts identified by commit
or content hash.

Collector CLI design target:

```bash
UV_CACHE_DIR=/tmp/orches-uv-cache uv run orches probe-trace-host --output artifacts/manifests/host.json
UV_CACHE_DIR=/tmp/orches-uv-cache uv run orches collect-text-trace --config configs/workloads/math500.yaml
UV_CACHE_DIR=/tmp/orches-uv-cache uv run orches collect-vision-trace --config configs/workloads/mathvista.yaml
UV_CACHE_DIR=/tmp/orches-uv-cache uv run orches validate-policy-trace artifacts/traces/policy.jsonl --manifest artifacts/manifests/policy.json --json
UV_CACHE_DIR=/tmp/orches-uv-cache uv run orches validate-trace artifacts/traces/replay.jsonl --json
```

`validate-policy-trace` and its strict manifest binding are implemented in
M4B.3. Host probing and both collection commands remain design targets.

## Artifact Contract

Each collection cell produces:

```text
artifacts/raw/<run-id>/pipeline-output.jsonl
artifacts/traces/<run-id>/replay.jsonl
artifacts/manifests/<run-id>.json
artifacts/reports/<run-id>-validation.json
```

The manifest includes source/model/tokenizer/dataset revisions, uv lock hash,
GPU/driver/CUDA identity, precision, seed, sampling, command, environment,
raw/replay hashes, request status, and active assumptions. A trace is eligible
for paper replay only when schema validation passes, every request has a
terminal selection, no provenance field is missing, and repeat collection has
a documented determinism result.

## Implementation Order

1. [Complete] Add schema v2 verifier calls, exact token/KV lineage,
   multi-selection semantics, and synthetic-only v1 migration.
2. [Complete] Replay schema v2 with per-token generation, multi-parent KV,
   exact selection-verifier work, and ordered physical pruning.
3. [Partial] Add the collection manifest contract and validation; host probing
   remains.
4. Implement the text trace sink and complete-tree export.
5. Implement layer-10/final PRM scoring with architecture-specific tests.
6. Run the 5070 Ti smallest-pair pilot.
7. Pin executable LLaVA-CoT and implement width-2/4 tournament export.
8. Run server 10-request, 50-request, then full matrices.
9. Replay the frozen traces with `paper_method`; run `orin_calibrated` only when
   target hardware is available.
