# ORCHES Techniques: Paper-to-Implementation Guide

This document explains the M3 implementation in source-reading order. M3 is a
functional implementation of the paper's three techniques. It is not yet a
paper-number reproduction: GPU/PIM rates, tier thresholds, cache timing, and
buffer/compaction parameters still require the calibration described in
`docs/calibration.md`.

## Code Style and Baseline Relationship

The implementation follows the useful conventions of AttAcc and Duplex without
copying their internal structure. AttAcc remains the native PIM command/timing
backend, while Duplex motivates the explicit model-to-operator and resource
timing decomposition. ORCHES-specific policy stays in typed Python modules so a
scheduler decision can be inspected without reading simulator internals.

The main conventions are:

1. Immutable dataclasses carry decisions and measurements; mutable classes are
   limited to real state machines such as timelines, caches, allocators, and
   speculation sessions.
2. Paper equations use MAC/s and elements/s. Conventional roofline estimates
   use FLOP/s and bytes/s. The two unit systems are not mixed.
3. Every undisclosed paper parameter is a constructor/configuration input and
   has an ID in `docs/assumptions.md`.
4. Third-party repositories are adapters/backends only. ORCHES does not patch
   them in place.
5. A result object retains alternatives, selected values, reasons, traffic, and
   stalls instead of returning only a device name or final latency.

## Technique 1A: Offline Assignment

Paper correspondence: Sec. 4.2.1, Fig. 8(a)-(e), and Eq. (1)-(4).

`scheduler/offline.py` implements the three branch-width regions:

| Region | Linear | Shared attention | Unique attention |
|---|---|---|---|
| Small | PIM primary | PIM | PIM |
| Medium | PIM primary | GPU | PIM |
| Large | GPU primary | GPU | PIM |

`TierThresholds` requires the medium and large lower bounds. They are not
hard-coded because the paper does not publish one universal pair of widths.
`solve_linear_balance_alpha` solves the crossing of Eq. (3) and Eq. (4), then
clamps alpha to `[0, 1]`. Co-processing is selected only when it improves the
tier's primary path and satisfies the paper condition:

```text
T_PIM >= max(T_GPU(alpha), T_PIM(alpha))
```

`OfflineAssignment` retains GPU-only, PIM-only, co-processed, and chosen
latencies; alpha; shared/unique placement; FP16 host-I/O element count; barrier
stall; and a human-readable reason.

## Technique 1B: Online Compensation

Paper correspondence: Sec. 4.2.2, Fig. 8(f), and Eq. (5)-(7).

`scheduler/online.py` represents each shared or unique KV segment as
`AttentionFragment(W_i, L_i, D)`. The implementation starts with every
`alpha_i=1`, sorts fragments from low to high `W_i`, and moves them to PIM until
the GPU/PIM critical path changes sign. All non-critical alphas stay binary;
only the crossing fragment receives an analytically solved continuous alpha.

`scheduler/generation.py` composes the T1A linear decision and T1B attention
decision into per-resource durations. It also computes the all-PIM duration
used during T2 speculative generation, where the paper disables T1.

## Technique 2A: Predictor and Speculation

Paper correspondence: Sec. 4.3.1, Fig. 9(a)-(c), Table 4, and Fig. 13.

`predictor/scores.py` preserves the current small-PRM score but replaces every
completed historical small-PRM score with the corresponding large-PRM score
when history alignment is enabled. Mean, minimum, and last-score aggregation
are explicit alternatives because the paper does not disclose the aggregate
function.

`predictor/partition.py` represents the Fig. 13 experiment in which the first
10 layers of one 8B PRM are the small prefix and the remaining layers are the
large suffix. The partition adds no model layers.

`predictor/speculation.py` is a single-use commit/rollback state machine. A
correct prediction commits speculative tokens. A mismatch records discarded
tokens and KV bytes, charges rollback/regeneration once, and returns the large
PRM's candidate as the only final branch.

## Technique 2B: Pipelined Verification

Paper correspondence: Sec. 4.3.2 and Fig. 9(d).

`predictor/pipeline.py` accumulates ready token batches until
`min_prefill_tokens` is reached. It then schedules a non-preemptive small-PRM
chunk on the GPU. Existing critical GPU work delays the chunk through
`EventTimeline`; completed chunks are never charged twice. A final below-
threshold remainder waits for generation completion. The serial comparison
pays one fixed launch overhead, while each pipelined chunk pays its own.

The implementation chooses the paper-permitted "delay" behavior rather than
preempting a running chunk. The threshold and timing rates are calibration
inputs (`A-T2-002`).

## Technique 3: Memory Structuring

Paper correspondence: Sec. 3.3, Fig. 10, Sec. 4.4, Table 5, and Sec. 5.5.

`memory/allocator.py` places immutable model weights first, then uses aligned
first fit for shared and unique KV blocks. Pruning removes physical blocks but
keeps the reasoning high-water mark, so holes remain observable. The selected
candidate is promoted from unique to shared KV without moving it.

The implemented fragmentation metric is:

```text
beta = holes within reasoning high-water span / reasoning high-water span
```

Model weights are excluded. This denominator interpretation is `A-T3-003`.
Compaction moves live KV blocks into a contiguous prefix, reduces the high-water
mark, and increments the generation of every moved block.

`memory/cache.py` stores `(candidate_id, start, length, generation)` in one
controller-wide fully-associative LRU cache. A hit models one SRAM plus one DRAM
access. A miss models one SRAM plus two dependent DRAM accesses. Generation
mismatch after compaction forces a miss and refreshes the mapping.

`memory/policy.py` supports both the paper's fixed 3/4/5 verification intervals
and an explicit beta threshold. `memory/trace.py` emits transaction-level DRAM
read/write commands for every move. `memory/buffer.py` models controller SRAM
staging, controller-to-bank writeback with zero PIM-host bytes, and GPU
synchronization from either a resident buffer segment or PIM banks.

## Request Replay Sequence

`replay.py` composes all techniques on one `EventTimeline`. For each step it:

1. schedules T1 GPU/PIM generation using the current shared and unique lengths;
2. allocates candidate KV through the controller buffer and resolves candidate
   physical addresses through the T3 cache;
3. emits token-triggered small-PRM chunks into GPU idle windows;
4. predicts the selected branch using optional history alignment;
5. runs the large PRM on GPU while PIM speculates on the next step with T1 off;
6. commits correct speculative work or discards its KV and schedules one
   rollback before regeneration;
7. promotes selected KV, prunes rejected candidates, evaluates the compaction
   policy, and schedules generated RD/WR traffic on PIM;
8. restores T1 after large-PRM completion and schedules the remaining or full
   next-step generation.

Replay results expose every event, per-step scheduler decisions, prediction
outcomes, memory snapshots, cache behavior, compaction traffic, and GPU/PIM/
controller utilization. Synthetic replay tests are control-flow evidence only
and must never be used as paper evaluation points.

`replay_v2.py` applies the same technique modules to exact-token traces. It
creates one generation plan per generated-token round, derives shared/unique
attention from all active parent KV lineages, allocates only recorded
materialized KV positions, and interleaves scalar/pairwise verifier calls with
ordered physical pruning. T2A remains limited to one retained beam because the
paper's predictor produces one speculative branch; multi-beam requests still
exercise T1, T2B, and T3 without silently choosing a winner. See
`docs/workload-trace.md` for the v2 invariants and remaining collector boundary.

## Verification

Run all tests:

```bash
uv run pytest
```

Run only M3 behavior tests:

```bash
uv run pytest \
  tests/test_offline_scheduler.py \
  tests/test_online_scheduler.py \
  tests/test_predictor.py \
  tests/test_verification_pipeline.py \
  tests/test_memory_structuring.py \
  tests/test_orches_replay.py
```

M3 completion establishes functional and control-flow equivalence. M4 and M5
must still add calibrated baseline timing/energy/area, real collected traces,
evaluation manifests, and comparison with the paper's figures and tables.
