# ORCHES Development Record

This document is the implementation ledger for the ORCHES reproduction. It is
updated at every milestone boundary and records what was implemented, how it
maps to the paper, what was verified, and what remains uncertain. Final figures
must be traceable through this document to source revisions, configuration
files, tests, and raw result manifests.

## Status Summary

| Milestone | Scope | Status | Evidence |
|---|---|---|---|
| M0 | Environment and source freeze | Complete | `uv.lock`, `.python-version`, `third_party.lock` |
| M1 | Hardware contract and configuration validation | Complete | 14 tests; validated GPU/PIM configs; bootstrap audit |
| M2 | TTC traces, GPU baseline, PIM microbenchmarks | In progress (M2B schema complete) | Exact-token/KV schema complete; real collectors and GPU runner pending |
| M3 | ORCHES Techniques 1, 2, and 3 | Complete (functional) | 105 tests; request replay; calibration/evaluation pending |
| M4 | Baselines, energy, area, utilization | In progress (M4B.2 complete) | 141 tests; exact-token replay/contracts/accounting complete; collectors/launchers pending |
| M5 | Paper evaluation and deviation report | Not started | - |

## Commit Policy

The first Git commit consolidates M0 through M2A because those checkpoints were
implemented before per-milestone commits were requested. It is intentionally
labelled as a consolidated foundation rather than presenting a reconstructed
history. Starting with M3, every completed milestone must update this ledger,
pass its declared verification checks, and be committed separately. Incomplete
milestone files are excluded from earlier checkpoint commits.

## M0: Environment and Source Freeze

### Objective

Create a reproducible boundary before implementing ORCHES-specific behavior.
The environment must identify the Python dependency graph, native toolchain,
upstream repositories, and the exact Ramulator2 revision to which AttAcc
patches are applied.

### Completed Work

1. Created a UV project using Python 3.10.12.
2. Locked the current Python dependencies in `uv.lock`.
3. Installed and audited the native compiler and build toolchain.
4. Cloned the four upstream projects required by the reproduction plan.
5. Built the AttAcc-modified Ramulator2 executable.
6. Generated a small bank-level PIM trace and ran the AttAcc smoke test.
7. Recorded all upstream revisions and licenses in `third_party.lock`.
8. Initialized `orches/` as an independent Git repository while excluding
   reproducible third-party checkouts and generated artifacts.
9. Recorded the paper DOI and local PDF SHA-256 in `third_party.lock`.

### Frozen Toolchain

| Component | Version |
|---|---:|
| Python | 3.10.12 |
| UV | 0.11.29 |
| CMake | 3.22.1 |
| Ninja | 1.10.1 |
| Clang | 14.0.0 |
| GCC | 11.4.0 |
| Git | 2.34.1 |
| Git LFS | 3.0.2 |

### Upstream Revisions

| Project | Commit | Intended use |
|---|---|---|
| AttAcc | `c60005143a6b492d7ef83231723386478b59a506` | PIM simulator foundation |
| AttAcc Ramulator gitlink | `0eafaa4c3df7b333f8645f1249afa52390c89616` | Parent-repository provenance |
| AttAcc Ramulator build base | `b7c70275f04126c647edb989270cc429776955d1` | Actual patched build base |
| Duplex LLMSimulator | `419252761fbdb95b789778a02256d458a5537ec7` | Duplex baseline |
| Duplex Ramulator2 | `b5262869b24790e5ec62d9d9974fca9bbb4f54e0` | Duplex memory simulator |
| Compute-optimal TTS | `0ee2578af1f8d6cac445c9c4c72780528bb94556` | Text reasoning traces |
| LLaVA-o1 | `fe3c9304fb08168fa3ae35d3380732439a2d4a36` | Vision reasoning traces |

### Important AttAcc Build Detail

The AttAcc parent repository records Ramulator2 commit `0eafaa4`, but
`set_pim_ramulator.sh` explicitly executes:

```text
git reset --hard b7c70275f04126c647edb989270cc429776955d1
```

It then copies AttAcc source files and applies its patch series. Therefore, the
dirty submodule status shown after installation is expected. Reproducibility
requires recording all three inputs: the AttAcc parent commit, the parent
gitlink, and the actual Ramulator2 build base.

### Verification Evidence

The following artifacts were observed after the smoke test:

| Artifact | Meaning |
|---|---|
| Paper SHA-256 `70ed78d47642...3d4106ac00` | Exact local paper input used for the reproduction |
| `third_party/attacc_simulator/ramulator2/ramulator2` | Built x86-64 simulator executable |
| `third_party/attacc_simulator/ramulator2/trace_gen/attacc_bank.trace` | Generated bank-level test trace |
| `third_party/attacc_simulator/ramulator2/log/attacc_bank/` | Per-channel command logs from execution |

The executable is an x86-64 ELF binary. The trace and command logs confirm that
the frontend, patched DRAM model, controller, scheduler, and trace recorder were
all exercised.

### Paper Relationship

This milestone implements the reproducibility prerequisite for ORCHES Sec. 5.1:
the paper states that its simulator extends AttAcc, uses a modified Ramulator2,
and retains AttAcc unit latency and energy values. No ORCHES performance claim
is evaluated at M0.

### Decisions

1. Third-party repositories are not copied into ORCHES history. Their immutable
   revisions are recorded in `third_party.lock` and will be reconstructed by a
   bootstrap script in a later M1 update.
2. ORCHES Python code will use a `src/` package layout and standard-library
   interfaces where practical. Dependencies are added only when they reduce
   meaningful complexity.
3. Hardware configuration values will carry provenance labels: `PAPER`,
   `INHERITED`, `CALIBRATED`, or `ASSUMED`.
4. LLaVA-o1 remains an external trace-generation tool because no license file
   was found in the frozen checkout. Its code will not be copied or modified in
   the ORCHES package until licensing is clarified.

### Open Issues

1. The current execution environment does not expose a GPU, so model trace
   collection and AGX Orin calibration cannot be validated here.
2. The exact ORCHES artifact is not available in the workspace; all unspecified
   parameters must remain explicit assumptions.
3. The paper reports 32 GB, while a Ramulator organization is naturally
   expressed using binary capacities. M1 must document and validate this unit
   interpretation rather than silently equating GB and GiB.

### Next Milestone

M1 will implement the hardware configuration contract, machine-readable source
metadata, derived 32 GiB/2048-bank checks, a command-line validator, and focused
unit tests. It will not yet implement the paper's scheduling equations.

## M1: Hardware Contract and Configuration Validation

### Objective

Translate the paper's hardware description into a strict, reviewable contract
before implementing performance equations or scheduling policies. Every
numeric input must identify whether it comes from ORCHES, an inherited
baseline/device specification, calibration, or an explicit assumption.

M1 deliberately does not implement Eq. (1)-(7), execute LLM workloads, or claim
any ORCHES speedup. It establishes the physical and provenance invariants that
those later implementations must consume.

### Implementation Sequence

1. Added a standard `src/orches` Python package and the `orches` CLI entry
   point.
2. Defined `SourcedValue` and the four evidence states: `PAPER`, `INHERITED`,
   `CALIBRATED`, and `ASSUMED`.
3. Implemented immutable GPU and PIM hardware data classes with physical
   consistency checks.
4. Implemented a strict YAML loader. It rejects missing keys, unknown keys,
   wrong types, missing provenance, and unsupported evidence states.
5. Added paper-mapped AGX Orin 32 GB and ORCHES PIM 32 GB configurations.
6. Added human-readable and JSON validation output. JSON includes every value,
   source, evidence state, note, and derived quantity.
7. Added a bootstrap implementation that reads `third_party.lock`, reconstructs
   missing repositories, verifies exact revisions, and builds AttAcc Ramulator2.
8. Added hardware-contract, paper-to-code, and assumption-register documents.
9. Added focused unit and CLI tests, then ran a read-only third-party audit.

### Code Structure

| Path | Responsibility |
|---|---|
| `src/orches/provenance.py` | Evidence status and sourced values |
| `src/orches/hardware.py` | Typed GPU/PIM contracts and derived physical invariants |
| `src/orches/config.py` | Strict YAML parsing with no hidden defaults |
| `src/orches/cli.py` | User and machine-readable configuration validation |
| `src/orches/bootstrap.py` | Safe reconstruction and revision verification from `third_party.lock` |
| `configs/hardware/agx_orin_32gb.yaml` | GPU baseline and bandwidth sensitivity points |
| `configs/hardware/orches_pim_32gb.yaml` | PIM hierarchy, capacity, lanes, timing, and host I/O limit |
| `docs/hardware-contract.md` | Contract semantics and failure policy |
| `docs/paper-to-code.md` | Paper requirement to code/test map |
| `docs/assumptions.md` | Open assumptions and required validation |
| `tests/` | Contract, failure-path, CLI, and lock-file tests |

### Style Relationship to AttAcc and Duplex

The implementation preserves AttAcc/Ramulator2 hierarchy names exactly:
`channel`, `pseudochannel`, `rank`, `bankgroup`, and `bank`. Timing preset,
refresh manager, controller clock ratio, and organization preset are explicit
fields so the later backend adapter can produce familiar Ramulator2 YAML.

The system-level separation follows Duplex: workload/scheduling logic remains
outside the memory simulator, while Ramulator2 is a timing backend. ORCHES code
does not copy either baseline's implicit defaults or modify their source trees
in place. Provenance metadata, frozen data classes, strict parsing, and tests
are local additions intended to make baseline review and deviation reporting
more reliable.

### Frozen Hardware Mapping

#### GPU

| Parameter | Value | Evidence |
|---|---:|---|
| Module | Jetson AGX Orin 32 GB | `PAPER` |
| Capacity | 32 GB | `PAPER` |
| SoC-visible bandwidth | 204.8 GB/s | `PAPER` |
| Bandwidth scales | 100%, 75%, 50% | `PAPER` |
| CUDA cores | 1792 | `INHERITED`, NVIDIA Technical Brief v1.2 |
| Tensor cores | 56 | `INHERITED`, NVIDIA Technical Brief v1.2 |
| Maximum GPU frequency | 930 MHz | `INHERITED`, NVIDIA Technical Brief v1.2 |

The official specification check corrected an intermediate 1300 MHz value:
1.3 GHz belongs to the 64 GB module, while the 32 GB module in Technical Brief
v1.2 is specified at 930 MHz. This peak remains only an analytical upper bound;
M2 must use calibrated effective throughput.

The derived bandwidth points are `204.8`, `153.6`, and `102.4 GB/s`. Scaling
bandwidth does not implicitly scale GPU compute capability.

#### PIM

| Parameter | Value | Evidence |
|---|---:|---|
| Reported capacity | 32 GB | `PAPER` |
| Simulated capacity | 32 GiB | `ASSUMED` unit mapping |
| Memory banks | 2048 | `PAPER` |
| GEMV lanes per bank | 16 | `PAPER` |
| Total GEMV lanes | 32768 | Derived |
| AttAcc organization preset | `HBM3_8Gb_2R` | `INHERITED` |
| Channels | 32 | `ASSUMED` from paper constraints and AttAcc organization |
| Pseudochannels/channel | 2 | `INHERITED` |
| Ranks/pseudochannel | 2 | `INHERITED` |
| Bank groups/rank | 4 | `INHERITED` |
| Banks/bank group | 4 | `INHERITED` |
| Timing preset | `HBM3_5.2Gbps` | `INHERITED` |
| Refresh policy | `AllBankHBM3` | `INHERITED` |
| Host/controller I/O | 204.8 GB/s | `PAPER` |

The validated organization equations are:

```text
banks = 32 channels * 2 pseudochannels * 2 ranks * 4 bank groups * 4 banks
      = 2048 banks

capacity = 32 channels * 8 Gibit/channel / 8
         = 32 GiB

GEMV lanes = 2048 banks * 16 lanes/bank
           = 32768 lanes
```

During the first test run, the plan's proposed 16-channel organization derived
only 1024 banks. The earlier expression `16*2*2*4*4=2048` was an arithmetic
error. The implementation was not weakened to accept the mismatch. Instead,
the plan and configuration were corrected to retain AttAcc's
`HBM3_8Gb_2R` preset and double AttAcc's channel count from 16 to 32. This is an
explicit assumption (`A-HW-002`) because the paper gives capacity and bank
count but does not disclose the exact Ramulator organization.

### Bootstrap Behavior

`bash scripts/bootstrap.sh` first performs `uv sync --frozen`, then reads exact
repository revisions from `third_party.lock`. Missing checkouts are cloned and
detached at their locked commits. Existing checkouts with a different commit
cause an error; the script does not reset or overwrite them.

AttAcc is handled specially because its upstream setup script intentionally
resets Ramulator2 from the parent gitlink revision to build-base revision
`b7c7027` before applying patches. An existing executable is verified against
that build-base revision. `--verify-only` performs no checkout or build and is
the audit mode used for M1 verification.

### Paper Relationship

| Paper location | M1 implementation |
|---|---|
| Sec. 5.1 | AGX Orin 32 GB baseline, 204.8 GB/s bandwidth, 32 GB PIM, 2048 banks |
| Fig. 11, Sec. 5.1 | 100%, 75%, and 50% bandwidth profiles |
| Sec. 4.1, Sec. 5.1 | 16 multiplier/adder GEMV lanes per memory bank |
| Fig. 7, Sec. 4.1 | Controller components documented as deferred fields with no hidden performance defaults |
| Sec. 5.1 | AttAcc timing/refresh inheritance and frozen modified Ramulator2 base |

Detailed row-level mappings are maintained in `docs/paper-to-code.md`.

### Verification Evidence

The final M1 commands were:

```bash
uv run --frozen pytest
bash scripts/bootstrap.sh --verify-only
uv run --frozen orches validate-config configs/hardware/agx_orin_32gb.yaml
uv run --frozen orches validate-config configs/hardware/orches_pim_32gb.yaml
python -m compileall -q src tests scripts
```

Results:

| Check | Result |
|---|---|
| Unit and CLI tests | 14 passed |
| Python syntax compilation | Passed |
| AGX Orin validation | Passed; bandwidth points `204.8/153.6/102.4 GB/s` |
| PIM validation | Passed; `2048 banks`, `32 GiB`, `32768 lanes` |
| PIM evidence report | 4 `PAPER`, 10 `INHERITED`, 0 `CALIBRATED`, 2 `ASSUMED` |
| Third-party audit | All four top-level repositories and both Ramulator2 revisions verified |
| AttAcc executable audit | Build-base commit and executable verified |

Tests cover valid paper configurations, derived quantities, inconsistent bank
count, inconsistent capacity, unknown fields, missing provenance, invalid
evidence status, invalid bandwidth ordering, non-mutation of input data, CLI
human/JSON/error output, and lock-file parsing.

### Remaining Assumptions and Limits

1. `A-HW-001`: the paper's 32 GB PIM is represented as 32 GiB in Ramulator2.
2. `A-HW-002`: the PIM uses AttAcc `HBM3_8Gb_2R` with 32 channels.
3. `A-GPU-001`: official peak GPU specifications are bounds, not achieved
   performance; M2 calibration is required.
4. `A-PIM-001`: each paper multiplier/adder pair maps to one configured GEMV
   lane; command-level semantics still require microbenchmark validation.
5. Controller address-cache, accumulation, softmax, state-machine, and
   shared-KV-buffer sizes or latencies are not disclosed. They remain deferred
   rather than receiving fabricated defaults.
6. No GPU is visible in the current environment, so AGX Orin calibration and
   model trace generation have not been run.

### Completion Decision

M1 is complete because all disclosed hardware quantities have typed fields and
sources, the inherited AttAcc hierarchy is explicit, contradictory physical
organizations fail before simulation, third-party reconstruction is auditable,
and the focused tests pass. Open assumptions are not considered resolved; they
are carried into manifests and later sensitivity analysis.

### Next Milestone

M2 will define the versioned TTC workload trace schema, add a deterministic
synthetic trace generator for simulator development, and implement analytical
or calibrated GPU/PIM primitive models. It will map GPU timing to Eq. (1), PIM
timing to Eq. (2)-(3), and preserve inherited, measured, and assumed parameters
separately.

## M2A: Trace, Model, Timing, and Native PIM Foundation

### Checkpoint Scope

M2 spans Phase 2-4 and is not complete yet. This checkpoint implements a
synthetic replay contract, Transformer shapes/operator expansion, unit-safe
analytical timing equations, reversible PIM addressing, and a native AttAcc
smoke path. Paper-facing schema v2 and real workload collection, the optional
AGX Orin calibration profile, full GPU baseline execution, and calibrated PIM
primitive sweeps remain open.

### Implementation Sequence

1. Defined strict TTC request/step/candidate dataclasses and schema version 1.
2. Added deterministic JSONL serialization, parsing, hashing, and full
   cross-step control-flow validation.
3. Added a seed-reproducible synthetic text/vision trace generator and CLI.
4. Added architecture profiles for all three text policy models and three PRM
   proxies from Sec. 5.1.
5. Expanded one decoder layer into an explicit operator DAG with conventional
   MAC/FLOP/byte counts and shared/unique KV fragments.
6. Implemented the paper's Eq. (1)-(4) with MAC/s and element/s types, separate
   from the conventional GPU FLOP/byte roofline model.
7. Extended the PIM hardware contract with row, column, and transaction sizes,
   deriving a 35-bit 32 GiB address space independently of density.
8. Implemented a reversible AttAcc-compatible address mapper, command trace
   parser/writer, Ramulator2 YAML renderer, native runner, and counter parser.
9. Added a deterministic multi-resource event timeline with dependency and
   overlap validation as the common substrate for later system simulation.
10. Added model/trace/PIM CLI commands, documentation, tests, and a native-run
    manifest.

### TTC Trace Contract

Each request records dataset identity/difficulty, modality, policy/PRM and
tokenizer revisions, seed/sampling inputs, prompt/image token counts, search
width, and every reasoning step. Each candidate records its parent, generated
and unique-KV token count, per-token completion timeline, small/large PRM
scores, and selected/pruned outcome.

The validator enforces:

```text
candidate_count(step) = search_width
next.parent = current.selected_candidate
next.shared_KV = current.shared_KV + selected.unique_KV
unique_KV = generated_tokens                  # schema v1
```

These are synthetic v1 invariants only. They do not claim that output
characters equal tokens or that generated tokens always equal materialized KV.
Paper-facing schema v2 must use exact model token IDs/masks, actual KV
materialization, ordered selection lineage, and multiple retained beams.

Prompt, completion, answer, and correctness bodies are intentionally excluded.
Correctness will be a separate record keyed by request and dataset IDs.
Synthetic traces are marked `source_kind=synthetic`; the CLI reports
`evaluation_eligible: False`.

### Model and Operator Mapping

Policy architecture profiles are based on official model configs:

| Model | Hidden / intermediate | Layers | Q / KV heads | Evidence |
|---|---:|---:|---:|---|
| Llama3.2-1B-Instruct | 2048 / 8192 | 16 | 32 / 8 | Official gated config; snapshot SHA still required |
| Qwen2.5-1.5B-Instruct | 1536 / 8960 | 28 | 12 / 2 | Official revision `989aa79` |
| Qwen2.5-3B-Instruct | 2048 / 11008 | 36 | 16 / 2 | Official revision `aa8e725` |

The paper's exact `*-PRM-Tuned` checkpoint revisions are not disclosed. The
Qwen 1.5B base, public Qwen2.5-Math-PRM-7B, and public RLHFlow Llama3.1-8B PRM
architectures are recorded as proxies with `ASSUMED` status. A collected trace
must identify the checkpoint actually executed.

The decoder DAG follows the same model-to-operator layering used by Duplex but
keeps ORCHES shared/unique KV fragments explicit. Shared and unique score
fragments feed one combined softmax; context fragments then feed one output
projection. Shared KV is read once for a batched query, while unique KV traffic
scales with width. Tests confirm shared-attention arithmetic intensity rises
with width and unique-attention intensity remains constant.

### Timing Convention Boundary

`models/timing.py` separates two incompatible counting conventions:

1. Paper equations use MAC/s and tensor-elements/s, follow printed Eq. (1)-(4)
   additively, and do not multiply operation counts by two.
2. The conventional GPU roofline uses FLOP/s and byte/s, counts one MAC as two
   FLOPs, and computes `max(compute, memory) + launch + synchronization`.

Distinct rate classes and function names prevent accidental cross-use. The
100/75/50% bandwidth helper changes memory rate only.

### PIM Address and Native Backend

The full hierarchy is:

```text
32 channels * 2 pseudochannels * 2 ranks * 4 bankgroups * 4 banks
* 16384 rows * 32 columns * 32 bytes = 32 GiB
```

This independently validates the 2048-bank and 35-bit address contract. The
adapter emits AttAcc's existing command vocabulary and renders the frozen
`HBM3-PIM`/`PIMDRAM`/PIM-scheduler/`AllBankHBM3` configuration without editing
third-party files.

The M2A smoke command was:

```bash
uv run orches pim-microbench /tmp/orches-pim-m2 --json
```

It generated 224 commands across 32 channels. Ramulator2 completed in 22 memory
cycles and reported 32 GEMV-buffer writes, 32 all-bank MACs, 32 moves to the
softmax buffer, and 32 softmax requests. Exact inputs, hashes, host information,
and counters are in `artifacts/manifests/m2_pim_smoke.json`. This run validates
the command path only; it is not an evaluation throughput point.

### Verification Evidence

| Check | Result |
|---|---|
| M0-M2A Python suite | 64 passed, including native integrations |
| Syntax compilation | Passed |
| Synthetic JSONL | Same seed produces byte-identical replay and SHA-256 |
| Trace failure paths | Selected/pruned/parent/shared-KV/timeline/schema errors rejected |
| Six text model profiles | Shapes and GQA invariants validated |
| Eq. (1)-(4) | Hand calculations, alpha boundaries, and unit conventions passed |
| Operator DAG | Topology, one combined softmax, traffic and intensity trends passed |
| PIM address mapping | Zero, maximum, interior round trips and invalid ranges passed |
| Native PIM integration | 32-channel trace executed and expected counters observed |

### Remaining M2 Work

1. Use schema v2 to instrument the frozen compute-optimal-TTS and LLaVA-CoT
   pipelines and collect exact-token/KV MATH500, LiveCodeBench, and MathVista
   traces; schema v1 remains synthetic-only.
2. Resolve the exact paper policy/PRM checkpoint and tokenizer revisions,
   including the custom tuned PRMs.
3. Build the `paper_method` AttAcc-style AGX Orin profile; optionally run AGX
   Orin GEMM/GEMV/attention/memory/launch calibration and store raw measurements
   as the separate `orin_calibrated` profile.
4. Integrate the generic event timeline into an executable pure-GPU request
   baseline; the current timeline is validated but does not yet replay requests.
5. Expand native PIM microbenchmarks for linear, shared/unique attention,
   host-I/O, row locality, refresh, barrier, accumulation, and softmax scaling.
6. Fit and validate PIM primitive rates or use raw Ramulator cycles directly;
   do not infer them from the paper's final speedup.

The required calibration matrix and acceptance criteria are in
`docs/calibration.md`.

## M3: Core ORCHES Techniques and Request Replay

### Checkpoint Scope

This checkpoint implements the functional behavior of Techniques 1, 2, and 3
from Sec. 4.2-4.4 and composes them on one request-level resource timeline. It
establishes scheduling, prediction, rollback, branch pruning, address
translation, and compaction invariants. It does not claim the paper's numerical
speedup, energy, area, or memory-saving results: real schema-v2 traces and a
frozen evidence profile remain M2/M4 prerequisites for M5 evaluation. The
profile may be `paper_method`; `orin_calibrated` is optional stronger evidence.

### Implementation Sequence

1. Implemented T1A's small/medium/large placement table around explicit,
   calibration-supplied width thresholds.
2. Solved the Eq. (3)-(4) GPU/PIM linear crossing analytically and retained all
   single-device and co-processing alternatives in each decision.
3. Enforced the paper's `T_PIM >= max(T_GPU(alpha), T_PIM(alpha))` condition and
   recorded host-I/O elements plus barrier imbalance.
4. Implemented T1B's Eq. (5)-(7) over explicit `(W_i, L_i, D)` KV fragments,
   sorting by width and solving at most one continuous critical alpha.
5. Composed T1A linear and T1B attention timing into one per-resource
   generation plan, including an all-PIM path for T2 speculation.
6. Implemented the history-aligned small-PRM predictor, stable tie breaking,
   explicit score aggregation alternatives, and the Fig. 13 prefix/suffix PRM
   partition.
7. Implemented a single-use speculation state machine with commit, rollback,
   discarded-token/KV accounting, and large-PRM branch authority.
8. Implemented token-thresholded T2B pre-verification on the shared GPU
   timeline, including per-chunk overhead and a one-launch serial comparison.
9. Implemented T3's weights-first KV allocator, physical holes, beta metric,
   address cache, fixed/beta compaction policy, transaction-level relocation
   trace, controller shared-KV buffer, and GPU synchronization accounting.
10. Added request replay that connects all three techniques to one deterministic
    GPU/PIM/controller/link resource timeline and exposes every decision and
    traffic component.

### Technique 1 Correspondence

`scheduler/offline.py` maps Fig. 8(b)-(d) as follows:

| Width tier | Linear primary | Shared KV query | Unique KV query |
|---|---|---|---|
| Small | PIM | PIM | PIM |
| Medium | PIM | GPU | PIM |
| Large | GPU | GPU | PIM |

`TierThresholds` does not embed unexplained model-independent constants. The
transition widths are assumption `A-T1-001` and must come from the frozen
calibration for each model/bandwidth point. `solve_linear_balance_alpha`
expresses both Eq. (3) and Eq. (4) as affine functions of alpha, solves their
crossing, and clamps the result to the legal range. The selected decision keeps
the original balance timing even when co-processing is rejected, allowing a
reviewer to reconstruct the inequality.

For T1B, each accumulated shared segment and each candidate-unique segment is
an `AttentionFragment`. `balance_attention_fragments` begins at all-GPU,
switches fragments to PIM in ascending width order, detects the first resource
crossing, and solves only that fragment's alpha. `OnlineBalanceDecision`
reports the complete alpha vector, critical fragment, ordered IDs, GPU/PIM
components, critical path, and imbalance stall.

### Technique 2 Correspondence

`CandidateScorePath` stores the current small-PRM score and equal-length small/
large historical arrays. With history alignment enabled, only completed
history is replaced by large-PRM scores; the current score remains from the
small PRM as shown in Fig. 9(c). Mean, minimum, and last aggregation are all
implemented because the paper does not disclose its aggregate function
(`A-T2-001`).

The T2A timeline gives the GPU priority to large-PRM verification and uses PIM
for speculative next-step generation. T1 is disabled during this window by
using the all-PIM generation plan. At large-PRM completion, a match preserves
completed speculative work and restores T1 for the remainder. A mismatch
materializes and prunes the wrong speculative KV, executes one controller
rollback event, and regenerates from the trace's large-PRM-selected candidate.

T2B turns token completion times into batches. Once the explicit
`min_prefill_tokens` threshold is met, the batch is submitted to the GPU event
timeline; existing generation/verification events delay it without overlap or
double charging. The current implementation uses non-preemptive chunks and the
paper-permitted delay behavior. Threshold and timing calibration are tracked by
`A-T2-002`.

### Technique 3 Correspondence

The allocator keeps model weights in an immutable prefix. Shared and unique KV
use aligned first fit in the reasoning region. Pruning deletes the physical
allocation but does not lower the high-water mark, so the removed range becomes
a real hole. The implemented interpretation of Sec. 4.4's metric is:

```text
beta = reasoning hole bytes / reasoning high-water span bytes
```

This excludes model weights and is recorded as `A-T3-003`. Compaction sorts
live KV by physical address, packs it after the weight prefix, increments every
moved block's generation, and reports moved/reclaimed bytes plus beta before
and after. `memory/trace.py` emits an ordered DRAM read and write for every
transaction of every move.

The controller address cache maps the logical candidate/block ID to start,
length, and generation. A valid hit models SRAM+DRAM; a miss models
SRAM+DRAM+DRAM. A stale generation after compaction is invalidated and refilled.
The cache is one controller-wide fully-associative LRU instance under
`A-T3-001`; capacities and access latencies remain explicit inputs.

The shared-KV buffer stages QKV output in controller SRAM and writes it to PIM
banks with zero PIM-host bytes. GPU synchronization identifies whether bytes
came directly from the resident controller buffer or were fetched from PIM
banks. Compaction can trigger at any explicit beta threshold or after a fixed
number of PRM verifications; intervals 3, 4, and 5 are tested because Sec. 5.5
reports that range.

### Request-Level Execution Order

For each trace step, `OrchesRequestReplayer` performs the following auditable
sequence:

1. plan and schedule T1 generation on GPU and PIM;
2. allocate candidate KV through the shared controller buffer;
3. resolve candidate addresses and schedule controller lookup latency;
4. feed token-ready batches into pipelined small-PRM verification;
5. predict one candidate and schedule authoritative large-PRM verification;
6. overlap next-step all-PIM speculation with the large PRM;
7. commit or rollback, promote selected KV, and prune rejected KV;
8. evaluate T3 policy and schedule generated compaction RD/WR traffic;
9. restore T1 and execute the uncompleted or full next-step work.

The replay result retains per-step decisions, all events, memory snapshots,
prediction accuracy, compaction bytes, and resource busy time/utilization. A
replayer is single-use so state from one request cannot leak into another.

### Verification Evidence

| Check | Result |
|---|---|
| Full Python suite | 105 passed |
| Syntax compilation | Passed for `src` and `tests` |
| T1A alpha | Analytical crossing matches a 10,001-point dense search |
| T1A paper guard | Rejects co-processing when PIM-only is already faster |
| T1B alpha vector | At most one fractional alpha; near small-grid optimum |
| T2 history alignment | Corrects a constructed ranking; deterministic ties |
| T2 speculation | Correct path commits; mismatch discards KV and rolls back once |
| T2B pipeline | 0/partial/100% overlap, shared timeline, and serial overhead tested |
| T3 allocation | Deterministic random allocate/prune/compact preserves live objects |
| T3 address cache | LRU, hit/miss timing, eviction, prune, and stale generation tested |
| T3 policies | Fixed intervals 3/4/5 and beta threshold tested |
| T3 traffic | Compaction emits balanced transaction-level read/write bytes |
| Request replay | Correct/mismatch prediction and T3 enabled/disabled compared |

### Remaining Calibration and Evaluation Work

1. Replace synthetic replay rates with a frozen `paper_method` GPU/PIM profile;
   add `orin_calibrated` separately when target measurements are available.
   Synthetic tests are not evaluation evidence.
2. Collect schema-v2 MATH500, LiveCodeBench, and MathVista traces with exact
   model/tokenizer revisions, token IDs, KV materialization, and selection
   lineage.
3. Derive per-model T1 tier thresholds and T2B prefill thresholds rather than
   selecting values from final paper speedups.
4. Calibrate cache SRAM, controller buffer, compaction bandwidth/energy, and
   area; reproduce the reported 12% area and 0.12% runtime overhead definitions.
5. Add GPU, AttAcc, Duplex, ORCHES-A/B/C adapters on identical trace inputs and
   implement M4 energy/utilization accounting.
6. Run the full evaluation matrix and compare Table 4/5, Fig. 11-13, and all
   paper aggregates with deviation attribution.

## M4A: Baseline Contracts and Metric Accounting

### Checkpoint Scope

M4A defines the comparison boundary before implementing native experiment
launchers. It freezes the paper's baseline/ablation switch matrix, enforces a
common workload and hardware fingerprint, parses frozen AttAcc and Duplex
output schemas into SI units, adapts full ORCHES replay, and implements shared
energy/area/utilization accounting. It does not yet claim executable parity for
GPU, ORCHES-A/B/C, or native end-to-end AttAcc/Duplex runs.

### Upstream Audit

The AttAcc source at `c60005143a6b492d7ef83231723386478b59a506`
returns layer-group time in milliseconds and energy in nanojoules. Its energy
model is activity based: memory/communication bytes and MAC operations are
multiplied by pJ coefficients. The adapter follows the CSV schema written by
upstream `main.py`; it does not import mutable third-party Python modules into
the ORCHES package.

The Duplex LLMSimulator source at
`419252761fbdb95b789778a02256d458a5537ec7` exports mixed iteration and request
records through `Cluster::exportToCSV`. Timing fields are nanoseconds and energy
fields are nanojoules. `iter_info=1` rows carry device activity; `type=e2e` rows
carry request completion. This distinction is preserved by the adapter.

### Paper Configuration Matrix

`baselines/definitions.py` makes each paper configuration explicit:

| Result set | Required systems |
|---|---|
| Fig. 11 | GPU, AttAcc, Duplex, full ORCHES |
| Fig. 12 | GPU, AttAcc, ORCHES-A, ORCHES-B, ORCHES-C |
| Table 6 | GPU, T1 only, T2 only, full ORCHES |

ORCHES-A places all computation on PIM. ORCHES-B enables adaptive linear
assignment. ORCHES-C adds T1B dynamic compensation. T2 is disabled in all
three, exactly as Sec. 5.3 states. T3 remains enabled because Sec. 5.3 does not
state that memory structuring is removed; this interpretation is
`A-BASE-001` and requires sensitivity validation.

### Fairness and Failure Contract

Every result contains one `FairnessContract` whose deterministic SHA-256 covers
the trace, ordered request IDs, all model/tokenizer identities and revisions,
weight/activation precision, GPU count, SoC bandwidth, PIM capacity, and
hardware-contract hash. `compare_baselines` rejects mixed fingerprints,
duplicate or missing required systems, and normalization without a successful
GPU reference.

Run status is one of success, OOM, failed, or missing. Non-success rows require
an error and cannot carry metrics. They remain in the normalized comparison
with null speedup/energy efficiency; no pre-aggregation filter can silently
remove them.

### Native Result Adapters

`parse_attacc_csv` converts `g_time (ms)` to seconds and all generation-energy
nJ columns to joules. It converts native GiB capacity to bytes and emits OOM
when `required_cap` exceeds it. `parse_duplex_csv` converts all ns/nJ columns,
uses mean E2E request latency, sums iteration activity energy, and propagates
the native OOM flag. Both reject missing, negative, non-numeric, `NaN`, and
infinite values.

The adapters translate results only. M4B launchers must still generate native
configs from the fairness contract and bind command/config/binary/output hashes
to each result.

### Energy, Area, and Utilization

`metrics/energy.py` keeps pJ/activity coefficients distinct from measured
energy. The inherited AttAcc bank-level constants are:

| Activity | Unit energy | Status |
|---|---:|---|
| GPU/PIM MAC | 0.32 pJ/MAC | `INHERITED` |
| GPU off-memory byte | 28.72 pJ/byte | `INHERITED` |
| PIM bank-level memory byte | 4.4 pJ/byte | `INHERITED` |
| Host communication byte | 10.4 pJ/byte | `INHERITED` |
| Controller SRAM byte | 0.0034 pJ/byte | `INHERITED` proxy |

Generation activity is reconstructed from the selected Eq. (1)-(7) alphas,
including discarded all-PIM speculation. Replay aggregation adds address-cache
SRAM/DRAM accesses, shared-buffer writeback, GPU synchronization, and
compaction RD/WR bytes. PRM verifier activity is a required explicit input; it
is never inferred from elapsed time.

`AreaReport` requires component mm2 and a named positive baseline denominator,
so a 12% overhead cannot be reported without defining "12% of what".
`UtilizationReport` uses busy/makespan under one common timeline and rejects
busy time greater than wall time.

### Verification Evidence

| Check | Result |
|---|---|
| Full Python suite | 121 passed |
| Syntax compilation and diff whitespace | Passed |
| Paper baseline sets | Fig. 11/12 and Table 6 definitions tested |
| Fairness fingerprint | Stable for equal inputs; changes with SoC bandwidth |
| Failure visibility | OOM remains a comparison row with null normalized value |
| AttAcc adapter | ms/nJ/GiB conversion, components, OOM, non-finite rejection |
| Duplex adapter | E2E/iteration separation and ns/nJ conversion |
| Energy accounting | AttAcc constants, pJ-to-J conversion, exact zero activity |
| Scheduler activity | GPU+PIM MACs conserve the Eq. (1)-(7) workload |
| Area accounting | Explicit components reproduce a constructed 12% overhead |
| Utilization | Cross-resource overlap uses one makespan without double counting |
| ORCHES adapter | Full replay produces common latency/energy/memory/utilization |

### Remaining M4 Work

1. Implement launch manifests and executable native AttAcc/Duplex commands from
   one fairness contract.
2. Implement analytical/native GPU and ORCHES-A/B/C/T1-only/T2-only request
   runners on the same TTC trace.
3. Bind real verifier operator activity, not development-only placeholder
   counters, to ORCHES energy.
4. Calibrate cache/buffer unit energy and component area; resolve
   `A-ENERGY-001` and `A-AREA-001`.
5. Produce one smoke comparison containing every required status and raw result
   hash before starting the paper evaluation matrix.

### M4B Design Checkpoint: GPU Evidence and Real Trace Sources

The Sec. 5.1 and upstream-source audit corrected an overly strict earlier
requirement: real AGX Orin measurements are not mandatory inputs to the paper's
simulation method. ORCHES says its simulator extends AttAcc, relies on prior
GPU/PIM validation, and retains prior unit latency and energy. AttAcc uses a
layer analytical model with fixed compute/memory utilization and limited A100
fits; Duplex uses operation-level `max(FLOPs/peak, bytes/bandwidth)` timing.
Neither consumes a measured target-workload GPU latency trace.

M4B will therefore expose `paper_method` and `orin_calibrated` as separate
profiles. The former is sufficient for a transparently labeled reproduction;
the latter remains optional stronger validation. RTX 5070 Ti wall time will
never be substituted for Orin timing or energy.

The workload-source audit found:

1. compute-optimal-TTS commit `0ee2578` supports the paper's one-parent beam
   shape with `num_sequence=1`, exposes token counts and final PRM history, but
   does not save the rejected tree, token-ready progress, or layer-10 scores;
2. the frozen LLaVA-o1 checkout is only a redirect; executable vision code must
   be pinned from LLaVA-CoT, with `8983878` selected as the closest public
   paper-era inference commit under `A-VISION-001`;
3. LLaVA-CoT stage beam uses four stages and pairwise model judging, so schema
   v1's scalar PRM fields cannot faithfully represent its verifier;
4. a 5070 Ti can develop and collect small unquantized text combinations, while
   the 11B BF16 vision and 7B/8B PRM matrices require a larger server GPU.

The trace contract was further tightened after review: all sizes come from the
exact token IDs/masks submitted to the model, never output characters or
re-tokenized text. Schema v2 must separate generated IDs from KV-materialized
positions and retain ordered KV lineage. Width greater than two follows actual
top-k or pairwise tournament decisions, including multiple surviving beams and
the precise point at which each losing candidate becomes dead.

The complete collection fields, uv environment split, local pilot, server
matrix, artifact contract, and implementation order are frozen in
`docs/trace-collection-plan.md`. The planned collector commands do not exist
yet; this checkpoint is design evidence, not a completed real-trace milestone.

### M4B.1 Implementation Checkpoint: Exact Trace Schema V2

#### Scope

This checkpoint implements the paper-facing representation required before a
real model pipeline can be instrumented. It does not collect MATH500,
LiveCodeBench, or MathVista data and does not claim an evaluation result. Schema
v1 remains available to the existing simulator as a synthetic-only contract;
schema v2 is a separate type so exact evidence is not forced into v1's
single-winner assumptions.

#### Implementation Sequence

1. Added `TokenTensorTrace`, which stores the exact submitted token IDs,
   attention mask, and model-visible token count. Text tensors require the model
   count to equal valid IDs; vision tensors can explicitly include expanded
   image positions.
2. Added `CandidateTraceV2` with exact generated token IDs, a request-global
   logical ready order, separate output-KV materialization, reused KV count,
   parent candidate, and terminal KV block.
3. Added append-only `KvBlockTraceV2` lineages. The validator derives lineage
   totals, rejects cycles/orphans/extra owner blocks, and checks each candidate's
   extension against actual input plus materialized output positions.
4. Separated `search_width` from `beam_size`. Each retained parent generates one
   width-sized child set, while all selected beams remain legal next-step
   parents.
5. Added scalar-PRM and pairwise-judge calls. Scalar calls retain exact score
   vectors; pairwise calls retain exact judge input/output tokens and may not
   fabricate scalar scores.
6. Added ordered selection events with considered/selected/pruned/live IDs and
   the full retained-KV set after each event. Pairwise events must follow the
   recorded judge winner.
7. Added strict version-dispatched JSONL I/O. Files cannot mix schema versions,
   v1 rejects collected provenance, and CLI summaries report schema/beam values.
8. Added deterministic v1-to-v2 migration only for synthetic fixtures. Its token
   IDs are explicit placeholders, provenance remains synthetic, and it cannot be
   used as evaluation evidence.

#### Core Invariants

```text
valid_token_count = sum(attention_mask)
generated_tokens = len(generated_token_ids)
parent_lineage_tokens = reused_kv_tokens
terminal_lineage_tokens = model_input_tokens + materialized_output_tokens
KV_extension = terminal_lineage_tokens - parent_lineage_tokens

children_per_selected_parent = search_width
final_live_candidates = selected_candidate_ids
count(final_live_candidates) <= beam_size
retained_KV = union(lineage(candidate) for candidate in final_live_candidates)
```

Token-ready indices are contiguous for the request and later steps cannot
precede earlier generation steps. Character length, whitespace splitting, and
decoded-text re-tokenization are absent from the schema and rejected as unknown
candidate fields.

#### Paper Correspondence

| Implemented evidence | Paper use |
|---|---|
| Exact policy/verifier token tensors | Sec. 2.2 TTC generation and verification sizes |
| Width per parent and retained beam list | Sec. 3.1 variable candidate parallelism |
| Parent and KV block lineage | Sec. 3.2 dependency and Technique 3 memory lifetime |
| Layer-named scalar PRM calls | Technique 2 layer-10 prediction/final verification input |
| Request-global token-ready order | Technique 2B pre-verification trigger order |
| Ordered pairwise loser pruning | LLaVA-CoT width-2/4 vision trace and KV release point |

#### Verification

Fourteen new focused tests cover exact-token byte-stable round trips, CLI v2
inspection, width 4 with non-first winners, two retained parents expanded in the
next step, `beam_size=3`, generated/output-KV count differences, broken parent
prefixes, incorrect KV extension sizes, retained pruned blocks, cross-step token
ordering, pairwise elimination/winner consistency, character-count rejection,
v1 collected rejection, synthetic migration, and mixed-version file rejection.

The next implementation checkpoint is host probing plus collection manifests,
followed by the compute-optimal-TTS event sink. Until that sink runs on a visible
GPU, the repository still has no real workload trace.

### M4B.2 Implementation Checkpoint: Schema-V2 ORCHES Replay

#### Scope

This checkpoint connects validated schema-v2 requests to the functional
ORCHES T1/T2/T3 simulator. It replaces v1 replay assumptions that counted one
generation operation per reasoning step, retained exactly one winner, and made
generated-token count equal KV growth. It does not add a real trace collector,
calibrate the analytical rates, or produce paper evaluation numbers.

#### Implementation Sequence

1. Added `OrchesV2RequestReplayer` as a separate adapter so the existing v1
   synthetic path remains byte- and behavior-compatible.
2. Expanded each reasoning step into one `GenerationPlan` per actual generated
   token round. Candidates that have already stopped are removed from later
   rounds, so active branch width follows the recorded output lengths.
3. Derived shared and unique attention fragments from parent KV ancestry. This
   supports several retained parents whose private lineages remain distinct in
   the next expansion.
4. Allocated physical KV from `materialized_output_tokens`, not decoded text or
   generated-token count. A zero-length extension reuses the parent terminal
   block and does not create a fake allocation.
5. Converted request-global logical token-ready indices into simulated readiness
   within the analytical generation interval. Collector wall time is never used.
6. Scheduled selection-driving scalar and pairwise verifier calls from their
   exact model-input token tensors plus generated decision tokens.
7. Interleaved every verifier call with its ordered selection event. Later
   pairwise judges depend explicitly on the previous prune event; physical KV
   blocks are freed from the event's exact retained-lineage set.
8. Kept T2A prediction/speculation enabled only for `beam_size=1`, which matches
   the paper's main text setting of one retained sequence. Multi-beam replay
   still executes T1, T2B, verifier decisions, and T3. Pairwise vision traces do
   not fabricate the scalar scores required by T2A.
9. Extended replay activity accounting to sum every generation/speculation
   phase rather than charging only the first token plan.

#### Per-Token Execution Model

For generated-token round `r`, let `A_r` be candidates whose recorded output
contains token `r`. The adapter computes:

```text
shared_r = tokens in the intersection of all active parent KV lineages
unique_i_r = candidate_i.model_input_tokens + r - shared_r
active_width_r = count(A_r)
```

`shared_r` and every nonzero `unique_i_r` become T1B attention fragments. The
formula counts the context visible when generating token `r`; the generated
token itself enters the next decode round. Physical output KV is handled
separately using the trace's materialization count.

A selection-driving verifier call uses:

```text
verifier_tokens = sum(input_tensor.model_token_count)
                + count(generated_decision_token_ids)
duration = verifier_tokens * configured_verifier_time_per_token
```

T2B currently models incremental pre-verification over policy tokens as they
become ready. Full verifier tensors are retained for exact final-call sizing;
cross-tokenizer alignment between a streamed policy token and newly available
PRM input positions must be supplied by the real collector before that timing
can be called checkpoint-exact (`A-T2-003`).

#### Paper Correspondence

| Implemented behavior | Paper relationship |
|---|---|
| Per-token active widths and parent-lineage fragments | Sec. 3.1-3.2 and Technique 1A/1B workload inputs |
| Logical readiness mapped onto analytical time | Technique 2B without importing RTX 5070 Ti timing |
| Single-retained-path predictor/speculation | Technique 2A and the paper's `num_sequence=1` text setup |
| Exact scalar/pairwise verifier workload | Sec. 2.2 TTC verification and LLaVA-CoT judging |
| Ordered physical KV release and retained ancestry | Sec. 3.3 and Technique 3 memory lifetime |
| All token phases included in activity | Sec. 5.2 energy-accounting prerequisite |

#### Verification

Six new replay tests cover per-token generation phases, `beam_size=3`, exact
materialized KV bytes, two retained parent lineages across steps, ordered
pairwise judge/prune dependencies, candidate-local small-PRM calls, single-use
state isolation, and activity accumulation over all phases. The complete Python
suite passes 141 tests. Syntax compilation and `git diff --check` also pass.

#### Current Boundary

The simulator can now consume a valid schema-v2 trace without collapsing its
token, beam, verifier, or KV semantics. This is the normal analytical PIM
accelerator workflow used by AttAcc/Duplex-style studies: workload control flow
feeds operator and memory models, with native Ramulator used for memory-command
validation. It is not RTL, cycle-accurate end-to-end GPU execution, or measured
ORCHES silicon. Quantitative alignment still requires real traces, frozen
`paper_method` rates, executable baselines, and the M5 deviation report.

## Change Log

| Date | Milestone | Change |
|---|---|---|
| 2026-07-27 | M0 | Froze environment and upstream source revisions; documented AttAcc's Ramulator2 revision override. |
| 2026-07-27 | M1 | Added strict provenance-aware hardware contracts, corrected the PIM channel mapping, verified AGX Orin specifications, added bootstrap audit, and passed 14 tests. |
| 2026-07-27 | M2A | Added deterministic TTC traces, six model profiles, operator/timing primitives, a generic event timeline, reversible PIM mapping, and native AttAcc smoke execution; 64 tests pass, while real traces and optional Orin calibration remain pending. |
| 2026-07-27 | M3 | Implemented T1A/T1B scheduling, history-aligned prediction, speculative rollback, pipelined verification, fragmentation-aware memory structuring, and request-level replay; 105 tests pass, while calibrated evaluation remains pending. |
| 2026-07-27 | M4A | Added paper baseline definitions, fairness fingerprints, strict AttAcc/Duplex/ORCHES adapters, and unit-explicit energy/area/utilization accounting; 121 tests pass, while executable launchers and calibration remain pending. |
| 2026-07-27 | M4B design | Audited AttAcc/Duplex GPU models and both workload sources; separated paper-method timing from optional Orin calibration and froze the 5070 Ti/server trace-collection plan. |
| 2026-07-27 | M4B.1 | Implemented schema v2 exact-token tensors, logical KV lineage, multi-beam selection, scalar/pairwise verifiers, strict versioned I/O, and synthetic-only migration; 135 tests pass. |
| 2026-07-27 | M4B.2 | Connected schema v2 to per-token ORCHES replay with multi-parent KV, exact verifier calls, ordered pruning, and phase-complete activity accounting; 141 tests pass, while real collection remains pending. |
