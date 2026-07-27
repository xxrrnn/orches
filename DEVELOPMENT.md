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
| M2 | TTC traces, GPU baseline, PIM microbenchmarks | In progress (M2A complete) | 64 tests; native PIM smoke; real traces/GPU calibration pending |
| M3 | ORCHES Techniques 1, 2, and 3 | Not started | - |
| M4 | Baselines, energy, area, utilization | Not started | - |
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
synthetic trace generator for simulator development, and implement calibrated
GPU/PIM primitive models. It will map GPU timing to Eq. (1), PIM timing to
Eq. (2)-(3), and preserve measured versus assumed parameters separately.

## M2A: Trace, Model, Timing, and Native PIM Foundation

### Checkpoint Scope

M2 spans Phase 2-4 and is not complete yet. This checkpoint implements the
replayable workload contract, Transformer shapes/operator expansion, unit-safe
analytical timing equations, reversible PIM addressing, and a native AttAcc
smoke path. Real paper workload collection, AGX Orin calibration, full GPU
baseline execution, and calibrated PIM primitive sweeps remain open.

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

1. Instrument the frozen compute-optimal-TTS and LLaVA-o1 pipelines to collect
   real MATH500/LiveCodeBench/MathVista schema-v1 traces.
2. Resolve the exact paper policy/PRM checkpoint and tokenizer revisions,
   including the custom tuned PRMs.
3. Run AGX Orin GEMM/GEMV/attention/memory/launch calibration and store raw
   measurements; the current machine has no target GPU.
4. Integrate the generic event timeline into an executable pure-GPU request
   baseline; the current timeline is validated but does not yet replay requests.
5. Expand native PIM microbenchmarks for linear, shared/unique attention,
   host-I/O, row locality, refresh, barrier, accumulation, and softmax scaling.
6. Fit and validate PIM primitive rates or use raw Ramulator cycles directly;
   do not infer them from the paper's final speedup.

The required calibration matrix and acceptance criteria are in
`docs/calibration.md`.

## Change Log

| Date | Milestone | Change |
|---|---|---|
| 2026-07-27 | M0 | Froze environment and upstream source revisions; documented AttAcc's Ramulator2 revision override. |
| 2026-07-27 | M1 | Added strict provenance-aware hardware contracts, corrected the PIM channel mapping, verified AGX Orin specifications, added bootstrap audit, and passed 14 tests. |
| 2026-07-27 | M2A | Added deterministic TTC traces, six model profiles, operator/timing primitives, a generic event timeline, reversible PIM mapping, and native AttAcc smoke execution; 64 tests pass, while real traces and GPU calibration remain pending. |
