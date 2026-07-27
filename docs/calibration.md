# Calibration Plan and Status

## Evidence Profiles

ORCHES Sec. 5.1 says that the simulator extends AttAcc, retains its unit latency
and energy, and relies on the prior work's hardware validation. Consequently,
target-Orin measurement is not a prerequisite for collecting workload control
flow or for running a clearly labeled reproduction of the paper's simulation
method.

Two profiles are kept separate:

| Profile | Purpose | Eligibility |
|---|---|---|
| `paper_method` | AttAcc-style analytical GPU model with paper bandwidth, inherited utilization, and explicit peak/assumed fields | Main paper-method reproduction with sensitivity |
| `orin_calibrated` | Rates fitted from target AGX Orin operator measurements | Independent validation and calibrated result set |

An RTX 5070 Ti may collect model-dependent traces. Its wall-clock rates and
energy are not valid inputs to either AGX Orin profile. See
`docs/trace-collection-plan.md`.

## Status

No `orin_calibrated` profile is available in the current environment. The host
does not expose the target GPU. Official core count, frequency, and memory
bandwidth remain upper bounds and must not be recorded as achieved rates. This
does not block the separate `paper_method` profile.

The AttAcc Ramulator2 backend is executable. M2A ran a functional all-bank MAC
smoke test, but one mixed command sequence is insufficient to infer the
separate Eq. (2)-(3) compute, internal-memory, host-I/O, accumulation, and
softmax rates.

## GPU Calibration Matrix

Run on a Jetson AGX Orin 32 GB with the exact power mode, JetPack, CUDA,
compiler, clocks, and thermal state recorded. Each point requires warmup,
multiple measured iterations, median and dispersion, and raw output retention.

| Primitive | Sweep | Derived parameter |
|---|---|---|
| FP16/BF16 GEMM | M/N/K covering model projections and widths 1-8 | effective FLOP/s, kernel efficiency |
| GEMV | hidden/intermediate sizes, width 1 | low-width effective FLOP/s |
| Batched linear | widths 2/4/6/8 | width-dependent effective FLOP/s |
| Device memory copy | 1 KiB through multi-GiB | effective byte/s and fixed cost |
| Shared KV attention | context and width sweep | batched attention roofline |
| Unique KV attention | fragment length and width sweep | irregular attention roofline |
| Empty/small kernels | repeated launches | launch and synchronization overhead |

The 100/75/50% profiles must alter only measured memory bandwidth. Compute
rates and fixed overhead remain unchanged unless an independently documented
power/clock mode changes them.

## PIM Calibration Matrix

Use generated traces against the frozen AttAcc build base. Run each primitive
alone before composing a layer:

| Primitive | Required checks |
|---|---|
| Bank MAC | command rounds, channel/bank scaling, power-constrained timing |
| GEMV input/output | host-I/O byte count and fixed barriers |
| Shared attention | broadcast behavior, internal KV reads, accumulation |
| Unique attention | per-fragment addressing and low arithmetic intensity |
| Softmax | sequence-length scaling and controller serialization |
| Row locality | row hit/conflict, ACT/PRE counters |
| Refresh | long trace with refresh enabled and disabled sensitivity |

For each run preserve generated YAML/trace hashes, executable hash, source
commits, command counters, memory cycles, and command log. Analytical rates may
be fitted only from primitive runs, never from the paper's final speedup.

## Acceptance

A calibrated value changes from `ASSUMED`/upper-bound status to `CALIBRATED`
only when its raw measurements, environment manifest, fitting method, and
error are committed or archived. Calibration is accepted when held-out shapes
are predicted within a documented tolerance; the tolerance must be chosen
before the final evaluation.
