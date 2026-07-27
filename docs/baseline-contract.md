# Baseline and Metrics Contract

This document records M4A, the common comparison boundary for GPU, AttAcc,
Duplex, and ORCHES. It prevents a native tool's convenient output format from
silently changing workload, units, or aggregation. M4A parses and validates
results; M4B must still build launchers and execute every system on real traces.

## Paper Configurations

`baselines/definitions.py` freezes the paper-facing systems:

| Name | Placement or native policy | ORCHES switches |
|---|---|---|
| GPU | all GPU | none |
| AttAcc | native prior work; attention on PIM | none |
| Duplex | native LLMSimulator policy | none |
| ORCHES-A | all computation on PIM | T3 |
| ORCHES-B | adaptive linear assignment | T1A + T3 |
| ORCHES-C | B plus dynamic compensation | T1A + T1B + T3 |
| ORCHES T1 only | adaptive + dynamic scheduling | T1A + T1B + T3 |
| ORCHES T2 only | all-PIM speculation policy | T2A + T2B + T3 |
| ORCHES | full policy | T1A + T1B + T2A + T2B + T3 |

Sec. 5.3 explicitly disables T2 for A/B/C but does not say T3 is removed. M4A
therefore retains T3 and registers that interpretation as `A-BASE-001`.

The required sets are constants rather than plotting-script selections:

```text
Fig. 11: GPU, AttAcc, Duplex, ORCHES
Fig. 12: GPU, AttAcc, ORCHES-A, ORCHES-B, ORCHES-C
Table 6: GPU, T1 only, T2 only, ORCHES
```

## Fairness Fingerprint

Every `BaselineRunResult` carries a `FairnessContract`. Normalization is
rejected unless all required rows have the same SHA-256 fingerprint over:

1. trace SHA-256 and ordered request IDs;
2. policy, small PRM, large PRM, and tokenizer names plus revisions;
3. weight and activation bytes per element;
4. GPU count, SoC bandwidth, and PIM capacity;
5. the hardware-contract SHA-256.

The fingerprint prevents a 100% bandwidth GPU result from being compared with
a 75% bandwidth ORCHES result, or a public proxy PRM from being mixed with a
different tuned checkpoint. It does not prove that a native command used those
inputs; M4B launch manifests must bind command/config hashes to the contract.

OOM, failed, and missing runs remain rows with no normalized metric. They are
never filtered before aggregation. A successful GPU reference is mandatory.

## Native Adapter Boundaries

### AttAcc

Frozen source: `c60005143a6b492d7ef83231723386478b59a506`.

`parse_attacc_csv` reads the schema written by upstream `main.py`. It converts:

```text
g_time (ms)   * 1e-3 -> latency_s
g_energy (nJ) * 1e-9 -> energy_j
component nJ  * 1e-9 -> component joules
cap (GiB)             -> capacity bytes
```

`required_cap > cap * GiB` becomes an explicit OOM result. Non-numeric,
negative, `NaN`, and infinite values are rejected.

### Duplex

Frozen source: `419252761fbdb95b789778a02256d458a5537ec7`.

`parse_duplex_csv` reads the `Cluster::exportToCSV` schema. It converts ns to
seconds and nJ to joules. `type=e2e` rows are request-completion records;
`iter_info=1` rows carry device activity and energy. Under `A-DUPLEX-001`, the
adapter uses mean E2E request latency and sums activity-row energy. Any nonzero
OOM field produces an explicit OOM result.

These parsers do not approximate either native simulator. They only translate
the frozen native output into the common result type.

## ORCHES Adapter

`adapt_orches_replay` converts a full M3 request replay into the same
`BaselineRunResult`. It requires verifier MAC/byte activity explicitly because
M3 intentionally does not infer PRM operation counts from elapsed time.
Generation activity is reconstructed from the selected Eq. (1)-(7) alphas;
speculative all-PIM work, cache accesses, buffer traffic, GPU synchronization,
and compaction reads/writes are then added.

## Energy Contract

Energy is always computed as activity multiplied by unit energy:

```text
energy_j = activity_count * unit_energy_pj * 1e-12
```

`attacc_bank_level_energy_rates` reproduces the upstream bank-level constants:

| Item | Value | Unit |
|---|---:|---|
| GPU/PIM compute | 0.32 | pJ/MAC |
| GPU off-memory | 28.72 | pJ/byte |
| Bank-level PIM memory | 4.4 | pJ/byte |
| Host communication | 10.4 | pJ/byte |
| Controller SRAM proxy | 0.0034 | pJ/byte |

The first four are inherited from AttAcc. Applying the SRAM value to the new
ORCHES cache/buffer is `A-ENERGY-001` and must be replaced by CACTI/synthesis
before final energy claims. Energy is not called power; no time normalization
is performed implicitly.

## Area and Utilization

`AreaReport` requires a named baseline area in mm2 and sourced component areas.
Its overhead is `added_area / baseline_area`. This makes the paper's 12% claim
unambiguous once the denominator is identified.

`UtilizationReport` uses one common makespan. For every resource:

```text
idle = makespan - busy
utilization = busy / makespan
```

The event timeline already forbids overlap on the same resource. GPU and PIM
may overlap each other, reducing wall time without reducing either resource's
accumulated busy time.

## M4B Requirements

1. Generate native AttAcc and Duplex configs from the fairness contract.
2. Bind command, config, binary, source revision, raw output, and trace hashes
   in a run manifest.
3. Implement executable GPU and ORCHES-A/B/C/T1-only/T2-only adapters.
4. Calibrate controller SRAM/buffer energy and identify the area denominator.
5. Compare all required rows without dropping OOM, failure, or missing runs.
