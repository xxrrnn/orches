# Hardware Contract

## Purpose

The hardware contract is the boundary between paper claims, analytical models,
and the AttAcc/Ramulator2 backend. A value may enter the simulator only through
a versioned configuration field carrying a source and evidence status. Hidden
defaults are not valid baseline inputs.

## Evidence Status

| Status | Meaning | Required treatment |
|---|---|---|
| `PAPER` | Stated directly by ORCHES | Preserve the paper value and citation. |
| `INHERITED` | Taken from a named baseline or device specification | Freeze the upstream revision or document. |
| `CALIBRATED` | Measured on the target platform | Store the raw measurement and fitting error. |
| `ASSUMED` | Needed but not disclosed | Explain the derivation and run sensitivity analysis. |

Every leaf field has `value`, `source`, `status`, and an optional `note`.
Unknown fields and missing metadata fail validation.

## GPU Contract

The GPU model defines physical memory capacity, maximum visible memory
bandwidth, core counts, a peak frequency bound, and the paper's SoC bandwidth
scales. The current contract does not treat peak cores or frequency as achieved
throughput. M2 will add calibrated kernel efficiency, launch overhead, and
effective bandwidth without overwriting the inherited peak values.

For base bandwidth `B` and scale `s`, the sensitivity point is:

```text
B_effective = B * s
```

The scales must be unique, descending, in `(0, 1]`, and include `1.0`.

## PIM Contract

The PIM hierarchy follows AttAcc/Ramulator2 naming exactly:

```text
channel -> pseudochannel -> rank -> bankgroup -> bank -> row -> column
```

The current M1 invariants are:

```text
bank_count = channels
           * pseudochannels_per_channel
           * ranks_per_pseudochannel
           * bankgroups_per_rank
           * banks_per_bankgroup

capacity_GiB = channels * channel_density_Gibits / 8

total_GEMV_lanes = bank_count * GEMV_lanes_per_bank

address_space_bytes = bank_count
                    * rows_per_bank
                    * columns_per_row
                    * transaction_bytes
```

For the paper configuration these derive `2048 banks`, `32 GiB`, and `32768`
GEMV lanes. The inherited `16384 rows x 32 columns x 32 bytes` per bank also
derives a 35-bit byte address space. Validation fails if either the density- or
address-derived organization conflicts with the declared bank count/capacity.

`host_io_bandwidth_gb_per_s` is the controller/host-visible bandwidth. It is
not aggregate bank-local bandwidth. The two limits must remain separate in the
later Eq. (2)-(3) implementation.

## Deferred Controller Contract

Fig. 7 and Sec. 4.1 also require an address cache, state machine, accumulator,
softmax unit, and shared-KV buffer on the controller die. Their capacities,
latencies, throughput, arbitration, and energy are not fully disclosed. M4
will add them as explicit fields after the command/dataflow interface is known;
until then they must not receive silent performance defaults.

## Failure Policy

Configuration loading fails on unknown or missing keys, type mismatches,
missing provenance, invalid evidence status, non-positive physical values,
inconsistent capacity, inconsistent bank count, or invalid bandwidth scales.
The CLI returns status `2` for user configuration errors.
