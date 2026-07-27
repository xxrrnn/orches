# AttAcc PIM Backend

## Compatibility Boundary

ORCHES emits the same two-column trace vocabulary consumed by AttAcc's
`PIMLoadStoreTrace`: load/store, all/same/per-bank MAC, GEMV-buffer movement,
softmax-buffer movement, softmax, model/head setup, and barrier commands.
Generated Ramulator2 YAML reuses AttAcc's `HBM3-PIM`, `PIMDRAM`, PIM scheduler,
`AllBankHBM3` refresh manager, trace recorder, and address mapper.

No file under `third_party/` is modified by the ORCHES adapter.

## Address Mapping

The byte mapper is reversible over:

```text
channel -> pseudochannel -> rank -> bankgroup -> bank
        -> row -> column -> byte offset
```

For the current contract the dimensions are
`32 x 2 x 2 x 4 x 4 x 16384 x 32 x 32`, producing exactly 32 GiB and a
35-bit byte address. Unit tests cover zero, maximum, interior, and invalid
coordinates.

## Native Microbenchmark

`orches pim-microbench OUTPUT_DIR` writes a deterministic command trace and
Ramulator2 YAML, then invokes the frozen AttAcc executable. The trace writes a
GEMV input on every channel, issues all-bank MAC rounds with channel barriers,
moves results to the softmax buffer, invokes softmax, and finishes with a
barrier. Output includes memory-system cycles and every AttAcc request counter.

This is a functional/timing smoke test, not a calibrated ORCHES GEMV result.
Later microbenchmarks must isolate linear, shared attention, unique attention,
host I/O, row locality, refresh, and controller operations before deriving
evaluation rates.
