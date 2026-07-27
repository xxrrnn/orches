# Assumption Register

Assumptions are baseline inputs that are not directly stated by the paper.
They remain visible until confirmed by an official artifact or replaced by a
calibration. Main evaluation runs must record the active assumption IDs.

| ID | Assumption | Reason | Validation or sensitivity requirement | State |
|---|---|---|---|---|
| `A-HW-001` | Paper's 32 GB PIM maps to a 32 GiB Ramulator2 address space. | The paper does not define decimal/binary capacity semantics. | Report the unit mapping and test any trace near the capacity boundary. | Open |
| `A-HW-002` | `HBM3_8Gb_2R` with 32 channels is the PIM organization. | It preserves AttAcc's preset and doubles AttAcc's 16-channel baseline, simultaneously realizing 32 GiB and 2048 banks. | Compare against an official ORCHES artifact if released; retain channel-count and organization sensitivity. | Open |
| `A-GPU-001` | NVIDIA peak core/frequency specifications bound the AGX Orin analytical model. | ORCHES names the module but does not disclose achieved throughput. | Replace achieved performance with M2 calibration; never use peak as measured throughput. | Open |
| `A-PIM-001` | One configured GEMV lane represents one multiplier-plus-adder datapath per bank. | This is the most direct representation of Sec. 4.1 wording. | Validate command issue and accumulation semantics with PIM microbenchmarks. | Open |
| `A-MODEL-001` | Public Qwen/RLHFlow PRM architectures proxy the paper's custom `*-PRM-Tuned` checkpoints. | The paper does not disclose exact tuned checkpoint revisions. | Replace proxies during trace collection and report any architecture mismatch. | Open |
| `A-CAL-001` | No achieved AGX Orin compute rate, launch overhead, or effective bandwidth is available yet. | The current host has no target GPU and the paper omits calibration details. | Run the documented operator microbenchmarks on the target module before evaluation. | Open |
| `A-T2-001` | Candidate path scores use an explicitly selected mean, minimum, or last-score aggregation. | The paper describes history replacement but does not disclose the full score aggregation function. | Freeze the chosen method per experiment and run all three as sensitivity points unless author code resolves it. | Open |
| `A-T1-001` | Small/medium/large branch-width boundaries are supplied by calibration rather than hard-coded. | Sec. 4.2.1 defines three regions but does not publish exact transition widths for every model and bandwidth. | Derive boundaries from frozen operator calibration and save them in every evaluation configuration. | Open |
| `A-T2-002` | The minimum pipelined-prefill token count and per-token verifier rates are explicit calibration inputs. | Sec. 4.3.2 requires enough arithmetic intensity to utilize the GPU but gives no universal threshold. | Sweep the threshold per PRM and freeze the selected calibration point before evaluation. | Open |
| `A-T3-001` | The controller address cache is fully associative with deterministic LRU replacement and an explicit entry count. | Sec. 4.4 states that one cache is shared across banks but omits capacity, associativity, and replacement policy. | Sweep entry count and replacement policy or replace them if an official artifact is released. | Open |
| `A-T3-002` | Address-cache SRAM and DRAM lookup latencies are supplied independently. | The paper gives only an order-of-magnitude SRAM/DRAM comparison. | Replace with CACTI and Ramulator calibration; retain hit-rate and latency sensitivity. | Open |
| `A-T3-003` | `memory_for_reasoning` is the allocated reasoning high-water span, excluding immutable model weights. | Sec. 4.4 defines beta as holes over memory for reasoning but does not define the denominator boundary. | Report both live bytes and high-water span; compare alternative total-capacity normalization. | Open |
| `A-T3-004` | Shared-KV buffer capacity, compaction bandwidth, transaction size, and optional beta threshold are explicit experiment inputs. | The paper reports 12% area and 0.12% runtime overhead but not the underlying hardware sizes/rates. | Calibrate buffer/transaction values and run interval 3/4/5 plus beta-threshold sensitivity. | Open |
