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
