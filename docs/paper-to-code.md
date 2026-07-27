# Paper-to-Code Map

This map is updated as implementation milestones finish. A row is complete only
when both implementation and verification evidence exist.

| Paper item | Requirement | Configuration/code | Verification | Status |
|---|---|---|---|---|
| Sec. 5.1 | Jetson AGX Orin 32 GB GPU baseline | `configs/hardware/agx_orin_32gb.yaml`; `GpuHardwareConfig` | `test_gpu_paper_configuration` | M1 complete |
| Fig. 11, Sec. 5.1 | 100%, 75%, 50% SoC bandwidth | `soc_bandwidth_scales`; `GpuHardwareConfig.summary` | `test_gpu_paper_configuration` | M1 contract complete; execution deferred to M2 |
| Sec. 5.1 | PIM capacity is 32 GB | `orches_pim_32gb.yaml`; `PimHardwareConfig.derived_capacity_gib` | `test_pim_paper_configuration`; inconsistent-capacity failure test | M1 complete with documented GB/GiB assumption |
| Sec. 5.1 | PIM has 2048 banks | hierarchy fields; `PimHardwareConfig.derived_bank_count` | `test_pim_paper_configuration`; inconsistent-bank failure test | M1 complete |
| Sec. 4.1, 5.1 | 16 multipliers/adders per bank | `gemv_lanes_per_bank`; `total_gemv_lanes` | `test_pim_paper_configuration` | M1 contract complete; timing deferred to M2/M4 |
| Sec. 5.1 | Timing and unit energy inherit AttAcc | timing/refresh fields; `third_party.lock` | config provenance tests; AttAcc smoke evidence in `DEVELOPMENT.md` | Timing contract complete; energy deferred to M4 |
| Sec. 2.2, Fig. 3 | Generation/verification TTC tree | `workload/schema.py`, `io.py`, `synthetic.py` | Workload invariant and round-trip tests | M2 schema complete; real collection pending |
| Sec. 3.1, Fig. 4-5 | Variable width and shared/unique KV behavior | `models/operators.py` | Arithmetic-intensity and DAG tests | M2 primitive complete |
| Sec. 4.1, Fig. 7 | Reversible PIM hierarchy and AttAcc commands | `sim/pim.py`, `sim/pim_trace.py` | Address and native integration tests | M2A complete |
| Sec. 5.1 | Policy/PRM model shapes | `configs/models/`; `models/transformer.py` | Six profile tests | Policy shapes inherited; tuned PRMs assumed |
| Fig. 7, Sec. 4.1 | Controller address cache, state machine, accumulation, softmax, shared-KV buffer | Deferred controller contract | Future unit and integration tests | Planned M4 |
| Sec. 4.2.1, Eq. (1)-(4) | Unit-exact linear timing equations | `models/timing.py` | Hand calculations and alpha boundaries | Primitive complete; scheduler pending M3 |
| Eq. (5)-(7) | T1B/T2 orchestration | Not implemented in M2A | Future scheduler tests | Planned M3 |

The complete phase sequence and evaluation matrix remain in
`REPRODUCTION_PLAN.md`.
