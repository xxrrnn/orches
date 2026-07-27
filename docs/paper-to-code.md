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
| Sec. 2.2, Fig. 3 | Generation/verification TTC tree | TTC v1/v2 plus generation-only `policy_schema.py`, `policy_io.py`, `policy_manifest.py`; `replay_v2.py` | Exact-token round trip/replay; width 4; beam 3; multi-parent KV; strict policy manifest and partial-evidence label | M4B.3 collection contract complete; real collector pending |
| Sec. 3.1, Fig. 4-5 | Variable width and shared/unique KV behavior | `models/operators.py` | Arithmetic-intensity and DAG tests | M2 primitive complete |
| Sec. 4.1, Fig. 7 | Reversible PIM hierarchy and AttAcc commands | `sim/pim.py`, `sim/pim_trace.py` | Address and native integration tests | M2A complete |
| Sec. 5.1 | Policy/PRM model shapes | `configs/models/`; `models/transformer.py` | Six profile tests | Policy shapes inherited; tuned PRMs assumed |
| Fig. 7, Sec. 4.1 | Controller address cache, state machine, accumulation, softmax, shared-KV buffer | `memory/cache.py`, `memory/buffer.py`; native accumulation/softmax adapter in `sim/pim_trace.py` | Cache coherence/LRU/timing, buffer writeback/sync, native command tests | Functional contract complete; calibrated area/energy deferred to M4 |
| Sec. 4.2.1, Fig. 8(a)-(e), Eq. (1)-(4) | T1A tier placement and linear co-processing | `models/timing.py`, `scheduler/offline.py`, `scheduler/generation.py` | Hand equations, analytical alpha versus dense search, paper inequality, tier placement | M3 complete; thresholds require calibration |
| Sec. 4.2.2, Fig. 8(f), Eq. (5)-(7) | T1B runtime compensation | `scheduler/online.py`, `scheduler/generation.py` | Fragment equations, width ordering, one critical alpha, grid comparison | M3 complete |
| Sec. 4.3.1, Fig. 9(a)-(c) | T2A history-aligned prediction, PIM speculation, commit/rollback | `predictor/scores.py`, `speculation.py`, `partition.py`; `replay.py` | Alignment ranking, Fig. 13 layer partition, correct/mismatch replay | M3 complete; score aggregation assumed |
| Sec. 4.3.2, Fig. 9(d) | T2B token-triggered pre-verification in GPU idle windows | `predictor/pipeline.py`; `replay_v2.py`; `sim/event.py` | zero/partial/full overlap, logical token-ready mapping, threshold, shared timeline IDs | M4B.2 functional; threshold and cross-tokenizer readiness require calibration/collection |
| Sec. 3.3, Fig. 10, Sec. 4.4 | T3 address cache, holes, compaction, shared-KV buffer | `memory/allocator.py`, `cache.py`, `policy.py`, `trace.py`, `buffer.py`; `replay_v2.py` | first-fit/prune, ordered pairwise release, retained multi-parent lineage, stale generation, RD/WR trace | M4B.2 functional; hardware sizes/rates assumed |
| Sec. 4.2-4.4 | Request-level composition of T1/T2/T3 | v1: `replay.py`; v2: `replay_v2.py` | v1 speculation/T3 ablation; v2 per-token phases, beam 3, multi-parent, pairwise order, activity | M4B.2 functional replay complete; no paper-number claim |
| Sec. 5.1, Fig. 11-12, Table 6 | GPU, AttAcc, Duplex, ORCHES and ablation definitions | `baselines/definitions.py` | exact required sets and technique-switch tests | M4A contract complete; launchers pending |
| Sec. 5.1-5.2 | Fair normalized latency/energy comparison | `baselines/results.py` | contract fingerprint, mismatch rejection, failure retention | M4A complete |
| Prior work [25, 40] | Native AttAcc ms/nJ and Duplex ns/nJ output adaptation | `baselines/adapters.py` | schema/unit/OOM/non-finite fixture tests | M4A parser complete; native execution pending |
| Table 1, Sec. 5.1 | Activity-counter energy using inherited unit values | `metrics/energy.py`, `activity.py` | exact constants, pJ-to-J, MAC conservation | M4A complete; controller calibration pending |
| Sec. 5.5-5.6 | Area overhead and GPU/PIM utilization | `metrics/area.py`, `utilization.py` | explicit denominator, 12% constructed case, overlap accounting | M4A contract complete; measurements pending |

The complete phase sequence and evaluation matrix remain in
`REPRODUCTION_PLAN.md`.
